import datetime
import hashlib
import json
import logging
import math
import shutil
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from core.data.repos.historical_repair import HistoricalRepairRepo
from core.domain.data_quality.historical_release import (
    HISTORICAL_QFQ_PRICE_TOLERANCE,
    HISTORICAL_VOLUME_TOLERANCE_SHARES,
    assess_quote_shard_release,
    classify_database_qfq_price,
    invalid_or_duplicate_date_axes,
)
from core.shared.infrastructure.async_task import TaskCancelled, task_manager
from core.shared.persistence import load_json, load_parquet, now_iso, save_json, save_json_atomic, save_parquet
from backend.service.qlib_access import qlib_mutation_locked


_batch_lock = threading.Lock()
_active_batch_job = None
_release_publish_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _historical_release_blocker(rule_id, message):
    message = str(message).strip()
    if message.startswith("DA04-") and ":" in message:
        return message
    return f"{rule_id}: {message}"


def _historical_release_blocker_ids(blockers):
    rule_ids = set()
    for blocker in blockers or []:
        value = str(blocker).strip()
        prefix = value.split(":", 1)[0]
        rule_ids.add(prefix if prefix.startswith("DA04-") else "DA04-LEGACY-UNCLASSIFIED-001")
    return sorted(rule_ids)


def _append_historical_release_blocker(manifest, rule_id, message):
    blockers = list(manifest.get("blockers") or [])
    blockers.append(_historical_release_blocker(rule_id, message))
    manifest["blockers"] = list(dict.fromkeys(blockers))
    manifest["blocker_rule_ids"] = _historical_release_blocker_ids(manifest["blockers"])


def _mark_historical_release_blocked(manifest, path, rule_id, message):
    manifest["status"] = "blocked"
    manifest["publishable"] = False
    _append_historical_release_blocker(manifest, rule_id, message)
    manifest["stale_at"] = now_iso()
    manifest["updated_at"] = manifest["stale_at"]
    save_json_atomic(manifest, path)


def _historical_release_gate(blocker_groups, assessment_status, qlib_status, blocker_rule_id_groups=None):
    blockers = list(dict.fromkeys(
        str(blocker).strip()
        for group in blocker_groups
        for blocker in (group or [])
        if str(blocker).strip()
    ))
    indexed_rule_ids = {
        str(rule_id).strip()
        for group in blocker_rule_id_groups or []
        for rule_id in (group if isinstance(group, (list, tuple, set)) else [group])
        if str(rule_id).strip()
    }
    missing_blocker_ids = indexed_rule_ids - set(_historical_release_blocker_ids(blockers))
    blockers.extend(
        _historical_release_blocker(rule_id, "持久化阻塞规则索引存在，但阻塞说明缺失；需重新执行G2预检")
        for rule_id in sorted(missing_blocker_ids)
    )
    if assessment_status != "ready":
        blockers.append(_historical_release_blocker("DA04-ASSESSMENT-001", "字段/交易状态评估未通过"))
    if qlib_status != "ready":
        blockers.append(_historical_release_blocker("DA04-QLIB-CANDIDATE-001", "Qlib候选校验未通过"))
    status = "ready" if not blockers else "blocked"
    return status, blockers


@contextmanager
def _historical_release_advisory_lock(release_id, timeout=30):
    from core.data.connection import Database

    lock_suffix = hashlib.sha256(str(release_id).encode("utf-8")).hexdigest()[:40]
    lock_name = f"historical_release_{lock_suffix}"
    with Database.connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, %s) AS acquired", (lock_name, timeout))
            acquired = cursor.fetchone()
        if not acquired or acquired.get("acquired") != 1:
            raise RuntimeError("无法获取历史发布互斥锁，拒绝并发写入")
        try:
            yield
        finally:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT RELEASE_LOCK(%s) AS released", (lock_name,))
                    released = cursor.fetchone()
                if not released or released.get("released") != 1:
                    raise RuntimeError("历史发布互斥锁未能正常释放")
            except Exception:
                connection.close()
                raise
_PIPELINE_CONTRACT_VERSION = "g1_quote_shadow_v9"


def _batch_root():
    return Path(__file__).resolve().parents[2] / "data/historical_repair/shadow"


def _manifest_path(batch_id):
    if not batch_id.isalnum() or len(batch_id) != 12:
        raise ValueError("batch_id 格式无效")
    return _batch_root() / batch_id / "manifest.json"


def prepare_shadow_batch(codes, start, end, anchor_date=None, denominator_batch_id=None):
    start_date = datetime.date.fromisoformat(str(start)[:10])
    end_date = datetime.date.fromisoformat(str(end)[:10])
    if end_date < start_date:
        raise ValueError("结束日期不能早于开始日期")
    if end_date > datetime.date.today():
        raise ValueError("影子历史批次不能包含未来日期")
    anchor_date = datetime.date.fromisoformat(str(anchor_date or datetime.date.today())[:10])
    if anchor_date < end_date or anchor_date > datetime.date.today():
        raise ValueError("复权锚点必须不早于行情结束日期，且不能晚于今天")
    normalized_codes = list(dict.fromkeys(str(code).strip().zfill(6) for code in codes if str(code).strip()))
    if not normalized_codes or any(not code.isdigit() or len(code) != 6 for code in normalized_codes):
        raise ValueError("请提供有效的六位股票代码")
    denominator_manifest = None
    if denominator_batch_id:
        denominator_batch_id = str(denominator_batch_id).strip()
        if not denominator_batch_id.isalnum() or len(denominator_batch_id) != 12:
            raise ValueError("G0 分母批次 ID 格式无效")
        denominator_root = Path(__file__).resolve().parents[2] / "data/historical_repair/denominator" / denominator_batch_id
        denominator_manifest = load_json(denominator_root / "manifest.json")
        if not denominator_manifest or denominator_manifest.get("batch_id") != denominator_batch_id:
            raise ValueError("G0 分母批次不存在或清单损坏")
        year_summaries = {int(item["year"]): item for item in denominator_manifest.get("year_summaries", [])}
        for year in range(start_date.year, end_date.year + 1):
            summary = year_summaries.get(year)
            expected_path = denominator_root / f"expected_keys_{year}.parquet"
            if not summary or not summary.get("denominator_verified") or not expected_path.is_file():
                raise ValueError(f"G0 分母批次缺少 {year} 年已核验键清单")
            if _file_sha256(expected_path) != summary.get("expected_keys_sha256"):
                raise ValueError(f"G0 分母批次 {year} 年键清单哈希不符")
    from core.data.sources.historical_quotes import get_source_versions
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = _batch_root() / batch_id
    if batch_dir.exists():
        raise RuntimeError("批次目录已存在，请重新生成批次")
    work_items = []
    identity_intervals = {}
    denominator_date_keys = None
    if denominator_manifest and start_date == end_date:
        denominator_path = Path(__file__).resolve().parents[2] / "data/historical_repair/denominator" / denominator_batch_id / f"expected_keys_{start_date.year}.parquet"
        denominator_date_keys = load_parquet(denominator_path)
        denominator_date_keys = denominator_date_keys.loc[
            pd.to_datetime(denominator_date_keys["trade_date"], errors="coerce").dt.date.eq(start_date),
            ["code", "trade_date", "delist_date"],
        ].copy()
    for code in normalized_codes:
        identity = HistoricalRepairRepo.get_shard_inputs(code, start_date.isoformat(), end_date.isoformat())["identity"]
        if denominator_manifest and start_date == end_date:
            denominator_key = denominator_date_keys.loc[
                denominator_date_keys["code"].astype(str).str.zfill(6).eq(code)
            ]
            if len(denominator_key) == 1:
                authoritative_delist_date = pd.to_datetime(denominator_key.iloc[0].get("delist_date"), errors="coerce")
                if pd.notna(authoritative_delist_date) and authoritative_delist_date.date() == start_date:
                    identity["delist_date"] = start_date.isoformat()
        ts_code = str(identity.get("ts_code") or "")
        exchange = ts_code.rsplit(".", 1)[-1].lower() if "." in ts_code else ""
        if not ts_code or exchange not in ("sh", "sz"):
            raise ValueError(f"{code} 缺少证券身份或交易所代码，不能构造双源请求")
        list_date = datetime.date.fromisoformat(str(identity.get("list_date"))[:10])
        delist_date = datetime.date.fromisoformat(str(identity["delist_date"])[:10]) if identity.get("delist_date") else end_date
        effective_start = max(start_date, list_date)
        effective_end = min(end_date, delist_date)
        if effective_end < effective_start:
            identity_intervals[code] = {
                "list_date": list_date.isoformat(),
                "delist_date": delist_date.isoformat() if identity.get("delist_date") else None,
                "effective_start": None,
                "effective_end": None,
                "status": "outside_requested_range",
            }
            continue
        years = list(range(effective_start.year, effective_end.year + 1))
        if len(years) > 8:
            raise ValueError("单个证券身份有效区间超过 8 个自然年，请拆分批次")
        identity_intervals[code] = {
            "list_date": list_date.isoformat(),
            "delist_date": delist_date.isoformat() if identity.get("delist_date") else None,
            "effective_start": effective_start.isoformat(),
            "effective_end": effective_end.isoformat(),
            "status": "in_range",
        }
        for year in years:
            shard_start = max(effective_start, datetime.date(year, 1, 1))
            shard_end = min(effective_end, datetime.date(year, 12, 31))
            inputs = HistoricalRepairRepo.get_shard_inputs(code, shard_start.isoformat(), shard_end.isoformat())
            work_items.append({
                "code": code, "ts_code": ts_code, "exchange": exchange, "year": year,
                "identity_interval": identity_intervals[code],
                "start": shard_start.isoformat(), "end": shard_end.isoformat(),
                "input_sha256": inputs["fingerprint"], "status": "pending",
            })
    if not work_items:
        raise ValueError("所选证券在请求日期范围内均无有效上市交易区间")
    if len(work_items) > 60:
        raise ValueError("单批最多 60 个股票年度分片，请拆分批次")
    batch_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "pipeline_contract_version": _PIPELINE_CONTRACT_VERSION,
        "batch_id": batch_id,
        "status": "prepared",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "anchor_date": anchor_date.isoformat(),
        "denominator_batch_id": denominator_batch_id,
        "requested_codes": normalized_codes,
        "identity_intervals": identity_intervals,
        "source_versions": get_source_versions(),
        "source_capture_contract": "SDK-returned original data frames retained alongside normalized parquet; HTTP wire bytes and provider history revision IDs are unavailable",
        "raw_close_comparison_threshold": "abs(TuShare close - BaoStock close) > 0.011 + abs(TuShare close) * 0.0005 is a review flag only; not a release tolerance",
        "price_basis": "raw_unadjusted_plus_adj_factor",
        "official_database_writes": False,
        "qlib_writes": False,
        "publishable": False,
        "work_items": work_items,
    }
    save_json(manifest, batch_dir / "manifest.json")
    return {
        "batch_id": batch_id, "status": manifest["status"],
        "work_items": len(work_items), "codes": len(normalized_codes),
        "years": sorted({int(item["year"]) for item in work_items}), "official_database_writes": False,
        "anchor_date": manifest["anchor_date"],
        "manifest": str((batch_dir / "manifest.json").relative_to(_batch_root().parent.parent)),
    }


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_shadow_batch(batch_id, job_id):
    global _active_batch_job
    manifest_path = _manifest_path(batch_id)
    manifest = load_json(manifest_path)
    if not manifest or manifest.get("batch_id") != batch_id:
        raise ValueError("历史影子批次不存在或清单损坏")
    if manifest.get("pipeline_contract_version") != _PIPELINE_CONTRACT_VERSION:
        raise RuntimeError("影子批次处理契约版本过旧，只能查看；请按当前版本新建批次")
    batch_dir = manifest_path.parent
    from core.data.sources.historical_quotes import (
        fetch_baostock_raw_daily, fetch_tushare_anchor_factor, fetch_tushare_raw_daily, get_source_versions,
    )
    from core.data.sources.historical_universe import fetch_tushare_suspensions
    anchor_cache = {
        item["code"]: item["source_anchor"]
        for item in manifest["work_items"]
        if item.get("source_anchor") and item.get("status") in ("collected_review", "quarantined_conflict", "non_trading_review")
    }

    source_versions = get_source_versions()
    if manifest.get("source_versions_at_collection") and manifest["source_versions_at_collection"] != source_versions:
        manifest["status"] = "invalidated"
        manifest["updated_at"] = now_iso()
        save_json(manifest, manifest_path)
        raise RuntimeError("数据源客户端版本变化，原批次失效；请新建批次")
    if manifest.get("source_versions") != source_versions:
        manifest["status"] = "invalidated"
        manifest["updated_at"] = now_iso()
        save_json(manifest, manifest_path)
        raise RuntimeError("数据源客户端版本与预检时不一致，原批次失效；请新建批次")
    manifest["source_versions_at_collection"] = source_versions
    manifest["status"] = "collecting"
    manifest["updated_at"] = now_iso()
    save_json(manifest, manifest_path)
    task_manager.update_progress(job_id, {
        "stage": "historical_shadow", "current": 0,
        "total": len(manifest["work_items"]), "message": f"开始影子采集批次 {batch_id}",
    })
    consecutive_baostock_login_failures = 0

    try:
        for index, item in enumerate(manifest["work_items"], 1):
            if task_manager.is_cancel_requested(job_id):
                manifest["status"] = "partial"
                manifest["updated_at"] = now_iso()
                save_json(manifest, manifest_path)
                raise TaskCancelled()
            if item.get("status") in ("collected_review", "quarantined_conflict", "non_trading_review"):
                inputs = HistoricalRepairRepo.get_shard_inputs(item["code"], item["start"], item["end"])
                if inputs["fingerprint"] != item.get("input_sha256"):
                    manifest["status"] = "invalidated"
                    manifest["updated_at"] = now_iso()
                    save_json(manifest, manifest_path)
                    raise RuntimeError(f"{item['code']} {item['year']} 现库输入变化，原批次失效，请重建批次")
                code_dir = batch_dir / "raw" / item["code"]
                stem = str(item["year"])
                artifact_paths = {
                    "tushare_daily_raw": code_dir / f"{stem}_tushare_daily_raw.parquet",
                    "tushare_factor_raw": code_dir / f"{stem}_tushare_factor_raw.parquet",
                    "tushare_normalized": code_dir / f"{stem}_tushare_normalized.parquet",
                    "baostock_raw": code_dir / f"{stem}_baostock_raw.parquet",
                    "baostock_normalized": code_dir / f"{stem}_baostock_normalized.parquet",
                    "qfq_candidate": code_dir / f"{stem}_qfq_candidate.parquet",
                    "comparison": code_dir / f"{stem}_comparison.parquet",
                    "anchor_factor_raw": code_dir / "anchor_factor_raw.parquet",
                    "tushare_suspensions_raw": code_dir / f"{stem}_tushare_suspensions_raw.parquet",
                }
                files = item.get("files", {})
                files_valid = bool(files) and all(
                    name in artifact_paths and artifact_paths[name].is_file()
                    and _file_sha256(artifact_paths[name]) == checksum
                    for name, checksum in files.items()
                )
                if not files_valid:
                    manifest["status"] = "invalidated"
                    manifest["updated_at"] = now_iso()
                    save_json(manifest, manifest_path)
                    raise RuntimeError(f"{item['code']} {item['year']} 影子分片哈希不符，原批次失效")
                task_manager.update_progress(job_id, {
                    "stage": "historical_shadow", "current": index,
                    "total": len(manifest["work_items"]), "message": f"复用已校验分片 {item['code']} {item['year']}",
                })
                continue

            inputs = HistoricalRepairRepo.get_shard_inputs(item["code"], item["start"], item["end"])
            if inputs["fingerprint"] != item.get("input_sha256"):
                manifest["status"] = "invalidated"
                manifest["updated_at"] = now_iso()
                save_json(manifest, manifest_path)
                raise RuntimeError(f"{item['code']} {item['year']} 现库输入已变化，拒绝在旧基线上采集")
            source_start = item["start"].replace("-", "")
            source_end = item["end"].replace("-", "")
            partial_paths = {}
            try:
                item["attempts"] = int(item.get("attempts") or 0) + 1
                code_dir = batch_dir / "raw" / item["code"]
                code_dir.mkdir(parents=True, exist_ok=True)
                anchor_raw_path = code_dir / "anchor_factor_raw.parquet"
                if item["code"] not in anchor_cache:
                    anchor_capture = fetch_tushare_anchor_factor(item["ts_code"], manifest.get("anchor_date") or manifest["date_range"]["end"])
                    anchor_cache[item["code"]] = {
                        "date": anchor_capture["date"], "factor": anchor_capture["factor"],
                        "raw_response_sha256": hashlib.sha256(anchor_capture["raw"].to_json(orient="records", date_format="iso").encode()).hexdigest(),
                    }
                    save_parquet(anchor_capture["raw"], anchor_raw_path)
                    anchor_cache[item["code"]]["file_sha256"] = _file_sha256(anchor_raw_path)
                source_anchor = anchor_cache[item["code"]]
                if anchor_raw_path.is_file() and source_anchor.get("file_sha256") != _file_sha256(anchor_raw_path):
                    item["batch_invalidation_reason"] = "anchor_file_hash_mismatch"
                    raise RuntimeError("锚点因子文件哈希不符，原批次失效")
                if not anchor_raw_path.is_file():
                    anchor_capture = fetch_tushare_anchor_factor(item["ts_code"], manifest.get("anchor_date") or manifest["date_range"]["end"])
                    if anchor_capture["date"] != source_anchor["date"] or anchor_capture["factor"] != source_anchor["factor"]:
                        item["batch_invalidation_reason"] = "anchor_factor_changed"
                        raise RuntimeError("锚点因子返回变化，原批次失效")
                    save_parquet(anchor_capture["raw"], anchor_raw_path)
                    source_anchor["file_sha256"] = _file_sha256(anchor_raw_path)
                stem = f"{item['year']}"
                partial_paths = {
                    "baostock_raw": code_dir / f"{stem}_baostock_raw.parquet",
                    "baostock_normalized": code_dir / f"{stem}_baostock_normalized.parquet",
                    "anchor_factor_raw": anchor_raw_path,
                }
                baostock_capture = fetch_baostock_raw_daily(item["code"], item["exchange"], item["start"], item["end"], allow_empty_daily=True)
                baostock = baostock_capture["normalized"]
                save_parquet(baostock_capture["raw"], partial_paths["baostock_raw"])
                save_parquet(baostock, partial_paths["baostock_normalized"])
                tushare_capture = fetch_tushare_raw_daily(item["ts_code"], source_start, source_end, allow_empty_daily=True)
                tushare = tushare_capture["normalized"]
                partial_paths.update({
                    "tushare_daily_raw": code_dir / f"{stem}_tushare_daily_raw.parquet",
                    "tushare_factor_raw": code_dir / f"{stem}_tushare_factor_raw.parquet",
                    "tushare_normalized": code_dir / f"{stem}_tushare_normalized.parquet",
                })
                save_parquet(tushare_capture["daily_raw"], partial_paths["tushare_daily_raw"])
                save_parquet(tushare_capture["factors_raw"], partial_paths["tushare_factor_raw"])
                save_parquet(tushare, partial_paths["tushare_normalized"])
                suspension_capture = None
                non_trading_review = False
                exact_delist_no_quote = False
                if tushare.empty:
                    suspension_capture = fetch_tushare_suspensions(item["start"], item["end"])
                    suspension_capture = suspension_capture.loc[
                        suspension_capture["ts_code"].astype(str).str.upper().eq(item["ts_code"].upper())
                    ].copy().reset_index(drop=True)
                    partial_paths["tushare_suspensions_raw"] = code_dir / f"{stem}_tushare_suspensions_raw.parquet"
                    save_parquet(suspension_capture, partial_paths["tushare_suspensions_raw"])
                    suspension_dates = set(pd.to_datetime(suspension_capture["trade_date"], errors="coerce").dt.date)
                    factor_dates = set(pd.to_datetime(tushare_capture["factors_raw"].get("trade_date", pd.Series(dtype="object")), format="%Y%m%d", errors="coerce").dt.date)
                    baostock_dates = set(pd.to_datetime(baostock["date"], errors="coerce").dt.date)
                    delist_date = item.get("identity_interval", {}).get("delist_date")
                    delist_date = datetime.date.fromisoformat(str(delist_date)[:10]) if delist_date else None
                    delist_boundary_dates = {day for day in baostock_dates if day == delist_date}
                    status_zero = pd.to_numeric(baostock.get("tradestatus"), errors="coerce").eq(0).all()
                    volume_missing = pd.to_numeric(baostock.get("volume"), errors="coerce").isna().all()
                    non_trading_review = bool(len(baostock)) and status_zero and volume_missing and (
                        baostock_dates == (suspension_dates | delist_boundary_dates)
                    ) and (
                        factor_dates == suspension_dates
                        or factor_dates == (suspension_dates | delist_boundary_dates)
                    ) and len(delist_boundary_dates) <= 1
                    delist_date = item.get("identity_interval", {}).get("delist_date")
                    delist_date = datetime.date.fromisoformat(str(delist_date)[:10]) if delist_date else None
                    exact_delist_no_quote = (
                        delist_date is not None
                        and datetime.date.fromisoformat(item["start"]) == delist_date
                        and datetime.date.fromisoformat(item["end"]) == delist_date
                        and tushare.empty
                        and baostock.empty
                    )
                    non_trading_review = non_trading_review or exact_delist_no_quote
                    if not non_trading_review and manifest.get("denominator_batch_id"):
                        denominator_root = Path(__file__).resolve().parents[2] / "data/historical_repair/denominator" / manifest["denominator_batch_id"]
                        denominator = load_json(denominator_root / "manifest.json")
                        year_summary = next((entry for entry in denominator.get("year_summaries", []) if int(entry["year"]) == int(item["year"])), None) if denominator else None
                        expected_path = denominator_root / f"expected_keys_{item['year']}.parquet"
                        if not year_summary or not year_summary.get("denominator_verified") or not expected_path.is_file():
                            raise RuntimeError(f"G0 {item['year']} 年分母证据缺失，不能确认无可交易行情")
                        if _file_sha256(expected_path) != year_summary.get("expected_keys_sha256"):
                            raise RuntimeError(f"G0 {item['year']} 年分母证据哈希变化，不能确认无可交易行情")
                        expected_keys = load_parquet(expected_path)
                        expected_keys = expected_keys.loc[
                            expected_keys["code"].astype(str).str.zfill(6).eq(item["code"])
                            & pd.to_datetime(expected_keys["trade_date"], errors="coerce").dt.date.between(
                                datetime.date.fromisoformat(item["start"]), datetime.date.fromisoformat(item["end"]),
                            )
                        ].copy()
                        expected_dates = set(pd.to_datetime(expected_keys["trade_date"], errors="coerce").dt.date)
                        expected_suspended = (
                            not expected_keys.empty
                            and expected_keys["full_day_suspended"].fillna(False).astype(bool).all()
                            and not expected_keys["partial_suspension"].fillna(False).astype(bool).any()
                        )
                        no_trade_quantity = (
                            pd.to_numeric(baostock.get("volume"), errors="coerce").fillna(0).eq(0).all()
                            and pd.to_numeric(baostock.get("amount"), errors="coerce").fillna(0).eq(0).all()
                        )
                        non_trading_review = bool(expected_suspended and len(baostock)) and (
                            baostock_dates == expected_dates and factor_dates == expected_dates
                            and status_zero and no_trade_quantity
                        )
                    if not non_trading_review:
                        raise RuntimeError(
                            "TuShare pro.daily 为空，且 BaoStock 停牌/退市边界证据不足，需人工复核；"
                            f"BaoStock={len(baostock)}，复权因子={len(factor_dates)}，停牌={len(suspension_dates)}"
                        )
                comparison = tushare[["trade_date", "close", "adj_factor"]].merge(
                    baostock[["date", "close", "tradestatus"]],
                    left_on="trade_date", right_on="date", how="outer", suffixes=("_tushare", "_baostock"),
                    indicator=True, validate="one_to_one",
                )
                close_tushare = pd.to_numeric(comparison["close_tushare"], errors="coerce")
                close_baostock = pd.to_numeric(comparison["close_baostock"], errors="coerce")
                difference = (close_tushare - close_baostock).abs()
                comparison["raw_close_difference"] = difference
                comparison["raw_close_status"] = "matched"
                comparison.loc[comparison["_merge"] == "left_only", "raw_close_status"] = "missing_baostock"
                comparison.loc[comparison["_merge"] == "right_only", "raw_close_status"] = "missing_tushare"
                baostock_volume = pd.to_numeric(baostock["volume"], errors="coerce")
                baostock_amount = pd.to_numeric(baostock["amount"], errors="coerce")
                baostock_no_trade_quantity = (baostock_volume.isna() | baostock_volume.eq(0)) & (
                    baostock_amount.isna() | baostock_amount.eq(0)
                )
                baostock_tradestatus_zero = pd.to_numeric(baostock["tradestatus"], errors="coerce").eq(0)
                status_conflict_dates = set(
                    baostock.loc[baostock_tradestatus_zero & ~baostock_no_trade_quantity, "date"]
                    .dropna().dt.normalize()
                )
                comparison["baostock_tradestatus_conflict"] = pd.to_datetime(
                    comparison["date"], errors="coerce",
                ).dt.normalize().isin(status_conflict_dates)
                suspended = comparison["tradestatus"].fillna(-1).astype(float).eq(0)
                suspended &= ~comparison["baostock_tradestatus_conflict"]
                comparison.loc[suspended, "raw_close_status"] = "baostock_suspended"
                conflict = difference > (0.011 + close_tushare.abs() * 0.0005)
                comparison.loc[conflict, "raw_close_status"] = "raw_close_conflict"
                comparison["date_key"] = comparison["trade_date"].fillna(comparison["date"])
                stored_rows = pd.DataFrame(inputs["rows"])
                if not stored_rows.empty:
                    stored_rows["date_key"] = pd.to_datetime(stored_rows.pop("trade_date"))
                    stored_anchor_factor = pd.to_numeric(pd.Series([inputs["anchor"].get("adj_factor")]), errors="coerce").iloc[0]
                    stored_close = pd.to_numeric(stored_rows["close"], errors="coerce")
                    stored_factor = pd.to_numeric(stored_rows["adj_factor"], errors="coerce")
                    stored_rows["database_implied_raw_close"] = pd.NA
                    valid_stored_factor = stored_factor.notna() & (stored_factor > 0)
                    if pd.notna(stored_anchor_factor) and stored_anchor_factor > 0:
                        stored_rows.loc[valid_stored_factor, "database_implied_raw_close"] = (
                            stored_close.loc[valid_stored_factor] * stored_anchor_factor / stored_factor.loc[valid_stored_factor]
                        )
                    stored_rows = stored_rows.rename(columns={"close": "database_qfq_close", "adj_factor": "database_adj_factor"})
                    stored_rows["database_row_present"] = True
                    comparison = comparison.merge(
                        stored_rows[["date_key", "database_qfq_close", "database_adj_factor", "database_implied_raw_close", "database_row_present"]],
                        on="date_key", how="outer", validate="one_to_one",
                    )
                else:
                    comparison["database_row_present"] = False
                    comparison["database_implied_raw_close"] = pd.NA
                qfq_candidate = tushare.copy()
                for column in ("open", "high", "low", "close"):
                    qfq_candidate[column] = (
                        qfq_candidate[column] * qfq_candidate["adj_factor"] / source_anchor["factor"]
                    ).round(4)
                qfq_candidate["date"] = qfq_candidate["trade_date"]
                qfq_candidate["volume"] = (pd.to_numeric(qfq_candidate["vol"], errors="coerce") * 100).round().astype("int64")
                qfq_candidate["amount"] = (pd.to_numeric(qfq_candidate["amount"], errors="coerce") * 1000).round().astype("int64")
                qfq_candidate["pct_chg"] = qfq_candidate["close"].pct_change(fill_method=None) * 100
                if not qfq_candidate.empty:
                    source_first_pct = pd.to_numeric(tushare["pct_chg"], errors="coerce").iloc[0]
                    if pd.notna(source_first_pct):
                        qfq_candidate.loc[qfq_candidate.index[0], "pct_chg"] = source_first_pct
                matched_dates = set(
                    pd.to_datetime(
                        comparison.loc[comparison["raw_close_status"].eq("matched"), "date_key"],
                        errors="coerce",
                    ).dt.normalize().dropna()
                )
                qfq_candidate = qfq_candidate.loc[
                    pd.to_datetime(qfq_candidate["date"], errors="coerce").dt.normalize().isin(matched_dates)
                ].copy()
                qfq_candidate["amplitude"] = float("nan")
                qfq_candidate["turnover"] = float("nan")
                candidate_prices = qfq_candidate[["date", "close"]].copy()
                candidate_prices["date_key"] = pd.to_datetime(candidate_prices.pop("date"), errors="coerce").dt.normalize()
                candidate_prices = candidate_prices.rename(columns={"close": "source_qfq_close"})
                comparison = comparison.merge(candidate_prices, on="date_key", how="left", validate="one_to_one")
                database_qfq_close = pd.to_numeric(comparison.get("database_qfq_close"), errors="coerce")
                source_qfq_close = pd.to_numeric(comparison.get("source_qfq_close"), errors="coerce")
                comparison["database_qfq_close_difference"] = (database_qfq_close - source_qfq_close).abs()
                comparison["database_price_status"] = comparison.apply(
                    lambda row: classify_database_qfq_price(
                        row.get("database_row_present"), row.get("database_qfq_close"), row.get("source_qfq_close"),
                    ),
                    axis=1,
                )
                paths = {
                    "tushare_daily_raw": code_dir / f"{stem}_tushare_daily_raw.parquet",
                    "tushare_factor_raw": code_dir / f"{stem}_tushare_factor_raw.parquet",
                    "tushare_normalized": code_dir / f"{stem}_tushare_normalized.parquet",
                    "baostock_raw": code_dir / f"{stem}_baostock_raw.parquet",
                    "baostock_normalized": code_dir / f"{stem}_baostock_normalized.parquet",
                    "qfq_candidate": code_dir / f"{stem}_qfq_candidate.parquet",
                    "comparison": code_dir / f"{stem}_comparison.parquet",
                    "anchor_factor_raw": anchor_raw_path,
                }
                if suspension_capture is not None:
                    paths["tushare_suspensions_raw"] = code_dir / f"{stem}_tushare_suspensions_raw.parquet"
                save_parquet(tushare_capture["daily_raw"], paths["tushare_daily_raw"])
                save_parquet(tushare_capture["factors_raw"], paths["tushare_factor_raw"])
                save_parquet(tushare, paths["tushare_normalized"])
                save_parquet(baostock_capture["raw"], paths["baostock_raw"])
                save_parquet(baostock, paths["baostock_normalized"])
                save_parquet(qfq_candidate, paths["qfq_candidate"])
                save_parquet(comparison, paths["comparison"])
                if suspension_capture is not None:
                    save_parquet(suspension_capture, paths["tushare_suspensions_raw"])
                counts = comparison["raw_close_status"].value_counts().to_dict()
                database_counts = comparison["database_price_status"].value_counts().to_dict()
                status_conflict_dates = sorted(
                    pd.to_datetime(comparison.loc[comparison["baostock_tradestatus_conflict"], "date_key"], errors="coerce")
                    .dt.strftime("%Y-%m-%d").dropna().unique().tolist()
                )
                item.update({
                    "status": "non_trading_review" if non_trading_review else "quarantined_conflict" if counts.get("raw_close_conflict", 0) or database_counts.get("database_price_conflict", 0) else "collected_review",
                    "data_availability_status": "no_quote_on_registry_delist_date_requires_official_evidence" if exact_delist_no_quote else "no_tradable_rows_confirmed_by_suspension_and_delist_evidence" if non_trading_review else "tradable_rows_available",
                    "collected_at": now_iso(),
                    "source_anchor": source_anchor,
                    "source_request": {
                        "anchor_factor": {"api": "pro.adj_factor", "ts_code": item["ts_code"],
                                          "start_date": (datetime.date.fromisoformat(manifest["date_range"]["end"]) - datetime.timedelta(days=3650)).strftime("%Y%m%d"),
                                          "end_date": (manifest.get("anchor_date") or manifest["date_range"]["end"]).replace("-", ""),
                                          "fields": "ts_code,trade_date,adj_factor",
                                          "requested_anchor_date": manifest.get("anchor_date") or manifest["date_range"]["end"],
                                          "resolved_anchor_date": source_anchor["date"],
                                          "resolved_anchor_factor": source_anchor["factor"]},
                        "tushare": {
                            "daily": {"api": "pro.daily", "ts_code": item["ts_code"], "start_date": source_start, "end_date": source_end,
                                      "fields": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"},
                            "adj_factor": {"api": "pro.adj_factor", "ts_code": item["ts_code"], "start_date": source_start, "end_date": source_end,
                                           "fields": "ts_code,trade_date,adj_factor"},
                        },
                        "baostock": {"api": "query_history_k_data_plus", "symbol": f"{item['exchange']}.{item['code']}", "adjustflag": "3", "frequency": "d", "start_date": item["start"], "end_date": item["end"]},
                        "tushare_suspensions": {"api": "pro.suspend_d", "ts_code": item["ts_code"], "start_date": source_start, "end_date": source_end, "suspend_type": "S"} if suspension_capture is not None else None,
                    },
                    "tushare_rows": len(tushare), "baostock_rows": len(baostock),
                    "overlap_rows": int((comparison["_merge"] == "both").sum()),
                    "comparison_status_counts": {str(key): int(value) for key, value in counts.items()},
                    "baostock_tradestatus_conflict_rows": len(status_conflict_dates),
                    "baostock_tradestatus_conflict_dates": status_conflict_dates,
                    "database_price_status_counts": {str(key): int(value) for key, value in database_counts.items()},
                    "comparison_threshold": "0.011 + abs(TuShare raw close) * 0.0005; review flag only",
                    "database_price_comparison_threshold": (
                        f"abs(database QFQ close - source QFQ close) <= {HISTORICAL_QFQ_PRICE_TOLERANCE}"
                    ),
                    "normalized_response_sha256": {
                        "tushare": hashlib.sha256(tushare.to_json(orient="records", date_format="iso").encode()).hexdigest(),
                        "baostock": hashlib.sha256(baostock.to_json(orient="records", date_format="iso").encode()).hexdigest(),
                    },
                    "raw_response_sha256": {
                        "tushare_daily": hashlib.sha256(tushare_capture["daily_raw"].to_json(orient="records", date_format="iso").encode()).hexdigest(),
                        "tushare_adj_factor": hashlib.sha256(tushare_capture["factors_raw"].to_json(orient="records", date_format="iso").encode()).hexdigest(),
                        "tushare_anchor_factor": source_anchor["raw_response_sha256"],
                        "baostock_daily": hashlib.sha256(baostock_capture["raw"].to_json(orient="records", date_format="iso").encode()).hexdigest(),
                    },
                    "qfq_candidate_contract": "only dates with raw_close_status=matched in the dual-source comparison; BaoStock status-zero rows with nonzero trading quantities are retained with an explicit status-conflict flag for G2 corroboration; raw OHLC * per-date adj_factor / common Tushare factor on resolved manifest anchor date; OHLC rounded to 4 decimals; TuShare volume converted from lots to shares and amount from thousand yuan to yuan; pct_chg recomputed from QFQ close with the first shard row sourced from TuShare pct_chg to preserve its prior-session return; unavailable amplitude and turnover remain null; shadow only",
                    "files": {key: _file_sha256(path) for key, path in paths.items()},
                    "independent_source_verified": False,
                    "requires_review": True,
                })
                consecutive_baostock_login_failures = 0
            except Exception as error:
                error_text = str(error)
                visible_error = error_text if len(error_text) <= 500 else f"…{error_text[-499:]}"
                item.setdefault("attempt_log", []).append({"attempt": item.get("attempts", 1), "at": now_iso(), "error": visible_error})
                partial_files = {
                    name: _file_sha256(path)
                    for name, path in partial_paths.items()
                    if path.is_file()
                }
                item.update({"status": "failed", "error": visible_error, "failed_at": now_iso(),
                             "partial_files": partial_files})
                if "BaoStock 3 次隔离采集均失败" in error_text and "BaoStock 登录失败:" in error_text:
                    consecutive_baostock_login_failures += 1
                else:
                    consecutive_baostock_login_failures = 0
                if item.get("batch_invalidation_reason"):
                    manifest["status"] = "invalidated"
                    manifest["updated_at"] = now_iso()
                    save_json(manifest, manifest_path)
                    raise RuntimeError("锚点因子变化，批次已失效；请重建批次") from error
            manifest["updated_at"] = now_iso()
            manifest["status"] = "collecting"
            save_json(manifest, manifest_path)
            task_manager.update_progress(job_id, {
                "stage": "historical_shadow", "current": index,
                "total": len(manifest["work_items"]),
                "message": f"已处理 {item['code']} {item['year']}：{item['status']}",
            })
            if consecutive_baostock_login_failures >= 3:
                manifest["status"] = "partial"
                manifest["halt_reason"] = "baostock_login_circuit_open"
                manifest["updated_at"] = now_iso()
                save_json(manifest, manifest_path)
                task_manager.update_progress(job_id, {
                    "stage": "historical_shadow_paused", "current": index,
                    "total": len(manifest["work_items"]),
                    "message": "BaoStock 连续登录失败 3 个分片，已暂停剩余采集；检查网络路由后可续跑",
                })
                break
        states = [item.get("status") for item in manifest["work_items"]]
        manifest["status"] = "review_required" if all(state in ("collected_review", "quarantined_conflict", "non_trading_review") for state in states) else "partial"
        manifest["publishable"] = False
        manifest["updated_at"] = now_iso()
        save_json(manifest, manifest_path)
        return {"batch_id": batch_id, "status": manifest["status"], "work_items": len(states), "publishable": False}
    finally:
        with _batch_lock:
            if _active_batch_job == job_id:
                _active_batch_job = None


def start_shadow_batch(batch_id):
    global _active_batch_job
    manifest_path = _manifest_path(batch_id)
    manifest = load_json(manifest_path)
    if not manifest:
        raise ValueError("历史影子批次不存在")
    if manifest.get("pipeline_contract_version") != _PIPELINE_CONTRACT_VERSION:
        raise RuntimeError("影子批次处理契约版本过旧，只能查看；请按当前版本新建批次")
    if manifest.get("status") == "invalidated":
        raise ValueError("批次已失效，必须重新创建")
    with _batch_lock:
        if _active_batch_job:
            active_batch_id = _active_batch_job[2:] if _active_batch_job.startswith("hr") else None
            return {"status": "busy", "job_id": _active_batch_job, "batch_id": active_batch_id, "message": "另一个历史影子批次正在运行"}
        job_id = f"hr{batch_id}"
        if task_manager.is_active(job_id):
            return {"status": "busy", "job_id": job_id, "batch_id": batch_id, "message": "该批次任务正在运行"}
        _active_batch_job = job_id
    try:
        manifest["job_id"] = job_id
        manifest["status"] = "queued"
        manifest["updated_at"] = now_iso()
        save_json(manifest, manifest_path)
        def run_batch():
            try:
                return _run_shadow_batch(batch_id, job_id)
            finally:
                global _active_batch_job
                with _batch_lock:
                    if _active_batch_job == job_id:
                        _active_batch_job = None

        task_manager.submit_with_id(job_id, run_batch,
                                    f"历史行情影子采集 {batch_id}", pool="default")
    except Exception:
        with _batch_lock:
            _active_batch_job = None
        raise
    return {"job_id": job_id, "batch_id": batch_id, "status": "pending", "publishable": False}


def get_shadow_batch(batch_id):
    manifest = load_json(_manifest_path(batch_id))
    if not manifest:
        return None
    return manifest


def cancel_shadow_batch(batch_id):
    manifest = load_json(_manifest_path(batch_id))
    if not manifest:
        raise ValueError("历史影子批次不存在")
    job_id = manifest.get("job_id")
    if not job_id:
        return {"status": "not_running", "batch_id": batch_id}
    requested = task_manager.request_cancel(job_id)
    return {"status": "cancel_requested" if requested else "not_running", "batch_id": batch_id, "job_id": job_id}


def get_shadow_conflicts(batch_id, code, year, limit=500, denominator_batch_id=None):
    manifest_path = _manifest_path(batch_id)
    manifest = load_json(manifest_path)
    if not manifest:
        return {"status": "not_found", "rows": []}
    code = str(code).zfill(6)
    if not code.isdigit() or len(code) != 6:
        raise ValueError("股票代码格式无效")
    year = int(year)
    item = next((row for row in manifest.get("work_items", [])
                 if row.get("code") == code and int(row.get("year", 0)) == year), None)
    path = manifest_path.parent / "raw" / code / f"{year}_comparison.parquet"
    if not item or not path.is_file():
        return {"status": "not_ready", "batch_id": batch_id, "code": code, "year": year, "rows": []}
    if _file_sha256(path) != item.get("files", {}).get("comparison"):
        return {"status": "integrity_error", "batch_id": batch_id, "code": code, "year": year, "rows": []}
    frame = load_parquet(path)
    flagged = (frame["raw_close_status"] != "matched") | frame["database_price_status"].isin(("database_price_conflict", "unverifiable"))
    frame = frame.loc[flagged].head(max(1, min(int(limit), 2000)))
    if denominator_batch_id:
        if not denominator_batch_id.isalnum() or len(denominator_batch_id) != 12:
            raise ValueError("G0 分母批次 ID 格式无效")
        denominator_root = Path(__file__).resolve().parents[2] / "data/historical_repair/denominator" / denominator_batch_id
        denominator_manifest = load_json(denominator_root / "manifest.json")
        denominator_path = denominator_root / f"expected_keys_{year}.parquet"
        if denominator_manifest and denominator_manifest.get("batch_id") == denominator_batch_id and denominator_path.is_file():
            denominator = load_parquet(denominator_path)
            denominator = denominator.loc[denominator["code"].astype(str).str.zfill(6).eq(code),
                                          ["trade_date", "coverage_status", "full_day_suspended", "partial_suspension"]].copy()
            denominator["date_key"] = pd.to_datetime(denominator.pop("trade_date"), errors="coerce")
            frame["date_key"] = pd.to_datetime(frame["date_key"], errors="coerce")
            frame = frame.merge(denominator, on="date_key", how="left", validate="many_to_one")
            frame = frame.rename(columns={
                "coverage_status": "g0_coverage_status",
                "full_day_suspended": "g0_full_day_suspended",
                "partial_suspension": "g0_partial_suspension",
            })
        else:
            frame["g0_coverage_status"] = None
            frame["g0_full_day_suspended"] = None
            frame["g0_partial_suspension"] = None
    rows = json.loads(frame.to_json(orient="records", date_format="iso"))
    source_flagged = sum(int(value) for key, value in item.get("comparison_status_counts", {}).items() if key != "matched")
    database_flagged = sum(int(value) for key, value in item.get("database_price_status_counts", {}).items()
                           if key not in ("matched", "no_database_row"))
    return {
        "status": item.get("status"), "batch_id": batch_id, "code": code, "year": year,
        "denominator_batch_id": denominator_batch_id,
        "total_flagged": int(flagged.sum()), "source_flagged": source_flagged, "database_flagged": database_flagged,
        "rows": rows,
    }


def _release_root():
    return Path(__file__).resolve().parents[2] / "data/historical_repair/releases"


def _release_manifest_path(release_id):
    if not str(release_id).isalnum() or len(str(release_id)) != 12:
        raise ValueError("历史发布 ID 格式无效")
    return _release_root() / str(release_id) / "manifest.json"


def _verify_qlib_generation_release_files(manifest, generation_id, expected_side):
    from core.shared import config

    if not isinstance(generation_id, str) or not generation_id or Path(generation_id).name != generation_id:
        raise RuntimeError("Qlib受管代ID无效，拒绝核验恢复文件")
    generation_dir = (Path(config.QLIB_DATA_ROOT) / "generations" / generation_id).resolve()
    stage_dir = Path(manifest.get("qlib_generation_stage_dir") or "").resolve()
    parent_dir = (Path(config.QLIB_DATA_ROOT) / "generations" / str(manifest.get("qlib_generation_parent_id"))).resolve()
    for field, record in (manifest.get("qlib", {}).get("files") or {}).items():
        expected = record.get("after_sha256" if expected_side == "after" else "before_sha256")
        relative = record.get("generation_relative_path")
        if relative is None:
            for source_key, base in (
                ("source", stage_dir),
                ("source", generation_dir),
                ("source_before_generation", parent_dir),
            ):
                source = Path(record[source_key]).resolve() if record.get(source_key) else None
                if source is not None and source.is_relative_to(base):
                    relative = source.relative_to(base)
                    break
        if relative is None:
            raise RuntimeError(f"Qlib {field} 无法定位到受管代内文件，拒绝回滚")
        target = (generation_dir / relative).resolve()
        if not target.is_relative_to(generation_dir):
            raise RuntimeError(f"Qlib {field} 路径超出受管代目录，拒绝回滚")
        current = _file_sha256(target) if target.is_file() else None
        if current != expected:
            raise RuntimeError(f"Qlib {field} 在generation {generation_id} 中已变化，拒绝回滚")


def _discard_unpublished_qlib_generation_candidate(manifest, release_id, generation_id):
    from core.adapters.qlib.generation import (
        has_qlib_generation_pointer,
        qlib_generation_mutation_lock,
        resolve_qlib_data_dir,
    )
    from core.shared import config

    if not isinstance(generation_id, str) or len(generation_id) != 32 or not generation_id.isalnum():
        raise RuntimeError("Qlib未激活候选代ID无效，拒绝清理")
    generations_dir = (Path(config.QLIB_DATA_ROOT) / "generations").resolve()
    candidate_dir = (generations_dir / generation_id).resolve()
    if candidate_dir.parent != generations_dir:
        raise RuntimeError("Qlib未激活候选代路径无效，拒绝清理")
    with qlib_generation_mutation_lock(config.QLIB_DATA_ROOT):
        current_generation_id = (
            resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
            if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
        )
        if current_generation_id != manifest.get("qlib_generation_parent_id"):
            raise RuntimeError("Qlib当前代已变化，拒绝清理未激活候选代")
        if not candidate_dir.exists():
            return False
        generation_manifest = load_json(candidate_dir / "generation.json")
        metadata = (generation_manifest or {}).get("metadata") or {}
        if (
            (generation_manifest or {}).get("generation_id") != generation_id
            or generation_manifest.get("status") != "ready"
            or generation_manifest.get("parent_generation_id") != manifest.get("qlib_generation_parent_id")
            or metadata.get("operation") not in ("historical_quote_repair", "historical_quote_repair_batch")
            or metadata.get("release_id") != release_id
        ):
            raise RuntimeError("未激活Qlib候选代身份与当前release不符，保留现场")
        _verify_qlib_generation_release_files(manifest, generation_id, "after")
        shutil.rmtree(candidate_dir)
    return True


def _shadow_files_valid(batch_dir, item):
    code_dir = batch_dir / "raw" / item["code"]
    paths = {
        "tushare_daily_raw": code_dir / f"{item['year']}_tushare_daily_raw.parquet",
        "tushare_factor_raw": code_dir / f"{item['year']}_tushare_factor_raw.parquet",
        "tushare_normalized": code_dir / f"{item['year']}_tushare_normalized.parquet",
        "baostock_raw": code_dir / f"{item['year']}_baostock_raw.parquet",
        "baostock_normalized": code_dir / f"{item['year']}_baostock_normalized.parquet",
        "qfq_candidate": code_dir / f"{item['year']}_qfq_candidate.parquet",
        "comparison": code_dir / f"{item['year']}_comparison.parquet",
        "anchor_factor_raw": code_dir / "anchor_factor_raw.parquet",
        "tushare_suspensions_raw": code_dir / f"{item['year']}_tushare_suspensions_raw.parquet",
    }
    files = item.get("files") or {}
    if not files:
        return False
    return all(
        name in paths and paths[name].is_file() and _file_sha256(paths[name]) == expected
        for name, expected in files.items()
    )


def prepare_shadow_release(batch_id, code, year, denominator_batch_id, replace_conflicts=False, repair_fields=None):
    from core.adapters.qlib.historical_patch import load_stock_features, prepare_stock_patch

    code = str(code).strip().zfill(6)
    year = int(year)
    if not code.isdigit() or len(code) != 6:
        raise ValueError("股票代码格式无效")
    if not denominator_batch_id.isalnum() or len(denominator_batch_id) != 12:
        raise ValueError("G0 分母批次 ID 格式无效")
    batch_dir = _manifest_path(batch_id).parent
    shadow = load_json(batch_dir / "manifest.json")
    if not shadow or shadow.get("batch_id") != batch_id:
        raise ValueError("G1 双源影子批次不存在")
    if shadow.get("pipeline_contract_version") != _PIPELINE_CONTRACT_VERSION:
        raise ValueError("G1 处理契约过期，请按当前契约重建影子批次")
    if shadow.get("status") not in ("review_required", "partial"):
        raise ValueError("G1 批次尚未完成或仍在采集中")
    item = next((row for row in shadow.get("work_items", [])
                 if row.get("code") == code and int(row.get("year", 0)) == year), None)
    source_conflicts = int((item or {}).get("comparison_status_counts", {}).get("raw_close_conflict", 0))
    database_conflicts = int((item or {}).get("database_price_status_counts", {}).get("database_price_conflict", 0))
    database_only_conflict = bool(
        item and item.get("status") == "quarantined_conflict"
        and source_conflicts == 0 and database_conflicts > 0
    )
    replace_conflicts = bool(replace_conflicts)
    repair_fields = tuple(repair_fields or ())
    if repair_fields not in ((), ("amount",), ("volume",), ("volume", "amount")):
        raise ValueError("字段级量额修复字段组合无效")
    if repair_fields and replace_conflicts:
        raise ValueError("字段级修复不能同时启用冲突覆盖")
    if not item or (item.get("status") != "collected_review" and not database_only_conflict):
        raise ValueError("G1 分片存在双源冲突，不能进入发布预检")
    if replace_conflicts and source_conflicts:
        raise ValueError("TuShare 与 BaoStock 原始收盘价存在冲突，不能启用双源冲突修复")
    if not _shadow_files_valid(batch_dir, item):
        raise RuntimeError("G1 原始、规范化或候选文件哈希校验失败")

    project_root = Path(__file__).resolve().parents[2]
    historical_repair_root = project_root / "data/historical_repair"
    adjudication_root = historical_repair_root / "adjudication"
    trade_status_adjudication_path = adjudication_root / "verified_baostock_status_trade_adjudications.json"
    trade_status_adjudication_manifest = load_json(trade_status_adjudication_path)
    trade_status_adjudications = {}
    if (
        trade_status_adjudication_manifest
        and trade_status_adjudication_manifest.get("schema_version") == 1
        and trade_status_adjudication_manifest.get("type") == "verified_baostock_status_zero_trade_adjudication"
        and isinstance(trade_status_adjudication_manifest.get("records"), list)
    ):
        for record in trade_status_adjudication_manifest["records"]:
            if (
                record.get("status") != "verified"
                or record.get("classification") != "baostock_status_zero_but_dual_source_traded"
                or str(record.get("code") or "").zfill(6) != code
                or str(record.get("batch_id") or "") != batch_id
                or int(record.get("year") or 0) != year
            ):
                continue
            evidence_by_kind = {entry.get("kind"): entry for entry in record.get("evidence", [])}
            required_kinds = {
                "official_exchange_trade_notice",
                "independent_tushare_daily_quote",
                "independent_baostock_daily_quote",
            }
            if len(evidence_by_kind) != 3 or set(evidence_by_kind) != required_kinds:
                continue
            official_notice = evidence_by_kind["official_exchange_trade_notice"]
            official_path = (historical_repair_root / str(official_notice.get("path") or "")).resolve()
            tushare_capture = evidence_by_kind["independent_tushare_daily_quote"]
            baostock_capture = evidence_by_kind["independent_baostock_daily_quote"]
            expected_tushare_path = batch_dir / "raw" / code / f"{year}_tushare_normalized.parquet"
            expected_baostock_path = batch_dir / "raw" / code / f"{year}_baostock_normalized.parquet"
            if (
                not official_path.is_file()
                or _file_sha256(official_path) != official_notice.get("sha256")
                or (historical_repair_root / str(tushare_capture.get("path") or "")).resolve() != expected_tushare_path.resolve()
                or tushare_capture.get("sha256") != item.get("files", {}).get("tushare_normalized")
                or (historical_repair_root / str(baostock_capture.get("path") or "")).resolve() != expected_baostock_path.resolve()
                or baostock_capture.get("sha256") != item.get("files", {}).get("baostock_normalized")
            ):
                continue
            trade_status_adjudications[str(record.get("trade_date") or "")[:10]] = record

    denominator_root = Path(__file__).resolve().parents[2] / "data/historical_repair/denominator" / denominator_batch_id
    denominator = load_json(denominator_root / "manifest.json")
    if not denominator or denominator.get("batch_id") != denominator_batch_id:
        raise ValueError("G0 分母批次不存在")
    year_summary = next((row for row in denominator.get("year_summaries", []) if int(row.get("year", 0)) == year), None)
    if not year_summary or not year_summary.get("denominator_verified"):
        raise ValueError("G0 该年度分母未认证")
    expected_path = denominator_root / f"expected_keys_{year}.parquet"
    if not expected_path.is_file() or _file_sha256(expected_path) != year_summary.get("expected_keys_sha256"):
        raise RuntimeError("G0 预期日期文件缺失或哈希不符")

    g0 = load_parquet(expected_path)
    g0 = g0.loc[g0["code"].astype(str).str.zfill(6).eq(code)].copy()
    verified_delist_dates = set()
    excluded_delist_path = denominator_root / f"excluded_delist_dates_{year}.parquet"
    excluded_delist_hash = year_summary.get("excluded_delist_dates_sha256")
    if excluded_delist_hash:
        if not excluded_delist_path.is_file() or _file_sha256(excluded_delist_path) != excluded_delist_hash:
            raise RuntimeError("G0 已核验摘牌日期快照缺失或哈希不符")
        excluded_delist_rows = load_parquet(excluded_delist_path)
        required_exclusion_columns = {"code", "trade_date", "verified_delisted"}
        if not required_exclusion_columns.issubset(excluded_delist_rows.columns):
            raise RuntimeError("G0 已核验摘牌日期快照字段不完整")
        code_exclusions = excluded_delist_rows.loc[
            excluded_delist_rows["code"].astype(str).str.zfill(6).eq(code)
            & excluded_delist_rows["verified_delisted"].fillna(False).astype(bool)
        ]
        verified_delist_dates = set(
            pd.to_datetime(code_exclusions["trade_date"], errors="coerce")
            .dropna().dt.strftime("%Y-%m-%d").astype(str)
        )
    candidate_path = batch_dir / "raw" / code / f"{year}_qfq_candidate.parquet"
    comparison_path = batch_dir / "raw" / code / f"{year}_comparison.parquet"
    candidate = load_parquet(candidate_path)
    comparison = load_parquet(comparison_path)
    source_status_evidence = {"status": "not_required", "rows": []}
    source_status_blockers = []
    source_suspension_evidence = {"status": "not_applicable", "rows": [], "verified_dates": []}
    verified_suspension_clear_dates = []
    suspension_rows = comparison.loc[
        comparison.get("raw_close_status", pd.Series(index=comparison.index, dtype=object)).eq("baostock_suspended")
    ].copy()
    if not suspension_rows.empty:
        tushare_suspension = load_parquet(batch_dir / "raw" / code / f"{year}_tushare_normalized.parquet").copy()
        baostock_suspension = load_parquet(batch_dir / "raw" / code / f"{year}_baostock_normalized.parquet").copy()
        tushare_suspension["date_key"] = pd.to_datetime(tushare_suspension["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        baostock_suspension["date_key"] = pd.to_datetime(baostock_suspension["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        g0_suspension = g0.copy()
        g0_suspension["date_key"] = pd.to_datetime(g0_suspension["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        suspension_evidence_rows = []
        for row in suspension_rows.itertuples(index=False):
            date_value = pd.Timestamp(row.date_key).strftime("%Y-%m-%d")
            expected_rows = g0_suspension.loc[g0_suspension["date_key"].eq(date_value)]
            tushare_rows = tushare_suspension.loc[tushare_suspension["date_key"].eq(date_value)]
            baostock_rows = baostock_suspension.loc[baostock_suspension["date_key"].eq(date_value)]
            g0_full_day = bool(
                len(expected_rows) == 1
                and expected_rows.iloc[0]["full_day_suspended"]
                and not expected_rows.iloc[0]["partial_suspension"]
            )
            status_zero = bool(
                len(baostock_rows) == 1
                and pd.to_numeric(baostock_rows["tradestatus"], errors="coerce").iloc[0] == 0
            )
            no_trade_quantity = False
            if len(baostock_rows) == 1:
                volume = pd.to_numeric(baostock_rows["volume"], errors="coerce").iloc[0]
                amount = pd.to_numeric(baostock_rows["amount"], errors="coerce").iloc[0]
                no_trade_quantity = bool((pd.isna(volume) or volume == 0) and (pd.isna(amount) or amount == 0))
            accepted = g0_full_day and status_zero and no_trade_quantity and tushare_rows.empty
            evidence = {
                "date": date_value,
                "g0_full_day_suspended": g0_full_day,
                "tushare_rows": len(tushare_rows),
                "baostock_rows": len(baostock_rows),
                "baostock_tradestatus_zero": status_zero,
                "baostock_quantity_zero_or_null": no_trade_quantity,
                "accepted_for_qlib_clear": accepted,
            }
            suspension_evidence_rows.append(evidence)
            if accepted:
                verified_suspension_clear_dates.append(date_value)
        source_suspension_evidence = {
            "status": "verified" if len(verified_suspension_clear_dates) == len(suspension_evidence_rows) else "partially_verified",
            "rows": suspension_evidence_rows,
            "verified_dates": sorted(verified_suspension_clear_dates),
            "tushare_normalized_sha256": item.get("files", {}).get("tushare_normalized"),
            "baostock_normalized_sha256": item.get("files", {}).get("baostock_normalized"),
            "expected_keys_sha256": year_summary.get("expected_keys_sha256"),
        }
    status_conflict_mask = comparison.get(
        "baostock_tradestatus_conflict", pd.Series(False, index=comparison.index),
    ).fillna(False).astype(bool)
    status_conflict_rows = comparison.loc[status_conflict_mask].copy()
    if not status_conflict_rows.empty:
        tushare_status = load_parquet(batch_dir / "raw" / code / f"{year}_tushare_normalized.parquet").copy()
        baostock_status = load_parquet(batch_dir / "raw" / code / f"{year}_baostock_normalized.parquet").copy()
        tushare_status["date_key"] = pd.to_datetime(tushare_status["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        baostock_status["date_key"] = pd.to_datetime(baostock_status["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        g0_status = g0.copy()
        g0_status["date_key"] = pd.to_datetime(g0_status["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        evidence_rows = []
        used_trade_status_adjudications = {}
        for row in status_conflict_rows.itertuples(index=False):
            date_value = pd.Timestamp(row.date_key).strftime("%Y-%m-%d")
            trade_status_adjudication = trade_status_adjudications.get(date_value)
            expected_rows = g0_status.loc[g0_status["date_key"].eq(date_value)]
            tushare_rows = tushare_status.loc[tushare_status["date_key"].eq(date_value)]
            baostock_rows = baostock_status.loc[baostock_status["date_key"].eq(date_value)]
            expected_partial = bool(
                len(expected_rows) == 1
                and expected_rows.iloc[0]["partial_suspension"]
                and not expected_rows.iloc[0]["full_day_suspended"]
            )
            evidence = {
                "date": date_value,
                "raw_close_status": row.raw_close_status,
                "g0_partial_suspension": expected_partial,
                "tushare_rows": len(tushare_rows),
                "baostock_rows": len(baostock_rows),
                "volume_difference_shares": None,
                "amount_difference_cny": None,
                "ohlc_match": False,
                "quantity_match": False,
                "verified_trade_status_adjudication_id": trade_status_adjudication.get("adjudication_id") if trade_status_adjudication else None,
            }
            if len(tushare_rows) == 1 and len(baostock_rows) == 1:
                source_tushare = tushare_rows.iloc[0]
                source_baostock = baostock_rows.iloc[0]
                ohlc_match = True
                for field in ("open", "high", "low", "close"):
                    value_tushare = pd.to_numeric(pd.Series([source_tushare[field]]), errors="coerce").iloc[0]
                    value_baostock = pd.to_numeric(pd.Series([source_baostock[field]]), errors="coerce").iloc[0]
                    if pd.isna(value_tushare) or pd.isna(value_baostock):
                        ohlc_match = False
                        break
                    if abs(value_tushare - value_baostock) > 0.011 + abs(value_tushare) * 0.0005:
                        ohlc_match = False
                        break
                tushare_volume = pd.to_numeric(pd.Series([source_tushare["vol"]]), errors="coerce").iloc[0] * 100
                baostock_volume = pd.to_numeric(pd.Series([source_baostock["volume"]]), errors="coerce").iloc[0]
                tushare_amount = pd.to_numeric(pd.Series([source_tushare["amount"]]), errors="coerce").iloc[0] * 1000
                baostock_amount = pd.to_numeric(pd.Series([source_baostock["amount"]]), errors="coerce").iloc[0]
                volume_difference = abs(tushare_volume - baostock_volume)
                amount_difference = abs(tushare_amount - baostock_amount)
                amount_tolerance = max(100.0, abs(tushare_amount) * 0.001) if pd.notna(tushare_amount) else 100.0
                quantity_match = bool(
                    pd.notna(tushare_volume) and pd.notna(baostock_volume)
                    and pd.notna(tushare_amount) and pd.notna(baostock_amount)
                    and tushare_volume > 0 and baostock_volume > 0
                    and tushare_amount > 0 and baostock_amount > 0
                    and volume_difference <= HISTORICAL_VOLUME_TOLERANCE_SHARES
                    and amount_difference <= amount_tolerance
                )
                evidence.update({
                    "volume_difference_shares": float(volume_difference) if pd.notna(volume_difference) else None,
                    "amount_difference_cny": float(amount_difference) if pd.notna(amount_difference) else None,
                    "ohlc_match": bool(ohlc_match),
                    "quantity_match": quantity_match,
                })
            accepted = (
                evidence["raw_close_status"] == "matched"
                and (evidence["g0_partial_suspension"] or trade_status_adjudication is not None)
                and evidence["ohlc_match"]
                and evidence["quantity_match"]
            )
            evidence["accepted"] = accepted
            evidence_rows.append(evidence)
            if accepted and trade_status_adjudication:
                used_trade_status_adjudications[trade_status_adjudication["adjudication_id"]] = trade_status_adjudication
            if not accepted:
                source_status_blockers.append(_historical_release_blocker(
                    "DA04-SOURCE-STATUS-001",
                    f"BaoStock tradestatus=0 与成交数据矛盾，{date_value} 缺少经核验的交易日证据及双源 OHLC/成交量/成交额一致证据",
                ))
        source_status_evidence = {
            "status": "matched" if not source_status_blockers else "blocked",
            "rows": evidence_rows,
            "tolerance": {
                "raw_ohlc": "0.011 + abs(TuShare value) * 0.0005",
                "volume_shares": 1,
                "amount_cny": "max(100, TuShare amount * 0.001)",
            },
            "tushare_normalized_sha256": item.get("files", {}).get("tushare_normalized"),
            "baostock_normalized_sha256": item.get("files", {}).get("baostock_normalized"),
            "expected_keys_sha256": year_summary.get("expected_keys_sha256"),
            "trade_status_adjudication_manifest_sha256": _file_sha256(trade_status_adjudication_path) if used_trade_status_adjudications and trade_status_adjudication_path.is_file() else None,
            "verified_trade_status_adjudications": list(used_trade_status_adjudications.values()),
        }
    source_quantity_evidence = None
    quantity_evidence_passed = False
    quantity_evidence_blocker = None
    if replace_conflicts:
        tushare = load_parquet(batch_dir / "raw" / code / f"{year}_tushare_normalized.parquet").copy()
        baostock = load_parquet(batch_dir / "raw" / code / f"{year}_baostock_normalized.parquet").copy()
        tushare["date"] = pd.to_datetime(tushare["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        baostock["date"] = pd.to_datetime(baostock["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        quantity_comparison = tushare[["date", "vol", "amount"]].merge(
            baostock[["date", "volume", "amount"]],
            on="date", how="outer", suffixes=("_tushare", "_baostock"), validate="one_to_one",
        )
        tushare_volume = pd.to_numeric(quantity_comparison["vol"], errors="coerce") * 100
        baostock_volume = pd.to_numeric(quantity_comparison["volume"], errors="coerce")
        tushare_amount = pd.to_numeric(quantity_comparison["amount_tushare"], errors="coerce") * 1000
        baostock_amount = pd.to_numeric(quantity_comparison["amount_baostock"], errors="coerce")
        volume_comparable = tushare_volume.notna() & baostock_volume.notna()
        amount_comparable = tushare_amount.notna() & baostock_amount.notna()
        volume_mismatch = volume_comparable & (
            (tushare_volume - baostock_volume).abs() > HISTORICAL_VOLUME_TOLERANCE_SHARES
        )
        amount_tolerance = (tushare_amount.abs() * 0.001).clip(lower=100)
        amount_mismatch = amount_comparable & ((tushare_amount - baostock_amount).abs() > amount_tolerance)
        candidate_date_series = pd.to_datetime(candidate["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        candidate_dates = set(candidate_date_series.dropna().astype(str))
        tushare_dates = set(tushare["date"].dropna().astype(str))
        baostock_dates = set(baostock["date"].dropna().astype(str))
        expected_dates = pd.to_datetime(g0["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        full_day_suspended = g0["full_day_suspended"].fillna(False).astype(bool)
        partial_suspension = g0["partial_suspension"].fillna(False).astype(bool)
        expected_tradable_dates = set(expected_dates.loc[~full_day_suspended].dropna().astype(str))
        verified_full_day_suspended_dates = set(
            expected_dates.loc[full_day_suspended & ~partial_suspension].dropna().astype(str)
        )
        baostock_only_dates = baostock_dates - tushare_dates
        baostock_tradestatus = pd.to_numeric(
            baostock.get("tradestatus", pd.Series(index=baostock.index, dtype=float)), errors="coerce",
        )
        baostock_volume = pd.to_numeric(baostock["volume"], errors="coerce")
        baostock_amount_values = pd.to_numeric(baostock["amount"], errors="coerce")
        no_trade_quantity = (baostock_volume.isna() | baostock_volume.eq(0)) & (
            baostock_amount_values.isna() | baostock_amount_values.eq(0)
        )
        verified_baostock_suspension_dates = set(
            baostock.loc[
                baostock_tradestatus.eq(0)
                & no_trade_quantity
                & baostock["date"].isin(verified_full_day_suspended_dates),
                "date",
            ].dropna().astype(str)
        )
        matched_quantity_dates = set(quantity_comparison.loc[
            volume_comparable & amount_comparable & ~volume_mismatch & ~amount_mismatch, "date",
        ].dropna().astype(str))
        quantity_evidence_passed = (
            not quantity_comparison["date"].isna().any()
            and not quantity_comparison["date"].duplicated().any()
            and not candidate_date_series.isna().any()
            and candidate_dates == tushare_dates
            and candidate_dates.issubset(expected_tradable_dates)
            and matched_quantity_dates == candidate_dates
            and baostock_only_dates == verified_baostock_suspension_dates
            and not (tushare_dates - baostock_dates)
        )
        source_quantity_evidence = {
            "status": "matched" if quantity_evidence_passed else "blocked",
            "candidate_rows": len(candidate_dates),
            "compared_rows": len(quantity_comparison),
            "matched_rows": len(matched_quantity_dates),
            "expected_tradable_rows": len(expected_tradable_dates),
            "expected_tradable_dates_match": candidate_dates == expected_tradable_dates,
            "candidate_dates_within_expected_tradable_dates": candidate_dates.issubset(expected_tradable_dates),
            "verified_suspended_extra_rows": len(verified_baostock_suspension_dates),
            "verified_suspended_extra_dates": sorted(verified_baostock_suspension_dates),
            "suspension_evidence_sha256": year_summary.get("expected_keys_sha256"),
            "volume_mismatch_rows": int(volume_mismatch.sum()),
            "amount_mismatch_rows": int(amount_mismatch.sum()),
            "volume_tolerance_shares": HISTORICAL_VOLUME_TOLERANCE_SHARES,
            "amount_tolerance": "max(100 CNY, TuShare amount * 0.001)",
            "tushare_normalized_sha256": item.get("files", {}).get("tushare_normalized"),
            "baostock_normalized_sha256": item.get("files", {}).get("baostock_normalized"),
        }
        if not quantity_evidence_passed:
            quantity_evidence_blocker = "TuShare 与 BaoStock 可交易日成交量/成交额未逐日匹配，或 BaoStock 额外停牌行缺少 G0 全日停牌证据"
        else:
            quantity_evidence_blocker = None
    inputs = HistoricalRepairRepo.get_shard_inputs(code, item["start"], item["end"])
    input_current = inputs["fingerprint"] == item.get("input_sha256")
    db_anchor = inputs.get("anchor") or {}
    anchor_date = db_anchor.get("trade_date")
    db_anchor_factor = float(db_anchor.get("adj_factor") or 0)
    source_anchor_date = str((item.get("source_anchor") or {}).get("date") or "")[:10] or None
    source_anchor_factor = float((item.get("source_anchor") or {}).get("factor") or 0)
    anchor_mismatch = bool(
        anchor_date and str(anchor_date)[:10] <= str(item["end"])
        and db_anchor_factor > 0 and source_anchor_factor > 0
        and not math.isclose(db_anchor_factor, source_anchor_factor, rel_tol=1e-12, abs_tol=1e-9)
    )
    anchor_compatibility = {
        "status": "no_existing_anchor",
        "database_anchor_date": str(anchor_date)[:10] if anchor_date else None,
        "database_anchor_factor": db_anchor_factor or None,
        "source_anchor_date": source_anchor_date,
        "source_anchor_factor": source_anchor_factor or None,
    }
    if anchor_mismatch and not repair_fields:
        anchor_compatibility["status"] = "full_stock_reanchor_pending_verification"
    elif anchor_date and str(anchor_date)[:10] <= str(item["end"]):
        anchor_compatibility["status"] = "same_factor_verified"
    elif anchor_date:
        anchor_compatibility["status"] = "candidate_before_database_anchor"

    blockers = []
    if quantity_evidence_blocker and repair_fields not in (("amount",), ("volume",), ("volume", "amount")):
        blockers.append(_historical_release_blocker("DA04-QUANTITY-EVIDENCE-001", quantity_evidence_blocker))
    elif source_quantity_evidence is not None and not quantity_evidence_passed and not repair_fields:
        blockers.append(_historical_release_blocker("DA04-QUANTITY-EVIDENCE-001", "TuShare 与 BaoStock 成交量/成交额未逐日匹配，不能覆盖现库冲突"))
    blockers.extend(source_status_blockers)
    if not input_current:
        blockers.append(_historical_release_blocker("DA04-INPUT-BASELINE-001", "现库分片指纹与 G1 采集时基线不同，需刷新影子批次"))
    full_inputs = None
    full_reanchor = None
    candidate_release = candidate.copy()
    stored_for_assessment = pd.DataFrame(
        inputs["rows"],
        columns=("trade_date", "open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover", "adj_factor"),
    )
    assessment = None
    if anchor_mismatch:
        full_inputs = HistoricalRepairRepo.get_full_stock_inputs(code)
        full_anchor = full_inputs.get("anchor") or {}
        if (
            not full_anchor.get("trade_date")
            or str(full_anchor.get("trade_date"))[:10] != str(anchor_date)[:10]
            or not math.isclose(float(full_anchor.get("adj_factor") or 0), db_anchor_factor, rel_tol=1e-12, abs_tol=1e-9)
        ):
            blockers.append(_historical_release_blocker("DA04-REANCHOR-SNAPSHOT-001", "整股快照与 G1 锚点不一致，拒绝重锚预检"))
        source_factor_path = batch_dir / "raw" / code / "anchor_factor_raw.parquet"
        source_factors = load_parquet(source_factor_path)
        source_factors = source_factors.copy()
        source_factors["trade_date"] = pd.to_datetime(
            source_factors["trade_date"], format="%Y%m%d", errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        source_factors["adj_factor"] = pd.to_numeric(source_factors["adj_factor"], errors="coerce")
        old_anchor_rows = source_factors.loc[
            source_factors["trade_date"].eq(str(anchor_date)[:10]), "adj_factor",
        ].dropna()
        source_old_anchor_factor = float(old_anchor_rows.iloc[-1]) if not old_anchor_rows.empty else 0.0
        candidate_last = candidate_release.sort_values("date").iloc[-1]
        target_anchor_date = pd.Timestamp(candidate_last["date"]).strftime("%Y-%m-%d")
        target_anchor_factor = float(pd.to_numeric(pd.Series([candidate_last["adj_factor"]]), errors="coerce").iloc[0])
        source_anchor_matches_database_date = (
            source_old_anchor_factor > 0
            and math.isclose(source_old_anchor_factor, db_anchor_factor, rel_tol=1e-12, abs_tol=1e-9)
        )
        if not full_inputs["rows"]:
            blockers.append(_historical_release_blocker("DA04-REANCHOR-PREIMAGE-001", "现库没有整股行情 pre-image，拒绝锚点重建"))
        if not source_anchor_matches_database_date:
            blockers.append(_historical_release_blocker("DA04-REANCHOR-FACTOR-001", "TuShare 历史因子与现库旧锚点日期不一致，无法证明只需统一重锚"))
        if target_anchor_factor <= 0 or not math.isfinite(target_anchor_factor):
            blockers.append(_historical_release_blocker("DA04-REANCHOR-TARGET-FACTOR-001", "候选末日没有合法复权因子，无法确定新锚点"))
        if not blockers:
            candidate_scale = source_anchor_factor / target_anchor_factor
            database_scale = db_anchor_factor / target_anchor_factor
            for field in ("open", "high", "low", "close"):
                candidate_release[field] = pd.to_numeric(candidate_release[field], errors="coerce") * candidate_scale
                stored_for_assessment[field] = pd.to_numeric(stored_for_assessment[field], errors="coerce") * database_scale
            assessment = assess_quote_shard_release(
                g0, candidate_release, comparison, stored_for_assessment,
                replace_price_conflicts=replace_conflicts and quantity_evidence_passed,
            )
            if assessment["status"] != "ready":
                blockers.extend(_historical_release_blocker("DA04-REANCHOR-ASSESSMENT-001", message)
                                for message in assessment["blockers"])
                blockers.append(_historical_release_blocker("DA04-REANCHOR-ASSESSMENT-001", "旧锚点缩放后仍与双源候选不一致，拒绝整股重锚"))
            else:
                anchor_compatibility.update({
                    "status": "full_stock_reanchor_verified",
                    "source_factor_at_database_anchor_date": source_old_anchor_factor,
                    "target_anchor_date": target_anchor_date,
                    "target_anchor_factor": target_anchor_factor,
                    "database_rebase_ratio": database_scale,
                    "candidate_rebase_ratio": candidate_scale,
                })
                full_reanchor = {
                    "old_anchor_date": str(anchor_date)[:10],
                    "old_anchor_factor": db_anchor_factor,
                    "source_factor_at_old_anchor_date": source_old_anchor_factor,
                    "target_anchor_date": target_anchor_date,
                    "target_anchor_factor": target_anchor_factor,
                    "database_rebase_ratio": database_scale,
                    "candidate_rebase_ratio": candidate_scale,
                    "full_stock_fingerprint": full_inputs["fingerprint"],
                    "full_stock_rows": len(full_inputs["rows"]),
                }
        else:
            anchor_compatibility["status"] = "full_stock_reanchor_blocked"
    elif repair_fields in (("amount",), ("volume",), ("volume", "amount")):
        tushare_amounts = load_parquet(batch_dir / "raw" / code / f"{year}_tushare_normalized.parquet").copy()
        baostock_amounts = load_parquet(batch_dir / "raw" / code / f"{year}_baostock_normalized.parquet").copy()
        tushare_amounts["date_key"] = pd.to_datetime(tushare_amounts["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        baostock_amounts["date_key"] = pd.to_datetime(baostock_amounts["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        ts_amount = pd.to_numeric(tushare_amounts["amount"], errors="coerce") * 1000
        bs_amount = pd.to_numeric(baostock_amounts["amount"], errors="coerce")
        ts_volume = pd.to_numeric(tushare_amounts["vol"], errors="coerce") * 100
        bs_volume = pd.to_numeric(baostock_amounts["volume"], errors="coerce")
        ts_by_date = dict(zip(tushare_amounts["date_key"], ts_amount))
        bs_by_date = dict(zip(baostock_amounts["date_key"], bs_amount))
        ts_volume_by_date = dict(zip(tushare_amounts["date_key"], ts_volume))
        bs_volume_by_date = dict(zip(baostock_amounts["date_key"], bs_volume))
        amount_evidence = []
        for date_key in sorted(set(ts_by_date) | set(bs_by_date)):
            first, second = ts_by_date.get(date_key), bs_by_date.get(date_key)
            first_volume, second_volume = ts_volume_by_date.get(date_key), bs_volume_by_date.get(date_key)
            tolerance = max(100.0, abs(first) * 0.001) if first is not None and pd.notna(first) else 100.0
            volume_match = bool(first_volume is not None and second_volume is not None
                                and pd.notna(first_volume) and pd.notna(second_volume)
                                and first_volume > 0 and second_volume > 0
                                and abs(first_volume - second_volume) <= HISTORICAL_VOLUME_TOLERANCE_SHARES)
            amount_match = bool(first is not None and second is not None and pd.notna(first) and pd.notna(second)
                                and first > 0 and second > 0 and abs(first - second) <= tolerance)
            amount_evidence.append({
                "date": date_key,
                "tushare_amount_cny": float(first) if first is not None and pd.notna(first) else None,
                "baostock_amount_cny": float(second) if second is not None and pd.notna(second) else None,
                "tushare_volume_shares": float(first_volume) if first_volume is not None and pd.notna(first_volume) else None,
                "baostock_volume_shares": float(second_volume) if second_volume is not None and pd.notna(second_volume) else None,
                "amount_matched": amount_match,
                "volume_matched": volume_match,
                "matched": (amount_match if "amount" in repair_fields else True)
                           and (volume_match if "volume" in repair_fields else True),
            })
        comparison_status_counts = item.get("comparison_status_counts", {})
        unreviewed_source_status_rows = sum(
            int(count)
            for status, count in comparison_status_counts.items()
            if status not in ("matched", "baostock_suspended")
        )
        comparison_status_ok = (
            source_conflicts == 0
            and int(comparison_status_counts.get("matched", 0)) > 0
            and unreviewed_source_status_rows == 0
            and not source_status_blockers
        )
        all_amounts_match = bool(amount_evidence) and all(row["matched"] for row in amount_evidence)
        candidate_dates = set(pd.to_datetime(candidate_release["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())
        g0_dates = pd.to_datetime(g0["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        expected_dates = set(g0.loc[
            g0_dates.between(item["start"], item["end"])
            & ~g0["full_day_suspended"].fillna(False).astype(bool), "trade_date",
        ].pipe(pd.to_datetime, errors="coerce").dt.strftime("%Y-%m-%d").dropna())
        comparison_date_column = "date_key" if "date_key" in comparison else "trade_date"
        invalid_date_axes = invalid_or_duplicate_date_axes({
            "G0": g0.get("trade_date"),
            "candidate": candidate_release.get("date"),
            "comparison": comparison.get(comparison_date_column),
            "TuShare": tushare_amounts.get("trade_date"),
            "BaoStock": baostock_amounts.get("date"),
        })
        field_blockers = [
            _historical_release_blocker(
                "DA04-DATE-KEY-001",
                f"{name}日期轴存在无效值或重复键，字段级量额修复不能按集合/字典合并",
            )
            for name in invalid_date_axes
        ]
        if not comparison_status_ok:
            field_blockers.append("DA04-SRC-STATUS-001: 双源比较存在未审阅来源状态，字段级量额发布必须先完成状态核验")
        if not all_amounts_match:
            field_blockers.append(_historical_release_blocker("DA04-QUANTITY-EVIDENCE-002", "TuShare 与 BaoStock 成交额或成交量未逐日匹配"))
        if candidate_dates != expected_dates:
            field_blockers.append(_historical_release_blocker("DA04-G0-COVERAGE-001", "候选交易日与 G0 可交易日分母不一致"))
        assessment = {
            "status": "ready" if not field_blockers else "blocked",
            "blockers": field_blockers,
            "repair_fields": list(repair_fields),
            "source_amount_evidence": {
                "status": "matched" if all_amounts_match else "blocked",
                "rows": len(amount_evidence),
            "matched_rows": sum(row["matched"] for row in amount_evidence),
                "tolerance": "max(100 CNY, TuShare amount * 0.001)",
                "source_files_sha256": {
                    "tushare": item.get("files", {}).get("tushare_normalized"),
                    "baostock": item.get("files", {}).get("baostock_normalized"),
                },
            },
            "candidate_rows": len(candidate_release),
            "matched_source_rows": len(candidate_release) if all_amounts_match and comparison_status_ok else 0,
        }
        blockers.extend(_historical_release_blocker("DA04-QUOTE-ASSESSMENT-001", message)
                        for message in assessment["blockers"])
    else:
        assessment = assess_quote_shard_release(
            g0, candidate_release, comparison, stored_for_assessment,
            replace_price_conflicts=replace_conflicts and quantity_evidence_passed,
            verified_delist_dates=verified_delist_dates,
        )
        blockers.extend(_historical_release_blocker("DA04-QUOTE-ASSESSMENT-001", message)
                        for message in assessment["blockers"])

    if repair_fields in (("amount",), ("volume",), ("volume", "amount")):
        stored_by_date = {str(row["trade_date"])[:10]: row for row in inputs["rows"]}
        amount_rows = []
        for row in candidate_release.itertuples(index=False):
            date_value = row.date.strftime("%Y-%m-%d")
            stored = stored_by_date.get(date_value)
            if stored is None or all(stored.get(field) is not None for field in repair_fields):
                continue
            merged = {**row._asdict(), "date": row.date}
            for field in ("volume", "amount"):
                if stored.get(field) is not None:
                    merged[field] = stored.get(field)
            amount_rows.append(merged)
        if not amount_rows:
            blockers.append(_historical_release_blocker("DA04-EMPTY-REPAIR-001", "现库没有可填补的空成交额，拒绝发布空修复"))
        else:
            candidate_release = pd.DataFrame(amount_rows)
            field_values = {field: [] for field in ("open", "high", "low", "close", "volume", "amplitude", "pct_chg", "turnover", "adj_factor")}
            vwap_values = []
            for row in candidate_release.itertuples(index=False):
                stored = stored_by_date[row.date.strftime("%Y-%m-%d")]
                for field in field_values:
                    value = stored.get(field)
                    field_values[field].append(float(value) if value is not None and pd.notna(value) else float("nan"))
                amount = pd.to_numeric(pd.Series([row.amount]), errors="coerce").iloc[0]
                volume = pd.to_numeric(pd.Series([stored.get("volume")]), errors="coerce").iloc[0]
                factor = pd.to_numeric(pd.Series([stored.get("adj_factor")]), errors="coerce").iloc[0]
                vwap_values.append(amount / volume * factor / source_anchor_factor if pd.notna(amount) and pd.notna(volume) and volume > 0 and pd.notna(factor) and source_anchor_factor > 0 else float("nan"))
            for field, values in field_values.items():
                candidate_release[field] = values
            candidate_release["vwap"] = vwap_values
            candidate_release["factor"] = 1.0

    release_id = uuid.uuid4().hex[:12]
    release_dir = _release_root() / release_id
    preimage_dir = release_dir / "preimage"
    qlib_backup_dir = preimage_dir / "qlib"
    qlib_stage_dir = release_dir / "qlib_stage"
    release_dir.mkdir(parents=True, exist_ok=False)
    from core.shared import config
    from core.adapters.qlib.generation import has_qlib_generation_pointer, resolve_qlib_data_dir
    qlib_managed = has_qlib_generation_pointer(config.QLIB_DATA_ROOT)
    qlib_dir = resolve_qlib_data_dir(config.QLIB_DATA_ROOT)
    qlib_generation_id = qlib_dir.name if qlib_managed else None
    if full_reanchor:
        before_rows = full_inputs["backup_rows"]
        full_dates = [str(row["trade_date"])[:10] for row in before_rows]
        rollback_range = {
            "start": min([*full_dates, str(candidate_release["date"].min())[:10]]),
            "end": max([*full_dates, str(candidate_release["date"].max())[:10]]),
        }
        candidate_path = release_dir / "candidate_qfq_rebased.parquet"
        save_parquet(candidate_release, candidate_path)
        database_scale = full_reanchor["database_rebase_ratio"]
        qlib_preimage = load_stock_features(qlib_dir, code)
        if qlib_preimage["status"] != "ready":
            blockers.extend(_historical_release_blocker("DA04-QLIB-PREIMAGE-001", message)
                            for message in qlib_preimage["blockers"])
        qlib_frame = qlib_preimage["frame"].copy()
        if not blockers:
            for field in ("open", "high", "low", "close", "vwap"):
                qlib_frame[field] = pd.to_numeric(qlib_frame[field], errors="coerce") * database_scale
            qlib_frame["date"] = pd.to_datetime(qlib_frame["date"], errors="coerce").dt.normalize()
            qlib_frame = qlib_frame.set_index("date")
        for row in candidate_release.itertuples(index=False):
            date_value = pd.Timestamp(row.date).normalize()
            if blockers:
                break
            if date_value not in qlib_frame.index:
                blockers.append(_historical_release_blocker("DA04-QLIB-CALENDAR-001", f"Qlib 日历缺少候选交易日 {date_value:%Y-%m-%d}"))
                break
            for field in ("open", "high", "low", "close", "volume", "amount"):
                value = pd.to_numeric(pd.Series([getattr(row, field)]), errors="coerce").iloc[0]
                if pd.notna(value):
                    qlib_frame.loc[date_value, field] = float(value)
            amount = pd.to_numeric(pd.Series([qlib_frame.loc[date_value, "amount"]]), errors="coerce").iloc[0]
            volume = pd.to_numeric(pd.Series([qlib_frame.loc[date_value, "volume"]]), errors="coerce").iloc[0]
            factor = pd.to_numeric(pd.Series([row.adj_factor]), errors="coerce").iloc[0]
            target_anchor = full_reanchor["target_anchor_factor"]
            qlib_frame.loc[date_value, "vwap"] = (
                amount / volume * factor / target_anchor
                if pd.notna(amount) and pd.notna(volume) and volume > 0 and pd.notna(factor)
                else float("nan")
            )
            qlib_frame.loc[date_value, "factor"] = 1.0
        if qlib_preimage["status"] == "ready":
            desired = qlib_frame.reset_index()
            desired = desired.loc[desired[["open", "high", "low", "close", "volume", "amount", "vwap"]].notna().any(axis=1)].copy()
        else:
            desired = pd.DataFrame(columns=("date", "open", "high", "low", "close", "volume", "amount", "vwap", "factor"))
    else:
        before_rows = HistoricalRepairRepo.get_shard_backup(code, item["start"], item["end"])
        rollback_range = {"start": item["start"], "end": item["end"]}
        desired_rows = []
        qfq_anchor = source_anchor_factor
        stored_by_date = {str(row["trade_date"])[:10]: row for row in inputs["rows"]}
        for row in candidate_release.itertuples(index=False):
            date_value = row.date.strftime("%Y-%m-%d")
            old = stored_by_date.get(date_value, {})
            merged = {"date": row.date}
            for field in ("open", "high", "low", "close", "volume", "amount"):
                old_value = old.get(field)
                candidate_value = getattr(row, field)
                replace_field = replace_conflicts and field in ("open", "high", "low", "close", "volume", "amount")
                selected = candidate_value if replace_field else old_value if old_value is not None and pd.notna(old_value) else candidate_value
                if pd.notna(selected):
                    if field in ("open", "high", "low", "close", "amount"):
                        selected = round(float(selected), 2)
                    elif field == "volume":
                        selected = int(round(float(selected)))
                merged[field] = selected
            old_factor = old.get("adj_factor")
            selected_factor = row.adj_factor if replace_conflicts else old_factor if old_factor is not None and pd.notna(old_factor) else row.adj_factor
            merged["adj_factor"] = round(float(selected_factor), 6) if pd.notna(selected_factor) else float("nan")
            amount = pd.to_numeric(pd.Series([merged["amount"]]), errors="coerce").iloc[0]
            volume = pd.to_numeric(pd.Series([merged["volume"]]), errors="coerce").iloc[0]
            factor = pd.to_numeric(pd.Series([merged["adj_factor"]]), errors="coerce").iloc[0]
            merged["vwap"] = amount / volume * factor / qfq_anchor if pd.notna(amount) and pd.notna(volume) and volume > 0 and pd.notna(factor) else float("nan")
            merged["factor"] = 1.0
            desired_rows.append(merged)
        desired = pd.DataFrame(desired_rows)
        if repair_fields in (("amount",), ("volume",), ("volume", "amount")):
            candidate_path = release_dir / "candidate_amount_patch.parquet"
            save_parquet(candidate_release, candidate_path)
    before_path = preimage_dir / "daily_quotes.parquet"
    save_parquet(pd.DataFrame(before_rows), before_path)
    allow_qlib_conflict_replacement = (
        assessment.get("status") == "ready"
        and assessment.get("matched_source_rows") == assessment.get("candidate_rows")
    )
    qlib_result = prepare_stock_patch(
        qlib_dir, code, desired, qlib_stage_dir,
        allow_conflicts=allow_qlib_conflict_replacement,
        clear_dates=verified_suspension_clear_dates,
        fields=(tuple(field for field in ("volume", "amount") if field in repair_fields) + ("vwap",))
               if repair_fields and "amount" in repair_fields and desired["vwap"].notna().all()
               else tuple(field for field in ("volume", "amount") if field in repair_fields)
               if repair_fields else None,
    )
    blockers.extend(_historical_release_blocker("DA04-QLIB-CANDIDATE-001", message)
                    for message in qlib_result["blockers"])
    database_repair_required = False
    if full_reanchor:
        database_repair_required = not math.isclose(
            float(full_reanchor["database_rebase_ratio"]), 1.0, rel_tol=1e-12, abs_tol=1e-12,
        )
        stored_by_date = {
            str(row["trade_date"])[:10]: row for row in full_inputs["backup_rows"]
        }
        fields_to_fill = ("open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "turnover", "adj_factor")
        for row in candidate_release.itertuples(index=False):
            date_value = row.date.strftime("%Y-%m-%d")
            stored = stored_by_date.get(date_value)
            if stored is None:
                database_repair_required = True
                break
            if any(
                pd.isna(stored.get(field)) and pd.notna(getattr(row, field))
                for field in fields_to_fill
            ):
                database_repair_required = True
                break
            factor = pd.to_numeric(pd.Series([getattr(row, "adj_factor")]), errors="coerce").iloc[0]
            stored_factor = pd.to_numeric(pd.Series([stored.get("adj_factor")]), errors="coerce").iloc[0]
            if pd.notna(factor) and (
                pd.isna(stored_factor)
                or not math.isclose(float(stored_factor), float(factor), rel_tol=1e-12, abs_tol=0.0000005)
            ):
                database_repair_required = True
                break
    elif repair_fields in (("amount",), ("volume",), ("volume", "amount")):
        stored_by_date = {str(row["trade_date"])[:10]: row for row in inputs["rows"]}
        database_repair_required = any(
            any(stored_by_date.get(row.date.strftime("%Y-%m-%d"), {}).get(field) is None for field in repair_fields)
            for row in candidate_release.itertuples(index=False)
        )
    else:
        stored_by_date = {str(row["trade_date"])[:10]: row for row in inputs["rows"]}
        fields_to_fill = ("open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "turnover", "adj_factor")
        for row in candidate_release.itertuples(index=False):
            date_value = row.date.strftime("%Y-%m-%d")
            stored = stored_by_date.get(date_value)
            if stored is None:
                database_repair_required = True
                break
            if replace_conflicts and any(
                pd.notna(getattr(row, field))
                and (pd.isna(stored.get(field)) or not math.isclose(float(stored[field]), float(getattr(row, field)), rel_tol=1e-12, abs_tol=1e-9))
                for field in ("open", "high", "low", "close", "volume", "amount")
            ):
                database_repair_required = True
                break
            if any(pd.isna(stored.get(field)) and pd.notna(getattr(row, field)) for field in fields_to_fill):
                database_repair_required = True
                break
    qlib_repair_required = any(
        record.get("before_sha256") != record.get("after_sha256")
        for record in qlib_result.get("files", {}).values()
    )
    if not database_repair_required and not qlib_repair_required:
        blockers.append(_historical_release_blocker("DA04-NO-OP-001", "候选与正式行情库及 Qlib 完全一致，没有需要修复的数据；保留双源证据，不发布空修复"))
    if qlib_result["files"]:
        qlib_backup_dir.mkdir(parents=True, exist_ok=True)
        for field, record in qlib_result["files"].items():
            backup_path = qlib_backup_dir / record.get("backup_name", f"{field}.day.bin")
            if record.get("before_sha256") is None:
                continue
            shutil.copy2(record["source"], backup_path)
            if _file_sha256(backup_path) != record["before_sha256"]:
                blockers.append(_historical_release_blocker("DA04-QLIB-BACKUP-001", f"Qlib {field} 备份哈希校验失败"))
    status, blockers = _historical_release_gate(
        [blockers],
        assessment.get("status") if isinstance(assessment, dict) else "unknown",
        qlib_result.get("status"),
    )
    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "status": status,
        "created_at": now_iso(),
        "batch_id": batch_id,
        "denominator_batch_id": denominator_batch_id,
        "code": code,
        "year": year,
        "date_range": {"start": item["start"], "end": item["end"]},
        "pipeline_contract_version": shadow["pipeline_contract_version"],
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": _file_sha256(candidate_path),
        "source_candidate_path": str((batch_dir / "raw" / code / f"{year}_qfq_candidate.parquet").resolve()),
        "source_candidate_sha256": _file_sha256(batch_dir / "raw" / code / f"{year}_qfq_candidate.parquet"),
        "comparison_sha256": _file_sha256(comparison_path),
        "expected_keys_sha256": year_summary["expected_keys_sha256"],
        "input_fingerprint": full_reanchor["full_stock_fingerprint"] if full_reanchor else inputs["fingerprint"],
        "preimage_fingerprint": full_reanchor["full_stock_fingerprint"] if full_reanchor else inputs["fingerprint"],
        "shard_input_fingerprint": inputs["fingerprint"],
        "anchor_compatibility": anchor_compatibility,
        "full_stock_reanchor": full_reanchor,
        "rollback_range": rollback_range,
        "price_basis": {
            "anchor_date": full_reanchor["target_anchor_date"] if full_reanchor else source_anchor_date,
            "anchor_factor": full_reanchor["target_anchor_factor"] if full_reanchor else source_anchor_factor,
        },
        "preimage_path": str(before_path.resolve()),
        "preimage_sha256": _file_sha256(before_path),
        "preimage_rows": len(before_rows),
        "qlib_dir": str(Path(qlib_dir).resolve()),
        "qlib_generation_id": qlib_generation_id,
        "qlib": qlib_result,
        "assessment": assessment,
        "database_repair_required": database_repair_required,
        "qlib_repair_required": qlib_repair_required,
        "release_change_contract": "g2_effective_change_v1",
        "source_verified_conflict_replacement": replace_conflicts,
        "repair_fields": list(repair_fields),
        "source_quantity_evidence": source_quantity_evidence,
        "source_status_evidence": source_status_evidence,
        "trade_status_adjudication_manifest_sha256": _file_sha256(trade_status_adjudication_path) if trade_status_adjudication_path.is_file() else None,
        "source_suspension_evidence": source_suspension_evidence,
        "price_conflicts_replacement_eligible": assessment.get("price_conflicts_replacement_eligible", 0),
        "blockers": blockers,
        "blocker_rule_ids": _historical_release_blocker_ids(blockers),
        "official_database_writes": False,
        "qlib_writes": False,
        "publishable": status == "ready",
    }
    save_json_atomic(manifest, release_dir / "manifest.json")
    return manifest


def get_historical_release(release_id):
    manifest = load_json(_release_manifest_path(release_id))
    return manifest or {"status": "not_found", "release_id": release_id}


def _historical_release_recovery_snapshot(manifest):
    from core.adapters.qlib.generation import has_qlib_generation_pointer, resolve_qlib_data_dir
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    if manifest.get("full_stock_reanchor"):
        database = HistoricalRepairRepo.get_full_stock_inputs(manifest["code"])
    else:
        database = HistoricalRepairRepo.get_shard_inputs(
            manifest["code"], manifest["date_range"]["start"], manifest["date_range"]["end"],
        )
    qlib_state = {"mode": "generation" if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else "files"}
    if qlib_state["mode"] == "generation":
        qlib_state["current_generation_id"] = resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
    else:
        qlib_state["files"] = {
            field: {
                "current_sha256": _file_sha256(Path(record["source"])) if Path(record["source"]).is_file() else None,
                "before_sha256": record.get("before_sha256"),
                "after_sha256": record.get("after_sha256"),
            }
            for field, record in (manifest.get("qlib", {}).get("files") or {}).items()
        }
    qlib_state_fingerprint = hashlib.sha256(
        json.dumps(qlib_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    preimage_fingerprint = manifest.get("preimage_fingerprint") or (
        (manifest.get("full_stock_reanchor") or {}).get("full_stock_fingerprint")
        or manifest.get("input_fingerprint")
    )
    return {
        "release_id": manifest.get("release_id"),
        "status": manifest.get("status"),
        "database_fingerprint": database["fingerprint"],
        "preimage_fingerprint": preimage_fingerprint,
        "db_after_fingerprint": manifest.get("db_after_fingerprint"),
        "qlib_state": qlib_state,
        "qlib_state_fingerprint": qlib_state_fingerprint,
        "manual_restore_required": bool(
            manifest.get("status") == "recovery_required"
            and manifest.get("db_after_fingerprint") is None
            and database["fingerprint"] != preimage_fingerprint
        ),
    }


def inspect_historical_release_recovery(release_id):
    manifest = load_json(_release_manifest_path(release_id))
    if not manifest or manifest.get("release_id") != release_id:
        raise ValueError("历史发布记录不存在")
    if manifest.get("status") not in ("staging", "publishing", "database_published", "rolling_back", "rollback_required", "recovery_required"):
        raise ValueError("当前发布状态无需恢复核验")
    return _historical_release_recovery_snapshot(manifest)


def authorize_historical_release_recovery_rollback(
    release_id, expected_database_fingerprint, expected_qlib_state_fingerprint, actor, reason, evidence_ref,
):
    if not isinstance(expected_database_fingerprint, str) or len(expected_database_fingerprint) != 64:
        raise ValueError("正式库核验指纹格式无效")
    if not isinstance(expected_qlib_state_fingerprint, str) or len(expected_qlib_state_fingerprint) != 64:
        raise ValueError("Qlib核验指纹格式无效")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("管理员身份不能为空")
    if not isinstance(reason, str) or len(reason.strip()) < 10:
        raise ValueError("恢复原因至少需要10个字符")
    if not isinstance(evidence_ref, str) or len(evidence_ref.strip()) < 3:
        raise ValueError("核验依据引用不能为空")
    with _release_publish_lock:
        with _historical_release_advisory_lock(release_id):
            path = _release_manifest_path(release_id)
            manifest = load_json(path)
            if not manifest or manifest.get("release_id") != release_id:
                raise ValueError("历史发布记录不存在")
            if manifest.get("status") != "recovery_required" or manifest.get("db_after_fingerprint") is not None:
                raise ValueError("只有后指纹缺失的recovery_required记录可走人工核验恢复")
            snapshot = _historical_release_recovery_snapshot(manifest)
            if snapshot["database_fingerprint"] != expected_database_fingerprint:
                raise RuntimeError("正式库已变化，请重新读取恢复核验快照")
            if snapshot["qlib_state_fingerprint"] != expected_qlib_state_fingerprint:
                raise RuntimeError("Qlib状态已变化，请重新读取恢复核验快照")
            if snapshot["database_fingerprint"] == snapshot["preimage_fingerprint"]:
                raise ValueError("正式库已等于pre-image；请使用普通回滚入口，不需要人工覆盖授权")
            decisions = list(manifest.get("manual_recovery_decisions") or [])
            decisions.append({
                "action": "restore_preimage",
                "actor": actor.strip(),
                "reason": reason.strip(),
                "evidence_ref": evidence_ref.strip(),
                "database_fingerprint": snapshot["database_fingerprint"],
                "qlib_state_fingerprint": snapshot["qlib_state_fingerprint"],
                "qlib_state": snapshot["qlib_state"],
                "recorded_at": now_iso(),
            })
            manifest["manual_recovery_decisions"] = decisions
            manifest["db_after_fingerprint"] = snapshot["database_fingerprint"]
            manifest["manual_recovery_authorized_at"] = decisions[-1]["recorded_at"]
            manifest["updated_at"] = decisions[-1]["recorded_at"]
            save_json_atomic(manifest, path)
            return _rollback_historical_release(release_id)


def publish_historical_release(release_id):
    with _release_publish_lock:
        with _historical_release_advisory_lock(release_id):
            return _publish_historical_release(release_id)


def _publish_historical_release(release_id):
    from core.adapters.qlib.historical_patch import promote_stock_patch, restore_stock_patch
    from core.adapters.qlib.generation import (
        clear_historical_release_active,
        discard_qlib_generation_stage,
        has_qlib_generation_pointer,
        mark_historical_release_active,
        publish_qlib_generation,
        qlib_generation_mutation_lock,
        resolve_qlib_data_dir,
        rollback_qlib_generation,
        stage_qlib_generation,
    )
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    path = _release_manifest_path(release_id)
    manifest = load_json(path)
    if not manifest or manifest.get("release_id") != release_id:
        raise ValueError("历史发布预检不存在")
    if manifest.get("status") != "ready":
        message = f"历史发布状态为{manifest.get('status') or 'unknown'}，仅ready状态允许提交"
        manifest["publishable"] = False
        _append_historical_release_blocker(manifest, "DA04-RELEASE-STATE-001", message)
        manifest["updated_at"] = now_iso()
        save_json_atomic(manifest, path)
        raise RuntimeError(_historical_release_blocker("DA04-RELEASE-STATE-001", message))
    if manifest.get("release_change_contract") != "g2_effective_change_v1":
        message = "历史发布预检版本缺少有效变更门禁，请重新运行 G2 预检"
        _mark_historical_release_blocked(manifest, path, "DA04-RELEASE-CONTRACT-001", message)
        raise RuntimeError(_historical_release_blocker("DA04-RELEASE-CONTRACT-001", message))
    gate_status, gate_blockers = _historical_release_gate(
        [manifest.get("blockers") or []],
        (manifest.get("assessment") or {}).get("status"),
        (manifest.get("qlib") or {}).get("status"),
        [manifest.get("blocker_rule_ids") or []],
    )
    if manifest.get("status") != "ready" or not manifest.get("publishable") or gate_status != "ready":
        if gate_blockers:
            manifest["blockers"] = gate_blockers
            manifest["blocker_rule_ids"] = _historical_release_blocker_ids(gate_blockers)
            manifest["status"] = "blocked"
            manifest["publishable"] = False
            manifest["updated_at"] = now_iso()
            save_json_atomic(manifest, path)
        else:
            message = "历史发布状态与publishable准入标记不一致，需重新执行G2预检"
            _mark_historical_release_blocked(manifest, path, "DA04-PUBLISHABLE-STATE-001", message)
        raise RuntimeError("历史发布未通过预检")
    if not manifest.get("database_repair_required") and not manifest.get("qlib_repair_required"):
        message = "候选与正式行情库及 Qlib 无有效差异，拒绝发布空修复"
        _mark_historical_release_blocked(manifest, path, "DA04-NO-OP-001", message)
        raise RuntimeError(_historical_release_blocker("DA04-NO-OP-001", message))
    release_dir = path.parent
    candidate_path = Path(manifest["candidate_path"])
    preimage_path = Path(manifest["preimage_path"])
    if _file_sha256(candidate_path) != manifest["candidate_sha256"]:
        _mark_historical_release_blocked(manifest, path, "DA04-CANDIDATE-HASH-001", "历史发布候选哈希变化，需重新预检")
        raise RuntimeError("历史发布候选哈希不符")
    if _file_sha256(preimage_path) != manifest["preimage_sha256"]:
        _mark_historical_release_blocked(manifest, path, "DA04-PREIMAGE-HASH-001", "历史发布 pre-image 哈希变化，需重新预检")
        raise RuntimeError("历史发布 pre-image 哈希不符")
    shadow_path = _manifest_path(manifest["batch_id"])
    shadow = load_json(shadow_path)
    shadow_item = next((row for row in (shadow or {}).get("work_items", [])
                        if row.get("code") == manifest["code"] and int(row.get("year", 0)) == int(manifest["year"])), None)
    source_candidate_path = Path(manifest.get("source_candidate_path") or manifest["candidate_path"])
    comparison_path = shadow_path.parent / "raw" / manifest["code"] / f"{manifest['year']}_comparison.parquet"
    if (
        not shadow_item or not _shadow_files_valid(shadow_path.parent, shadow_item)
        or not source_candidate_path.is_file()
        or _file_sha256(source_candidate_path) != manifest.get("source_candidate_sha256")
        or not comparison_path.is_file()
        or _file_sha256(comparison_path) != manifest.get("comparison_sha256")
    ):
        _mark_historical_release_blocked(manifest, path, "DA04-SOURCE-EVIDENCE-HASH-001", "G1 双源证据或原始候选哈希变化，需重新预检")
        raise RuntimeError("G1 双源证据或原始候选已变化，拒绝发布")
    from core.data.repos.historical_repair import HistoricalRepairRepo
    full_reanchor = manifest.get("full_stock_reanchor")
    if full_reanchor:
        current = HistoricalRepairRepo.get_full_stock_inputs(manifest["code"])
    else:
        current = HistoricalRepairRepo.get_shard_inputs(manifest["code"], manifest["date_range"]["start"], manifest["date_range"]["end"])
    if current["fingerprint"] != manifest["input_fingerprint"]:
        blocker = "正式库整股指纹与预检快照不同，需基于最新基线重新采集并预检" if full_reanchor else "正式库分片指纹与预检快照不同，需基于最新基线重新采集并预检"
        _mark_historical_release_blocked(manifest, path, "DA04-INPUT-BASELINE-001", blocker)
        raise RuntimeError("正式库基线在预检后变化，拒绝发布")
    qlib_managed = has_qlib_generation_pointer(config.QLIB_DATA_ROOT)
    current_qlib_dir = resolve_qlib_data_dir(config.QLIB_DATA_ROOT)
    current_qlib_generation_id = current_qlib_dir.name if qlib_managed else None
    if current_qlib_generation_id != manifest.get("qlib_generation_id"):
        _mark_historical_release_blocked(manifest, path, "DA04-QLIB-GENERATION-001", "Qlib当前代与历史预检快照不同，需重新预检")
        raise RuntimeError("Qlib当前代与历史预检快照不同，拒绝发布")
    if qlib_managed and Path(manifest["qlib_dir"]).resolve() != current_qlib_dir.resolve():
        message = "历史预检记录的Qlib路径与当前代目录不一致，拒绝发布"
        _mark_historical_release_blocked(manifest, path, "DA04-QLIB-PATH-001", message)
        raise RuntimeError(_historical_release_blocker("DA04-QLIB-PATH-001", message))
    changed_qlib_files = []
    for field, record in manifest.get("qlib", {}).get("files", {}).items():
        source = Path(record["source"])
        before_sha256 = record.get("before_sha256")
        if before_sha256 is None:
            if source.exists():
                changed_qlib_files.append(field)
        elif not source.is_file() or _file_sha256(source) != before_sha256:
            changed_qlib_files.append(field)
    if changed_qlib_files:
        blocker = f"Qlib {', '.join(changed_qlib_files)} 自预检后已变化，需基于当前 Qlib 重新预检"
        _mark_historical_release_blocked(manifest, path, "DA04-QLIB-FILE-HASH-001", blocker)
        raise RuntimeError(blocker)
    qlib_generation_stage = None
    db_result = None
    before_rows = []
    provenance = {}
    qlib_promoted = False
    qlib_generation_published_id = None
    qlib_unpublished_generation_candidate_id = None
    qlib_commit_lock = None
    qlib_commit_lock_acquired = False
    try:
        if qlib_managed and manifest.get("qlib_repair_required"):
            stage_id = uuid.uuid4().hex
            stage_dir = Path(config.QLIB_DATA_ROOT).resolve() / "generations" / f".staging-{stage_id}"
            manifest["qlib_generation_parent_id"] = current_qlib_generation_id
            manifest["qlib_generation_stage_id"] = stage_id
            manifest["qlib_generation_stage_dir"] = str(stage_dir.resolve())
            manifest["qlib_generation_stage_state"] = "copying"
            manifest["status"] = "staging"
            manifest["stage_started_at"] = now_iso()
            manifest["updated_at"] = manifest["stage_started_at"]
            save_json_atomic(manifest, path)
            qlib_generation_stage = stage_qlib_generation(
                data_root=config.QLIB_DATA_ROOT,
                generation_id=stage_id,
                expected_parent_generation_id=current_qlib_generation_id,
                lifecycle_lease=True,
                owner={"type": "historical_release", "release_id": release_id},
            )
        if qlib_generation_stage:
            if Path(qlib_generation_stage["stage_dir"]).resolve() != Path(manifest["qlib_generation_stage_dir"]).resolve():
                raise RuntimeError("Qlib候选暂存目录与已持久化恢复路径不一致")
            for record in manifest.get("qlib", {}).get("files", {}).values():
                source = Path(record["source"]).resolve()
                relative = source.relative_to(current_qlib_dir.resolve())
                staged_source = Path(qlib_generation_stage["stage_dir"]) / relative
                if record.get("before_sha256") is None:
                    if staged_source.exists():
                        raise RuntimeError(f"Qlib候选文件预期不存在但已存在: {relative}")
                elif not staged_source.is_file() or _file_sha256(staged_source) != record["before_sha256"]:
                    raise RuntimeError(f"Qlib候选基线哈希不符: {relative}")
                record["generation_relative_path"] = relative.as_posix()
                record["source_before_generation"] = str(source)
                record["source"] = str(staged_source)
            manifest["qlib_generation_stage_state"] = "ready"
            manifest["updated_at"] = now_iso()
            save_json_atomic(manifest, path)
        candidate = load_parquet(candidate_path)
        before_rows = load_parquet(preimage_path).to_dict("records")
        provenance = {
            "batch_id": release_id,
            "task_id": f"historical-{release_id}",
            "source": "tushare+baostock",
            "provider_version": json.dumps(load_json(_manifest_path(manifest["batch_id"]))["source_versions_at_collection"], sort_keys=True),
            "adjustment_mode": "qfq",
            "anchor_date": manifest["price_basis"]["anchor_date"],
            "input_payload_sha256": manifest["candidate_sha256"],
            "source_verified_conflict_replacement": bool(manifest.get("source_verified_conflict_replacement")),
        }
        qlib_commit_lock = qlib_generation_mutation_lock(config.QLIB_DATA_ROOT)
        qlib_commit_lock.__enter__()
        qlib_commit_lock_acquired = True
        mark_historical_release_active(release_id, path, data_root=config.QLIB_DATA_ROOT)
        manifest["status"] = "publishing"
        manifest["publish_started_at"] = now_iso()
        save_json_atomic(manifest, path)
        if full_reanchor:
            db_result = HistoricalRepairRepo.publish_full_reanchor(
                manifest["code"], candidate, manifest["input_fingerprint"],
                full_reanchor["old_anchor_factor"], full_reanchor["target_anchor_factor"], provenance,
            )
        else:
            db_result = HistoricalRepairRepo.publish_missing_values(
                manifest["code"], manifest["date_range"]["start"], manifest["date_range"]["end"],
                candidate, manifest["input_fingerprint"], provenance,
                replace_source_verified_prices=bool(manifest.get("source_verified_conflict_replacement")),
                repair_fields=manifest.get("repair_fields") or None,
            )
        manifest["db_after_fingerprint"] = db_result["after_fingerprint"]
        after_anchor = db_result.get("after_anchor") or {}
        manifest["db_after_anchor"] = {
            key: value.isoformat() if hasattr(value, "isoformat") else str(value) if value is not None else None
            for key, value in after_anchor.items()
        }
        manifest["db_input_rows"] = db_result["input_rows"]
        manifest["db_audit_rows"] = db_result["audit_rows"]
        manifest["status"] = "database_published"
        save_json_atomic(manifest, path)
        if qlib_generation_stage:
            promote_stock_patch(manifest["qlib"]["files"])
        else:
            with qlib_generation_mutation_lock(config.QLIB_DATA_ROOT, require_legacy=True):
                promote_stock_patch(manifest["qlib"]["files"])
        if qlib_generation_stage:
            published = publish_qlib_generation(
                qlib_generation_stage,
                metadata={"operation": "historical_quote_repair", "release_id": release_id},
                expected_generation_id=qlib_generation_stage["parent_generation_id"],
            )
            qlib_generation_published_id = published["generation_id"]
            manifest["qlib_generation_published_id"] = published["generation_id"]
            manifest["qlib_dir"] = published["data_dir"]
            final_root = Path(published["data_dir"])
            for record in manifest["qlib"]["files"].values():
                relative = Path(record["source"]).relative_to(Path(qlib_generation_stage["stage_dir"]))
                record["source"] = str(final_root / relative)
            qlib_promoted = True
        else:
            qlib_promoted = bool(manifest["qlib"].get("files"))
        manifest["qlib_promoted_at"] = now_iso()
        manifest["status"] = "published"
        manifest["official_database_writes"] = True
        manifest["qlib_writes"] = True
        manifest["publishable"] = False
        manifest["published_at"] = now_iso()
        save_json_atomic(manifest, path)
        try:
            clear_historical_release_active(release_id, data_root=config.QLIB_DATA_ROOT)
        except Exception as cleanup_error:
            logger.error("历史发布已完成，但清理Qlib阻断标记失败: %s", cleanup_error)
        return manifest
    except Exception as error:
        rollback_errors = []
        try:
            if qlib_generation_stage:
                current_generation_id = (
                    resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
                    if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
                )
                staged_generation_id = qlib_generation_stage["generation_id"]
                parent_generation_id = qlib_generation_stage["parent_generation_id"]
                if current_generation_id == staged_generation_id:
                    manifest["qlib_generation_published_id"] = staged_generation_id
                    manifest["qlib_generation_parent_id"] = parent_generation_id
                    _verify_qlib_generation_release_files(manifest, staged_generation_id, "after")
                    _verify_qlib_generation_release_files(manifest, parent_generation_id, "before")
                    rollback_qlib_generation(
                        parent_generation_id,
                        data_root=config.QLIB_DATA_ROOT,
                        expected_current_generation_id=staged_generation_id,
                    )
                elif current_generation_id == parent_generation_id:
                    _verify_qlib_generation_release_files(manifest, parent_generation_id, "before")
                    candidate_dir = (
                        Path(config.QLIB_DATA_ROOT).resolve()
                        / "generations" / staged_generation_id
                    )
                    if candidate_dir.is_dir():
                        qlib_unpublished_generation_candidate_id = staged_generation_id
                else:
                    raise RuntimeError("Qlib当前代与发布前/候选代均不一致，拒绝自动回滚")
                manifest["qlib_dir"] = str(resolve_qlib_data_dir(config.QLIB_DATA_ROOT))
                for record in manifest["qlib"]["files"].values():
                    if record.get("source_before_generation"):
                        record["source"] = record["source_before_generation"]
            elif manifest.get("qlib_generation_stage_id"):
                current_generation_id = (
                    resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
                    if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
                )
                if current_generation_id != manifest.get("qlib_generation_parent_id"):
                    raise RuntimeError("Qlib当前代在暂存开始前已变化，拒绝将发布标记为已回滚")
                _verify_qlib_generation_release_files(
                    manifest, manifest["qlib_generation_parent_id"], "before",
                )
            elif qlib_generation_published_id:
                rollback_qlib_generation(
                    manifest["qlib_generation_parent_id"],
                    data_root=config.QLIB_DATA_ROOT,
                    expected_current_generation_id=qlib_generation_published_id,
                )
                manifest["qlib_dir"] = str(resolve_qlib_data_dir(config.QLIB_DATA_ROOT))
                for record in manifest["qlib"]["files"].values():
                    if record.get("source_before_generation"):
                        record["source"] = record["source_before_generation"]
            elif not qlib_generation_stage and (qlib_promoted or manifest.get("status") == "database_published"):
                with qlib_generation_mutation_lock(config.QLIB_DATA_ROOT, require_legacy=True):
                    restore_stock_patch(
                        manifest["qlib"]["files"], release_dir / "preimage" / "qlib",
                        expected_after=qlib_promoted, allow_partial=not qlib_promoted,
                    )
        except Exception as rollback_error:
            rollback_errors.append(f"Qlib 回滚失败：{rollback_error}")
        try:
            if db_result is not None:
                restored = HistoricalRepairRepo.restore_shard(
                    manifest["code"], manifest["rollback_range"]["start"], manifest["rollback_range"]["end"],
                    before_rows, db_result["after_fingerprint"], provenance,
                )
                manifest["db_rollback_fingerprint"] = restored["restored_fingerprint"]
        except Exception as rollback_error:
            rollback_errors.append(f"数据库回滚失败：{rollback_error}")
        if qlib_generation_stage and Path(qlib_generation_stage["stage_dir"]).exists():
            try:
                discard_qlib_generation_stage(qlib_generation_stage)
                manifest["qlib_generation_stage_cleanup"] = "complete"
            except Exception as cleanup_error:
                manifest["qlib_generation_stage_cleanup"] = "required"
                manifest["qlib_generation_stage_cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                rollback_errors.append(f"Qlib候选代临时目录清理失败：{cleanup_error}")
        if qlib_unpublished_generation_candidate_id:
            try:
                _discard_unpublished_qlib_generation_candidate(
                    manifest, release_id, qlib_unpublished_generation_candidate_id,
                )
                manifest["qlib_generation_orphan_cleanup"] = "complete"
            except Exception as cleanup_error:
                manifest["qlib_generation_orphan_cleanup"] = "required"
                manifest["qlib_generation_orphan_cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                rollback_errors.append(f"Qlib未激活候选代清理失败：{cleanup_error}")
        manifest["status"] = "rollback_required" if rollback_errors else "rolled_back"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["rollback_errors"] = rollback_errors
        manifest["updated_at"] = now_iso()
        save_json_atomic(manifest, path)
        if manifest["status"] == "rolled_back":
            try:
                clear_historical_release_active(release_id, data_root=config.QLIB_DATA_ROOT)
            except Exception as cleanup_error:
                logger.error("历史发布已回滚，但清理Qlib阻断标记失败: %s", cleanup_error)
        raise RuntimeError(f"历史发布失败；状态 {manifest['status']}；{manifest['error']}; {rollback_errors}") from error
    finally:
        if qlib_commit_lock_acquired:
            qlib_commit_lock.__exit__(None, None, None)


def rollback_historical_release(release_id):
    with _release_publish_lock:
        with _historical_release_advisory_lock(release_id):
            return _rollback_historical_release(release_id)


@qlib_mutation_locked
def _rollback_historical_release(release_id):
    from core.adapters.qlib.historical_patch import restore_stock_patch
    from core.adapters.qlib.generation import (
        clear_historical_release_active,
        discard_qlib_generation_stage,
        has_qlib_generation_pointer,
        qlib_generation_mutation_lock,
        mark_historical_release_active,
        resolve_qlib_data_dir,
        rollback_qlib_generation,
    )
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    path = _release_manifest_path(release_id)
    manifest = load_json(path)
    if not manifest or manifest.get("release_id") != release_id or manifest.get("status") not in (
        "staging", "publishing", "database_published", "published", "rolling_back", "rollback_required", "recovery_required",
    ):
        raise ValueError("只有发布中、已发布或待恢复的批次可以回滚")
    rollback_state = manifest.get("rollback_state") or {}
    if manifest.get("full_stock_reanchor"):
        current = HistoricalRepairRepo.get_full_stock_inputs(manifest["code"])
    else:
        current = HistoricalRepairRepo.get_shard_inputs(manifest["code"], manifest["date_range"]["start"], manifest["date_range"]["end"])
    preimage_fingerprint = manifest.get("preimage_fingerprint") or (
        (manifest.get("full_stock_reanchor") or {}).get("full_stock_fingerprint")
        or manifest.get("input_fingerprint")
    )
    if current["fingerprint"] not in (manifest.get("db_after_fingerprint"), preimage_fingerprint):
        if manifest.get("db_after_fingerprint") is None and manifest.get("status") in ("staging", "publishing", "recovery_required"):
            manifest["status"] = "recovery_required"
            manifest["rollback_error"] = "正式库与pre-image不一致，但发布清单缺少已提交后指纹；拒绝推测并覆盖"
            manifest["updated_at"] = now_iso()
            save_json_atomic(manifest, path)
            raise RuntimeError(manifest["rollback_error"])
        raise RuntimeError("正式库分片在发布后已变化，拒绝覆盖式回滚")
    qlib_generation_published_id = manifest.get("qlib_generation_published_id")
    qlib_generation_parent_id = manifest.get("qlib_generation_parent_id")
    qlib_generation_stage_id = manifest.get("qlib_generation_stage_id")
    if qlib_generation_stage_id and (
        not isinstance(qlib_generation_stage_id, str)
        or len(qlib_generation_stage_id) != 32
        or not qlib_generation_stage_id.isalnum()
    ):
        raise RuntimeError("Qlib候选代ID无效，拒绝恢复")
    if not qlib_generation_published_id and qlib_generation_stage_id:
        current_qlib_id = (
            resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
            if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
        )
        if current_qlib_id not in (qlib_generation_stage_id, qlib_generation_parent_id):
            raise RuntimeError("Qlib当前代既非发布候选代也非父代，拒绝恢复")
        if current_qlib_id == qlib_generation_stage_id:
            qlib_generation_published_id = qlib_generation_stage_id
            manifest["qlib_generation_published_id"] = qlib_generation_stage_id
    if qlib_generation_stage_id:
        current_qlib_id = (
            resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
            if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
        )
        if current_qlib_id in (qlib_generation_stage_id, qlib_generation_published_id):
            _verify_qlib_generation_release_files(manifest, current_qlib_id, "after")
            _verify_qlib_generation_release_files(manifest, qlib_generation_parent_id, "before")
        elif current_qlib_id == qlib_generation_parent_id:
            _verify_qlib_generation_release_files(manifest, qlib_generation_parent_id, "before")
        else:
            raise RuntimeError("Qlib当前代已变化，既非发布候选代也非父代，拒绝恢复")
        qlib_already_restored = current_qlib_id == qlib_generation_parent_id
    elif qlib_generation_published_id:
        current_qlib_id = (
            resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
            if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
        )
        if current_qlib_id not in (qlib_generation_published_id, qlib_generation_parent_id):
            raise RuntimeError("Qlib当前代已变化，拒绝回滚历史发布")
        qlib_already_restored = current_qlib_id == qlib_generation_parent_id
    else:
        qlib_file_states = []
        for field, record in manifest["qlib"]["files"].items():
            source = Path(record["source"])
            current_hash = _file_sha256(source) if source.is_file() else None
            before_hash = record.get("before_sha256")
            after_hash = record.get("after_sha256")
            if current_hash not in (before_hash, after_hash):
                raise RuntimeError(f"Qlib {field} 在发布后已变化，拒绝覆盖式回滚")
            qlib_file_states.append(current_hash == before_hash)
        qlib_already_restored = all(qlib_file_states)
    candidate_path = Path(manifest["candidate_path"])
    if _file_sha256(candidate_path) != manifest["candidate_sha256"]:
        raise RuntimeError("回滚 provenance 候选文件哈希不符")
    batch = load_json(_manifest_path(manifest["batch_id"]))
    provenance = {
        "batch_id": release_id, "task_id": f"historical-{release_id}", "source": "tushare+baostock",
        "provider_version": json.dumps(batch["source_versions_at_collection"], sort_keys=True),
        "anchor_date": manifest["date_range"]["end"], "input_payload_sha256": manifest["candidate_sha256"],
    }
    before_rows = load_parquet(manifest["preimage_path"]).to_dict("records")
    database_already_restored = current["fingerprint"] == preimage_fingerprint
    rollback_state = {
        "qlib": "complete" if qlib_already_restored else rollback_state.get("qlib", "pending"),
        "database": "complete" if database_already_restored else rollback_state.get("database", "pending"),
    }
    manifest["status"] = "rolling_back"
    mark_historical_release_active(release_id, path, data_root=config.QLIB_DATA_ROOT)
    manifest["rollback_state"] = rollback_state
    manifest["rollback_started_at"] = now_iso()
    save_json_atomic(manifest, path)
    try:
        if rollback_state["qlib"] != "complete":
            if qlib_generation_published_id:
                current_qlib_id = (
                    resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
                    if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
                )
                if current_qlib_id == qlib_generation_published_id:
                    rollback_qlib_generation(
                        qlib_generation_parent_id,
                        data_root=config.QLIB_DATA_ROOT,
                        expected_current_generation_id=qlib_generation_published_id,
                    )
                elif current_qlib_id != qlib_generation_parent_id:
                    raise RuntimeError("Qlib当前代在回滚期间变化，拒绝覆盖")
                manifest["qlib_dir"] = str(resolve_qlib_data_dir(config.QLIB_DATA_ROOT))
            else:
                with qlib_generation_mutation_lock(config.QLIB_DATA_ROOT, require_legacy=True):
                    restore_stock_patch(
                        manifest["qlib"]["files"], path.parent / "preimage" / "qlib",
                        expected_after=True, allow_partial=True,
                    )
            rollback_state["qlib"] = "complete"
            manifest["rollback_state"] = rollback_state
            save_json_atomic(manifest, path)
        if rollback_state["database"] != "complete":
            current = HistoricalRepairRepo.get_full_stock_inputs(manifest["code"]) if manifest.get("full_stock_reanchor") else HistoricalRepairRepo.get_shard_inputs(
                manifest["code"], manifest["date_range"]["start"], manifest["date_range"]["end"],
            )
            if current["fingerprint"] == preimage_fingerprint:
                restored = {"restored_fingerprint": preimage_fingerprint, "rows": len(before_rows), "audit_rows": 0}
            elif current["fingerprint"] == manifest.get("db_after_fingerprint"):
                restored = HistoricalRepairRepo.restore_shard(
                    manifest["code"], manifest["rollback_range"]["start"], manifest["rollback_range"]["end"],
                    before_rows, manifest["db_after_fingerprint"], provenance,
                )
            else:
                raise RuntimeError("正式库在回滚期间发生变化，拒绝覆盖")
            rollback_state["database"] = "complete"
            manifest["rollback_state"] = rollback_state
            manifest["db_rollback_fingerprint"] = restored["restored_fingerprint"]
            manifest["rollback_rows"] = restored["rows"]
            manifest["rollback_audit_rows"] = restored["audit_rows"]
            save_json_atomic(manifest, path)
        if qlib_generation_stage_id:
            stage_dir = (Path(config.QLIB_DATA_ROOT) / "generations" / f".staging-{qlib_generation_stage_id}").resolve()
            generations_dir = (Path(config.QLIB_DATA_ROOT) / "generations").resolve()
            if stage_dir.is_relative_to(generations_dir):
                discard_qlib_generation_stage({
                    "data_root": config.QLIB_DATA_ROOT,
                    "generation_id": qlib_generation_stage_id,
                    "stage_dir": stage_dir,
                    "parent_generation_id": qlib_generation_parent_id,
                    "lifecycle_lease": True,
                    "owner": {"type": "historical_release", "release_id": release_id},
                })
            candidate_dir = (generations_dir / qlib_generation_stage_id).resolve()
            if candidate_dir.is_relative_to(generations_dir) and candidate_dir.is_dir() and not manifest.get("qlib_generation_published_id"):
                _discard_unpublished_qlib_generation_candidate(
                    manifest, release_id, qlib_generation_stage_id,
                )
                manifest["qlib_generation_orphan_cleanup"] = "complete"
            manifest["qlib_generation_stage_cleanup"] = "complete"
        final_database = HistoricalRepairRepo.get_full_stock_inputs(manifest["code"]) if manifest.get("full_stock_reanchor") else HistoricalRepairRepo.get_shard_inputs(
            manifest["code"], manifest["date_range"]["start"], manifest["date_range"]["end"],
        )
        if final_database["fingerprint"] != preimage_fingerprint:
            raise RuntimeError("正式库在回滚终态核验时已偏离pre-image，拒绝标记回滚完成")
        if has_qlib_generation_pointer(config.QLIB_DATA_ROOT):
            expected_qlib_generation_id = qlib_generation_parent_id or manifest.get("qlib_generation_id")
            current_qlib_generation_id = resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
            if current_qlib_generation_id != expected_qlib_generation_id:
                raise RuntimeError("Qlib在回滚终态核验时已偏离父代，拒绝标记回滚完成")
            if manifest.get("qlib", {}).get("files"):
                _verify_qlib_generation_release_files(manifest, expected_qlib_generation_id, "before")
        else:
            for field, record in manifest.get("qlib", {}).get("files", {}).items():
                source = Path(record["source"])
                current_hash = _file_sha256(source) if source.is_file() else None
                if current_hash != record.get("before_sha256"):
                    raise RuntimeError(f"Qlib {field}在回滚终态核验时已变化，拒绝标记回滚完成")
        manifest["status"] = "rolled_back"
        manifest["official_database_writes"] = False
        manifest["qlib_writes"] = False
        manifest["rolled_back_at"] = now_iso()
        manifest.pop("rollback_error", None)
        save_json_atomic(manifest, path)
        try:
            clear_historical_release_active(release_id, data_root=config.QLIB_DATA_ROOT)
        except Exception as cleanup_error:
            logger.error("历史发布回滚已完成，但清理Qlib阻断标记失败: %s", cleanup_error)
        return manifest
    except Exception as error:
        manifest["status"] = "rollback_required"
        manifest["rollback_error"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = now_iso()
        save_json_atomic(manifest, path)
        raise RuntimeError(f"历史发布回滚未完成，可重试恢复：{manifest['rollback_error']}") from error
