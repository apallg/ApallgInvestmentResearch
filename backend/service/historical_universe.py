import datetime
import hashlib
import threading
import uuid
from pathlib import Path

import pandas as pd

from core.data.repos.historical_quality import HistoricalQuoteQualityRepo
from core.data.sources.historical_universe import (
    fetch_tushare_open_calendar,
    fetch_tushare_stock_registry,
    fetch_tushare_suspensions,
    get_historical_universe_source_version,
)
from core.domain.data_quality.historical_denominator import build_historical_denominator
from core.shared.infrastructure.async_task import TaskCancelled, task_manager
from core.shared.persistence import load_json, now_iso, save_json_atomic, save_parquet


_active_lock = threading.Lock()
_active_job_id = None


def _root():
    return Path(__file__).resolve().parents[2] / "data/historical_repair/denominator"


def _manifest_path(batch_id):
    if not batch_id.isalnum() or len(batch_id) != 12:
        raise ValueError("G0 批次 ID 格式无效")
    return _root() / batch_id / "manifest.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_range(start, end):
    start_date = datetime.date.fromisoformat(str(start)[:10])
    end_date = datetime.date.fromisoformat(str(end)[:10])
    if end_date < start_date:
        raise ValueError("结束日期不能早于开始日期")
    if end_date > datetime.date.today():
        raise ValueError("G0 历史分母批次不能包含未来日期")
    if end_date.year - start_date.year > 7:
        raise ValueError("单批最多覆盖 8 个自然年，请拆分批次")
    return start_date, end_date


def _execute_batch(batch_id, job_id, start, end, progress_cb=None):
    global _active_job_id
    manifest_path = _manifest_path(batch_id)
    batch_dir = manifest_path.parent
    manifest = load_json(manifest_path)
    if not manifest or manifest.get("batch_id") != batch_id:
        raise ValueError("G0 历史分母批次不存在或清单损坏")
    try:
        if task_manager.is_cancel_requested(job_id):
            raise TaskCancelled()
        registry_capture = fetch_tushare_stock_registry()
        calendar = fetch_tushare_open_calendar(start, end)
        suspensions = fetch_tushare_suspensions(start, end)
        source_dir = batch_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        trading_override_path = _root().parent / "adjudication" / "verified_trading_day_overrides.json"
        trading_overrides = load_json(trading_override_path)
        if (
            not trading_overrides
            or trading_overrides.get("schema_version") != 1
            or trading_overrides.get("type") != "verified_trading_day_overrides"
            or not isinstance(trading_overrides.get("records"), list)
        ):
            raise RuntimeError("已核验交易日覆盖裁决清单缺失或版本不支持")
        trading_override_snapshot_path = source_dir / "verified_trading_day_overrides.json"
        save_json_atomic(trading_overrides, trading_override_snapshot_path)
        trading_override_keys = set()
        trading_override_checks = []
        historical_repair_root = _root().parent.resolve()
        calendar_dates = set(pd.to_datetime(calendar["cal_date"], errors="coerce").dt.date.dropna())
        for record in trading_overrides["records"]:
            if record.get("status") != "verified":
                continue
            if record.get("classification") != "traded_on_tushare_suspension_date":
                raise RuntimeError(f"交易日覆盖裁决类型不支持：{record.get('adjudication_id')}")
            code = str(record.get("code") or "").strip().zfill(6)
            ts_code = str(record.get("ts_code") or "").strip().upper()
            trade_date = datetime.date.fromisoformat(str(record.get("trade_date"))[:10])
            scope_start = datetime.date.fromisoformat(str(record.get("verified_scope_start_date"))[:10])
            scope_end = datetime.date.fromisoformat(str(record.get("verified_scope_end_date_exclusive"))[:10])
            if not code.isdigit() or ts_code != f"{code}.SH" or scope_start != trade_date or scope_end != trade_date + datetime.timedelta(days=1):
                raise RuntimeError(f"交易日覆盖裁决身份或范围无效：{record.get('adjudication_id')}")
            evidence_by_kind = {evidence.get("kind"): evidence for evidence in record.get("evidence", [])}
            required_kinds = {"official_exchange_trade_notice", "independent_tushare_daily_quote", "independent_baostock_daily_quote"}
            if len(evidence_by_kind) != 3 or set(evidence_by_kind) != required_kinds:
                raise RuntimeError(f"交易日覆盖裁决缺少交易所公告或双源行情证据：{record.get('adjudication_id')}")
            evidence_paths = {}
            evidence_checks = []
            for kind, evidence in evidence_by_kind.items():
                evidence_file = (trading_override_path.parent / str(evidence.get("path") or "")).resolve()
                if (
                    historical_repair_root not in evidence_file.parents
                    or not evidence_file.is_file()
                    or _sha256(evidence_file) != evidence.get("sha256")
                ):
                    raise RuntimeError(f"交易日覆盖证据缺失或哈希不符：{record.get('adjudication_id')}")
                if kind == "official_exchange_trade_notice" and (
                    not str(evidence.get("url") or "").startswith("https://")
                    or trade_date.isoformat() not in str(evidence.get("fact") or "")
                ):
                    raise RuntimeError(f"交易日覆盖交易所公告未明确交易日期：{record.get('adjudication_id')}")
                evidence_paths[kind] = evidence_file
                evidence_checks.append({"kind": kind, "path": str(evidence_file), "sha256": evidence["sha256"]})
            tushare_capture = pd.read_parquet(evidence_paths["independent_tushare_daily_quote"])
            tushare_capture["date_key"] = pd.to_datetime(tushare_capture["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            tushare_rows = tushare_capture.loc[tushare_capture["date_key"].eq(trade_date.isoformat())]
            baostock_capture = pd.read_parquet(evidence_paths["independent_baostock_daily_quote"])
            baostock_capture["date_key"] = pd.to_datetime(baostock_capture["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            baostock_rows = baostock_capture.loc[baostock_capture["date_key"].eq(trade_date.isoformat())]
            if len(tushare_rows) != 1 or len(baostock_rows) != 1:
                raise RuntimeError(f"交易日覆盖双源行情缺少唯一记录：{record.get('adjudication_id')}")
            source_tushare = tushare_rows.iloc[0]
            source_baostock = baostock_rows.iloc[0]
            ohlc_match = all(
                pd.notna(pd.to_numeric(pd.Series([source_tushare[field]]), errors="coerce").iloc[0])
                and pd.notna(pd.to_numeric(pd.Series([source_baostock[field]]), errors="coerce").iloc[0])
                and abs(float(source_tushare[field]) - float(source_baostock[field])) <= 0.011 + abs(float(source_tushare[field])) * 0.0005
                for field in ("open", "high", "low", "close")
            )
            tushare_volume = pd.to_numeric(pd.Series([source_tushare["vol"]]), errors="coerce").iloc[0] * 100
            baostock_volume = pd.to_numeric(pd.Series([source_baostock["volume"]]), errors="coerce").iloc[0]
            tushare_amount = pd.to_numeric(pd.Series([source_tushare["amount"]]), errors="coerce").iloc[0] * 1000
            baostock_amount = pd.to_numeric(pd.Series([source_baostock["amount"]]), errors="coerce").iloc[0]
            quantity_match = bool(
                pd.notna(tushare_volume) and pd.notna(baostock_volume)
                and pd.notna(tushare_amount) and pd.notna(baostock_amount)
                and tushare_volume > 0 and baostock_volume > 0
                and tushare_amount > 0 and baostock_amount > 0
                and abs(tushare_volume - baostock_volume) <= 1
                and abs(tushare_amount - baostock_amount) <= max(100.0, abs(tushare_amount) * 0.001)
                and str(source_baostock["tradestatus"]) == "1"
            )
            if not ohlc_match or not quantity_match:
                raise RuntimeError(f"交易日覆盖双源行情状态或数值不一致：{record.get('adjudication_id')}")
            if start <= trade_date <= end:
                if trade_date not in calendar_dates:
                    raise RuntimeError(f"交易日覆盖日期不在交易日历中：{record.get('adjudication_id')}")
                matching_suspensions = suspensions.loc[
                    suspensions["ts_code"].astype(str).str.upper().eq(ts_code)
                    & pd.to_datetime(suspensions["trade_date"], errors="coerce").dt.date.eq(trade_date)
                ]
                if len(matching_suspensions) != 1:
                    raise RuntimeError(f"交易日覆盖裁决与 TuShare 停牌记录不匹配：{record.get('adjudication_id')}")
                trading_override_keys.add((ts_code, pd.Timestamp(trade_date)))
            trading_override_checks.append({
                "adjudication_id": record["adjudication_id"], "ts_code": ts_code,
                "trade_date": trade_date.isoformat(), "ohlc_match": ohlc_match,
                "quantity_match": quantity_match, "overrode_tushare_suspension": start <= trade_date <= end,
                "evidence": evidence_checks,
            })
        if trading_override_keys:
            suspension_dates = pd.to_datetime(suspensions["trade_date"], errors="coerce").dt.normalize()
            override_mask = [
                (str(ts_code).upper(), trade_date) in trading_override_keys
                for ts_code, trade_date in zip(suspensions["ts_code"], suspension_dates)
            ]
            suspensions = suspensions.loc[~pd.Series(override_mask, index=suspensions.index)].copy()
        adjudication_path = _root().parent / "adjudication" / "verified_suspension_intervals.json"
        adjudication = load_json(adjudication_path)
        if not adjudication or adjudication.get("schema_version") != 1 or not isinstance(adjudication.get("records"), list):
            raise RuntimeError("已核验停牌区间裁决清单缺失或版本不支持")
        adjudication_snapshot_path = source_dir / "verified_suspension_intervals.json"
        save_json_atomic(adjudication, adjudication_snapshot_path)
        delist_adjudication_path = _root().parent / "adjudication" / "verified_delist_date_adjudications.json"
        delist_adjudication = load_json(delist_adjudication_path)
        if (
            not delist_adjudication
            or delist_adjudication.get("schema_version") != 1
            or delist_adjudication.get("type") != "verified_delist_date_nontrading_adjudication"
            or not isinstance(delist_adjudication.get("records"), list)
        ):
            raise RuntimeError("已核验退市日裁决清单缺失或版本不支持")
        delist_adjudication_snapshot_path = source_dir / "verified_delist_date_adjudications.json"
        save_json_atomic(delist_adjudication, delist_adjudication_snapshot_path)
        delist_date_exclusion_rows = []
        delist_adjudication_checks = []
        registry_rows = registry_capture["normalized"].copy()
        registry_rows["symbol"] = registry_rows["symbol"].astype(str).str.strip().str.zfill(6)
        registry_rows["ts_code"] = registry_rows["ts_code"].astype(str).str.upper()
        registry_rows["delist_date"] = pd.to_datetime(registry_rows["delist_date"], errors="coerce").dt.date
        calendar_dates = set(pd.to_datetime(calendar["cal_date"], errors="coerce").dt.date.dropna())
        for record in delist_adjudication["records"]:
            if record.get("status") != "verified":
                continue
            if record.get("classification") != "delisted_on_registry_date_nontrading":
                raise RuntimeError(f"退市日裁决类型不支持：{record.get('adjudication_id')}")
            ts_code = str(record.get("ts_code") or "").strip().upper()
            code = str(record.get("code") or "").strip().zfill(6)
            official_delist_date = datetime.date.fromisoformat(str(record.get("official_delist_date"))[:10])
            scope_start = datetime.date.fromisoformat(str(record.get("verified_scope_start_date"))[:10])
            scope_end = datetime.date.fromisoformat(str(record.get("verified_scope_end_date_exclusive"))[:10])
            if (
                not ts_code
                or not code.isdigit()
                or scope_start != official_delist_date
                or scope_end != official_delist_date + datetime.timedelta(days=1)
            ):
                raise RuntimeError(f"退市日裁决身份或范围无效：{record.get('adjudication_id')}")
            identity = registry_rows.loc[
                registry_rows["symbol"].eq(code) & registry_rows["ts_code"].eq(ts_code)
            ]
            if len(identity) != 1 or identity.iloc[0]["list_status"] != "D" or identity.iloc[0]["delist_date"] != official_delist_date:
                raise RuntimeError(f"退市日裁决与当前证券身份快照不一致：{record.get('adjudication_id')}")
            evidence_list = record.get("evidence", [])
            evidence_by_kind = {evidence.get("kind"): evidence for evidence in evidence_list}
            official_notice_kinds = {"official_company_notice", "official_exchange_notice"}
            status_capture_kind = "independent_daily_status_capture"
            universe_capture_kind = "independent_daily_universe_absence_capture"
            if (
                len(evidence_list) != 3
                or len(evidence_by_kind) != 3
                or not official_notice_kinds.intersection(evidence_by_kind)
                or not ({status_capture_kind, "independent_daily_quote_capture"}.issubset(evidence_by_kind)
                        or {universe_capture_kind, "independent_daily_quote_capture"}.issubset(evidence_by_kind))
            ):
                raise RuntimeError(f"退市日裁决缺少公告或独立行情证据：{record.get('adjudication_id')}")
            notice = evidence_by_kind[next(iter(official_notice_kinds.intersection(evidence_by_kind)))]
            if (
                not str(notice.get("url") or "").startswith("https://")
                or official_delist_date.isoformat() not in str(notice.get("fact") or "")
            ):
                raise RuntimeError(f"退市日裁决公告来源或事实日期无效：{record.get('adjudication_id')}")
            evidence_checks = []
            evidence_paths = {}
            for kind, evidence in evidence_by_kind.items():
                evidence_file = (delist_adjudication_path.parent / str(evidence.get("path") or "")).resolve()
                if (
                    _root().parent.resolve() not in evidence_file.parents
                    or not evidence_file.is_file()
                    or _sha256(evidence_file) != evidence.get("sha256")
                ):
                    raise RuntimeError(f"退市日裁决证据缺失或哈希不符：{record.get('adjudication_id')}")
                evidence_paths[kind] = evidence_file
                evidence_checks.append({"kind": kind, "path": str(evidence_file.resolve()), "sha256": evidence["sha256"]})
            if status_capture_kind in evidence_paths:
                status_capture = pd.read_parquet(evidence_paths[status_capture_kind])
                status_capture["date"] = pd.to_datetime(status_capture["date"], errors="coerce").dt.date
                status_rows = status_capture.loc[status_capture["date"].eq(official_delist_date)]
                volume_values = pd.to_numeric(status_rows["volume"], errors="coerce") if "volume" in status_rows else pd.Series(dtype=float)
                amount_values = pd.to_numeric(status_rows["amount"], errors="coerce") if "amount" in status_rows else pd.Series(dtype=float)
                if len(status_rows) == 1 and (
                    str(status_rows.iloc[0]["tradestatus"]) != "0"
                    or not (pd.isna(volume_values.iloc[0]) or volume_values.iloc[0] == 0)
                    or not (pd.isna(amount_values.iloc[0]) or amount_values.iloc[0] == 0)
                ):
                    raise RuntimeError(f"BaoStock 退市日状态与裁决不一致：{record.get('adjudication_id')}")
                if len(status_rows) == 0:
                    capture_batch_id = str(evidence_by_kind[status_capture_kind].get("capture_batch_id") or "")
                    capture_manifest_path = _root().parent / "shadow" / capture_batch_id / "manifest.json"
                    capture_manifest = load_json(capture_manifest_path)
                    capture_items = [
                        item for item in (capture_manifest or {}).get("work_items", [])
                        if str(item.get("ts_code") or "").upper() == ts_code
                        and int(item.get("year") or 0) == official_delist_date.year
                    ]
                    capture_item = capture_items[0] if len(capture_items) == 1 else {}
                    baostock_request = capture_item.get("source_request", {}).get("baostock", {})
                    if (
                        not capture_batch_id.isalnum()
                        or len(capture_batch_id) != 12
                        or (capture_manifest or {}).get("batch_id") != capture_batch_id
                        or capture_item.get("status") not in ("collected_review", "non_trading_review")
                        or (capture_item.get("status") == "non_trading_review" and capture_item.get("data_availability_status") not in {"no_tradable_rows_confirmed_by_suspension_and_delist_evidence", "no_quote_on_registry_delist_date_requires_official_evidence"})
                        or capture_item.get("code") != code
                        or not (str(capture_item.get("start") or "") <= official_delist_date.isoformat() <= str(capture_item.get("end") or ""))
                        or baostock_request.get("api") != "query_history_k_data_plus"
                        or baostock_request.get("symbol") != f"{'sh' if code.startswith('6') else 'sz'}.{code}"
                        or not (str(baostock_request.get("start_date") or "") <= official_delist_date.isoformat() <= str(baostock_request.get("end_date") or ""))
                        or int(capture_item.get("baostock_rows", -1)) != len(status_capture)
                        or capture_item.get("normalized_response_sha256", {}).get("baostock") != hashlib.sha256(status_capture.to_json(orient="records", date_format="iso").encode()).hexdigest()
                    ):
                        raise RuntimeError(f"BaoStock 退市日无行证据未能通过已完成采集批次核验：{record.get('adjudication_id')}")
                    evidence_checks.append({
                        "kind": "independent_source_capture_manifest",
                        "source": "baostock",
                        "path": str(capture_manifest_path.resolve()),
                        "sha256": _sha256(capture_manifest_path),
                        "batch_id": capture_batch_id,
                    })
                elif len(status_rows) != 1:
                    raise RuntimeError(f"BaoStock 退市日状态记录不唯一：{record.get('adjudication_id')}")
            else:
                capture_manifest_path = evidence_paths[universe_capture_kind]
                capture_manifest = load_json(capture_manifest_path)
                expected_symbol = f"{'sh' if code.startswith('6') else 'sz'}.{code}"
                raw_path = (capture_manifest_path.parent / str(capture_manifest.get("raw_file") or "")).resolve()
                if (
                    capture_manifest.get("source") != "baostock.query_all_stock"
                    or capture_manifest.get("status") != "captured"
                    or capture_manifest.get("date") != official_delist_date.isoformat()
                    or capture_manifest.get("error_code") != "0"
                    or capture_manifest.get("fields") != ["code", "tradeStatus", "code_name"]
                    or capture_manifest.get("target_code") != code
                    or capture_manifest.get("target_symbol") != expected_symbol
                    or capture_manifest.get("target_present") is not False
                    or _root().parent.resolve() not in raw_path.parents
                    or not raw_path.is_file()
                    or _sha256(raw_path) != capture_manifest.get("raw_sha256")
                ):
                    raise RuntimeError(f"BaoStock 全市场历史名册证据无效：{record.get('adjudication_id')}")
                universe_capture = pd.read_parquet(raw_path)
                universe_codes = universe_capture.get("code", pd.Series(dtype=object)).astype(str)
                if (
                    len(universe_capture) != int(capture_manifest.get("universe_rows", -1))
                    or universe_capture.empty
                    or universe_codes.isna().any()
                    or universe_codes.duplicated().any()
                    or universe_codes.eq(expected_symbol).any()
                ):
                    raise RuntimeError(f"BaoStock 全市场历史名册不完整或包含目标证券：{record.get('adjudication_id')}")
            quote_capture = pd.read_parquet(evidence_paths["independent_daily_quote_capture"])
            quote_capture["trade_date"] = pd.to_datetime(quote_capture["trade_date"], errors="coerce").dt.date
            if quote_capture["trade_date"].eq(official_delist_date).any():
                raise RuntimeError(f"TuShare 在裁决退市日存在行情，与排除裁决冲突：{record.get('adjudication_id')}")
            quote_evidence = evidence_by_kind["independent_daily_quote_capture"]
            capture_batch_id = str(quote_evidence.get("capture_batch_id") or "")
            if capture_batch_id:
                capture_manifest_path = _root().parent / "shadow" / capture_batch_id / "manifest.json"
                capture_manifest = load_json(capture_manifest_path)
                capture_items = [
                    item for item in (capture_manifest or {}).get("work_items", [])
                    if str(item.get("ts_code") or "").upper() == ts_code
                    and int(item.get("year") or 0) == official_delist_date.year
                ]
                capture_item = capture_items[0] if len(capture_items) == 1 else {}
                tushare_request = capture_item.get("source_request", {}).get("tushare", {}).get("daily", {})
                if (
                    not capture_batch_id.isalnum()
                    or len(capture_batch_id) != 12
                    or (capture_manifest or {}).get("batch_id") != capture_batch_id
                    or capture_item.get("status") not in ("collected_review", "non_trading_review")
                    or (capture_item.get("status") == "non_trading_review" and capture_item.get("data_availability_status") not in {"no_tradable_rows_confirmed_by_suspension_and_delist_evidence", "no_quote_on_registry_delist_date_requires_official_evidence"})
                    or capture_item.get("code") != code
                    or not (str(capture_item.get("start") or "") <= official_delist_date.isoformat() <= str(capture_item.get("end") or ""))
                    or tushare_request.get("api") != "pro.daily"
                    or tushare_request.get("ts_code") != ts_code
                    or not (str(tushare_request.get("start_date") or "") <= official_delist_date.strftime("%Y%m%d") <= str(tushare_request.get("end_date") or ""))
                    or int(capture_item.get("tushare_rows", -1)) != len(quote_capture)
                    or capture_item.get("normalized_response_sha256", {}).get("tushare") != hashlib.sha256(quote_capture.to_json(orient="records", date_format="iso").encode()).hexdigest()
                ):
                    raise RuntimeError(f"TuShare 退市日无行证据未能通过已完成采集批次核验：{record.get('adjudication_id')}")
                evidence_checks.append({
                    "kind": "independent_source_capture_manifest",
                    "source": "tushare",
                    "path": str(capture_manifest_path.resolve()),
                    "sha256": _sha256(capture_manifest_path),
                    "batch_id": capture_batch_id,
                })
            applies_to_batch = start <= official_delist_date <= end
            if applies_to_batch and official_delist_date not in calendar_dates:
                raise RuntimeError(f"退市日裁决不在交易日历中：{record.get('adjudication_id')}")
            if applies_to_batch:
                delist_date_exclusion_rows.append({
                    "code": code,
                    "ts_code": ts_code,
                    "trade_date": pd.Timestamp(official_delist_date),
                    "adjudication_id": record["adjudication_id"],
                })
            delist_adjudication_checks.append({
                "adjudication_id": record["adjudication_id"],
                "ts_code": ts_code,
                "code": code,
                "official_delist_date": official_delist_date.isoformat(),
                "generated_exclusion": applies_to_batch,
                "evidence": evidence_checks,
            })
        delist_date_exclusions = pd.DataFrame(delist_date_exclusion_rows, columns=[
            "code", "ts_code", "trade_date", "adjudication_id",
        ])
        adjudicated_rows = []
        adjudication_checks = []
        for record in adjudication["records"]:
            if record.get("status") != "verified":
                continue
            if record.get("classification") != "full_day_suspended":
                raise RuntimeError(f"停牌裁决类型不支持：{record.get('adjudication_id')}")
            ts_code = str(record.get("ts_code") or "").strip().upper()
            code = str(record.get("code") or "").strip().zfill(6)
            interval_start = datetime.date.fromisoformat(str(record.get("suspension_start_date"))[:10])
            resume_date = datetime.date.fromisoformat(str(record.get("resumption_date_exclusive"))[:10])
            scope_start = datetime.date.fromisoformat(str(record.get("verified_scope_start_date"))[:10])
            scope_end = datetime.date.fromisoformat(str(record.get("verified_scope_end_date_exclusive"))[:10])
            if not ts_code or not code.isdigit() or interval_start >= resume_date or scope_start >= scope_end:
                raise RuntimeError(f"停牌裁决身份或有效区间无效：{record.get('adjudication_id')}")
            evidence_checks = []
            for evidence in record.get("evidence", []):
                evidence_file = adjudication_path.parent / str(evidence.get("path") or "")
                if not evidence_file.is_file() or _sha256(evidence_file) != evidence.get("sha256"):
                    raise RuntimeError(f"停牌裁决证据缺失或哈希不符：{record.get('adjudication_id')}")
                evidence_checks.append({"path": str(evidence_file.resolve()), "sha256": evidence["sha256"]})
            if not evidence_checks:
                raise RuntimeError(f"停牌裁决缺少证据：{record.get('adjudication_id')}")
            applies_from = max(start, interval_start, scope_start)
            applies_until = min(end + datetime.timedelta(days=1), resume_date, scope_end)
            trade_dates = pd.to_datetime(calendar["cal_date"], errors="coerce").dt.date
            applicable_dates = trade_dates.loc[
                trade_dates.ge(applies_from) & trade_dates.lt(applies_until)
            ]
            if (
                applies_from == scope_start
                and applies_until == scope_end
                and len(applicable_dates) != int(record.get("verified_trade_date_count", -1))
            ):
                raise RuntimeError(f"停牌裁决交易日数量与核验证据不一致：{record.get('adjudication_id')}")
            for trade_date in applicable_dates:
                adjudicated_rows.append({
                    "ts_code": ts_code,
                    "trade_date": pd.Timestamp(trade_date),
                    "suspend_timing": "",
                    "suspend_type": "S",
                    "adjudication_id": record["adjudication_id"],
                    "evidence_scope": f"{scope_start.isoformat()}..{(applies_until - datetime.timedelta(days=1)).isoformat()}",
                })
            adjudication_checks.append({
                "adjudication_id": record["adjudication_id"],
                "ts_code": ts_code,
                "code": code,
                "generated_trade_dates": int(len(applicable_dates)),
                "evidence": evidence_checks,
            })
        adjudicated_suspensions = pd.DataFrame(adjudicated_rows, columns=[
            "ts_code", "trade_date", "suspend_timing", "suspend_type", "adjudication_id", "evidence_scope",
        ])
        tushare_suspension_keys = set(zip(
            suspensions["ts_code"].astype(str).str.upper(),
            pd.to_datetime(suspensions["trade_date"], errors="coerce").dt.normalize(),
        ))
        if not adjudicated_suspensions.empty:
            adjudicated_suspensions["already_reported_by_tushare"] = [
                (row.ts_code, row.trade_date.normalize()) in tushare_suspension_keys
                for row in adjudicated_suspensions.itertuples(index=False)
            ]
            additional_suspensions = adjudicated_suspensions.loc[
                ~adjudicated_suspensions["already_reported_by_tushare"],
                ["ts_code", "trade_date", "suspend_timing", "suspend_type"],
            ]
        else:
            adjudicated_suspensions["already_reported_by_tushare"] = pd.Series(dtype=bool)
            additional_suspensions = pd.DataFrame(columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"])
        adjudicated_keys = set(zip(
            adjudicated_suspensions["ts_code"].astype(str).str.upper(),
            pd.to_datetime(adjudicated_suspensions["trade_date"], errors="coerce").dt.normalize(),
        ))
        suspension_keys = list(zip(
            suspensions["ts_code"].astype(str).str.upper(),
            pd.to_datetime(suspensions["trade_date"], errors="coerce").dt.normalize(),
        ))
        superseded_suspensions = suspensions.loc[
            [key in adjudicated_keys for key in suspension_keys]
        ]
        effective_suspensions = pd.concat([
            suspensions.loc[[key not in adjudicated_keys for key in suspension_keys]],
            adjudicated_suspensions[["ts_code", "trade_date", "suspend_timing", "suspend_type"]],
        ], ignore_index=True)
        if effective_suspensions.duplicated(["ts_code", "trade_date"]).any():
            raise RuntimeError("TuShare 与裁决停牌日期合并后存在重复键")
        adjudicated_suspensions_path = source_dir / "adjudicated_suspensions.parquet"
        effective_suspensions_path = source_dir / "effective_suspensions.parquet"
        save_parquet(adjudicated_suspensions, adjudicated_suspensions_path)
        save_parquet(effective_suspensions, effective_suspensions_path)
        alias_catalog_path = _root().parent / "identity" / "stock_code_aliases.json"
        alias_catalog = load_json(alias_catalog_path)
        if not alias_catalog or alias_catalog.get("schema_version") != 1:
            raise RuntimeError("证券代码别名清单缺失或版本不支持")
        alias_snapshot_path = source_dir / "stock_code_aliases.json"
        save_json_atomic(alias_catalog, alias_snapshot_path)
        source_files = {}
        denominator_logic_path = Path(__file__).resolve().parents[2] / "core/domain/data_quality/historical_denominator.py"
        source_files["historical_denominator_logic"] = _sha256(denominator_logic_path)
        source_files["stock_code_aliases"] = _sha256(alias_snapshot_path)
        source_files["suspension_adjudication_manifest"] = _sha256(adjudication_snapshot_path)
        source_files["delist_date_adjudication_manifest"] = _sha256(delist_adjudication_snapshot_path)
        source_files["trading_day_override_adjudication_manifest"] = _sha256(trading_override_snapshot_path)
        source_files["adjudicated_suspensions"] = _sha256(adjudicated_suspensions_path)
        source_files["effective_suspensions"] = _sha256(effective_suspensions_path)
        delist_date_exclusions_path = source_dir / "delist_date_exclusions.parquet"
        save_parquet(delist_date_exclusions, delist_date_exclusions_path)
        source_files["delist_date_exclusions"] = _sha256(delist_date_exclusions_path)
        manifest["identity_alias_catalog_sha256"] = source_files["stock_code_aliases"]
        for status, frame in registry_capture["raw"].items():
            path = source_dir / f"stock_basic_{status}.parquet"
            save_parquet(frame, path)
            source_files[f"stock_basic_{status}"] = _sha256(path)
        for name, frame in (("trade_calendar", calendar), ("suspensions", suspensions)):
            path = source_dir / f"{name}.parquet"
            save_parquet(frame, path)
            source_files[name] = _sha256(path)
        manifest.update({
            "status": "processing",
            "updated_at": now_iso(),
            "source_version": get_historical_universe_source_version(),
            "denominator_contract_version": "active_listing_interval_v6_dual_source_no_quote_delist_evidence_adjudicated_suspension_precedence",
            "denominator_logic_sha256": source_files["historical_denominator_logic"],
            "source_observed_at": now_iso(),
            "source_semantics": "current TuShare registry snapshot with effective list/delist dates; separately hashed evidence-verified full-day suspensions supersede same-key TuShare suspension rows; an exact-date official delist notice plus independent BaoStock no-trade status with null or zero volume/amount, or a completed BaoStock daily capture covering the date with no quote row, and an independent TuShare daily capture with no quote row may exclude exactly the verified delist date from the candidate denominator; absent-row evidence is tied to completed source batch manifests and normalized capture hashes; a quote row on a verified non-trading delist date is retained as an adjudicated anomaly and excluded from coverage; registry is not a historical publication-time snapshot",
            "source_files": source_files,
            "suspension_adjudication": {
                "manifest_sha256": source_files["suspension_adjudication_manifest"],
                "verified_records": adjudication_checks,
                "generated_rows": int(len(adjudicated_suspensions)),
                "additional_rows_not_in_tushare_suspend_d": int(len(additional_suspensions)),
                "tushare_rows_superseded_by_adjudication": int(len(superseded_suspensions)),
                "effective_rows": int(len(effective_suspensions)),
            },
            "delist_date_adjudication": {
                "manifest_sha256": source_files["delist_date_adjudication_manifest"],
                "verified_records": delist_adjudication_checks,
                "generated_exclusions": int(len(delist_date_exclusions)),
            },
            "trading_day_overrides": {
                "manifest_sha256": source_files["trading_day_override_adjudication_manifest"],
                "verified_records": trading_override_checks,
                "overridden_tushare_suspension_rows": len(trading_override_keys),
            },
            "registry_counts": {
                status: int(len(frame)) for status, frame in registry_capture["raw"].items()
            },
            "calendar_trade_dates": int(len(calendar)),
            "suspension_rows": int(len(suspensions)),
            "adjudicated_suspension_rows": int(len(adjudicated_suspensions)),
            "effective_suspension_rows": int(len(effective_suspensions)),
            "suspension_query_contract": "month partitions; recursively bisected when a request returns 5000 rows; single-day saturation fails closed",
        })
        save_json_atomic(manifest, manifest_path)

        year_summaries = []
        years = list(range(start.year, end.year + 1))
        for index, year in enumerate(years, 1):
            if task_manager.is_cancel_requested(job_id):
                raise TaskCancelled()
            year_start = max(start, datetime.date(year, 1, 1)).isoformat()
            year_end = min(end, datetime.date(year, 12, 31)).isoformat()
            quotes = pd.DataFrame(HistoricalQuoteQualityRepo.quote_snapshot(year_start, year_end))
            result = build_historical_denominator(
                registry_capture["normalized"], calendar, effective_suspensions, quotes, year_start, year_end,
                code_aliases=alias_catalog.get("aliases", []),
                delist_date_exclusions=delist_date_exclusions,
            )
            key_path = batch_dir / f"expected_keys_{year}.parquet"
            summary_path = batch_dir / f"stock_year_summary_{year}.parquet"
            unexpected_path = batch_dir / f"unexpected_quote_keys_{year}.parquet"
            unmapped_suspension_path = batch_dir / f"unmapped_suspensions_{year}.parquet"
            excluded_suspension_path = batch_dir / f"excluded_suspensions_{year}.parquet"
            outside_suspension_path = batch_dir / f"suspensions_outside_expected_{year}.parquet"
            out_of_scope_suspension_path = batch_dir / f"suspensions_out_of_scope_{year}.parquet"
            unclassified_suspension_path = batch_dir / f"suspensions_unclassified_scope_{year}.parquet"
            legacy_alias_path = batch_dir / f"legacy_alias_quotes_{year}.parquet"
            legacy_alias_unverified_path = batch_dir / f"legacy_alias_unverified_quotes_{year}.parquet"
            excluded_delist_dates_path = batch_dir / f"excluded_delist_dates_{year}.parquet"
            adjudicated_delist_quotes_path = batch_dir / f"adjudicated_delist_quote_keys_{year}.parquet"
            save_parquet(result["expected_keys"], key_path)
            save_parquet(result["stock_year_summary"], summary_path)
            save_parquet(result["unexpected_quote_keys"], unexpected_path)
            save_parquet(result["unmapped_suspensions"], unmapped_suspension_path)
            save_parquet(result["excluded_suspensions"], excluded_suspension_path)
            save_parquet(result["suspensions_outside_expected"], outside_suspension_path)
            save_parquet(result["out_of_scope_suspensions"], out_of_scope_suspension_path)
            save_parquet(result["unclassified_suspensions"], unclassified_suspension_path)
            save_parquet(result["legacy_alias_quotes"], legacy_alias_path)
            save_parquet(result["legacy_alias_unverified_quotes"], legacy_alias_unverified_path)
            save_parquet(result["excluded_delist_dates"], excluded_delist_dates_path)
            save_parquet(result["adjudicated_delist_quote_keys"], adjudicated_delist_quotes_path)
            year_summaries.append({
                "year": year,
                **result["summary"],
                "expected_keys_sha256": _sha256(key_path),
                "stock_year_summary_sha256": _sha256(summary_path),
                "unexpected_quote_keys_sha256": _sha256(unexpected_path),
                "unmapped_suspensions_sha256": _sha256(unmapped_suspension_path),
                "excluded_suspensions_sha256": _sha256(excluded_suspension_path),
                "suspensions_outside_expected_sha256": _sha256(outside_suspension_path),
                "suspensions_out_of_scope_sha256": _sha256(out_of_scope_suspension_path),
                "suspensions_unclassified_scope_sha256": _sha256(unclassified_suspension_path),
                "legacy_alias_quotes_sha256": _sha256(legacy_alias_path),
                "legacy_alias_unverified_quotes_sha256": _sha256(legacy_alias_unverified_path),
                "excluded_delist_dates_sha256": _sha256(excluded_delist_dates_path),
                "adjudicated_delist_quote_keys_sha256": _sha256(adjudicated_delist_quotes_path),
                "quote_snapshot_rows": int(len(quotes)),
            })
            manifest["year_summaries"] = year_summaries
            manifest["updated_at"] = now_iso()
            save_json_atomic(manifest, manifest_path)
            if progress_cb:
                progress_cb({
                    "stage": "historical_denominator",
                    "current": index,
                    "total": len(years),
                    "message": f"G0 分母核算 {year} 年完成",
                })

        blockers = sorted({blocker for item in year_summaries for blocker in item["source_blockers"]})
        manifest.update({
            "status": "review_required",
            "updated_at": now_iso(),
            "year_summaries": year_summaries,
            "denominator_verified": not blockers,
            "denominator_blockers": blockers,
            "strict_pit_status_verified": False,
            "official_database_writes": False,
            "qlib_writes": False,
            "publishable": False,
        })
        save_json_atomic(manifest, manifest_path)
        return {
            "batch_id": batch_id,
            "status": manifest["status"],
            "denominator_verified": manifest["denominator_verified"],
            "denominator_blockers": blockers,
            "year_summaries": year_summaries,
            "manifest": str(manifest_path.relative_to(_root().parent.parent)),
            "official_database_writes": False,
            "qlib_writes": False,
        }
    except TaskCancelled:
        manifest["status"] = "partial"
        manifest["updated_at"] = now_iso()
        save_json_atomic(manifest, manifest_path)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["updated_at"] = now_iso()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save_json_atomic(manifest, manifest_path)
        raise
    finally:
        with _active_lock:
            if _active_job_id == job_id:
                _active_job_id = None


def start_historical_denominator_shadow(start, end):
    global _active_job_id
    start_date, end_date = _validate_range(start, end)
    with _active_lock:
        if _active_job_id:
            return {"status": "busy", "job_id": _active_job_id, "message": "已有 G0 历史分母任务运行中"}
        batch_id = uuid.uuid4().hex[:12]
        job_id = f"g0{batch_id}"
        batch_dir = _root() / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "batch_id": batch_id,
            "job_id": job_id,
            "status": "queued",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "basis": "effective_date_reconstruction",
            "universe_scope": {
                "exchanges": ["SSE", "SZSE"],
                "reason": "当前 Qlib instruments/all.txt 仅含 sh/sz 标的；BaoStock 日线采集也以 sh/sz 为范围",
                "excluded_exchanges": ["BSE"],
            },
            "strict_pit_status_verified": False,
            "official_database_writes": False,
            "qlib_writes": False,
            "publishable": False,
        }
        save_json_atomic(manifest, batch_dir / "manifest.json")
        _active_job_id = job_id

    def run(progress_cb):
        return _execute_batch(batch_id, job_id, start_date, end_date, progress_cb)

    try:
        task_manager.submit_with_id(
            job_id, run, f"G0 历史证券池分母 {start_date} 至 {end_date}",
            progress_cb=lambda progress: task_manager.update_progress(job_id, progress), pool="light",
        )
    except Exception:
        with _active_lock:
            if _active_job_id == job_id:
                _active_job_id = None
        manifest["status"] = "failed_to_start"
        manifest["updated_at"] = now_iso()
        save_json_atomic(manifest, batch_dir / "manifest.json")
        raise
    return {"batch_id": batch_id, "job_id": job_id, "status": "queued", "official_database_writes": False}


def get_historical_denominator_shadow(batch_id):
    manifest_path = _manifest_path(batch_id)
    manifest = load_json(manifest_path)
    if not manifest or manifest.get("batch_id") != batch_id:
        return {"batch_id": batch_id, "status": "not_found"}
    return manifest


def cancel_historical_denominator_shadow(batch_id):
    manifest_path = _manifest_path(batch_id)
    manifest = load_json(manifest_path)
    if not manifest or manifest.get("batch_id") != batch_id:
        return {"batch_id": batch_id, "status": "not_found"}
    job_id = manifest.get("job_id")
    if not job_id:
        return {"batch_id": batch_id, "status": "not_running"}
    requested = task_manager.request_cancel(job_id)
    return {
        "batch_id": batch_id,
        "job_id": job_id,
        "status": "cancel_requested" if requested else "not_running",
    }
