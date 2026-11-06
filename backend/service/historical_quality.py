import datetime
import hashlib
from pathlib import Path
from backend.service.qlib_access import qlib_read_locked

from core.data.repos.historical_quality import HistoricalQuoteQualityRepo
from core.shared.persistence import load_json, load_pickle
from core.shared.persistence import now_iso
from core.shared.persistence import save_json_atomic


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def search_quote_ingest_audit(code=None, task_id=None, start_date=None, end_date=None, page=1, page_size=50):
    from core.data.repos.quote_audit import DailyQuoteAuditRepo
    return DailyQuoteAuditRepo.search(code, task_id, start_date, end_date, page, page_size)


def _find_verified_denominator(start, end):
    root = Path(__file__).resolve().parents[2] / "data/historical_repair/denominator"
    candidates = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            manifest = load_json(manifest_path)
            if (
                manifest.get("denominator_verified") is True
                and manifest.get("date_range", {}).get("start", "") <= start
                and manifest.get("date_range", {}).get("end", "") >= end
                and set(manifest.get("universe_scope", {}).get("exchanges", [])) == {"SSE", "SZSE"}
            ):
                candidates.append((manifest.get("updated_at", ""), manifest_path.parent, manifest))
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        return None
    _, batch_dir, manifest = max(candidates, key=lambda item: item[0])
    summaries = {
        int(item["year"]): item for item in manifest.get("year_summaries", [])
        if isinstance(item, dict) and item.get("year") is not None
    }
    yearly = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        summary = summaries.get(year)
        keys = batch_dir / f"expected_keys_{year}.parquet"
        if (
            not summary or summary.get("denominator_verified") is not True or not keys.is_file()
            or _file_sha256(keys) != summary.get("expected_keys_sha256")
        ):
            return None
        expected = int(summary.get("expected_trading_stock_dates", 0))
        complete = int(summary.get("observed_price_complete", 0))
        missing = int(summary.get("missing_row_unresolved", 0))
        invalid = int(summary.get("observed_row_invalid_price", 0))
        if expected <= 0 or complete != expected or missing != 0 or invalid != 0:
            return None
        yearly.append(summary)
    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "date_range": manifest.get("date_range"),
        "strict_pit_status_verified": manifest.get("strict_pit_status_verified") is True,
        "denominator_verified": True,
        "stockyear_coverage_verified": all(
            item.get("denominator_verified") is True and int(item.get("missing_row_unresolved", 0)) == 0
            for item in yearly
        ),
        "expected_rows": sum(int(item.get("expected_trading_stock_dates", 0)) for item in yearly),
        "observed_price_complete": sum(int(item.get("observed_price_complete", 0)) for item in yearly),
        "missing_row_unresolved": sum(int(item.get("missing_row_unresolved", 0)) for item in yearly),
        "invalid_price_rows": sum(int(item.get("observed_row_invalid_price", 0)) for item in yearly),
        "yearly": [{
            "year": int(item["year"]),
            "expected_rows": int(item.get("expected_trading_stock_dates", 0)),
            "observed_price_complete": int(item.get("observed_price_complete", 0)),
            "missing_row_unresolved": int(item.get("missing_row_unresolved", 0)),
        } for item in yearly],
    }


def create_historical_baseline(start="2019-01-01", end=None):
    end = end or datetime.date.today().isoformat()
    snapshot = HistoricalQuoteQualityRepo.baseline_snapshot(start, end)
    baseline_id = hashlib.sha256(
        f"{snapshot['fingerprint']}|{now_iso()}".encode("utf-8")
    ).hexdigest()[:20]
    artifact = {
        "baseline_id": baseline_id,
        "created_at": now_iso(),
        "status": "frozen_unverified",
        "scope": "read_only_database_aggregate_snapshot",
        **snapshot,
        "writes_data": False,
        "can_certify_history_denominator": False,
    }
    output_dir = Path(__file__).resolve().parents[2] / "data/historical_repair/baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{baseline_id}.json"
    save_json_atomic(artifact, output_path)
    response = {key: value for key, value in artifact.items() if key != "stock_years"}
    response["stock_years_count"] = len(snapshot["stock_years"])
    response["artifact"] = str(output_path.relative_to(output_dir.parents[1]))
    return response


@qlib_read_locked
def get_historical_quality(start="2019-01-01", end=None):
    from core.adapters.qlib.generation import assert_no_active_historical_release
    from core.shared import config

    assert_no_active_historical_release(config.QLIB_DATA_ROOT)
    end = end or HistoricalQuoteQualityRepo.latest_trade_date() or datetime.date.today().isoformat()
    rows = HistoricalQuoteQualityRepo.yearly_rows(start, end)
    years = []
    for row in rows:
        total = int(row["rows_total"] or 0)
        missing = int(row["price_missing"] or 0)
        invalid = int(row.get("price_invalid") or 0)
        years.append({
            "year": int(row["year"]), "rows": total,
            "price_missing": missing,
            "price_missing_rate": missing / total if total else 1.0,
            "price_invalid": invalid,
            "price_invalid_rate": invalid / total if total else 1.0,
            "price_unusable_rate": (missing + invalid) / total if total else 1.0,
            "liquidity_missing": int(row["liquidity_missing"] or 0),
            "factor_only": int(row["factor_only"] or 0),
        })
    project_root = Path(__file__).resolve().parents[2]
    root = project_root / "data/experiments"
    reports = []
    for path in root.glob("stock_context_p1_*/coverage.json"):
        try:
            report = load_json(path)
            if isinstance(report, dict):
                reports.append((str(report.get("as_of") or ""), path.stat().st_mtime, report, path))
        except (OSError, ValueError, TypeError):
            continue
    panel_item = max(reports, default=None, key=lambda item: (item[0], item[1]))
    panel = panel_item[2] if panel_item else {}
    try:
        cache_root = project_root / 'data/fund_cache'
        pointer = cache_root / 'current.json'
        if pointer.exists():
            current = load_json(pointer)
            generation = current.get('generation', '')
            if not generation or Path(generation).name != generation:
                raise ValueError('非法基本面缓存代号')
            fundamental_cache_meta = load_pickle(cache_root / 'generations' / generation / 'fund_meta.pkl') or {}
        else:
            fundamental_cache_meta = load_pickle(cache_root / 'fund_meta.pkl') or {}
    except (OSError, ValueError, TypeError, EOFError):
        fundamental_cache_meta = {}
    from core.shared import config
    qlib_root = Path(config.QLIB_DATA_DIR)
    if not qlib_root.is_absolute():
        qlib_root = project_root / qlib_root
    try:
        qlib_fundamental_meta = load_json(qlib_root / "fundamental_pit_manifest.json") or {}
    except (OSError, ValueError, TypeError):
        qlib_fundamental_meta = {}
    if not isinstance(fundamental_cache_meta, dict):
        fundamental_cache_meta = {}
    if not isinstance(qlib_fundamental_meta, dict):
        qlib_fundamental_meta = {}
    fundamental_pit_verified = (
        fundamental_cache_meta.get("fundamental_pit_verified") is True
        and fundamental_cache_meta.get("historical_source_vintages_certified") is True
        and qlib_fundamental_meta.get("fundamental_pit_verified") is True
        and qlib_fundamental_meta.get("historical_source_vintages_certified") is True
    )
    missing = panel.get("missing_indicator_rate_before_value_fill") or panel.get("source_missing_indicator_rates") or {}
    contract_current = panel.get("feature_contract_version") == 2
    coverage_pass = bool(missing) and all(float(value) <= 0.02 for value in missing.values())
    quote_rows_pass = bool(years) and all(item["price_unusable_rate"] <= 0.02 for item in years)
    denominator_verified = panel.get("history_denominator_verified") is True
    stockyear_verified = panel.get("stockyear_coverage_verified") is True
    industry_pit_verified = panel.get("industry_pit_verified") is True
    certified = panel.get("certification_status") == "certified"
    quote_coverage = HistoricalQuoteQualityRepo.expected_quote_coverage(start, end)
    denominator = _find_verified_denominator(start, end)
    if denominator:
        quote_coverage.update({
            "expected_rows": denominator["expected_rows"],
            "covered_rows": denominator["observed_price_complete"],
            "missing_row_unresolved": denominator["missing_row_unresolved"],
            "coverage_rate": denominator["observed_price_complete"] / denominator["expected_rows"] if denominator["expected_rows"] else 0.0,
            "denominator_verified": denominator["denominator_verified"],
            "denominator_label": "G0 沪深历史有效上市区间，排除已核验停牌与退市日",
            "denominator_batch_id": denominator["batch_id"],
        })
    return {
        "years": years,
        "quote_denominator": quote_coverage.get("denominator_label", "现库已有行；完整历史池分母尚未核验"),
        "quote_coverage": quote_coverage,
        "panel_rows": panel.get("rows"),
        "panel_path": str(panel_item[3].relative_to(root)) if panel_item else None,
        "panel_as_of": panel.get("as_of"),
        "panel_missing_rates": missing,
        "panel_contract_current": contract_current,
        "panel_coverage_pass": coverage_pass,
        "quote_rows_pass": quote_rows_pass,
        "history_denominator_verified": denominator["denominator_verified"] if denominator else denominator_verified,
        "stockyear_coverage_verified": denominator["stockyear_coverage_verified"] if denominator else stockyear_verified,
        "strict_pit_status_verified": denominator["strict_pit_status_verified"] if denominator else False,
        "industry_pit_verified": industry_pit_verified,
        "certification_status": "certified" if certified else "uncertified",
        "fundamental_pit_verified": fundamental_pit_verified,
        "fundamental_pit_status": {
            "cache_metadata_present": bool(fundamental_cache_meta),
            "qlib_manifest_present": bool(qlib_fundamental_meta),
            "cache_source_vintages_certified": fundamental_cache_meta.get("historical_source_vintages_certified") is True,
            "qlib_source_vintages_certified": qlib_fundamental_meta.get("historical_source_vintages_certified") is True,
        },
        "training_ready": certified and contract_current and coverage_pass and quote_rows_pass and bool(denominator)
                          and denominator["stockyear_coverage_verified"] and denominator["strict_pit_status_verified"]
                          and industry_pit_verified and fundamental_pit_verified,
        "denominator": denominator,
        "industry_vintage": panel.get("industry_vintage", "未核实"),
    }


@qlib_read_locked
def get_historical_preflight(start="2019-01-01", end=None):
    from core.adapters.qlib.bridge import QlibDataBridge
    from core.adapters.qlib.generation import assert_no_active_historical_release
    from core.data.repos.historical_quality import HistoricalQuoteQualityRepo
    from core.shared import config

    assert_no_active_historical_release(config.QLIB_DATA_ROOT)
    end = end or HistoricalQuoteQualityRepo.latest_trade_date() or datetime.date.today().isoformat()
    quality = get_historical_quality(start, end)
    footprint_error = None
    try:
        footprint = HistoricalQuoteQualityRepo.historical_footprint(start, end)
    except Exception as exc:
        footprint = None
        footprint_error = str(exc)[:300]
    try:
        qlib_status = QlibDataBridge(getattr(config, "QLIB_DATA_DIR", "data/qlib_data")).get_status()
    except Exception as exc:
        qlib_status = None
        qlib_error = str(exc)[:300]
    else:
        qlib_error = None

    try:
        from backend.service.stock_industry_history import get_stock_industry_history_status
        industry_history = get_stock_industry_history_status(start, end, ensure=False)
    except Exception as exc:
        industry_history = {"status": "unavailable", "error": str(exc)[:300]}
    if footprint:
        footprint["historical_industry_snapshots"] = industry_history

    blockers = []
    if footprint_error:
        blockers.append(f"历史行情或证券身份表无法读取：{footprint_error}")
    if not quality.get("history_denominator_verified"):
        blockers.append("缺少按历史上市、退市与停牌状态构建并认证的股票年度分母")
    if not quality.get("stockyear_coverage_verified"):
        blockers.append("股票年度覆盖尚未认证，现库已有行不能作为完整性分母")
    if not quality.get("industry_pit_verified"):
        blockers.append("行业关系尚无严格的历史可见时间版本证据")
    if not quality.get("fundamental_pit_verified"):
        blockers.append("基本面供应商历史修订版本未认证，现有来源观察时间不能证明历史决策日可见值")
    if not quality.get("strict_pit_status_verified"):
        blockers.append("证券历史状态分母仅完成生效日期重建，严格 PIT 发布时点未认证")
    if not quality.get("panel_contract_current"):
        blockers.append("P1 特征面板契约版本未通过")
    if not quality.get("panel_coverage_pass"):
        blockers.append("P1 核心字段有效覆盖未达到 98% 门槛")
    if not quality.get("quote_rows_pass"):
        blockers.append("现库已有行情行仍有超过门槛的缺价或非法价格；该指标不替代固定分母覆盖率")
    if quality.get("certification_status") != "certified":
        blockers.append("没有独立质量认证签名")
    if qlib_error:
        blockers.append(f"Qlib 当前数据目录不可读：{qlib_error}")
    if not qlib_status or not qlib_status.get("synced"):
        blockers.append("Qlib 当前数据目录尚未提供可确认的同步状态")
    if footprint and not footprint["industry_observations"]["rows"] and not industry_history.get("summary", {}).get("snapshots"):
        blockers.append("尚无行业关系历史观察数据，不能证明历史关系可见时间")

    return {
        "status": "ready" if not blockers else "blocked",
        "checked_at": now_iso(),
        "range": {"start": start, "end": end},
        "summary": "历史数据发布与实验准入检查已通过" if not blockers else "只读检查未通过正式历史修复准入条件",
        "blockers": blockers,
        "quality": quality,
        "current_footprint": footprint,
        "qlib": qlib_status,
        "writes_data": False,
    }
