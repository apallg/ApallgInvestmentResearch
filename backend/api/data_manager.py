"""
数据中心 API — 全量数据完整性概览 + 逐表修复
"""
import datetime
import logging
import anyio
from fastapi import APIRouter, Body, Header, HTTPException, Query
from backend.service.data_repair import data_repair_service
from core.shared.qlib_errors import QlibRestartRequired

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/history/quote-audit")
async def history_quote_audit(
    code: str | None = None,
    task_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    from backend.service.historical_quality import search_quote_ingest_audit
    return {"data": await anyio.to_thread.run_sync(
        search_quote_ingest_audit, code, task_id, start, end, page, page_size,
    )}


@router.get("/history-quality")
async def history_quality():
    from backend.service.historical_quality import get_historical_quality
    return {"data": await anyio.to_thread.run_sync(get_historical_quality)}


@router.post("/history/preflight")
async def history_preflight():
    from backend.service.historical_quality import get_historical_preflight
    return {"data": await anyio.to_thread.run_sync(get_historical_preflight)}


@router.post("/history/industry-snapshots")
async def start_history_industry_snapshots(payload: dict = Body(default={})):
    from backend.service.stock_industry_history import start_stock_industry_history
    return {"data": await anyio.to_thread.run_sync(
        start_stock_industry_history,
        payload.get("start_date", "2018-12-24"), payload.get("end_date"), payload.get("force") is True,
    )}


@router.get("/history/industry-snapshots")
async def history_industry_snapshots(start_date: str | None = None, end_date: str | None = None):
    from backend.service.stock_industry_history import get_stock_industry_history_status
    return {"data": await anyio.to_thread.run_sync(get_stock_industry_history_status, start_date, end_date)}


@router.post("/history/industry-snapshots/{job_id}/cancel")
async def cancel_history_industry_snapshots(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    return {"data": {"job_id": job_id, "status": "cancel_requested" if task_manager.request_cancel(job_id) else "not_running"}}


@router.post("/history/baseline")
async def history_baseline(start: str = "2019-01-01", end: str | None = None):
    from backend.service.historical_quality import create_historical_baseline
    return {"data": await anyio.to_thread.run_sync(create_historical_baseline, start, end)}


@router.post("/history/denominator-shadow")
async def history_denominator_shadow_start(payload: dict = Body(...)):
    from backend.service.historical_universe import start_historical_denominator_shadow
    return {"data": await anyio.to_thread.run_sync(
        start_historical_denominator_shadow, payload.get("start"), payload.get("end"),
    )}


@router.get("/history/denominator-shadow/{batch_id}")
async def history_denominator_shadow_status(batch_id: str):
    from backend.service.historical_universe import get_historical_denominator_shadow
    return {"data": await anyio.to_thread.run_sync(get_historical_denominator_shadow, batch_id)}


@router.post("/history/denominator-shadow/{batch_id}/cancel")
async def history_denominator_shadow_cancel(batch_id: str):
    from backend.service.historical_universe import cancel_historical_denominator_shadow
    return {"data": await anyio.to_thread.run_sync(cancel_historical_denominator_shadow, batch_id)}


@router.post("/history/shadow-batches/preview")
async def history_shadow_batch_preview(payload: dict = Body(...)):
    from backend.service.historical_repair import prepare_shadow_batch
    return {"data": await anyio.to_thread.run_sync(
        prepare_shadow_batch, payload.get("codes") or [], payload.get("start"), payload.get("end"), payload.get("anchor_date"), payload.get("denominator_batch_id"),
    )}


@router.post("/history/shadow-batches/{batch_id}/start")
async def history_shadow_batch_start(batch_id: str):
    from backend.service.historical_repair import start_shadow_batch
    return {"data": await anyio.to_thread.run_sync(start_shadow_batch, batch_id)}


@router.get("/history/shadow-batches/{batch_id}")
async def history_shadow_batch_status(batch_id: str):
    from backend.service.historical_repair import get_shadow_batch
    manifest = await anyio.to_thread.run_sync(get_shadow_batch, batch_id)
    return {"data": manifest or {"status": "not_found", "batch_id": batch_id}}


@router.get("/history/shadow-batches/{batch_id}/{code}/{year}/conflicts")
async def history_shadow_batch_conflicts(batch_id: str, code: str, year: int, limit: int = 500,
                                         denominator_batch_id: str | None = None):
    from backend.service.historical_repair import get_shadow_conflicts
    return {"data": await anyio.to_thread.run_sync(
        get_shadow_conflicts, batch_id, code, year, limit, denominator_batch_id,
    )}


@router.post("/history/shadow-batches/{batch_id}/cancel")
async def history_shadow_batch_cancel(batch_id: str):
    from backend.service.historical_repair import cancel_shadow_batch
    return {"data": await anyio.to_thread.run_sync(cancel_shadow_batch, batch_id)}


@router.post("/history/shadow-batches/{batch_id}/{code}/{year}/release-preview")
async def history_shadow_release_preview(batch_id: str, code: str, year: int, payload: dict = Body(...)):
    from backend.service.historical_repair import prepare_shadow_release
    return {"data": await anyio.to_thread.run_sync(
        prepare_shadow_release, batch_id, code, year, payload.get("denominator_batch_id", ""),
        payload.get("replace_conflicts") is True,
        tuple(payload.get("repair_fields") or ()),
    )}


@router.get("/history/releases/{release_id}")
async def history_release_status(release_id: str):
    from backend.service.historical_repair import get_historical_release
    return {"data": await anyio.to_thread.run_sync(get_historical_release, release_id)}


@router.post("/history/releases/{release_id}/publish")
async def history_release_publish(release_id: str, authorization: str | None = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.historical_repair import publish_historical_release

    require_admin_actor(authorization)
    return {"data": await anyio.to_thread.run_sync(publish_historical_release, release_id)}


@router.post("/history/releases/{release_id}/rollback")
async def history_release_rollback(release_id: str, authorization: str | None = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.historical_repair import rollback_historical_release

    require_admin_actor(authorization)
    return {"data": await anyio.to_thread.run_sync(rollback_historical_release, release_id)}


@router.get("/history/releases/{release_id}/recovery-check")
async def history_release_recovery_check(release_id: str, authorization: str | None = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.historical_repair import inspect_historical_release_recovery

    require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(inspect_historical_release_recovery, release_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.post("/history/releases/{release_id}/recovery-restore")
async def history_release_recovery_restore(
    release_id: str, payload: dict = Body(...), authorization: str | None = Header(None),
):
    from backend.api.auth import require_admin_actor
    from backend.service.historical_repair import authorize_historical_release_recovery_rollback

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            authorize_historical_release_recovery_rollback,
            release_id,
            payload.get("expected_database_fingerprint"),
            payload.get("expected_qlib_state_fingerprint"),
            actor,
            payload.get("reason"),
            payload.get("evidence_ref"),
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}

# ── 分类映射 ──

CATEGORY_MAP = {
    "daily_quotes":    ("market",       "行情数据",   "📈"),
    "ths_daily_kline": ("market",       "行情数据",   "📈"),
    "daily_basic":     ("valuation",    "估值与资金", "💰"),
    "capital_flow_daily": ("events", "事件与资金结构", "💹"),
    "stock_sectors":   ("identity",     "股票身份与关系", "🏷️"),
    "north_flow":      ("valuation",    "估值与资金", "💰"),
    "etf_fund_flow":   ("valuation",    "估值与资金", "💰"),
    "margin_daily":    ("events",       "事件与资金结构", "💹"),
    "lhb_daily":       ("events",       "事件与资金结构", "💹"),
    "macro_shibor":    ("macro",        "宏观数据",   "🌐"),
    "macro_lpr":       ("macro",        "宏观数据",   "🌐"),
    "macro_money_supply": ("macro",     "宏观数据",   "🌐"),
    "macro_cpi":       ("macro",        "宏观数据",   "🌐"),
    "macro_ppi":       ("macro",        "宏观数据",   "🌐"),
    "macro_pmi":       ("macro",        "宏观数据",   "🌐"),
    "fundamental_data": ("fundamental", "基本面财务", "📊"),
    "forecast_data":   ("fundamental",  "基本面财务", "📊"),
    "analyst_forecast_report_captures": ("fundamental", "基本面财务", "📊"),
    "analyst_forecast_report_versions": ("fundamental", "基本面财务", "📊"),
    "income_data":     ("fundamental",  "基本面财务", "📊"),
    "shareholder_quarterly": ("events", "事件与资金结构", "💹"),
    "stock_identity": ("identity", "股票身份与关系", "🏷️"),
    "stock_index_membership": ("identity", "行业成员关系", "🏷️"),
    "stock_dividend_events": ("events", "事件与资金结构", "💹"),
    "raw_news":        ("news",         "新闻公告",   "📰"),
    "stock_notices":   ("events",       "事件与资金结构", "💹"),
}


def _resolve_repair_func(table: str):
    return data_repair_service.resolve_repair_func(table)


@router.get("/overview")
async def data_overview(force: bool = False):
    def _run():
        from core.data.collection_gate import GateCheckService, GateStatus, RULES
        from core.adapters.qlib.bridge import QlibDataBridge

        service = GateCheckService()
        results = service.check_all(force=force)
        try:
            latest_td = data_repair_service.latest_completed_trading_day()
        except Exception:
            latest_td = datetime.date.today()

        # ── 按分类聚合 ──
        categories: dict = {}
        for rule in RULES:
            gr = results.get(rule.table)
            cat_key, cat_name, cat_icon = CATEGORY_MAP.get(
                rule.table, ("other", "其他", "📦")
            )
            if cat_key not in categories:
                categories[cat_key] = {
                    "key": cat_key,
                    "name": cat_name,
                    "icon": cat_icon,
                    "tables": [],
                }

            if gr is None:
                categories[cat_key]["tables"].append({
                    "table": rule.table,
                    "desc": rule.desc,
                    "status": "error",
                    "latest_date": None,
                    "row_count": 0,
                    "expected_min": rule.min_rows,
                    "frequency": rule.frequency,
                    "data_type": rule.data_type,
                    "issues": ["检查未返回结果"],
                    "repairable": _resolve_repair_func(rule.table) is not None,
                })
                continue

            issues = []
            if gr.status == GateStatus.STALE:
                issues.append(f"数据过期: {gr.latest_in_db}")
            elif gr.status == GateStatus.MISSING:
                issues.append("表无数据或不存在")
            elif gr.status == GateStatus.INCOMPLETE:
                if gr.reason:
                    issues.append(gr.reason)
                if gr.row_count < gr.expected_min_rows:
                    issues.append(f"行数不足: {gr.row_count}/{gr.expected_min_rows}")
                if gr.null_field_ratio:
                    bad = {f: r for f, r in gr.null_field_ratio.items()
                           if r > rule.null_tolerance}
                    for f, r in bad.items():
                        issues.append(f"字段 {f} 空值率 {r:.1%}")
            elif gr.status == GateStatus.ERROR:
                issues.append(gr.reason)
            elif gr.status == GateStatus.UNAVAILABLE:
                issues.append(gr.reason)

            latest_date = None
            if gr.latest_in_db:
                if isinstance(gr.latest_in_db, datetime.date):
                    latest_date = gr.latest_in_db.isoformat()
                else:
                    latest_date = str(gr.latest_in_db)

            categories[cat_key]["tables"].append({
                "table": rule.table,
                "desc": rule.desc,
                "status": gr.status.value,
                "latest_date": latest_date,
                "row_count": gr.row_count,
                "expected_min": gr.expected_min_rows or rule.min_rows,
                "frequency": rule.frequency,
                "data_type": rule.data_type,
                "issues": issues,
                "repairable": _resolve_repair_func(rule.table) is not None,
            })

        # ── 分类健康 ──
        cat_list = []
        for ck in ["market", "valuation", "margin", "macro", "fundamental", "identity", "events", "news"]:
            if ck in categories:
                cat = categories[ck]
                statuses = [t["status"] for t in cat["tables"]]
                if any(s in ("missing", "error") for s in statuses):
                    cat["health"] = "error"
                elif any(s in ("stale", "incomplete") for s in statuses):
                    cat["health"] = "warning"
                else:
                    cat["health"] = "healthy"
                cat_list.append(cat)

        # ── Qlib 状态 ──
        qlib_status = {"synced": False, "stocks": 0, "dates": 0,
                       "first_date": None, "last_date": None}
        try:
            from core.shared import config
            from backend.service.qlib_access import run_with_qlib_read_access
            qlib_dir = getattr(config, "QLIB_DATA_DIR", "data/qlib_data")
            qs = run_with_qlib_read_access(
                lambda: QlibDataBridge(qlib_dir).get_status()
            )
            qlib_status = {
                "synced": qs.get("synced", False),
                "stocks": qs.get("stocks", 0),
                "dates": qs.get("dates", 0),
                "first_date": qs.get("first_date"),
                "last_date": qs.get("last_date"),
                "generation_id": qs.get("generation_id"),
                "current_generation_id": qs.get("current_generation_id"),
                "restart_required": qs.get("restart_required", False),
            }
        except QlibRestartRequired as e:
            qlib_status.update({
                "restart_required": True,
                "generation_id": e.generation_id,
                "current_generation_id": e.current_generation_id,
                "reason": str(e),
            })
        except Exception as e:
            logger.warning(f"Qlib 状态获取失败: {e}")

        # ── Qlib 作为独立分类 ──
        qlib_last = str(qlib_status.get("last_date") or "")[:10]
        qlib_fresh = bool(
            qlib_status["synced"] and qlib_status["stocks"] > 0
            and not qlib_status.get("restart_required")
            and qlib_last >= str(latest_td)[:10]
        )
        qlib_health = "healthy"
        if not qlib_status["synced"] or qlib_status["stocks"] == 0:
            qlib_health = "error"
        elif not qlib_fresh or qlib_status["dates"] < 100:
            qlib_health = "warning"

        cat_list.append({
            "key": "qlib",
            "name": "Qlib存储",
            "icon": "💾",
            "health": qlib_health,
            "tables": [{
                "table": "qlib_binary",
                "desc": "Qlib二进制数据",
                "status": "error" if qlib_status.get("restart_required") else "fresh" if qlib_fresh else ("stale" if qlib_status["synced"] else "missing"),
                "latest_date": qlib_status.get("last_date"),
                "row_count": qlib_status["stocks"],
                "expected_min": 4000,
                "frequency": "日",
                "data_type": "qlib",
                "issues": [qlib_status.get("reason") or "Qlib当前数据代已变化，请重启后端后验证"] if qlib_status.get("restart_required") else [] if qlib_fresh else (
                    [f"Qlib 数据过期: {qlib_last or '无日期'}"]
                    if qlib_status["synced"] else ["Qlib 数据未同步"]
                ),
                "repairable": not qlib_status.get("restart_required", False),
            }],
        })

        # ── 全局健康 ──
        all_tables = []
        for cat in cat_list:
            all_tables.extend(cat["tables"])

        fresh = sum(1 for t in all_tables if t["status"] == "fresh")
        stale = sum(1 for t in all_tables if t["status"] == "stale")
        missing = sum(1 for t in all_tables if t["status"] == "missing")
        incomplete = sum(1 for t in all_tables if t["status"] == "incomplete")
        error = sum(1 for t in all_tables if t["status"] == "error")
        unavailable = sum(1 for t in all_tables if t["status"] == "unavailable")
        monitored_total = len(all_tables) - unavailable

        overall = {
            "total": len(all_tables),
            "monitored_total": monitored_total,
            "fresh": fresh,
            "stale": stale,
            "missing": missing,
            "incomplete": incomplete,
            "error": error,
            "unavailable": unavailable,
            "health_pct": round(fresh / monitored_total * 100) if monitored_total else 0,
            "recent_update_rate": round(fresh / monitored_total * 100) if monitored_total else 0,
            "metric_scope": "按数据频率检查最近一期新鲜度，不代表历史完整度、Qlib截面覆盖或实验准入",
        }

        return {
            "as_of": datetime.date.today().isoformat(),
            "latest_trading_day": latest_td.isoformat() if isinstance(latest_td, datetime.date) else str(latest_td),
            "overall_health": overall,
            "categories": cat_list,
            "qlib": {
                "stocks": qlib_status["stocks"],
                "dates": qlib_status["dates"],
                "synced": qlib_status["synced"],
                "restart_required": qlib_status.get("restart_required", False),
                "generation_id": qlib_status.get("generation_id"),
                "current_generation_id": qlib_status.get("current_generation_id"),
                "fresh": qlib_fresh,
                "freshness_only": True,
                "cross_section_validated": False,
                "validation_note": "当前仅按末日判断新鲜度；逐日截面覆盖须单独验证",
                "first_date": qlib_status.get("first_date"),
                "last_date": qlib_status.get("last_date"),
            },
        }

    return {"data": await anyio.to_thread.run_sync(_run)}


@router.post("/repair/{table}")
async def repair_table(table: str):
    return {"data": await anyio.to_thread.run_sync(data_repair_service.start_table, table)}


@router.post("/repair-all")
async def repair_all(payload: dict | None = Body(default=None)):
    preview_id = (payload or {}).get("preview_id")
    return {"data": await anyio.to_thread.run_sync(data_repair_service.start_all, preview_id)}


@router.post("/repair-preview")
async def repair_preview():
    return {"data": await anyio.to_thread.run_sync(data_repair_service.preview_all)}


@router.get("/jobs")
async def data_jobs(limit: int = 50):
    from core.shared.infrastructure.async_task import task_manager
    from backend.service.collection import list_jobs

    limit = max(1, min(limit, 200))
    task_jobs = task_manager.list_recent(limit)
    qlib_jobs = list_jobs(limit)
    jobs = task_jobs + qlib_jobs
    jobs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return {"data": jobs[:limit]}


@router.post("/context/collect/{kind}")
async def collect_stock_context(kind: str):
    from core.shared.infrastructure.async_task import task_manager
    import uuid

    collectors = {
        "identity": ("股票身份与关系", ("collect_stock_identity", "collect_stock_identity_industry_fallback", "collect_stock_index_membership", "collect_stock_concept_membership")),
        "events": ("分红事件", ("collect_dividend_events",)),
    }
    if kind not in collectors:
        return {"data": {"status": "error", "message": "未知的数据方向"}}
    title, methods = collectors[kind]
    job_id = f"context{uuid.uuid4().hex[:8]}"

    def run():
        from backend.service import stock_context
        results = {}
        for index, method in enumerate(methods, 1):
            task_manager.update_progress(job_id, {
                "stage": "collecting", "current": index - 1, "total": len(methods),
                "message": f"正在采集 {method}",
            })
            results[method] = getattr(stock_context, method)()
            task_manager.update_progress(job_id, {
                "stage": "collecting", "current": index, "total": len(methods),
                "message": f"已完成 {method}",
            })
        return {"results": results}

    task_manager.submit_with_id(job_id, run, f"采集: {title}", pool="default")
    return {"data": {"job_id": job_id, "status": "pending", "message": f"已提交{title}采集"}}
