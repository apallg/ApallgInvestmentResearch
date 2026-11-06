"""定时调度 API — 持仓风险扫描 + 数据采集 + 情绪分析"""
import anyio
from fastapi import APIRouter

router = APIRouter()


# ═══ 统一状态 ═══

@router.get("/all-status")
async def all_status():
    from core.shared.infrastructure.jobs import get_scheduler, get_collect_scheduler, get_smoke_scheduler
    from core.adapters.nlp.scheduler import get_sentiment_scheduler
    from core.adapters.nlp.notice_scheduler import get_notice_scheduler

    def safe_status(getter, name):
        try:
            s = getter()
            st = s.status
            return st() if callable(st) else st
        except Exception as e:
            return {"error": str(e), "name": name}

    north_flow = await anyio.to_thread.run_sync(_north_flow_status)
    capital_flow = await anyio.to_thread.run_sync(_capital_flow_status)

    return {
        "data": {
            "risk_scan": safe_status(get_scheduler, "risk_scan"),
            "data_collection": safe_status(get_collect_scheduler, "data_collection"),
            "sentiment": safe_status(get_sentiment_scheduler, "sentiment"),
            "notice": safe_status(get_notice_scheduler, "notice"),
            "north_flow": north_flow,
            "capital_flow": capital_flow,
            "smoke_test": safe_status(get_smoke_scheduler, "smoke_test"),
        }
    }


# ═══ 持仓风险扫描调度器 ═══

@router.get("/status")
async def scheduler_status():
    from core.shared.infrastructure.jobs import get_scheduler
    return {"data": get_scheduler().status}


@router.post("/start")
async def scheduler_start():
    from core.shared.infrastructure.jobs import get_scheduler
    s = get_scheduler()
    s.start()
    return {"data": s.status}


@router.post("/stop")
async def scheduler_stop():
    from core.shared.infrastructure.jobs import get_scheduler
    s = get_scheduler()
    s.stop()
    return {"data": s.status}


@router.post("/trigger")
async def scheduler_trigger():
    from core.shared.infrastructure.jobs import get_scheduler
    get_scheduler().trigger()
    return {"data": {"triggered": True}}


# ═══ 数据采集调度器 ═══

@router.get("/collection/status")
async def collection_status():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    return {"data": get_collect_scheduler().status}


@router.post("/collection/start")
async def collection_start():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    get_collect_scheduler().start()
    return {"data": get_collect_scheduler().status}


@router.post("/collection/stop")
async def collection_stop():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    get_collect_scheduler().stop()
    return {"data": get_collect_scheduler().status}


@router.post("/collection/trigger")
async def collection_trigger():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    get_collect_scheduler().trigger()
    return {"data": {"triggered": True}}


# ═══ 冒烟测试调度器 ═══

@router.get("/smoke/status")
async def smoke_status():
    from core.shared.infrastructure.jobs import get_smoke_scheduler
    return {"data": get_smoke_scheduler().status}


@router.post("/smoke/start")
async def smoke_start():
    from core.shared.infrastructure.jobs import get_smoke_scheduler
    get_smoke_scheduler().start()
    return {"data": get_smoke_scheduler().status}


@router.post("/smoke/stop")
async def smoke_stop():
    from core.shared.infrastructure.jobs import get_smoke_scheduler
    get_smoke_scheduler().stop()
    return {"data": get_smoke_scheduler().status}


@router.post("/smoke/trigger")
async def smoke_trigger():
    from core.shared.infrastructure.jobs import get_smoke_scheduler
    get_smoke_scheduler().trigger()
    return {"data": {"triggered": True}}


@router.get("/smoke/history")
async def smoke_history(limit: int = 30):
    from core.shared.infrastructure.smoke_test import get_smoke_history
    return {"data": await anyio.to_thread.run_sync(get_smoke_history, limit)}


# ═══ 情绪分析调度器 ═══

@router.get("/sentiment/status")
async def sentiment_status():
    from core.adapters.nlp.scheduler import get_sentiment_scheduler
    return {"data": get_sentiment_scheduler().status}


@router.post("/sentiment/start")
async def sentiment_start():
    from core.adapters.nlp.scheduler import get_sentiment_scheduler
    get_sentiment_scheduler().start()
    return {"data": get_sentiment_scheduler().status}


@router.post("/sentiment/stop")
async def sentiment_stop():
    from core.adapters.nlp.scheduler import get_sentiment_scheduler
    get_sentiment_scheduler().stop()
    return {"data": get_sentiment_scheduler().status}


@router.post("/sentiment/trigger")
async def sentiment_trigger():
    from core.adapters.nlp.scheduler import get_sentiment_scheduler
    get_sentiment_scheduler().trigger()
    return {"data": {"triggered": True}}


# ═══ 公告采集调度器 ═══

@router.get("/notice/status")
async def notice_status():
    from core.adapters.nlp.notice_scheduler import get_notice_scheduler
    return {"data": get_notice_scheduler().status}


@router.post("/notice/start")
async def notice_start():
    from core.adapters.nlp.notice_scheduler import get_notice_scheduler
    get_notice_scheduler().start()
    return {"data": get_notice_scheduler().status}


@router.post("/notice/stop")
async def notice_stop():
    from core.adapters.nlp.notice_scheduler import get_notice_scheduler
    get_notice_scheduler().stop()
    return {"data": get_notice_scheduler().status}


@router.post("/notice/trigger")
async def notice_trigger():
    from core.adapters.nlp.notice_scheduler import get_notice_scheduler
    get_notice_scheduler().trigger()
    return {"data": {"triggered": True}}


# ═══ 北向资金采集 ═══

def _north_flow_status():
    try:
        from core.data import Database
        rows = Database.fetchall(
            "SELECT MAX(trade_date) as last_date, COUNT(*) as total, "
            "COUNT(DISTINCT code) as stocks FROM north_flow"
        )
        r = rows[0] if rows else {}
        last_date = r.get("last_date")
        return {
            "last_date": last_date.isoformat() if hasattr(last_date, "isoformat") else str(last_date) if last_date else None,
            "total_rows": r.get("total", 0),
            "stock_count": r.get("stocks", 0),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/north-flow/status")
async def north_flow_status():
    return {"data": await anyio.to_thread.run_sync(_north_flow_status)}


@router.post("/north-flow/collect")
async def north_flow_collect():
    from core.domain.screening._screener import fetch_and_save_north_flow
    n = await anyio.to_thread.run_sync(fetch_and_save_north_flow)
    return {"data": {"collected": n}}


@router.post("/north-flow/backfill")
async def north_flow_backfill(date_start: str, date_end: str):
    import datetime as dt
    from core.domain.screening._screener import backfill_north_flow
    start = dt.date.fromisoformat(date_start)
    end = dt.date.fromisoformat(date_end)
    results = await anyio.to_thread.run_sync(backfill_north_flow, start, end)
    ok = sum(1 for r in results if r["status"] == "ok" and r["count"] > 0)
    return {"data": {"total": len(results), "ok": ok, "results": results}}


# ═══ 资金流向采集 ═══

def _capital_flow_status():
    try:
        from core.data import Database
        rows = Database.fetchall(
            "SELECT MAX(trade_date) as last_date, COUNT(*) as total, "
            "COUNT(DISTINCT code) as stocks FROM capital_flow_daily"
        )
        r = rows[0] if rows else {}
        last_date = r.get("last_date")
        return {
            "last_date": last_date.isoformat() if hasattr(last_date, "isoformat") else str(last_date) if last_date else None,
            "total_rows": r.get("total", 0),
            "stock_count": r.get("stocks", 0),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/capital-flow/status")
async def capital_flow_status():
    return {"data": await anyio.to_thread.run_sync(_capital_flow_status)}


@router.post("/capital-flow/collect")
async def capital_flow_collect():
    from core.domain.screening._screener import fetch_and_save_capital_flow
    import datetime as dt
    n = await anyio.to_thread.run_sync(fetch_and_save_capital_flow, date=dt.date.today())
    return {"data": {"collected": n}}


@router.post("/capital-flow/backfill")
async def capital_flow_backfill(date_start: str, date_end: str):
    import datetime as dt
    from core.domain.screening._screener import backfill_capital_flow
    start = dt.date.fromisoformat(date_start)
    end = dt.date.fromisoformat(date_end)
    results = await anyio.to_thread.run_sync(backfill_capital_flow, start, end)
    ok = sum(1 for r in results if r["status"] == "ok" and r["count"] > 0)
    return {"data": {"total": len(results), "ok": ok, "results": results}}
