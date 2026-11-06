"""
系统管理 API — 健康检查 + 异步任务查询 + 配置管理
统一前缀: /api/system
"""
import anyio
from fastapi import APIRouter, BackgroundTasks
from core.shared.infrastructure.async_task import task_manager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health():
    from core.data import Database
    from core.shared.infrastructure import redis_client as rds

    def _check():
        return {
            "mysql": Database.is_available(),
            "redis": rds.redis_recheck(),
        }
    result = await anyio.to_thread.run_sync(_check)
    return {"data": {"status": "ok", **result}}


@router.get("/tasks")
async def list_tasks(limit: int = 50):
    return {"data": await anyio.to_thread.run_sync(task_manager.list_recent, limit)}


@router.get("/tasks/{job_id}")
async def get_task(job_id: str):
    task = await anyio.to_thread.run_sync(task_manager.get_dict, job_id)
    if task is None:
        return {"data": None, "error": f"任务 {job_id} 不存在"}
    return {"data": task}


@router.get("/health/redis-stats")
async def redis_stats():
    from core.shared.infrastructure.redis_client import get_redis_stats
    from core.shared.infrastructure import redis_client as rds

    def _run():
        rds.redis_recheck()
        return get_redis_stats()
    return {"data": await anyio.to_thread.run_sync(_run)}


@router.get("/health/data-sources")
async def data_source_health():
    """检查所有数据源的可用性"""
    import time

    def _check():
        results = {}

        # Redis
        try:
            from core.shared.infrastructure import redis_client as rds
            t0 = time.time()
            available = rds.redis_recheck()
            results["redis"] = {"available": available, "latency_ms": round((time.time()-t0)*1000)}
        except Exception as e:
            results["redis"] = {"available": False, "error": str(e)}

        # MySQL
        try:
            from core.data import Database
            t0 = time.time()
            if Database.is_available():
                Database.fetchall("SELECT 1")
                results["mysql"] = {"available": True, "latency_ms": round((time.time()-t0)*1000)}
            else:
                results["mysql"] = {"available": False, "error": "未启用"}
        except Exception as e:
            results["mysql"] = {"available": False, "error": str(e)}

        # TuShare
        try:
            from core.data.sources.tushare import get_pro
            t0 = time.time()
            pro = get_pro()
            if pro:
                results["tushare"] = {"available": True, "latency_ms": round((time.time()-t0)*1000)}
            else:
                results["tushare"] = {"available": False, "error": "Token 未配置"}
        except Exception as e:
            results["tushare"] = {"available": False, "error": str(e)}

        # BaoStock
        try:
            import baostock as bs
            t0 = time.time()
            lg = bs.login()
            bs.logout()
            results["baostock"] = {"available": lg.error_code == "0", "latency_ms": round((time.time()-t0)*1000)}
        except Exception as e:
            results["baostock"] = {"available": False, "error": str(e)}

        realtime_sources = {"available": True, "sources": {}}
        for source_key, module_name in (("tencent", "core.data.sources.tencent"), ("sina", "core.data.sources.sina")):
            try:
                __import__(module_name)
                realtime_sources["sources"][source_key] = {"available": True}
            except Exception as e:
                realtime_sources["available"] = False
                realtime_sources["sources"][source_key] = {"available": False, "error": str(e)}
        results["realtime_sources"] = realtime_sources

        # Factor pool
        from core.domain.factors.registry import get_pool
        results["factor_pool"] = {"total_factors": len(get_pool().get_active_factors())}

        return results

    return {"data": await anyio.to_thread.run_sync(_check)}


# ── forward return 预热 ──

@router.post("/forward-returns/warm-up")
async def warm_up_forward_returns(start: str = None, end: str = None):
    """预热 daily_forward_ret 物化表 (MySQL 8.0 LEAD 窗口函数)。
    不传参数默认覆盖最近 3 年。
    完成后因子校验/调优的 IC 计算将从 O(n×stocks) 降到 O(1) 查表。

    GET /api/system/forward-returns/status  查看表状态
    """
    import datetime as _dt
    from core.domain.factors.forward_returns import get_forward_cache

    if not start:
        start = (_dt.date.today() - _dt.timedelta(days=365 * 3)).isoformat()
    if not end:
        end = _dt.date.today().isoformat()

    def _run():
        ok = get_forward_cache().warm_up(start, end)
        return {"success": ok, "start": start, "end": end}

    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except Exception as e:
        return {"data": {"success": False, "error": str(e)}}


@router.get("/forward-returns/status")
async def forward_returns_status():
    """检查 daily_forward_ret 物化表状态"""
    def _check():
        from core.data import Database
        if not Database.is_available():
            return {"available": False, "error": "MySQL 不可用"}

        cnt = Database.fetchone("SELECT COUNT(*) AS n FROM daily_forward_ret")
        total = cnt["n"] if cnt else 0

        daterange = Database.fetchone(
            "SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx FROM daily_forward_ret")
        return {
            "available": total > 0,
            "total_rows": total,
            "date_from": str(daterange["mn"]) if daterange and daterange["mn"] else None,
            "date_to": str(daterange["mx"]) if daterange and daterange["mx"] else None,
        }

    try:
        return {"data": await anyio.to_thread.run_sync(_check)}
    except Exception as e:
        return {"data": {"available": False, "error": str(e)}}


# ── 线程池监控 ──

@router.get("/threads")
async def get_thread_status():
    """获取所有线程池的实时状态"""
    import threading
    from core.shared.infrastructure.thread_pool import thread_pool

    stats = thread_pool.stats()
    all_threads = threading.enumerate()
    stats["global"] = {
        "total_threads": len(all_threads),
        "daemon_threads": sum(1 for t in all_threads if t.daemon),
        "pool_threads": sum(1 for t in all_threads if t.name.startswith("qt-")),
    }
    return {"data": stats}


@router.get("/threads/dump")
async def dump_thread_stacks():
    """dump 所有线程的调用栈 (诊断死锁/卡死用)"""
    import threading
    import sys
    import traceback

    frames = []
    for th in threading.enumerate():
        frame = sys._current_frames().get(th.ident)
        stack = traceback.format_stack(frame) if frame else []
        frames.append({
            "name": th.name,
            "ident": th.ident,
            "daemon": th.daemon,
            "alive": th.is_alive(),
            "stack": "".join(stack[-8:]) if stack else "",
        })

    return {"data": {"threads": frames}}


# ── 错误日志查询 ──

@router.get("/errors")
async def list_errors(date: str = None, level: str = None,
                      source: str = None, limit: int = 100, offset: int = 0):
    from core.shared.error_store import query_errors
    rows = await anyio.to_thread.run_sync(
        query_errors, date_str=date, level=level, source=source,
        limit=min(limit, 500), offset=offset)
    return {"data": rows}


@router.get("/errors/summary")
async def error_summary(days: int = 7):
    from core.shared.error_store import query_date_summary
    rows = await anyio.to_thread.run_sync(
        query_date_summary, days=min(days, 365))
    return {"data": rows}


@router.post("/errors/cleanup")
async def cleanup_errors(before: str):
    from core.shared.error_store import cleanup
    deleted = await anyio.to_thread.run_sync(cleanup, before)
    return {"data": {"deleted": deleted, "before": before}}


@router.post("/rebuild-fund-cache")
async def rebuild_fund_cache(background_tasks: BackgroundTasks):
    """手动重建基本面因子缓存（从 MySQL 重新预计算）"""
    import time as _time

    def _rebuild():
        try:
            t0 = _time.time()
            from backend.service.fundamental_cache import FundamentalCacheService
            result = FundamentalCacheService.rebuild()
            elapsed = _time.time() - t0
            logger.info(f"[FundCache] 重建服务返回: {result}, {elapsed:.0f}s")
        except Exception as e:
            logger.exception(f"[FundCache] 手动重建异常: {e}")

    background_tasks.add_task(_rebuild)
    return {"data": {"status": "started", "message": "基本面缓存重建已启动，查看后端日志获取进度"}}


@router.get("/integrity")
async def data_integrity():
    """查询最近交易日数据完整性状态"""
    from core.data.integrity import get_status
    result = await anyio.to_thread.run_sync(get_status)
    return {"data": result}


@router.post("/integrity/check")
async def run_integrity_check(dry_run: bool = False):
    """手动触发数据完整性校验 (dry_run=true 仅检查不回补)"""
    from core.data.integrity import check_latest
    result = await anyio.to_thread.run_sync(lambda: check_latest(auto_repair=not dry_run, dry_run=dry_run))
    return {"data": result}


@router.post("/integrity/repair")
async def run_integrity_repair(date: str = None):
    """手动触发指定日期的数据回补 (不指定=最近交易日)"""
    from core.data.integrity import check_and_repair
    result = await anyio.to_thread.run_sync(lambda: check_and_repair(trade_date=date, auto_repair=True))
    return {"data": result}
