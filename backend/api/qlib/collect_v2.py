"""Qlib 集成 API — 数据采集/同步端点（薄接口，委托 CollectionService）"""
from fastapi import APIRouter, BackgroundTasks, Body, Header

from backend.api.qlib.models import SyncRequest, CollectRequest
from backend.service.collection import CollectionService

router = APIRouter()


@router.post("/data/sync")
async def sync_data(req: SyncRequest, background_tasks: BackgroundTasks):
    """触发 MySQL → qlib 二进制格式数据同步"""
    return CollectionService.sync_data(req, background_tasks)


@router.post("/data/sync-benchmarks")
async def sync_benchmarks(background_tasks: BackgroundTasks):
    return CollectionService.sync_benchmarks(background_tasks)


@router.post("/data/collect")
async def collect_data(req: CollectRequest, background_tasks: BackgroundTasks):
    """采集最新行情数据到 MySQL，然后增量同步到 qlib"""
    return CollectionService.collect_data(req, background_tasks)


@router.get("/data/sync/{job_id}")
async def sync_status(job_id: str):
    """查询同步/采集任务状态"""
    return CollectionService.sync_status(job_id)


@router.get("/data/status")
async def data_status():
    """获取 qlib 数据同步状态"""
    return CollectionService.data_status()


@router.get("/data/collection/status")
async def collection_status():
    return CollectionService.collection_status()


@router.post("/data/collection/start")
async def collection_start():
    return CollectionService.collection_start()


@router.post("/data/collection/stop")
async def collection_stop():
    return CollectionService.collection_stop()


@router.post("/data/collection/trigger")
async def collection_trigger():
    return CollectionService.collection_trigger()


@router.post("/data/collect-fundamentals")
async def collect_fundamentals(pool: str = Body("all"), max_workers: int = Body(4)):
    """手动触发基本面数据采集。Tushare 多线程优先，akshare 补充。"""
    return CollectionService.collect_fundamentals(pool, max_workers)


@router.get("/data/collection/history")
async def collection_history(days: int = 7):
    return CollectionService.collection_history(days)


@router.post("/data/collect-incremental")
async def collect_incremental(
    pool: str = Body("all"),
    lookback_days: int = Body(5),
    threads: int = Body(32),
    background_tasks: BackgroundTasks = None,
):
    """Tencent 增量更新：多线程拉取最近行情，只更新有缺口的股票"""
    return CollectionService.collect_incremental(pool, lookback_days, threads, background_tasks)


@router.post("/data/collect-incremental/{job_id}/retry-failed")
async def retry_failed_incremental(job_id: str, background_tasks: BackgroundTasks, threads: int = Body(32, embed=True)):
    """仅重试指定历史任务中明确记录失败的证券。"""
    return CollectionService.retry_failed_incremental(job_id, threads, background_tasks)


@router.post("/data/sync-fundamentals-pit")
async def sync_fundamentals_pit():
    """同步基本面 PIT 数据到 qlib 特征目录"""
    return CollectionService.sync_fundamental_pit()


@router.post("/data/fundamental-cache/rebuild")
def rebuild_fundamental_cache(authorization: str | None = Header(default=None)):
    from backend.api.auth import require_admin_actor
    actor = require_admin_actor(authorization)
    return CollectionService.rebuild_fundamental_cache(actor)


@router.get("/data/pit/check")
async def check_pit():
    """检查基本面 PIT 文件是否已生成且字段完整"""
    from backend.service.pit import PitService
    return PitService.check_pit()


@router.post("/data/collect-full")
async def collect_full(
    pool: str = Body("all"),
    threads: int = Body(4),
    delay: float = Body(0.5),
    background_tasks: BackgroundTasks = None,
):
    """BaoStock 全量下载：逐只拉取全部历史日线，排除退市/ST，volume=股"""
    return CollectionService.collect_full(pool, threads, delay, background_tasks)
