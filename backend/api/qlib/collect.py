"""DEPRECATED: 已由 backend/api/qlib/collect_v2.py 和 backend/service/collection.py 取代，请勿新增逻辑。"""
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from core.data.sources.qfq_normalizer import prepare_qfq_daily_frame
from backend.api.qlib.common import (  # noqa: F403
    logger, _train_jobs, _sync_lock, _collect_job_id,
    _OPTUNA_DIR, _save_optuna_result, _load_optuna_result,
    _get_qlib_data_dir, _resolve_instruments, _build_filtered_instruments,
    _filter_st_stocks, _get_all_stock_codes, _resolve_factor_label,
    _make_display_name, _assert_run_path,
    fetch_tencent_kline_raw, task_manager, thread_pool, uuid,
)
from backend.api.qlib.models import SyncRequest, CollectRequest

router = APIRouter()

@router.post("/data/sync")
async def sync_data(req: SyncRequest, background_tasks: BackgroundTasks):
    from backend.service.collection import CollectionService
    return CollectionService.sync_data(req, background_tasks)

@router.get("/data/sync/{job_id}")
async def sync_status(job_id: str):
    """查询同步/采集任务状态"""
    job = _train_jobs.get(job_id)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(404, "任务不存在")
    return {"data": job}
@router.get("/data/status")
async def data_status():
    """获取 qlib 数据同步状态"""
    from core.adapters.qlib.bridge import QlibDataBridge
    bridge = QlibDataBridge(_get_qlib_data_dir())
    return {"data": bridge.get_status()}
@router.get("/data/collection/status")
async def collection_status():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    return {"data": get_collect_scheduler().status()}
@router.post("/data/collection/start")
async def collection_start():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    get_collect_scheduler().start()
    return {"data": get_collect_scheduler().status()}
@router.post("/data/collection/stop")
async def collection_stop():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    get_collect_scheduler().stop()
    return {"data": get_collect_scheduler().status()}
@router.post("/data/collection/trigger")
async def collection_trigger():
    from core.shared.infrastructure.jobs import get_collect_scheduler
    get_collect_scheduler().trigger()
    return {"data": {"triggered": True}}
@router.post("/data/collect-fundamentals")
async def collect_fundamentals(pool: str = Body("all"), max_workers: int = Body(4)):
    """手动触发基本面数据采集。Tushare 多线程优先，akshare 补充。"""
    import threading
    from core.data.sources.tushare import get_all_stock_codes, get_index_components, get_small_cap_codes

    codes = None
    if pool == "csi300":
        codes = get_index_components("000300")
    elif pool == "csi500":
        codes = get_index_components("000905")
    elif pool == "small_cap":
        codes = get_small_cap_codes()

    def _run():
        try:
            from backend.service.fundamental_collection import fetch_and_save_fundamentals
            result = fetch_and_save_fundamentals(codes=codes, max_workers=max_workers)
            logger.info(f"[API] 基本面采集完成: {result}")
        except Exception as e:
            logger.error(f"[API] 基本面采集失败: {e}")

    thread_pool.light.submit(_run)
    return {"data": {"status": "started", "pool": pool, "message": "基本面采集已启动，请查看日志"}}
@router.get("/data/collection/history")
async def collection_history(days: int = 7):
    import datetime
    from core.data import Database
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    rows = Database.fetchall(
        "SELECT * FROM daily_collection_log WHERE created_at >= %s ORDER BY created_at DESC LIMIT 100",
        (cutoff,),
    )
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        result.append(d)
    return {"data": result}
def _get_active_codes():
    """获取当前上市股票列表（排除退市股）"""
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        codes = df["code"].tolist()
        return set(str(c) for c in codes)
    except Exception:
        return None
def _exclude_delisted(codes):
    """过滤掉退市股票"""
    active = _get_active_codes()
    if active is None:
        return codes
    before = len(codes)
    filtered = [c for c in codes if c in active]
    if before != len(filtered):
        logger.info(f"退市过滤: {before} → {len(filtered)} (排除 {before - len(filtered)} 只退市股)")
    return filtered
@router.post("/data/collect-incremental")
async def collect_incremental(
    pool: str = Body("all"),
    lookback_days: int = Body(5),
    threads: int = Body(32),
    background_tasks: BackgroundTasks = None,
):
    """Tencent 增量更新：多线程拉取最近行情，只更新有缺口的股票"""
    from scripts.fast_collect import collect_incremental as do_incremental, get_all_codes

    codes = get_all_codes(pool)
    codes = _exclude_delisted(codes)
    if not codes:
        return {"data": {"status": "error", "message": "无有效股票"}}

    job_id = str(uuid.uuid4())[:8]
    _train_jobs[job_id] = {
        "status": "running",
        "message": f"Tencent 增量更新: {len(codes)} 只, 回溯{lookback_days}天, {threads}线程"
    }

    def _do():
        try:
            n = do_incremental(codes, source="tencent", threads=threads, lookback_days=lookback_days)
            _train_jobs[job_id] = {
                "status": "completed",
                "message": f"增量更新完成: 采集 {n} 只新数据"
            }
            try:
                from core.adapters.qlib.bridge import QlibDataBridge
                bridge = QlibDataBridge(_get_qlib_data_dir())
                r = bridge.sync_incremental()
                _train_jobs[job_id]["message"] += f", Qlib同步: {r.get('new_dates', 0)}新日期"
            except Exception:
                pass
        except Exception as e:
            logger.exception("增量更新失败")
            _train_jobs[job_id] = {"status": "error", "message": str(e)}

    background_tasks.add_task(_do)
    return {"data": {"job_id": job_id, "status": "started",
                    "message": f"Tencent 增量更新 {len(codes)} 只, 回溯{lookback_days}天"}}
@router.post("/data/collect-full")
async def collect_full(
    pool: str = Body("all"),
    threads: int = Body(4),
    delay: float = Body(0.5),
    background_tasks: BackgroundTasks = None,
):
    """BaoStock 全量下载：逐只拉取全部历史日线，排除退市/ST，volume=股"""
    import time
    from scripts.fast_collect import get_all_codes

    codes = get_all_codes(pool)
    codes = _exclude_delisted(codes)
    if not codes:
        return {"data": {"status": "error", "message": "无有效股票"}}

    # 排除 ST
    try:
        from core.data.fetcher import DataFetcher
        fetcher = DataFetcher()
        non_st = []
        for c in codes:
            try:
                name = fetcher.get_stock_name(c) or ""
                if not (name.upper().startswith("ST") or name.upper().startswith("*ST")):
                    non_st.append(c)
            except Exception:
                non_st.append(c)
        codes = non_st
        logger.info(f"ST过滤后: {len(codes)} 只")
    except Exception:
        pass

    job_id = str(uuid.uuid4())[:8]
    total = len(codes)
    _train_jobs[job_id] = {
        "status": "running",
        "message": f"BaoStock 全量下载: {total} 只, {threads}线程"
    }

    def _do():
        import baostock as bs
        import pandas as pd
        from core.data import DailyQuotesRepo
        from concurrent.futures import as_completed
        from core.shared.infrastructure.thread_pool import thread_pool
        import threading as th

        lg = bs.login()
        if lg.error_code != "0":
            _train_jobs[job_id] = {"status": "error", "message": f"BaoStock 登录失败: {lg.error_msg}"}
            return

        success = [0]; fail = [0]
        lock = th.Lock()
        err_samples = []
        done = [0]

        def _fetch_one(code):
            mkt = "sh" if code.startswith(("6", "5")) else "sz"
            try:
                rs = bs.query_history_k_data_plus(
                    f"{mkt}.{code}",
                    "date,open,high,low,close,volume,amount,turn,pctChg",
                    start_date="2012-01-01", end_date="2026-12-31",
                    frequency="d", adjustflag="3",
                )
                if rs.error_code != "0":
                    raise RuntimeError(rs.error_msg)
                rows = []
                while rs.next():
                    row = rs.get_row_data()
                    try:
                        if not row[0]:
                            continue
                        rows.append({
                            "date": row[0].replace("-", ""),
                            "open": float(row[1]) if row[1] not in ("", None) else None,
                            "high": float(row[2]) if row[2] not in ("", None) else None,
                            "low": float(row[3]) if row[3] not in ("", None) else None,
                            "close": float(row[4]) if row[4] not in ("", None) else None,
                            "volume": float(row[5]) if row[5] not in ("", None) else None,
                            "amount": float(row[6]) if row[6] not in ("", None) else None,
                            "turnover": float(row[7]) if row[7] not in ("", None) else None,
                            "pct_chg": float(row[8]) if row[8] not in ("", None) else None,
                        })
                    except (ValueError, TypeError, IndexError):
                        continue
                if not rows:
                    raise RuntimeError("无数据")
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                df["amplitude"] = None
                DailyQuotesRepo.save_batch(code, prepare_qfq_daily_frame(code, df, source="baostock", task_id=job_id))
                with lock:
                    success[0] += 1
            except Exception as e:
                with lock:
                    fail[0] += 1
                    if len(err_samples) < 5:
                        err_samples.append(f"{code}: {e}")
            finally:
                with lock:
                    done[0] += 1
                if delay > 0:
                    time.sleep(delay)

        with thread_pool.default.borrow() as ex:
            futs = {ex.submit(_fetch_one, c): c for c in codes}
            for f in as_completed(futs):
                f.result()
                if done[0] % 100 == 0:
                    _train_jobs[job_id]["message"] = (
                        f"全量下载: {done[0]}/{total}, 成功 {success[0]}, 失败 {fail[0]}"
                    )

        bs.logout()
        msg = f"全量下载完成: {success[0]} 成功, {fail[0]} 失败"
        if err_samples:
            msg += f" | 错误: {'; '.join(err_samples[:3])}"
        _train_jobs[job_id] = {"status": "completed", "message": msg}

    background_tasks.add_task(_do)
    return {"data": {"job_id": job_id, "status": "started",
                    "message": f"BaoStock 全量下载 {total} 只 (预计数小时)"}}
