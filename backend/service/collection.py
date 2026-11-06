"""采集服务 — 数据采集/同步/调度编排。"""
import json
import logging
import os
import threading
import uuid

from core.data.sources.tencent_kline import fetch_tencent_kline_raw
from core.data.sources.qfq_normalizer import prepare_qfq_daily_frame
from core.domain.screening.exclusion import B_SHARE_PREFIXES
from core.shared import config
from backend.service.qlib_access import qlib_read_locked
from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)

from backend.service.qlib_benchmarks import sync_benchmark_indices

_sync_lock = threading.Lock()
_fundamental_cache_rebuild_lock = threading.Lock()
_fundamental_cache_rebuild_job_id = None
_train_jobs = {}
_job_history = []
_collect_job_id = None


def _qlib_data_dir():
    return config.QLIB_DATA_DIR


def _set_job(job_id, data):
    _train_jobs[job_id] = data
    if job_id not in _job_history:
        _job_history.append(job_id)
    if len(_job_history) > 200:
        del _job_history[:-200]
    try:
        import redis
        _r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
        _r.setex(f"qlib:job:{job_id}", 7 * 24 * 3600, json.dumps(data, ensure_ascii=False, default=str))
        _r.lrem("qlib:job_history", 0, job_id)
        _r.lpush("qlib:job_history", job_id)
        _r.ltrim("qlib:job_history", 0, 199)
    except Exception:
        pass


def get_job(job_id):
    job = _train_jobs.get(job_id)
    if job is not None:
        return job
    try:
        import redis
        value = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2).get(f"qlib:job:{job_id}")
        return json.loads(value) if value else None
    except Exception:
        return None


def list_jobs(limit=50):
    count = max(1, min(int(limit), 200))
    job_ids = list(reversed(_job_history[-count:]))
    if len(job_ids) < count:
        try:
            import redis
            client = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
            persisted = client.lrange("qlib:job_history", 0, count - 1)
            known = set(job_ids)
            job_ids.extend(job_id.decode() for job_id in persisted if job_id.decode() not in known)
        except Exception:
            pass
    jobs = []
    for job_id in job_ids[:count]:
        data = get_job(job_id) or {}
        jobs.append({"job_id": job_id, "source": "qlib", **data})
    return jobs


class CollectionService:

    @staticmethod
    def sync_data(req, background_tasks):
        from core.adapters.qlib.bridge import QlibDataBridge

        if req.full_resync:
            return {"data": {
                "status": "disabled",
                "message": "安全版全量候选构建与逐日验证尚未接入，原地全量重建已暂停；日常增量同步可用",
            }}

        if not _sync_lock.acquire(blocking=False):
            return {"data": {"status": "syncing", "message": "同步正在进行中，请稍后"}}

        job_id = str(uuid.uuid4())[:8]
        try:
            bridge = QlibDataBridge(_qlib_data_dir())
            result = bridge.sync_incremental()
            _set_job(job_id, {"status": "completed", "result": result})
            return {"data": {"job_id": job_id, "status": "completed", "result": result}}
        except Exception as error:
            logger.exception("增量同步失败")
            _set_job(job_id, {"status": "error", "error": str(error)})
            return {"data": {"job_id": job_id, "status": "error", "error": str(error)}}
        finally:
            _sync_lock.release()

    @staticmethod
    def sync_benchmarks(background_tasks):
        from core.adapters.qlib.bridge import QlibDataBridge
        from core.adapters.qlib.generation import (
            discard_qlib_generation_stage,
            has_qlib_generation_pointer,
            publish_qlib_generation,
            qlib_generation_mutation_lock,
            stage_qlib_generation,
        )
        from core.shared import config

        if not _sync_lock.acquire(blocking=False):
            return {"data": {"status": "syncing", "message": "Qlib 同步正在进行中，请完成后再更新基准指数"}}

        job_id = str(uuid.uuid4())[:8]

        def _do_sync():
            stage = None
            try:
                stage = stage_qlib_generation(data_root=config.QLIB_DATA_ROOT, lifecycle_lease=True, owner={"type": "benchmark_sync"}) if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
                bridge = QlibDataBridge(stage["stage_dir"] if stage else _qlib_data_dir())
                calendar_path = bridge.calendar_dir / "day.txt"
                if not calendar_path.exists():
                    raise RuntimeError("Qlib 交易日历不存在，请先执行全量重建")
                dates = calendar_path.read_text(encoding="utf-8").strip().splitlines()
                if not dates:
                    raise RuntimeError("Qlib 交易日历为空，请先执行全量重建")
                date_to_idx = {date: index for index, date in enumerate(dates)}
                _set_job(job_id, {"status": "running", "message": "正在获取并写入基准指数行情..."})
                if stage:
                    result = sync_benchmark_indices(bridge, dates, date_to_idx, require_current=True)
                else:
                    with qlib_generation_mutation_lock(config.QLIB_DATA_ROOT, require_legacy=True):
                        result = sync_benchmark_indices(bridge, dates, date_to_idx, require_current=True)
                if stage:
                    published = publish_qlib_generation(
                        stage,
                        metadata={"operation": "benchmark_sync", "result": result},
                    )
                    result["generation_id"] = published["generation_id"]
                    result["restart_required"] = True
                _set_job(job_id, {"status": "completed", "result": result})
            except Exception as e:
                logger.exception("基准指数 Qlib 同步失败")
                error_message = str(e) or repr(e)
                if stage:
                    try:
                        discard_qlib_generation_stage(stage)
                    except Exception as cleanup_error:
                        logger.exception("基准指数 Qlib 候选代清理失败")
                        error_message = f"{error_message}; Qlib候选暂存清理失败: {cleanup_error}"
                _set_job(job_id, {"status": "error", "error": error_message})
            finally:
                _sync_lock.release()

        _set_job(job_id, {"status": "running", "message": "基准指数同步任务已启动"})
        background_tasks.add_task(_do_sync)
        return {"data": {"job_id": job_id, "status": "started", "message": "基准指数同步任务已启动"}}

    @staticmethod
    def sync_status(job_id):
        """查询同步/采集任务状态"""
        job = get_job(job_id)
        if not job:
            from fastapi import HTTPException
            raise HTTPException(404, "任务不存在")
        return {"data": job}

    @staticmethod
    @qlib_read_locked
    def data_status():
        """获取 qlib 数据同步状态"""
        from core.adapters.qlib.bridge import QlibDataBridge
        bridge = QlibDataBridge(_qlib_data_dir())
        return {"data": bridge.get_status()}

    @staticmethod
    def collection_status():
        from core.shared.infrastructure.jobs import get_collect_scheduler
        return {"data": get_collect_scheduler().status()}

    @staticmethod
    def collection_start():
        from core.shared.infrastructure.jobs import get_collect_scheduler
        get_collect_scheduler().start()
        return {"data": get_collect_scheduler().status()}

    @staticmethod
    def collection_stop():
        from core.shared.infrastructure.jobs import get_collect_scheduler
        get_collect_scheduler().stop()
        return {"data": get_collect_scheduler().status()}

    @staticmethod
    def collection_trigger():
        from core.shared.infrastructure.jobs import get_collect_scheduler
        get_collect_scheduler().trigger()
        return {"data": {"triggered": True}}

    @staticmethod
    def collect_fundamentals(pool, max_workers):
        """手动触发基本面数据采集。Tushare 多线程优先，akshare 补充。"""
        from core.data.sources.tushare import get_all_stock_codes, get_index_components, get_small_cap_codes

        codes = None
        if pool == "csi300":
            codes = get_index_components("000300")
        elif pool == "csi500":
            codes = get_index_components("000905")
        elif pool == "small_cap":
            codes = get_small_cap_codes()

        from core.shared.infrastructure.async_task import task_manager
        job_id = f"fund{uuid.uuid4().hex[:8]}"

        def _run():
            from backend.service.fundamental_collection import fetch_and_save_fundamentals
            return fetch_and_save_fundamentals(codes=codes, max_workers=max_workers)

        task_manager.submit_with_id(job_id, _run, f"采集基本面: {pool}", pool="light")
        return {"data": {"job_id": job_id, "status": "pending", "pool": pool, "message": "基本面采集任务已提交"}}

    @staticmethod
    def rebuild_fundamental_cache(actor):
        global _fundamental_cache_rebuild_job_id
        from core.shared.fundamental_contract import FUNDVAL_CONTRACT_VERSION, fundamental_field_contract_sha256
        from core.shared.infrastructure.async_task import task_manager

        if not _fundamental_cache_rebuild_lock.acquire(blocking=False):
            return {"data": {"job_id": _fundamental_cache_rebuild_job_id, "status": "running", "message": "FundVal缓存重建已在运行"}}

        contract_sha256 = fundamental_field_contract_sha256()
        job_id = f"fundcache-{uuid.uuid4().hex[:12]}"
        _fundamental_cache_rebuild_job_id = job_id

        def _run(progress_cb):
            global _fundamental_cache_rebuild_job_id
            try:
                progress_cb({"stage": "rebuilding", "message": "正在从当前数据库快照重建FundVal缓存"})
                from core.data.fundamental import get_cache_identity, preload_cache
                if not preload_cache(force=True):
                    raise RuntimeError("FundVal缓存重建失败，旧代次保持有效但不符合当前契约")
                identity = get_cache_identity()
                if identity.get("field_contract_sha256") != contract_sha256:
                    raise RuntimeError("新缓存字段契约指纹与提交任务时不一致")
                return {
                    "generation": identity["generation"],
                    "field_contract_sha256": identity["field_contract_sha256"],
                    "actor": actor,
                }
            finally:
                _fundamental_cache_rebuild_job_id = None
                _fundamental_cache_rebuild_lock.release()

        try:
            task_manager.submit_with_id(
                job_id,
                _run,
                f"重建FundVal缓存 v{FUNDVAL_CONTRACT_VERSION} by {actor}",
                pool="light",
            )
        except Exception:
            _fundamental_cache_rebuild_job_id = None
            _fundamental_cache_rebuild_lock.release()
            raise
        return {"data": {"job_id": job_id, "status": "pending", "message": "FundVal缓存重建已提交；可在系统任务中查看结果"}}

    @staticmethod
    def sync_fundamental_pit():
        """同步基本面 PIT 数据到 qlib 特征目录。"""
        from backend.service.pit import PitService
        return PitService.sync_fundamental_pit()

    @staticmethod
    def collection_history(days):
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

    @staticmethod
    def collect_incremental(pool, lookback_days, threads, background_tasks=None):
        """Tencent 增量更新：多线程拉取最近行情，只更新有缺口的股票"""
        from scripts.fast_collect import collect_incremental as do_incremental, get_all_codes

        codes = get_all_codes(pool)
        codes = CollectionService._exclude_delisted(codes)
        if not codes:
            return {"data": {"status": "error", "message": "无有效股票"}}

        job_id = str(uuid.uuid4())[:8]
        _set_job(job_id, {
            "status": "running",
            "message": f"Tencent 增量更新: {len(codes)} 只, 回溯{lookback_days}天, {threads}线程"
        })

        def _do():
            CollectionService._run_incremental_job(job_id, codes, lookback_days, threads)

        if background_tasks is not None:
            background_tasks.add_task(_do)
        else:
            _do()
        return {"data": {"job_id": job_id, "status": "started",
                        "message": f"Tencent 增量更新 {len(codes)} 只, 回溯{lookback_days}天"}}

    @staticmethod
    def retry_failed_incremental(parent_job_id, threads=32, background_tasks=None):
        parent = get_job(parent_job_id)
        if not isinstance(parent, dict):
            return {"data": {"status": "error", "message": "原采集任务不存在或状态不可读取"}}
        collection_result = parent.get("collection_result")
        failed_codes = collection_result.get("failed_codes") if isinstance(collection_result, dict) else None
        if parent.get("collection_status") != "partial" or not isinstance(failed_codes, list):
            return {"data": {"status": "error", "message": "原任务没有可重试的采集失败明细"}}
        codes = list(dict.fromkeys(
            str(item.get("code", "")).strip()
            for item in failed_codes
            if isinstance(item, dict) and item.get("code")
        ))
        if not codes:
            return {"data": {"status": "error", "message": "原任务未提供有效失败证券代码"}}
        raw_lookback_days = collection_result.get("lookback_days", 5)
        lookback_days = raw_lookback_days if isinstance(raw_lookback_days, int) and not isinstance(raw_lookback_days, bool) and raw_lookback_days > 0 else 5

        job_id = str(uuid.uuid4())[:8]
        _set_job(job_id, {
            "status": "running",
            "parent_job_id": parent_job_id,
            "retry_codes": codes,
            "message": f"仅重试原任务失败的{len(codes)}只证券",
        })

        def _do():
            CollectionService._run_incremental_job(
                job_id,
                codes,
                lookback_days,
                threads,
                parent_job_id=parent_job_id,
            )

        if background_tasks is not None:
            background_tasks.add_task(_do)
        else:
            _do()
        return {"data": {
            "job_id": job_id,
            "parent_job_id": parent_job_id,
            "status": "started",
            "retry_codes": codes,
            "message": f"已提交仅针对原失败证券的重试：{len(codes)}只",
        }}

    @staticmethod
    def _run_incremental_job(job_id, codes, lookback_days, threads, parent_job_id=None):
        from scripts.fast_collect import collect_incremental as do_incremental

        try:
            collection_result = do_incremental(
                codes, source="tencent", threads=threads, lookback_days=lookback_days,
                return_details=True,
            )
            required = ("expected", "success", "failed", "skipped", "successful_codes", "failed_codes", "actual_date_range")
            if not isinstance(collection_result, dict) or any(key not in collection_result for key in required):
                raise RuntimeError("增量采集返回缺少结构化阶段结果")
            counts = [collection_result[key] for key in ("expected", "success", "failed", "skipped")]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
                raise RuntimeError("增量采集阶段计数非法")
            if not isinstance(collection_result["successful_codes"], list) or not isinstance(collection_result["failed_codes"], list):
                raise RuntimeError("增量采集证券明细格式非法")
            if collection_result["expected"] != len(codes) or sum(counts[1:]) != collection_result["expected"]:
                raise RuntimeError("增量采集阶段计数不满足期望=成功+失败+跳过")
            if len(collection_result["successful_codes"]) != collection_result["success"] or len(collection_result["failed_codes"]) != collection_result["failed"]:
                raise RuntimeError("增量采集证券明细数与阶段计数不一致")
            failed = collection_result["failed"] > 0
            common = {
                "parent_job_id": parent_job_id,
                "collection_status": "partial" if failed else "completed",
                "collection_result": {**collection_result, "lookback_days": lookback_days},
                "collected": collection_result["success"],
            }
            _set_job(job_id, {
                **common,
                "status": "collect_done",
                "message": (
                    f"采集阶段完成: 期望{collection_result['expected']}只，成功{collection_result['success']}只，"
                    f"失败{collection_result['failed']}只，跳过{collection_result['skipped']}只，正在同步Qlib"
                ),
            })
            try:
                from core.adapters.qlib.bridge import QlibDataBridge
                sync_result = QlibDataBridge(_qlib_data_dir()).sync_incremental()
                if not isinstance(sync_result, dict):
                    raise RuntimeError("Qlib同步未返回结构化结果")
                _set_job(job_id, {
                    **common,
                    "status": "partial" if failed else "completed",
                    "sync_status": "completed",
                    "sync_result": sync_result,
                    "message": (
                        f"增量采集{'部分完成' if failed else '完成'}: 期望{collection_result['expected']}只，"
                        f"成功{collection_result['success']}只，失败{collection_result['failed']}只，"
                        f"跳过{collection_result['skipped']}只；Qlib同步{sync_result.get('new_dates', 0)}个新日期"
                    ),
                })
            except Exception as sync_error:
                logger.exception("Tencent增量采集完成但Qlib同步失败")
                _set_job(job_id, {
                    **common,
                    "status": "partial",
                    "sync_status": "error",
                    "sync_error": str(sync_error),
                    "message": (
                        f"采集阶段{'部分完成' if failed else '完成'}"
                        f"（期望{collection_result['expected']}只，成功{collection_result['success']}只，"
                        f"失败{collection_result['failed']}只，跳过{collection_result['skipped']}只），"
                        f"但Qlib同步失败：{sync_error}"
                    ),
                })
        except Exception as error:
            logger.exception("增量更新失败")
            _set_job(job_id, {
                "status": "error",
                "parent_job_id": parent_job_id,
                "collection_status": "error",
                "message": str(error),
            })

    @staticmethod
    def collect_full(pool, threads, delay, background_tasks=None):
        """旧的全量下载会按当前存续状态过滤历史股票，暂时停用。"""
        return {"data": {
            "status": "disabled",
            "message": "旧全量下载使用当前股票名单筛除退市/ST，可能遗漏历史样本；请先完成影子批次核验",
        }}

        """BaoStock 全量下载：逐只拉取全部历史日线，排除退市/ST，volume=股"""
        import time
        from scripts.fast_collect import get_all_codes

        codes = get_all_codes(pool)
        codes = CollectionService._exclude_delisted(codes)
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
        _set_job(job_id, {
            "status": "running",
            "message": f"BaoStock 全量下载: {total} 只, {threads}线程"
        })

        def _do():
            import baostock as bs
            import pandas as pd
            from core.data import DailyQuotesRepo
            from concurrent.futures import as_completed
            from core.shared.infrastructure.thread_pool import thread_pool
            import threading as th

            lg = bs.login()
            if lg.error_code != "0":
                _set_job(job_id, {"status": "error", "message": f"BaoStock 登录失败: {lg.error_msg}"})
                return

            success = [0]
            fail = [0]
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
                        _set_job(job_id, {
                            "status": "running",
                            "message": f"全量下载: {done[0]}/{total}, 成功 {success[0]}, 失败 {fail[0]}"
                        })

            bs.logout()
            msg = f"全量下载完成: {success[0]} 成功, {fail[0]} 失败"
            if err_samples:
                msg += f" | 错误: {'; '.join(err_samples[:3])}"
            _set_job(job_id, {"status": "completed", "message": msg})

        if background_tasks is not None:
            background_tasks.add_task(_do)
        else:
            _do()
        return {"data": {"job_id": job_id, "status": "started",
                        "message": f"BaoStock 全量下载 {total} 只 (预计数小时)"}}

    @staticmethod
    def collect_data(req, background_tasks):
        """采集最新行情数据到 MySQL，然后增量同步到 qlib"""
        global _collect_job_id
        import datetime

        # 防止重复启动
        if _collect_job_id and _collect_job_id in _train_jobs:
            job = _train_jobs[_collect_job_id]
            if job.get("status") in ("running", "collect_done"):
                return {"data": {"job_id": _collect_job_id, "status": "running", "message": f"采集已在运行中 ({_collect_job_id}): {job.get('message', '')}"}}

        today = datetime.date.today()
        end_date = today.strftime("%Y%m%d")
        start_date = "20150101"

        # ── retry_missing: 从数据库查缺数据的股票，仅 Tencent ──
        if req.retry_missing:
            try:
                from core.data import Database
                rows = Database.fetchall("SELECT code, COUNT(*) as cnt FROM daily_quotes GROUP BY code HAVING cnt < 100")
                codes = [r["code"] for r in rows]
            except Exception as e:
                return {"data": {"status": "error", "message": f"查询缺失股票失败: {e}"}}

            if not codes:
                return {"data": {"status": "completed", "message": "没有缺数据的股票"}}

            job_id = str(uuid.uuid4())[:8]
            _collect_job_id = job_id
            _set_job(job_id, {"status": "running", "message": f"补采 {len(codes)} 只缺失股票 (仅Tencent)"})

            def _do_retry():
                global _collect_job_id
                try:
                    import time, random, pandas as pd
                    from concurrent.futures import as_completed
                    from core.shared.infrastructure.thread_pool import thread_pool
                    from core.data import DailyQuotesRepo

                    total = len(codes)
                    success = 0
                    fail = 0
                    lock = threading.Lock()
                    err_samples = []

                    def _fetch_raw(code):
                        try:
                            _, rows = fetch_tencent_kline_raw(code, adj="", timeout=10)
                            if rows:
                                return (code, True, rows)
                            return (code, False, None)
                        except Exception as e:
                            with lock:
                                if len(err_samples) < 3:
                                    err_samples.append(f"{code}: {e}")
                            return (code, False, None)

                    workers = max(1, min(8, (os.cpu_count() or 4) - 2))
                    with thread_pool.default.borrow() as pool:
                        futures = {pool.submit(_fetch_raw, c): c for c in codes}
                        done = 0
                        for f in as_completed(futures):
                            done += 1
                            code, ok, rows_list = f.result()
                            if ok and rows_list:
                                try:
                                    df = pd.DataFrame(rows_list)
                                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                                    df["amplitude"] = None
                                    df["turnover"] = None
                                    DailyQuotesRepo.save_batch(code, prepare_qfq_daily_frame(code, df, source="tencent", task_id=job_id))
                                    success += 1
                                except Exception as e:
                                    with lock:
                                        if len(err_samples) < 3:
                                            err_samples.append(f"{code}(save): {e}")
                                    fail += 1
                            else:
                                fail += 1
                            if done % 100 == 0:
                                sc, fl = success, fail
                                samples = err_samples[:3]
                                msg = f"补采进度: {done}/{total}, 成功 {sc}, 失败 {fl} ({workers}线程 Tencent)"
                                if samples:
                                    msg += f" | 错误样本: {samples[0]}"
                                _set_job(job_id, {"status": "running", "message": msg})

                    retry_msg = f"补采完成: 成功 {success}, 失败 {fail}"
                    if err_samples:
                        retry_msg += f" | 错误样本: {'; '.join(err_samples[:3])}"
                    _set_job(job_id, {"status": "collect_done", "message": retry_msg})
                    if success > 0:
                        from core.adapters.qlib.bridge import QlibDataBridge
                        from core.shared import config as app_config
                        bridge = QlibDataBridge(app_config.QLIB_DATA_DIR)
                        sync_result = bridge.sync_incremental()
                        _set_job(job_id, {
                            "status": "completed",
                            "message": f"补采+同步完成: {sync_result.get('new_dates',0)} 新交易日, {sync_result.get('new_stocks',0)} 新股票",
                            "collect": {"total": total, "success": success, "fail": fail},
                            "sync": sync_result,
                        })
                except Exception as e:
                    logger.exception("补采失败")
                    _set_job(job_id, {"status": "error", "error": str(e)})
                finally:
                    if _collect_job_id == job_id:
                        _collect_job_id = None

            background_tasks.add_task(_do_retry)
            return {"data": {"job_id": job_id, "status": "started", "message": f"开始补采 {len(codes)} 只缺失股票 (仅Tencent)"}}

        # ── 正常采集 ──
        codes = []
        if req.pool == "all":
            try:
                import baostock as bs
                bs.login()
                rs = bs.query_all_stock()
                while rs.next():
                    row_data = rs.get_row_data()
                    if len(row_data) >= 1:
                        raw = row_data[0]
                        if "." in raw:
                            raw = raw.split(".")[1]
                        if raw.isdigit() and len(raw) == 6 and not raw.startswith(B_SHARE_PREFIXES):
                            codes.append(raw)
                bs.logout()
            except Exception:
                pass
            if not codes:
                try:
                    import akshare as ak
                    df = ak.stock_info_a_code_name()
                    codes = df["code"].tolist()
                except Exception as e:
                    logger.warning(f"获取全部A股列表失败: {e}")

        if req.pool in ("csi300", "csi_all") or (req.pool == "all" and not codes):
            try:
                import akshare as ak
                df = ak.index_stock_cons(symbol="000300")
                if "品种代码" in df.columns:
                    codes.extend(df["品种代码"].tolist())
                elif "code" in df.columns:
                    codes.extend(df["code"].tolist())
                else:
                    codes.extend(df.iloc[:, 0].tolist())
            except Exception as e:
                logger.warning(f"获取沪深300成分股失败: {e}")
        if req.pool in ("csi500", "csi_all"):
            try:
                import akshare as ak
                df = ak.index_stock_cons(symbol="000905")
                if "品种代码" in df.columns:
                    codes.extend(df["品种代码"].tolist())
                elif "code" in df.columns:
                    codes.extend(df["code"].tolist())
                else:
                    codes.extend(df.iloc[:, 0].tolist())
            except Exception as e:
                logger.warning(f"获取中证500成分股失败: {e}")

        if not codes:
            return {"data": {"status": "error", "message": "无法获取成分股列表"}}

        codes = list(set(codes))

        # ── 查已有足量数据的股票，Tencent 降级时跳过（防止覆盖 BaoStock 数据）──
        rich_codes = set()
        try:
            from core.data import Database
            rows = Database.fetchall("SELECT code, COUNT(*) as cnt FROM daily_quotes GROUP BY code HAVING cnt >= 500")
            rich_codes = {r["code"] for r in rows}
        except Exception:
            pass

        job_id = str(uuid.uuid4())[:8]
        _collect_job_id = job_id
        use_bs = not req.force_tencent
        src_label = "BaoStock+Tencent" if use_bs else f"Tencent(跳过{len(rich_codes)}只已有数据)"
        _set_job(job_id, {"status": "running", "message": f"开始采集 {len(codes)} 只股票, {start_date}~{end_date} ({src_label})"})

        BATCH_SIZE = 120
        BATCH_PAUSE = 40
        STOCK_DELAY = 0.15

        def _do_collect():
            global _collect_job_id
            try:
                import time, random, pandas as pd
                from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
                from core.shared.infrastructure.thread_pool import thread_pool
                from core.data import DailyQuotesRepo

                success = fail = 0
                total = len(codes)
                baostock_ok = False
                bs_timeouts = 0
                lock = threading.Lock()
                err_samples = []

                if use_bs:
                    try:
                        import baostock as bs
                        lg = bs.login()
                        if lg.error_code == "0":
                            rs = bs.query_history_k_data_plus(
                                "sh.600519", "date", start_date="2025-01-01", end_date="2025-01-02",
                                frequency="d", adjustflag="3"
                            )
                            baostock_ok = (rs is not None and rs.error_code == "0")
                            if baostock_ok:
                                _set_job(job_id, {"status": "running", "message": f"BaoStock 已就绪，开始采集 {total} 只"})
                    except Exception:
                        pass

                # ── 并行 Tencent 模式 ──
                if not baostock_ok:
                    if req.retry_missing:
                        tcodes = [c for c in codes if c not in rich_codes]
                    else:
                        tcodes = list(codes)
                    if len(tcodes) < len(codes):
                        _set_job(job_id, {"status": "running", "message": f"Tencent: 跳过 {len(codes)-len(tcodes)} 只已有数据，采集 {len(tcodes)} 只"})
                    total = len(tcodes)
                    _set_job(job_id, {"status": "running", "message": _train_jobs[job_id].get("message", f"开始采集 {total} 只 (Tencent)")})

                    def _fetch_raw(code):
                        try:
                            _, rows = fetch_tencent_kline_raw(code, adj="", timeout=10)
                            if rows:
                                return (code, True, rows)
                            return (code, False, None)
                        except Exception as e:
                            with lock:
                                if len(err_samples) < 3:
                                    err_samples.append(f"{code}: {e}")
                            return (code, False, None)

                    import os as _os
                    workers = min(8, (_os.cpu_count() or 4) - 2)
                    with thread_pool.default.borrow() as pool:
                        futures = {pool.submit(_fetch_raw, c): c for c in tcodes}
                        done = 0
                        for f in as_completed(futures):
                            done += 1
                            code, ok, rows_list = f.result()
                            if ok and rows_list:
                                try:
                                    df = pd.DataFrame(rows_list)
                                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                                    df["amplitude"] = None
                                    df["turnover"] = None
                                    DailyQuotesRepo.save_batch(code, prepare_qfq_daily_frame(code, df, source="tencent", task_id=job_id))
                                    success += 1
                                except Exception as e:
                                    with lock:
                                        if len(err_samples) < 3:
                                            err_samples.append(f"{code}(save): {e}")
                                    fail += 1
                            else:
                                fail += 1
                            if done % 100 == 0:
                                sc, fl = success, fail
                                samples = err_samples[:3]
                                msg = f"采集进度: {done}/{total}, 成功 {sc}, 失败 {fl} ({workers}线程 Tencent)"
                                if samples:
                                    msg += f" | 错误样本: {samples[0]}"
                                _set_job(job_id, {"status": "running", "message": msg})

                # ── 串行 BaoStock + Tencent 降级模式 ──
                else:
                    for i, code in enumerate(codes):
                        df = None
                        ok = False
                        mkt = "sh" if code.startswith(("6", "5")) else "sz"

                        if baostock_ok:
                            try:
                                def _bs_query():
                                    import baostock as bs
                                    rs = bs.query_history_k_data_plus(
                                        f"{mkt}.{code}", "date,open,high,low,close,volume,amount",
                                        start_date="2015-01-01", end_date=str(today),
                                        frequency="d", adjustflag="3",
                                    )
                                    if rs is not None and rs.error_code == "0":
                                        rows_list = []
                                        while rs.next():
                                            rows_list.append(rs.get_row_data())
                                        return (True, rows_list, None)
                                    elif rs is not None and rs.error_code == "10001011":
                                        return (False, None, "ban")
                                    return (False, None, rs.error_code if rs else "rs_none")
                                success_flag, rows_list, err = thread_pool.default.submit(_bs_query).result(timeout=15)
                                if err == "ban":
                                    baostock_ok = False
                                    _set_job(job_id, {"status": "running", "message": f"BaoStock IP 被限 (第 {i+1}/{total})，降级到 Tencent"})
                                elif success_flag and rows_list:
                                    df = pd.DataFrame(rows_list, columns=["date", "open", "high", "low", "close", "volume", "amount"])
                                    for col in ["open", "high", "low", "close", "volume", "amount"]:
                                        df[col] = pd.to_numeric(df[col], errors="coerce")
                                    df["date"] = pd.to_datetime(df["date"])
                                    df["amplitude"] = None
                                    df["turnover"] = None
                                    quote_source = "baostock"
                                    ok = True
                            except FutureTimeoutError:
                                bs_timeouts += 1
                                if bs_timeouts >= 3:
                                    baostock_ok = False
                            except Exception:
                                pass

                        if not ok:
                            try:
                                _, rows_list = fetch_tencent_kline_raw(code, adj="", timeout=10)
                                if rows_list:
                                    df = pd.DataFrame(rows_list)
                                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                                    df["amplitude"] = None
                                    df["turnover"] = None
                                    quote_source = "tencent"
                                    ok = True
                            except Exception:
                                pass

                        if ok and df is not None and not df.empty:
                            try:
                                DailyQuotesRepo.save_batch(code, prepare_qfq_daily_frame(code, df, source=quote_source, task_id=job_id))
                                success += 1
                            except Exception:
                                fail += 1
                        else:
                            fail += 1

                        if (i + 1) % 50 == 0:
                            _set_job(job_id, {"status": "running", "message": f"采集进度: {i+1}/{total}, 成功 {success}, 失败 {fail} (BaoStock+Tencent)"})

                        if (i + 1) % BATCH_SIZE == 0 and (i + 1) < total:
                            pause = BATCH_PAUSE + random.randint(-5, 5)
                            _set_job(job_id, {"status": "running", "message": f"采集进度: {i+1}/{total}, 成功 {success}, 失败 {fail} — 暂停 {pause}s 防封..."})
                            time.sleep(pause)

                        time.sleep(STOCK_DELAY + random.uniform(0, 0.05))

                    if baostock_ok:
                        try:
                            import baostock as bs
                            bs.logout()
                        except Exception:
                            pass

                msg = f"采集完成: 成功 {success}, 失败 {fail}"
                if err_samples:
                    msg += f" | 错误样本: {'; '.join(err_samples[:3])}"
                _set_job(job_id, {
                    "status": "collect_done",
                    "message": msg,
                    "total": total, "success": success, "fail": fail,
                    "date_range": f"{start_date}~{end_date}",
                })

                if success > 0:
                    from core.adapters.qlib.bridge import QlibDataBridge
                    from core.shared import config as app_config
                    bridge = QlibDataBridge(app_config.QLIB_DATA_DIR)
                    sync_result = bridge.sync_incremental()
                    new_stocks = sync_result.get("new_stocks", 0)
                    recommend = ""
                    if new_stocks > 100:
                        recommend = f"新增 {new_stocks} 只股票，建议运行「全量同步」以获取完整历史数据"
                    elif sync_result.get("new_dates", 0) == 0 and new_stocks == 0:
                        recommend = "数据已是最新"

                    _set_job(job_id, {
                        "status": "completed",
                        "message": f"采集+同步完成: {sync_result.get('new_dates', 0)} 个新交易日, {new_stocks} 只新股票",
                        "collect": {"total": total, "success": success, "fail": fail},
                        "sync": sync_result,
                        "recommend": recommend,
                        "date_range": f"{start_date}~{end_date}",
                    })

            except Exception as e:
                logger.exception("数据采集失败")
                _set_job(job_id, {"status": "error", "error": str(e)})
            finally:
                if _collect_job_id == job_id:
                    _collect_job_id = None

        background_tasks.add_task(_do_collect)
        return {"data": {"job_id": job_id, "status": "started", "message": f"开始采集 {len(codes)} 只股票, {start_date}~{end_date}"}}

    @staticmethod
    def _get_active_codes():
        """获取当前上市股票列表（排除退市股）"""
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            codes = df["code"].tolist()
            return set(str(c) for c in codes)
        except Exception:
            return None

    @staticmethod
    def _exclude_delisted(codes):
        """过滤掉退市股票"""
        active = CollectionService._get_active_codes()
        if active is None:
            return codes
        before = len(codes)
        filtered = [c for c in codes if c in active]
        if before != len(filtered):
            logger.info(f"退市过滤: {before} → {len(filtered)} (排除 {before - len(filtered)} 只退市股)")
        return filtered
