"""
时序滚动因子筛选 — 报告修正1 (最致命): 杜绝全样本一次性筛选的数据窥探。

从 start 到 end 每 step_months 生成一个 anchor, 每个 anchor 只用其之前的数据做七/八层校验
(check_all_layers(anchor_date=anchor)), L1 Walk-Forward 用 anchor 之后的数据做严格样本外测试。
产出"每期活跃因子集"时间序列, 即动态因子池。

parallel=True: 使用 multiprocessing.Pool 并行跑锚点, Windows spawn 安全。
"""

__all__ = ["rolling_factor_screen", "_gen_anchors", "_build_master_indicators",
           "_ensure_daily_in_redis"]

import json
import logging
import gc
import time
import threading
import datetime as dt
from concurrent.futures import wait, FIRST_COMPLETED
from core.shared import config
from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)


def _add_months(d, months):
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return dt.date(y, m, day)


def _gen_anchors(start, end, step_months):
    anchors = []
    d = start
    while d <= end:
        anchors.append(d)
        d = _add_months(d, step_months)
    return anchors


def _ensure_daily_in_redis(start_date, end_date, pool_codes):
    """一次 SQL 批量加载全日期段日线数据, 按 code 分组写 Redis daily:{code}。
    幂等: 已有数据覆盖刷新。返回缓存的股票数。
    """
    try:
        from core.data import Database
        if not Database.is_available():
            return 0
        from core.shared.infrastructure.cache import CacheManager
        import polars as pl
        cm = CacheManager()
        logger.info(f"[Preload] 批量加载 daily_quotes {start_date} ~ {end_date}")
        rows = Database.fetchall(
            "SELECT code, trade_date, open, high, low, close, volume, amount, "
            "pct_chg, turnover FROM daily_quotes "
            "WHERE trade_date >= %s AND trade_date <= %s ORDER BY code, trade_date",
            [start_date, end_date])
        if not rows:
            return 0
        df_all = pl.from_dicts(rows, infer_schema_length=len(rows) + 1)
        target = set(pool_codes) if pool_codes else None
        grouped = df_all.group_by("code", maintain_order=True)
        count = 0
        for name, group in grouped:
            code = name[0] if isinstance(name, tuple) else name
            if target is not None and code not in target:
                continue
            pdf = group.to_pandas()
            ck = CacheManager.make_key("daily", "qfq", code)
            cm.set_redis(ck, pdf, expire=86400 * 7)
            count += 1
        logger.info(f"[Preload] Redis daily:{{code}} 写入 {count} 只股票")
        return count
    except Exception as e:
        logger.warning(f"[Preload] 批量加载失败: {e}")
        return 0


def _wait_futures(futs, timeout, total, done_counter, progress_cb):
    """等待所有 future 完成, 整体超时自动跳过, 防止 I/O 死锁卡死整个 master 构建。"""
    from concurrent.futures import wait as _fut_wait, FIRST_COMPLETED
    deadline = time.time() + timeout * 2
    pending = set(futs.keys())
    while pending and time.time() < deadline:
        completed, pending = _fut_wait(pending, timeout=min(10, max(1, deadline - time.time())),
                                        return_when=FIRST_COMPLETED)
        for f in completed:
            try:
                f.result(timeout=0)  # 已完成, 立即返回或抛异常
            except Exception:
                pass
    remaining = total - done_counter[0]
    if remaining > 0:
        logger.warning(f"[Master] {remaining}/{total} 只股票超时未完成, 标记为跳过")
        done_counter[0] = total
        # 确保进度上报不卡住
        if progress_cb:
            try:
                progress_cb({"stage": "master", "current": total, "total": total})
            except Exception:
                pass


def _build_master_indicators(pool_codes, progress_cb=None, max_workers=None):
    """从 Redis 读取全量日线, 每只股票 add_indicators 只算一次。

    返回 {code: pd.DataFrame (全量, 含全部技术指标列)}。
    内存估算: N_stocks × ~700 行 × 30 列 × 8B ≈ 2000 × 168KB ≈ 336MB。
    max_workers: None=用全局线程池, int=用本地 ThreadPoolExecutor(N)
    """
    from core.shared.infrastructure.cache import CacheManager
    from core.domain.screening.indicators import Analyzer
    import pandas as pd
    import numpy as np
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cm = CacheManager()
    analyzer = Analyzer()
    master = {}
    lock = threading.Lock()
    total = len(pool_codes) if pool_codes else 0
    done = [0]

    def _build_one(code):
        try:
            ck = CacheManager.make_key("daily", "qfq", code)
            # 缓存链: Redis → diskcache → MySQL
            df = cm.get_cached(ck)
            if df is None or isinstance(df, bool) or (hasattr(df, 'empty') and df.empty):
                try:
                    from core.data import DailyQuotesRepo
                    df = DailyQuotesRepo.get_range(code, "20100101", "20300101")
                except Exception:
                    pass
            if df is None or isinstance(df, bool) or (hasattr(df, 'empty') and df.empty):
                return
            if len(df) < 100:
                return

            num_cols = [c for c in ("open", "high", "low", "close", "volume", "amount")
                        if c in df.columns]
            if num_cols:
                df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

            # 确保 trade_date 是日期类型, 供后续切片使用
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

            df = analyzer.add_indicators(df)

            with lock:
                master[code] = df
        except Exception:
            pass
        finally:
            done[0] += 1
            if done[0] % 500 == 0:
                logger.info(f"[Master] 指标构建进度: {done[0]}/{total}")
                if progress_cb:
                    progress_cb({"stage": "master", "current": done[0], "total": total})

    STOCK_TIMEOUT = 30  # 单只股票最长 30s, 防止 Redis/MySQL I/O 死锁
    if max_workers:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_build_one, code): code for code in pool_codes}
            _wait_futures(futs, STOCK_TIMEOUT, len(pool_codes), done, progress_cb)
    else:
        with thread_pool.default.borrow() as ex:
            futs = {ex.submit(_build_one, code): code for code in pool_codes}
            _wait_futures(futs, STOCK_TIMEOUT, len(pool_codes), done, progress_cb)

    logger.info(f"[Master] 指标构建完成: {len(master)}/{total} 只")
    if progress_cb:
        try:
            progress_cb({"stage": "master", "current": len(master), "total": total})
        except Exception:
            pass
    return master


def _precompute_for_anchors(factor_variants, anchors, pool_codes, pool,
                            lookback_months, eval_weeks, all_dates_sorted,
                            code_prefixes, job_id, progress_cb):
    """主进程只构建 master → 写磁盘, worker 各自从磁盘加载 + 构建自己锚点的 snapshots。
    避免主进程重复调用 batch_snapshots_from_master 的巨额开销。
    返回 {cache_dir, master_path} (不预计算 snapshots/因子面板)。
    """
    import os
    import pickle as _pk

    cache_dir = os.path.join("data", "cache", "rolling_screen")
    if job_id:
        cache_dir = os.path.join(cache_dir, str(job_id))
    os.makedirs(cache_dir, exist_ok=True)

    from core.shared.anchor_cache import clear_mem_cache as _clear_mem_cache

    # 1. 构建 master (一次性, ~540MB)
    t0 = time.time()
    master = _build_master_indicators(pool_codes, max_workers=3)
    logger.info(f"[Precompute] master {len(master)}只, {time.time()-t0:.0f}s")

    # 2. master→磁盘
    master_path = os.path.join(cache_dir, "master.pkl")
    from core.shared.persistence import save_pickle
    save_pickle(master, master_path)

    master.clear()
    _clear_mem_cache()
    gc.collect()

    logger.info(f"[Precompute] master 写入 {master_path}")
    return {"cache_dir": cache_dir, "master_path": master_path}

    # 4. master→磁盘
    master_path = os.path.join(cache_dir, "master.pkl")
    from core.shared.persistence import save_pickle
    save_pickle(master, master_path)

    snapshots.clear()
    master.clear()
    _clear_mem_cache()
    gc.collect()

    logger.info(f"[Precompute] 完成: {write_i} snap + {n_expr} expr, 总耗时 {time.time()-t0:.0f}s")
    return {
        "snap_dir": snap_dir,
        "master_path": master_path,
        "expr_dir": expr_dir,
        "cache_dir": cache_dir,
    }


def _check_factors_batch(factor_variants, var_by_key, snapshots, pool, code_prefixes,
                         eval_weeks, mode, skip_layers, n_shuffles, anchor_date, akey,
                         job_id, preloaded_expr, progress_reporter, l1_dates=None,
                         label_horizon=20, max_eval_dates=8):
    """共享的因子批量校验逻辑 (旧worker和新worker共用)。
    preloaded_expr: 预计算的 _date_expr_scores dict, 注入 shared_checker 实现全缓存命中。
    progress_reporter(stage, current, total, **extra): 进度回调。
    返回 (passed, detail, directions)。
    """
    import threading as _thr
    from concurrent.futures import wait as _wait, FIRST_COMPLETED as _FC, ThreadPoolExecutor
    from core.domain.factors.validation import OverfittingChecker

    passed, detail = [], {}
    directions = {}
    detail_lock = _thr.Lock()
    checker_local = _thr.local()

    def _thread_checker():
        checker = getattr(checker_local, "checker", None)
        if checker is None:
            checker = OverfittingChecker()
            checker.set_expr_variants(factor_variants)
            if preloaded_expr:
                checker._date_expr_scores = preloaded_expr
            checker_local.checker = checker
        return checker

    def _check_one(fi, v):
        key = v.get("variant_key") or v.get("template_key") or "?"
        checker = None
        try:
            checker = _thread_checker()
            isolation, _signature, use_expr, _ast, _outer = checker._resolve_eval(v, quiet=True)
            if isolation is None and not use_expr:
                return key, {"error": "因子无法隔离或解析，未执行综合分回退"}, False
            r = checker.check_all_layers(
                v, eval_weeks=eval_weeks, pool=pool, code_prefixes=code_prefixes,
                snapshots=snapshots, mode=mode, anchor_date=anchor_date,
                skip_layers=skip_layers, n_shuffles=n_shuffles,
                l1_dates=l1_dates, label_horizon=label_horizon,
                max_eval_dates=max_eval_dates)
            d = {"all_pass": r["all_pass"], "passed_count": r["passed_count"],
                 "direction": r.get("direction", "long"),
                 "layers": [{"id": l["id"], "name": l["name"], "pass": l["pass"],
                             "detail": l["detail"]} for l in r["layers"]],
                 "daily_ics": r.get("daily_ics"),
                 "ic_decay": r.get("ic_decay"),
                 "walkforward_metrics": r.get("walkforward_metrics"),
                 "shuffle_stats": r.get("shuffle_stats"),
                 "max_corr": r.get("max_corr")}
            return key, d, r["all_pass"]
        except Exception as e:
            return key, {"error": str(e)}, False
        finally:
            if checker is not None:
                checker._screen_cache.clear()
                checker._neutral_cache.clear()
                checker._resid_cache.clear()
                checker._injected_scores = None

    CHK_TIMEOUT_PER = 1800
    BATCH_SIZE = 8
    n_factors = len(factor_variants)

    with ThreadPoolExecutor(max_workers=3) as chk_ex:
        for batch_start in range(0, n_factors, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_factors)
            batch_variants = factor_variants[batch_start:batch_end]

            chk_futs = {}
            for i, v in enumerate(batch_variants):
                fi = batch_start + i
                for _ in range(3):
                    try:
                        chk_futs[chk_ex.submit(_check_one, fi, v)] = (fi, time.time())
                        break
                    except RuntimeError:
                        time.sleep(0.1)
                else:
                    logger.warning(f"[Worker] 提交失败 {v.get('variant_key','?')}")

            pending = set(chk_futs.keys())
            batch_deadline = time.time() + CHK_TIMEOUT_PER + 60

            while pending and time.time() < batch_deadline:
                completed, pending = _wait(pending, timeout=15, return_when=_FC)
                for f in completed:
                    fi, _ = chk_futs[f]
                    v = factor_variants[fi]
                    key = v.get("variant_key") or v.get("template_key") or "?"
                    try:
                        _, d, passed_flag = f.result(timeout=5)
                        with detail_lock:
                            detail[key] = d
                            if passed_flag:
                                passed.append(key)
                                directions[key] = d.get("direction", "long")
                    except Exception as e:
                        logger.warning(f"[Worker] 校验失败 {key}: {e}")
                        with detail_lock:
                            detail[key] = {"error": str(e)}
                    with detail_lock:
                        progress_reporter("factor", len(detail), n_factors)

            if pending:
                timeout_keys = []
                for f in pending:
                    fi, _ = chk_futs[f]
                    timeout_keys.append(
                        factor_variants[fi].get("variant_key")
                        or factor_variants[fi].get("template_key")
                        or "?"
                    )
                    f.cancel()
                raise TimeoutError(
                    f"锚点 {akey} 因子校验超时，拒绝生成不完整证据: {timeout_keys}"
                )

    return passed, detail, directions


def _batch_deduplicate_passed(passed, detail, var_by_key, snapshots, pool,
                              code_prefixes, anchor_date, l1_dates, preloaded_expr):
    if len(passed) <= 1:
        return list(passed)
    from core.domain.factors.validation import L5_CORR_THRESHOLD, OverfittingChecker

    variants = [var_by_key[key] for key in passed if key in var_by_key]
    valid_dates = list((l1_dates or {}).get("valid") or [])
    train_dates = list((l1_dates or {}).get("train") or [])
    eval_date = (valid_dates or train_dates or [anchor_date])[-1]
    quality = {}
    for key in passed:
        wf = (detail.get(key) or {}).get("walkforward_metrics") or {}
        sign = int(wf.get("direction_sign") or 1)
        train_ic = wf.get("train_ic")
        valid_ic = wf.get("oriented_valid_ic")
        oriented_train = float(train_ic) * sign if train_ic is not None else 0.0
        quality[key] = min(oriented_train, float(valid_ic or 0.0))

    checker = OverfittingChecker()
    checker.set_expr_variants(variants)
    if preloaded_expr:
        checker._date_expr_scores = preloaded_expr
    dedup = checker.deduplicate_variants(
        variants, pool=pool, code_prefixes=code_prefixes, eval_date=eval_date,
        corr_thr=L5_CORR_THRESHOLD, snapshots=snapshots, selection_scores=quality,
        return_details=True,
    )
    kept = dedup["kept"]
    kept_set = set(kept)
    unresolved = set(dedup["unresolved"])
    for key in passed:
        item = detail.get(key) or {}
        layers = item.get("layers") or []
        l5 = next((layer for layer in layers if layer.get("id") == 5), None)
        if key in kept_set:
            if l5 is not None:
                l5.update({
                    "pass": True, "status": "passed",
                    "detail": f"锚点级批量去冗余通过，代表日={str(eval_date)[:10]}",
                })
            item["passed_count"] = int(item.get("passed_count") or 0) + 1
            item["all_pass"] = True
        else:
            if l5 is not None:
                l5.update({
                    "pass": False, "status": "failed",
                    "detail": (
                        "代表日截面不足，无法完成锚点级相关性检查"
                        if key in unresolved
                        else f"锚点级批量相关去冗余淘汰，阈值={L5_CORR_THRESHOLD:.2f}"
                    ),
                })
            item["all_pass"] = False
        detail[key] = item
    return [key for key in passed if key in kept_set]


def _save_anchor_results(job_id, anchor_date, var_by_key, passed, detail, directions):
    try:
        from core.data import Database
        if not Database.is_available():
            return
        sql = (
            "INSERT INTO rolling_screen_results "
            "(job_id, anchor_date, factor_key, label, passed, direction, passed_count, layer_detail, error) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        )
        rows = []
        for key, v in var_by_key.items():
            d = detail.get(key, {})
            rows.append((
                job_id, anchor_date, key,
                v.get("label", key),
                1 if key in passed else 0,
                directions.get(key, "long"),
                d.get("passed_count", 0),
                json.dumps(d, ensure_ascii=False) if d and "error" not in d else None,
                d.get("error") if "error" in d else None,
            ))
        if rows:
            Database.executemany(sql, rows)
        logger.info(f"[Rolling] anchor={anchor_date} 写入 {len(var_by_key)} 条结果")

        # 同时写入 factor_validation_scores (供跨锚点/跨市场状态 IC 聚合)
        _save_validation_scores(anchor_date, var_by_key, passed, detail, directions)

    except Exception as e:
        logger.warning(f"[Rolling] 写DB失败 anchor={anchor_date}: {e}")


def _save_validation_scores(anchor_date, var_by_key, passed, detail, directions):
    """从 check_all_layers 结果中提取 IC 指标写入 factor_validation_scores。"""
    try:
        from core.data import Database
        if not Database.is_available():
            return
        sql = (
            "REPLACE INTO factor_validation_scores "
            "(factor_key, anchor_date, icir, ic_mean, ic_std, n_ic_dates, daily_ics, "
            "ic_decay, shuffle_p_value, shuffle_p95, shuffle_n, max_corr, "
            "all_pass, direction, "
            "l1_score, l2_score, l3_score, l4_score, l5_score, l7_score, l8_score) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s, %s,%s,%s,%s,%s,%s,%s)"
        )
        anchor_str = anchor_date.isoformat() if hasattr(anchor_date, "isoformat") else str(anchor_date)
        rows = []
        for key, v in var_by_key.items():
            d = detail.get(key, {})
            daily_ics = d.get("daily_ics")
            layers = d.get("layers", [])
            ls = {ly["id"]: ly.get("score", 0) for ly in layers}

            ic_mean = None
            ic_std = None
            icir = None
            n_dates = len(daily_ics) if daily_ics else 0
            if daily_ics and n_dates > 0:
                ics = [v[1] for v in daily_ics if len(v) >= 2]
                if ics:
                    ic_mean = sum(ics) / len(ics)
                    ic_std = (sum((x - ic_mean) ** 2 for x in ics) / len(ics)) ** 0.5
                    icir = ic_mean / ic_std if ic_std > 0 else 0
            if ic_mean is None and n_dates == 0:
                continue

            ic_decay = d.get("ic_decay")
            ss = d.get("shuffle_stats") or {}

            rows.append((
                key, anchor_str,
                icir, ic_mean, ic_std,
                n_dates,
                json.dumps(daily_ics) if daily_ics else None,
                json.dumps(ic_decay) if ic_decay else None,
                ss.get("p_value"), ss.get("p95"), ss.get("n_shuffles"),
                d.get("max_corr"),
                1 if d.get("all_pass") else 0,
                directions.get(key, "long"),
                ls.get(1), ls.get(2), ls.get(3), ls.get(4), ls.get(5), ls.get(7), ls.get(8),
            ))
        if rows:
            try:
                Database.executemany(sql, rows)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[Rolling] 写 factor_validation_scores 失败 anchor={anchor_date}: {e}")


def _run_anchor_worker(args_dict):
    """独立子进程执行一个锚点的完整校验 (multiprocessing worker)。
    接收完全可序列化的 dict, 重新 import 所有模块。
    """
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _logger = _log.getLogger("rolling.worker")

    import threading as _thr, time as _time, gc as _gc, datetime as _dt
    from concurrent.futures import wait as _wait, FIRST_COMPLETED as _FC, ThreadPoolExecutor
    from core.domain.factors.validation import OverfittingChecker
    from core.domain.screening._screener import get_screener
    from core.shared.anchor_cache import clear_mem_cache as _clear_mem_cache

    factor_variants = args_dict["factor_variants"]
    anchor_date = _dt.date.fromisoformat(args_dict["anchor_date"])
    all_dates_sorted = [_dt.date.fromisoformat(d) for d in args_dict["all_dates_sorted"]]
    pool_codes = args_dict["pool_codes"]
    pool = args_dict["pool"]
    eval_weeks = args_dict.get("eval_weeks", 18)
    mode = args_dict.get("mode", "full")
    code_prefixes = args_dict.get("code_prefixes")
    skip_layers = args_dict.get("skip_layers")
    n_shuffles = args_dict.get("n_shuffles", 200)
    label_horizon = int(args_dict.get("label_horizon", 20))
    max_eval_dates = args_dict.get("max_eval_dates", 8)
    train_months = int(args_dict.get("train_months", 60))
    valid_months = int(args_dict.get("valid_months", 6))
    test_months = int(args_dict.get("test_months", 3))
    purge_gap = int(args_dict.get("purge_gap", 11))
    batch_dedup = bool(args_dict.get("batch_dedup", True))
    job_id = args_dict.get("job_id")
    shared_progress = args_dict.get("shared_progress")  # Manager().dict() proxy

    akey = anchor_date.isoformat()
    var_by_key = {(v.get("variant_key") or v.get("template_key")): v for v in factor_variants}

    _redis_progress_key = f"rolling:progress:{job_id}:{akey}"
    _report_redis = None
    try:
        import redis as _rds
        _report_redis = _rds.Redis(host='127.0.0.1', port=6379, db=0,
                                   socket_timeout=2, socket_connect_timeout=2)
    except Exception:
        pass

    _factor_current = [0]

    def _report(stage, current, total, **extra):
        nonlocal _report_redis
        if stage == "factor":
            _factor_current[0] = max(_factor_current[0], int(current or 0))
        ratio = float(current or 0) / max(float(total or 0), 1.0)
        if stage == "snapshot":
            anchor_fraction = 0.1 * ratio
        elif stage == "factor_panel":
            anchor_fraction = 0.1 + 0.1 * ratio
        elif stage == "factor":
            anchor_fraction = 0.2 + 0.75 * ratio
        elif stage in ("dedup", "cluster"):
            anchor_fraction = 0.95 + 0.04 * ratio
        elif stage == "done":
            anchor_fraction = 1.0
        else:
            anchor_fraction = 0.0
        payload = {
            "stage": stage, "current": current, "total": total,
            "factor_current": _factor_current[0], "anchor_fraction": anchor_fraction,
            "updated_at": _time.time(), **extra,
        }
        data = json.dumps(payload)
        if _report_redis is not None:
            try:
                _report_redis.setex(_redis_progress_key, 1200, data)
            except Exception:
                _report_redis = None
        # Manager dict 作为备用
        if shared_progress is not None:
            try:
                shared_progress[akey] = payload
            except Exception:
                pass

    # ── 加载 master (优先从磁盘, 节省 3/4 构建开销) ──
    master_path = args_dict.get("master_path")
    preloaded_expr = args_dict.get("_preloaded_expr")
    screener = get_screener()
    helper = OverfittingChecker()
    train_start = _add_months(anchor_date, -(train_months + valid_months))
    valid_start = _add_months(anchor_date, -valid_months)
    test_end = _add_months(anchor_date, test_months) - _dt.timedelta(days=1)
    calendar_dates = helper._dates_between(train_start, test_end, step=1)
    train_d = helper._dates_between(train_start, valid_start - _dt.timedelta(days=1), step=5)
    valid_d = helper._dates_between(valid_start, anchor_date - _dt.timedelta(days=1), step=5)
    test_d = helper._dates_between(anchor_date, test_end, step=5)
    from core.domain.factors.walkforward import evenly_sample_dates, purge_dates_before_boundary
    train_d = purge_dates_before_boundary(train_d, valid_start, calendar_dates, purge_gap)
    valid_d = purge_dates_before_boundary(valid_d, anchor_date, calendar_dates, purge_gap)
    train_d = evenly_sample_dates(train_d, max_eval_dates)
    valid_d = evenly_sample_dates(valid_d, max_eval_dates)
    test_d = evenly_sample_dates(test_d, max_eval_dates)
    future_dates = helper._dates_between(test_end, test_end + _dt.timedelta(days=60), step=1)
    matured_at = future_dates[label_horizon] if len(future_dates) > label_horizon else None
    l1_dates = {"train": train_d, "valid": valid_d, "test": test_d}
    window = {
        "train_start": train_d[0].isoformat() if train_d else train_start.isoformat(),
        "train_end": train_d[-1].isoformat() if train_d else None,
        "valid_start": valid_d[0].isoformat() if valid_d else valid_start.isoformat(),
        "valid_end": valid_d[-1].isoformat() if valid_d else None,
        "test_start": test_d[0].isoformat() if test_d else anchor_date.isoformat(),
        "test_end": test_d[-1].isoformat() if test_d else test_end.isoformat(),
        "label_matured_at": matured_at.isoformat() if matured_at else None,
        "label_horizon": label_horizon,
        "purge_gap": purge_gap,
    }
    all_snap_dates = sorted(set(train_d + valid_d + test_d))
    injected_snapshots = args_dict.get("_snapshots")
    master = None

    if injected_snapshots is not None:
        snapshots = injected_snapshots
        _report("snapshot", len(all_snap_dates), len(all_snap_dates),
                backing_mode=getattr(snapshots, "backing_mode", "injected"))
        try:
            from core.domain.screening._scoring_legacy import _prefetch_financials
            fin_cache = _prefetch_financials(pool_codes)
            screener.set_financial_cache(fin_cache)
        except Exception:
            pass
    elif master_path:
        # ── v2 路径: 从磁盘加载 master, 自行构建 snapshots ──
        from core.shared.persistence import load_pickle
        _logger.info(f"[WorkerV2] anchor={akey} 从磁盘加载 master...")
        _report("load_master", 0, 1)
        t0 = _time.time()
        master = load_pickle(master_path)
        _logger.info(f"[WorkerV2] anchor={akey} master {len(master)}只加载完成, {_time.time()-t0:.1f}s")
        _report("master", len(master), len(pool_codes))

        screener = get_screener()
        try:
            from core.domain.screening._scoring_legacy import _prefetch_financials
            fin_cache = _prefetch_financials(pool_codes)
            screener.set_financial_cache(fin_cache)
        except Exception:
            pass

        _report("snapshot", 0, len(all_snap_dates))
        t0 = _time.time()
        try:
            from core.domain.factors.panel import ResearchSnapshotStore
            snapshots = ResearchSnapshotStore(
                master,
                all_snap_dates,
                adj_latest=screener._get_adj_latest_map(list(master)),
                min_turnover=getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 1e7),
                skip_limit_up=True,
                fields="all",
                dtype=getattr(config, "FACTOR_PANEL_DTYPE", "float64"),
            )
        except Exception as _snap_exc:
            _logger.exception(f"[Worker] anchor={akey} 快照构建失败: {_snap_exc}")
            return {"akey": akey, "passed": [], "detail": {"error": f"快照构建失败: {_snap_exc}"},
                    "directions": {}, "error": str(_snap_exc)}
        _logger.info(f"[WorkerV2] anchor={akey} 快照 {len(snapshots)}/{len(all_snap_dates)}日期, "
                     f"{_time.time()-t0:.1f}s")
        _report("snapshot", len(all_snap_dates), len(all_snap_dates))

        master.clear()
        gc.collect()
    else:
        # ── 旧路径: 独立构建 master + 快照 ──
        _logger.info(f"[Worker] anchor={akey} 构建 master ({len(pool_codes)} 只)...")
        _report("master", 0, len(pool_codes))
        t0 = _time.time()
        try:
            master = _build_master_indicators(pool_codes, max_workers=3)
        except Exception as _master_exc:
            _logger.exception(f"[Worker] anchor={akey} master构建失败: {_master_exc}")
            return {"akey": akey, "passed": [], "detail": {"error": f"master构建失败: {_master_exc}"},
                    "directions": {}, "error": str(_master_exc)}
        _logger.info(f"[Worker] anchor={akey} master {len(master)} 只, {_time.time()-t0:.1f}s")
        _report("master", len(master), len(pool_codes))

        screener = get_screener()
        try:
            from core.domain.screening._scoring_legacy import _prefetch_financials
            fin_cache = _prefetch_financials(pool_codes)
            screener.set_financial_cache(fin_cache)
        except Exception:
            pass

        _report("snapshot", 0, len(all_snap_dates))
        t0 = _time.time()
        try:
            from core.domain.factors.panel import ResearchSnapshotStore
            snapshots = ResearchSnapshotStore(
                master,
                all_snap_dates,
                adj_latest=screener._get_adj_latest_map(list(master)),
                min_turnover=getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 1e7),
                skip_limit_up=True,
                fields="all",
                dtype=getattr(config, "FACTOR_PANEL_DTYPE", "float64"),
            )
        except Exception as _snap_exc:
            _logger.exception(f"[Worker] anchor={akey} 快照构建失败: {_snap_exc}")
            return {"akey": akey, "passed": [], "detail": {"error": f"快照构建失败: {_snap_exc}"},
                    "directions": {}, "error": str(_snap_exc)}
        _logger.info(f"[Worker] anchor={akey} 快照 {len(snapshots)}/{len(all_snap_dates)} 日期 "
                     f"(全窗270d), {_time.time()-t0:.1f}s")
        _report("snapshot", len(all_snap_dates), len(all_snap_dates))

    if batch_dedup and snapshots and preloaded_expr is None:
        panel_checker = OverfittingChecker()
        panel_checker.set_expr_variants(factor_variants)
        panel_dates = sorted(set(train_d + valid_d + test_d))
        _report("factor_panel", 0, len(panel_dates))
        for index, eval_date in enumerate(panel_dates, 1):
            date_key = eval_date.isoformat()
            panel_checker._ensure_date_expr_scores(
                date_key, eval_date, pool, code_prefixes, snapshots, screener,
            )
            _report("factor_panel", index, len(panel_dates))
        preloaded_expr = panel_checker._date_expr_scores
    elif batch_dedup and snapshots:
        _report("factor_panel", len(all_snap_dates), len(all_snap_dates),
                cache_stats=args_dict.get("expr_cache_stats") or {})

    # ── 因子校验 (新旧路径共用, v2 模式下全部缓存命中) ──
    n_factors = len(factor_variants)
    _report("factor", 0, n_factors)

    def _report_wrapper(stage, current, total, **extra):
        _report(stage, current, total, **extra)

    effective_skip_layers = list(skip_layers or [])
    if batch_dedup and 5 not in effective_skip_layers:
        effective_skip_layers.append(5)
    passed, detail, directions = _check_factors_batch(
        factor_variants, var_by_key, snapshots, pool, code_prefixes,
        eval_weeks, mode, effective_skip_layers, n_shuffles, anchor_date, akey,
        job_id, preloaded_expr, _report_wrapper, l1_dates=l1_dates,
        label_horizon=label_horizon, max_eval_dates=max_eval_dates)

    _logger.info(f"[Worker] anchor={akey} 通过 {len(passed)}/{n_factors}")

    if batch_dedup and passed and snapshots:
        _report("dedup", 0, len(passed), factor_current=n_factors)
        before = len(passed)
        passed = _batch_deduplicate_passed(
            passed, detail, var_by_key, snapshots, pool, code_prefixes,
            anchor_date, l1_dates, preloaded_expr,
        )
        _report("dedup", len(passed), before, factor_current=n_factors)
        _logger.info(f"[Worker] anchor={akey} 锚点级批量去重: {before} → {len(passed)}")

    # 兼容旧策略的后置聚类去重
    if not batch_dedup and len(passed) > 40 and snapshots:
        _report("cluster", 0, len(passed))
        try:
            before = len(passed)

            def _cluster_progress(info):
                stage = info.get("stage", "cluster")
                phase = info.get("phase", "")
                if phase == "eval_factor":
                    _report("cluster", info.get("factor_idx", 0), info.get("factor_total", len(passed)),
                            phase="eval", date=f"{info.get('date_idx',0)}/{info.get('date_total',0)}")
                elif phase == "corr_date":
                    _report("cluster", info.get("date_idx", 0), info.get("date_total", 0),
                            phase="corr", n_factors=info.get("n_factors", 0))
                elif phase == "eval":
                    _report("cluster", info.get("date_idx", 0), info.get("date_total", 0),
                            phase="eval")
                else:
                    _report("cluster", 0, len(passed), phase=phase)

            dedup_helper = OverfittingChecker()
            passed = dedup_helper.cluster_deduplicate(
                passed, var_by_key, snapshots, pool, code_prefixes, screener,
                progress_cb=_cluster_progress)
            _logger.info(f"[Worker] anchor={akey} 聚类去重: {before} → {len(passed)}")
        except Exception as e:
            _logger.warning(f"[Worker] 聚类去重失败: {e}")

    # 持久化
    if job_id:
        try:
            _save_anchor_results(job_id, anchor_date, var_by_key, passed, detail, directions)
        except Exception as e:
            _logger.warning(f"[Worker] 写DB失败: {e}")

    # 释放内存
    snapshots.clear()
    if master is not None:
        master.clear()
    _clear_mem_cache()
    _gc.collect()

    _report("done", len(passed), n_factors, status="completed", passed_count=len(passed))

    return {
        "akey": akey,
        "passed": passed,
        "detail": detail,
        "directions": directions,
        "window": window,
    }


def rolling_factor_screen(factor_variants, start, end, step_months=3, lookback_months=60,
                          pool="stable_history_500d", code_prefixes=None, custom_sectors=None, concept_sectors=None,
                          mode="full", eval_weeks=18,
                          skip_layers=None, progress_cb=None, write_pool=False, job_id=None,
                          parallel=False, min_periods_frac=0.5,
                          precomputed=None, worker_fn=None, pool_codes=None,
                          cleanup_precomputed=True, label_horizon=10,
                          max_eval_dates=8, train_months=60, valid_months=6,
                          test_months=3, purge_gap=11, batch_dedup=True,
                          cancel_check=None):
    """滚动筛选 (v2: 主进程预计算, 子进程轻量校验)。

    factor_variants: [variant dict] (需含 variant_key; 公开因子含 expr)
    start/end: datetime.date, 滚动区间
    write_pool: True 时用投票制写回动态因子池 (min_periods_frac 控制阈值)
    min_periods_frac: 投票阈值 — 因子至少出现在该比例的锚点中才激活 (0=最后一期覆盖)
    parallel: 已废弃 (始终使用 v2 预计算流程)。
    """
    import math, os, shutil
    from core.domain.factors.validation import OverfittingChecker
    from core.domain.screening._screener import get_screener
    from core.shared.anchor_cache import clear_mem_cache as _clear_mem_cache

    anchors = _gen_anchors(start, end, step_months)
    helper = OverfittingChecker()
    var_by_key = {(v.get("variant_key") or v.get("template_key")): v for v in factor_variants}

    # ── 一次性: 收集全锚点日期范围 → Redis预加载 ──
    all_dates_union = set()
    for anchor in anchors:
        data_start = _add_months(anchor, -max(train_months + valid_months, lookback_months))
        data_end = _add_months(anchor, test_months) + dt.timedelta(days=60)
        all_dates_union |= set(helper._dates_between(data_start, data_end, step=5))
        all_dates_union |= set(helper._get_eval_dates(weeks=eval_weeks, step=5, anchor_date=anchor))
    all_dates_sorted = sorted(all_dates_union)
    min_date = all_dates_sorted[0].isoformat() if all_dates_sorted else start.isoformat()
    max_date = all_dates_sorted[-1].isoformat() if all_dates_sorted else end.isoformat()

    if precomputed is None:
        from core.domain.screening._screener import _get_all_stock_codes, filter_pool_codes
        pool_codes = filter_pool_codes(_get_all_stock_codes(), pool, as_of=start)
        _ensure_daily_in_redis(min_date, max_date, pool_codes)
        precomp = _precompute_for_anchors(
            factor_variants, anchors, pool_codes, pool,
            lookback_months, eval_weeks, all_dates_sorted,
            code_prefixes, job_id, progress_cb)
    else:
        precomp = precomputed
        pool_codes = list(pool_codes or precomp.get("pool_codes") or [])
        if not pool_codes:
            raise RuntimeError("预计算面板缺少股票代码")
    cache_dir = precomp["cache_dir"]
    execute_worker = worker_fn or _run_anchor_worker

    try:
        # ── 分发锚点 ──
        if len(anchors) >= 2:
            from multiprocessing import Pool as _Pool, cpu_count, Manager

            mgr = Manager()
            shared_progress = mgr.dict()

            all_dates_iso = [d.isoformat() for d in all_dates_sorted]
            args_list = []
            for anchor in anchors:
                args_list.append({
                    "factor_variants": factor_variants,
                    "anchor_date": anchor.isoformat(),
                    "all_dates_sorted": all_dates_iso,
                    "pool_codes": pool_codes,
                    "pool": pool,
                    "eval_weeks": eval_weeks,
                    "mode": mode,
                    "code_prefixes": code_prefixes,
                    "skip_layers": skip_layers,
                    "n_shuffles": 200,
                    "label_horizon": label_horizon,
                    "max_eval_dates": max_eval_dates,
                    "train_months": train_months,
                    "valid_months": valid_months,
                    "test_months": test_months,
                    "purge_gap": purge_gap,
                    "batch_dedup": batch_dedup,
                    "job_id": job_id,
                    "shared_progress": shared_progress,
                    "master_path": precomp.get("master_path"),
                    "panel_manifest": precomp.get("panel_manifest"),
                    "expr_cache_dir": precomp.get("expr_cache_dir"),
                    "expr_overlay_dir": precomp.get("expr_overlay_dir"),
                })

            configured_workers = max(1, min(int(config.FACTOR_ROLLING_WORKERS), 6))
            if not precomp.get("expr_cache_dir"):
                configured_workers = min(configured_workers, 4)
            n_procs = min(configured_workers, cpu_count() or configured_workers, len(anchors))
            logger.info(f"[Rolling] 启动 {n_procs} 个子进程 ({len(anchors)} 锚点, v2预计算)")

            t_start = time.time()
            per_anchor = {}
            factor_timeline = {}
            directions = {}
            final_active = []

            import threading as _thr
            stop_poll = [False]

            def _poll_progress():
                import redis as _poll_rds
                try:
                    _poll_r = _poll_rds.Redis(host='127.0.0.1', port=6379, db=0,
                                              socket_timeout=2, socket_connect_timeout=2,
                                              decode_responses=True)
                except Exception:
                    _poll_r = None
                while not stop_poll[0]:
                    try:
                        anchors_detail = {}
                        if _poll_r:
                            try:
                                keys = _poll_r.keys(f"rolling:progress:{job_id}:*")
                                for k in (keys or []):
                                    ak = k.split(":")[-1]
                                    v = _poll_r.get(k)
                                    if v:
                                        try:
                                            d = json.loads(v)
                                            anchors_detail[ak] = d
                                        except Exception:
                                            pass
                            except Exception:
                                _poll_r = None
                        try:
                            for ak, v in shared_progress.items():
                                if not isinstance(v, dict):
                                    continue
                                current_value = anchors_detail.get(ak) or {}
                                if float(v.get("updated_at") or 0) >= float(current_value.get("updated_at") or 0):
                                    anchors_detail[ak] = dict(v)
                        except Exception:
                            pass
                        total_done = sum(
                            1 for value in anchors_detail.values() if value.get("stage") == "done"
                        )
                        factor_done = sum(
                            int(value.get("factor_current") or 0) for value in anchors_detail.values()
                        )
                        factor_total = len(anchors) * len(factor_variants)
                        if progress_cb:
                            progress_cb({
                                "stage": "parallel",
                                "current": total_done, "total": len(anchors),
                                "anchor_total": len(anchors),
                                "factor_current": factor_done,
                                "factor_total": factor_total,
                                "progress_fraction": sum(
                                    float(value.get("anchor_fraction") or 0.0)
                                    for value in anchors_detail.values()
                                ) / max(len(anchors), 1),
                                "anchors": anchors_detail,
                            })
                    except Exception:
                        pass
                    time.sleep(3)

            poll_thread = _thr.Thread(target=_poll_progress, daemon=True)
            poll_thread.start()

            results = []
            try:
                from core.shared.infrastructure.async_task import TaskCancelled
                from multiprocessing import TimeoutError as _PoolTimeout
                with _Pool(processes=n_procs) as proc_pool:
                    iterator = proc_pool.imap_unordered(execute_worker, args_list)
                    while len(results) < len(args_list):
                        if cancel_check and cancel_check():
                            proc_pool.terminate()
                            raise TaskCancelled(job_id or "rolling")
                        try:
                            results.append(iterator.next(timeout=3))
                        except _PoolTimeout:
                            continue
                        except StopIteration:
                            break
            except TaskCancelled:
                raise
            except Exception as e:
                logger.error(f"[Rolling] 并行异常 (worker可能崩溃): {e}")
                raise
            finally:
                stop_poll[0] = True
                poll_thread.join(timeout=5)
                try:
                    import redis as _cl_rds
                    _cl = _cl_rds.Redis(host='127.0.0.1', port=6379, db=0,
                                        socket_timeout=2, socket_connect_timeout=2)
                    keys = _cl.keys(f"rolling:progress:{job_id}:*")
                    if keys:
                        _cl.delete(*keys)
                except Exception:
                    pass
                mgr.shutdown()

            if len(results) != len(args_list):
                raise RuntimeError(
                    f"滚动锚点结果不完整: 预期 {len(args_list)}, 实际 {len(results)}"
                )

            for r in results:
                akey = r["akey"]
                passed = r["passed"]
                detail = r.get("detail", {})
                window = r.get("window", {})
                r_directions = r.get("directions", {})
                directions.update(r_directions)
                per_anchor[akey] = {"passed": passed, "detail": detail, "window": window}

        else:
            # 单锚点: 主进程直接调 worker (免 spawn 开销)
            logger.info(f"[Rolling] 单锚点模式 (v2预计算)")
            t_start = time.time()
            per_anchor = {}
            factor_timeline = {}
            directions = {}

            args = {
                "factor_variants": factor_variants,
                "anchor_date": anchors[0].isoformat(),
                "all_dates_sorted": [d.isoformat() for d in all_dates_sorted],
                "pool_codes": pool_codes,
                "pool": pool,
                "eval_weeks": eval_weeks,
                "mode": mode,
                "code_prefixes": code_prefixes,
                "skip_layers": skip_layers,
                "n_shuffles": 200,
                "label_horizon": label_horizon,
                "max_eval_dates": max_eval_dates,
                "train_months": train_months,
                "valid_months": valid_months,
                "test_months": test_months,
                "purge_gap": purge_gap,
                "batch_dedup": batch_dedup,
                "job_id": job_id,
                "master_path": precomp.get("master_path"),
                "panel_manifest": precomp.get("panel_manifest"),
                "expr_cache_dir": precomp.get("expr_cache_dir"),
                "expr_overlay_dir": precomp.get("expr_overlay_dir"),
            }

            if cancel_check and cancel_check():
                from core.shared.infrastructure.async_task import TaskCancelled
                raise TaskCancelled(job_id or "rolling")
            r = execute_worker(args)
            akey = r["akey"]
            per_anchor[akey] = {"passed": r["passed"], "detail": r.get("detail", {}),
                                "window": r.get("window", {})}
            directions.update(r.get("directions", {}))

        factor_timeline = {}
        for akey, result in per_anchor.items():
            window = result.get("window", {})
            matured_raw = window.get("label_matured_at")
            try:
                matured_at = dt.date.fromisoformat(str(matured_raw)[:10])
            except (TypeError, ValueError):
                logger.warning(f"[Rolling] 锚点 {akey} 缺少有效标签成熟日，不计入历史因子池")
                continue
            if matured_at > end:
                continue
            passed = result.get("passed", [])
            for key in passed:
                factor_timeline.setdefault(key, []).append(akey)
            if write_pool:
                try:
                    from core.domain.factors.registry import get_pool
                    factor_specs = {
                        key: {field: var_by_key[key][field] for field in
                              ("label", "template_key", "params", "expr", "data_safety", "source", "category")
                              if field in var_by_key[key]}
                        for key in passed if key in var_by_key
                    }
                    get_pool().record_history(
                        passed, matured_at, category="white_box",
                        source_anchor=dt.date.fromisoformat(akey),
                        factor_specs=factor_specs,
                    )
                except Exception as e:
                    logger.warning(f"[Rolling] 记录历史失败 {akey}: {e}")

        # ── 投票 + 因子池写回 (新旧路径共用) ──
        final_active = []
        if write_pool and min_periods_frac > 0 and factor_timeline:
            threshold = max(1, math.ceil(len(anchors) * min_periods_frac))
            voted = [k for k, periods in factor_timeline.items() if len(periods) >= threshold]
            logger.info(f"[Rolling] 投票制: {len(voted)} 因子 >= {threshold}/{len(anchors)} 锚点 → 激活")
            try:
                from core.domain.factors.registry import get_pool
                meta = {}
                for k in voted:
                    v = var_by_key.get(k, {})
                    meta[k] = {"label": v.get("label", k), "template_key": v.get("template_key"),
                               "params": v.get("params", {}), "expr": v.get("expr"),
                               "data_safety": v.get("data_safety") or "unknown",
                               "source": v.get("source"), "category": v.get("category"),
                               "metrics": {"direction": directions.get(k, "long")}}
                get_pool().sync_active_set(
                    voted, factor_meta=meta, category="white_box", anchor_date=end,
                )
            except Exception as e:
                logger.warning(f"[Rolling] 投票写回失败: {e}")
            final_active = voted
        elif not write_pool:
            final_active = list(factor_timeline.keys()) if factor_timeline else []
        else:
            final_active = list(factor_timeline.keys()) if factor_timeline else []
            if not final_active:
                logger.warning("[Rolling] write_pool=True 但 final_active 为空")

        logger.info(f"[Rolling] 完成: {len(anchors)} 锚点, "
                    f"耗时 {time.time()-t_start:.0f}s (v2预计算)")

        return {
            "anchors": [a.isoformat() for a in anchors],
            "per_anchor": per_anchor,
            "factor_timeline": factor_timeline,
            "final_active": final_active,
            "evidence_config": {
                "label_horizon": label_horizon,
                "max_eval_dates": max_eval_dates,
                "train_months": train_months,
                "valid_months": valid_months,
                "test_months": test_months,
                "purge_gap": purge_gap,
            },
        }

    finally:
        # 清理临时磁盘文件 (保留 cache 目录结构, 仅删本次 job 的文件)
        try:
            if cleanup_precomputed and cache_dir and os.path.exists(cache_dir):
                from core.shared.persistence import remove_dir
                remove_dir(cache_dir)
                logger.debug(f"[Rolling] 清理缓存: {cache_dir}")
        except Exception:
            pass
