import datetime as dt
import json
import logging
import os
import time
import uuid

from backend.service.factor_panel import FactorPanelService
from core.domain.factors.rolling import _add_months, _gen_anchors, _run_anchor_worker, rolling_factor_screen
from core.domain.factors.walkforward import evenly_sample_dates
from core.domain.factors.walkforward import purge_dates_before_boundary
from core.domain.factors.validation import OverfittingChecker, PackedExpressionScores
from core.shared.infrastructure.async_task import TaskCancelled


logger = logging.getLogger(__name__)
_expr_worker_checker = None
_expr_worker_screener = None
_expr_worker_signature = None


def _anchor_expression_dates(anchor, train_months, valid_months, test_months,
                             purge_gap, max_eval_dates):
    helper = OverfittingChecker()
    train_start = _add_months(anchor, -(train_months + valid_months))
    valid_start = _add_months(anchor, -valid_months)
    test_end = _add_months(anchor, test_months) - dt.timedelta(days=1)
    train_dates = helper._dates_between(
        train_start, valid_start - dt.timedelta(days=1), step=5,
    )
    valid_dates = helper._dates_between(
        valid_start, anchor - dt.timedelta(days=1), step=5,
    )
    test_dates = helper._dates_between(anchor, test_end, step=5)
    calendar_dates = helper._dates_between(train_start, test_end, step=1)
    train_dates = purge_dates_before_boundary(
        train_dates, valid_start, calendar_dates, purge_gap,
    )
    valid_dates = purge_dates_before_boundary(
        valid_dates, anchor, calendar_dates, purge_gap,
    )
    train_dates = evenly_sample_dates(train_dates, max_eval_dates)
    valid_dates = evenly_sample_dates(valid_dates, max_eval_dates)
    test_dates = evenly_sample_dates(test_dates, max_eval_dates)
    return sorted(set(train_dates + valid_dates + test_dates))


def _load_or_compute_expr_scores(cache_dir, date_key, compute):
    from core.shared.persistence import load_pickle, save_pickle

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{date_key}.pkl")
    lock_path = f"{cache_path}.lock"
    if os.path.exists(cache_path):
        return PackedExpressionScores.from_mapping(load_pickle(cache_path)), True
    lock_fd = None
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            if os.path.exists(cache_path):
                return PackedExpressionScores.from_mapping(load_pickle(cache_path)), True
            try:
                if time.time() - os.path.getmtime(lock_path) > 1800:
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.25)
    try:
        if os.path.exists(cache_path):
            return PackedExpressionScores.from_mapping(load_pickle(cache_path)), True
        scores = PackedExpressionScores.from_mapping(compute())
        temp_path = f"{cache_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        save_pickle(scores, temp_path)
        os.replace(temp_path, cache_path)
        return scores, False
    finally:
        os.close(lock_fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _merge_expr_overlay(scores, overlay_dir, date_key):
    if not overlay_dir:
        return scores
    from core.shared.persistence import load_pickle

    overlay_path = os.path.join(overlay_dir, f"{date_key}.pkl")
    if not os.path.exists(overlay_path):
        return scores
    overlay = PackedExpressionScores.from_mapping(load_pickle(overlay_path))
    for signature in overlay:
        scores[signature] = overlay[signature]
    return scores


def run_expression_date_worker(args):
    global _expr_worker_checker, _expr_worker_screener, _expr_worker_signature
    eval_date = dt.date.fromisoformat(args["eval_date"])
    service = FactorPanelService()
    store = arrays = None
    try:
        store, arrays, _manifest = service.open_shared_memmap(
            args["panel_manifest"], [eval_date],
        )
        worker_signature = json.dumps(
            args.get("factor_variants") or [], sort_keys=True, default=str,
        )
        if _expr_worker_checker is None or _expr_worker_signature != worker_signature:
            _expr_worker_checker = OverfittingChecker()
            _expr_worker_checker.set_expr_variants(args.get("factor_variants") or [])
            _expr_worker_signature = worker_signature
        checker = _expr_worker_checker

        def compute_scores():
            global _expr_worker_screener
            if _expr_worker_screener is None:
                from core.domain.screening._screener import get_screener
                _expr_worker_screener = get_screener()
            date_key = eval_date.isoformat()
            checker._ensure_date_expr_scores(
                date_key, eval_date, args.get("pool", "a500"),
                args.get("code_prefixes"), store, _expr_worker_screener,
            )
            return checker._date_expr_scores.pop(date_key)

        scores, cache_hit = _load_or_compute_expr_scores(
            args["expr_cache_dir"], eval_date.isoformat(), compute_scores,
        )
        scores = _merge_expr_overlay(
            scores, args.get("expr_overlay_dir"), eval_date.isoformat(),
        )
        signature_variants = {}
        for variant in args.get("factor_variants") or []:
            _iso, signature, use_expr, _ast, _outer = checker._resolve_eval(variant, quiet=True)
            if use_expr:
                signature_variants.setdefault(signature, variant)
        missing_signatures = [
            signature for signature in signature_variants if signature not in scores
        ]
        if missing_signatures and args.get("expr_overlay_dir"):
            global _expr_worker_screener
            if _expr_worker_screener is None:
                from core.domain.screening._screener import get_screener
                _expr_worker_screener = get_screener()
            missing_checker = OverfittingChecker()
            missing_checker.set_expr_variants([
                signature_variants[signature] for signature in missing_signatures
            ])
            date_key = eval_date.isoformat()
            missing_checker._ensure_date_expr_scores(
                date_key, eval_date, args.get("pool", "a500"),
                args.get("code_prefixes"), store, _expr_worker_screener,
            )
            additions = PackedExpressionScores.from_mapping(
                missing_checker._date_expr_scores.pop(date_key, {}),
            )
            os.makedirs(args["expr_overlay_dir"], exist_ok=True)
            from core.shared.persistence import save_pickle
            overlay_path = os.path.join(args["expr_overlay_dir"], f"{date_key}.pkl")
            temp_path = f"{overlay_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            overlay = _merge_expr_overlay(
                PackedExpressionScores.from_mapping({}), args["expr_overlay_dir"], date_key,
            )
            for signature in additions:
                overlay[signature] = additions[signature]
            save_pickle(overlay, temp_path)
            os.replace(temp_path, overlay_path)
            for signature in additions:
                scores[signature] = additions[signature]
        missing_factor_keys = [
            (
                signature_variants[signature].get("variant_key")
                or signature_variants[signature].get("template_key")
            )
            for signature in signature_variants
            if signature not in scores
        ]
        return {
            "eval_date": eval_date.isoformat(),
            "cache_hit": cache_hit,
            "packed_bytes": scores.memory_bytes,
            "factor_count": len(scores),
            "expected_factor_count": len(signature_variants),
            "missing_factor_keys": missing_factor_keys,
        }
    finally:
        if hasattr(store, "clear"):
            store.clear()
        if arrays is not None:
            service.close_shared_memmap(arrays)


def run_memmap_anchor_worker(args):
    anchor = dt.date.fromisoformat(args["anchor_date"])
    train_months = int(args.get("train_months", 60))
    valid_months = int(args.get("valid_months", 6))
    test_months = int(args.get("test_months", 3))
    purge_gap = int(args.get("purge_gap", 11))
    max_eval_dates = args.get("max_eval_dates", 8)
    dates = _anchor_expression_dates(
        anchor, train_months, valid_months, test_months,
        purge_gap, max_eval_dates,
    )
    service = FactorPanelService()
    store = arrays = None
    try:
        akey = anchor.isoformat()
        shared_progress = args.get("shared_progress")
        redis_progress = None
        try:
            import redis
            redis_progress = redis.Redis(
                host="127.0.0.1", port=6379, db=0, socket_timeout=2,
            )
        except Exception:
            pass

        def snapshot_progress(current, total, _date):
            ratio = float(current or 0) / max(float(total or 0), 1.0)
            payload = {
                "stage": "snapshot", "current": current, "total": total,
                "factor_current": 0, "anchor_fraction": 0.1 * ratio,
                "updated_at": time.time(),
            }
            if redis_progress is not None:
                try:
                    redis_progress.setex(
                        f"rolling:progress:{args.get('job_id')}:{akey}",
                        1200,
                        json.dumps(payload),
                    )
                except Exception:
                    pass
            if shared_progress is not None:
                try:
                    shared_progress[akey] = payload
                except Exception:
                    pass

        store, arrays, _manifest = service.open_shared_memmap(
            args["panel_manifest"], dates, progress_cb=snapshot_progress,
        )
        worker_args = dict(args)
        worker_args["_snapshots"] = store
        worker_args["master_path"] = None
        expr_cache_dir = args.get("expr_cache_dir")
        factor_variants = args.get("factor_variants") or []
        if expr_cache_dir and factor_variants and args.get("batch_dedup", True):
            checker = OverfittingChecker()
            checker.set_expr_variants(factor_variants)
            screener = None
            preloaded_expr = {}
            cache_hits = 0
            for index, eval_date in enumerate(dates, 1):
                date_key = eval_date.isoformat()

                def compute_scores():
                    nonlocal screener
                    if screener is None:
                        from core.domain.screening._screener import get_screener
                        screener = get_screener()
                    checker._ensure_date_expr_scores(
                        date_key, eval_date, args.get("pool", "a500"),
                        args.get("code_prefixes"), store, screener,
                    )
                    return checker._date_expr_scores.pop(date_key)

                scores, cache_hit = _load_or_compute_expr_scores(
                    expr_cache_dir, date_key, compute_scores,
                )
                scores = _merge_expr_overlay(
                    scores, args.get("expr_overlay_dir"), date_key,
                )
                preloaded_expr[date_key] = scores
                cache_hits += int(cache_hit)
                snapshot_progress(index, len(dates), date_key)
            worker_args["_preloaded_expr"] = preloaded_expr
            worker_args["expr_cache_stats"] = {
                "hits": cache_hits, "misses": len(dates) - cache_hits,
            }
        return _run_anchor_worker(worker_args)
    finally:
        if hasattr(store, "clear"):
            store.clear()
        if arrays is not None:
            service.close_shared_memmap(arrays)


class FactorRollingService:
    @staticmethod
    def resolve_variants(factor_source, source, factor_keys):
        if factor_source == "pool":
            from core.domain.factors.registry import get_pool

            active = get_pool().get_active_factors()
            checker = OverfittingChecker()
            keys = factor_keys or list(active)
            variants = [
                checker._pool_entry_to_variant(key, active[key])
                for key in keys if key in active
            ]
        else:
            from core.domain.factors.adapter import PublicFactorAdapter

            adapter = PublicFactorAdapter()
            keys = factor_keys or list(adapter.get_all(source=source))
            variants = [adapter.to_variant(key) for key in keys]
        return [variant for variant in variants if variant]

    def run(self, variants, start, end, step_months=3, lookback_months=60,
            pool="stable_history_500d", code_prefixes=None, mode="full", eval_weeks=18,
            skip_layers=None, progress_cb=None, write_pool=False, job_id=None,
            min_periods_frac=0.5, label_horizon=10, max_eval_dates=8,
            train_months=60, valid_months=6, test_months=3, purge_gap=11,
            batch_dedup=True, cancel_check=None, include_tech=True,
            reuse_cache_dir=None, reuse_expr_overlay_dir=None):
        anchors = _gen_anchors(start, end, step_months)
        if not anchors:
            return {"anchors": [], "per_anchor": {}, "factor_timeline": {}, "final_active": []}
        helper = OverfittingChecker()
        all_dates = set()
        for anchor in anchors:
            data_start = _add_months(anchor, -max(train_months + valid_months, lookback_months))
            data_end = _add_months(anchor, test_months) + dt.timedelta(days=60)
            all_dates |= set(helper._dates_between(
                data_start,
                data_end,
                step=5,
            ))
            all_dates |= set(helper._get_eval_dates(
                weeks=eval_weeks, step=5, anchor_date=anchor,
            ))
        dates = sorted(all_dates)
        cache_id = str(job_id or uuid.uuid4().hex[:12])
        cache_dir = (
            os.path.abspath(reuse_cache_dir)
            if reuse_cache_dir
            else os.path.join("data", "cache", "rolling_screen", cache_id)
        )
        expr_cache_dir = os.path.join(cache_dir, "expr_scores")
        expr_overlay_dir = (
            os.path.abspath(reuse_expr_overlay_dir)
            if reuse_expr_overlay_dir
            else os.path.join("data", "cache", "rolling_screen", f"{cache_id}_repair_overlay")
            if reuse_cache_dir else None
        )
        panel_service = FactorPanelService()
        try:
            if cancel_check and cancel_check():
                raise TaskCancelled(job_id or "rolling")
            if progress_cb:
                progress_cb({
                    "stage": "shared_panel", "current": 0, "total": len(dates),
                    "backing_mode": "memmap", "progress_fraction": 0.0,
                    "message": "正在构建共享映射面板",
                })
            try:
                def shared_panel_progress(current, total, phase):
                    if cancel_check and cancel_check():
                        raise TaskCancelled(job_id or "rolling")
                    if progress_cb:
                        ratio = float(current or 0) / max(float(total or 0), 1.0)
                        progress_cb({
                            "stage": "shared_panel", "current": current, "total": total,
                            "phase": phase, "backing_mode": "memmap",
                            "progress_fraction": 0.1 * ratio,
                            "message": "正在构建共享映射面板",
                        })

                if reuse_cache_dir:
                    from core.shared.persistence import load_json

                    manifest_path = os.path.join(cache_dir, "manifest.json")
                    if not os.path.exists(manifest_path):
                        raise RuntimeError(f"复用缓存缺少 manifest.json: {cache_dir}")
                    manifest = load_json(manifest_path)
                    pool_codes = list(manifest.get("codes") or [])
                    if not pool_codes:
                        raise RuntimeError("复用缓存没有股票代码")
                else:
                    manifest_path, manifest, pool_codes = panel_service.build_shared_memmap(
                        pool,
                        dates,
                        cache_dir,
                        code_prefixes=code_prefixes,
                        include_tech=include_tech,
                        progress_cb=shared_panel_progress,
                    )
                if cancel_check and cancel_check():
                    raise TaskCancelled(job_id or "rolling")
            except TaskCancelled:
                raise
            except Exception:
                logger.warning("滚动筛选共享面板构建失败，回退兼容路径", exc_info=True)
                return rolling_factor_screen(
                    variants,
                    start,
                    end,
                    step_months=step_months,
                    lookback_months=lookback_months,
                    pool=pool,
                    code_prefixes=code_prefixes,
                    mode=mode,
                    eval_weeks=eval_weeks,
                    skip_layers=skip_layers,
                    progress_cb=progress_cb,
                    write_pool=write_pool,
                    job_id=job_id,
                    min_periods_frac=min_periods_frac,
                    label_horizon=label_horizon,
                    max_eval_dates=max_eval_dates,
                    train_months=train_months,
                    valid_months=valid_months,
                    test_months=test_months,
                    purge_gap=purge_gap,
                    batch_dedup=batch_dedup,
                    cancel_check=cancel_check,
                )
            if progress_cb:
                progress_cb({
                    "stage": "shared_panel", "current": len(dates), "total": len(dates),
                    "backing_mode": "memmap",
                    "mapped_mb": round(manifest["mapped_bytes"] / (1024 * 1024), 1),
                    "progress_fraction": 0.1,
                    "message": "共享映射面板已完成",
                })
            expression_checker = OverfittingChecker()
            expression_checker.set_expr_variants(variants)
            effective_expr_cache_dir = (
                expr_cache_dir
                if batch_dedup and expression_checker._expr_variants
                else None
            )
            if effective_expr_cache_dir:
                expression_dates = sorted({
                    value
                    for anchor in anchors
                    for value in _anchor_expression_dates(
                        anchor, train_months, valid_months, test_months,
                        purge_gap, max_eval_dates,
                    )
                })
                self._precompute_expression_scores(
                    variants, expression_dates, manifest_path, effective_expr_cache_dir,
                    pool, code_prefixes, progress_cb, cancel_check,
                    expr_overlay_dir=expr_overlay_dir,
                )
            def rolling_progress(payload):
                if not progress_cb:
                    return
                value = dict(payload or {})
                local_fraction = value.get("progress_fraction")
                if local_fraction is not None:
                    value["progress_fraction"] = 0.45 + 0.55 * float(local_fraction)
                value.setdefault("message", "正在生成滚动因子证据")
                progress_cb(value)

            return rolling_factor_screen(
                variants,
                start,
                end,
                step_months=step_months,
                lookback_months=lookback_months,
                pool=pool,
                code_prefixes=code_prefixes,
                mode=mode,
                eval_weeks=eval_weeks,
                skip_layers=skip_layers,
                progress_cb=rolling_progress,
                write_pool=write_pool,
                job_id=job_id,
                min_periods_frac=min_periods_frac,
                precomputed={
                    "cache_dir": cache_dir,
                    "panel_manifest": manifest_path,
                    "pool_codes": pool_codes,
                    "expr_cache_dir": effective_expr_cache_dir,
                    "expr_overlay_dir": expr_overlay_dir,
                },
                worker_fn=run_memmap_anchor_worker,
                pool_codes=pool_codes,
                cleanup_precomputed=False,
                label_horizon=label_horizon,
                max_eval_dates=max_eval_dates,
                train_months=train_months,
                valid_months=valid_months,
                test_months=test_months,
                purge_gap=purge_gap,
                batch_dedup=batch_dedup,
                cancel_check=cancel_check,
            )
        finally:
            if not reuse_cache_dir:
                panel_service.cleanup_shared_memmap(cache_dir)

    @staticmethod
    def _precompute_expression_scores(variants, dates, manifest_path, expr_cache_dir,
                                      pool, code_prefixes, progress_cb, cancel_check,
                                      expr_overlay_dir=None):
        from multiprocessing import Pool, TimeoutError as PoolTimeout, cpu_count
        from core.shared import config

        if not dates:
            return
        worker_count = min(
            max(1, int(config.FACTOR_EXPR_PRECOMPUTE_WORKERS)),
            cpu_count() or 1,
            len(dates),
        )
        args = [{
            "eval_date": value.isoformat(),
            "panel_manifest": manifest_path,
            "expr_cache_dir": expr_cache_dir,
            "factor_variants": variants,
            "pool": pool,
            "code_prefixes": code_prefixes,
            "expr_overlay_dir": expr_overlay_dir,
        } for value in dates]
        if progress_cb:
            progress_cb({
                "stage": "expression_panel", "current": 0, "total": len(args),
                "workers": worker_count,
                "progress_fraction": 0.1,
                "message": "正在预计算表达式面板",
            })
        completed = 0
        cache_hits = 0
        packed_bytes = 0
        missing_by_date = {}
        with Pool(processes=worker_count) as process_pool:
            iterator = process_pool.imap_unordered(run_expression_date_worker, args)
            while completed < len(args):
                if cancel_check and cancel_check():
                    process_pool.terminate()
                    raise TaskCancelled("expression_panel")
                try:
                    result = iterator.next(timeout=1)
                except PoolTimeout:
                    continue
                completed += 1
                cache_hits += int(result.get("cache_hit", False))
                packed_bytes += int(result.get("packed_bytes") or 0)
                missing = list(result.get("missing_factor_keys") or [])
                if missing:
                    missing_by_date[result.get("eval_date") or "unknown"] = missing
                if progress_cb:
                    ratio = completed / max(len(args), 1)
                    progress_cb({
                        "stage": "expression_panel",
                        "current": completed,
                        "total": len(args),
                        "workers": worker_count,
                        "cache_hits": cache_hits,
                        "packed_mb": round(packed_bytes / (1024 * 1024), 1),
                        "progress_fraction": 0.1 + 0.35 * ratio,
                        "message": "正在预计算表达式面板",
                    })
        if missing_by_date:
            first_date, first_missing = sorted(missing_by_date.items())[0]
            raise RuntimeError(
                f"表达式预计算不完整: {len(missing_by_date)} 个日期缺少因子; "
                f"示例 {first_date} 缺少 {len(first_missing)} 个: {first_missing[:20]}"
            )
