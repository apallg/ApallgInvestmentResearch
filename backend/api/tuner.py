"""配置调优 API"""
import logging
import uuid
import threading
import anyio
from datetime import datetime, timedelta
from core.shared.infrastructure.thread_pool import thread_pool
from backend.service.qlib_access import run_with_qlib_read_access

from fastapi import APIRouter, HTTPException, Body

router = APIRouter()
logger = logging.getLogger(__name__)

_results: dict = {}
_history: list = []
_lock = threading.Lock()


def _run_sweep(job_id: str, configs: list, pool: str, top_n: int, code_prefixes):
    from core.domain.tuning._tuner import get_tuner
    tuner = get_tuner()
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "running", "stage": "sweep"}
    try:
        result = tuner.quick_sweep(configs, pool=pool, top_n=top_n, code_prefixes=code_prefixes)
        with _lock:
            _results[job_id] = {
                "job_id": job_id,
                "status": "done",
                "type": "sweep",
                "data": result,
            }
            _history.append({
                "job_id": job_id,
                "created_at": datetime.now().isoformat(),
                "type": "sweep",
                "config_count": len(configs),
                "completed": result.get("completed", 0),
                "best_label": result["best"]["label"] if result.get("best") else None,
                "best_score": result["best"]["score_mean"] if result.get("best") else None,
            })
            if len(_history) > 200:
                _history[:] = _history[-200:]
    except Exception as e:
        logger.error(f"快扫失败 {job_id}: {e}")
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}


def _run_validate(job_id: str, configs: list, eval_weeks: int, top_k: int, pool: str, code_prefixes):
    from core.domain.tuning._tuner import get_tuner
    tuner = get_tuner()
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "running", "stage": "validate"}
    try:
        result = tuner.historical_validate(configs, eval_weeks=eval_weeks, top_k=top_k, pool=pool,
                                           code_prefixes=code_prefixes)
        with _lock:
            _results[job_id] = {
                "job_id": job_id,
                "status": "done",
                "type": "validate",
                "data": result,
            }
            _history.append({
                "job_id": job_id,
                "created_at": datetime.now().isoformat(),
                "type": "validate",
                "config_count": len(configs),
                "eval_dates": result.get("total_dates", 0),
                "best_label": result["best"]["label"] if result.get("best") else None,
                "best_fitness": result["best"]["fitness"] if result.get("best") else None,
            })
            if len(_history) > 200:
                _history[:] = _history[-200:]
    except Exception as e:
        logger.error(f"历史验证失败 {job_id}: {e}")
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}


@router.post("/sweep")
async def start_sweep(
    configs: list = Body(...),
    pool: str = Body("all"),
    top_n: int = Body(50),
    code_prefixes: list = Body(None),
):
    if not configs:
        raise HTTPException(400, "configs 不能为空")
    if len(configs) > 40:
        raise HTTPException(400, "最多 40 组配置")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "pending"}
    thread_pool.default.submit(run_with_qlib_read_access, _run_sweep, job_id, configs, pool, top_n, code_prefixes)
    return {"data": {"job_id": job_id, "status": "pending"}}


@router.post("/validate")
async def start_validate(
    configs: list = Body(...),
    eval_weeks: int = Body(12),
    top_k: int = Body(10),
    pool: str = Body("all"),
    code_prefixes: list = Body(None),
):
    if not configs:
        raise HTTPException(400, "configs 不能为空")
    if len(configs) > 20:
        raise HTTPException(400, "最多 20 组配置")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "pending"}
    thread_pool.default.submit(run_with_qlib_read_access, _run_validate, job_id, configs, eval_weeks, top_k, pool, code_prefixes)
    return {"data": {"job_id": job_id, "status": "pending"}}


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    with _lock:
        result = _results.get(job_id)
    if not result:
        raise HTTPException(404, f"任务不存在: {job_id}")
    return {"data": result}


@router.get("/history")
async def get_history(days: int = 7):
    cutoff = datetime.now() - timedelta(days=days)
    with _lock:
        recent = [h for h in _history
                  if datetime.fromisoformat(h["created_at"]) >= cutoff]
    return {"data": recent}


def _run_auto(job_id: str, weights: dict, skips: dict, tech_presets: list,
               l0_thresholds: list, pool: str, top_n: int, eval_weeks: int, top_k: int,
               date_start, date_end, code_prefixes):
    from backend.service.tuning import TuningService
    progress = {"stage": "init", "current": 0, "total": 0, "message": "准备中..."}
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "running", "stage": "auto", "progress": progress}
    try:
        result = TuningService().auto_tune(
            weights=weights, skips=skips, tech_presets=tech_presets,
            l0_thresholds=l0_thresholds, pool=pool, top_n=top_n,
            eval_weeks=eval_weeks, top_k=top_k,
            date_start=date_start, date_end=date_end,
            code_prefixes=code_prefixes,
            progress=progress,
            job_id=job_id,
        )
        with _lock:
            _results[job_id] = {
                "job_id": job_id,
                "status": "done",
                "type": "auto",
                "data": result,
                "progress": {"stage": "done", "current": 1, "total": 1, "message": "调优完成"},
            }
            _history.append({
                "job_id": job_id,
                "created_at": datetime.now().isoformat(),
                "type": "auto",
                "config_count": result.get("sweep", {}).get("total", 0),
                "top3_selected": result.get("top3_selected"),
                "best_label": result["validate"]["best"]["label"] if result.get("validate", {}).get("best") else None,
                "best_fitness": result["validate"]["best"]["fitness"] if result.get("validate", {}).get("best") else None,
            })
            if len(_history) > 200:
                _history[:] = _history[-200:]
    except Exception as e:
        logger.error(f"自动调优失败 {job_id}: {e}")
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}


@router.post("/auto")
async def start_auto_tune(
    weights: dict = Body(...),
    skips: dict = Body(...),
    tech_presets: list = Body(["all"]),
    l0_thresholds: list = Body([0.30]),
    pool: str = Body("all"),
    top_n: int = Body(30),
    eval_weeks: int = Body(12),
    top_k: int = Body(10),
    date_start: str = Body(None),
    date_end: str = Body(None),
    code_prefixes: list = Body(None),
):
    if len(tech_presets) * len(l0_thresholds) > 30:
        raise HTTPException(400, "配置组合过多，请减少技术预设或L0阈值")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "pending"}
    thread_pool.default.submit(run_with_qlib_read_access, _run_auto,
        job_id, weights, skips, tech_presets, l0_thresholds, pool, top_n, eval_weeks, top_k,
        date_start, date_end, code_prefixes)
    return {"data": {"job_id": job_id, "status": "pending"}}


@router.post("/phase1-signal-test")
async def start_phase1(
    pool: str = Body("all"),
    eval_weeks: int = Body(12),
    date_start: str = Body(None),
    date_end: str = Body(None),
    top_k: int = Body(10),
    code_prefixes: list = Body(None),
    top_n: int = Body(None),
    qlib_model: list = Body(None),
    thresholds: dict = Body(None),
    skip_tech_subs: bool = Body(True),
    date_step: int = Body(5),
    skip_financial_prefetch: bool = Body(True),
    regime: str = Body(None),
    signal_filter: list = Body(None),  # 可选,指定要测的信号key列表; None=全部
):
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "pending"}
    thread_pool.default.submit(run_with_qlib_read_access, _run_phase1,
        job_id, pool, eval_weeks, date_start, date_end, top_k, code_prefixes, top_n,
        qlib_model, thresholds, skip_tech_subs, date_step, skip_financial_prefetch, regime, signal_filter,
    )
    return {"data": {"job_id": job_id, "status": "pending"}}


def _run_phase1(job_id, pool, eval_weeks, date_start, date_end, top_k, code_prefixes, top_n,
                qlib_model, thresholds, skip_tech_subs, date_step, skip_financial_prefetch=False,
                regime=None, signal_filter=None):
    from core.domain.tuning._tuner import get_tuner
    tuner = get_tuner()
    progress = {"stage": "phase1", "current": 0, "total": 0, "message": "准备检测..."}
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "running", "stage": "phase1", "progress": progress}
    try:
        result = tuner.signal_validity_test(
            pool=pool, eval_weeks=eval_weeks, date_start=date_start, date_end=date_end,
            top_k=top_k, code_prefixes=code_prefixes, top_n=top_n,
            qlib_model=qlib_model, thresholds=thresholds,
            skip_tech_subs=skip_tech_subs, date_step=date_step,
            skip_financial_prefetch=skip_financial_prefetch, progress=progress,
            regime=regime, signal_filter=signal_filter)
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "done", "type": "phase1", "data": result,
                                "progress": {"stage": "done", "current": 1, "total": 1, "message": "检测完成"}}
            _history.append({
                "job_id": job_id, "created_at": datetime.now().isoformat(), "type": "phase1",
                "valid_count": result.get("summary", {}).get("valid_count", 0),
                "invalid_count": result.get("summary", {}).get("invalid_count", 0),
            })
            if len(_history) > 200:
                _history[:] = _history[-200:]
    except Exception as e:
        logger.error(f"Phase1 失败 {job_id}: {e}")
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}


@router.post("/phase2-weight-opt")
async def start_phase2(
    active_signals: list = Body(...),
    pool: str = Body("all"),
    top_n: int = Body(30),
    eval_weeks: int = Body(12),
    date_start: str = Body(None),
    date_end: str = Body(None),
    top_k: int = Body(10),
    code_prefixes: list = Body(None),
    regime: str = Body(None),
):
    if not active_signals:
        raise HTTPException(400, "active_signals 不能为空")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "pending"}
    thread_pool.default.submit(run_with_qlib_read_access, _run_phase2,
        job_id, active_signals, pool, top_n, eval_weeks, date_start, date_end, top_k, code_prefixes, regime,
    )
    return {"data": {"job_id": job_id, "status": "pending"}}


def _run_phase2(job_id, active_signals, pool, top_n, eval_weeks, date_start, date_end, top_k, code_prefixes, regime=None):
    from backend.service.tuning import TuningService
    progress = {"stage": "init", "current": 0, "total": 0, "message": "准备优化..."}
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "running", "stage": "phase2", "progress": progress}
    try:
        result = TuningService().phase2_weight_optimization(
            active_signals=active_signals, pool=pool, top_n=top_n,
            eval_weeks=eval_weeks, date_start=date_start, date_end=date_end,
            top_k=top_k, code_prefixes=code_prefixes, progress=progress,
            regime=regime, job_id=job_id)
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "done", "type": "phase2", "data": result,
                                "progress": {"stage": "done", "current": 1, "total": 1, "message": "优化完成"}}
            _history.append({
                "job_id": job_id, "created_at": datetime.now().isoformat(), "type": "phase2",
                "active_signals": active_signals,
                "best_fitness": result.get("validate", {}).get("best", {}).get("fitness"),
            })
            if len(_history) > 200:
                _history[:] = _history[-200:]
    except Exception as e:
        logger.error(f"Phase2 失败 {job_id}: {e}")
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}


@router.get("/presets")
async def get_presets():
    from core.domain.tuning._tuner import get_presets
    result = await anyio.to_thread.run_sync(get_presets)
    return {"data": result}


@router.post("/regime-walk-forward")
async def start_regime_walk_forward(
    pool: str = Body("all", embed=True),
    eval_weeks: int = Body(52, embed=True),
    date_start: str = Body(None, embed=True),
    date_end: str = Body(None, embed=True),
    top_k: int = Body(10, embed=True),
    code_prefixes: list = Body(None, embed=True),
    date_step: int = Body(5, embed=True),
):
    """Walk-forward回测: regime-aware动态权重 vs 固定权重"""
    job_id = str(uuid.uuid4())[:8]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "running",
                           "type": "regime_walk_forward", "stage": "wf"}
    thread_pool.default.submit(run_with_qlib_read_access, _run_regime_wf,
        job_id, pool, eval_weeks, date_start, date_end, top_k, code_prefixes, date_step
    )
    return {"data": {"job_id": job_id, "status": "started"}}


def _run_regime_wf(job_id, pool, eval_weeks, date_start, date_end, top_k, code_prefixes, date_step):
    try:
        from core.domain.tuning._tuner import get_tuner
        tuner = get_tuner()
        with _lock:
            _results[job_id]["message"] = f"Regime Walk-Forward: {eval_weeks}周 {pool}"
        result = tuner.walk_forward_regime_test(
            pool=pool, eval_weeks=eval_weeks, date_start=date_start, date_end=date_end,
            top_k=top_k, code_prefixes=code_prefixes, date_step=date_step,
        )
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "done",
                              "type": "regime_walk_forward", "data": result}
        _history.append({
            "job_id": job_id, "type": "regime_walk_forward",
            "valid_dates": result.get("valid_dates", 0),
            "regime_wins": result.get("summary", {}).get("regime_wins", 0),
            "baseline_wins": result.get("summary", {}).get("baseline_wins", 0),
            "created_at": datetime.now().isoformat(),
        })
        if len(_history) > 200:
            _history[:] = _history[-200:]
    except Exception as e:
        logger.error(f"Regime WF失败 {job_id}: {e}")
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}


@router.post("/regime-replay")
async def start_regime_replay(
    eval_weeks: int = Body(52, embed=True),
    date_start: str = Body(None, embed=True),
    date_end: str = Body(None, embed=True),
    date_step: int = Body(5, embed=True),
    index_name: str = Body("沪深300", embed=True),
):
    """验证分类器质量: 各状态下后续指数实际涨跌"""
    job_id = str(uuid.uuid4())[:8]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "running",
                           "type": "regime_replay", "stage": "replay"}
    thread_pool.default.submit(run_with_qlib_read_access, _run_regime_replay,
        job_id, eval_weeks, date_start, date_end, date_step, index_name
    )
    return {"data": {"job_id": job_id, "status": "started"}}


def _run_regime_replay(job_id, eval_weeks, date_start, date_end, date_step, index_name):
    try:
        from core.domain.tuning._tuner import get_tuner
        tuner = get_tuner()
        progress = {"stage": "replay", "current": 0, "total": 0, "message": "获取指数数据..."}
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "running",
                              "type": "regime_replay", "progress": progress}
        result = tuner.regime_historical_replay(
            eval_weeks=eval_weeks, date_start=date_start, date_end=date_end,
            date_step=date_step, index_name=index_name, progress=progress,
        )
        with _lock:
            progress["message"] = "完成"
            _results[job_id] = {"job_id": job_id, "status": "done",
                              "type": "regime_replay", "data": result,
                              "progress": progress}
        _history.append({
            "job_id": job_id, "type": "regime_replay",
            "valid_dates": result.get("valid_dates", 0),
            "created_at": datetime.now().isoformat(),
        })
        if len(_history) > 200:
            _history[:] = _history[-200:]
    except Exception as e:
        logger.error(f"Regime Replay失败 {job_id}: {e}")
        with _lock:
            _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}


@router.post("/generate-regime-weights")
async def generate_regime_weights(regime_breakdown: dict = Body(...)):
    """根据 Phase 1 市态分组表自动生成并保存权重到 regime_weights.py"""
    def _do():
        from core.domain.tuning._tuner import get_tuner
        tuner = get_tuner()
        result = tuner.generate_regime_weights_from_breakdown(regime_breakdown)
        import importlib
        from core import regime_weights
        importlib.reload(regime_weights)
        return result
    result = await anyio.to_thread.run_sync(_do)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"data": result}


@router.post("/preload-cache")
async def preload_cache(
    pool: str = Body("all"),
    eval_weeks: int = Body(12),
    date_start: str = Body(None),
    date_end: str = Body(None),
    date_step: int = Body(5),
    code_prefixes: list = Body(None),
    max_workers: int = Body(4),
):
    """预热缓存：一次性把全部快照 + Qlib 预测加载进 Redis/磁盘。
    跑完后再跑 Phase 1/2，全部命中，零计算量。"""
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "pending"}

    def _run():
        from core.shared.infrastructure.redis_client import ensure_redis_running
        ensure_redis_running()
        from core.domain.tuning._tuner import get_tuner, _load_cached_snapshot, _save_cached_snapshot
        from core.domain.screening._screener import get_screener
        tuner = get_tuner()
        screener = get_screener()
        progress = {"stage": "preload", "current": 0, "total": 0, "message": "获取交易日列表..."}
        try:
            with _lock:
                _results[job_id] = {"job_id": job_id, "status": "running", "progress": progress}

            # 1. 获取交易日
            eval_dates = tuner._get_eval_dates(
                weeks=eval_weeks, date_start=date_start, date_end=date_end, step=date_step)
            n_dates = len(eval_dates)

            # 2. 串行预计算快照：一次只保留一个在内存
            ok = 0
            for i, d in enumerate(eval_dates):
                # 先查缓存
                snap = _load_cached_snapshot(d, pool)
                if snap is not None:
                    ok += 1
                    progress.update({"stage": "snapshots", "current": i+1, "total": n_dates,
                                   "message": f"命中缓存 ({i+1}/{n_dates}) {d}"})
                    continue

                # 计算新快照
                progress.update({"stage": "snapshots", "current": i+1, "total": n_dates,
                               "message": f"计算快照 ({i+1}/{n_dates}) {d}"})
                try:
                    snap = screener.precompute_snapshot(
                        pool=pool, code_prefixes=code_prefixes, skip_tech=False,
                        eval_date=d, fast=True)
                    if snap:
                        _save_cached_snapshot(d, pool, snap)
                        ok += 1
                except Exception as e:
                    logger.error(f"[预热] {d} 快照失败: {e}")
                # 立即释放，控制内存 ~1 个快照
                del snap

            # 3. 批量加载 Qlib 预测
            progress.update({"stage": "qlib", "current": n_dates, "total": n_dates,
                           "message": "加载 Qlib 预测..."})
            best_qlib = screener._find_best_qlib_model()
            if best_qlib:
                bulk = screener._load_qlib_predictions_bulk(best_qlib, eval_dates, pool=pool)
                if bulk:
                    screener.set_qlib_predictions_by_date(bulk)

            progress.update({"stage": "done", "current": n_dates, "total": n_dates,
                           "message": f"预热完成: {ok}/{n_dates} 快照 + Qlib 预测已缓存"})
            with _lock:
                _results[job_id] = {"job_id": job_id, "status": "done",
                                   "data": {"dates": n_dates, "snapshots_ok": ok,
                                           "qlib": bool(best_qlib and bulk)},
                                   "progress": progress}
        except Exception as e:
            logger.error(f"预热缓存失败: {e}")
            with _lock:
                _results[job_id] = {"job_id": job_id, "status": "error", "error": str(e)}

    thread_pool.default.submit(run_with_qlib_read_access, _run)
    return {"data": {"job_id": job_id, "status": "started", "message": f"开始预热 {eval_weeks} 周缓存"}}


# ═══ 因子池管理 ═══

@router.get("/factor-pool")
async def get_factor_pool():
    """获取因子池完整状态"""
    def _do():
        from core.domain.factors.registry import get_pool
        pool = get_pool()
        return {
            "factors": pool.get_all_factors(),
            "monitoring": pool.state.get("monitoring", {}),
            "updated_at": pool.state.get("updated_at", ""),
        }
    result = await anyio.to_thread.run_sync(_do)
    return {"data": result}


@router.post("/factor-pool/add")
async def factor_pool_add(signals: list = Body(...)):
    """从 Phase 1 结果批量入库"""
    def _do():
        from core.domain.factors.registry import get_pool
        return get_pool().batch_add_from_phase1(signals)
    count = await anyio.to_thread.run_sync(_do)
    return {"data": {"added": count}}


@router.post("/factor-pool/toggle")
async def factor_pool_toggle(key: str = Body(...), active: bool = Body(...)):
    """切换因子活跃状态"""
    def _do():
        from core.domain.factors.registry import get_pool
        return get_pool().toggle_factor(key, active)
    ok = await anyio.to_thread.run_sync(_do)
    return {"data": {"key": key, "active": active, "ok": ok}}


@router.post("/factor-pool/remove")
async def factor_pool_remove(key: str = Body(...)):
    """手动移除因子 (软删除,标记为停用)"""
    def _do():
        from core.domain.factors.registry import get_pool
        return get_pool().remove_factor(key)
    ok = await anyio.to_thread.run_sync(_do)
    return {"data": {"key": key, "ok": ok}}


@router.delete("/factor-pool/{key}")
async def factor_pool_delete(key: str):
    """硬删除因子 (从状态文件中彻底移除)"""
    def _do():
        from core.domain.factors.registry import get_pool
        return get_pool().delete_factor(key)
    ok = await anyio.to_thread.run_sync(_do)
    return {"data": {"key": key, "ok": ok}}
