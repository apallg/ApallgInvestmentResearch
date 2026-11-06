"""
因子管理 API — 合并 factor_editor.py + whitebox_factors.py
统一前缀: /api/factor
"""
import logging
import threading
import uuid
from datetime import date
from typing import Optional

import anyio
from fastapi import APIRouter, Body, HTTPException
from backend.service.qlib_access import run_with_qlib_read_access

router = APIRouter()
logger = logging.getLogger(__name__)
_lock = threading.Lock()

# ══════════════════════════════════════════════
# 模板浏览
# ══════════════════════════════════════════════

@router.get("/templates")
def get_templates(source: str = "whitebox"):
    if source == "whitebox":
        from core.domain.factors.whitebox import FACTOR_TEMPLATES, get_categories
        return {"data": {
            "categories": get_categories(),
            "templates": {
                k: {
                    "key": k, "category": v["category"], "label": v["label"],
                    "dataflow": v.get("dataflow", "price_only"),
                    "data_safety": v.get("data_safety", "unknown"),
                    "logic_desc": v["logic_desc"], "operator": v.get("operator", ""),
                    "default_n": v.get("default_n", 20),
                    "param_definitions": v.get("param_definitions", []),
                    "qlib_expression": v.get("qlib_expression", ""),
                }
                for k, v in FACTOR_TEMPLATES.items()
            },
        }}
    else:
        return {"data": {
            "error": "codegen 数据流模板已迁移至 v2，请使用 /api/factor/register 接口",
            "types": [],
        }}


# ══════════════════════════════════════════════
# 因子生成/预览/注册
# ══════════════════════════════════════════════

@router.post("/preview")
def preview_factor(
    key: str = Body(...), label: str = Body(...),
    logic_desc: str = Body(""), code: str = Body(None),
    template_key: str = Body(None),
):
    try:
        from core.domain.factors.codegen import generate_scoring_function
        result = generate_scoring_function(key=key, label=label, logic_desc=logic_desc)
        return {"data": result}
    except Exception as e:
        return {"data": {"error": f"代码生成失败: {e}"}}


@router.post("/generate")
def generate_factor(
    logic_desc: str = Body(...), key: str = Body(None),
    dataflow: str = Body("price_only"), model: str = Body(None),
):
    return {"data": {"error": "AI 因子生成功能尚未实现，请使用 /api/factor/register 手动注册"}}


@router.post("/register")
def register_factor_endpoint(
    key: str = Body(...), label: str = Body(...),
    logic_desc: str = Body(""), code: str = Body(None),
    source: str = Body("manual"),
    data_safety: str = Body("unknown"),
):
    data_safety = str(data_safety or "unknown").lower()
    if data_safety not in {"safe", "pit", "point_in_time", "caveat", "unsafe", "unknown"}:
        raise HTTPException(422, "data_safety 必须为 safe、pit、point_in_time、caveat、unsafe 或 unknown")
    if data_safety in {"safe", "pit", "point_in_time"}:
        raise HTTPException(422, "人工注册不能自行声明因子通过PIT安全审查；请使用已审查的因子定义")
    with _lock:
        from core.domain.factors.codegen import register_factor_safe
        from core.domain.factors.registry import get_pool
        ok = register_factor_safe(
            key=key, label=label, func=None,  # 自定义因子 func 后续 eval
            category="custom", signal_type="continuous",
            data_safety=data_safety, logic_desc=logic_desc,
        )
        return {"data": {
            "registered": ok, "key": key,
            "total_factors": len(get_pool().get_all_factors()),
        }}


@router.delete("/unregister/{key}")
def unregister_factor_endpoint(key: str):
    with _lock:
        from core.domain.factors.codegen import unregister_factor_safe
        ok = unregister_factor_safe(key)
        return {"data": {"unregistered": ok, "key": key}}


# ══════════════════════════════════════════════
# 参数遍历
# ══════════════════════════════════════════════

@router.post("/sweep")
def sweep_template(
    template_key: str = Body(...), coarse: bool = Body(True),
):
    from core.domain.tuning.sweep_engine import ParamSweepEngine
    engine = ParamSweepEngine()
    variants = engine.sweep_all_variants(template_key, coarse=coarse)
    summary = engine.get_sweep_summary([template_key])
    return {"data": {"template_key": template_key, "variants": variants, "summary": summary}}


@router.post("/sweep-all")
def sweep_all_templates():
    from core.domain.tuning.sweep_engine import ParamSweepEngine
    engine = ParamSweepEngine()
    summary = engine.get_sweep_summary()
    return {"data": summary}


# ══════════════════════════════════════════════
# 七层反过拟合校验
# ══════════════════════════════════════════════

@router.post("/validate")
def validate_factor(
    variant_key: str = Body(...), template_key: str = Body(...),
    params: dict = Body({}), default_n: int = Body(20),
    eval_weeks: int = Body(12), pool: str = Body("all"),
    top_k: int = Body(10),
):
    from core.domain.factors.validation import OverfittingChecker
    from core.shared.infrastructure.async_task import task_manager
    variant = {
        "variant_key": variant_key, "template_key": template_key,
        "params": params, "default_n": default_n,
    }
    checker = OverfittingChecker()

    def _run(progress=None):
        return checker.check_all_layers(
            variant, eval_weeks=eval_weeks, pool=pool, top_k=top_k,
            progress_cb=lambda p: task_manager.update_progress(job_id, {
                "stage": "validating", "current": p["current"], "total": p["total"],
                "name": p["name"], "layers": p["layers"],
            }),
        )

    job_id = task_manager.submit(lambda: run_with_qlib_read_access(_run), f"七层校验 {variant_key}")
    return {"data": {"job_id": job_id}}


@router.get("/validate-progress/{job_id}")
def validate_progress(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(job_id)
    if task is None:
        return {"data": {"status": "not_found"}}
    return {"data": task}


@router.post("/rolling-screen")
def rolling_screen(
    source: str = Body(None), factor_keys: list = Body(None),
    factor_source: str = Body("public"),
    start: str = Body(...), end: str = Body(...),
    step_months: int = Body(3), lookback_months: int = Body(60),
    pool: str = Body("stable_history_500d"), code_prefixes: list = Body(None),
    mode: str = Body("full"), eval_weeks: int = Body(18),
    write_pool: bool = Body(False),
    parallel: bool = Body(False),
    min_periods_frac: float = Body(0.5),
    label_horizon: int = Body(10),
    max_eval_dates: Optional[int] = Body(8),
    train_months: int = Body(60),
    valid_months: int = Body(6),
    test_months: int = Body(3),
    purge_gap: int = Body(11),
):
    """时序滚动因子筛选 (报告修正1): 每 step_months 用之前数据滚动筛选, 严格样本外。异步。
    factor_source: public=公开因子集(source/factor_keys) / pool=当前因子池活跃因子。
    write_pool=True 时每期通过因子集写回动态因子池, 供 use_factor_pool 训练读取 (修正6闭环)。"""
    import datetime as dt
    from backend.service.factor_rolling import FactorRollingService
    from core.shared.infrastructure.async_task import task_manager

    service = FactorRollingService()
    variants = service.resolve_variants(factor_source, source, factor_keys)
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)

    job_id = task_manager._make_id()

    def _run(progress=None):
        return service.run(
            variants, start_d, end_d, step_months=step_months,
            lookback_months=lookback_months, pool=pool, code_prefixes=code_prefixes,
            mode=mode, eval_weeks=eval_weeks,
            write_pool=write_pool, job_id=job_id,
            parallel=parallel, min_periods_frac=min_periods_frac,
            label_horizon=label_horizon, max_eval_dates=max_eval_dates,
            train_months=train_months, valid_months=valid_months,
            test_months=test_months, purge_gap=purge_gap,
            progress_cb=lambda p: task_manager.update_progress(job_id, {
                "stage": p.get("stage", "rolling"),
                "current": p.get("current", 0), "total": p.get("total", 0),
                "anchor": p.get("anchor", ""), "factor": p.get("factor", ""),
                "anchor_idx": p.get("anchor_idx", 0), "anchor_total": p.get("anchor_total", 0),
                "anchors": p.get("anchors", {}),
                "backing_mode": p.get("backing_mode"), "mapped_mb": p.get("mapped_mb"),
            }),
        )

    task_manager.submit_with_id(job_id, lambda: run_with_qlib_read_access(_run),
                                f"滚动筛选 {len(variants)}因子 {start}~{end}")
    return {"data": {"job_id": job_id, "n_factors": len(variants)}}


@router.get("/rolling-screen/history")
def rolling_screen_history(limit: int = 20):
    """列出最近的滚动筛选任务 (从 DB 查，重启不丢)"""
    try:
        from core.data import Database
        if not Database.is_available():
            return {"data": []}
        rows = Database.fetchall(
            "SELECT job_id, "
            "COUNT(DISTINCT anchor_date) as anchor_count, "
            "COUNT(*) as n_rows, "
            "SUM(passed=1) as n_passed, "
            "SUM(passed=0) as n_failed, "
            "MIN(created_at) as created_at "
            "FROM rolling_screen_results GROUP BY job_id "
            "ORDER BY MIN(created_at) DESC LIMIT %s",
            (limit,)
        )
        jobs = []
        for r in rows:
            jobs.append({
                "job_id": r["job_id"],
                "anchor_count": int(r["anchor_count"] or 0),
                "n_passed": int(r["n_passed"] or 0),
                "n_failed": int(r["n_failed"] or 0),
                "created_at": str(r["created_at"]) if r["created_at"] else None,
            })
        return {"data": jobs}
    except Exception as e:
        return {"data": [], "error": str(e)}


@router.get("/rolling-screen/db-result/{job_id}")
def rolling_screen_db_result(job_id: str):
    """查询某次滚动筛选的持久化结果 (从 DB，重启不丢)"""
    try:
        from core.data import Database
        if not Database.is_available():
            return {"data": {"status": "db_unavailable"}}
        rows = Database.fetchall(
            "SELECT anchor_date, factor_key, label, passed, direction, "
            "passed_count, layer_detail, error "
            "FROM rolling_screen_results WHERE job_id=%s "
            "ORDER BY anchor_date, factor_key",
            (job_id,)
        )
        if not rows:
            return {"data": {"status": "not_found"}}
        import json
        per_anchor = {}
        factor_timeline = {}
        for r in rows:
            akey = str(r["anchor_date"])
            if akey not in per_anchor:
                per_anchor[akey] = {"passed": [], "failed": [], "detail": {}}
            detail = {}
            if r["layer_detail"]:
                detail = json.loads(r["layer_detail"]) if isinstance(r["layer_detail"], str) else r["layer_detail"]
            if r["error"]:
                detail = {"error": r["error"]}
            per_anchor[akey]["detail"][r["factor_key"]] = detail
            if r["passed"]:
                per_anchor[akey]["passed"].append(r["factor_key"])
                factor_timeline.setdefault(r["factor_key"], []).append(akey)
            else:
                per_anchor[akey]["failed"].append(r["factor_key"])
        return {"data": {
            "job_id": job_id,
            "anchors": sorted(per_anchor.keys()),
            "per_anchor": per_anchor,
            "factor_timeline": factor_timeline,
        }}
    except Exception as e:
        return {"data": {"status": "error", "error": str(e)}}


@router.get("/rolling-screen/{job_id}")
def rolling_screen_progress(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(job_id)
    if task is None:
        return {"data": {"status": "not_found"}}
    return {"data": task}


# ══════════════════════════════════════════════
# 核心基底与风格卫星证据
# ══════════════════════════════════════════════

@router.post("/architecture/run")
def create_factor_architecture_run(payload: dict = Body(...)):
    try:
        from backend.service.factor_architecture import factor_architecture_service
        return {"data": factor_architecture_service.create(payload)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/architecture/history")
def list_factor_architecture_runs():
    from core.data.repos.factor_architecture import factor_architecture_repo
    return {"data": factor_architecture_repo.list_runs()}


@router.get("/architecture/{run_id}")
def get_factor_architecture_run(run_id: str):
    from backend.service.factor_architecture import factor_architecture_service
    result = factor_architecture_service.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="因子架构任务不存在")
    return {"data": result}


@router.get("/architecture/{run_id}/evidence")
def get_factor_architecture_evidence(run_id: str, factor_key: str = None):
    from core.data.repos.factor_architecture import factor_architecture_repo
    return {"data": factor_architecture_repo.list_evidence(run_id, factor_key=factor_key)}


@router.get("/architecture/{run_id}/candidates")
def get_factor_architecture_candidates(run_id: str):
    from core.data.repos.factor_architecture import factor_architecture_repo
    return {"data": factor_architecture_repo.list_candidates(run_id)}


@router.post("/architecture/{run_id}/repair-evidence")
def repair_factor_architecture_evidence(run_id: str):
    try:
        from backend.service.factor_architecture import factor_architecture_service
        return {"data": factor_architecture_service.create_evidence_repair(run_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/architecture/{run_id}/regroup-states")
def regroup_factor_architecture_states(run_id: str):
    try:
        from backend.service.factor_state_regroup import factor_state_regroup_service
        return {"data": factor_state_regroup_service.create(run_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/architecture/{run_id}/residual")
def create_factor_architecture_residual(run_id: str, payload: dict = Body(default={})):
    try:
        from backend.service.factor_architecture import factor_architecture_service
        return {"data": factor_architecture_service.create_residual(run_id, payload)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/architecture/{run_id}/model-ab")
def create_factor_architecture_model_ab(run_id: str, payload: dict = Body(default={})):
    try:
        from backend.service.factor_architecture import factor_architecture_service
        return {"data": factor_architecture_service.create_model_ab(run_id, payload)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/architecture/{run_id}/balanced-core")
def create_balanced_factor_architecture_core(run_id: str, payload: dict = Body(default={})):
    try:
        from backend.service.factor_architecture import factor_architecture_service
        return {"data": factor_architecture_service.create_balanced_core(run_id, payload)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/architecture/{run_id}/cancel")
def cancel_factor_architecture(run_id: str):
    try:
        from backend.service.factor_architecture import factor_architecture_service
        return {"data": factor_architecture_service.cancel(run_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/architecture/{run_id}/publish-satellites")
def publish_factor_architecture_satellites(run_id: str):
    try:
        from backend.service.factor_architecture import factor_architecture_service
        return {"data": factor_architecture_service.publish_satellites(run_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/architecture/{run_id}/evaluations")
def get_factor_architecture_evaluations(run_id: str, set_id: str = None):
    from core.data.repos.factor_architecture import factor_architecture_repo
    return {"data": factor_architecture_repo.list_evaluations(run_id, set_id=set_id)}


@router.get("/architecture-task/{task_id}")
def get_factor_architecture_task(task_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="因子架构子任务不存在")
    return {"data": task}


# ══════════════════════════════════════════════
# 因子池管理
# ══════════════════════════════════════════════

@router.get("/pool/all")
def pool_all():
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    all_f = pool.get_all_factors()
    factors = []
    for key, info in sorted(all_f.items()):
        factors.append({
            "key": key,
            "label": info.get("label", key),
            "active": info.get("active", True),
            "pool_category": info.get("pool_category", ""),
            "category": info.get("category", ""),
            "template_key": info.get("template_key", ""),
            "params": info.get("params", {}),
            "added_date": info.get("added_date", ""),
            "expr": info.get("expr"),
        })
    return {"data": {"factors": factors, "count": len(factors)}}


@router.get("/pool/for-training")
def pool_factors_for_training():
    """返回因子池中所有活跃因子，附带 Qlib 表达式，供训练时选为因子集"""
    from core.domain.factors.registry import get_pool
    from core.domain.factors.whitebox import FACTOR_TEMPLATES
    pool = get_pool()
    active = pool.get_active_factors()
    factors = []
    for key, info in sorted(active.items()):
        tk = info.get("template_key", "")
        tmpl = FACTOR_TEMPLATES.get(tk, {})
        params = info.get("params", {})
        expr = tmpl.get("qlib_expression", "")
        for pk, pv in params.items():
            expr = expr.replace(f"{{{pk}}}", str(pv))
        import re
        expr = re.sub(r"\{[^}]+\}", "0", expr)
        factors.append({
            "key": key,
            "label": info.get("label", key),
            "category": info.get("category", "white_box"),
            "qlib_expression": expr.strip() or None,
            "template_key": tk,
            "params": params,
            "active": info.get("active", True),
        })
    return {"data": {"factors": factors, "count": len(factors)}}


@router.get("/pool-status")
def factor_pool_status():
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    return {"data": {
        "active_count": len(pool.get_active_factors()),
        "total_count": len(pool.get_all_factors()),
        "active_factors": pool.get_active_factors(),
    }}


@router.post("/monthly-check")
def monthly_check():
    from core.domain.factors.monitor import FactorMonitor
    monitor = FactorMonitor()
    report = monitor.monthly_check(as_of=date.today())
    return {"data": report}


@router.get("/report/monthly")
def monthly_report():
    from core.domain.factors.monitor import FactorMonitor
    monitor = FactorMonitor()
    path = monitor.generate_monthly_report(as_of=date.today())
    return {"data": {"path": path}}


# ══════════════════════════════════════════════
# 公共因子 (Alpha158/101)
# ══════════════════════════════════════════════

@router.get("/public-factors")
def get_public_factors(source: str = "alpha158"):
    from core.domain.factors.adapter import PublicFactorAdapter
    adapter = PublicFactorAdapter()
    factors = adapter.list_factors(source=source)
    return {"data": {"source": source, "factors": factors}}


@router.post("/public-factors/test")
def test_public_factors(
    source: str = Body("alpha158"), eval_weeks: int = Body(12),
    pool: str = Body("all"), top_k: int = Body(10), dedupe: bool = Body(True),
    anchor_date: str = Body(None),
):
    from backend.service.factor_validation import FactorValidationService
    result = FactorValidationService().validate_public_factors(
        source=source, eval_weeks=eval_weeks, pool=pool, top_k=top_k, dedupe=dedupe,
        anchor_date=anchor_date,
    )
    return {"data": result}


@router.post("/public-factors/compare")
def compare_factor_sets(data: dict = Body(...)):
    from core.domain.factors.adapter import PublicFactorAdapter
    adapter = PublicFactorAdapter()
    result = adapter.compare_factor_sets(data=data)
    return {"data": result}


@router.post("/public-factors/adapt")
def adapt_public_factor(
    factor_key: str = Body(...), source: str = Body("alpha158"),
):
    from core.domain.factors.adapter import PublicFactorAdapter
    adapter = PublicFactorAdapter()
    result = adapter.adapt_factor(factor_key=factor_key, source=source)
    return {"data": result}


# ══════════════════════════════════════════════
# 因子集管理
# ══════════════════════════════════════════════

@router.get("/factor-sets")
def get_factor_sets():
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    return {"data": pool.get_factor_sets()}


@router.post("/factor-sets")
def save_factor_set(name: str = Body(...), factor_keys: list = Body(...)):
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    pool.save_factor_set(name, factor_keys)
    return {"data": {"saved": True, "name": name}}


@router.delete("/factor-sets/{name}")
def delete_factor_set(name: str):
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    pool.delete_factor_set(name)
    return {"data": {"deleted": True, "name": name}}


@router.post("/factor-sets/export-qlib")
def export_qlib_expressions(name: str = Body(...)):
    from core.domain.factors.registry import get_pool
    from core.adapters.qlib.factor_expression import factor_to_qlib_expression
    pool = get_pool()
    factors = pool.get_factor_set(name)
    expressions = {}
    for fk in factors:
        expr = factor_to_qlib_expression(fk)
        if expr:
            expressions[fk] = expr
    return {"data": {"name": name, "expressions": expressions}}


# ══════════════════════════════════════════════
# 一键研发流水线
# ══════════════════════════════════════════════

@router.post("/pipeline")
def run_pipeline(
    template_key: str = Body(None),
    run_all: bool = Body(False),
    eval_weeks: int = Body(12), pool: str = Body("all"),
):
    job_id = uuid.uuid4().hex[:12]

    def _run():
        if run_all:
            from core.domain.factors.whitebox import FACTOR_TEMPLATES
            keys = list(FACTOR_TEMPLATES.keys())
        else:
            keys = [template_key] if template_key else []
        results = {}
        for tk in keys:
            from core.domain.tuning.sweep_engine import ParamSweepEngine
            engine = ParamSweepEngine()
            variants = engine.sweep_all_variants(tk, coarse=True)
            results[tk] = {"variants": len(variants)}
        return results

    import threading
    threading.Thread(target=run_with_qlib_read_access, args=(_run,), daemon=True).start()
    return {"data": {"job_id": job_id, "template_keys": [template_key] if not run_all else "all"}}


@router.post("/preload-cache")
def preload_cache(pool: str = Body("all")):
    import datetime as _dt
    from core.domain.factors.rolling import _ensure_daily_in_redis
    today = _dt.date.today()
    start = (today - _dt.timedelta(days=365 * 3)).isoformat()
    end = today.isoformat()
    n = _ensure_daily_in_redis(start, end, None)
    # 同时预热 snapshot 缓存 (兼容旧逻辑)
    from core.domain.tuning._tuner import get_tuner
    tuner = get_tuner()
    tuner.signal_validity_test(pool=pool, eval_weeks=2, skip_tech_subs=True,
                               date_step=20, skip_financial_prefetch=True)
    return {"data": {"status": "done", "daily_cached": n}}


# ══════════════════════════════════════════════
# 因子池开关 (批量)
# ══════════════════════════════════════════════

@router.post("/pool-toggle")
def pool_toggle(active: bool = Body(...), keys: list = Body(None)):
    from core.domain.factors.registry import get_pool
    if not keys:
        # 防止空请求体误触发全量翻转
        raise HTTPException(400, "必须指定至少一个因子 key")
    pool = get_pool()
    for k in keys:
        pool.toggle_factor(k, active)
    return {"data": {"toggled": keys, "active": active}}


# ══════════════════════════════════════════════
# 精选集管理
# ══════════════════════════════════════════════

@router.get("/curated-sets")
def list_curated_sets():
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    sets = pool.list_curated_sets()
    return {"data": sets}


@router.get("/curated-sets/{name}")
def get_curated_set(name: str):
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    entry = pool.get_curated_set(name)
    if entry is None:
        return {"data": None}
    return {"data": entry}


@router.delete("/curated-sets/{name}")
def delete_curated_set(name: str):
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    ok = pool.delete_curated_set(name)
    return {"data": {"deleted": ok, "name": name}}


# ══════════════════════════════════════════════
# 市场状态
# ══════════════════════════════════════════════

@router.get("/regime-status")
def regime_status():
    try:
        import datetime as _dt
        from core.domain.regime.classifier import get_current_regime
        rr = get_current_regime()
        td = rr.as_of_date
        if not td or td == "latest":
            td = str(_dt.date.today())
        return {"data": {"current": {
            "regime": rr.regime.value,
            "confidence": rr.confidence,
            "trade_date": td,
        }}}
    except Exception:
        pass
    return {"data": {"current": None}}


# ══════════════════════════════════════════════
# 拥挤度检测 (异步)
# ══════════════════════════════════════════════

@router.post("/crowding-check")
def crowding_check(
    factor_keys: list = Body(...), pool: str = Body("a500"),
    eval_weeks: int = Body(26),
):
    from core.domain.factors.crowding import run_crowding_check
    from core.domain.factors.registry import get_pool
    from core.shared.infrastructure.async_task import task_manager

    pool_mgr = get_pool()
    all_f = pool_mgr.get_all_factors()
    factor_infos = []
    for k in factor_keys:
        info = all_f.get(k, {})
        factor_infos.append({
            "key": k,
            "label": info.get("label", k),
            "expr": info.get("expr"),
        })

    job_id = task_manager._make_id()

    def _run(progress=None):
        import datetime as dt
        return run_crowding_check(factor_infos, pool=pool, eval_weeks=eval_weeks, as_of=dt.date.today())

    task_manager.submit_with_id(job_id, lambda: run_with_qlib_read_access(_run),
                                f"拥挤度检测 {len(factor_keys)}因子")
    return {"data": {"job_id": job_id}}


@router.get("/crowding-check/{job_id}")
def crowding_check_progress(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(job_id)
    if task is None:
        return {"data": {"status": "not_found"}}
    return {"data": task}


# ══════════════════════════════════════════════
# Portfolio Sort 分组回测 (异步)
# ══════════════════════════════════════════════

@router.post("/portfolio-sort")
def portfolio_sort(
    factor_keys: list = Body(...), pool: str = Body("a500"),
    eval_weeks: int = Body(12), forward_days: int = Body(20),
):
    from core.shared.infrastructure.async_task import task_manager

    job_id = task_manager._make_id()

    def _run(progress=None):
        import numpy as np
        from backend.service.factor_validation import FactorValidationService
        results = FactorValidationService().portfolio_sort(
            factor_keys,
            pool=pool,
            eval_weeks=eval_weeks,
            forward_days=forward_days,
            progress_cb=(
                (lambda current, total: progress.update({"current": current, "total": total}))
                if progress is not None else None
            ),
        )
        return [{
            "factor_key": r.factor_key, "factor_label": r.factor_label,
            "n_dates": r.n_dates, "annual_return": r.annual_return,
            "sharpe": r.sharpe, "monotonicity": r.monotonicity_score,
            "hit_rate": r.hit_rate, "max_drawdown": r.max_drawdown,
            "mean_spread": r.mean_spread,
            "q_returns": {
                str(q): float(np.mean(vals)) if vals else 0
                for q, vals in r.quintile_returns.items()
            },
        } for r in results]

    task_manager.submit_with_id(job_id, lambda: run_with_qlib_read_access(_run),
                                f"分组回测 {len(factor_keys)}因子")
    return {"data": {"job_id": job_id}}


@router.get("/portfolio-sort/{job_id}")
def portfolio_sort_progress(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(job_id)
    if task is None:
        return {"data": {"status": "not_found"}}
    return {"data": task}


# ══════════════════════════════════════════════
# Fama-MacBeth 回归 (异步)
# ══════════════════════════════════════════════

@router.post("/fama-macbeth")
def fama_macbeth(
    factor_keys: list = Body(...), pool: str = Body("a500"),
    eval_weeks: int = Body(52), forward_days: int = Body(20),
):
    from core.shared.infrastructure.async_task import task_manager

    job_id = task_manager._make_id()

    def _run(progress=None):
        import datetime as dt
        from core.domain.factors.fama_macbeth import run_fama_macbeth
        end = dt.date.today()
        start = end - dt.timedelta(days=eval_weeks * 7 + 30)
        return run_fama_macbeth(
            factor_keys=factor_keys, pool=pool,
            start=start, end=end,
            eval_weeks=eval_weeks, forward_days=forward_days,
        )

    task_manager.submit_with_id(job_id, lambda: run_with_qlib_read_access(_run),
                                f"Fama-MacBeth {len(factor_keys)}因子")
    return {"data": {"job_id": job_id}}


@router.get("/fama-macbeth/{job_id}")
def fama_macbeth_progress(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(job_id)
    if task is None:
        return {"data": {"status": "not_found"}}
    return {"data": task}


# ══════════════════════════════════════════════
# 因子精选 (异步)
# ══════════════════════════════════════════════

@router.post("/curate")
def curate_factors(
    pool: str = Body("all"), target_lower: int = Body(60),
    target_upper: int = Body(80), eval_weeks: int = Body(12),
):
    from core.shared.infrastructure.async_task import task_manager

    job_id = task_manager._make_id()

    def _run(progress=None):
        from core.domain.factors.registry import get_pool
        from core.domain.factors.adapter import PublicFactorAdapter
        from core.domain.factors.selector import curate

        pool_mgr = get_pool()
        active = pool_mgr.get_active_factors()
        active_keys = list(active.keys())
        adapter = PublicFactorAdapter()
        target_range = (target_lower, target_upper)

        task_manager.update_progress(job_id, {"progress": 10, "message": "开始精选..."})

        flat_list = curate(
            active_keys=active_keys, adapter=adapter,
            validation_results=None, pool=pool,
            target_range=target_range, method="category",
        )

        if isinstance(flat_list, tuple):
            flat_list = flat_list[1] if len(flat_list) > 1 else flat_list[0]
        if not isinstance(flat_list, list):
            flat_list = []

        return {
            "input_count": len(active_keys),
            "output_count": len(flat_list),
            "flat_list": flat_list,
        }

    task_manager.submit_with_id(job_id, lambda: run_with_qlib_read_access(_run), "因子精选")
    return {"data": {"job_id": job_id}}


@router.get("/curate-progress/{job_id}")
def curate_progress(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(job_id)
    if task is None:
        return {"data": {"status": "not_found"}}
    return {"data": task}


# ══════════════════════════════════════════════
# 策略方向精选 + ICIR 回填
# ══════════════════════════════════════════════

@router.post("/curate-direction")
def curate_direction_endpoint(payload: dict = Body(...)):
    direction = payload.get("direction", "")
    as_of = payload.get("as_of")  # 可选 'YYYY-MM-DD' — 时序截断, 无前视
    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter
    from core.domain.factors.selector import curate_direction

    pool_mgr = get_pool()
    active = pool_mgr.get_active_factors(as_of=as_of or None)
    active_keys = list(active.keys())
    adapter = PublicFactorAdapter()

    name, flat_list, count = curate_direction(
        direction_name=direction, active_keys=active_keys,
        adapter=adapter, validation_results=None, save=True,
        regime_aware=True, pool_active=active, as_of=as_of or None,
    )
    return {"data": {"name": name, "count": count, "factors": flat_list}}


@router.post("/backfill-icir/{dirKey}")
def backfill_icir(dirKey: str):
    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter
    from core.domain.factors.selector import (
        curate_direction, _load_regime_dates_for_direction, _compute_regime_icir,
    )

    pool_mgr = get_pool()
    active = pool_mgr.get_active_factors()
    active_keys = list(active.keys())

    regime_dates = _load_regime_dates_for_direction(dirKey)
    if not regime_dates:
        return {"data": {"error": f"方向 {dirKey} 无对应市场状态数据", "count": 0}}

    icir_map = _compute_regime_icir(active_keys, regime_dates, min_dates=2)
    if not icir_map:
        return {"data": {"error": "ICIR 计算无结果", "count": 0}}

    adapter = PublicFactorAdapter()
    name, flat_list, count = curate_direction(
        direction_name=dirKey, active_keys=active_keys,
        adapter=adapter, validation_results=None, save=True,
        regime_aware=True, pool_active=active,
    )
    return {"data": {"name": name, "count": count,
                     "factors": flat_list, "icir_count": len(icir_map)}}


@router.post("/recalc-all-icir")
def recalc_all_icir():
    import json as _json
    from core.domain.factors.registry import get_pool
    from core.data import Database

    pool_mgr = get_pool()
    active = pool_mgr.get_active_factors()
    active_keys = list(active.keys())

    if not Database.is_available():
        return {"data": {"error": "数据库不可用", "count": 0}}

    BATCH = 200
    factor_ics = {}
    for i in range(0, len(active_keys), BATCH):
        batch = active_keys[i:i + BATCH]
        ph = ','.join(['%s'] * len(batch))
        try:
            rows = Database.fetchall(
                f"SELECT factor_key, daily_ics FROM factor_validation_scores "
                f"WHERE daily_ics IS NOT NULL AND factor_key IN ({ph})",
                batch
            )
        except Exception:
            continue
        for r in rows:
            key = r["factor_key"]
            daily_ics = r["daily_ics"]
            if not daily_ics:
                continue
            try:
                pairs = _json.loads(daily_ics) if isinstance(daily_ics, str) else daily_ics
            except Exception:
                continue
            ics = [float(p[1]) for p in pairs if len(p) >= 2 and p[1] is not None]
            if key not in factor_ics:
                factor_ics[key] = []
            factor_ics[key].extend(ics)

    updated = 0
    skipped = 0
    for key in active_keys:
        all_ics = factor_ics.get(key, [])
        if len(all_ics) < 3:
            skipped += 1
            continue
        ic_mean = sum(all_ics) / len(all_ics)
        ic_std = (sum((x - ic_mean) ** 2 for x in all_ics) / len(all_ics)) ** 0.5
        icir = ic_mean / ic_std if ic_std > 0 else 0
        try:
            Database.execute(
                "UPDATE factor_validation_scores SET icir=%s, ic_mean=%s, ic_std=%s, "
                "n_ic_dates=%s WHERE factor_key=%s",
                (round(icir, 6), round(ic_mean, 6), round(ic_std, 6),
                 len(all_ics), key)
            )
            updated += 1
        except Exception:
            pass

    return {"data": {
        "total_active": len(active_keys),
        "updated": updated,
        "skipped": skipped,
        "message": f"已更新 {updated} 个因子的ICIR (跨锚点聚合, 跳过 {skipped} 个数据不足)",
    }}


@router.post("/genetic-mine")
def genetic_mine(
    pool: str = Body("csi300"),
    train_start: str = Body(...), train_end: str = Body(...),
    valid_start: str = Body(...), valid_end: str = Body(...),
    step: int = Body(5),
    population: int = Body(200), generations: int = Body(20),
    top_n: int = Body(10), forward_days: int = Body(20),
    risk_neutralize: bool = Body(True),
    write_pool: bool = Body(False),
    seed: int = Body(42),
):
    """旧遗传挖矿接口兼容包装器。write_pool 已废弃，不会自动写入生产因子池。"""
    from backend.service.factor_mining import factor_mining_service
    try:
        result = factor_mining_service.create_run({
            "pool": pool, "train_start": train_start, "train_end": train_end,
            "valid_start": valid_start, "valid_end": valid_end, "step": step,
            "population": population, "generations": generations, "top_n": top_n,
            "forward_days": forward_days, "risk_neutralize": risk_neutralize, "seed": seed,
            "validation_profile": "default", "baseline_set": "current_curated",
            "final_oos_start": valid_start, "final_oos_end": valid_end,
            "compatibility_mode": True,
        })
        return {"data": {"job_id": result["run_id"], "run_id": result["run_id"], "write_pool_deprecated": bool(write_pool)}}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/genetic-mine/{job_id}")
def genetic_mine_progress(job_id: str):
    from backend.service.factor_mining import factor_mining_service
    task = factor_mining_service.get_run(job_id)
    if task is None:
        return {"data": {"status": "not_found"}}
    return {"data": task}


@router.post("/mining/runs")
def create_factor_mining_run(
    pool: str = Body("csi300"),
    train_start: str = Body(...), train_end: str = Body(...),
    valid_start: str = Body(...), valid_end: str = Body(...),
    step: int = Body(5), population: int = Body(100), generations: int = Body(10),
    max_depth: int = Body(5),
    top_n: int = Body(20), forward_days: int = Body(20),
    risk_neutralize: bool = Body(True), validation_profile: str = Body("default"),
    baseline_set: str = Body("current_curated"), seed: int = Body(42),
    baseline_model_experiment_id: str = Body(""), baseline_model_run_id: str = Body(""),
    backtest_profile: str = Body("production_screening"), backtest_freq: str = Body("week"),
    backtest_top_n: int = Body(10), backtest_use_timing: bool = Body(True),
    backtest_macro_bias: bool = Body(True), backtest_sector_fusion: Optional[bool] = Body(None),
    overlay_weight: float = Body(0.15), final_oos_start: Optional[str] = Body(None),
    final_oos_end: Optional[str] = Body(None),
    candidate_sources: list[str] = Body(["gp"]),
    llm_provider: str = Body("deepseek"), llm_candidate_count: int = Body(10),
    llm_temperature: float = Body(0.2), llm_max_tokens: int = Body(4096),
    llm_prompt_profile: str = Body("factor_hypothesis_v1"),
    llm_research_focus: str = Body(""),
):
    from backend.service.factor_mining import factor_mining_service
    try:
        return {"data": factor_mining_service.create_run({
            "pool": pool, "train_start": train_start, "train_end": train_end,
            "valid_start": valid_start, "valid_end": valid_end, "step": step,
            "population": population, "generations": generations, "max_depth": max_depth,
            "top_n": top_n,
            "forward_days": forward_days, "risk_neutralize": risk_neutralize,
            "validation_profile": validation_profile, "baseline_set": baseline_set, "seed": seed,
            "baseline_model_experiment_id": baseline_model_experiment_id,
            "baseline_model_run_id": baseline_model_run_id,
            "backtest_profile": backtest_profile, "backtest_freq": backtest_freq,
            "backtest_top_n": backtest_top_n, "backtest_use_timing": backtest_use_timing,
            "backtest_macro_bias": backtest_macro_bias, "backtest_sector_fusion": backtest_sector_fusion,
            "overlay_weight": overlay_weight, "final_oos_start": final_oos_start,
            "final_oos_end": final_oos_end,
            "candidate_sources": candidate_sources, "llm_provider": llm_provider,
            "llm_candidate_count": llm_candidate_count, "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens, "llm_prompt_profile": llm_prompt_profile,
            "llm_research_focus": llm_research_focus,
        })}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/mining/campaigns")
def create_factor_mining_campaign(payload: dict = Body(...)):
    from backend.service.factor_mining_campaign import factor_mining_campaign_service
    try:
        return {"data": factor_mining_campaign_service.create(payload)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/mining/campaigns/{campaign_id}")
def get_factor_mining_campaign(campaign_id: str):
    from backend.service.factor_mining_campaign import factor_mining_campaign_service
    result = factor_mining_campaign_service.get(campaign_id)
    if result is None:
        raise HTTPException(status_code=404, detail="持续挖掘任务不存在")
    return {"data": result}


@router.post("/mining/campaigns/{campaign_id}/cancel")
def cancel_factor_mining_campaign(campaign_id: str):
    from backend.service.factor_mining_campaign import factor_mining_campaign_service
    result = factor_mining_campaign_service.cancel(campaign_id)
    if not result["cancel_requested"]:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    return {"data": result}


@router.get("/mining/archive")
def list_factor_mining_archive(limit: int = 100, status: str = None):
    from backend.service.factor_mining_campaign import factor_mining_campaign_service
    return {"data": {
        "candidates": factor_mining_campaign_service.archive(
            limit=max(1, min(int(limit), 500)), status=status),
    }}


@router.post("/mining/runs/{run_id}/train")
def train_factor_mining_candidates(run_id: str, payload: dict = Body(...)):
    from backend.service.factor_mining_training import factor_mining_training_service
    try:
        return {"data": factor_mining_training_service.create(run_id, payload)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/mining/training/{training_id}")
def get_factor_mining_training(training_id: str):
    from backend.service.factor_mining_training import factor_mining_training_service
    result = factor_mining_training_service.get(training_id)
    if result is None:
        raise HTTPException(status_code=404, detail="研究训练任务不存在")
    return {"data": result}


@router.get("/mining/models")
def list_factor_mining_models(
    pool: str = "csi300",
    baseline_set: str = "current_curated",
    train_end: str = None,
):
    from backend.service.factor_mining import factor_mining_service
    if not train_end:
        raise HTTPException(status_code=400, detail="必须提供 train_end")
    try:
        return {"data": factor_mining_service.list_models(pool, baseline_set, train_end)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/mining/runs/{run_id}")
def get_factor_mining_run(run_id: str):
    from backend.service.factor_mining import factor_mining_service
    result = factor_mining_service.get_run(run_id)
    if result is None:
        return {"data": {"status": "not_found", "run_id": run_id}}
    return {"data": result}


@router.post("/mining/runs/{run_id}/resume")
def resume_factor_mining_run(run_id: str):
    from backend.service.factor_mining import factor_mining_service
    try:
        return {"data": factor_mining_service.resume_run(run_id)}
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/mining/runs/{run_id}/candidates")
def get_factor_mining_candidates(run_id: str):
    from backend.service.factor_mining import factor_mining_service
    return {"data": {"run_id": run_id, "candidates": factor_mining_service.list_candidates(run_id)}}


@router.get("/mining/runs/{run_id}/proposals")
def get_factor_mining_proposals(run_id: str):
    from backend.service.factor_mining import factor_mining_service
    return {"data": {"run_id": run_id, "proposals": factor_mining_service.list_proposals(run_id)}}


@router.get("/mining/runs/{run_id}/candidates/{candidate_key}")
def get_factor_mining_candidate_detail(run_id: str, candidate_key: str):
    from backend.service.factor_mining import factor_mining_service
    candidate = factor_mining_service.get_candidate_detail(run_id, candidate_key)
    if candidate is None:
        raise HTTPException(status_code=404, detail="候选不存在")
    return {"data": {"run_id": run_id, "candidate": candidate}}


@router.get("/mining/runs/{run_id}/report")
def get_factor_mining_report(run_id: str):
    from backend.service.factor_mining import factor_mining_service
    return {"data": {"run_id": run_id, "report": factor_mining_service.get_report(run_id)}}


@router.get("/mining/runs/{run_id}/backtests")
def get_factor_mining_backtests(run_id: str, candidate_key: str = None):
    from backend.service.factor_mining import factor_mining_service
    return {"data": {
        "run_id": run_id,
        "backtests": factor_mining_service.list_backtests(run_id, candidate_key),
    }}


@router.post("/mining/runs/{run_id}/promote")
def decide_factor_mining_candidate(
    run_id: str,
    candidate_key: str = Body(...),
    target_pool: str = Body("current_curated"),
    action: str = Body("promote"),
    reason: str = Body(""),
):
    from backend.service.factor_mining import factor_mining_service
    try:
        candidate = factor_mining_service.decide(
            run_id, candidate_key, target_pool, action=action, reason=reason)
        return {"data": {"run_id": run_id, "candidate": candidate}}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/mining/runs/{run_id}/cancel")
def cancel_factor_mining_run(run_id: str):
    from backend.service.factor_mining import factor_mining_service
    result = factor_mining_service.cancel(run_id)
    if not result["cancel_requested"]:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    return {"data": result}
