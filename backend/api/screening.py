"""选股筛选 API"""
import anyio
import copy
import logging
import uuid
import threading
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Body, Header
from fastapi.encoders import jsonable_encoder

from core.shared.infrastructure.thread_pool import thread_pool
from backend.service.qlib_access import run_with_qlib_read_access

router = APIRouter()
logger = logging.getLogger(__name__)

_results: dict = {}
_history: list = []
_lock = threading.Lock()
_worker_id = uuid.uuid4().hex
_MAX_RESULTS = 100  # 防止内存泄漏, 超出时删除最旧条目
_MAX_HISTORY = 200  # 历史记录上限


def _cleanup_old_results():
    """清理旧结果, 保留最近 _MAX_RESULTS 条"""
    if len(_results) > _MAX_RESULTS:
        old = sorted(_results.keys(),
                     key=lambda k: _results[k].get("started_at", ""))[:len(_results) - _MAX_RESULTS]
        for k in old:
            del _results[k]


def _run_screening_task(job_id, *args):
    from core.data import ScreeningResultRepo

    stop_heartbeat = threading.Event()
    heartbeat_started = False

    def heartbeat():
        while not stop_heartbeat.wait(15):
            progress = {"stage": "running"}
            try:
                from core.domain.screening._screener import get_screener
                task = get_screener()._jobs.get(job_id) or {}
                progress = task.get("progress") or progress
            except Exception:
                pass
            try:
                ScreeningResultRepo.heartbeat(job_id, _worker_id, progress)
            except Exception as error:
                logger.warning(f"[Screening] 任务 {job_id} 心跳持久化失败: {error}")

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    try:
        ScreeningResultRepo.mark_running(job_id, _worker_id)
        with _lock:
            if job_id in _results:
                _results[job_id]["status"] = "running"
        heartbeat_thread.start()
        heartbeat_started = True
        return run_with_qlib_read_access(_run_screening, job_id, *args)
    except Exception as error:
        try:
            ScreeningResultRepo.fail(job_id, _worker_id, error)
        except Exception as persist_error:
            logger.error(f"[Screening] 任务 {job_id} 失败状态持久化失败: {persist_error}")
        with _lock:
            _results[job_id] = {
                "job_id": job_id, "status": "error", "error": str(error),
                "started_at": _results.get(job_id, {}).get("started_at"),
            }
        raise
    finally:
        stop_heartbeat.set()
        if heartbeat_started:
            heartbeat_thread.join(timeout=2)


def _run_screening(job_id: str, pool: str, top_n: int, debate_top: int,
                   debate_agents: int,
                   skip_debate: bool, skip_l0: bool, skip_l1: bool, skip_l2: bool,
                   skip_tech: bool, skip_qlib: bool, skip_sent: bool,
                   skip_fundamental: bool, skip_flow: bool,
                   skip_north_flow: bool, skip_size_turnover: bool,
                   skip_momentum: bool, skip_volatility: bool,
                   skip_illiq: bool, skip_overnight_gap: bool,
                   skip_wb_industry: bool, skip_wb_divergence: bool, skip_wb_moneyflow: bool,
                   wb_industry_n: int, wb_divergence_n: int, wb_moneyflow_n: int,
                   illiq_n: int, overnight_gap_n: int,
                   source_weights: dict, tech_preset: str, qlib_model: list,
                   hot_sector_count: int, custom_sectors: list,
                   code_prefixes: list, concept_sectors: list,
                   regime: str = None, eval_date: str = None,
                   macro_bias: bool = False, sector_fusion: bool = False,
                   macro_regime: bool = False, smooth_regime: bool = False,
                   use_vol_switch: bool = False, use_conditional_switch: bool = True,
                   state_models: dict = None,
                   neutralize_qlib: bool = True,
                   decision_day_filter: bool = True,
                   shadow_models: list = None,
                   strategy_profile: dict = None):
    from backend.service.market_data import MarketDataService
    from core.domain.screening._screener import get_screener
    from datetime import date as dt_date
    screener = get_screener()
    qlib_model_tuple = tuple(qlib_model) if qlib_model and len(qlib_model) == 2 else None
    requested_eval_d = dt_date.fromisoformat(eval_date) if eval_date else None
    eval_d = MarketDataService.get_screening_decision_date(requested_eval_d)
    source_weights = {"qlib": 1.0}
    conditional_market_env = {}
    if use_vol_switch and use_conditional_switch:
        try:
            conditional_market_env = MarketDataService.get_conditional_market_env(end_date=eval_d)
        except Exception as e:
            logger.warning(f"[VolSwitch] 条件切换环境特征获取失败, 回退二值切换: {e}")
    operational_snapshot = None
    profile_warnings = []
    if strategy_profile:
        from backend.service.strategy_profile import StrategyProfileService
        cfg = strategy_profile.get("config") or {}
        research_only = bool((cfg.get("factors") or {}).get("research_only"))
        if requested_eval_d and eval_d != requested_eval_d:
            raise RuntimeError("策略版本选股不允许决策日回退，请选择有效交易日")
        first_date = str((cfg.get("availability") or {}).get("first_decision_date") or "")[:10]
        if first_date and eval_d.isoformat() < first_date:
            raise RuntimeError(f"该策略最早可运行日期为 {first_date}")
        min_history_days = int((cfg.get("universe") or {}).get("min_history_days") or 500)
        operational_snapshot = StrategyProfileService.build_data_snapshot(eval_d, min_history_days)
        if (not StrategyProfileService.snapshot_has_verified_pit(operational_snapshot)
                or operational_snapshot.get("freshness_status") != "fresh"):
            raise RuntimeError("当前数据快照未通过时点一致性或新鲜度检查")
        if operational_snapshot.get("security_reference_status") != "pass":
            if not research_only:
                raise RuntimeError("当前历史证券状态快照不完整，禁止正式预测")
            profile_warnings.append("历史行业、特别处理、上市和停牌状态快照尚未完整")

    qlib_readiness = MarketDataService.get_qlib_readiness(eval_d)
    if not qlib_readiness["ready"]:
        if not qlib_readiness.get("latest_date") or qlib_readiness["latest_date"] < qlib_readiness["decision_date"]:
            raise RuntimeError(
                f"Qlib数据仅同步到 {qlib_readiness['latest_date'] or '未知'}，"
                f"落后决策日 {qlib_readiness['decision_date']}；请先执行Qlib增量同步后重试"
            )
        raise RuntimeError(
            f"Qlib决策日截面不完整：{qlib_readiness['available']}/"
            f"{qlib_readiness['expected']}；请先修复缺失股票并重新同步Qlib"
        )

    result = screener.screen(
        pool=pool, top_n=top_n, debate_top=debate_top,
        debate_agents=debate_agents,
        job_id=job_id, skip_debate=skip_debate,
        skip_l0=skip_l0, skip_l1=skip_l1, skip_l2=skip_l2,
        skip_tech=skip_tech, skip_qlib=skip_qlib, skip_sent=skip_sent,
        skip_fundamental=skip_fundamental, skip_flow=skip_flow,
        skip_north_flow=skip_north_flow, skip_size_turnover=skip_size_turnover,
        skip_momentum=skip_momentum, skip_volatility=skip_volatility,
        skip_illiq=skip_illiq, skip_overnight_gap=skip_overnight_gap,
        skip_wb_industry=skip_wb_industry, skip_wb_divergence=skip_wb_divergence,
        skip_wb_moneyflow=skip_wb_moneyflow,
        wb_industry_n=wb_industry_n, wb_divergence_n=wb_divergence_n, wb_moneyflow_n=wb_moneyflow_n,
        illiq_n=illiq_n, overnight_gap_n=overnight_gap_n,
        source_weights=source_weights or None,
        tech_preset=tech_preset, qlib_model=qlib_model_tuple,
        hot_sector_count=hot_sector_count,
        custom_sectors=custom_sectors,
        code_prefixes=code_prefixes,
        concept_sectors=concept_sectors,
        regime=regime,
        eval_date=eval_d,
        macro_bias=macro_bias,
        sector_fusion=sector_fusion,
        macro_regime=macro_regime,
        smooth_regime=smooth_regime,
        use_vol_switch=use_vol_switch,
        use_conditional_switch=use_conditional_switch,
        state_models=state_models,
        neutralize_qlib=neutralize_qlib,
        conditional_market_env=conditional_market_env,
        ranking_mode="qlib_primary",
        decision_day_filter=decision_day_filter,
    )
    if result.get("status") == "error":
        raise RuntimeError(result.get("error") or "选股失败")
    shadow_observations = []
    main_top5 = [row.get("code") for row in result.get("candidates", [])[:5] if row.get("code")]
    for item in shadow_models or []:
        model = tuple(item.get("model") or [])
        if len(model) != 2:
            continue
        try:
            _, scores = screener.get_qlib_predictions(
                as_of=eval_d, pool=pool, qlib_model=model,
                neutralize_qlib=neutralize_qlib,
            )
            ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
            shadow_top5 = [code for code, _ in ranked[:5]]
            denominator = min(5, len(main_top5), len(shadow_top5))
            shadow_observations.append({
                **item,
                "status": "completed",
                "changes_selection": False,
                "top5": [{"code": code, "score": float(score)} for code, score in ranked[:5]],
                "top5_overlap": len(set(main_top5) & set(shadow_top5)) / denominator if denominator else 0.0,
            })
        except Exception as error:
            logger.warning(f"[ShadowModel] {model[0]}/{model[1]} 预测失败: {error}")
            shadow_observations.append({
                **item,
                "status": "failed",
                "changes_selection": False,
                "error": str(error),
            })
    result["shadow_model_observations"] = shadow_observations
    result["warnings"] = profile_warnings
    from core.domain.portfolio.decision import decision_fingerprint
    actual_eval_date = eval_d
    score_context = (result.get("score_processing") or {}).get("context") or {}
    actual_prediction_date = str(
        score_context.get("prediction_date") or score_context.get("ref_date") or ""
    )[:10]
    if actual_prediction_date != actual_eval_date.isoformat():
        raise RuntimeError(
            f"Qlib预测日期 {actual_prediction_date or '未知'} 与决策日 "
            f"{actual_eval_date.isoformat()} 不一致，已拒绝保存结果"
        )
    l0_stage = (result.get("stages") or {}).get("l0") or {}
    coverage_count = int(result.get("qlib_coverage") or 0)
    universe_count = int(l0_stage.get("input_count") or 0)
    actual_model_id = score_context.get("model_id")
    if not actual_model_id and qlib_model and len(qlib_model) == 2:
        actual_model_id = f"{qlib_model[0]}/{qlib_model[1]}"
    prediction = {
        "decision_date": actual_eval_date.isoformat(),
        "prediction_date": actual_prediction_date,
        "fresh": True,
        "model_id": actual_model_id,
        "coverage_count": coverage_count,
        "universe_count": universe_count,
        "coverage_rate": round(coverage_count / universe_count, 6) if universe_count else None,
    }
    result["prediction"] = prediction
    request_params = {
        "pool": pool, "top_n": top_n, "debate_top": debate_top, "debate_agents": debate_agents,
        "skip_debate": skip_debate, "skip_l0": skip_l0, "skip_l1": skip_l1, "skip_l2": skip_l2,
        "skip_tech": skip_tech, "skip_qlib": skip_qlib, "skip_sent": skip_sent,
        "skip_fundamental": skip_fundamental, "skip_flow": skip_flow,
        "skip_north_flow": skip_north_flow, "skip_size_turnover": skip_size_turnover,
        "skip_momentum": skip_momentum, "skip_volatility": skip_volatility,
        "skip_illiq": skip_illiq, "skip_overnight_gap": skip_overnight_gap,
        "skip_wb_industry": skip_wb_industry, "skip_wb_divergence": skip_wb_divergence,
        "skip_wb_moneyflow": skip_wb_moneyflow, "wb_industry_n": wb_industry_n,
        "wb_divergence_n": wb_divergence_n, "wb_moneyflow_n": wb_moneyflow_n,
        "illiq_n": illiq_n, "overnight_gap_n": overnight_gap_n,
        "source_weights": source_weights, "tech_preset": tech_preset, "qlib_model": qlib_model,
        "hot_sector_count": hot_sector_count, "custom_sectors": custom_sectors,
        "code_prefixes": code_prefixes, "concept_sectors": concept_sectors, "regime": regime,
        "eval_date": actual_eval_date.isoformat(), "macro_bias": macro_bias,
        "sector_fusion": sector_fusion, "macro_regime": macro_regime,
        "smooth_regime": smooth_regime, "use_vol_switch": use_vol_switch,
        "use_conditional_switch": use_conditional_switch,
        "state_models": state_models, "neutralize_qlib": neutralize_qlib,
        "decision_day_filter": decision_day_filter,
        "shadow_models": shadow_models or [],
    }
    snapshot = {
        "job_id": job_id,
        "eval_date": actual_eval_date.isoformat(),
        "prediction": prediction,
        "request": request_params,
        "candidates": result.get("candidates", []),
    }
    result["pool"] = pool
    result["eval_date"] = actual_eval_date.isoformat()
    result["data_cutoff"] = actual_prediction_date
    result["request_params"] = request_params
    score_processing = dict(result.get("score_processing") or {})
    score_processing.setdefault("neutralize_qlib", bool(neutralize_qlib))
    score_processing.setdefault("percentile_mapping", True)
    score_processing.update({
        "neutralize_qlib_requested": bool(neutralize_qlib),
        "ranking_mode": "qlib_primary",
        "ranking_source": "qlib_score",
        "auxiliary_only": ["fundamental", "flow", "north_flow", "sentiment", "etf_flow"],
        "auxiliary_affects_rank": False,
        "macro_sector_bias": bool(macro_bias),
        "macro_regime": bool(macro_regime),
        "decision_day_filter": bool(decision_day_filter),
    })
    result["score_processing"] = score_processing
    requested_model_routing = {
        "mode": "volatility" if use_vol_switch else "regime",
        "explicit": bool(state_models),
        "state_models": state_models or {},
        "qlib_model": qlib_model,
        "conditional_switch": bool(use_conditional_switch) if use_vol_switch else None,
    }
    result["snapshot_fingerprint"] = decision_fingerprint(snapshot)
    result["provenance"] = {
        "eval_date": actual_eval_date.isoformat(),
        "requested_eval_date": requested_eval_d.isoformat() if requested_eval_d else None,
        "actual_prediction_date": actual_prediction_date,
        "data_cutoff": actual_prediction_date,
        "prediction": prediction,
        "request_params": request_params,
        "requested_model_routing": requested_model_routing,
        "runtime_model_routing": result.get("model_routing"),
        "snapshot_fingerprint": result["snapshot_fingerprint"],
    }
    if strategy_profile:
        cfg = strategy_profile["config"]
        profile_info = {
            "profile_id": strategy_profile["profile_id"],
            "version": strategy_profile["version"],
            "status": strategy_profile["status"],
            "fingerprint": strategy_profile["fingerprint"],
            "data_fingerprint": operational_snapshot["snapshot_id"],
            "profile_data_snapshot": cfg["data"].get("snapshot_id"),
            "model_fingerprint": cfg.get("fingerprints", {}).get("model_refs"),
            "factor_fingerprint": cfg.get("fingerprints", {}).get("factor"),
            "research_only": bool(cfg.get("factors", {}).get("research_only")),
            "promotable": not bool(cfg.get("factors", {}).get("research_only")),
        }
        result["strategy_profile"] = profile_info
        result["provenance"]["strategy_profile"] = profile_info
        gate_model = cfg.get("score", {}).get("absolute_gate_model")
        if gate_model:
            from core.domain.screening.absolute_gate import apply_absolute_gate
            raw_scores = {
                row.get("code"): row.get("raw_model_score")
                for row in result.get("candidates", []) if row.get("code")
            }
            gate = apply_absolute_gate(gate_model, raw_scores)
            for row in result.get("candidates", []):
                row["absolute_gate_v1"] = gate["predictions"].get(row.get("code"))
            result["absolute_gate_v1"] = {key: value for key, value in gate.items() if key != "predictions"}
        else:
            result["absolute_gate_v1"] = {
                "version": "absolute_gate_v1", "shadow": True,
                "status": "untrained", "changes_ranking": False,
            }
    else:
        result["strategy_profile"] = {"legacy": True, "promotable": False}
    result["result_id"] = job_id
    with _lock:
        result["started_at"] = _results.get(job_id, {}).get("started_at") or datetime.now().isoformat()
        completed_result = copy.deepcopy(result)
        completed_result["completed_at"] = datetime.now().isoformat()
    from core.data import ScreeningResultRepo
    ScreeningResultRepo.complete(job_id, _worker_id, jsonable_encoder(completed_result))
    with _lock:
        _results[job_id] = completed_result
        _cleanup_old_results()
        _history.append({
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "pool": pool,
            "candidate_count": len(result.get("candidates", [])),
            "recommended_count": len(result.get("recommended", [])),
            "status": result.get("status", "unknown"),
        })
        if len(_history) > 200:
            _history[:] = _history[-200:]


@router.post("/run")
async def run_screening(
    pool: str = Body("stable_history_500d"),
    top_n: int = Body(50),
    debate_top: int = Body(5),
    debate_agents: int = Body(6),
    skip_debate: bool = Body(False),
    skip_l0: bool = Body(False),
    skip_l1: bool = Body(False),
    skip_l2: bool = Body(False),
    skip_tech: bool = Body(False),
    skip_qlib: bool = Body(False),
    skip_sent: bool = Body(False),
    skip_fundamental: bool = Body(False),
    skip_flow: bool = Body(False),
    skip_north_flow: bool = Body(False),
    skip_size_turnover: bool = Body(False),
    skip_momentum: bool = Body(False),
    skip_volatility: bool = Body(False),
    skip_illiq: bool = Body(False),
    skip_overnight_gap: bool = Body(False),
    skip_wb_industry: bool = Body(False),
    skip_wb_divergence: bool = Body(False),
    skip_wb_moneyflow: bool = Body(False),
    wb_industry_n: int = Body(20),
    wb_divergence_n: int = Body(10),
    wb_moneyflow_n: int = Body(5),
    illiq_n: int = Body(20),
    overnight_gap_n: int = Body(20),
    source_weights: Optional[dict] = Body(None),
    tech_preset: str = Body("all"),
    qlib_model: Optional[list] = Body(None),
    async_mode: bool = Body(True),
    hot_sector_count: int = Body(10),
    custom_sectors: Optional[list] = Body(None),
    code_prefixes: Optional[list] = Body([
        "600", "601", "603", "605", "000", "001", "002", "003",
    ]),
    concept_sectors: Optional[list] = Body(None),
    regime: Optional[str] = Body(None),
    eval_date: Optional[str] = Body(None),
    macro_bias: bool = Body(False),
    sector_fusion: bool = Body(False),
    macro_regime: bool = Body(False),
    smooth_regime: bool = Body(False),
    use_vol_switch: bool = Body(False),
    use_conditional_switch: bool = Body(True),
    state_models: Optional[dict] = Body(None),
    neutralize_qlib: bool = Body(False),
    decision_day_filter: bool = Body(True),
    profile_id: Optional[str] = Body(None),
    profile_version: Optional[int] = Body(None),
    profile_overrides: Optional[dict] = Body(None),
):
    if skip_qlib or skip_l0:
        raise HTTPException(422, "当前选股必须先按Qlib预测排序，不能跳过Qlib排序阶段")
    from core.domain.regime.model_routing import validate_state_models
    from backend.service.strategy_profile import StrategyProfileService
    try:
        strategy_profile = StrategyProfileService.resolve(profile_id, profile_version) if profile_id else StrategyProfileService.champion()
        if strategy_profile:
            research_only = bool((strategy_profile.get("config") or {}).get("factors", {}).get("research_only"))
            if strategy_profile["status"] not in {"challenger", "paper", "champion"} and not research_only:
                raise ValueError(f"策略状态 {strategy_profile['status']} 不允许当前操作")
            locked = StrategyProfileService.bind_operation(
                strategy_profile, "screening", profile_overrides,
            )
            pool = locked["pool"]
            top_n = locked["top_n"]
            qlib_model = locked["qlib_model"]
            code_prefixes = locked["code_prefixes"]
            macro_bias = locked["macro_bias"]
            sector_fusion = locked["sector_fusion"]
            macro_regime = locked["macro_regime"]
            smooth_regime = locked["smooth_regime"]
            use_vol_switch = locked["use_vol_switch"]
            state_models = locked["state_models"]
            neutralize_qlib = locked["neutralize_qlib"]
            decision_day_filter = locked["decision_day_filter"]
            shadow_models = locked["shadow_models"]
            from core.shared.trading_time import get_latest_trading_day
            from datetime import date as dt_date
            requested_date = dt_date.fromisoformat(eval_date) if eval_date else None
            decision_date = get_latest_trading_day(requested_date) if requested_date else get_latest_trading_day()
            if requested_date and decision_date != requested_date:
                raise ValueError("策略版本选股不允许决策日回退，请选择有效交易日")
            first_date = str((strategy_profile["config"].get("availability") or {}).get("first_decision_date") or "")[:10]
            if first_date and decision_date.isoformat() < first_date:
                raise ValueError(f"该策略最早可运行日期为 {first_date}")
            model_ref, scheduled_model = StrategyProfileService.model_ref_for_date(
                strategy_profile, decision_date,
                require_schedule=strategy_profile["status"] == "challenger",
            )
            qlib_model = [model_ref["experiment_id"], model_ref["run_id"]]
            cfg = strategy_profile["config"]
            from core.adapters.qlib.model_selection import get_model_for_factor_set
            checked = get_model_for_factor_set(
                model_ref["experiment_id"], model_ref["run_id"],
                (cfg.get("factors") or {}).get("keys") or [], decision_date.isoformat(),
                "a500", allow_test_window=scheduled_model,
            )
            if int(checked.get("label_horizon") or 0) != 10:
                raise ValueError("当前主模型的预测周期不是10日")
            if checked.get("artifact_fingerprint") != model_ref.get("artifact_fingerprint"):
                raise ValueError("当前主模型内容指纹已变化")
            for shadow_ref in (cfg.get("model") or {}).get("shadow_refs") or []:
                checked_shadow = get_model_for_factor_set(
                    shadow_ref["experiment_id"], shadow_ref["run_id"],
                    shadow_ref.get("factor_keys") or [], decision_date.isoformat(), "a500",
                )
                if int(checked_shadow.get("label_horizon") or 0) != 10:
                    raise ValueError(f"影子模型 {shadow_ref['run_id']} 的预测周期不是10日")
                if checked_shadow.get("artifact_fingerprint") != shadow_ref.get("artifact_fingerprint"):
                    raise ValueError(f"影子模型 {shadow_ref['run_id']} 内容指纹已变化")
        else:
            shadow_models = []
    except ValueError as error:
        raise HTTPException(422, str(error))
    try:
        state_models = validate_state_models(state_models, use_vol_switch)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if pool in ("concept_rotation", "concept"):
        from backend.service.concept_data import require_fresh_concept_snapshot
        try:
            await anyio.to_thread.run_sync(require_fresh_concept_snapshot)
        except ValueError as e:
            raise HTTPException(409, str(e))
    if state_models and use_vol_switch:
        try:
            from core.data import SettingsRepo
            instruments = {
                "csi300": "csi300", "a500": "a500", "stable_history_500d": "a500", "small_cap": "small_cap",
                "all": "all", "non_csi300": "all",
            }.get(pool, "all")
            SettingsRepo.set("lgb_lstm_prewarm", {
                "models": {key: list(value) for key, value in state_models.items()},
                "pools": [instruments],
            })
        except Exception as e:
            logger.warning(f"[LGB+LSTM] 保存预热配置失败: {e}")
    job_id = uuid.uuid4().hex[:12]
    from core.data import ScreeningResultRepo
    try:
        ScreeningResultRepo.create_pending(job_id, pool)
    except Exception as error:
        logger.error(f"[Screening] 任务 {job_id} 初始化持久化失败: {error}")
        raise HTTPException(503, "选股任务状态存储不可用，未启动任务")
    with _lock:
        _results[job_id] = {"job_id": job_id, "status": "pending", "started_at": datetime.now().isoformat(),
                            "progress": {"stage": "pending", "current": 0, "total": 0}}
        _cleanup_old_results()

    if async_mode:
        try:
            future = thread_pool.default.submit(_run_screening_task,
                                 job_id, pool, top_n, debate_top,
                                 debate_agents,
                                 skip_debate, skip_l0, skip_l1, skip_l2,
                                 skip_tech, skip_qlib, skip_sent,
                                 skip_fundamental, skip_flow,
                                 skip_north_flow, skip_size_turnover,
                                 skip_momentum, skip_volatility,
                                 skip_illiq, skip_overnight_gap,
                                 skip_wb_industry, skip_wb_divergence, skip_wb_moneyflow,
                                 wb_industry_n, wb_divergence_n, wb_moneyflow_n,
                                 illiq_n, overnight_gap_n,
                                 source_weights, tech_preset, qlib_model,
                                 hot_sector_count, custom_sectors,
                                 code_prefixes, concept_sectors, regime, eval_date, macro_bias,
                                 sector_fusion, macro_regime, smooth_regime, use_vol_switch,
                                 use_conditional_switch,
                                 state_models, neutralize_qlib, decision_day_filter, shadow_models, strategy_profile)
        except Exception as error:
            try:
                ScreeningResultRepo.fail(job_id, _worker_id, error)
            except Exception as persist_error:
                logger.error(f"[Screening] 任务 {job_id} 提交失败状态持久化失败: {persist_error}")
            with _lock:
                _results[job_id] = {"job_id": job_id, "status": "error", "error": str(error)}
            raise HTTPException(503, "选股任务无法启动") from error
        def _on_screen_done(fut):
            try:
                exc = fut.exception()
                if exc is not None:
                    logger.error(f"[Screening] 异步任务 {job_id} 异常: {exc}", exc_info=exc)
            except Exception:
                pass
        future.add_done_callback(_on_screen_done)
        return {"data": {"job_id": job_id, "status": "pending"}}
    else:
        result = await anyio.to_thread.run_sync(
            _run_screening_task, job_id, pool, top_n, debate_top, debate_agents,
            skip_debate, skip_l0, skip_l1, skip_l2, skip_tech, skip_qlib, skip_sent,
            skip_fundamental, skip_flow, skip_north_flow, skip_size_turnover,
            skip_momentum, skip_volatility, skip_illiq, skip_overnight_gap,
            skip_wb_industry, skip_wb_divergence, skip_wb_moneyflow,
            wb_industry_n, wb_divergence_n, wb_moneyflow_n, illiq_n, overnight_gap_n,
            source_weights, tech_preset, qlib_model, hot_sector_count, custom_sectors,
            code_prefixes, concept_sectors, regime, eval_date, macro_bias, sector_fusion,
            macro_regime, smooth_regime, use_vol_switch, use_conditional_switch,
            state_models, neutralize_qlib, decision_day_filter, shadow_models, strategy_profile,
        )
        return {"data": result}


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    from core.data import ScreeningResultRepo
    try:
        task = ScreeningResultRepo.get(job_id)
    except Exception as error:
        logger.error(f"[Screening] 读取任务 {job_id} 持久化状态失败: {error}")
        raise HTTPException(503, "选股任务状态存储不可用")
    if not task:
        raise HTTPException(404, f"任务不存在: {job_id}")
    if task.get("status") in ("pending", "running"):
        task = ScreeningResultRepo.interrupt_if_stale(job_id)
    if task.get("status") == "done":
        result = task.get("result_json")
        if (not isinstance(result, dict) or result.get("status") != "done"
                or result.get("job_id") != job_id or result.get("result_id") != job_id):
            raise HTTPException(500, "选股任务终态与持久化结果不一致")
        return {"data": result}
    if task.get("status") in ("pending", "running"):
        live = None
        try:
            from core.domain.screening._screener import get_screener
            live = get_screener()._jobs.get(job_id)
        except Exception:
            pass
        if live and live.get("status") in ("pending", "running"):
            return {"data": {
                "job_id": job_id,
                "status": live.get("status"),
                "progress": copy.deepcopy(live.get("progress") or {}),
            }}
        if task.get("status") == "running" and task.get("worker_id") != _worker_id:
            return {"data": {"job_id": job_id, "status": "running", "progress": task.get("progress") or {}}}
        if task.get("status") == "running":
            return {"data": {
                "job_id": job_id, "status": "finalizing",
                "progress": {"stage": "finalizing", "current": 0, "total": 0},
            }}
        return {"data": {"job_id": job_id, "status": "pending", "progress": task.get("progress") or {}}}
    return {"data": {
        "job_id": job_id, "status": task.get("status"), "error": task.get("error"),
        "progress": task.get("progress") or {}, "updated_at": task.get("updated_at"),
    }}


@router.get("/latest")
async def get_latest():
    from core.data import ScreeningResultRepo
    result = ScreeningResultRepo.latest()
    if not result:
        return {"data": None}
    if result.get("status") == "done":
        return {"data": result.get("result_json")}
    return {"data": {"job_id": result["job_id"], "status": result["status"], "error": result.get("error")}}


@router.get("/history")
async def get_history(days: int = 7):
    from core.data import ScreeningResultRepo
    return {"data": ScreeningResultRepo.list_history(days)}


@router.get("/hot-sectors")
async def get_hot_sectors(top_n: int = 5):
    """获取当前最活跃板块列表（不执行筛选）"""
    def _run():
        try:
            from core.data.fetcher import DataFetcher
            import akshare as ak

            fetcher = DataFetcher()

            try:
                perf = fetcher.get_sector_performance()
            except Exception:
                return {"data": {"sectors": [], "error": "板块数据暂不可用"}}

            if perf is None or perf.empty:
                return {"data": {"sectors": [], "error": "板块数据为空"}}

            if "涨跌幅" in perf.columns:
                perf = perf.copy()
                perf["abs_chg"] = perf["涨跌幅"].abs()
                perf = perf.sort_values("abs_chg", ascending=False)

            hot = perf.head(top_n)

            summary_map = {}
            try:
                summary = ak.stock_board_industry_summary_ths()
                if summary is not None and not summary.empty:
                    cols = summary.columns.tolist()
                    name_col = cols[1] if len(cols) > 1 else cols[0]
                    up_col = cols[6] if len(cols) > 6 else "上涨家数"
                    down_col = cols[7] if len(cols) > 7 else "下跌家数"
                    for _, row in summary.iterrows():
                        sname = str(row[name_col])
                        up = int(row.get(up_col, 0) or 0)
                        down = int(row.get(down_col, 0) or 0)
                        summary_map[sname] = up + down
            except Exception:
                pass

            sectors = []
            total_stocks = 0
            for _, row in hot.iterrows():
                name = row.get("板块名称", "")
                chg = round(float(row.get("涨跌幅", 0)), 2) if "涨跌幅" in perf.columns else 0
                count = summary_map.get(name, 0)
                total_stocks += count
                sectors.append({"name": name, "change": chg, "stock_count": count})

            return {"data": {"sectors": sectors, "total_stocks": total_stocks}}
        except Exception as e:
            return {"data": {"sectors": [], "error": str(e)}}
    return await anyio.to_thread.run_sync(_run)


@router.get("/sectors")
async def get_available_sectors():
    def _run():
        try:
            from core.domain.screening.stock_scanner import SectorScanner
            scanner = SectorScanner()
            df = scanner.get_all_sectors()
            if df is not None and not df.empty:
                col = "板块名称" if "板块名称" in df.columns else df.columns[0]
                sector_names = sorted(set(str(s) for s in df[col].tolist() if str(s).strip()))

                sector_counts = {}
                try:
                    import akshare as ak
                    summary = ak.stock_board_industry_summary_ths()
                    if summary is not None and not summary.empty:
                        cols = summary.columns.tolist()
                        name_col = cols[1] if len(cols) > 1 else cols[0]
                        up_col = cols[6] if len(cols) > 6 else "上涨家数"
                        down_col = cols[7] if len(cols) > 7 else "下跌家数"
                        for _, row in summary.iterrows():
                            sname = str(row[name_col])
                            up = int(row.get(up_col, 0) or 0)
                            down = int(row.get(down_col, 0) or 0)
                            sector_counts[sname] = up + down
                except Exception:
                    pass

                return {"data": [{"name": s, "stock_count": sector_counts.get(s, 0)} for s in sector_names]}
        except Exception as e:
            logger.warning(f"获取板块列表失败: {e}")

        try:
            from core.data import Database
            if Database.is_available():
                rows = Database.fetchall(
                    "SELECT DISTINCT sector_name FROM stock_sectors "
                    "WHERE sector_type='industry' OR sector_type IS NULL ORDER BY sector_name")
                if rows:
                    sectors = [r["sector_name"] for r in rows]
                    return {"data": [{"name": s} for s in sectors]}
        except Exception:
            return {"data": [], "error": "板块数据不可用"}
    return await anyio.to_thread.run_sync(_run)


@router.get("/concept-sectors")
async def get_concept_sectors():
    """获取同花顺概念板块列表（~370个）"""
    def _run():
        import akshare as ak
        try:
            df = ak.stock_board_concept_name_ths()
            if df is not None and not df.empty:
                col = "name" if "name" in df.columns else df.columns[0]
                sector_names = sorted(set(str(s) for s in df[col].tolist() if str(s).strip()))

                sector_counts = {}
                try:
                    summary = ak.stock_board_concept_summary_ths()
                    if summary is not None and not summary.empty:
                        cols = summary.columns.tolist()
                        name_col = cols[1] if len(cols) > 1 else cols[0]
                        count_col = "成分股数量"
                        for _, row in summary.iterrows():
                            sname = str(row[name_col])
                            cnt = int(row.get(count_col, 0) or 0)
                            sector_counts[sname] = cnt
                except Exception:
                    pass

                cache_loaded = 0
                try:
                    from backend.service.concept_data import require_fresh_concept_snapshot
                    require_fresh_concept_snapshot()
                    from core.data import Database
                    if Database.is_available():
                        rows = Database.fetchall(
                            "SELECT sector_name, COUNT(*) as cnt FROM stock_sectors "
                            "WHERE sector_type='concept' GROUP BY sector_name")
                        for r in rows:
                            sname = r["sector_name"]
                            cnt = int(r["cnt"])
                            if sname not in sector_counts and cnt > 0:
                                sector_counts[sname] = cnt
                                cache_loaded += 1
                except Exception:
                    pass

                if cache_loaded:
                    logger.info(f"概念板块数量: summary={len([v for v in sector_counts.values() if v>0])}个 (含MySQL补充{cache_loaded}个)")

                return {"data": [{"name": s, "stock_count": sector_counts.get(s, 0)} for s in sector_names]}
        except Exception as e:
            logger.warning(f"获取概念板块列表失败: {e}")
        return {"data": [], "error": "概念板块数据不可用"}
    return await anyio.to_thread.run_sync(_run)


@router.get("/retail-hot-sectors")
async def get_retail_hot_sectors(top_n: int = 10, days: int = 1):
    """散户热门板块排行 — 概念板块按散户资金结构+换手+涨幅加权"""
    from backend.service.concept_data import require_fresh_concept_snapshot
    try:
        await anyio.to_thread.run_sync(require_fresh_concept_snapshot)
    except ValueError as e:
        raise HTTPException(409, str(e))

    def _run():
        import pandas as pd
        from core.data import Database
        from core.domain.screening.retail_heat import compute_retail_heat

        try:
            if not Database.is_available():
                return {"data": [], "error": "数据库不可用"}

            latest = Database.fetchone("SELECT MAX(trade_date) AS d FROM daily_quotes")
            if not latest or not latest.get("d"):
                return {"data": [], "error": "无行情数据"}
            end = str(latest["d"])

            trade_dates = [r["trade_date"] for r in Database.fetchall(
                "SELECT DISTINCT trade_date FROM capital_flow_daily ORDER BY trade_date DESC LIMIT %s",
                (max(days, 1),))]
            if not trade_dates:
                return {"data": [], "error": "无资金流数据"}
            ph = ",".join(["%s"] * len(trade_dates))
            flow = pd.DataFrame(Database.fetchall(
                "SELECT code, SUM(COALESCE(sm_net_flow,0)) AS sm_net_flow, "
                "SUM(COALESCE(md_net_flow,0)) AS md_net_flow, "
                "SUM(COALESCE(lg_net_flow,0)) AS lg_net_flow, "
                "SUM(COALESCE(elg_net_flow,0)) AS elg_net_flow, "
                "SUM(COALESCE(main_net_flow,0)) AS main_net_flow "
                f"FROM capital_flow_daily WHERE trade_date IN ({ph}) GROUP BY code",
                trade_dates))
            if flow.empty:
                return {"data": [], "error": "无资金流数据"}

            basic = pd.DataFrame(Database.fetchall(
                "SELECT code, circ_mv, turnover_rate FROM daily_basic WHERE trade_date=%s", (end,)))
            # daily_quotes.pct_chg 采集源常填 0, 用当日/前一日 close 自算涨跌幅
            cur = pd.DataFrame(Database.fetchall(
                "SELECT code, close FROM daily_quotes WHERE trade_date=%s", (end,)))
            prev_rows = Database.fetchall(
                "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date < %s "
                "ORDER BY trade_date DESC LIMIT 1", (end,))
            if prev_rows:
                prev = pd.DataFrame(Database.fetchall(
                    "SELECT code, close AS prev_close FROM daily_quotes WHERE trade_date=%s",
                    (prev_rows[0]["trade_date"],)))
                quotes = cur.merge(prev, on="code", how="left")
                quotes["pct_chg"] = (quotes["close"] - quotes["prev_close"]) / quotes["prev_close"] * 100
            else:
                quotes = cur
                quotes["pct_chg"] = 0.0
            sector_rows = Database.fetchall(
                "SELECT code, sector_name FROM stock_sectors WHERE sector_type='concept'")

            stock = flow.merge(basic, on="code", how="left").merge(quotes, on="code", how="left")
            sector_map = {}
            for r in sector_rows:
                sector_map.setdefault(r["code"], []).append(r["sector_name"])

            result = compute_retail_heat(stock, sector_map).head(top_n)
            result = result.where(pd.notna(result), None)
            return {"data": {"trade_date": str(end), "sectors": result.to_dict(orient="records")}}
        except Exception as e:
            logger.warning(f"散户热门板块查询失败: {e}")
            return {"data": {"trade_date": None, "sectors": []}, "error": str(e)}
    return await anyio.to_thread.run_sync(_run)


@router.post("/set-predictions")
async def set_predictions(predictions: dict = Body(...), authorization: str | None = Header(None)):
    from backend.api.settings import _verify_admin
    if not _verify_admin(authorization):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    from core.domain.screening._screener import get_screener
    screener = get_screener()
    screener.set_qlib_predictions(predictions)
    return {"data": {"count": len(predictions)}}


# ─── 周频调仓建议 ───

_rebalance_jobs: dict = {}
_rebalance_lock = threading.Lock()
_MAX_REBALANCE_JOBS = 50

@router.post("/weekly-advice")
async def generate_weekly_advice(
    pool: str = Body("stable_history_500d"),
    sectors: Optional[list] = Body(None),
    code_prefixes: Optional[list] = Body(None),
    top_n: int = Body(5),
    sizing: str = Body("score_weighted"),
    eval_date: Optional[str] = Body(None),
    async_mode: bool = Body(True),
    commission_rate: Optional[float] = Body(None),
    slippage_rate: Optional[float] = Body(None),
    stamp_duty: Optional[float] = Body(None),
    stop_loss_pct: Optional[float] = Body(None),
    max_position_pct: Optional[float] = Body(None),
    max_sector_pct: Optional[float] = Body(None),
):
    from datetime import date as dt_date
    from core.shared.trading_time import get_latest_trading_day

    latest_day = get_latest_trading_day()
    try:
        eval_d = dt_date.fromisoformat(eval_date) if eval_date else latest_day
    except (TypeError, ValueError):
        raise HTTPException(422, "eval_date 必须为 YYYY-MM-DD") from None
    if eval_d != latest_day:
        raise HTTPException(409, "历史日期缺少当时账户现金与持仓快照，不能生成当前账本调仓建议")
    if not async_mode:
        def _sync_advice():
            from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor
            from backend.service.local_ledger import LocalLedgerService
            advisor = WeeklyRebalanceAdvisor(
                top_n=top_n, pool=pool, sizing=sizing,
                **{k: v for k, v in {
                    "stop_loss_pct": stop_loss_pct,
                    "max_position_pct": max_position_pct,
                    "max_sector_pct": max_sector_pct,
                    "commission_rate": commission_rate,
                    "slippage_rate": slippage_rate,
                    "stamp_duty": stamp_duty,
                }.items() if v is not None},
            )
            account_snapshot = LocalLedgerService.get_advice_snapshot()
            report = advisor.generate_advice(
                pool=pool, sectors=sectors, code_prefixes=code_prefixes,
                sizing=sizing, eval_date=eval_d,
                account_snapshot=account_snapshot,
            )
            if "error" not in report:
                report["account_source"] = "local_ledger"
                report["account_version"] = account_snapshot["account"]["version"]
                report["execution"] = {"executable": False,
                                       "reason": "该建议仅供预览，请从本地账本调仓计划生成可执行信号"}
            return report
        try:
            report = await anyio.to_thread.run_sync(
                run_with_qlib_read_access, _sync_advice,
            )
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        if "error" in report:
            raise HTTPException(400, report["error"])
        return {"data": report}

    job_id = uuid.uuid4().hex[:12]
    with _rebalance_lock:
        _rebalance_jobs[job_id] = {"status": "running", "progress": "初始化...", "started_at": datetime.now().isoformat()}
        # 防止内存泄漏: 保留最近 50 条
        if len(_rebalance_jobs) > _MAX_REBALANCE_JOBS:
            oldest = sorted(_rebalance_jobs.keys(), key=lambda k: _rebalance_jobs[k].get("started_at", ""))[:len(_rebalance_jobs) - _MAX_REBALANCE_JOBS]
            for k in oldest:
                del _rebalance_jobs[k]

    def _cleanup():
        t = threading.Timer(7200, lambda: _rebalance_jobs.pop(job_id, None))
        t.daemon = True; t.start()

    def _run():
        try:
            from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor
            from backend.service.local_ledger import LocalLedgerService
            advisor = WeeklyRebalanceAdvisor(
                top_n=top_n, pool=pool, sizing=sizing,
                **{k: v for k, v in {
                    "stop_loss_pct": stop_loss_pct,
                    "max_position_pct": max_position_pct,
                    "max_sector_pct": max_sector_pct,
                    "commission_rate": commission_rate,
                    "slippage_rate": slippage_rate,
                    "stamp_duty": stamp_duty,
                }.items() if v is not None},
            )
            with _rebalance_lock:
                _rebalance_jobs[job_id]["progress"] = "正在加载 Qlib 预测..."
            account_snapshot = LocalLedgerService.get_advice_snapshot()
            report = advisor.generate_advice(
                pool=pool, sectors=sectors, code_prefixes=code_prefixes,
                sizing=sizing, eval_date=eval_d,
                account_snapshot=account_snapshot,
            )
            if "error" not in report:
                report["account_source"] = "local_ledger"
                report["account_version"] = account_snapshot["account"]["version"]
                report["execution"] = {"executable": False,
                                       "reason": "该建议仅供预览，请从本地账本调仓计划生成可执行信号"}
            with _rebalance_lock:
                if "error" in report:
                    _rebalance_jobs[job_id] = {"status": "error", "error": report["error"]}
                else:
                    _rebalance_jobs[job_id] = {"status": "done", "result": report}
                    _rebalance_jobs[job_id]["text_report"] = advisor.format_text_report(report)
        except Exception as e:
            logger.error(f"周频调仓失败: {e}", exc_info=True)
            with _rebalance_lock:
                _rebalance_jobs[job_id] = {"status": "error", "error": str(e)}
        _cleanup()

    thread_pool.default.submit(run_with_qlib_read_access, _run)
    return {"data": {"job_id": job_id, "status": "pending"}}


@router.get("/weekly-advice/{job_id}")
async def get_weekly_advice_result(job_id: str):
    with _rebalance_lock:
        job = _rebalance_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在或已过期")
    return {"data": job}


# ═══ 选股结果保存 ═══

@router.post("/{job_id}/save")
async def save_screening_result(
    job_id: str,
    save_name: str = Body(...),
    note: str = Body(""),
):
    import anyio
    from backend.service.rebalance import RebalanceService

    def _save():
        from core.data import ScreeningResultRepo
        task = ScreeningResultRepo.get(str(job_id))
        screening_result = copy.deepcopy(task.get("result_json")) if task else None
        if (not screening_result or screening_result.get("status") != "done"
                or task.get("status") != "done"
                or screening_result.get("result_id") != str(job_id)):
            raise HTTPException(409, "选股任务尚未通过服务端最终校验，不能保存为正式快照")
        return RebalanceService.save_screening(job_id, save_name, note, screening_result=screening_result)

    result = await anyio.to_thread.run_sync(_save)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"data": result}


@router.get("/saves")
async def list_saved_screenings(
    days: int = 30,
    pool: Optional[str] = None,
    limit: int = 20,
):
    from backend.service.rebalance import RebalanceService
    rows = RebalanceService.list_saves(days=days, pool=pool, limit=limit)
    return {"data": rows}


@router.get("/saves/{save_id}")
async def get_saved_screening(save_id: int):
    from backend.service.rebalance import RebalanceService
    r = RebalanceService.get_save(save_id)
    if not r:
        raise HTTPException(404, f"保存 #{save_id} 不存在")
    return {"data": r}


@router.delete("/saves/{save_id}")
async def delete_saved_screening(save_id: int):
    from backend.service.rebalance import RebalanceService
    RebalanceService.delete_save(save_id)
    return {"data": {"status": "ok"}}
