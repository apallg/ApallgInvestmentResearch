"""DEPRECATED: 已由 backend/api/qlib/train_v2.py 和 backend/service/training.py 取代。"""
import anyio
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from backend.api.qlib.common import (  # noqa: F403
    logger,
    _get_qlib_data_dir,
    is_excluded, EXCLUDED_PREFIXES, B_SHARE_PREFIXES,
    fetch_tencent_kline_raw, task_manager, thread_pool, uuid,
)
from backend.service.qlib_job_store import (
    train_jobs as _train_jobs,
    train_lock as _train_lock,
    sync_lock as _sync_lock,
    collect_job_id as _collect_job_id,
)
from backend.service.instrument_resolver import (
    resolve_instruments as _resolve_instruments,
    build_filtered_instruments,
    filter_instrument_selection,
)
from backend.service.model_presenter import resolve_factor_label, make_display_name
from backend.service.optuna_store import (
    OPTUNA_DIR as _OPTUNA_DIR,
    save_optuna_result as _save_optuna_result,
    load_optuna_result,
)
from backend.service.qlib_path_guard import assert_run_path
from backend.api.qlib.models import TrainRequest, OptunaSearchRequest, RollingTrainRequest, ModelToggleRequest

router = APIRouter()

@router.get("/models")
async def list_models():
    """列出可用的模型类型"""
    from core.adapters.qlib.models import SUPPORTED_MODELS, DEFAULT_PARAMS
    models = []
    for name, (cls_name, module) in SUPPORTED_MODELS.items():
        models.append({
            "name": name,
            "class": cls_name,
            "module": module,
            "default_params": DEFAULT_PARAMS.get(name, {}),
        })
    return {"data": models}
@router.post("/models/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """启动模型训练（异步）"""
    job_id = str(uuid.uuid4())[:8]

    def _do_train():
        try:
            from core.adapters.qlib.models import QlibModelManager
            manager = QlibModelManager()

            factor_set = req.factor_set
            custom_factor_keys = req.custom_factor_keys
            # 动态因子池绑定 (报告修正6): 取池活跃因子替代默认 Alpha158 全量
            if req.use_factor_pool:
                from core.domain.factors.registry import get_pool
                # 按训练截止日查对应时期的活跃因子集, 避免反向数据窥探
                # (真正滚动训练应每 fold 独立筛选, 此为当前折中方案)
                active = get_pool().get_active_factors(category="white_box", as_of=req.train_end)
                pool_keys = list(active.keys())
                if pool_keys:
                    # 训练前去同源: 剔除高相关冗余因子 (报告修正2 覆盖训练输入)
                    try:
                        from core.domain.factors.validation import OverfittingChecker
                        variants = [OverfittingChecker()._pool_entry_to_variant(k, active[k]) for k in pool_keys]
                        kept = OverfittingChecker().deduplicate_variants(
                            variants, pool=req.instruments if isinstance(req.instruments, str) else "all",
                            corr_thr=0.8)
                        if kept:
                            logger.info(f"[Train] 因子池去同源: {len(pool_keys)} → {len(kept)}")
                            pool_keys = kept
                    except Exception as de:
                        logger.warning(f"[Train] 因子池去同源失败, 用原始池: {de}")
                    factor_set = "WhiteBoxHandler"
                    custom_factor_keys = pool_keys
                    logger.info(f"[Train] 使用动态因子池: {len(pool_keys)} 个活跃因子 {pool_keys}")
                else:
                    logger.warning("[Train] use_factor_pool=True 但因子池无活跃白盒因子, 回退默认 factor_set")

            instruments = _resolve_instruments(req.instruments)
            if req.code_prefixes:
                instruments = filter_instrument_selection(instruments, req.code_prefixes)
                logger.info(f"[Train] 前缀过滤: {req.code_prefixes} → {len(instruments)} stocks")

            # 精选因子集 (use_curated): 仅在未使用因子池时生效，避免覆盖
            if req.use_curated and not req.use_factor_pool:
                from core.domain.factors.registry import get_pool
                curated_keys = get_pool().get_curated_factors(req.use_curated)
                if curated_keys:
                    factor_set = "WhiteBoxHandler"
                    custom_factor_keys = curated_keys
                    logger.info(f"[Train] 使用精选因子集 '{req.use_curated}': {len(curated_keys)} 个因子")
                else:
                    logger.warning(f"[Train] 精选因子集 '{req.use_curated}' 为空, 回退默认")

            # LGB + LSTM 融合 (0.5/0.5 秩融合)
            if req.train_lgb_lstm:
                logger.info("[Train] LGB+LSTM 融合模式 (0.5/0.5)")
                if not custom_factor_keys:
                    logger.warning("[Train] LGB+LSTM 融合建议配合 use_curated 或 use_factor_pool")
                from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion
                _fr = train_lgb_lstm_fusion(
                    factor_keys=custom_factor_keys,
                    instruments=instruments,
                    train_period=(req.train_start, req.train_end),
                    valid_period=(req.valid_start, req.valid_end),
                    test_period=(req.test_start, req.test_end),
                    label_horizon=req.label_horizon,
                    label_mode=req.label_mode,
                    label_norm=req.label_norm,
                    fusion_weight=0.5,
                    param_profile=req.param_profile if req.param_profile != "optuna" else "conservative",
                )
                result = {
                    "predictions": _fr["fused_pred"],
                    "label": _fr["label"],
                    "ic_mean": _fr["fused"]["ic_mean"],
                    "icir": _fr["fused"]["icir"],
                    "lgb_ic_mean": _fr["lgb"]["ic_mean"],
                    "lstm_ic_mean": _fr["lstm"]["ic_mean"],
                    "degraded": [],
                }
                backtest_result = None
                if req.auto_backtest and result.get("predictions") is not None and result.get("label") is not None:
                    try:
                        from core.adapters.qlib.backtest import QlibBacktestRunner
                        runner = QlibBacktestRunner()
                        bt = runner.run_from_predictions(
                            predictions=result["predictions"],
                            label=result["label"],
                            topk=req.topk, n_drop=req.n_drop,
                            start_time=req.test_start, end_time=req.test_end,
                            freq=req.freq,
                        )
                        backtest_result = bt
                    except Exception as e:
                        logger.warning(f"[Train] 回测失败 (训练结果不受影响): {e}")
                _train_jobs[job_id] = {
                    "status": "completed",
                    "result": {
                        "ic_mean": result.get("ic_mean"),
                        "icir": result.get("icir"),
                        "lgb_ic_mean": result.get("lgb_ic_mean"),
                        "lstm_ic_mean": result.get("lstm_ic_mean"),
                        "degraded": result.get("degraded"),
                        "backtest": backtest_result,
                    },
                }
                try:
                    import redis as _redis, json as _json
                    _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                    _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
                except Exception:
                    pass
                return

            # 双模型训练
            if req.train_dual:
                logger.info(f"[Train] 双模型模式: LightGBM + XGBoost")
                if not custom_factor_keys:
                    logger.warning("[Train] 双模型模式建议配合 use_curated 或 use_factor_pool 使用")
                # Optuna 参数自动加载: param_profile="optuna" 时从磁盘读取对应因子集的搜索结果
                _lgb_params = req.lgb_params or None
                _xgb_params = req.xgb_params or None
                if req.param_profile == "optuna":
                    _opt = load_optuna_result(req.use_curated or "default")
                    if _opt:
                        _lgb_params = _opt.get("lgb_best_params") or _lgb_params
                        _xgb_params = _opt.get("xgb_best_params") or _xgb_params
                        logger.info(f"[Train] 已加载 Optuna 参数: {req.use_curated or 'default'} "
                                    f"(LGB_IC={_opt.get('lgb_best_ic','?')}, XGB_IC={_opt.get('xgb_best_ic','?')})")
                    else:
                        logger.warning(f"[Train] Optuna 参数未找到: {req.use_curated or 'default'}, 回退 conservative")
                result = manager.train_dual_model(
                    factor_set=factor_set,
                    instruments=instruments,
                    train_period=(req.train_start, req.train_end),
                    valid_period=(req.valid_start, req.valid_end),
                    test_period=(req.test_start, req.test_end),
                    label_horizon=req.label_horizon,
                    label_mode=req.label_mode,
                    custom_factor_keys=custom_factor_keys,
                    enable_cross_sectional=req.enable_cross_sectional,
                    enable_crossover=req.enable_crossover,
                    enable_winsorize=req.enable_winsorize,
                    label_norm=req.label_norm,
                    early_stop_metric=req.early_stop_metric,
                    param_profile=req.param_profile,
                    use_reweighter=req.use_reweighter,
                    with_linear=req.with_linear,
                    with_catboost=req.with_catboost,
                    catboost_params=req.catboost_params or None,
                    lgb_params=_lgb_params,
                    xgb_params=_xgb_params,
                )
                backtest_result = None
                if req.auto_backtest and result.get("predictions") is not None and result.get("label") is not None:
                    try:
                        from core.adapters.qlib.backtest import QlibBacktestRunner
                        runner = QlibBacktestRunner()
                        bt = runner.run_from_predictions(
                            predictions=result["predictions"],
                            label=result["label"],
                            topk=req.topk, n_drop=req.n_drop,
                            start_time=req.test_start, end_time=req.test_end,
                            freq=req.freq,
                        )
                        backtest_result = bt
                    except Exception as e:
                        logger.warning(f"[Train] 回测失败 (训练结果不受影响): {e}")

                _train_jobs[job_id] = {
                    "status": "completed",
                    "result": {
                        "ic_mean": result.get("ic_mean"),
                        "ric_mean": result.get("ric_mean"),
                        "icir": result.get("icir"),
                        "lgb_ic_mean": result.get("lgb_ic_mean"),
                        "xgb_ic_mean": result.get("xgb_ic_mean"),
                        "degraded": result.get("degraded"),
                        "backtest": backtest_result,
                    },
                }
                try:
                    import redis as _redis, json as _json
                    _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                    _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
                except Exception:
                    pass
                return

            logger.info(f"[Train] instruments={instruments if isinstance(instruments, str) else f'list[{len(instruments)}]'}")
            logger.info(f"[Train] train={req.train_start}~{req.train_end} valid={req.valid_start}~{req.valid_end} test={req.test_start}~{req.test_end}")
            result = manager.train(
                model_name=req.model_name,
                factor_set=factor_set,
                instruments=instruments,
                train_period=(req.train_start, req.train_end),
                valid_period=(req.valid_start, req.valid_end),
                test_period=(req.test_start, req.test_end),
                model_params=req.model_params,
                label_horizon=req.label_horizon,
                label_mode=req.label_mode,
                custom_factor_keys=custom_factor_keys,
                label_norm=req.label_norm,
                enable_winsorize=req.enable_winsorize,
            )
            backtest_result = None
            if req.auto_backtest and result.get("predictions") is not None and result.get("label") is not None:
                try:
                    from core.adapters.qlib.backtest import QlibBacktestRunner
                    runner = QlibBacktestRunner()
                    bt = runner.run_from_predictions(
                        predictions=result["predictions"],
                        label=result["label"],
                        topk=req.topk, n_drop=req.n_drop,
                        start_time=req.test_start, end_time=req.test_end,
                        freq=req.freq,
                    )
                    backtest_result = bt
                except Exception as e:
                    logger.warning(f"[Train] 回测失败 (训练结果不受影响): {e}")

            _train_jobs[job_id] = {
                "status": "completed",
                "result": {
                    "ic_mean": result.get("ic_mean"),
                    "ric_mean": result.get("ric_mean"),
                    "icir": result.get("icir"),
                    "backtest": backtest_result,
                },
            }
            try:
                import redis as _redis, json as _json
                _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
            except Exception:
                pass
        except Exception as e:
            logger.exception("模型训练失败")
            _train_jobs[job_id] = {"status": "error", "error": str(e)}
            try:
                import redis as _redis, json as _json
                _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
            except Exception:
                pass

    _train_jobs[job_id] = {"status": "running"}
    try:
        import redis as _redis, json as _json
        _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
        _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False))
    except Exception:
        pass
    background_tasks.add_task(_do_train)
    return {"data": {"job_id": job_id, "status": "started"}}
@router.get("/models/train/{job_id}")
async def train_status(job_id: str):
    """查询训练任务状态 (内存 → Redis 回退)"""
    import json as _json
    job = _train_jobs.get(job_id)
    if not job:
        try:
            import redis
            r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2, decode_responses=True)
            for key in (f"qlib:job:{job_id}", f"rolling:train:{job_id}"):
                raw = r.get(key)
                if raw:
                    job = _json.loads(raw)
                    break
        except Exception:
            pass
    if not job:
        raise HTTPException(404, "任务不存在")
    return {"data": job}
@router.get("/optuna-params")
async def list_optuna_params():
    """列出所有已保存的 Optuna 搜索结果"""
    results = []
    if _OPTUNA_DIR.exists():
        import json as _json
        for f in sorted(_OPTUNA_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = _json.load(fp)
                results.append({
                    "factor_set": d.get("factor_set"),
                    "saved_at": d.get("saved_at", "")[:16],
                    "lgb_best_ic": d.get("lgb_best_ic"),
                    "xgb_best_ic": d.get("xgb_best_ic"),
                })
            except Exception:
                pass
    return {"data": results}
@router.post("/models/optuna-search")
async def optuna_search(req: OptunaSearchRequest, background_tasks: BackgroundTasks):
    """Optuna 超参自动搜索 (异步). search_dual=True 依次搜LGB+XGB后融合重训."""
    job_id = str(uuid.uuid4())[:8]

    def _do_optuna():
        try:
            from core.adapters.qlib.models import QlibModelManager
            manager = QlibModelManager()

            factor_set = req.factor_set
            custom_factor_keys = req.custom_factor_keys
            instruments = _resolve_instruments(req.instruments)

            if req.use_curated:
                from core.domain.factors.registry import get_pool
                curated_keys = get_pool().get_curated_factors(req.use_curated)
                if curated_keys:
                    factor_set = "WhiteBoxHandler"
                    custom_factor_keys = curated_keys

            if req.search_dual:
                # 依次搜索 LGB 和 XGB，各自最优参数
                lgb_timeout = (req.timeout or 0) // 2 if req.timeout else None
                xgb_timeout = (req.timeout or 0) // 2 if req.timeout else None

                _train_jobs[job_id] = {"status": "running", "phase": "lgb",
                                       "message": f"搜索 LGB ({req.n_trials}轮)..."}
                lgb_result = manager.train_with_optuna(
                    factor_set=factor_set, instruments=instruments,
                    train_period=(req.train_start, req.train_end),
                    valid_period=(req.valid_start, req.valid_end),
                    test_period=(req.test_start, req.test_end),
                    label_horizon=req.label_horizon, n_trials=req.n_trials,
                    custom_factor_keys=custom_factor_keys,
                    label_norm=req.label_norm, model_name="lightgbm",
                    timeout=lgb_timeout, skip_final_train=True,
                )

                _train_jobs[job_id] = {"status": "running", "phase": "xgb",
                                       "message": f"搜索 XGB ({req.n_trials}轮)...",
                                       "lgb_best_ic": lgb_result["best_ic"]}
                xgb_result = manager.train_with_optuna(
                    factor_set=factor_set, instruments=instruments,
                    train_period=(req.train_start, req.train_end),
                    valid_period=(req.valid_start, req.valid_end),
                    test_period=(req.test_start, req.test_end),
                    label_horizon=req.label_horizon, n_trials=req.n_trials,
                    custom_factor_keys=custom_factor_keys,
                    label_norm=req.label_norm, model_name="xgboost",
                    timeout=xgb_timeout, skip_final_train=True,
                )

                # 直接从 Optuna valid IC 算融合权重, 跳过 train_dual_model (heavy)
                from core.adapters.qlib.models import _ic_fusion_weights
                w_lgb, w_xgb = _ic_fusion_weights(lgb_result["best_ic"], xgb_result["best_ic"])

                result_data = {
                    "lgb_best_params": lgb_result["best_params"],
                    "lgb_best_ic": lgb_result["best_ic"],
                    "xgb_best_params": xgb_result["best_params"],
                    "xgb_best_ic": xgb_result["best_ic"],
                    "lgb_weight": w_lgb,
                    "xgb_weight": w_xgb,
                    "n_trials": req.n_trials,
                }
                _save_optuna_result(req.use_curated or "default", result_data)
                _train_jobs[job_id] = {"status": "completed", "result": result_data}
                try:
                    import redis as _redis, json as _json
                    _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                    _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
                except Exception:
                    pass
                return

            # 单模型搜索
            result = manager.train_with_optuna(
                factor_set=factor_set, instruments=instruments,
                train_period=(req.train_start, req.train_end),
                valid_period=(req.valid_start, req.valid_end),
                test_period=(req.test_start, req.test_end),
                label_horizon=req.label_horizon, n_trials=req.n_trials,
                custom_factor_keys=custom_factor_keys,
                label_norm=req.label_norm, model_name=req.model_name,
                timeout=req.timeout or None,
                param_profile=req.param_profile,
            )
            _model_key = "lgb" if req.model_name == "lightgbm" else "xgb" if req.model_name == "xgboost" else req.model_name
            result_data = {
                f"{_model_key}_best_params": result["best_params"],
                f"{_model_key}_best_ic": result["best_ic"],
                "best_trial": result.get("best_trial"),
                "n_trials": req.n_trials,
            }
            _save_optuna_result(req.use_curated or "default", result_data)
            _train_jobs[job_id] = {"status": "completed", "result": result_data}
            try:
                import redis as _redis, json as _json
                _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
            except Exception:
                pass
        except Exception as e:
            logger.exception("Optuna 搜索失败")
            _train_jobs[job_id] = {"status": "error", "error": str(e)}
            try:
                import redis as _redis, json as _json
                _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
            except Exception:
                pass

    _train_jobs[job_id] = {"status": "running"}
    try:
        import redis as _redis, json as _json
        _r = _redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
        _r.setex(f"qlib:job:{job_id}", 3600, _json.dumps(_train_jobs[job_id], ensure_ascii=False))
    except Exception:
        pass
    background_tasks.add_task(_do_optuna)
    return {"data": {"job_id": job_id, "status": "started"}}
@router.post("/models/rolling-train")
async def rolling_train(req: RollingTrainRequest, background_tasks: BackgroundTasks):
    """滚动时序训练 (异步): 季度 anchor 滚动, 每轮独立 MLflow experiment,
    结束后输出各轮 test 段拼接的全期 OOS 评估"""
    job_id = str(uuid.uuid4())[:8]

    def _do_rolling():
        try:
            from core.adapters.qlib.rolling_train import rolling_train_dual
            from pathlib import Path as _Path
            from core.shared import config as _app_config

            # 处理组合股票池 (csi300+csi500), 读 txt 合并, 不依赖 qlib init
            instruments = req.pool
            if "+" in str(instruments):
                parts = str(instruments).split("+")
                combined = []
                for p in parts:
                    p = p.strip()
                    try:
                        fpath = _Path(_get_qlib_data_dir()) / "instruments" / f"{p}.txt"
                        if fpath.exists():
                            for line in fpath.read_text().splitlines():
                                c = line.strip().split("\t")[0]
                                if c and c not in combined:
                                    combined.append(c)
                    except Exception:
                        pass
                instruments = combined if combined else "all"

            factor_set = "Alpha158"
            custom_factor_keys = None
            if req.use_factor_pool:
                from core.domain.factors.registry import get_pool
                active = get_pool().get_active_factors(category="white_box")
                keys = list(active.keys())
                if keys:
                    factor_set = "WhiteBoxHandler"
                    custom_factor_keys = keys
            elif req.use_curated:
                from core.domain.factors.registry import get_pool
                keys = get_pool().get_curated_factors(req.use_curated)
                if keys:
                    factor_set = "WhiteBoxHandler"
                    custom_factor_keys = keys

            # 嵌入式二次筛选: 构建 qlib 表达式 → factor_key 映射
            feature_key_map = None
            track_importance = False
            if req.filter_mode and req.use_factor_pool and custom_factor_keys:
                from core.adapters.qlib.factor_expression import to_qlib_expression
                from core.domain.factors.whitebox import FACTOR_TEMPLATES
                feature_key_map = {}
                for key in custom_factor_keys:
                    info = active.get(key, {})
                    # 优先用因子池 entry 自带的 expr（与 WhiteBoxHandler 一致）
                    expr = info.get("expr", "")
                    if not expr:
                        tk = info.get("template_key", "")
                        tmpl = FACTOR_TEMPLATES.get(tk, {})
                        expr = to_qlib_expression(tmpl, info.get("params", {}))
                    if expr and not expr.startswith("#"):
                        feature_key_map[expr] = key
                logger.info(f"[Rolling] filter_mode 开启, 构建 {len(feature_key_map)} 个因子映射, "
                            f"自定义因子 {len(custom_factor_keys)} 个")
                track_importance = True

            def _progress(i, total, anchor):
                data = {"status": "running", "progress": i, "total": total,
                        "current_anchor": anchor}
                _train_jobs[job_id] = data
                try:
                    import redis, json
                    r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                    r.setex(f"rolling:train:{job_id}", 1800, json.dumps(data, ensure_ascii=False))
                except Exception:
                    pass

            if not isinstance(instruments, list):
                instruments = _resolve_instruments(req.pool)
            result = rolling_train_dual(
                pool=instruments, pool_label=req.pool, start_anchor=req.start_anchor,
                end_anchor=req.end_anchor or None,
                step_months=req.step_months, train_years=req.train_years,
                valid_months=req.valid_months,
                model_name=req.model_name, train_dual=req.train_dual,
                factor_set=factor_set, custom_factor_keys=custom_factor_keys,
                code_prefixes=req.code_prefixes,
                label_horizon=req.label_horizon, label_norm=req.label_norm,
                label_mode=req.label_mode,
                early_stop_metric=req.early_stop_metric,
                param_profile=req.param_profile,
                enable_crossover=req.enable_crossover,
                enable_cross_sectional=req.enable_cross_sectional,
                enable_winsorize=req.enable_winsorize,
                use_reweighter=req.use_reweighter,
                with_linear=req.with_linear,
                with_catboost=req.with_catboost,
                progress_cb=_progress,
                track_importance=track_importance,
                feature_key_map=feature_key_map,
                feature_names=custom_factor_keys if track_importance else None,
            )

            # 筛选模式: 写回因子池 (仅停用被淘汰的因子，不动其他)
            if req.filter_mode and result.get("elimination_candidates"):
                eliminated = set(result["elimination_candidates"])
                all_keys = custom_factor_keys or []
                passed = [k for k in all_keys if k not in eliminated]
                failed = [k for k in all_keys if k in eliminated]
                pool = get_pool()
                imp_summary = result.get("importance_summary") or {}
                active_factors = pool.get_active_factors()
                try:
                    from core.data import Database
                    db_ok = Database.is_available()
                except Exception:
                    db_ok = False
                for k in failed:
                    pool.toggle_factor(k, active=False)
                    if db_ok:
                        try:
                            info = imp_summary.get(k, {})
                            label = active_factors.get(k, {}).get("label", k)
                            Database.execute(
                                "INSERT INTO factor_elimination_log "
                                "(job_id, factor_key, factor_label, reason, mean_split, win_rate) "
                                "VALUES (%s,%s,%s,%s,%s,%s)",
                                (job_id, k, label, "low_importance",
                                 info.get("mean_split"), info.get("win_rate")),
                            )
                        except Exception as e:
                            logger.warning(f"[Rolling] 淘汰记录入库失败 {k}: {e}")
                result["filter_result"] = {
                    "passed": passed, "failed": failed,
                    "n_passed": len(passed), "n_failed": len(failed),
                }
                logger.info(f"[Rolling] 因子池已更新: 保留 {len(passed)}, 淘汰 {len(failed)}")

            _train_jobs[job_id] = {"status": "completed", "result": result}
            try:
                import redis, json
                r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                r.setex(f"rolling:train:{job_id}", 1800, json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
            except Exception:
                pass
        except Exception as e:
            logger.exception("滚动训练失败")
            _train_jobs[job_id] = {"status": "error", "error": str(e)}
            try:
                import redis, json
                r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
                r.setex(f"rolling:train:{job_id}", 1800, json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
            except Exception:
                pass

    _train_jobs[job_id] = {"status": "running", "progress": 0}
    background_tasks.add_task(_do_rolling)
    return {"data": {"job_id": job_id, "status": "started"}}
@router.get("/trained-models")
async def list_trained_models():
    """列出所有已训练的模型（扫描 MLflow 实验）"""
    import yaml
    from pathlib import Path
    from core.shared import config as app_config

    def _scan():
        mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", ""))
        models = []
        if not mlruns_dir.exists():
            return models

        try:
            for exp_dir in mlruns_dir.iterdir():
                if not exp_dir.is_dir() or exp_dir.name == "0":
                    continue
                exp_id = exp_dir.name
                exp_meta = exp_dir / "meta.yaml"
                exp_name = exp_id
                if exp_meta.exists():
                    try:
                        with open(exp_meta, encoding="utf-8") as f:
                            meta = yaml.safe_load(f)
                        exp_name = meta.get("name", exp_id)
                    except Exception:
                        pass

                for run_dir in exp_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    artifacts = run_dir / "artifacts"
                    is_dual = (artifacts / "params_lgb.pkl").exists()
                    has_params = (artifacts / "params.pkl").exists()
                    if not has_params and not is_dual:
                        continue
                    run_id = run_dir.name
                    is_rolling = exp_name.startswith("rolling_")
                    rolling_anchor = ""
                    if is_rolling:
                        parts = exp_name.rsplit("_", 1)
                        if len(parts) == 2 and len(parts[1]) == 8:
                            ds = parts[1]
                            rolling_anchor = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"
                    model_info = {
                        "experiment_id": exp_id,
                        "experiment_name": exp_name,
                        "run_id": run_id,
                        "model_class": "LGB+XGB" if is_dual else "",
                        "is_dual": is_dual,
                        "is_rolling": is_rolling,
                        "rolling_anchor": rolling_anchor,
                        "factor_set": "",
                        "instruments": "",
                        "train_period": "",
                        "valid_period": "",
                        "test_period": "",
                        "ic": None,
                        "ric": None,
                        "icir": None,
                        "lgb_ic": None,
                        "xgb_ic": None,
                        "trained_at": None,
                        "model_kwargs": {},
                        "has_predictions": (artifacts / "pred.pkl").exists(),
                        "has_label": (artifacts / "label.pkl").exists(),
                    }

                    params_dir = run_dir / "params"
                    if params_dir.exists():
                        for pf in params_dir.iterdir():
                            try:
                                val = pf.read_text().strip()
                                if pf.name == "model.class":
                                    model_info["model_class"] = val
                                elif pf.name == "dataset.kwargs.handler.class":
                                    model_info["factor_set"] = val
                                elif pf.name == "dataset.kwargs.handler.kwargs.instruments":
                                    model_info["instruments"] = val
                                elif pf.name == "dataset.kwargs.segments.train":
                                    model_info["train_period"] = val
                                elif pf.name == "dataset.kwargs.segments.valid":
                                    model_info["valid_period"] = val
                                elif pf.name == "dataset.kwargs.segments.test":
                                    model_info["test_period"] = val
                                elif pf.name == "dataset.kwargs.handler.kwargs.label":
                                    import re
                                    m = re.search(r'-(\d+)\)', val)
                                    model_info["label_period"] = int(m.group(1)) if m else None
                                elif pf.name in ("label_norm", "early_stop_metric", "param_profile"):
                                    model_info[pf.name] = val
                                elif pf.name.startswith("model.kwargs."):
                                    key = pf.name.replace("model.kwargs.", "")
                                    try:
                                        model_info["model_kwargs"][key] = float(val)
                                    except ValueError:
                                        model_info["model_kwargs"][key] = val
                            except Exception:
                                pass

                    metrics_dir = run_dir / "metrics"
                    if metrics_dir.exists():
                        for metric_name, key in [("IC", "ic"), ("Rank_IC", "ric"), ("ICIR", "icir"),
                                                  ("LGB_IC", "lgb_ic"), ("XGB_IC", "xgb_ic"),
                                                  ("LGB_W", "lgb_w"), ("XGB_W", "xgb_w"),
                                                  ("IC_WIN_RATE_M", "ic_win_rate_m"),
                                                  ("MAX_NEG_IC_MONTHS", "max_neg_ic_months"),
                                                  ("ICIR_YEARLY_MIN", "icir_yearly_min"),
                                                  ("Q5_Q1_SPREAD", "q5_q1_spread"),
                                                  ("LS_SHARPE", "ls_sharpe"),
                                                  ("LINEAR_IC", "linear_ic"),
                                                  ("LINEAR_W", "linear_w"),
                                                  ("CAT_IC", "cat_ic"),
                                                  ("CAT_W", "cat_w"),
                                                  ("IC_NONOVERLAP", "ic_nonoverlap"),
                                                  ("ICIR_NONOVERLAP", "icir_nonoverlap"),
                                                  ("IC_DECAY_1M", "ic_decay_1m")]:
                            mf = metrics_dir / metric_name
                            if mf.exists():
                                try:
                                    parts = mf.read_text().strip().split()
                                    if len(parts) >= 2:
                                        model_info[key] = float(parts[1])
                                except Exception:
                                    pass

                    meta_file = run_dir / "meta.yaml"
                    if meta_file.exists():
                        try:
                            with open(meta_file, encoding="utf-8") as f:
                                run_meta = yaml.safe_load(f)
                            raw_time = run_meta.get("start_time")
                            if raw_time is not None:
                                from datetime import datetime, timezone
                                model_info["trained_at"] = datetime.fromtimestamp(raw_time / 1000, tz=timezone.utc).isoformat()
                            model_info["user_id"] = run_meta.get("user_id")
                        except Exception:
                            pass

                    try:
                        model_info["factor_label"] = resolve_factor_label(model_info, run_dir)
                        if model_info["factor_label"]:
                            model_info["factor_set"] = model_info["factor_label"]
                        elif model_info["factor_set"] and "WhiteBox" not in model_info["factor_set"]:
                            cls = model_info["factor_set"].split(".")[-1] if "." in model_info["factor_set"] else model_info["factor_set"]
                            model_info["factor_set"] = cls
                    except Exception:
                        pass

                    if model_info["is_dual"] and model_info["model_class"] != "LGB+XGB":
                        model_info["model_class"] = "LGB+XGB"

                    try:
                        model_info["display_name"] = make_display_name(model_info)
                    except Exception:
                        model_info["display_name"] = f"{model_info.get('model_class','?')}_{run_id[:8]}"

                    models.append(model_info)
        except Exception:
            logger.exception("扫描预测模型失败，跳过")

        # 扫描择时模型
        timing_dir = Path("data/timing_model")
        if timing_dir.exists():
            for pkl_file in sorted(timing_dir.glob("timing_h*.pkl")):
                try:
                    with open(pkl_file, "rb") as f:
                        import pickle
                        tdata = pickle.load(f)
                    mt = tdata.get("model_type", "tree")
                    lh = tdata.get("label_horizon", "?")
                    suffix = "_linear" if mt == "linear" else ""
                    model_info = {
                        "experiment_id": "timing",
                        "experiment_name": "择时模型",
                        "run_id": f"timing_h{lh}{suffix}",
                        "model_class": "LGB+XGB" if mt == "tree" else "EN+MLP",
                        "is_dual": mt == "tree",
                        "is_rolling": False,
                        "rolling_anchor": "",
                        "factor_set": "择时模型",
                        "instruments": "指数",
                        "train_period": "",
                        "valid_period": "",
                        "test_period": "",
                        "ic": tdata.get("valid_auc"),
                        "ric": None,
                        "icir": None,
                        "lgb_ic": tdata.get("lgb_auc"),
                        "xgb_ic": tdata.get("xgb_auc"),
                        "trained_at": str(tdata.get("last_train_date", "")),
                        "model_kwargs": {"label_horizon": lh},
                        "has_predictions": True,
                        "has_label": False,
                        "label_period": lh,
                        "model_type": "timing",
                        "display_name": f"择时_h{lh}_{'Tree' if mt == 'tree' else 'Linear'}",
                    }
                    models.append(model_info)
                except Exception:
                    logger.exception(f"扫描择时模型失败: {pkl_file}")

            for pkl_file in sorted(timing_dir.glob("regime_h*.pkl")):
                try:
                    with open(pkl_file, "rb") as f:
                        import pickle
                        tdata = pickle.load(f)
                    mt = tdata.get("model_type", "linear")
                    lh = tdata.get("label_horizon", "?")
                    suffix = "_linear" if mt == "linear" else ""
                    model_info = {
                        "experiment_id": "regime",
                        "experiment_name": "市场状态模型",
                        "run_id": f"regime_h{lh}{suffix}",
                        "model_class": "LGB+XGB" if mt == "tree" else "EN+MLP",
                        "is_dual": mt == "tree",
                        "is_rolling": False,
                        "rolling_anchor": "",
                        "factor_set": "市场状态",
                        "instruments": "指数",
                        "train_period": "",
                        "valid_period": "",
                        "test_period": "",
                        "ic": tdata.get("valid_macro_auc") or tdata.get("valid_acc"),
                        "ric": None,
                        "icir": None,
                        "lgb_ic": tdata.get("lgb_acc"),
                        "xgb_ic": tdata.get("xgb_acc"),
                        "trained_at": str(tdata.get("last_train_date", "")),
                        "model_kwargs": {"label_horizon": lh},
                        "has_predictions": True,
                        "has_label": False,
                        "label_period": lh,
                        "model_type": "regime",
                        "display_name": f"Regime_h{lh}_{'Tree' if mt == 'tree' else 'Linear'}",
                    }
                    models.append(model_info)
                except Exception:
                    logger.exception(f"扫描Regime模型失败: {pkl_file}")

            for pkl_file in sorted(timing_dir.glob("regime_v2_h*.pkl")):
                try:
                    with open(pkl_file, "rb") as f:
                        import pickle
                        tdata = pickle.load(f)
                    lh = tdata.get("label_horizon", "?")
                    model_info = {
                        "experiment_id": "regime_v2",
                        "experiment_name": "市场状态模型 V2",
                        "run_id": f"regime_v2_h{lh}",
                        "model_class": "GBR-Vol",
                        "is_dual": False,
                        "is_rolling": False,
                        "rolling_anchor": "",
                        "factor_set": "波动率预测",
                        "instruments": "指数",
                        "train_period": "",
                        "valid_period": "",
                        "test_period": "",
                        "ic": tdata.get("vol_r2") if tdata.get("vol_r2") is not None else None,
                        "ric": tdata.get("vol_mae") if tdata.get("vol_mae") is not None else None,
                        "icir": None,
                        "lgb_ic": None,
                        "xgb_ic": None,
                        "trained_at": str(tdata.get("last_train_date", "")),
                        "model_kwargs": {"label_horizon": lh},
                        "has_predictions": True,
                        "has_label": False,
                        "label_period": lh,
                        "model_type": "regime_v2",
                        "display_name": f"RegimeV2_h{lh}_GBR",
                    }
                    models.append(model_info)
                except Exception:
                    logger.exception(f"扫描RegimeV2模型失败: {pkl_file}")

        # 标记已停用模型
        from core.shared.disabled_models import get_all_disabled
        disabled_map = get_all_disabled()
        for m in models:
            key = f"{m.get('experiment_id')}/{m.get('run_id')}"
            m["disabled"] = key in disabled_map

        models.sort(key=lambda m: m["ic"] or 0, reverse=True)
        return {"data": models}

    return await anyio.to_thread.run_sync(_scan)
@router.get("/training/results")
async def list_training_results(
    factor_set: str = None,
    instruments: str = None,
    limit: int = 50,
    sort_by: str = "created_at",
):
    """查询历史训练结果 (从 MySQL training_results 表)。
    支持按 factor_set/instruments 过滤，默认按创建时间降序。
    """
    try:
        from core.data import Database
        if not Database.is_available():
            return {"data": [], "message": "数据库不可用"}
        conditions = []
        params = []
        if factor_set:
            conditions.append("factor_set = %s")
            params.append(factor_set)
        if instruments:
            conditions.append("instruments = %s")
            params.append(instruments)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sort_cols = {"icir": "icir", "ic_mean": "ic_mean", "created_at": "created_at"}
        order = f" ORDER BY {sort_cols.get(sort_by, 'created_at')} DESC LIMIT %s"
        params.append(limit)
        rows = Database.fetchall(
            "SELECT id, run_id, model_type, factor_set, instruments, "
            "train_start, train_end, valid_start, valid_end, test_start, test_end, "
            "label_horizon, ic_mean, ric_mean, icir, "
            "lgb_ic_mean, lgb_weight, xgb_ic_mean, xgb_weight, "
            "linear_ic_mean, linear_weight, cat_ic_mean, cat_weight, "
            "monthly_win_rate, annual_icir, long_short_spread, max_neg_ic_months, "
            "custom_factor_keys, created_at "
            "FROM training_results" + where + order,
            tuple(params),
        )
        results = []
        for r in rows:
            d = dict(r)
            for k in ("created_at", "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"):
                if d.get(k):
                    d[k] = str(d[k])
            # 解析 JSON 字段
            if d.get("custom_factor_keys"):
                try:
                    import json as _json
                    d["custom_factor_keys"] = _json.loads(d["custom_factor_keys"]) if isinstance(d["custom_factor_keys"], str) else d["custom_factor_keys"]
                except Exception:
                    pass
            results.append(d)
        return {"data": results, "total": len(results)}
    except Exception as e:
        logger.warning(f"[TrainingResults] 查询失败: {e}")
        return {"data": [], "error": str(e)}
@router.delete("/trained-models/{experiment_id}/{run_id}")
async def delete_trained_model(experiment_id: str, run_id: str):
    """删除指定训练模型"""
    import shutil
    from pathlib import Path
    from core.shared import config as app_config

    mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", ""))
    run_dir = assert_run_path(mlruns_dir, experiment_id, run_id)
    if not run_dir.exists():
        raise HTTPException(404, f"模型不存在: {experiment_id}/{run_id}")

    try:
        shutil.rmtree(run_dir)
        # 如果实验下没有其他 run 了，删除整个实验目录
        exp_dir = run_dir.parent
        remaining = [d for d in exp_dir.iterdir() if d.is_dir() and d.name != "metrics"]
        if not remaining:
            shutil.rmtree(exp_dir)
        # 清除 direction_models 缓存，避免回测引用已删除的模型
        try:
            from core.data import SettingsRepo
            SettingsRepo.set("direction_models", {})
        except Exception:
            pass
        # 清除停用状态
        try:
            from core.shared.disabled_models import set_disabled
            set_disabled(experiment_id, run_id, disabled=False)
        except Exception:
            pass
        return {"data": {"deleted": True}}
    except Exception as e:
        raise HTTPException(500, f"删除失败: {e}")
@router.put("/trained-models/{run_id}/disable")
async def disable_model(run_id: str, req: ModelToggleRequest):
    """停用指定模型 — 停用后所有使用该模型的地方将跳过该模型"""
    from core.shared.disabled_models import set_disabled
    set_disabled(req.experiment_id, run_id, disabled=True)
    return {"data": {"disabled": True, "experiment_id": req.experiment_id, "run_id": run_id}}
@router.put("/trained-models/{run_id}/enable")
async def enable_model(run_id: str, req: ModelToggleRequest):
    """恢复已停用的模型"""
    from core.shared.disabled_models import set_disabled
    set_disabled(req.experiment_id, run_id, disabled=False)
    return {"data": {"disabled": False, "experiment_id": req.experiment_id, "run_id": run_id}}
