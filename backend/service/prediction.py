"""预测服务 — Qlib 预测/回测用例编排。"""
import logging

from fastapi import HTTPException

from core.shared import config
from core.shared.infrastructure.async_task import task_manager
from backend.service.qlib_path_guard import assert_run_path
from backend.service.qlib_access import qlib_read_locked, run_with_qlib_read_access

logger = logging.getLogger(__name__)


class PredictionService:

    @staticmethod
    def _get_top_features(model, dataset, top_n=10):
        """从模型和数据集提取 Top-N 特征重要性及名称"""
        import numpy as np

        top_features = []
        try:
            if hasattr(model, "booster_"):
                booster = model.booster_
                imp = booster.feature_importance()
                names = booster.feature_name()
                idx = np.argsort(imp)[::-1][:top_n]
                for rank, i in enumerate(idx, 1):
                    top_features.append({
                        "name": names[i],
                        "importance": round(float(imp[i]), 2),
                        "rank": rank,
                    })
        except Exception:
            logger.warning("特征重要性提取失败", exc_info=True)
        return top_features

    @staticmethod
    def _get_feature_values(dataset, pred_index, top_features):
        """从 dataset 提取预测日期的因子值"""
        import pandas as pd

        try:
            X_test, _ = dataset.prepare("test")
            if not isinstance(X_test, pd.DataFrame):
                return {}
            feature_names = [f["name"] for f in top_features if f["name"] in X_test.columns]
            if not feature_names:
                return {}

            X_sub = X_test[feature_names]
            result = {}
            for dt, inst in pred_index:
                key = (pd.Timestamp(dt), inst) if not isinstance(dt, pd.Timestamp) else (dt, inst)
                try:
                    row = X_sub.loc[key]
                    result[f"{dt}_{inst}"] = {
                        fn: round(float(row[fn]), 4)
                        for fn in feature_names
                    }
                except (KeyError, TypeError):
                    continue
            return result
        except Exception:
            logger.warning("因子值提取失败", exc_info=True)
            return {}

    @staticmethod
    def _format_predictions(pred_df, dataset, top_features, topk, neutralize_scores=True,
                            industry_map=None, model_input_semantics="raw_model_score"):
        from core.domain.screening._screener import normalize_qlib_scores

        dates_summary = {}
        records = []
        score_audits = {}
        ranked_rows = []
        for dt, group in pred_df.groupby("datetime", sort=True):
            raw_scores = {}
            instruments = {}
            for _, row in group.iterrows():
                instrument = str(row["instrument"])
                code = instrument[2:] if instrument.startswith(("sh", "sz")) else instrument
                raw_scores[code] = float(row["score"])
                instruments[code] = instrument
            details = normalize_qlib_scores(
                raw_scores, str(dt)[:10], industry_map=industry_map,
                neutralize=neutralize_scores, return_details=True,
                model_input_semantics=model_input_semantics,
            )
            score_audits[str(dt)[:10]] = details["audit"]
            day_rows = []
            for code, ledger in details["ledger"].items():
                day_rows.append({
                    "datetime": str(dt)[:10],
                    "instrument": instruments.get(code, code),
                    "model_input_score": ledger.get("model_input_score"),
                    "model_input_semantics": ledger.get("model_input_semantics"),
                    "raw_score": ledger.get("raw_model_score"),
                    "neutralized_score": ledger.get("neutralized_score"),
                    "percentile_score": ledger.get("final_rank_score"),
                    "rank_percentile": ledger.get("rank_percentile"),
                    "rank": ledger.get("final_rank"),
                    "score": ledger.get("raw_model_score"),
                })
            day_rows.sort(key=lambda row: (row.get("rank") or 10 ** 9, row["instrument"]))
            dates_summary[str(dt)[:10]] = day_rows[:topk]
            ranked_rows.extend(day_rows[:topk * 5])

        pred_index = [(row["datetime"], row["instrument"]) for row in ranked_rows]
        feature_values = PredictionService._get_feature_values(dataset, pred_index, top_features) if top_features else {}
        for row in ranked_rows:
            row["factors"] = feature_values.get(f"{row['datetime']}_{row['instrument']}", {})
        for rows in dates_summary.values():
            for row in rows:
                row["factors"] = feature_values.get(f"{row['datetime']}_{row['instrument']}", {})
        records.extend(ranked_rows)
        return records, dates_summary, score_audits

    @staticmethod
    def _do_predict(experiment_id: str, run_id: str, instruments: str,
                    start_time: str, end_time: str, topk: int,
                    skip_features: bool, code_prefixes: list,
                    strategy_profile=None, profile_factor_keys=None,
                    neutralize_scores=True, industry_map=None,
                    fit_start_time=None, fit_end_time=None):
        """预测执行体 — 在后台线程中运行"""
        import pickle
        from pathlib import Path as _Path

        mlruns_dir = _Path(config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", ""))
        model_path = assert_run_path(mlruns_dir, experiment_id, run_id) / "artifacts" / "params.pkl"
        from core.adapters.qlib.fundval import require_run_contract
        trained_keys = require_run_contract(model_path.parent, profile_factor_keys or None)

        factor_set = "Alpha158"
        params_dir = mlruns_dir / experiment_id / run_id / "params"
        if params_dir.exists():
            for pf in params_dir.iterdir():
                try:
                    if pf.name == "dataset.kwargs.handler.class":
                        factor_set = pf.read_text().strip()
                except Exception:
                    pass

        from core.adapters.qlib.config import init_qlib
        init_qlib()

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        handler_kwargs = {
            "instruments": instruments,
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": fit_start_time or start_time,
            "fit_end_time": fit_end_time or end_time,
        }
        if factor_set.startswith("WhiteBox") and profile_factor_keys:
            handler_kwargs["factor_keys"] = list(profile_factor_keys)
        elif factor_set.startswith('WhiteBox'):
            if not trained_keys:
                raise RuntimeError('无法还原模型的实际因子清单')
            handler_kwargs['factor_keys'] = trained_keys
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": factor_set,
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": handler_kwargs,
                },
                "segments": {"test": (start_time, end_time)},
            },
        }

        handler_module = "core.adapters.qlib.custom_handler" if factor_set in ("WhiteBoxHandler",) or factor_set.startswith("WhiteBox") else "qlib.contrib.data.handler"
        dataset_config["kwargs"]["handler"]["module_path"] = handler_module

        from core.adapters.qlib.config import build_dataset_cached
        dataset = build_dataset_cached(dataset_config)
        top_features = [] if skip_features else PredictionService._get_top_features(model, dataset)
        pred = model.predict(dataset)

        if code_prefixes:
            prefixes = tuple(str(p) for p in code_prefixes)

        dates_summary = {}
        records = []
        score_audits = {}
        if hasattr(pred, "reset_index"):
            pred_df = pred.reset_index()
            if len(pred_df.columns) >= 3:
                pred_df.columns = ["datetime", "instrument", "score"]
            pred_df["datetime"] = pred_df["datetime"].astype(str).str[:10]
            pred_df["instrument"] = pred_df["instrument"].astype(str)

            if code_prefixes:
                raw_code = pred_df["instrument"].str[2:]
                pred_df = pred_df[raw_code.str.startswith(prefixes)]

            records, dates_summary, score_audits = PredictionService._format_predictions(
                pred_df, dataset, top_features, topk,
                neutralize_scores=neutralize_scores, industry_map=industry_map,
            )

            logger.info(f"预测完成: records={len(records)}, dates={len(dates_summary)}")

            try:
                model_id = f"{experiment_id}/{run_id}"
                db_rows = []
                all_pred = pred_df.copy() if hasattr(pred, "reset_index") else None
                if all_pred is not None and not all_pred.empty:
                    for _, r in all_pred.iterrows():
                        inst = str(r["instrument"])
                        code = inst[2:] if inst.startswith(("sh", "sz")) else inst
                        dt = str(r["datetime"])[:10]
                        db_rows.append((model_id, code, dt, float(r["score"])))
                if db_rows:
                    from core.data import Database
                    batch_size = 500
                    for bs in range(0, len(db_rows), batch_size):
                        batch = db_rows[bs:bs + batch_size]
                        ph = ",".join(["(%s,%s,%s,%s)"] * len(batch))
                        flat = []
                        for b in batch:
                            flat.extend(b)
                        Database.execute(
                            f"INSERT INTO qlib_predictions (model_id, code, trade_date, score) "
                            f"VALUES {ph} ON DUPLICATE KEY UPDATE score=VALUES(score)",
                            tuple(flat),
                        )
                    logger.info(f"预测入库: {len(db_rows)} 条")
            except Exception as e:
                logger.warning(f"预测入库失败: {e}")
        elif hasattr(pred, "to_dict"):
            records = [{"score": float(v)} for v in list(pred)]

        return {
            "predictions": records,
            "count": len(records),
            "topk": topk,
            "by_date": dict(list(dates_summary.items())[:30]),
            "feature_importance": top_features,
            "score_audits": score_audits,
            "score_semantics": {
                "raw_score": "模型原始输出，仅在同一模型内部解释",
                "neutralized_score": "市值和行业残差化输出",
                "percentile_score": "仅代表同日横截面排序，不能跨日期比较或解释为预期收益",
            },
            "strategy_profile": strategy_profile,
        }

    @staticmethod
    def submit_predict(req):
        """校验请求并提交预测后台任务"""
        from pathlib import Path as _Path

        profile_meta = None
        factor_keys = None
        neutralize_scores = True
        industry_map = None
        experiment_id = req.experiment_id
        run_id = req.run_id
        instruments = req.instruments
        topk = req.topk
        code_prefixes = req.code_prefixes
        start_time = req.start_time
        end_time = req.end_time
        fit_start_time = None
        fit_end_time = None
        shadow_models = []
        profile_warnings = []
        if req.profile_id or req.profile_version is not None:
            if not req.profile_id or req.profile_version is None:
                raise HTTPException(422, "profile_id和profile_version必须同时提供")
            from core.data import StrategyProfileRepo
            from backend.service.strategy_profile import StrategyProfileService
            row = StrategyProfileRepo.get(req.profile_id, int(req.profile_version))
            if not row:
                raise HTTPException(422, "策略版本不存在")
            profile = row["config_json"]
            research_only = bool(profile.get("factors", {}).get("research_only"))
            if row["status"] not in ("challenger", "paper", "champion") and not research_only:
                raise HTTPException(409, "只有候选策略、模拟运行、正式策略或仅研究版本可以运行预测")
            first_date = str((profile.get("availability") or {}).get("first_decision_date") or "")[:10]
            if first_date and (start_time < first_date or end_time < first_date):
                raise HTTPException(422, f"该策略最早可运行日期为 {first_date}")
            if len(profile["model"].get("shadow_refs") or []) > 3:
                raise HTTPException(422, "影子模型最多允许3个")
            try:
                model_ref, scheduled_model = StrategyProfileService.model_ref_for_date(
                    row, start_time, require_schedule=row["status"] == "challenger",
                )
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            locked = {
                "experiment_id": model_ref["experiment_id"],
                "run_id": model_ref["run_id"],
                "instruments": "a500",
                "topk": int(profile["selection"]["candidate_pool"]),
                "code_prefixes": list(profile["universe"].get("code_prefixes") or []),
            }
            supplied = getattr(req, "model_fields_set", set())
            conflicts = []
            for field, value in locked.items():
                if field in supplied and getattr(req, field) != value and not (
                    field == "instruments" and getattr(req, field) == "stable_history_500d"
                ):
                    conflicts.append(field)
            if conflicts:
                raise HTTPException(422, f"请求覆盖了Profile锁定参数: {', '.join(conflicts)}")
            if start_time != end_time:
                raise HTTPException(422, "Profile预测必须使用单一固定决策日，开始和结束日期需相同")
            windows = profile["windows"]
            if row["status"] == "challenger" and not (
                str(windows["oos_start"]) <= start_time <= str(windows["oos_end"])
            ):
                raise HTTPException(422, "候选策略的预测日期必须位于独立样本外窗口内")
            snapshot = StrategyProfileService.build_data_snapshot(start_time, profile["universe"]["min_history_days"])
            if (not StrategyProfileService.snapshot_has_current_integrity(snapshot)
                    or snapshot["freshness_status"] != "fresh"):
                raise HTTPException(409, "预测日期的数据快照未通过时点一致性或新鲜度检查")
            if snapshot.get("security_reference_status") != "pass":
                if not research_only:
                    raise HTTPException(409, "预测日期的历史证券状态快照不完整")
                profile_warnings.append("历史行业、特别处理、上市和停牌状态快照尚未完整")
            references = StrategyProfileRepo.get_security_references(start_time)
            industry_map = {item["code"]: item.get("industry", "") for item in references}
            experiment_id = locked["experiment_id"]
            run_id = locked["run_id"]
            instruments = locked["instruments"]
            topk = locked["topk"]
            code_prefixes = locked["code_prefixes"]
            factor_keys = profile["factors"]["keys"]
            fit_start_time = profile["windows"]["train_start"]
            fit_end_time = profile["windows"]["train_end"]
            neutralize_scores = bool(profile["score"]["neutralize"])
            from core.adapters.qlib.model_selection import get_model_for_factor_set
            refs = [(model_ref, factor_keys)]
            refs.extend((ref, ref.get("factor_keys") or []) for ref in profile["model"].get("shadow_refs") or [])
            for ref, keys in refs:
                try:
                    checked = get_model_for_factor_set(
                        ref["experiment_id"], ref["run_id"], keys,
                        start_time, "a500", allow_test_window=scheduled_model and ref is model_ref,
                    )
                except ValueError as error:
                    raise HTTPException(409, str(error)) from error
                if int(checked.get("label_horizon") or 0) != 10:
                    raise HTTPException(409, f"模型 {ref['run_id']} 的预测周期不是10日")
                if checked.get("artifact_fingerprint") != ref.get("artifact_fingerprint"):
                    raise HTTPException(409, f"模型 {ref['run_id']} 的内容指纹已变化")
            shadow_models = list(profile["model"].get("shadow_refs") or [])
            profile_meta = {
                "profile_id": row["profile_id"], "version": int(row["version"]),
                "name": row.get("name") or row["profile_id"],
                "status": row["status"], "fingerprint": row["fingerprint"],
                "data_fingerprint": snapshot["snapshot_id"],
                "model_fingerprint": profile.get("fingerprints", {}).get("model_refs"),
                "factor_fingerprint": profile.get("fingerprints", {}).get("factor"),
                "research_only": research_only,
                "promotable": not research_only,
                "active_model": [model_ref["experiment_id"], model_ref["run_id"]],
                "uses_scheduled_oos_model": scheduled_model,
            }

        if not experiment_id or not run_id:
            raise HTTPException(400, "必须指定 experiment_id 和 run_id")

        from core.shared.disabled_models import is_disabled
        if is_disabled(experiment_id, run_id):
            raise HTTPException(400, "模型已停用，请先恢复后再使用")

        mlruns_dir = _Path(config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", ""))
        run_path = assert_run_path(mlruns_dir, experiment_id, run_id)
        model_path = run_path / "artifacts" / "params.pkl"
        if not model_path.exists():
            raise HTTPException(400, f"模型文件不存在: {model_path}")

        predict_fn = PredictionService._do_predict_dual if any(
            (run_path / "artifacts" / name).exists() for name in ("params_lstm.pkl", "params_lgb.pkl")
        ) else PredictionService._do_predict
        predict_args = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "instruments": instruments,
            "start_time": start_time,
            "end_time": end_time,
            "topk": topk,
            "skip_features": req.skip_features,
            "code_prefixes": code_prefixes,
            "strategy_profile": profile_meta,
            "profile_factor_keys": factor_keys,
            "neutralize_scores": neutralize_scores,
            "industry_map": industry_map,
            "fit_start_time": fit_start_time,
            "fit_end_time": fit_end_time,
        }
        runner = (
            lambda: PredictionService._do_profile_predict(
                predict_fn, predict_args, shadow_models, profile_warnings,
            )
        ) if profile_meta else lambda: predict_fn(**predict_args)
        job_id = task_manager.submit(
            lambda: run_with_qlib_read_access(runner),
            description=f"预测 {experiment_id}/{run_id}",
            pool="default",
        )
        return {"data": {"job_id": job_id, "status": "started"}}

    @staticmethod
    def _do_profile_predict(predict_fn, primary_args, shadow_models, warnings):
        result = predict_fn(**primary_args)
        observations = []
        from pathlib import Path
        mlruns_dir = Path(config.QLIB_MLFLOW_URI.replace("file:", ""))
        for ref in shadow_models:
            try:
                run_path = assert_run_path(mlruns_dir, ref["experiment_id"], ref["run_id"])
                shadow_fn = PredictionService._do_predict_dual if any(
                    (run_path / "artifacts" / name).exists() for name in ("params_lstm.pkl", "params_lgb.pkl")
                ) else PredictionService._do_predict
                args = {
                    **primary_args,
                    "experiment_id": ref["experiment_id"],
                    "run_id": ref["run_id"],
                    "strategy_profile": None,
                    "profile_factor_keys": ref.get("factor_keys") or [],
                }
                shadow = shadow_fn(**args)
                overlap = {}
                for date, rows in (result.get("by_date") or {}).items():
                    main_codes = {row.get("instrument") for row in rows[:5]}
                    shadow_codes = {
                        row.get("instrument")
                        for row in (shadow.get("by_date") or {}).get(date, [])[:5]
                    }
                    denominator = min(5, len(main_codes), len(shadow_codes))
                    overlap[date] = len(main_codes & shadow_codes) / denominator if denominator else 0.0
                observations.append({
                    "model": [ref["experiment_id"], ref["run_id"]],
                    "factor_set_name": ref.get("factor_set_name"),
                    "role": ref.get("role"),
                    "status": "completed",
                    "changes_selection": False,
                    "by_date": shadow.get("by_date") or {},
                    "top5_overlap": overlap,
                })
            except Exception as error:
                logger.warning(f"[影子模型] {ref.get('run_id')} 预测失败: {error}")
                observations.append({
                    "model": [ref.get("experiment_id"), ref.get("run_id")],
                    "factor_set_name": ref.get("factor_set_name"),
                    "role": ref.get("role"),
                    "status": "failed",
                    "changes_selection": False,
                    "error": str(error),
                })
        result["shadow_model_observations"] = observations
        result["warnings"] = list(warnings or [])
        return result

    @staticmethod
    def _do_predict_dual(experiment_id: str, run_id: str, instruments: str,
                         start_time: str, end_time: str, topk: int,
                         skip_features: bool, code_prefixes: list,
                         strategy_profile=None, profile_factor_keys=None,
                         neutralize_scores=True, industry_map=None,
                         fit_start_time=None, fit_end_time=None):
        """双模型预测执行体 — 在后台线程中运行"""
        import numpy as np
        import pandas as pd

        from core.adapters.qlib.models import QlibModelManager
        from core.adapters.qlib.config import init_qlib
        init_qlib()

        manager = QlibModelManager()
        fused, dataset = manager.predict_dual(
            experiment_id, run_id,
            predict_start=start_time, predict_end=end_time,
            instruments=instruments,
        )

        if hasattr(fused, 'reset_index'):
            pred_df = fused.reset_index()
            if len(pred_df.columns) >= 3:
                pred_df.columns = ["datetime", "instrument", "score"]
            pred_df["datetime"] = pred_df["datetime"].astype(str).str[:10]
            pred_df["instrument"] = pred_df["instrument"].astype(str)
        else:
            arr = np.array(fused).ravel()
            pred_df = pd.DataFrame({"score": arr, "instrument": "", "datetime": ""})

        if code_prefixes:
            prefixes = tuple(str(p) for p in code_prefixes)
            raw_code = pred_df["instrument"].str[2:]
            pred_df = pred_df[raw_code.str.startswith(prefixes)]

        records, dates_summary, score_audits = PredictionService._format_predictions(
            pred_df, dataset, [], topk,
            neutralize_scores=neutralize_scores, industry_map=industry_map,
            model_input_semantics="daily_cross_sectional_rank_fusion",
        )

        try:
            model_id = f"{experiment_id}/{run_id}/dual"
            db_rows = []
            for _, r in pred_df.iterrows():
                inst = str(r["instrument"])
                code = inst[2:] if inst.startswith(("sh", "sz")) else inst
                db_rows.append((model_id, code, str(r["datetime"])[:10], float(r["score"])))
            if db_rows:
                from core.data import Database
                for bs in range(0, len(db_rows), 500):
                    batch = db_rows[bs:bs + 500]
                    ph = ",".join(["(%s,%s,%s,%s)"] * len(batch))
                    flat = [x for b in batch for x in b]
                    Database.execute(
                        f"INSERT INTO qlib_predictions (model_id, code, trade_date, score) "
                        f"VALUES {ph} ON DUPLICATE KEY UPDATE score=VALUES(score)", tuple(flat))
                logger.info(f"[PredictDual] 入库: {len(db_rows)} 条")
        except Exception as e:
            logger.warning(f"[PredictDual] 入库失败: {e}")

        return {
            "predictions": records, "count": len(records), "topk": topk,
            "by_date": dict(list(dates_summary.items())[:30]),
            "score_audits": score_audits,
            "score_semantics": {
                "model_input_score": "融合run的同日排名输入，不是单模型原始收益预测",
                "raw_score": "融合run不提供可比较的单一原始模型输出",
                "neutralized_score": "市值和行业残差化输出",
                "percentile_score": "仅代表同日横截面排序，不能跨日期比较或解释为预期收益",
            },
            "strategy_profile": strategy_profile,
        }

    @staticmethod
    def submit_predict_dual(req):
        """校验请求并提交双模型预测后台任务"""
        if not req.experiment_id or not req.run_id:
            raise HTTPException(400, "必须指定 experiment_id 和 run_id")

        from core.shared.disabled_models import is_disabled
        if is_disabled(req.experiment_id, req.run_id):
            raise HTTPException(400, "模型已停用，请先恢复后再使用")

        job_id = task_manager.submit(
            lambda: run_with_qlib_read_access(
                PredictionService._do_predict_dual,
                experiment_id=req.experiment_id,
                run_id=req.run_id,
                instruments=req.instruments,
                start_time=req.start_time,
                end_time=req.end_time,
                topk=req.topk,
                skip_features=req.skip_features,
                code_prefixes=req.code_prefixes,
            ),
            description=f"双模型预测 {req.experiment_id}/{req.run_id}",
            pool="default",
        )
        return {"data": {"job_id": job_id, "status": "started"}}

    @staticmethod
    def get_predict_result(job_id: str):
        """查询预测任务结果"""
        task = task_manager.get_dict(job_id)
        if task is None:
            return {"data": None, "error": f"任务 {job_id} 不存在"}
        return {"data": task}

    @staticmethod
    def get_stock_predictions(code: str, days: int = 60, model_id: str = None):
        """查询某只股票的历史预测分数"""
        from core.data import Database

        if model_id:
            rows = Database.fetchall(
                "SELECT trade_date, score FROM qlib_predictions "
                "WHERE code=%s AND model_id=%s AND trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY) "
                "ORDER BY trade_date DESC LIMIT 200",
                (code, model_id, days),
            )
        else:
            rows = Database.fetchall(
                "SELECT trade_date, score, model_id FROM qlib_predictions "
                "WHERE code=%s AND trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY) "
                "ORDER BY trade_date DESC LIMIT 200",
                (code, days),
            )
        return {"data": [{"date": str(r["trade_date"]), "score": float(r["score"]),
                         "model_id": r.get("model_id", "")} for r in rows]}

    @staticmethod
    @qlib_read_locked
    def run_backtest(req):
        """运行 qlib 回测"""
        from core.adapters.qlib.generation import assert_no_active_historical_release

        assert_no_active_historical_release(config.QLIB_DATA_ROOT)
        import pickle
        from pathlib import Path

        pred_path = Path(req.predictions_file).resolve()
        mlruns_dir = Path(config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", "")).resolve()
        data_root = Path(config.QLIB_DATA_DIR).resolve()
        try:
            pred_path.relative_to(mlruns_dir)
        except ValueError:
            try:
                pred_path.relative_to(data_root)
            except ValueError:
                raise HTTPException(400, f"预测文件必须在服务端数据目录内: {req.predictions_file}")
        if not pred_path.exists():
            raise HTTPException(400, f"预测文件不存在: {req.predictions_file}")

        data = pickle.loads(pred_path.read_bytes())

        from core.adapters.qlib.backtest import QlibBacktestRunner
        runner = QlibBacktestRunner()

        if isinstance(data, tuple) and len(data) == 2:
            predictions, label = data
        else:
            predictions = data
            label = None

        if label is None:
            raise HTTPException(400, "预测文件缺少标签数据，请保存为 (predictions, label) 元组")

        result = runner.run_from_predictions(
            predictions=predictions,
            label=label,
            topk=req.topk,
            n_drop=req.n_drop,
            start_time=req.start_time,
            end_time=req.end_time,
        )
        from backend.service.backtest import BacktestService
        report = BacktestService.build_standard_report(result)
        # 不把 pandas DataFrame 直接返回给前端
        result.pop("report_normal", None)
        result["standard_report"] = report
        return {"data": result}
