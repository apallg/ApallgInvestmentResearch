import datetime as dt
import traceback
import uuid

from backend.service.factor_mining import factor_mining_service
from core.adapters.qlib.factor_resolver import resolve_factor_expression
from core.data.repos.factor_mining import factor_mining_repo
from core.shared.infrastructure.async_task import task_manager


class FactorMiningTrainingService:
    def create(self, run_id, request):
        normalized = self._normalize(run_id, request)
        training_id = uuid.uuid4().hex[:12]
        factor_mining_repo.ensure_tables()
        factor_mining_repo.create_training_run(
            training_id,
            run_id,
            normalized["candidate_keys"],
            normalized["baseline_set"],
            normalized,
        )
        task_manager.submit_with_id(
            training_id,
            lambda: self._run(training_id, normalized),
            f"候选因子研究训练 {','.join(normalized['candidate_keys'])}",
            pool="default",
        )
        return {"training_id": training_id, "status": "pending", "request": normalized}

    @staticmethod
    def _normalize(run_id, request):
        run = factor_mining_repo.get_run(run_id)
        if not run:
            raise ValueError("挖矿任务不存在")
        candidate_keys = list(dict.fromkeys(request.get("candidate_keys") or []))
        if not candidate_keys:
            raise ValueError("至少选择一个候选因子")
        if len(candidate_keys) > 20:
            raise ValueError("单次研究训练最多选择 20 个候选因子")
        candidates = {item["candidate_key"]: item for item in factor_mining_repo.list_candidates(run_id)}
        missing = [key for key in candidate_keys if key not in candidates]
        if missing:
            raise ValueError(f"候选不属于该任务: {', '.join(missing)}")
        unresolved = [key for key in candidate_keys if not resolve_factor_expression(key)[0]]
        if unresolved:
            raise ValueError(f"候选表达式无法被 Qlib 解析: {', '.join(unresolved)}")
        config = run.get("config") or {}
        normalized = {
            "run_id": run_id,
            "candidate_keys": candidate_keys,
            "pool": request.get("pool") or config.get("pool", "csi300"),
            "baseline_set": request.get("baseline_set") or config.get("baseline_set", "current_curated"),
            "include_baseline": bool(request.get("include_baseline", True)),
            "train_baseline_control": bool(request.get("train_baseline_control", True)),
            "train_start": request.get("train_start") or config.get("train_start"),
            "train_end": request.get("train_end") or config.get("train_end"),
            "valid_start": request.get("valid_start") or config.get("valid_start"),
            "valid_end": request.get("valid_end") or config.get("valid_end"),
            "test_start": request.get("test_start") or config.get("final_oos_start"),
            "test_end": request.get("test_end") or config.get("final_oos_end"),
            "label_horizon": int(request.get("label_horizon") or config.get("forward_days", 5)),
            "param_profile": request.get("param_profile", "conservative"),
            "enable_cross_sectional": bool(request.get("enable_cross_sectional", False)),
            "enable_winsorize": bool(request.get("enable_winsorize", True)),
        }
        dates = [normalized[name] for name in (
            "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")]
        if not all(dates):
            raise ValueError("研究训练的训练、验证、测试日期必须完整")
        parsed = [dt.date.fromisoformat(value) for value in dates]
        if not all(left < right for left, right in zip(parsed, parsed[1:])):
            raise ValueError("研究训练的训练、验证、测试日期必须完整且按时间递增")
        if normalized["label_horizon"] < 2:
            raise ValueError("label_horizon 至少为 2，避免与 T+1 执行口径错配")
        if normalized["param_profile"] not in ("default", "conservative", "aggressive"):
            raise ValueError("不支持的 param_profile")
        return normalized

    def _run(self, training_id, request):
        try:
            factor_mining_repo.update_training_run(
                training_id, status="running", qlib_job_id=training_id)
            baseline_keys = []
            if request["include_baseline"]:
                _, resolved_baseline, baseline_keys = factor_mining_service._resolve_baseline(
                    request["baseline_set"])
                request["resolved_baseline_set"] = resolved_baseline
            baseline_keys = list(dict.fromkeys(baseline_keys))
            factor_keys = list(dict.fromkeys(baseline_keys + request["candidate_keys"]))
            for key in factor_keys:
                expr, reason = resolve_factor_expression(key)
                if not expr:
                    raise RuntimeError(f"因子 {key} 无法解析: {reason}")

            task_manager.update_progress(training_id, {
                "stage": "qlib_training", "current": 0,
                "total": 2 if baseline_keys and request["train_baseline_control"] else 1,
                "factor_count": len(factor_keys), "candidate_count": len(request["candidate_keys"]),
                "message": "正在训练同口径基线控制组" if baseline_keys and request["train_baseline_control"] else "正在训练候选增强模型",
            })
            from core.adapters.qlib.models import QlibModelManager
            manager = QlibModelManager()
            shared = {
                "factor_set": "WhiteBoxHandler",
                "instruments": request["pool"],
                "train_period": (request["train_start"], request["train_end"]),
                "valid_period": (request["valid_start"], request["valid_end"]),
                "test_period": (request["test_start"], request["test_end"]),
                "label_horizon": request["label_horizon"],
                "enable_cross_sectional": request["enable_cross_sectional"],
                "enable_winsorize": request["enable_winsorize"],
                "param_profile": request["param_profile"],
            }
            baseline_result = None
            if baseline_keys and request["train_baseline_control"]:
                baseline_result = manager.train_dual_model(
                    **shared,
                    experiment_name=f"factor_mining_{request['run_id']}_baseline",
                    custom_factor_keys=baseline_keys,
                )
                task_manager.update_progress(training_id, {
                    "stage": "qlib_training", "current": 1, "total": 2,
                    "factor_count": len(factor_keys), "candidate_count": len(request["candidate_keys"]),
                    "message": "基线完成，正在训练候选增强组",
                })
            result = manager.train_dual_model(
                **shared,
                experiment_name=f"factor_mining_{request['run_id']}",
                custom_factor_keys=factor_keys,
            )
            baseline_ic = baseline_result.get("ic_mean") if baseline_result else None
            enhanced_ic = result.get("ic_mean")
            compact = {
                "factor_keys": factor_keys,
                "candidate_keys": request["candidate_keys"],
                "resolved_baseline_set": request.get("resolved_baseline_set"),
                "ic_mean": result.get("ic_mean"),
                "icir": result.get("icir"),
                "baseline_ic_mean": baseline_ic,
                "ic_delta": (
                    float(enhanced_ic) - float(baseline_ic)
                    if enhanced_ic is not None and baseline_ic is not None else None
                ),
                "lgb_ic_mean": result.get("lgb_ic_mean"),
                "xgb_ic_mean": result.get("xgb_ic_mean"),
                "lgb_weight": result.get("lgb_weight"),
                "xgb_weight": result.get("xgb_weight"),
                "degraded": result.get("degraded") or [],
                "research_only": True,
                "activated": False,
            }
            factor_mining_repo.update_training_run(training_id, status="completed", result=compact)
            task_manager.update_progress(training_id, {
                "stage": "completed",
                "current": 2 if baseline_result else 1,
                "total": 2 if baseline_result else 1,
                "factor_count": len(factor_keys), "message": "候选增强模型训练完成，尚未激活",
            })
            return compact
        except Exception:
            error = traceback.format_exc()
            factor_mining_repo.update_training_run(training_id, status="failed", error=error)
            raise

    @staticmethod
    def get(training_id):
        persisted = factor_mining_repo.get_training_run(training_id)
        if not persisted:
            return None
        task = task_manager.get_dict(training_id) or {}
        return {
            **persisted,
            "status": task.get("status") or persisted["status"],
            "progress": task.get("progress") or {},
            "error": task.get("error") or persisted.get("error") or "",
        }


factor_mining_training_service = FactorMiningTrainingService()
