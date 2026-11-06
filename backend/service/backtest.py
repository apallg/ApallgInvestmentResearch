"""回测报告服务 — 把 qlib 原始回测结果整理成标准风险指标和图表数据。"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class BacktestService:

    @staticmethod
    def validate_direction_request(req):
        from core.domain.regime.model_routing import normalize_model_ref, validate_state_models

        params = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        profile = None
        if params.get("profile_id"):
            from backend.service.strategy_profile import StrategyProfileService
            profile = StrategyProfileService.resolve(
                params["profile_id"], params.get("profile_version"),
                require_status={"challenger", "paper", "champion"},
            )
            explicit_fields = set(req.model_fields_set if hasattr(req, "model_fields_set") else req.__fields_set__)
            requested = {key: params.get(key) for key in explicit_fields if key not in {"start", "end", "profile_id", "profile_version"}}
            locked = StrategyProfileService.bind_operation(profile, "backtest", requested)
            params.update(locked)
            windows = profile["config"]["windows"]
            if params["start"] < windows["oos_start"] or params["end"] > windows["oos_end"]:
                raise ValueError(
                    f"回测区间必须位于Profile独立OOS窗口 {windows['oos_start']}~{windows['oos_end']}"
                )
            fold_model = next((item for item in profile["config"]["model"].get("oos_schedule") or []
                               if item.get("window_start") == params["start"] and item.get("window_end") == params["end"]), None)
            if not fold_model:
                raise ValueError("回测区间必须精确匹配Profile冻结的walk-forward折窗口")
            from core.adapters.qlib.model_selection import get_model_for_factor_set
            checked_model = get_model_for_factor_set(
                fold_model["experiment_id"], fold_model["run_id"],
                (profile["config"].get("factors") or {}).get("keys") or [],
                fold_model["window_start"], "a500", allow_test_window=True,
            )
            if checked_model.get("artifact_fingerprint") != fold_model.get("artifact_fingerprint"):
                raise ValueError("walk-forward折模型内容指纹已变化")
            params["fixed_model"] = [fold_model["experiment_id"], fold_model["run_id"]]
            params["_strategy_profile_model"] = fold_model
        params["state_models"] = validate_state_models(
            params.get("state_models"), params.get("use_vol_switch", False)
        )
        params["fixed_model"] = normalize_model_ref(params.get("fixed_model"))
        if params["state_models"] and params["fixed_model"]:
            raise ValueError("固定单模型与状态模型映射不能同时启用")
        params["_strategy_profile"] = profile
        return params

    @staticmethod
    def run_direction(req):
        from core.domain.backtest.engine import DirectionBacktestRunner
        from core.data.repos.factor_panel import FactorPanelRepo

        params = BacktestService.validate_direction_request(req)
        profile = params.pop("_strategy_profile", None)
        profile_model = params.pop("_strategy_profile_model", None)
        conditional_fallback_policy = params.pop("conditional_fallback_policy", "error")
        params.pop("profile_id", None)
        params.pop("profile_version", None)
        if profile:
            from core.data import StrategyProfileRepo
            params["benchmark_membership_by_date"] = StrategyProfileRepo.get_universe_memberships(
                "stable_history_500d", params["start"], params["end"],
            )
            params["security_references_by_date"] = StrategyProfileRepo.get_security_references_between(
                params["start"], params["end"],
            )
        if params["state_models"] and params["use_vol_switch"]:
            try:
                from core.data import SettingsRepo
                instruments = {
                    "csi300": "csi300", "a500": "a500", "stable_history_500d": "a500", "small_cap": "small_cap",
                    "all": "all", "non_csi300": "all",
                }.get(params["pool"], "all")
                remembered = SettingsRepo.get("lgb_lstm_prewarm", {}) or {}
                SettingsRepo.set("lgb_lstm_prewarm", {
                    "models": {key: list(value) for key, value in params["state_models"].items()},
                    "pools": remembered.get("pools") or [instruments],
                })
            except Exception as e:
                logger.warning(f"[LGB+LSTM] 保存回测预热配置失败: {e}")
        conditional_switch = {
            "requested": "disabled",
            "effective": "disabled",
            "fallback_applied": False,
            "fallback_reason": None,
        }
        if params["use_vol_switch"] and params["use_conditional_switch"]:
            from backend.service.market_data import MarketDataService
            conditional_switch["requested"] = "conditional"
            try:
                environment = MarketDataService.get_conditional_market_env(
                    start_date=params["start"], end_date=params["end"]
                )
            except Exception as e:
                environment = {}
                failure_reason = str(e)
            else:
                failure_reason = None if environment else "条件切换环境特征为空"
            if not environment and conditional_fallback_policy == "error":
                raise RuntimeError(failure_reason)
            if not environment:
                logger.warning(f"[VolSwitch] 条件切换不可用, 显式回退二值切换: {failure_reason}")
                params["conditional_market_env"] = {}
                conditional_switch.update({
                    "effective": "binary",
                    "fallback_applied": True,
                    "fallback_reason": failure_reason,
                })
            else:
                params["conditional_market_env"] = environment
                conditional_switch["effective"] = "conditional"
        else:
            params["conditional_market_env"] = {}
            if params["use_vol_switch"]:
                conditional_switch.update({"requested": "binary", "effective": "binary"})
        result = DirectionBacktestRunner(
            latest_adj_factor_loader=FactorPanelRepo.load_latest_adj_factors,
        ).run(**params)
        if profile:
            cfg = profile["config"]
            result["strategy_profile"] = {
                "profile_id": profile["profile_id"],
                "version": profile["version"],
                "status": profile["status"],
                "fingerprint": profile["fingerprint"],
                "data_fingerprint": cfg["data"].get("snapshot_id"),
                "model_fingerprint": cfg.get("fingerprints", {}).get("model_refs"),
                "factor_fingerprint": cfg.get("fingerprints", {}).get("factor"),
                "oos_model": profile_model,
            }
        else:
            result["strategy_profile"] = {"legacy": True, "promotable": False}
        result["conditional_switch"] = conditional_switch
        return result

    @staticmethod
    def build_standard_report(backtest_result: dict) -> dict:
        """从 QlibBacktestRunner 返回值生成标准报告。

        期望 backtest_result 包含:
          - metrics: dict
          - indicator: dict
          - report_normal: pd.DataFrame（可选）
        """
        from core.adapters.qlib.report import build_standard_report as _build_report

        metrics = backtest_result.get("metrics") or {}
        indicator = backtest_result.get("indicator") or {}
        report_normal = backtest_result.get("report_normal")

        report = {
            "metrics": metrics,
            "indicator": indicator,
            "standard": {},
            "curves": {},
        }

        if report_normal is not None:
            try:
                standard = _build_report(report_normal)
                report["standard"] = {
                    "risk": standard.get("risk", {}),
                    "risk_without_cost": standard.get("risk_without_cost", {}),
                    "risk_with_cost": standard.get("risk_with_cost", {}),
                }
                report["curves"] = {
                    "dates": standard.get("dates", []),
                    "cumulative_return": standard.get("cumulative_return", []),
                    "benchmark_return": standard.get("benchmark_return", []),
                    "excess_return_without_cost": standard.get("excess_return_without_cost", []),
                    "excess_return_with_cost": standard.get("excess_return_with_cost", []),
                }
            except Exception as e:
                logger.warning(f"[BacktestService] 标准报告生成失败: {e}")
                report["standard"] = {}
                report["curves"] = {}

        return report
