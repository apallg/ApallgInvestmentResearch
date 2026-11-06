import hashlib
import datetime
import json
import math
import random


PROFILE_STATUSES = {
    "draft", "challenger", "paper", "champion", "superseded", "blocked", "retired",
}

DEFAULT_DIAGNOSTIC_CONFIG = {
    "objective": {
        "primary": "post_cost_excess_calmar",
        "primary_benchmark": "stable_history_500d_equal_weight",
        "secondary_benchmark": "csi300",
    },
    "universe": {
        "name": "stable_history_500d",
        "min_history_days": 500,
        "point_in_time": True,
        "code_prefixes": ["600", "601", "603", "605", "000", "001", "002", "003"],
    },
    "data": {
        "snapshot_id": None,
        "price_adjustment": "qfq_point_in_time",
        "require_tradable_filter": True,
    },
    "factors": {
        "set_name": "core_base_30",
        "version": None,
        "keys": [],
        "evidence": [],
        "research_only": False,
    },
    "model": {
        "refs": [],
        "shadow_refs": [],
        "oos_schedule": [],
        "fusion": "single",
        "weights": [],
        "label_mode": "tradable_open_to_open",
        "label_horizon": 10,
    },
    "training": {
        "train_years": 5,
        "valid_months": 6,
        "test_months": 3,
        "step_months": 3,
        "purge_days": 11,
        "min_oos_folds": 5,
        "random_split": False,
    },
    "windows": {
        "train_start": None,
        "train_end": None,
        "valid_start": None,
        "valid_end": None,
        "oos_start": None,
        "oos_end": None,
    },
    "score": {
        "primary": "raw_score",
        "neutralize": False,
        "percentile_mapping": True,
        "percentile_semantics": "same_date_cross_sectional_rank_only",
        "absolute_gate": "absolute_gate_v1_shadow",
        "legacy_shadow_v2_semantics": "rank_overlay",
    },
    "selection": {
        "candidate_pool": 50,
        "top_n": 5,
        "frequency": "week",
        "sizing": "equal",
    },
    "portfolio": {
        "target_exposure": 0.95,
        "max_position_pct": 0.20,
        "max_sector_pct": 0.30,
        "max_one_way_turnover": 0.30,
    },
    "execution": {
        "decision_time": "close",
        "fill_time": "t_plus_1_open",
        "commission_rate": 0.0003,
        "slippage_rate": 0.0001,
        "stamp_duty": 0.0005,
        "respect_limit_rules": True,
        "respect_suspension": True,
        "refill_failed_buy": False,
        "manual_confirmation": True,
    },
    "features": {
        "timing": False,
        "macro_bias": False,
        "macro_regime": False,
        "sector_fusion": False,
        "model_routing": False,
        "vol_switch": False,
        "hmm": False,
        "crf": False,
        "stop_source": "off",
        "cooldown_stop_weeks": 0,
        "cooldown_profit_weeks": 0,
        "intraday_changes_selection": False,
    },
    "promotion": {
        "max_drawdown_floor": -0.20,
        "max_one_way_turnover": 0.30,
        "min_oos_folds": 5,
        "min_positive_folds": 3,
        "bootstrap_lower_bound": 0.0,
        "max_single_fold_contribution": 0.50,
        "min_paper_periods": 20,
        "signal_consistency": 1.0,
    },
    "experiment": {
        "kind": "diagnostic_baseline",
        "changed_field": None,
    },
}


def _merge(base, override):
    result = {}
    for key, value in base.items():
        if isinstance(value, dict):
            incoming = override.get(key, {}) if isinstance(override, dict) else {}
            result[key] = _merge(value, incoming if isinstance(incoming, dict) else {})
        else:
            result[key] = override.get(key, value) if isinstance(override, dict) else value
    if isinstance(override, dict):
        for key, value in override.items():
            if key not in result:
                result[key] = value
    return result


def canonical_pool_name(value):
    return "stable_history_500d" if value in ("a500", "stable_history_500d") else value


def normalize_profile_config(config):
    result = _merge(DEFAULT_DIAGNOSTIC_CONFIG, config or {})
    result["universe"]["name"] = canonical_pool_name(result["universe"].get("name"))
    result["universe"]["code_prefixes"] = sorted(set(result["universe"].get("code_prefixes") or []))
    result["factors"]["keys"] = list(dict.fromkeys(str(key) for key in result["factors"].get("keys") or []))
    refs = []
    for ref in result["model"].get("refs") or []:
        if isinstance(ref, dict):
            exp_id, run_id = ref.get("experiment_id"), ref.get("run_id")
        elif isinstance(ref, (list, tuple)) and len(ref) == 2:
            exp_id, run_id = ref
        else:
            continue
        if exp_id and run_id:
            normalized = {"experiment_id": str(exp_id), "run_id": str(run_id)}
            if isinstance(ref, dict) and ref.get("artifact_fingerprint"):
                normalized["artifact_fingerprint"] = str(ref["artifact_fingerprint"])
            refs.append(normalized)
    result["model"]["refs"] = refs
    shadow_refs = []
    for ref in result["model"].get("shadow_refs") or []:
        if not isinstance(ref, dict) or not ref.get("experiment_id") or not ref.get("run_id"):
            continue
        normalized = {
            "experiment_id": str(ref["experiment_id"]),
            "run_id": str(ref["run_id"]),
            "role": str(ref.get("role") or "observer"),
            "factor_set_name": str(ref.get("factor_set_name") or ""),
            "factor_set_version": str(ref.get("factor_set_version") or ""),
            "factor_keys": list(dict.fromkeys(str(key) for key in ref.get("factor_keys") or [])),
            "affects_selection": bool(ref.get("affects_selection", False)),
            "research_only": bool(ref.get("research_only", True)),
        }
        for key in ("artifact_fingerprint", "factor_definitions_fingerprint"):
            if ref.get(key):
                normalized[key] = str(ref[key])
        shadow_refs.append(normalized)
    result["model"]["shadow_refs"] = shadow_refs
    schedule = []
    for item in result["model"].get("oos_schedule") or []:
        if not isinstance(item, dict):
            continue
        schedule.append({
            key: str(item[key])
            for key in ("window_start", "window_end", "experiment_id", "run_id", "artifact_fingerprint", "train_start", "train_end", "valid_start", "valid_end")
            if item.get(key) is not None
        })
    result["model"]["oos_schedule"] = schedule
    weights = [float(value) for value in result["model"].get("weights") or []]
    if refs and not weights:
        weights = [1.0 / len(refs)] * len(refs)
    total = sum(max(0.0, value) for value in weights)
    result["model"]["weights"] = [max(0.0, value) / total for value in weights] if total else []
    return result


def profile_fingerprint(config):
    payload = json.dumps(
        normalize_profile_config(config), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_profile_contract(config):
    cfg = normalize_profile_config(config)
    errors = []
    warnings = []
    universe = cfg["universe"]
    model = cfg["model"]
    training = cfg["training"]
    execution = cfg["execution"]
    selection = cfg["selection"]
    portfolio = cfg["portfolio"]
    windows = cfg["windows"]

    if universe.get("name") != "stable_history_500d":
        errors.append("生产Profile必须使用稳定历史池500D")
    if int(universe.get("min_history_days") or 0) != 500 or not universe.get("point_in_time"):
        errors.append("股票池必须按决策时点以前500个有效交易观测构造")
    if not model.get("refs"):
        errors.append("必须绑定精确模型experiment_id/run_id，禁止自动选择最新模型")
    elif len(model["refs"]) != 1:
        errors.append("运行时Profile必须绑定一个已落盘模型run，融合模型也必须先固化为独立run")
    if len(model.get("weights") or []) != len(model.get("refs") or []):
        errors.append("模型融合权重数量与模型数量不一致")
    for ref in model.get("refs") or []:
        fingerprint = str(ref.get("artifact_fingerprint") or "")
        if len(fingerprint) != 64:
            errors.append(f"模型 {ref.get('run_id')} 必须冻结64位内容指纹")
    primary_models = {(ref["experiment_id"], ref["run_id"]) for ref in model.get("refs") or []}
    if len(model.get("shadow_refs") or []) > 3:
        errors.append("影子模型最多允许3个，避免预测资源失控")
    for ref in model.get("shadow_refs") or []:
        if (ref["experiment_id"], ref["run_id"]) in primary_models:
            errors.append("影子模型不得与主模型重复")
        if ref.get("affects_selection"):
            errors.append("影子模型不得改变排名、选股或仓位")
        if not ref.get("research_only"):
            errors.append("影子模型必须标记为仅研究")
        if not ref.get("factor_set_name") or not ref.get("factor_set_version") or not ref.get("factor_keys"):
            errors.append(f"影子模型 {ref.get('run_id')} 必须冻结因子集名称、版本和顺序")
        if len(str(ref.get("artifact_fingerprint") or "")) != 64:
            errors.append(f"影子模型 {ref.get('run_id')} 必须冻结64位内容指纹")
        if len(str(ref.get("factor_definitions_fingerprint") or "")) != 64:
            errors.append(f"影子模型 {ref.get('run_id')} 必须冻结因子表达式指纹")
    schedule = model.get("oos_schedule") or []
    if len(schedule) < 5:
        errors.append("必须冻结至少5个季度walk-forward折模型")
    schedule_windows = []
    schedule_models = []
    for item in schedule:
        required = ("window_start", "window_end", "experiment_id", "run_id", "artifact_fingerprint", "train_start", "train_end", "valid_start", "valid_end")
        if any(not item.get(field) for field in required):
            errors.append("walk-forward折缺少窗口、训练截止或精确模型内容指纹")
            continue
        try:
            start = datetime.date.fromisoformat(item["window_start"][:10])
            end = datetime.date.fromisoformat(item["window_end"][:10])
            train_start = datetime.date.fromisoformat(item["train_start"][:10])
            train_end = datetime.date.fromisoformat(item["train_end"][:10])
            valid_start = datetime.date.fromisoformat(item["valid_start"][:10])
            valid_end = datetime.date.fromisoformat(item["valid_end"][:10])
            if not 75 <= (end - start).days <= 100:
                errors.append("walk-forward折必须覆盖完整3个月")
            if not train_start < train_end < valid_start < valid_end < start:
                errors.append("walk-forward折训练、验证和OOS窗口顺序无效")
            if not 1750 <= (train_end - train_start).days <= 1860:
                errors.append("walk-forward折训练窗口必须接近完整5年")
            if not 140 <= (valid_end - valid_start).days <= 190:
                errors.append("walk-forward折验证窗口必须接近完整6个月并保留purge")
            if len(item["artifact_fingerprint"]) != 64:
                errors.append("walk-forward折模型必须冻结64位内容指纹")
            schedule_windows.append((start, end))
            schedule_models.append((item["experiment_id"], item["run_id"]))
        except ValueError:
            errors.append("walk-forward折包含无效日期")
    ordered_schedule = sorted(schedule_windows)
    if any(current[0] <= previous[1] for previous, current in zip(ordered_schedule, ordered_schedule[1:])):
        errors.append("walk-forward折窗口不得重复或重叠")
    if len(schedule_models) != len(set(schedule_models)):
        errors.append("每个walk-forward折必须绑定独立模型run")
    if model.get("weights") and abs(sum(model["weights"]) - 1.0) > 1e-8:
        errors.append("模型融合权重之和必须为1")
    if model.get("label_mode") != "tradable_open_to_open" or int(model.get("label_horizon") or 0) != 10:
        errors.append("标签必须为T+1开盘到T+11开盘，周期10日")
    if training.get("random_split"):
        errors.append("训练和评估禁止随机切分")
    expected = {"train_years": 5, "valid_months": 6, "test_months": 3, "step_months": 3, "purge_days": 11}
    for field, value in expected.items():
        if int(training.get(field) or 0) != value:
            errors.append(f"训练合同 {field} 必须为 {value}")
    if int(training.get("min_oos_folds") or 0) < 5:
        errors.append("至少需要5个完整OOS折")
    if not all(windows.get(field) for field in ("train_start", "train_end", "valid_start", "valid_end", "oos_start", "oos_end")):
        errors.append("必须冻结训练、验证和独立OOS起止日期")
    elif not (
        str(windows["train_start"]) <= str(windows["train_end"]) < str(windows["valid_start"])
        <= str(windows["valid_end"]) < str(windows["oos_start"]) <= str(windows["oos_end"])
    ):
        errors.append("训练、验证和独立OOS窗口必须严格按时间顺序且不重叠")
    elif schedule_windows and any(
        start < datetime.date.fromisoformat(str(windows["oos_start"])[:10])
        or end > datetime.date.fromisoformat(str(windows["oos_end"])[:10])
        for start, end in schedule_windows
    ):
        errors.append("walk-forward折必须完全位于Profile独立OOS窗口")
    if selection.get("frequency") != "week" or int(selection.get("top_n") or 0) != 5:
        errors.append("生产基线必须为周频Top5")
    if selection.get("sizing") != "equal":
        errors.append("生产基线必须使用Top5等权")
    if cfg["data"].get("price_adjustment") != "qfq_point_in_time":
        errors.append("生产证据必须使用决策时点可见的前复权价格")
    if execution.get("decision_time") != "close" or execution.get("fill_time") != "t_plus_1_open":
        errors.append("决策与成交时点必须为D日收盘决策、T+1开盘成交")
    expected_costs = {"commission_rate": 0.0003, "slippage_rate": 0.0001, "stamp_duty": 0.0005}
    for field, expected_cost in expected_costs.items():
        try:
            actual_cost = float(execution.get(field))
        except (TypeError, ValueError):
            actual_cost = float("nan")
        if not math.isfinite(actual_cost) or actual_cost < 0 or abs(actual_cost - expected_cost) > 1e-12:
            errors.append(f"生产成本 {field} 必须冻结为 {expected_cost}")
    if not execution.get("respect_limit_rules") or not execution.get("respect_suspension"):
        errors.append("必须模拟涨跌停和停牌约束")
    if execution.get("refill_failed_buy"):
        errors.append("买入失败后禁止递补")
    if float(portfolio.get("target_exposure") or 0) > 0.95:
        errors.append("目标总仓位不得超过95%")
    if float(portfolio.get("max_position_pct") or 0) > 0.20:
        errors.append("单股仓位不得超过20%")
    if float(portfolio.get("max_sector_pct") or 0) > 0.30:
        errors.append("单行业仓位不得超过30%")
    if float(portfolio.get("max_one_way_turnover") or 0) > 0.30:
        errors.append("周度单边换手率不得超过30%")
    if cfg["score"].get("percentile_semantics") != "same_date_cross_sectional_rank_only":
        errors.append("分位分必须标记为仅同日横截面排序")
    if cfg["features"].get("intraday_changes_selection"):
        errors.append("日内辅助信号不得改变目标股票或权重")
    controlled_sections = ("score", "selection", "portfolio", "features")
    changed = []
    for section in controlled_sections:
        current = cfg[section]
        baseline = DEFAULT_DIAGNOSTIC_CONFIG[section]
        for key in sorted(set(current) | set(baseline)):
            if current.get(key) != baseline.get(key):
                changed.append(f"{section}.{key}")
    experiment = cfg.get("experiment") or {}
    if experiment.get("kind") == "diagnostic_baseline":
        if changed:
            errors.append("诊断基线不得同时启用或修改增强功能: " + ", ".join(changed))
    elif experiment.get("kind") == "single_change":
        declared = experiment.get("changed_field")
        valid_research_change = not changed and declared in ("model", "factors")
        if not valid_research_change and (len(changed) != 1 or declared != (changed[0] if changed else None)):
            errors.append("单变量消融必须且只能修改experiment.changed_field声明的一项")
    elif experiment.get("kind") == "research_candidate":
        if not cfg["factors"].get("research_only"):
            errors.append("研究候选必须标记为仅研究")
    else:
        errors.append("experiment.kind只能是诊断基线、单变量消融或研究候选")
    factor_keys = cfg["factors"].get("keys") or []
    expected_factor_counts = {
        "core_base_30": 30,
        "core_trend_satellite": 40,
        "core_rebound_satellite": 41,
    }
    expected_factor_count = expected_factor_counts.get(cfg["factors"].get("set_name"))
    if not factor_keys:
        errors.append("必须冻结非空的可追溯因子清单")
    elif expected_factor_count and len(factor_keys) != expected_factor_count:
        errors.append(
            f"{cfg['factors'].get('set_name')}必须冻结为{expected_factor_count}个可追溯因子"
        )
    if len(str((cfg.get("fingerprints") or {}).get("factor_definitions") or "")) != 64:
        errors.append("必须冻结因子表达式内容指纹")
    evidence = {row.get("key"): row for row in cfg["factors"].get("evidence") or [] if isinstance(row, dict)}
    if cfg["factors"].get("research_only"):
        errors.append("该版本是研究候选策略，不得晋级为正式策略")
    for key in ([] if cfg["factors"].get("research_only") else factor_keys):
        row = evidence.get(key)
        if not row:
            errors.append(f"因子 {key} 缺少OOF准入证据")
            continue
        if float(row.get("rank_ic") or 0) < 0.02:
            errors.append(f"因子 {key} OOS RankIC低于0.02")
        if float(row.get("icir") or 0) < 0.30:
            errors.append(f"因子 {key} ICIR低于0.30")
        if float(row.get("positive_date_ratio") or 0) < 0.60:
            errors.append(f"因子 {key} 正IC日期比例低于60%")
        if int(row.get("positive_folds") or 0) < 3 or int(row.get("fold_count") or 0) < 5:
            errors.append(f"因子 {key} 至少需5折且3折为正")
        train_ic = abs(float(row.get("train_rank_ic") or 0))
        oos_ic = abs(float(row.get("rank_ic") or 0))
        if train_ic > 0 and oos_ic / train_ic < 0.50:
            errors.append(f"因子 {key} 训练到OOS衰减超过50%")
        if float(row.get("coverage") or 0) < 0.80:
            errors.append(f"因子 {key} 有效覆盖率低于80%")
    if not cfg["data"].get("snapshot_id"):
        errors.append("必须绑定数据快照")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "config": cfg}


def operation_config(config, operation):
    cfg = normalize_profile_config(config)
    universe = cfg["universe"]
    model = cfg["model"]
    selection = cfg["selection"]
    portfolio = cfg["portfolio"]
    execution = cfg["execution"]
    features = cfg["features"]
    model_ref = None
    if model["refs"]:
        first = model["refs"][0]
        model_ref = [first["experiment_id"], first["run_id"]]
    common = {
        "pool": universe["name"],
        "code_prefixes": universe["code_prefixes"],
    }
    if operation == "screening":
        return {
            **common,
            "top_n": int(selection["candidate_pool"]),
            "qlib_model": model_ref,
            "macro_bias": bool(features["macro_bias"]),
            "sector_fusion": bool(features["sector_fusion"]),
            "macro_regime": bool(features["macro_regime"]),
            "smooth_regime": bool(features["crf"]),
            "use_vol_switch": bool(features["vol_switch"]),
            "state_models": {},
            "neutralize_qlib": bool(cfg["score"]["neutralize"]),
            "decision_day_filter": bool(cfg["data"]["require_tradable_filter"]),
            "shadow_models": [
                {
                    "model": [ref["experiment_id"], ref["run_id"]],
                    "role": ref["role"],
                    "factor_set_name": ref["factor_set_name"],
                    "affects_selection": False,
                }
                for ref in model.get("shadow_refs") or []
            ],
        }
    if operation == "backtest":
        return {
            **common,
            "top_n": int(selection["top_n"]),
            "freq": selection["frequency"],
            "sizing": selection["sizing"],
            "target_exposure": float(portfolio["target_exposure"]),
            "max_one_way_turnover": float(portfolio["max_one_way_turnover"]),
            "fixed_model": model_ref,
            "use_hmm": bool(features["hmm"]),
            "use_timing": bool(features["timing"]),
            "full_position": not bool(features["timing"]),
            "use_crf": bool(features["crf"]),
            "use_vol_switch": bool(features["vol_switch"]),
            "state_models": {},
            "commission_rate": float(execution["commission_rate"]),
            "slippage_rate": float(execution["slippage_rate"]),
            "stamp_duty": float(execution["stamp_duty"]),
            "stop_source": features["stop_source"],
            "macro_bias": bool(features["macro_bias"]),
            "macro_regime": bool(features["macro_regime"]),
            "neutralize_qlib": bool(cfg["score"]["neutralize"]),
            "sector_fusion": bool(features["sector_fusion"]),
            "decision_day_filter": bool(cfg["data"]["require_tradable_filter"]),
            "max_position_pct": float(portfolio["max_position_pct"]),
            "max_sector_pct": float(portfolio["max_sector_pct"]),
            "cash_yield": False,
        }
    if operation == "rebalance":
        return {
            "top_n": int(selection["top_n"]),
            "sizing": selection["sizing"],
            "frequency": "weekly",
            "code_prefixes": universe["code_prefixes"],
            "cooldown_stop_weeks": int(features["cooldown_stop_weeks"]),
            "cooldown_profit_weeks": int(features["cooldown_profit_weeks"]),
        }
    raise ValueError(f"未知Profile操作: {operation}")


def assert_single_change(base_config, candidate_config):
    base = normalize_profile_config(base_config)
    candidate = normalize_profile_config(candidate_config)
    changed = []

    def walk(left, right, prefix=""):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            lv, rv = left.get(key), right.get(key)
            if isinstance(lv, dict) and isinstance(rv, dict):
                walk(lv, rv, path)
            elif lv != rv:
                changed.append(path)

    walk(base, candidate)
    meaningful = []
    for path in changed:
        if path == "data.snapshot_id" or path.startswith("fingerprints.") or path.startswith("factors.definitions") or path.startswith("experiment."):
            continue
        grouped = "model" if path.startswith("model.") else "factors" if path.startswith("factors.") else path
        if grouped not in meaningful:
            meaningful.append(grouped)
    return {"valid": len(meaningful) == 1, "changed_fields": meaningful}


def bootstrap_lower_bound(date_excess_returns, samples=2000):
    values = [float(value) for value in date_excess_returns if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    seed = int(hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return means[max(0, int(samples * 0.025) - 1)]


def evaluate_oos_evidence(folds, champion_excess_calmar=None):
    normalized = []
    all_date_excess = []
    for fold in folds or []:
        row = dict(fold)
        row["annual_excess"] = float(row.get("annual_excess") or 0)
        row["max_drawdown"] = float(row.get("max_drawdown") or 0)
        row["one_way_turnover"] = float(row.get("one_way_turnover") or 0)
        row["excess_calmar"] = float(row.get("excess_calmar") or 0)
        row["date_excess_returns"] = [float(value) for value in row.get("date_excess_returns") or []]
        row["decision_dates"] = [str(value)[:10] for value in row.get("decision_dates") or []]
        row["benchmark_coverage"] = float(row.get("benchmark_coverage") or 0)
        all_date_excess.extend(row["date_excess_returns"])
        normalized.append(row)

    failures = []
    if len(normalized) < 5:
        failures.append("完整OOS折少于5个")
    windows = []
    for row in normalized:
        start = str(row.get("window_start") or "")[:10]
        end = str(row.get("window_end") or "")[:10]
        if not start or not end or start > end:
            failures.append("OOS折缺少有效的起止窗口")
            continue
        windows.append((start, end))
        try:
            start_date = datetime.date.fromisoformat(start)
            end_date = datetime.date.fromisoformat(end)
            if not 75 <= (end_date - start_date).days <= 100:
                failures.append("每个OOS折必须覆盖完整3个月窗口")
            parsed_dates = [datetime.date.fromisoformat(value) for value in row["decision_dates"]]
            unique_decision_weeks = {value.isocalendar()[:2] for value in parsed_dates}
            if len(parsed_dates) < 10 or len(unique_decision_weeks) < 10:
                failures.append("每个OOS折至少需要10个独立周度决策日期")
            if len(unique_decision_weeks) != len(parsed_dates):
                failures.append("OOS折每个自然周只能包含一个决策日期")
            if len(parsed_dates) != len(row["date_excess_returns"]):
                failures.append("OOS折的周度决策日期与超额收益数量不一致")
            if any(value < start_date or value > end_date for value in parsed_dates):
                failures.append("OOS折的决策日期超出冻结窗口")
        except ValueError:
            failures.append("OOS折包含无效日期")
    if len(set(windows)) != len(windows):
        failures.append("OOS折时间窗口重复")
    ordered_windows = sorted(windows)
    if any(current[0] <= previous[1] for previous, current in zip(ordered_windows, ordered_windows[1:])):
        failures.append("OOS折时间窗口重叠")
    positive = sum(row["annual_excess"] > 0 for row in normalized)
    if positive < 3:
        failures.append("净超额为正的OOS折少于3个")
    aggregate_excess = sum(row["annual_excess"] for row in normalized) / len(normalized) if normalized else 0
    if aggregate_excess <= 0:
        failures.append("聚合OOS净超额不为正")
    worst_drawdown = min((row["max_drawdown"] for row in normalized), default=0)
    if worst_drawdown < -0.20:
        failures.append("最大回撤差于-20%")
    max_turnover = max((row["one_way_turnover"] for row in normalized), default=0)
    if max_turnover > 0.30:
        failures.append("周度单边换手率超过30%")
    if normalized and min(row["benchmark_coverage"] for row in normalized) < 0.80:
        failures.append("同池等权基准有效覆盖率低于80%")
    excess_calmar = sum(row["excess_calmar"] for row in normalized) / len(normalized) if normalized else 0
    if champion_excess_calmar is not None and excess_calmar <= float(champion_excess_calmar):
        failures.append("Excess Calmar未超过当前Champion")
    lower_bound = bootstrap_lower_bound(all_date_excess)
    if lower_bound is None or lower_bound < 0:
        failures.append("决策日期聚类Bootstrap 95%下界小于0")
    positive_contributions = [max(0.0, row["annual_excess"]) for row in normalized]
    positive_total = sum(positive_contributions)
    max_contribution = max(positive_contributions, default=0) / positive_total if positive_total > 0 else 1.0
    if max_contribution > 0.50:
        failures.append("单一OOS折贡献超过50%")
    return {
        "passed": not failures,
        "failures": failures,
        "summary": {
            "fold_count": len(normalized),
            "positive_fold_count": positive,
            "annual_excess": aggregate_excess,
            "max_drawdown": worst_drawdown,
            "max_one_way_turnover": max_turnover,
            "excess_calmar": excess_calmar,
            "bootstrap_lower_bound": lower_bound,
            "max_single_fold_contribution": max_contribution,
        },
        "folds": normalized,
    }


def evaluate_paper_evidence(periods, champion_excess_calmar=None):
    valid = [row for row in periods or [] if row.get("matured") and row.get("valid")]
    failures = []
    decision_dates = []
    try:
        decision_dates = [datetime.date.fromisoformat(str(row.get("decision_date") or "")[:10]) for row in valid]
    except ValueError:
        failures.append("模拟盘包含无效决策日期")
    unique_weeks = {value.isocalendar()[:2] for value in decision_dates}
    if valid and (len(decision_dates) != len(set(decision_dates)) or len(unique_weeks) != len(valid)):
        failures.append("模拟盘必须由互不重复的独立周度决策期组成")
    if len(valid) < 20:
        failures.append(f"有效成熟模拟盘周期不足20期，当前{len(valid)}期")
    excess_returns = [float(row.get("excess_return") or 0) for row in valid]
    excess_equity = 1.0
    for value in excess_returns:
        excess_equity *= 1.0 + value
    net_excess = excess_equity - 1.0
    if valid and net_excess <= 0:
        failures.append("模拟盘净超额不为正")
    max_drawdown = min((float(row.get("drawdown") or 0) for row in valid), default=0)
    if max_drawdown < -0.20:
        failures.append("模拟盘最大回撤差于-20%")
    turnover = max((float(row.get("one_way_turnover") or 0) for row in valid), default=0)
    if turnover > 0.30:
        failures.append("模拟盘周度单边换手率超过30%")
    consistency = min((float(row.get("signal_consistency") or 0) for row in valid), default=0)
    if consistency < 1.0:
        failures.append("模拟盘信号与正式选股结果未达到100%一致")
    annual_excess = (excess_equity ** (52 / len(valid)) - 1.0) if valid and excess_equity > 0 else 0.0
    excess_calmar = annual_excess / abs(max_drawdown) if max_drawdown < 0 else (999.0 if annual_excess > 0 else 0.0)
    if champion_excess_calmar is not None and excess_calmar <= float(champion_excess_calmar):
        failures.append("模拟盘Excess Calmar未超过当前Champion")
    return {
        "passed": not failures,
        "failures": failures,
        "summary": {
            "valid_periods": len(valid),
            "net_excess": net_excess,
            "annual_excess": annual_excess,
            "max_drawdown": max_drawdown,
            "max_one_way_turnover": turnover,
            "signal_consistency": consistency,
            "excess_calmar": excess_calmar,
        },
    }
