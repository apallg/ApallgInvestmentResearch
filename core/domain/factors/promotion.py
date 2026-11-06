

DEFAULT_THRESHOLDS = {
    "oos_ic": 0.02,
    "oos_icir": 0.30,
    "positive_ratio": 0.60,
    "passed_folds": 3,
    "folds": 5,
    "decay_max": 0.50,
    "max_correlation": 0.75,
    "coverage": 0.80,
    "baseline_margin": 0.0,
    "selection_adjusted_oos_ic": 0.02,
}


def evaluate_promotion_gate(metrics, thresholds=None, baseline=None):
    rules = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons = []
    oos_ic = float(metrics.get("oos_ic") or 0.0)
    oos_icir = float(metrics.get("oos_icir", metrics.get("icir")) or 0.0)
    positive = float(metrics.get("positive_ratio") or 0.0)
    passed_folds = int(metrics.get("passed_folds") or 0)
    folds = int(metrics.get("folds") or rules["folds"])
    decay = metrics.get("decay")
    coverage = float(metrics.get("coverage") or 0.0)
    max_corr = float(metrics.get("max_correlation") or 0.0)
    adjusted_oos_ic = metrics.get("selection_adjusted_oos_ic")
    if oos_ic < rules["oos_ic"]:
        reasons.append(f"OOS IC {oos_ic:.4f} < {rules['oos_ic']:.4f}")
    if adjusted_oos_ic is not None and float(adjusted_oos_ic) < rules["selection_adjusted_oos_ic"]:
        reasons.append(
            f"候选选择校正 OOS IC {float(adjusted_oos_ic):.4f} < "
            f"{rules['selection_adjusted_oos_ic']:.4f}")
    if oos_icir < rules["oos_icir"]:
        reasons.append(f"OOS ICIR {oos_icir:.4f} < {rules['oos_icir']:.4f}")
    if positive < rules["positive_ratio"]:
        reasons.append(f"正 IC 比例 {positive:.1%} < {rules['positive_ratio']:.1%}")
    if passed_folds < rules["passed_folds"] or folds < rules["folds"]:
        reasons.append(f"滚动折通过 {passed_folds}/{folds} < {rules['passed_folds']}/{rules['folds']}")
    if decay is not None and float(decay) > rules["decay_max"]:
        reasons.append(f"训练到验证衰减 {float(decay):.1%} > {rules['decay_max']:.1%}")
    if max_corr >= rules["max_correlation"]:
        reasons.append(f"与现有因子最大相关性 {max_corr:.3f} >= {rules['max_correlation']:.3f}")
    if coverage < rules["coverage"]:
        reasons.append(f"覆盖率 {coverage:.1%} < {rules['coverage']:.1%}")
    baseline_result = (baseline or {}).get("net_return")
    strategy_result = metrics.get("strategy_net_return")
    if metrics.get("strategy_passed") is False:
        reasons.extend(metrics.get("strategy_reasons") or ["策略级 A/B 未通过"])
    if metrics.get("oos_independent") is False:
        reasons.append("最终 OOS 与验证区间重叠，不允许晋级")
    if baseline_result is not None and strategy_result is not None and strategy_result < baseline_result + rules["baseline_margin"]:
        reasons.append("策略级增量未超过基线")
    return {"passed": not reasons, "reasons": reasons, "thresholds": rules}
