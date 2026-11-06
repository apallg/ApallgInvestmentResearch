import statistics


def paired_fold_delta(control, challenger, metric="rank_ic"):
    control_folds = {
        str(item.get("fold_id")): item.get(metric)
        for item in control.get("folds", []) if item.get(metric) is not None
    }
    challenger_folds = {
        str(item.get("fold_id")): item.get(metric)
        for item in challenger.get("folds", []) if item.get(metric) is not None
    }
    values = [
        float(challenger_folds[key]) - float(control_folds[key])
        for key in sorted(control_folds.keys() & challenger_folds.keys())
    ]
    return {
        "fold_count": len(values),
        "median_delta": statistics.median(values) if values else None,
        "mean_delta": sum(values) / len(values) if values else None,
        "positive_ratio": sum(value > 0 for value in values) / len(values) if values else 0.0,
        "fold_deltas": values,
    }


def analyze_core_ab(results, family_arms, min_folds=3, min_positive_ratio=0.5,
                    min_random_percentile=0.75, min_random_count=10):
    full = results.get("core_full") or {}
    family_results = {}
    harmful_families = []
    for family, arm in family_arms.items():
        comparison = paired_fold_delta(results.get(arm) or {}, full)
        keep = (
            comparison["fold_count"] >= int(min_folds)
            and comparison["median_delta"] is not None
            and comparison["median_delta"] > 0
            and comparison["positive_ratio"] >= float(min_positive_ratio)
        )
        family_results[family] = {**comparison, "keep": keep, "arm": arm}
        if not keep:
            harmful_families.append(family)
    random_scores = sorted(
        float(value["rank_ic"])
        for key, value in results.items()
        if key.startswith("random_") and value.get("rank_ic") is not None
    )
    full_score = full.get("rank_ic")
    percentile = (
        sum(score <= float(full_score) for score in random_scores) / len(random_scores)
        if random_scores and full_score is not None else None
    )
    fold_count = len([item for item in full.get("folds", []) if item.get("rank_ic") is not None])
    validated = (
        fold_count >= int(min_folds)
        and full_score is not None
        and float(full_score) > 0
        and percentile is not None
        and percentile >= float(min_random_percentile)
        and len(random_scores) >= int(min_random_count)
        and bool(family_results)
        and not harmful_families
    )
    return {
        "validated": validated,
        "scope": "development",
        "full_rank_ic": full_score,
        "fold_count": fold_count,
        "random_count": len(random_scores),
        "random_percentile": percentile,
        "family_results": family_results,
        "harmful_families": harmful_families,
    }


def analyze_core_oos(results, min_folds=3, min_positive_ratio=0.6,
                     min_random_percentile=0.75, min_random_count=10):
    random_scores = sorted(
        float(value["rank_ic"])
        for key, value in results.items()
        if key.startswith("random_") and value.get("rank_ic") is not None
    )
    candidates = {}
    for arm in ("core_full", "core_reference", "core_minimal"):
        result = results.get(arm) or {}
        score = result.get("rank_ic")
        folds = [item for item in result.get("folds", []) if item.get("rank_ic") is not None]
        fold_positive_ratio = (
            sum(float(item["rank_ic"]) > 0 for item in folds) / len(folds) if folds else 0.0
        )
        percentile = (
            sum(value <= float(score) for value in random_scores) / len(random_scores)
            if random_scores and score is not None else None
        )
        eligible = (
            len(folds) >= int(min_folds)
            and score is not None
            and float(score) > 0
            and fold_positive_ratio >= float(min_positive_ratio)
            and len(random_scores) >= int(min_random_count)
            and percentile is not None
            and percentile >= float(min_random_percentile)
        )
        if result:
            candidates[arm] = {
                "rank_ic": score,
                "icir": result.get("icir"),
                "fold_count": len(folds),
                "positive_fold_ratio": fold_positive_ratio,
                "random_percentile": percentile,
                "eligible": eligible,
            }
    eligible_arms = [arm for arm, item in candidates.items() if item["eligible"]]
    winner_arm = max(
        eligible_arms,
        key=lambda arm: (
            float(candidates[arm]["rank_ic"]),
            float(candidates[arm].get("icir") or float("-inf")),
        ),
        default=None,
    )
    return {
        "validated": winner_arm is not None,
        "scope": "fixed_oos",
        "winner_arm": winner_arm,
        "candidate_results": candidates,
        "random_count": len(random_scores),
    }


def _fold_state_mean(item, target_state):
    by_state = (item or {}).get("daily_ic_by_state") or {}
    state = by_state.get(target_state) or {}
    mean = state.get("mean")
    return float(mean) if mean is not None else None


def paired_fold_delta_state(control, challenger, target_state):
    control_folds = {}
    challenger_folds = {}
    state_days = 0
    for item in control.get("folds", []):
        mean = _fold_state_mean(item, target_state)
        if mean is not None:
            control_folds[str(item.get("fold_id"))] = mean
    for item in challenger.get("folds", []):
        mean = _fold_state_mean(item, target_state)
        if mean is not None:
            challenger_folds[str(item.get("fold_id"))] = mean
    common = sorted(control_folds.keys() & challenger_folds.keys())
    values = [float(challenger_folds[key]) - float(control_folds[key]) for key in common]
    for item in control.get("folds", []):
        if str(item.get("fold_id")) in challenger_folds:
            state = ((item.get("daily_ic_by_state") or {}).get(target_state) or {})
            state_days += int(state.get("n") or 0)
    return {
        "fold_count": len(values),
        "median_delta": statistics.median(values) if values else None,
        "mean_delta": sum(values) / len(values) if values else None,
        "positive_ratio": sum(value > 0 for value in values) / len(values) if values else 0.0,
        "fold_deltas": values,
        "state_days": state_days,
    }


def analyze_satellite_ab(results, role, min_folds=3, min_positive_ratio=0.6,
                         min_median_delta=0.0, min_unconditional_delta=None):
    target_state = "low_vol" if role == "trend" else "high_vol"
    comparison = paired_fold_delta_state(
        results.get("core_control") or {}, results.get(f"core_plus_{role}") or {},
        target_state,
    )
    unconditional = paired_fold_delta(
        results.get("core_control") or {}, results.get(f"core_plus_{role}") or {},
    )
    validated = (
        comparison["fold_count"] >= int(min_folds)
        and comparison["median_delta"] is not None
        and comparison["median_delta"] > float(min_median_delta)
        and comparison["positive_ratio"] >= float(min_positive_ratio)
    )
    if validated and min_unconditional_delta is not None:
        validated = (
            unconditional["median_delta"] is not None
            and unconditional["median_delta"] > float(min_unconditional_delta)
        )
    return {
        "validated": validated,
        "role": role,
        "scope": "state_conditional",
        "target_state": target_state,
        "fold_count": comparison["fold_count"],
        "median_delta": comparison["median_delta"],
        "mean_delta": comparison["mean_delta"],
        "positive_ratio": comparison["positive_ratio"],
        "fold_deltas": comparison["fold_deltas"],
        "state_days": comparison["state_days"],
        "unconditional": unconditional,
    }
