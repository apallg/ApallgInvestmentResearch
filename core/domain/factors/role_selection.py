import math


def select_core_candidates(summaries, min_folds=3, min_positive_ratio=0.6,
                           min_state_count=2, min_state_floor=-0.005,
                           min_coverage=0.8):
    selected = []
    rejected = []
    for summary in summaries:
        reasons = []
        if int(summary.get("fold_count") or 0) < int(min_folds):
            reasons.append("insufficient_folds")
        if float(summary.get("positive_fold_ratio") or 0.0) < float(min_positive_ratio):
            reasons.append("unstable_oos_direction")
        if int(summary.get("high_vol_count") or 0) < int(min_state_count):
            reasons.append("insufficient_high_vol_evidence")
        if int(summary.get("low_vol_count") or 0) < int(min_state_count):
            reasons.append("insufficient_low_vol_evidence")
        if summary.get("state_floor_ic") is None or float(summary["state_floor_ic"]) < float(min_state_floor):
            reasons.append("state_floor_failed")
        if summary.get("coverage") is not None and float(summary["coverage"]) < float(min_coverage):
            reasons.append("insufficient_coverage")
        item = dict(summary)
        item["selection_reasons"] = reasons
        if reasons:
            rejected.append(item)
        else:
            selected.append(item)
    selected.sort(key=lambda item: (
        item.get("core_robustness") is not None,
        item.get("core_robustness") or float("-inf"),
        item.get("median_oos_ic") or float("-inf"),
    ), reverse=True)
    return selected, rejected


def select_satellite_candidates(summaries, target_state, min_folds=3,
                                min_positive_ratio=0.6, min_target_ic=0.0,
                                outside_floor=-0.01, min_state_specificity=0.0,
                                core_family_counts=None, family_overlap_limit=None):
    target_key = f"{target_state}_ic"
    outside_key = "low_vol_ic" if target_state == "high_vol" else "high_vol_ic"
    selected = []
    rejected = []
    for summary in summaries:
        reasons = []
        target = summary.get(target_key)
        outside = summary.get(outside_key)
        family = summary.get("family") or "未分类"
        if int(summary.get("fold_count") or 0) < int(min_folds):
            reasons.append("insufficient_folds")
        if float(summary.get("positive_fold_ratio") or 0.0) < float(min_positive_ratio):
            reasons.append("unstable_residual_direction")
        if target is None or float(target) <= float(min_target_ic):
            reasons.append("target_state_no_increment")
        if outside is not None and float(outside) < float(outside_floor):
            reasons.append("outside_state_harm")
        if min_state_specificity and target is not None and outside is not None \
                and float(target) - float(outside) < float(min_state_specificity):
            reasons.append("not_state_specific")
        if family_overlap_limit is not None and core_family_counts \
                and int(core_family_counts.get(family, 0)) >= int(family_overlap_limit):
            reasons.append("core_family_saturated")
        item = dict(summary)
        item["target_state"] = target_state
        item["satellite_score"] = (
            float(target) - 0.5 * float(summary.get("oos_mad") or 0.0)
            - max(0.0, -float(outside or 0.0))
            if target is not None else None
        )
        item["selection_reasons"] = reasons
        if reasons:
            rejected.append(item)
        else:
            selected.append(item)
    selected.sort(key=lambda item: item.get("satellite_score") or float("-inf"), reverse=True)
    return selected, rejected


def build_balanced_set(candidates, max_count, family_fraction=0.2):
    if max_count <= 0:
        return []
    family_limit = max(2, int(math.ceil(max_count * family_fraction)))
    selected = []
    family_counts = {}
    for candidate in candidates:
        family = candidate.get("family") or "未分类"
        if family_counts.get(family, 0) >= family_limit:
            continue
        selected.append(candidate)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= max_count:
            break
    return selected


def build_balanced_extension(base_keys, candidates, target_count=15, family_fraction=0.25):
    if target_count <= 0:
        return []
    by_key = {item["factor_key"]: item for item in candidates}
    selected = [by_key[key] for key in base_keys if key in by_key]
    selected_keys = {item["factor_key"] for item in selected}
    family_counts = {}
    for item in selected:
        family = item.get("family") or "未分类"
        family_counts[family] = family_counts.get(family, 0) + 1
    family_limit = max(2, int(math.ceil(target_count * family_fraction)))
    missing_families = {
        item.get("family") or "未分类" for item in candidates
    } - set(family_counts)
    for family in sorted(missing_families):
        item = next((
            candidate for candidate in candidates
            if (candidate.get("family") or "未分类") == family
            and candidate["factor_key"] not in selected_keys
        ), None)
        if item is None:
            continue
        selected.append(item)
        selected_keys.add(item["factor_key"])
        family_counts[family] = 1
        if len(selected) >= target_count:
            return selected
    for item in candidates:
        key = item["factor_key"]
        family = item.get("family") or "未分类"
        if key in selected_keys or family_counts.get(family, 0) >= family_limit:
            continue
        selected.append(item)
        selected_keys.add(key)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= target_count:
            break
    return selected
