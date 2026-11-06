import math


def adjust_selection_bias(oos_ic, candidate_count, fold_count, penalty_scale=0.0025):
    count = max(int(candidate_count or 0), 1)
    folds = max(int(fold_count or 0), 1)
    penalty = float(penalty_scale) * math.sqrt(math.log2(count)) * math.sqrt(5.0 / folds)
    raw = float(oos_ic or 0.0)
    return {
        "raw_oos_ic": raw,
        "selection_haircut": penalty,
        "selection_adjusted_oos_ic": raw - penalty,
        "selection_candidate_count": count,
        "selection_fold_count": folds,
        "selection_method": "log_candidate_haircut",
    }
