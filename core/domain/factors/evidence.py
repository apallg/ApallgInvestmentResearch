import math
import statistics


def median(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def mad(values):
    center = median(values)
    if center is None:
        return None
    return median([abs(float(value) - center) for value in values if value is not None])


def orient_ic(train_ic, value):
    if train_ic is None or value is None:
        return None
    return float(value) * (-1 if float(train_ic) < 0 else 1)


def summarize_factor_evidence(rows):
    by_factor = {}
    for row in rows:
        key = row.get("factor_key")
        state = row.get("state") or "all"
        if not key:
            continue
        item = by_factor.setdefault(key, {
            "factor_key": key,
            "family": row.get("family") or "未分类",
            "folds": set(),
            "all": [],
            "high_vol": [],
            "low_vol": [],
            "coverage": [],
            "l1_pass": [],
            "dedup_pass": [],
        })
        if row.get("fold_id") is not None:
            item["folds"].add(str(row["fold_id"]))
        value = row.get("oriented_ic")
        if value is not None and state in ("all", "high_vol", "low_vol"):
            item[state].append(float(value))
        if row.get("coverage") is not None:
            item["coverage"].append(float(row["coverage"]))
        if state == "all":
            metrics = row.get("metrics") or {}
            if "l1_pass" in metrics:
                item["l1_pass"].append(bool(metrics["l1_pass"]))
            if "dedup_pass" in metrics:
                item["dedup_pass"].append(bool(metrics["dedup_pass"]))

    summaries = []
    for item in by_factor.values():
        all_values = item["all"]
        high_values = item["high_vol"]
        low_values = item["low_vol"]
        all_median = median(all_values)
        high_median = median(high_values)
        low_median = median(low_values)
        dispersion = mad(all_values) or 0.0
        state_floor = min(high_median, low_median) if high_median is not None and low_median is not None else None
        summaries.append({
            "factor_key": item["factor_key"],
            "family": item["family"],
            "fold_count": len(item["folds"]),
            "median_oos_ic": all_median,
            "oos_mad": dispersion,
            "positive_fold_ratio": (
                sum(value > 0 for value in all_values) / len(all_values) if all_values else 0.0
            ),
            "high_vol_ic": high_median,
            "low_vol_ic": low_median,
            "high_vol_count": len(high_values),
            "low_vol_count": len(low_values),
            "state_floor_ic": state_floor,
            "core_robustness": (
                state_floor - 0.5 * dispersion if state_floor is not None else None
            ),
            "coverage": median(item["coverage"]),
            "l1_pass_ratio": (
                sum(item["l1_pass"]) / len(item["l1_pass"])
                if item["l1_pass"] else None
            ),
            "dedup_pass_ratio": (
                sum(item["dedup_pass"]) / len(item["dedup_pass"])
                if item["dedup_pass"] else None
            ),
        })
    return summaries


def summarize_ic_values(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"ic_mean": None, "icir": None, "positive_ratio": 0.0, "count": 0}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    std = variance ** 0.5
    return {
        "ic_mean": mean,
        "icir": mean / std if std > 1e-12 else mean,
        "positive_ratio": sum(value > 0 for value in clean) / len(clean),
        "count": len(clean),
    }


def regroup_evidence_by_state(all_rows, state_map, run_id, family_map=None):
    family_map = family_map or {}
    result = []
    for source in all_rows:
        row = dict(source)
        row["run_id"] = run_id
        row["state"] = "all"
        row["family"] = family_map.get(row.get("factor_key"), row.get("family") or "未分类")
        result.append(row)
        metrics = row.get("metrics") or {}
        daily_pairs = [
            (str(pair[0])[:10], float(pair[1]))
            for pair in (metrics.get("daily_ics") or [])
            if len(pair) >= 2 and pair[1] is not None
        ]
        direction_sign = int(row.get("direction_sign") or 1)
        for state in ("high_vol", "low_vol"):
            values = [value for date, value in daily_pairs if state_map.get(date) == state]
            if not values:
                continue
            stats = summarize_ic_values(values)
            result.append({
                **row,
                "state": state,
                "raw_ic": stats["ic_mean"] * direction_sign,
                "oriented_ic": stats["ic_mean"],
                "icir": stats["icir"],
                "positive_ratio": stats["positive_ratio"],
                "sample_count": stats["count"],
                "metrics": {
                    "state_source": "market_regime_history_volatility_axis_daily_v1",
                    "l1_pass": metrics.get("l1_pass"),
                    "dedup_pass": metrics.get("dedup_pass"),
                },
            })
    return result


def orient_residual_rows_walkforward(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("factor_key"), []).append(dict(row))
    result = []
    for factor_rows in grouped.values():
        fold_values = {}
        for row in factor_rows:
            fold_values.setdefault(str(row.get("fold_id")), []).append(float(row["residual_ic"]))
        history = []
        signs = {}
        for fold_id in sorted(fold_values):
            if history:
                signs[fold_id] = -1 if median(history) < 0 else 1
            history.extend(fold_values[fold_id])
        for row in factor_rows:
            fold_id = str(row.get("fold_id"))
            if fold_id not in signs:
                continue
            row["direction_sign"] = signs[fold_id]
            row["oriented_ic"] = float(row["residual_ic"]) * signs[fold_id]
            result.append(row)
    return result
