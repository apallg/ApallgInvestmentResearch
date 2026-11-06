import math

from core.domain.portfolio.strategy_profile import bootstrap_lower_bound
from core.domain.screening.score_pipeline import percentile_map, value_fingerprint


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def fit_absolute_gate(oof_rows, bins=10):
    rows = []
    for raw in oof_rows or []:
        score = _finite(raw.get("raw_model_score"))
        realized = _finite(raw.get("realized_tradable_return"))
        date_key = str(raw.get("decision_date") or "")[:10]
        if not raw.get("is_oof"):
            raise ValueError("absolute_gate_v1只能使用明确标记为OOF的预测")
        if score is not None and realized is not None and date_key:
            rows.append({"date": date_key, "code": str(raw.get("code") or ""), "score": score, "realized": realized})
    dates = sorted({row["date"] for row in rows})
    if len(dates) < 20:
        raise ValueError("absolute_gate_v1至少需要20个独立OOF决策日")
    by_date = {}
    for row in rows:
        by_date.setdefault(row["date"], []).append(row)
    calibrated = []
    date_features = []
    for date_key, group in by_date.items():
        mapped, percentiles = percentile_map({row["code"]: row["score"] for row in group})
        scores = [row["score"] for row in group]
        mean = sum(scores) / len(scores)
        dispersion = (sum((value - mean) ** 2 for value in scores) / len(scores)) ** 0.5
        top = [row["realized"] for row in group if percentiles.get(row["code"], 0) >= 0.8]
        date_features.append({
            "date": date_key,
            "dispersion": dispersion,
            "quality": 1.0 if top and sum(top) / len(top) > 0 else 0.0,
        })
        for row in group:
            calibrated.append({**row, "percentile": percentiles[row["code"]]})
    bin_rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        values = [
            row["realized"] for row in calibrated
            if row["percentile"] >= low and (row["percentile"] < high or index == bins - 1)
        ]
        if not values:
            bin_rows.append({"low": low, "high": high, "expected_return": None, "ci_low": None, "ci_high": None, "count": 0})
            continue
        ordered = sorted(values)
        bin_rows.append({
            "low": low,
            "high": high,
            "expected_return": sum(values) / len(values),
            "ci_low": bootstrap_lower_bound(values),
            "ci_high": ordered[min(len(ordered) - 1, int(len(ordered) * 0.975))],
            "count": len(values),
        })
    dispersions = sorted(row["dispersion"] for row in date_features)
    thresholds = [
        dispersions[min(len(dispersions) - 1, int(len(dispersions) / 3))],
        dispersions[min(len(dispersions) - 1, int(len(dispersions) * 2 / 3))],
    ]
    quality_buckets = []
    for index in range(3):
        values = [
            row["quality"] for row in date_features
            if (index == 0 and row["dispersion"] <= thresholds[0])
            or (index == 1 and thresholds[0] < row["dispersion"] <= thresholds[1])
            or (index == 2 and row["dispersion"] > thresholds[1])
        ]
        quality_buckets.append(sum(values) / len(values) if values else 0.0)
    model = {
        "version": "absolute_gate_v1",
        "source": "oof_only",
        "shadow": True,
        "decision_dates": dates,
        "rank_bins": bin_rows,
        "dispersion_thresholds": thresholds,
        "date_quality_probability": quality_buckets,
        "training_rows": len(calibrated),
    }
    model["fingerprint"] = value_fingerprint(model)
    return model


def apply_absolute_gate(model, raw_scores):
    if not model or model.get("version") != "absolute_gate_v1" or model.get("source") != "oof_only":
        raise ValueError("absolute_gate_v1模型缺失或来源不是OOF")
    clean = {str(code): value for code, raw in (raw_scores or {}).items() if (value := _finite(raw)) is not None}
    _, percentiles = percentile_map(clean)
    values = list(clean.values())
    mean = sum(values) / len(values) if values else 0.0
    dispersion = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5 if values else 0.0
    thresholds = model["dispersion_thresholds"]
    bucket = 0 if dispersion <= thresholds[0] else 1 if dispersion <= thresholds[1] else 2
    date_quality = float(model["date_quality_probability"][bucket])
    result = {}
    for code, percentile in percentiles.items():
        row = next((item for item in model["rank_bins"] if percentile >= item["low"] and percentile <= item["high"]), None)
        result[code] = {
            "expected_tradable_return": row.get("expected_return") if row else None,
            "calibration_interval": [row.get("ci_low"), row.get("ci_high")] if row else [None, None],
            "date_quality_probability": date_quality,
            "rank_percentile": percentile,
            "percentile_semantics": "same_date_cross_sectional_rank_only",
            "shadow": True,
            "changes_ranking": False,
            "model_fingerprint": model["fingerprint"],
        }
    return {
        "version": "absolute_gate_v1",
        "shadow": True,
        "date_quality_probability": date_quality,
        "dispersion": dispersion,
        "predictions": result,
    }
