import hashlib
import math


def normalize_overlay_scores(scores):
    valid = [(code, float(value)) for code, value in (scores or {}).items()
             if value is not None and math.isfinite(float(value))]
    if not valid:
        return {}
    valid.sort(key=lambda item: (item[1], item[0]))
    denominator = max(len(valid) - 1, 1)
    result = {}
    index = 0
    while index < len(valid):
        end = index + 1
        while end < len(valid) and valid[end][1] == valid[index][1]:
            end += 1
        rank = ((index + end - 1) / 2.0) / denominator if len(valid) > 1 else 0.5
        for offset in range(index, end):
            result[valid[offset][0]] = rank
        index = end
    return result


def build_factor_overlay(expr, scores, weight=0.15, source="candidate"):
    normalized_weight = max(0.0, min(float(weight), 0.5))
    return {
        "expr": str(expr),
        "source": source,
        "weight": normalized_weight,
        "scores": normalize_overlay_scores(scores),
    }


def overlay_fingerprint(overlay):
    scores = overlay.get("scores") or {}
    payload = "|".join(
        [str(overlay.get("expr", "")), str(overlay.get("weight", 0)),
         *[f"{code}:{scores[code]:.8f}" for code in sorted(scores)]]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
