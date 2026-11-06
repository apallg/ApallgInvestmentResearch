import hashlib
import math

from core.domain.factors.candidate import canonical_expression


def expression_fingerprint(expr):
    canonical = canonical_expression(expr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def campaign_seed(base_seed, campaign_id, batch_index):
    payload = f"{int(base_seed)}:{campaign_id}:{int(batch_index)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def novelty_decision(expr, known_fingerprints=None, max_correlation=None, threshold=0.9):
    fingerprint = expression_fingerprint(expr)
    if fingerprint in set(known_fingerprints or ()):
        return {
            "accepted": False,
            "fingerprint": fingerprint,
            "reason": "历史任务已挖掘等价表达式",
            "novelty_score": 0.0,
        }
    if max_correlation is not None and math.isfinite(float(max_correlation)):
        correlation = abs(float(max_correlation))
        if correlation >= float(threshold):
            return {
                "accepted": False,
                "fingerprint": fingerprint,
                "reason": f"与历史或因子池行为相关性 {correlation:.2f} 超过 {threshold:.2f}",
                "novelty_score": max(0.0, 1.0 - correlation),
            }
        score = max(0.0, 1.0 - correlation)
    else:
        score = 1.0
    return {
        "accepted": True,
        "fingerprint": fingerprint,
        "reason": "",
        "novelty_score": score,
    }


def rank_hall_of_fame(candidates, limit):
    best = {}
    for candidate in candidates or []:
        fingerprint = candidate.get("expr_fingerprint") or expression_fingerprint(candidate.get("expr"))
        score = float(
            candidate.get("train_fitness")
            or (candidate.get("train_metrics") or {}).get("fitness")
            or 0.0
        )
        current = best.get(fingerprint)
        current_score = float(
            (current or {}).get("train_fitness")
            or ((current or {}).get("train_metrics") or {}).get("fitness")
            or 0.0
        )
        if current is None or score > current_score:
            best[fingerprint] = {
                **candidate,
                "expr_fingerprint": fingerprint,
                "train_fitness": score,
            }
    ranked = sorted(
        best.values(),
        key=lambda item: (
            float(item.get("train_fitness") or 0.0),
            float(item.get("novelty_score") or 0.0),
            -int(item.get("complexity") or 0),
        ),
        reverse=True,
    )
    return ranked[:max(0, int(limit))]
