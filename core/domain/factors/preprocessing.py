import hashlib


def expression_fingerprint(expression):
    normalized = "".join(str(expression or "").split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def preprocess_candidates(candidates, min_coverage=0.8):
    eligible = []
    rejected = []
    seen = {}
    for candidate in candidates:
        item = dict(candidate)
        key = str(item.get("factor_key") or "").strip()
        expression = item.get("expression")
        raw_evaluation_mode = item.get("evaluation_mode", "expression")
        evaluation_mode = str(raw_evaluation_mode) if raw_evaluation_mode is not None else ""
        evaluation_signature = str(item.get("evaluation_signature") or key)
        fingerprint = (
            expression_fingerprint(f"isolation:{evaluation_signature}")
            if evaluation_mode == "isolation"
            else expression_fingerprint(expression)
        )
        reason = None
        if not key:
            reason = "missing_factor_key"
        elif evaluation_mode not in ("expression", "isolation"):
            reason = item.get("skip_reason") or "runtime_unresolved_expression"
        elif evaluation_mode == "expression" and not expression:
            reason = item.get("skip_reason") or "unresolved_expression"
        elif str(item.get("data_safety") or "unknown").lower() not in ("safe", "pit", "point_in_time"):
            reason = "unsafe_data"
        elif item.get("coverage") is not None and float(item["coverage"]) < float(min_coverage):
            reason = "insufficient_coverage"
        elif fingerprint in seen:
            reason = f"duplicate_expression:{seen[fingerprint]}"
        item["factor_key"] = key
        item["expression_fingerprint"] = fingerprint
        if reason:
            item["reject_reason"] = reason
            rejected.append(item)
            continue
        seen[fingerprint] = key
        eligible.append(item)
    return {
        "eligible": eligible,
        "rejected": rejected,
        "eligible_count": len(eligible),
        "rejected_count": len(rejected),
    }
