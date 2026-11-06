import hashlib
import json
import math
from copy import deepcopy
from bisect import bisect_left


SCORE_PIPELINE_VERSION = "score-ledger-v2"


def clone_score_result(score_result):
    return deepcopy(score_result)


def _finite_scores(scores):
    result = {}
    for code, value in (scores or {}).items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[str(code)] = value
    return result


def score_fingerprint(scores):
    clean = _finite_scores(scores)
    payload = [[code, format(clean[code], ".17g")] for code in sorted(clean)]
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def value_fingerprint(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def percentile_map(scores):
    clean = _finite_scores(scores)
    if not clean:
        return {}, {}
    ordered = sorted(clean.values())
    count = len(ordered)
    percentiles = {
        code: bisect_left(ordered, value) / count
        for code, value in clean.items()
    }
    mapped = {code: (percentile - 0.5) * 2 for code, percentile in percentiles.items()}
    return mapped, percentiles


def _rank_positions(scores):
    clean = _finite_scores(scores)
    ordered = sorted(clean, key=lambda code: (-clean[code], code))
    return {code: index + 1 for index, code in enumerate(ordered)}


def build_score_result(raw_scores, basis_scores=None, neutralize_requested=True,
                       neutralize_status="applied", neutralize_error=None,
                       context=None, model_input_semantics="raw_model_score"):
    raw = _finite_scores(raw_scores)
    basis = _finite_scores(raw if basis_scores is None else basis_scores)
    common = set(raw) & set(basis)
    raw = {code: raw[code] for code in raw if code in common}
    basis = {code: basis[code] for code in basis if code in common}
    mapped, percentiles = percentile_map(basis)
    raw_ranks = _rank_positions(raw)
    basis_ranks = _rank_positions(basis)
    final_ranks = _rank_positions(mapped)
    ledger = {}
    for code in mapped:
        ledger[code] = {
            "model_input_score": raw[code],
            "model_input_rank": raw_ranks[code],
            "model_input_semantics": model_input_semantics,
            "raw_model_score": raw[code] if model_input_semantics == "raw_model_score" else None,
            "raw_rank": raw_ranks[code] if model_input_semantics == "raw_model_score" else None,
            "neutralized_score": basis[code],
            "neutralized_rank": basis_ranks[code],
            "rank_percentile": percentiles[code],
            "final_rank_score": mapped[code],
            "final_rank": final_ranks[code],
            "score_trace": {
                "model_input": {
                    "score": raw[code], "rank": raw_ranks[code],
                    "semantics": model_input_semantics,
                },
                "neutralize": {
                    "score": basis[code],
                    "rank": basis_ranks[code],
                    "rank_delta": raw_ranks[code] - basis_ranks[code],
                    "status": neutralize_status,
                },
                "percentile": {
                    "percentile": percentiles[code],
                    "score": mapped[code],
                    "rank": final_ranks[code],
                    "status": "applied",
                },
            },
        }
    audit = {
        "pipeline_version": SCORE_PIPELINE_VERSION,
        "source_fingerprint": score_fingerprint(raw),
        "basis_fingerprint": score_fingerprint(basis),
        "processed_fingerprint": score_fingerprint(mapped),
        "input_count": len(raw_scores or {}),
        "valid_count": len(mapped),
        "raw_score_available": model_input_semantics == "raw_model_score",
        "model_input_score_available": True,
        "raw_model_score_available": model_input_semantics == "raw_model_score",
        "model_input_semantics": model_input_semantics,
        "neutralize_qlib": bool(neutralize_requested),
        "neutralize_status": neutralize_status,
        "neutralize_qlib_applied": neutralize_status in ("applied", "partial"),
        "percentile_mapping": True,
        "percentile_status": "applied",
        "score_semantics": "cross_sectional_rank_not_expected_return",
        "steps": [
            {"name": "model", "status": "applied"},
            {"name": "neutralize", "status": neutralize_status},
            {"name": "percentile", "status": "applied"},
        ],
    }
    if neutralize_error:
        audit["neutralize_error"] = str(neutralize_error)
    if context:
        audit["context"] = dict(context)
    return {"scores": mapped, "ledger": ledger, "audit": audit}


def build_legacy_processed_result(scores, context=None):
    clean = _finite_scores(scores)
    ranks = _rank_positions(clean)
    ledger = {
        code: {
            "model_input_score": None,
            "model_input_rank": None,
            "model_input_semantics": "legacy_processed_score_unknown_provenance",
            "raw_model_score": None,
            "raw_rank": None,
            "neutralized_score": None,
            "neutralized_rank": None,
            "rank_percentile": None,
            "final_rank_score": value,
            "final_rank": ranks[code],
            "score_trace": {
                "legacy_processed_cache": {
                    "score": value,
                    "rank": ranks[code],
                    "status": "source_unavailable",
                }
            },
        }
        for code, value in clean.items()
    }
    audit = {
        "pipeline_version": SCORE_PIPELINE_VERSION,
        "source_fingerprint": None,
        "basis_fingerprint": score_fingerprint(clean),
        "processed_fingerprint": score_fingerprint(clean),
        "input_count": len(scores or {}),
        "valid_count": len(clean),
        "raw_score_available": False,
        "neutralize_status": "unknown_legacy_cache",
        "neutralize_qlib_applied": None,
        "percentile_mapping": None,
        "percentile_status": "unknown_legacy_cache",
        "score_semantics": "legacy_processed_score_unknown_provenance",
        "steps": [{"name": "legacy_processed_cache", "status": "source_unavailable"}],
    }
    if context:
        audit["context"] = dict(context)
    return {"scores": clean, "ledger": ledger, "audit": audit}


def apply_score_step(score_result, name, scores, status="applied", detail=None):
    input_fingerprint = score_fingerprint(score_result.get("scores", {}))
    clean = _finite_scores(scores)
    ledger = score_result.get("ledger", {})
    clean = {code: value for code, value in clean.items() if code in ledger}
    ranks = _rank_positions(clean)
    for code, row in ledger.items():
        if code not in clean:
            continue
        previous_rank = row.get("final_rank")
        row["final_rank_score"] = clean[code]
        row["final_rank"] = ranks[code]
        row.setdefault("score_trace", {})[name] = {
            "score": clean[code],
            "rank": ranks[code],
            "rank_delta": (previous_rank - ranks[code]) if previous_rank is not None else 0,
            "status": status,
        }
    score_result["scores"] = clean
    output_fingerprint = score_fingerprint(clean)
    step = {
        "name": name,
        "status": status,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": output_fingerprint,
    }
    if detail:
        step["detail"] = str(detail)
    score_result.setdefault("audit", {}).setdefault("steps", []).append(step)
    score_result["audit"][f"{name}_status"] = status
    score_result["audit"]["processed_fingerprint"] = output_fingerprint
    return score_result


def selected_score_audit(score_result, codes):
    ledger = score_result.get("ledger", {})
    return {
        "audit": dict(score_result.get("audit", {})),
        "selected": {code: dict(ledger[code]) for code in codes if code in ledger},
    }
