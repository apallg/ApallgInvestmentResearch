import hashlib
import json
import math
import random

from core.domain.screening.score_pipeline import score_fingerprint


def _variant_fingerprint(name, config):
    payload = json.dumps({"name": name, "config": config or {}}, sort_keys=True,
                         ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scores_for_variant(date_key, raw_scores, spec):
    if "scores_by_date" in spec:
        result = spec["scores_by_date"].get(date_key, {})
    else:
        result = spec["processor"](date_key, dict(raw_scores))
    audit = {}
    if isinstance(result, dict) and "scores" in result:
        audit = dict(result.get("audit") or {})
        result = result["scores"]
    clean = {}
    for code, value in (result or {}).items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean[str(code)] = value
    return clean, audit


def _summary(periods):
    gross_equity = 1.0
    net_equity = 1.0
    peak_equity = 1.0
    max_drawdown = 0.0
    for row in periods:
        gross_equity *= 1 + row["gross_return"]
        net_equity *= 1 + row["net_return"]
        peak_equity = max(peak_equity, net_equity)
        if peak_equity > 0:
            max_drawdown = min(max_drawdown, net_equity / peak_equity - 1)
    count = len(periods)
    return {
        "period_count": count,
        "gross_return": gross_equity - 1,
        "net_return": net_equity - 1,
        "mean_period_return": sum(row["net_return"] for row in periods) / count if count else 0.0,
        "mean_one_way_turnover": sum(row["one_way_turnover"] for row in periods) / count if count else 0.0,
        "total_cost": sum(row["cost"] for row in periods),
        "max_drawdown": max_drawdown,
        "missing_selected_returns": sum(row["missing_return_count"] for row in periods),
    }


def _bootstrap_mean_ci(values, samples=2000, seed=0):
    if len(values) < 2:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]]


def _finite_return(returns, code):
    try:
        value = float(returns[code])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def replay_score_variants(raw_by_date, variants, top_n, realized_returns_by_date=None,
                          baseline="baseline", buy_cost_rate=0.0004,
                          sell_cost_rate=0.0009, missing_return_policy="cash",
                          entry_eligible_by_date=None):
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if missing_return_policy not in ("cash", "error"):
        raise ValueError("missing_return_policy must be cash or error")
    raw_by_date = {str(date_key)[:10]: value for date_key, value in raw_by_date.items()}
    realized_returns_by_date = {
        str(date_key)[:10]: value
        for date_key, value in (realized_returns_by_date or {}).items()
    }
    entry_eligible_by_date = {
        str(date_key)[:10]: set(value)
        for date_key, value in (entry_eligible_by_date or {}).items()
    }
    dates = sorted(raw_by_date)
    output = {"baseline": baseline, "dates": dates, "variants": {}}
    for name, spec in variants.items():
        periods = []
        pretrade_weights = {}
        for date_key in dates:
            raw_scores = raw_by_date[date_key]
            scores, processing_audit = _scores_for_variant(date_key, raw_scores, spec)
            selected = sorted(scores, key=lambda code: (-scores[code], code))[:top_n]
            eligible = entry_eligible_by_date.get(date_key)
            executed = [code for code in selected if eligible is None or code in eligible]
            entry_rejected = [code for code in selected if code not in executed]
            selected_weight = 1 / len(selected) if selected else 0.0
            target = {code: selected_weight for code in executed}
            all_codes = set(pretrade_weights) | set(target)
            buys = sum(max(0.0, target.get(code, 0.0) - pretrade_weights.get(code, 0.0))
                       for code in all_codes)
            sells = sum(max(0.0, pretrade_weights.get(code, 0.0) - target.get(code, 0.0))
                        for code in all_codes)
            cost = buys * buy_cost_rate + sells * sell_cost_rate
            returns = realized_returns_by_date.get(date_key, {})
            period_returns = {code: _finite_return(returns, code) for code in target}
            missing = sorted(code for code, value in period_returns.items() if value is None)
            if missing and missing_return_policy == "error":
                raise ValueError(f"{name} {date_key} missing realized returns: {missing[:5]}")
            realized = {code: (value if value is not None else 0.0)
                        for code, value in period_returns.items()}
            gross_return = sum(target[code] * realized[code] for code in target)
            net_return = (1 - cost) * (1 + gross_return) - 1
            denominator = 1 + gross_return
            if denominator > 0:
                pretrade_weights = {
                    code: weight * (1 + realized[code]) / denominator
                    for code, weight in target.items()
                }
            else:
                pretrade_weights = dict(target)
            periods.append({
                "date": date_key,
                "source_fingerprint": score_fingerprint(raw_scores),
                "selected": selected,
                "executed": executed,
                "entry_rejected": entry_rejected,
                "selected_scores": {code: scores[code] for code in selected},
                "processing_audit": processing_audit,
                "buy_turnover": buys,
                "sell_turnover": sells,
                "one_way_turnover": (buys + sells) / 2,
                "cost": cost,
                "gross_return": gross_return,
                "net_return": net_return,
                "missing_return_count": len(missing),
                "missing_return_codes": missing,
            })
        if periods and pretrade_weights:
            terminal_sell = sum(pretrade_weights.values())
            terminal_cost = terminal_sell * sell_cost_rate
            last = periods[-1]
            last["sell_turnover"] += terminal_sell
            last["one_way_turnover"] = (last["buy_turnover"] + last["sell_turnover"]) / 2
            last["cost"] += terminal_cost
            last["net_return"] = (1 + last["net_return"]) * (1 - terminal_cost) - 1
            last["terminal_liquidation"] = True
        output["variants"][name] = {
            "config": dict(spec.get("config") or {}),
            "variant_fingerprint": _variant_fingerprint(name, spec.get("config")),
            "periods": periods,
            "summary": _summary(periods),
        }

    base = output["variants"].get(baseline)
    if base:
        base_by_date = {row["date"]: row for row in base["periods"]}
        for name, result in output["variants"].items():
            paired = []
            for row in result["periods"]:
                base_row = base_by_date.get(row["date"])
                if base_row is None:
                    continue
                union = set(row["selected"]) | set(base_row["selected"])
                overlap = len(set(row["selected"]) & set(base_row["selected"]))
                paired.append({
                    "date": row["date"],
                    "net_return_delta": row["net_return"] - base_row["net_return"],
                    "turnover_delta": row["one_way_turnover"] - base_row["one_way_turnover"],
                    "selection_jaccard": overlap / len(union) if union else 1.0,
                    "affected": overlap < len(union),
                })
            affected = [row for row in paired if row["affected"]]
            affected_deltas = [row["net_return_delta"] for row in affected]
            result["paired_vs_baseline"] = {
                "periods": paired,
                "mean_net_return_delta": (
                    sum(row["net_return_delta"] for row in paired) / len(paired) if paired else 0.0
                ),
                "mean_turnover_delta": (
                    sum(row["turnover_delta"] for row in paired) / len(paired) if paired else 0.0
                ),
                "mean_selection_jaccard": (
                    sum(row["selection_jaccard"] for row in paired) / len(paired) if paired else 0.0
                ),
                "positive_delta_rate": (
                    sum(row["net_return_delta"] > 0 for row in paired) / len(paired) if paired else 0.0
                ),
                "affected_period_count": len(affected),
                "coverage_rate": len(affected) / len(paired) if paired else 0.0,
                "conditional_mean_net_return_delta": (
                    sum(affected_deltas) / len(affected_deltas) if affected_deltas else None
                ),
                "conditional_positive_delta_rate": (
                    sum(value > 0 for value in affected_deltas) / len(affected_deltas)
                    if affected_deltas else None
                ),
                "conditional_bootstrap_mean_delta_ci95": _bootstrap_mean_ci(affected_deltas),
            }
            if len(paired) > 1:
                deltas = [row["net_return_delta"] for row in paired]
                mean_delta = result["paired_vs_baseline"]["mean_net_return_delta"]
                variance = sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
                standard_error = math.sqrt(variance / len(deltas))
                result["paired_vs_baseline"]["paired_t_stat"] = (
                    mean_delta / standard_error if standard_error > 0 else None
                )
                result["paired_vs_baseline"]["bootstrap_mean_delta_ci95"] = _bootstrap_mean_ci(deltas)
                midpoint = len(deltas) // 2
                result["paired_vs_baseline"]["early_mean_delta"] = (
                    sum(deltas[:midpoint]) / midpoint if midpoint else None
                )
                late = deltas[midpoint:]
                result["paired_vs_baseline"]["late_mean_delta"] = (
                    sum(late) / len(late) if late else None
                )
            else:
                result["paired_vs_baseline"]["paired_t_stat"] = None
                result["paired_vs_baseline"]["bootstrap_mean_delta_ci95"] = [None, None]
                result["paired_vs_baseline"]["early_mean_delta"] = None
                result["paired_vs_baseline"]["late_mean_delta"] = None
    return output
