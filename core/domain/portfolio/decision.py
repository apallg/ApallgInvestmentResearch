import hashlib
import json
from datetime import date, datetime


LEGACY_POLICY = {
    "policy_id": "legacy",
    "version": 1,
    "mode": "legacy",
    "candidate_pool": 50,
    "buffer_multiple": 2,
    "overlay_weight": 0.0,
    "risk_weight": 0.0,
    "cost_weight": 0.0,
}

SHADOW_V1_POLICY = {
    "policy_id": "shadow_v1",
    "version": 1,
    "mode": "shadow",
    "evidence_status": "unvalidated",
    "candidate_pool": 50,
    "buffer_multiple": 2,
    "overlay_weight": 0.35,
    "risk_weight": 1.0,
    "cost_weight": 1.0,
}

SHADOW_V2_POLICY = {
    "policy_id": "shadow_v2",
    "version": 1,
    "mode": "shadow",
    "evidence_status": "unvalidated",
    "score_mode": "rank_overlay_v2",
    "candidate_pool": 50,
    "buffer_multiple": 2,
    "base_weight": 0.70,
    "overlay_weight": 1.0,
    "risk_weight": 0.0,
    "cost_weight": 0.0,
    "max_rank_jump": 10,
    "min_dimension_coverage": 0.80,
    "min_field_coverage": 0.80,
    "min_field_informative_coverage": 0.20,
    "dimension_weights": {
        "trend_confirmation": 0.10,
        "capital_confirmation": 0.08,
        "defensive": 0.07,
        "liquidity": 0.05,
    },
}

SHADOW_DIMENSION_WEIGHTS = {
    "trend_confirmation": 0.30,
    "capital_confirmation": 0.25,
    "liquidity": 0.15,
    "volatility": 0.15,
    "overnight": 0.10,
    "sentiment": 0.05,
}

SHADOW_V2_DIMENSIONS = {
    "trend_confirmation": (
        ("tech_score", 1),
        ("momentum_score", 1),
        ("wb_industry_score", 1),
    ),
    "capital_confirmation": (
        ("flow_score", 1),
        ("north_flow_score", 1),
        ("wb_moneyflow_score", 1),
    ),
    "defensive": (
        ("volatility_score", 1),
        ("gap_score", 1),
    ),
    "liquidity": (
        ("recent_amount", 1),
        ("illiq_score", 1),
        ("impact_cost", -1),
    ),
}

SHADOW_V2_LABELS = {
    "trend_confirmation": "趋势确认",
    "capital_confirmation": "资金预期",
    "defensive": "低风险",
    "liquidity": "流动性",
}


def _number(value, default=0.0):
    try:
        value = float(value)
        return value if value == value else default
    except (TypeError, ValueError):
        return default


def _optional_number(value):
    try:
        value = float(value)
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def decision_fingerprint(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def holdings_fingerprint_rows(rows):
    normalized = []
    for row in rows or []:
        buy_date = row.get("buy_date")
        if isinstance(buy_date, datetime):
            buy_date = buy_date.date().isoformat()
        elif isinstance(buy_date, date):
            buy_date = buy_date.isoformat()
        elif buy_date is not None:
            buy_date = str(buy_date).strip() or None
            if buy_date:
                try:
                    buy_date = date.fromisoformat(buy_date[:10]).isoformat()
                except ValueError:
                    pass
        normalized.append({
            "code": row.get("code"),
            "shares": int(row.get("shares") or 0),
            "cost_price": float(row.get("cost_price") or 0),
            "buy_date": buy_date,
        })
    return sorted(normalized, key=lambda row: row.get("code") or "")


def decision_can_execute(decision):
    return (
        decision.get("summary", {}).get("mode") != "shadow"
        and decision.get("policy", {}).get("status") == "active"
    )


def _average(row, fields):
    values = [_number(row.get(field)) for field in fields if row.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def build_shadow_overlays(candidates, weights=None):
    weights = dict(weights or SHADOW_DIMENSION_WEIGHTS)
    rows = [dict(candidate) for candidate in candidates]
    raw = {
        "trend_confirmation": [
            _average(row, ("tech_score", "momentum_score", "wb_industry_score")) for row in rows
        ],
        "capital_confirmation": [
            _average(row, ("flow_score", "north_flow_score", "wb_moneyflow_score")) for row in rows
        ],
        "liquidity": [
            _average(row, ("size_turnover_score", "illiq_score")) for row in rows
        ],
        "volatility": [_number(row.get("volatility_score")) for row in rows],
        "overnight": [_number(row.get("gap_score")) for row in rows],
        "sentiment": [_number(row.get("sentiment_score")) for row in rows],
    }
    normalized = {}
    for dimension, values in raw.items():
        unique = sorted(set(values))
        if len(unique) <= 1:
            normalized[dimension] = [0.0] * len(values)
            continue
        ranks = {value: index / (len(unique) - 1) - 0.5 for index, value in enumerate(unique)}
        normalized[dimension] = [ranks[value] for value in values]
    for index, row in enumerate(rows):
        components = {dimension: normalized[dimension][index] for dimension in weights}
        row["overlay_components"] = components
        row["overlay_delta"] = sum(weights[dimension] * components[dimension] for dimension in weights)
        row["cost_penalty"] = max(0.0, _number(row.get("impact_cost", 0)))
    return rows


def _percentile_scores(values, direction=1):
    ranked = [(index, value) for index, value in enumerate(values) if value is not None]
    scores = [0.0] * len(values)
    if len(ranked) <= 1 or len({value for _, value in ranked}) <= 1:
        return scores
    ranked.sort(key=lambda item: item[1])
    position = 0
    while position < len(ranked):
        end = position + 1
        while end < len(ranked) and ranked[end][1] == ranked[position][1]:
            end += 1
        average_position = (position + end - 1) / 2
        score = (average_position / (len(ranked) - 1) - 0.5) * direction
        for offset in range(position, end):
            scores[ranked[offset][0]] = score
        position = end
    return scores


def _dimension_overlay(rows, fields, policy):
    total = len(rows)
    min_field_coverage = _number(policy.get("min_field_coverage"), 0.80)
    min_informative = _number(policy.get("min_field_informative_coverage"), 0.20)
    min_dimension_coverage = _number(policy.get("min_dimension_coverage"), 0.80)
    raw_fields = {}
    active_fields = []
    disabled_fields = []
    field_scores = {}

    for field, direction in fields:
        values = [_optional_number(row.get(field)) for row in rows]
        raw_fields[field] = values
        numeric_coverage = sum(value is not None for value in values) / total if total else 0.0
        informative_coverage = sum(value is not None and abs(value) > 1e-12 for value in values) / total if total else 0.0
        has_variation = len({value for value in values if value is not None}) > 1
        if numeric_coverage >= min_field_coverage and informative_coverage >= min_informative and has_variation:
            active_fields.append(field)
            field_scores[field] = _percentile_scores(values, direction)
        else:
            disabled_fields.append(field)

    informative_rows = sum(
        any(value is not None and abs(value) > 1e-12 for value in (raw_fields[field][index] for field, _ in fields))
        for index in range(total)
    )
    coverage = informative_rows / total if total else 0.0
    active = bool(active_fields) and coverage >= min_dimension_coverage
    if not active:
        disabled_fields = [field for field, _ in fields]
        active_fields = []
        scores = [0.0] * total
    else:
        scores = [
            sum(field_scores[field][index] for field in active_fields) / len(active_fields)
            for index in range(total)
        ]

    if active:
        reason = ""
    elif coverage < min_dimension_coverage:
        reason = f"有效覆盖{coverage:.0%}，低于{min_dimension_coverage:.0%}"
    else:
        reason = "截面无有效差异"
    return scores, {
        "status": "active" if active else "disabled",
        "coverage": coverage,
        "fields_used": active_fields,
        "disabled_fields": disabled_fields,
        "reason": reason,
    }


def build_shadow_v2_overlays(candidates, policy=None):
    active_policy = dict(SHADOW_V2_POLICY)
    if policy:
        active_policy.update(policy)
    weights = dict(SHADOW_V2_POLICY["dimension_weights"])
    weights.update(active_policy.get("dimension_weights") or {})
    rows = [dict(candidate) for candidate in candidates]
    eligible_indices = [index for index, row in enumerate(rows) if not row.get("hard_excluded")]
    eligible_rows = [rows[index] for index in eligible_indices]
    eligible_positions = {row_index: position for position, row_index in enumerate(eligible_indices)}
    total = len(eligible_rows)
    dimensions = {}
    dimension_scores = {}

    for dimension, fields in SHADOW_V2_DIMENSIONS.items():
        scores, quality = _dimension_overlay(eligible_rows, fields, active_policy)
        quality["label"] = SHADOW_V2_LABELS[dimension]
        quality["weight"] = weights.get(dimension, 0.0)
        dimensions[dimension] = quality
        dimension_scores[dimension] = scores

    active_overlay_weight = sum(
        weights.get(dimension, 0.0)
        for dimension, quality in dimensions.items()
        if quality["status"] == "active"
    )
    base_weight = _number(active_policy.get("base_weight"), 0.70)
    for index, row in enumerate(rows):
        eligible_position = eligible_positions.get(index)
        if eligible_position is None:
            components = {dimension: 0.0 for dimension in SHADOW_V2_DIMENSIONS}
            base_rank_score = 0.0
        else:
            components = {
                dimension: dimension_scores[dimension][eligible_position]
                for dimension in SHADOW_V2_DIMENSIONS
            }
            base_rank_score = 0.5 - eligible_position / (total - 1) if total > 1 else 0.0
        contributions = {
            dimension: weights.get(dimension, 0.0) * components[dimension]
            for dimension in SHADOW_V2_DIMENSIONS
        }
        strategy_score = sum(contributions.values())
        positive = sorted(
            ((dimension, value) for dimension, value in contributions.items() if value > 1e-12),
            key=lambda item: item[1],
            reverse=True,
        )
        if eligible_position is None:
            reasons = ["已由硬规则排除"]
        elif positive:
            reasons = [f"{SHADOW_V2_LABELS[dimension]} {value:+.3f}" for dimension, value in positive[:2]]
        else:
            reasons = ["辅助信号未提供正向确认"]
        row.update({
            "base_rank_score": base_rank_score,
            "base_contribution": base_weight * base_rank_score,
            "strategy_components": components,
            "strategy_contributions": contributions,
            "overlay_components": components,
            "overlay_delta": strategy_score,
            "strategy_score": strategy_score,
            "cost_penalty": 0.0,
            "decision_reasons": reasons,
        })
    return rows, {
        "evidence_status": active_policy.get("evidence_status", "unvalidated"),
        "base_weight": base_weight,
        "active_overlay_weight": active_overlay_weight,
        "max_rank_jump": int(active_policy.get("max_rank_jump", 10)),
        "eligible_count": total,
        "dimensions": dimensions,
    }


def _rank_with_promotion_guard(rows, max_rank_jump):
    remaining = list(rows)
    ranked = []
    for position in range(1, len(rows) + 1):
        allowed = [row for row in remaining if row["original_rank"] - position <= max_rank_jump]
        bypassed = not allowed
        if bypassed:
            allowed = [min(remaining, key=lambda row: row["original_rank"])]
        selected = max(allowed, key=lambda row: (row["utility_score"], -row["original_rank"]))
        selected["rank_guard_bypassed"] = bypassed
        ranked.append(selected)
        remaining.remove(selected)
    return ranked


def build_candidate_decision(candidates, top_n, policy=None, current_codes=None):
    active_policy = dict(LEGACY_POLICY)
    if policy:
        active_policy.update(policy)
    top_n = max(1, int(top_n))
    pool_size = max(top_n, int(active_policy.get("candidate_pool", 50)))
    buffer_multiple = max(1, int(active_policy.get("buffer_multiple", 2)))
    current_codes = set(current_codes or [])
    rows = []
    candidate_rows = list(candidates)[:pool_size]

    for index, candidate in enumerate(candidate_rows):
        row = dict(candidate)
        base_score = _number(row.get("base_score", row.get("qlib_score", row.get("l0_score", row.get("score", 0)))))
        overlay_delta = _number(row.get("overlay_delta", 0))
        risk_penalty = max(0.0, _number(row.get("risk_penalty", 0)))
        cost_penalty = max(0.0, _number(row.get("cost_penalty", row.get("impact_cost", 0))))
        if active_policy.get("score_mode") == "rank_overlay_v2":
            fallback_rank_score = 0.5 - index / (len(candidate_rows) - 1) if len(candidate_rows) > 1 else 0.0
            base_rank_score = _number(row.get("base_rank_score"), fallback_rank_score)
            strategy_score = _number(row.get("strategy_score", overlay_delta))
            utility = (
                _number(active_policy.get("base_weight"), 0.70) * base_rank_score
                + _number(active_policy.get("overlay_weight"), 1.0) * strategy_score
                - _number(active_policy.get("risk_weight", 0)) * risk_penalty
                - _number(active_policy.get("cost_weight", 0)) * cost_penalty
            )
        else:
            base_rank_score = _number(row.get("base_rank_score", 0))
            strategy_score = overlay_delta
            utility = (
                base_score
                + _number(active_policy.get("overlay_weight", 0)) * overlay_delta
                - _number(active_policy.get("risk_weight", 0)) * risk_penalty
                - _number(active_policy.get("cost_weight", 0)) * cost_penalty
            )
        row.update({
            "original_rank": index + 1,
            "base_score": base_score,
            "base_rank_score": base_rank_score,
            "strategy_score": strategy_score,
            "overlay_delta": overlay_delta,
            "risk_penalty": risk_penalty,
            "cost_penalty": cost_penalty,
            "utility_score": utility,
            "hard_excluded": bool(row.get("hard_excluded", False)),
            "exclusion_reason": row.get("exclusion_reason") or row.get("reason") or "",
        })
        rows.append(row)

    eligible = [row for row in rows if not row["hard_excluded"]]
    excluded = [row for row in rows if row["hard_excluded"]]
    if active_policy.get("mode") != "legacy":
        if active_policy.get("score_mode") == "rank_overlay_v2":
            eligible = _rank_with_promotion_guard(eligible, max(0, int(active_policy.get("max_rank_jump", 10))))
        else:
            eligible.sort(key=lambda row: (-row["utility_score"], row["original_rank"]))

    selected = eligible[:top_n]
    buffer_limit = top_n * buffer_multiple
    buffer = []
    observe = []
    for rank, row in enumerate(eligible, 1):
        row["final_rank"] = rank
        row["rank_change"] = row["original_rank"] - rank
        reasons = list(row.get("decision_reasons") or [])
        if row["rank_change"] > 0:
            reasons.insert(0, f"较原排序提升{row['rank_change']}名")
        elif row["rank_change"] < 0:
            reasons.insert(0, f"较原排序下降{abs(row['rank_change'])}名")
        row["decision_reasons"] = reasons
        code = row.get("code")
        if rank <= top_n:
            row["decision_status"] = "selected"
        elif rank <= buffer_limit and code in current_codes:
            row["decision_status"] = "hold_buffer"
            buffer.append(row)
        else:
            row["decision_status"] = "observe"
            observe.append(row)
    for row in excluded:
        row["final_rank"] = None
        row["rank_change"] = None
        row["decision_status"] = "excluded"

    ranked = eligible + excluded
    summary = {
        "policy_id": active_policy.get("policy_id", "custom"),
        "policy_version": active_policy.get("version", 1),
        "mode": active_policy.get("mode", "legacy"),
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "buffer_count": len(buffer),
        "excluded_count": len(excluded),
    }
    if active_policy.get("score_mode") == "rank_overlay_v2":
        summary.update({
            "score_mode": "rank_overlay_v2",
            "evidence_status": active_policy.get("evidence_status", "unvalidated"),
            "base_weight": _number(active_policy.get("base_weight"), 0.70),
            "max_rank_jump": max(0, int(active_policy.get("max_rank_jump", 10))),
        })
    return {
        "policy": active_policy,
        "ranked_candidates": ranked,
        "selected": selected,
        "buffer": buffer,
        "observe": observe,
        "excluded": excluded,
        "summary": summary,
        "fingerprint": decision_fingerprint({"policy": active_policy, "rows": ranked}),
    }
