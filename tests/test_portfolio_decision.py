import pytest

from core.domain.portfolio.decision import (
    SHADOW_V2_POLICY,
    build_candidate_decision,
    build_shadow_overlays,
    build_shadow_v2_overlays,
    decision_can_execute,
    decision_fingerprint,
)


pytestmark = [pytest.mark.p0, pytest.mark.unit]


def test_legacy_policy_preserves_candidate_order():
    candidates = [
        {"code": "A", "qlib_score": 0.1, "overlay_delta": 10},
        {"code": "B", "qlib_score": 0.9, "overlay_delta": -10},
        {"code": "C", "qlib_score": 0.5},
    ]

    result = build_candidate_decision(candidates, 2)

    assert [row["code"] for row in result["selected"]] == ["A", "B"]
    assert [row["original_rank"] for row in result["ranked_candidates"]] == [1, 2, 3]


def test_hard_exclusion_is_not_selected():
    candidates = [
        {"code": "A", "qlib_score": 1, "hard_excluded": True, "exclusion_reason": "停牌"},
        {"code": "B", "qlib_score": 0.9},
    ]

    result = build_candidate_decision(candidates, 1)

    assert [row["code"] for row in result["selected"]] == ["B"]
    assert result["excluded"][0]["decision_status"] == "excluded"


def test_shadow_policy_applies_overlay_only_when_explicit():
    candidates = [
        {"code": "A", "qlib_score": 0.8, "overlay_delta": 0},
        {"code": "B", "qlib_score": 0.7, "overlay_delta": 0.3},
    ]

    legacy = build_candidate_decision(candidates, 1)
    shadow = build_candidate_decision(candidates, 1, {"mode": "shadow", "overlay_weight": 1})

    assert legacy["selected"][0]["code"] == "A"
    assert shadow["selected"][0]["code"] == "B"


def test_current_holding_in_buffer_is_identified():
    candidates = [{"code": code, "qlib_score": 1 - i / 10} for i, code in enumerate("ABCDE")]

    result = build_candidate_decision(candidates, 2, current_codes={"C", "E"})

    assert [row["code"] for row in result["buffer"]] == ["C"]
    assert next(row for row in result["observe"] if row["code"] == "E")["decision_status"] == "observe"


def test_fingerprint_is_order_independent_for_mapping_keys():
    assert decision_fingerprint({"a": 1, "b": 2}) == decision_fingerprint({"b": 2, "a": 1})


def test_shadow_overlay_uses_non_fundamental_dimensions_without_mutating_input():
    candidates = [
        {"code": "A", "tech_score": 1, "momentum_score": 1, "fundamental_score": -99},
        {"code": "B", "tech_score": 0, "momentum_score": 0, "fundamental_score": 99},
    ]

    result = build_shadow_overlays(candidates)

    assert result[0]["overlay_delta"] > result[1]["overlay_delta"]
    assert "overlay_delta" not in candidates[0]
    assert "fundamental" not in result[0]["overlay_components"]


def test_only_active_non_shadow_decision_can_execute():
    assert decision_can_execute({"summary": {"mode": "legacy"}, "policy": {"status": "active"}})
    assert not decision_can_execute({"summary": {"mode": "shadow"}, "policy": {"status": "active"}})
    assert not decision_can_execute({"summary": {"mode": "ranked"}, "policy": {"status": "shadow"}})


def test_shadow_v2_uses_rank_scale_and_disables_sparse_dimensions():
    candidates = []
    for index in range(10):
        candidates.append({
            "code": str(index),
            "qlib_score": 1 - index * 0.001,
            "momentum_score": index / 10,
            "tech_score": index / 20,
            "wb_industry_score": index / 30,
            "flow_score": 1 if index == 0 else 0,
            "north_flow_score": 0,
            "wb_moneyflow_score": 0,
            "volatility_score": 1 - index / 20,
            "gap_score": 1 - index / 30,
            "recent_amount": 100 + index,
            "illiq_score": index / 50,
            "impact_cost": 0.0003,
        })

    rows, audit = build_shadow_v2_overlays(candidates, SHADOW_V2_POLICY)

    assert audit["evidence_status"] == "unvalidated"
    assert rows[0]["base_rank_score"] == pytest.approx(0.5)
    assert rows[-1]["base_rank_score"] == pytest.approx(-0.5)
    assert audit["dimensions"]["capital_confirmation"]["status"] == "disabled"
    assert audit["dimensions"]["capital_confirmation"]["coverage"] == pytest.approx(0.1)
    assert audit["dimensions"]["liquidity"]["fields_used"] == ["recent_amount", "illiq_score"]
    assert "impact_cost" in audit["dimensions"]["liquidity"]["disabled_fields"]
    assert all("strategy_contributions" in row for row in rows)


def test_shadow_v2_rank_guard_prevents_large_promotion():
    candidates = []
    for index in range(20):
        candidates.append({
            "code": str(index + 1),
            "qlib_score": 1 - index / 100,
            "base_rank_score": 0.5 - index / 19,
            "strategy_score": 10 if index == 19 else 0,
        })

    result = build_candidate_decision(candidates, 5, SHADOW_V2_POLICY)
    promoted = next(row for row in result["ranked_candidates"] if row["code"] == "20")

    assert promoted["final_rank"] >= 10
    assert promoted["rank_change"] <= SHADOW_V2_POLICY["max_rank_jump"]
    assert all(row["original_rank"] <= 15 for row in result["selected"])


def test_shadow_v2_missing_dimension_weight_is_not_redistributed():
    candidates = [
        {"code": "A", "momentum_score": 1, "volatility_score": 1, "recent_amount": 2, "illiq_score": 1},
        {"code": "B", "momentum_score": -1, "volatility_score": -1, "recent_amount": 1, "illiq_score": -1},
    ]

    _, audit = build_shadow_v2_overlays(candidates, SHADOW_V2_POLICY)

    assert audit["active_overlay_weight"] == pytest.approx(0.22)


def test_shadow_v2_excluded_rows_do_not_affect_dimension_coverage():
    candidates = [
        {"code": str(index), "hard_excluded": index < 8, "flow_score": 1 if index < 8 else 0}
        for index in range(10)
    ]

    rows, audit = build_shadow_v2_overlays(candidates, SHADOW_V2_POLICY)

    assert audit["eligible_count"] == 2
    assert audit["dimensions"]["capital_confirmation"]["coverage"] == 0
    assert rows[0]["decision_reasons"] == ["已由硬规则排除"]
