import pytest

from core.domain.screening.score_pipeline import (
    apply_score_step,
    build_score_result,
    clone_score_result,
    percentile_map,
    score_fingerprint,
)
from core.domain.screening.score_replay import replay_score_variants


@pytest.mark.p0
@pytest.mark.unit
def test_score_pipeline_preserves_raw_neutralized_and_rank_scores():
    raw = {f"{index:06d}": float(index) for index in range(10)}
    neutralized = dict(raw)
    neutralized["000000"], neutralized["000009"] = 9.0, 0.0

    result = build_score_result(raw, neutralized, True, "applied")

    assert result["scores"]["000000"] == pytest.approx(0.8)
    assert result["ledger"]["000000"]["raw_model_score"] == 0.0
    assert result["ledger"]["000000"]["neutralized_score"] == 9.0
    assert result["ledger"]["000000"]["raw_rank"] == 10
    assert result["ledger"]["000000"]["final_rank"] == 1
    assert result["audit"]["score_semantics"] == "cross_sectional_rank_not_expected_return"


@pytest.mark.p0
@pytest.mark.unit
def test_percentile_mapping_keeps_legacy_scale_and_order():
    raw = {f"{index:06d}": float(index) for index in range(10)}
    mapped, percentiles = percentile_map(raw)

    assert mapped["000000"] == pytest.approx(-1.0)
    assert mapped["000009"] == pytest.approx(0.8)
    assert percentiles["000009"] == pytest.approx(0.9)
    assert sorted(mapped, key=mapped.get) == list(raw)


@pytest.mark.p0
@pytest.mark.unit
def test_score_step_records_rank_change_without_losing_source():
    result = build_score_result({"a": 3, "b": 2, "c": 1}, neutralize_requested=False,
                                neutralize_status="disabled")
    source_fingerprint = result["audit"]["source_fingerprint"]

    apply_score_step(result, "macro_sector_bias", {"a": 0, "b": 2, "c": 3})

    assert result["ledger"]["a"]["raw_model_score"] == 3
    assert result["ledger"]["a"]["final_rank"] == 3
    assert result["ledger"]["a"]["score_trace"]["macro_sector_bias"]["rank_delta"] == -2
    assert result["audit"]["source_fingerprint"] == source_fingerprint
    assert result["audit"]["processed_fingerprint"] == score_fingerprint(result["scores"])
    assert result["audit"]["steps"][-1]["input_fingerprint"] != result["audit"]["steps"][-1]["output_fingerprint"]


@pytest.mark.p0
@pytest.mark.unit
def test_cloned_score_result_isolated_from_request_level_adjustments():
    cached = build_score_result({"a": 3, "b": 2, "c": 1}, neutralize_requested=False,
                                neutralize_status="disabled")
    request_result = clone_score_result(cached)

    apply_score_step(request_result, "macro_sector_bias", {"a": 0, "b": 2, "c": 3})

    assert cached["scores"]["a"] > cached["scores"]["c"]
    assert "macro_sector_bias_status" not in cached["audit"]


@pytest.mark.p0
@pytest.mark.unit
def test_score_fingerprint_is_order_independent():
    assert score_fingerprint({"a": 1, "b": 2}) == score_fingerprint({"b": 2, "a": 1})


@pytest.mark.p0
@pytest.mark.unit
def test_rank_fusion_input_is_not_labeled_as_raw_model_score():
    result = build_score_result(
        {"a": 0.9, "b": 0.1}, neutralize_requested=False,
        neutralize_status="disabled",
        model_input_semantics="daily_cross_sectional_rank_fusion",
    )

    assert result["ledger"]["a"]["model_input_score"] == 0.9
    assert result["ledger"]["a"]["model_input_rank"] == 1
    assert result["ledger"]["a"]["raw_model_score"] is None
    assert result["ledger"]["a"]["raw_rank"] is None
    assert result["audit"]["raw_model_score_available"] is False


@pytest.mark.p0
@pytest.mark.unit
def test_counterfactual_replay_pairs_variants_on_same_raw_source_and_costs():
    raw = {
        "2026-01-01": {"a": 3, "b": 2, "c": 1},
        "2026-01-08": {"a": 1, "b": 2, "c": 3},
    }
    returns = {
        "2026-01-01": {"a": 0.10, "b": 0.02, "c": -0.05},
        "2026-01-08": {"a": 0.01, "b": 0.02, "c": 0.03},
    }
    variants = {
        "baseline": {"processor": lambda _, scores: scores, "config": {"mode": "raw"}},
        "reverse": {
            "processor": lambda _, scores: {code: -value for code, value in scores.items()},
            "config": {"mode": "reverse"},
        },
    }

    result = replay_score_variants(raw, variants, 1, returns)

    baseline = result["variants"]["baseline"]
    reverse = result["variants"]["reverse"]
    assert baseline["periods"][0]["selected"] == ["a"]
    assert reverse["periods"][0]["selected"] == ["c"]
    assert baseline["periods"][0]["source_fingerprint"] == reverse["periods"][0]["source_fingerprint"]
    assert baseline["periods"][0]["cost"] == pytest.approx(0.0004)
    assert len(reverse["paired_vs_baseline"]["periods"]) == 2
