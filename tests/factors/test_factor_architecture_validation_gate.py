import pytest

from backend.service.factor_architecture import (
    FactorArchitectureService,
    _fingerprint,
    _build_validation_evidence,
    _has_current_validation_evidence,
    _validation_policy,
)


pytestmark = pytest.mark.unit


def _core_set():
    factor_set = {
        "set_id": "run_core_v1",
        "run_id": "run",
        "role": "core",
        "status": "validated",
        "fingerprint": "set-fingerprint",
        "factor_keys": ["a", "b"],
        "metrics": {},
    }
    analysis = {
        "validated": True,
        "scope": "fixed_oos",
        "winner_arm": "core_full",
        "random_count": 10,
        "candidate_results": {
            "core_full": {
                "eligible": True,
                "fold_count": 3,
                "rank_ic": 0.02,
                "icir": 0.5,
                "positive_fold_ratio": 2 / 3,
                "random_percentile": 1.0,
            },
        },
    }
    results = {
        "core_full": {
            "factor_keys": ["a", "b"], "rank_ic": 0.02, "icir": 0.5,
            "folds": [
                {"fold_id": "f1", "rank_ic": 0.03},
                {"fold_id": "f2", "rank_ic": -0.01},
                {"fold_id": "f3", "rank_ic": 0.04},
            ],
        },
    }
    for index in range(1, 11):
        results[f"random_{index:02d}"] = {
            "factor_keys": [f"r{index}"], "rank_ic": -0.01, "icir": -0.2,
            "folds": [
                {"fold_id": "f1", "rank_ic": -0.02},
                {"fold_id": "f2", "rank_ic": 0.01},
                {"fold_id": "f3", "rank_ic": -0.01},
            ],
        }
    factor_set["metrics"]["validation_evidence"] = _build_validation_evidence(
        "run", factor_set, "core", "fixed_oos", analysis,
        {"min_folds": 3, "oos_positive_ratio": 0.6, "min_random_percentile": 0.75,
         "min_random_count": 1}, results,
    )
    factor_set["_test_results"] = results
    return factor_set


def test_validation_evidence_is_bound_to_exact_factor_set_and_analysis():
    factor_set = _core_set()

    assert _has_current_validation_evidence(factor_set, "run", "core", "fixed_oos")

    factor_set["factor_keys"] = ["a", "changed"]
    assert not _has_current_validation_evidence(factor_set, "run", "core", "fixed_oos")


def test_publication_policy_thresholds_cannot_be_lowered_by_request_config():
    policy = _validation_policy("core", "fixed_oos", {
        "min_folds": 1,
        "oos_positive_ratio": 0.1,
        "min_random_percentile": 0.1,
        "min_random_count": 1,
    })

    assert policy["min_folds"] == 3
    assert policy["min_positive_ratio"] == 0.6
    assert policy["min_random_percentile"] == 0.75
    assert policy["min_random_count"] == 10


def test_status_validated_without_current_attestation_is_not_publishable():
    factor_set = _core_set()
    factor_set["metrics"].pop("validation_evidence")

    assert not FactorArchitectureService._has_persisted_validation_evidence(
        "run", factor_set, "core", "fixed_oos",
    )


def test_persisted_validation_folds_must_match_attested_fold_ids(monkeypatch):
    factor_set = _core_set()
    evidence = factor_set["metrics"]["validation_evidence"]
    rows = []
    for arm, result in factor_set["_test_results"].items():
        rows.append({
            "arm": arm, "fold_id": None,
            "metrics": {key: value for key, value in result.items() if key != "folds"},
        })
        rows.extend({"arm": arm, "fold_id": fold["fold_id"], "metrics": fold} for fold in result["folds"])
    factor_set.pop("_test_results")
    monkeypatch.setattr(
        "backend.service.factor_architecture.factor_architecture_repo.list_evaluations",
        lambda *_args, **_kwargs: rows,
    )

    assert FactorArchitectureService._has_persisted_validation_evidence(
        "run", factor_set, "core", "fixed_oos",
    )

    rows[-1]["metrics"]["rank_ic"] = 0.4
    assert not FactorArchitectureService._has_persisted_validation_evidence(
        "run", factor_set, "core", "fixed_oos",
    )


def test_validation_thresholds_are_rechecked_even_if_attestation_hash_is_rewritten():
    factor_set = _core_set()
    evidence = factor_set["metrics"]["validation_evidence"]
    winner = evidence["analysis"]["candidate_results"]["core_full"]
    winner["rank_ic"] = -0.01
    evidence["analysis_fingerprint"] = _fingerprint(evidence["analysis"])

    assert not _has_current_validation_evidence(factor_set, "run", "core", "fixed_oos")


def test_malformed_validation_metrics_fail_closed():
    factor_set = _core_set()
    evidence = factor_set["metrics"]["validation_evidence"]
    evidence["analysis"]["candidate_results"]["core_full"]["positive_fold_ratio"] = "not-a-ratio"
    evidence["analysis_fingerprint"] = _fingerprint(evidence["analysis"])

    assert not _has_current_validation_evidence(factor_set, "run", "core", "fixed_oos")


def test_malformed_attestation_objects_fail_closed():
    factor_set = _core_set()
    factor_set["metrics"]["validation_evidence"]["analysis"] = []

    assert not _has_current_validation_evidence(factor_set, "run", "core", "fixed_oos")


def test_fixed_oos_winner_can_reference_evaluations_saved_under_target_set():
    factor_set = {
        "set_id": "winning_reference_set",
        "fingerprint": "set-fingerprint",
        "factor_keys": ["a", "b"],
    }
    results = {
        "core_full": {"factor_keys": ["c"], "rank_ic": 0.02, "folds": [{"fold_id": "f1"}]},
    }
    analysis = {"validated": False, "scope": "fixed_oos"}
    evidence = _build_validation_evidence(
        "run", factor_set, "core", "fixed_oos", analysis, {}, results,
        evaluation_set_id="evaluated_target_set",
    )

    assert evidence["set_id"] == "winning_reference_set"
    assert evidence["evaluation_set_id"] == "evaluated_target_set"


def test_publish_satellites_rejects_legacy_validated_status_without_evidence(monkeypatch):
    service = FactorArchitectureService()
    sets = [
        {"set_id": "core", "role": "core", "status": "validated", "factor_keys": ["core_factor"]},
        {"set_id": "trend", "role": "trend", "status": "validated", "factor_keys": ["trend_factor"]},
    ]
    monkeypatch.setattr(
        "backend.service.factor_architecture.factor_architecture_repo.get_run",
        lambda _run_id: {"run_id": "run"},
    )
    monkeypatch.setattr(
        "backend.service.factor_architecture.factor_architecture_repo.list_sets",
        lambda _run_id: sets,
    )
    monkeypatch.setattr(
        "backend.service.factor_architecture.factor_architecture_repo.list_evaluations",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "core.domain.factors.registry.get_pool",
        lambda: pytest.fail("legacy status must be rejected before the factor pool is accessed"),
    )

    with pytest.raises(ValueError, match="固定OOS证据"):
        service.publish_satellites("run")


def test_publish_satellites_marks_verified_sets_research_only(monkeypatch):
    service = FactorArchitectureService()
    core = {
        "set_id": "core", "role": "core", "status": "validated", "version_no": 2,
        "factor_keys": ["core_factor"], "metrics": {"validation_evidence": {"test": "core"}},
    }
    trend = {
        "set_id": "trend", "role": "trend", "status": "validated", "version_no": 1,
        "factor_keys": ["trend_factor"], "metrics": {"validation_evidence": {"test": "trend"}},
    }
    candidates = [
        {"factor_key": key, "metadata": {
            "label": key, "expression": "$close", "data_safety": "safe", "family": "趋势",
        }}
        for key in ("core_factor", "trend_factor")
    ]
    saved = []

    class Pool:
        def save_curated_set(self, *args, **kwargs):
            saved.append((args, kwargs))

    monkeypatch.setattr(
        "backend.service.factor_architecture.factor_architecture_repo.get_run",
        lambda _run_id: {"run_id": "run"},
    )
    monkeypatch.setattr(
        "backend.service.factor_architecture.factor_architecture_repo.list_sets",
        lambda _run_id: [core, trend],
    )
    monkeypatch.setattr(
        "backend.service.factor_architecture.factor_architecture_repo.list_candidates",
        lambda _run_id: candidates,
    )
    monkeypatch.setattr(service, "_has_persisted_validation_evidence", lambda *_args: True)
    monkeypatch.setattr(service, "_validate_model_ab_factor_keys", lambda *_args: None)
    monkeypatch.setattr("core.domain.factors.registry.get_pool", lambda: Pool())

    result = service.publish_satellites("run")

    assert len(result["published"]) == 3
    assert all(item["qualification"] == "research_only" for item in result["published"])
    assert all(item["production_eligible"] is False for item in result["published"])
    for args, kwargs in saved:
        assert kwargs["qualification"] == "research_only"
        assert kwargs["factor_specs"]
        assert all(spec["data_safety"] == "safe" for spec in kwargs["factor_specs"].values())
        assert kwargs["validation_evidence"]["source_sets"]
