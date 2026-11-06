from core.domain.factors.archive import (
    campaign_seed,
    expression_fingerprint,
    novelty_decision,
    rank_hall_of_fame,
)
from core.domain.factors.candidate import expression_windows, replace_expression_window
import pytest


def test_expression_fingerprint_deduplicates_commutative_expression():
    assert expression_fingerprint("($close + $open)") == expression_fingerprint("($open+$close)")
    assert novelty_decision(
        "($open + $close)",
        {expression_fingerprint("($close + $open)")},
    )["accepted"] is False


def test_campaign_seed_is_reproducible_and_changes_by_batch():
    assert campaign_seed(42, "campaign", 1) == campaign_seed(42, "campaign", 1)
    assert campaign_seed(42, "campaign", 1) != campaign_seed(42, "campaign", 2)


def test_hall_of_fame_keeps_best_equivalent_candidate():
    result = rank_hall_of_fame([
        {"expr": "($close + $open)", "train_metrics": {"fitness": 0.1}},
        {"expr": "($open + $close)", "train_metrics": {"fitness": 0.2}},
        {"expr": "$volume", "train_metrics": {"fitness": 0.15}},
    ], 10)
    assert len(result) == 2
    assert result[0]["train_fitness"] == 0.2


def test_embedded_expression_window_can_be_perturbed():
    expr = "Mean(Std($open, 3), 60)"
    assert expression_windows(expr) == [3, 60]
    assert replace_expression_window(expr, 3, 5) == "Mean(Std($open, 5), 60)"


def test_campaign_config_enables_deferred_final_oos():
    from backend.service.factor_mining_campaign import FactorMiningCampaignService
    config = FactorMiningCampaignService._normalize({
        "train_start": "2020-01-01", "train_end": "2023-12-31",
        "valid_start": "2024-01-01", "valid_end": "2024-12-31",
        "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
        "max_depth": 7, "max_batches": 0, "max_hours": 0,
    })
    assert config["defer_final_oos"] is True
    assert config["max_depth"] == 7
    assert config["max_batches"] == 0


def test_max_depth_outside_guardrail_is_rejected():
    from backend.service.factor_mining import FactorMiningService
    with pytest.raises(ValueError, match="max_depth"):
        FactorMiningService._normalize_config({
            "train_start": "2020-01-01", "train_end": "2023-12-31",
            "valid_start": "2024-01-01", "valid_end": "2024-12-31",
            "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
            "max_depth": 9,
        })


def test_research_training_accepts_unpromoted_candidate(monkeypatch):
    import backend.service.factor_mining_training as training_module
    monkeypatch.setattr(training_module.factor_mining_repo, "get_run", lambda run_id: {
        "run_id": run_id,
        "config": {
            "pool": "csi300", "baseline_set": "trend_momentum",
            "train_start": "2020-01-01", "train_end": "2023-12-31",
            "valid_start": "2024-01-01", "valid_end": "2024-12-31",
            "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
            "forward_days": 20,
        },
    })
    monkeypatch.setattr(training_module.factor_mining_repo, "list_candidates", lambda run_id: [{
        "candidate_key": "gp_test", "expr": "Mean($close, 20)", "status": "train_passed",
    }])
    monkeypatch.setattr(training_module, "resolve_factor_expression", lambda key: ("Mean($close, 20)", None))
    normalized = training_module.FactorMiningTrainingService._normalize(
        "run1", {"candidate_keys": ["gp_test"]})
    assert normalized["candidate_keys"] == ["gp_test"]
    assert normalized["include_baseline"] is True


def test_research_training_compares_same_config_control_and_enhanced(monkeypatch):
    import backend.service.factor_mining_training as training_module
    import core.adapters.qlib.models as qlib_models
    calls = []

    class FakeManager:
        def __init__(self):
            pass

        def train_dual_model(self, **kwargs):
            calls.append(kwargs)
            return {"ic_mean": 0.02 if len(calls) == 1 else 0.03, "icir": 0.4}

    monkeypatch.setattr(qlib_models, "QlibModelManager", FakeManager)
    monkeypatch.setattr(training_module.factor_mining_repo, "update_training_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_module.factor_mining_service, "_resolve_baseline", lambda value: (
        value, "trend_momentum", ["mom_20d"],
    ))
    monkeypatch.setattr(training_module, "resolve_factor_expression", lambda key: ("$close", None))
    request = {
        "run_id": "run1", "candidate_keys": ["gp_test"], "pool": "csi300",
        "baseline_set": "trend_momentum", "include_baseline": True,
        "train_baseline_control": True,
        "train_start": "2020-01-01", "train_end": "2023-12-31",
        "valid_start": "2024-01-01", "valid_end": "2024-12-31",
        "test_start": "2025-01-01", "test_end": "2025-12-31",
        "label_horizon": 20, "param_profile": "conservative",
        "enable_cross_sectional": False, "enable_winsorize": True,
    }
    result = training_module.FactorMiningTrainingService()._run("training1", request)

    assert calls[0]["custom_factor_keys"] == ["mom_20d"]
    assert calls[1]["custom_factor_keys"] == ["mom_20d", "gp_test"]
    assert calls[0]["train_period"] == calls[1]["train_period"]
    assert result["baseline_ic_mean"] == 0.02
    assert result["ic_delta"] == pytest.approx(0.01)
    assert result["activated"] is False
