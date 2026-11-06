from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import HTTPException

from backend.service.strategy_profile import PIT_CONTRACT_VERSION
from backend.service.prediction import PredictionService


pytestmark = [pytest.mark.unit, pytest.mark.p0]


def profile_row():
    return {
        "profile_id": "baseline", "version": 1, "status": "challenger", "fingerprint": "p" * 64,
        "config_json": {
            "model": {
                "refs": [{"experiment_id": "7", "run_id": "run-7", "artifact_fingerprint": "a" * 64}],
                "oos_schedule": [{
                    "experiment_id": "7", "run_id": "fold-7", "artifact_fingerprint": "b" * 64,
                    "window_start": "2023-07-01", "window_end": "2024-12-31",
                }],
            },
            "selection": {"candidate_pool": 50},
            "universe": {"code_prefixes": ["600"], "min_history_days": 500},
            "windows": {"train_start": "2018-01-01", "train_end": "2022-12-31", "oos_start": "2023-07-01", "oos_end": "2024-12-31"},
            "factors": {"keys": [f"f{i}" for i in range(30)]},
            "score": {"neutralize": False},
            "fingerprints": {"model_refs": "m" * 64, "factor": "f" * 64},
        },
    }


def request(**overrides):
    values = {
        "profile_id": "baseline", "profile_version": 1,
        "experiment_id": "", "run_id": "", "instruments": "csi300",
        "start_time": "2024-01-05", "end_time": "2024-01-05", "topk": 30,
        "skip_features": True, "code_prefixes": None,
        "model_fields_set": {"profile_id", "profile_version", "start_time", "end_time"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prediction_frame_exposes_distinct_score_semantics():
    frame = pd.DataFrame([
        {"datetime": "2024-01-05", "instrument": "sh600001", "score": 0.2},
        {"datetime": "2024-01-05", "instrument": "sh600002", "score": -0.1},
    ])
    records, by_date, audits = PredictionService._format_predictions(
        frame, None, [], 2, neutralize_scores=False,
    )
    assert [row["instrument"] for row in by_date["2024-01-05"]] == ["sh600001", "sh600002"]
    assert records[0]["raw_score"] == 0.2
    assert records[0]["neutralized_score"] == 0.2
    assert records[0]["percentile_score"] != records[0]["raw_score"]
    assert audits["2024-01-05"]["score_semantics"] == "cross_sectional_rank_not_expected_return"


def test_profile_prediction_injects_locked_run_and_snapshot(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "params.pkl").write_bytes(b"model")
    captured = {}

    def submit(fn, description, pool):
        captured["fn"] = fn
        captured["description"] = description
        return "job-1"

    with patch("core.data.StrategyProfileRepo.get", return_value=profile_row()), \
         patch("core.data.StrategyProfileRepo.get_security_references", return_value=[]), \
         patch("backend.service.strategy_profile.StrategyProfileService.build_data_snapshot", return_value={
             "snapshot_id": "d" * 64, "pit_status": "pass", "freshness_status": "fresh",
             "pit_contract_version": PIT_CONTRACT_VERSION,
             "security_reference_status": "pass", "security_reference_coverage": 1.0,
             "security_reference_fingerprint": "b" * 64,
             "security_reference_input_fingerprint": "a" * 64,
         }), \
         patch("backend.service.strategy_profile.StrategyProfileService._snapshot_content_matches", return_value=True), \
         patch("backend.service.prediction.assert_run_path", return_value=tmp_path), \
         patch("core.adapters.qlib.model_selection.get_model_for_factor_set", return_value={
             "label_horizon": 10, "artifact_fingerprint": "b" * 64,
         }), \
         patch("backend.service.prediction.task_manager.submit", side_effect=submit):
        result = PredictionService.submit_predict(request())

    assert result["data"]["job_id"] == "job-1"
    assert captured["description"] == "预测 7/fold-7"


def test_profile_prediction_rejects_legacy_pit_snapshot():
    with patch("core.data.StrategyProfileRepo.get", return_value=profile_row()), \
         patch("backend.service.strategy_profile.StrategyProfileService.build_data_snapshot", return_value={
             "snapshot_id": "d" * 64, "pit_status": "pass", "freshness_status": "fresh",
             "security_reference_status": "pass",
         }):
        with pytest.raises(HTTPException) as exc:
            PredictionService.submit_predict(request())

    assert exc.value.status_code == 409
    assert "PIT" in exc.value.detail or "时点" in exc.value.detail


def test_research_profile_rejects_date_before_model_availability():
    row = profile_row()
    row["status"] = "draft"
    row["config_json"]["factors"]["research_only"] = True
    row["config_json"]["availability"] = {"first_decision_date": "2026-09-03"}
    with patch("core.data.StrategyProfileRepo.get", return_value=row):
        with pytest.raises(HTTPException) as exc:
            PredictionService.submit_predict(request(start_time="2026-09-02", end_time="2026-09-02"))
    assert exc.value.status_code == 422
    assert "2026-09-03" in exc.value.detail


def test_profile_prediction_rejects_locked_parameter_override(tmp_path):
    req = request(experiment_id="other", model_fields_set={
        "profile_id", "profile_version", "start_time", "end_time", "experiment_id",
    })
    with patch("core.data.StrategyProfileRepo.get", return_value=profile_row()):
        with pytest.raises(HTTPException) as exc:
            PredictionService.submit_predict(req)
    assert exc.value.status_code == 422
    assert "experiment_id" in exc.value.detail


def test_batch_execution_uses_single_signal_validation():
    from backend.service.rebalance import RebalanceService
    with patch.object(RebalanceService, "mark_executed", side_effect=[
        {"signal_id": 1, "execution_status": "executed"}, {"error": "价格无效"},
    ]) as execute:
        result = RebalanceService.batch_execute([
            {"signal_id": 1, "shares": 100, "price": 10},
            {"signal_id": 2, "shares": 100, "price": -1},
        ])
    assert execute.call_count == 2
    assert result == {"executed": 1, "failures": [{"signal_id": 2, "error": "价格无效"}]}
