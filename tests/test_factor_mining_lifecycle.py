from unittest.mock import patch

import pandas as pd

from backend.service.factor_mining import FactorMiningService


def test_factor_mining_get_run_reports_interrupted_without_resubmitting():
    service = FactorMiningService()
    persisted = {
        "run_id": "run-interrupted",
        "status": "running",
        "config": {"population": 20},
        "progress": {"stage": "generate", "current": 3, "total": 10},
    }
    task = {
        "job_id": "run-interrupted",
        "status": "interrupted",
        "progress": {"stage": "interrupted"},
    }

    with patch("backend.service.factor_mining.task_manager.get_dict", return_value=task), \
         patch("backend.service.factor_mining.task_manager.is_active", return_value=False), \
         patch("backend.service.factor_mining.task_manager.submit_with_id") as submit, \
         patch("backend.service.factor_mining.factor_mining_repo.ensure_tables"), \
         patch("backend.service.factor_mining.factor_mining_repo.get_checkpoint", return_value=None), \
         patch("backend.service.factor_mining.factor_mining_repo.get_run", return_value=persisted):
        result = service.get_run("run-interrupted")

    assert result["status"] == "interrupted"
    assert result["resume_available"] is False
    assert result["progress"]["current"] == 3
    assert "显式重新提交" in result["progress"]["message"]
    submit.assert_not_called()


def test_factor_mining_get_run_returns_active_task_without_side_effects():
    service = FactorMiningService()
    persisted = {"run_id": "run-active", "status": "running", "progress": {}}
    task = {"job_id": "run-active", "status": "running", "progress": {"stage": "snapshot"}}

    with patch("backend.service.factor_mining.task_manager.get_dict", return_value=task), \
         patch("backend.service.factor_mining.task_manager.is_active", return_value=True), \
         patch("backend.service.factor_mining.task_manager.submit_with_id") as submit, \
         patch("backend.service.factor_mining.factor_mining_repo.ensure_tables"), \
         patch("backend.service.factor_mining.factor_mining_repo.get_checkpoint", return_value=None), \
         patch("backend.service.factor_mining.factor_mining_repo.get_run", return_value=persisted):
        result = service.get_run("run-active")

    assert result == task
    submit.assert_not_called()


def test_factor_mining_get_run_exposes_only_supported_generation_checkpoint():
    service = FactorMiningService()
    persisted = {
        "run_id": "run-checkpoint",
        "status": "running",
        "checkpoint_available": True,
        "config": {"candidate_sources": ["gp"]},
        "progress": {"stage": "generate", "current": 2, "total": 8},
    }
    checkpoint = {"version": 1, "stage": "gp_generation"}
    task = {"job_id": "run-checkpoint", "status": "interrupted", "progress": {}}

    with patch("backend.service.factor_mining.task_manager.get_dict", return_value=task), \
         patch("backend.service.factor_mining.task_manager.is_active", return_value=False), \
         patch("backend.service.factor_mining.factor_mining_repo.ensure_tables"), \
         patch("backend.service.factor_mining.factor_mining_repo.get_checkpoint", return_value=checkpoint), \
         patch("backend.service.factor_mining.factor_mining_repo.get_run", return_value=persisted):
        result = service.get_run("run-checkpoint")

    assert result["status"] == "interrupted"
    assert result["resume_available"] is True
    assert "代际边界检查点" in result["progress"]["message"]


def test_factor_mining_resume_claims_checkpoint_and_submits_original_run_id():
    service = FactorMiningService()
    claimed = {
        "config": {"pool": "csi300"},
        "checkpoint": {
            "version": 1,
            "stage": "gp_generation",
            "progress": {"stage": "generate", "current": 2, "total": 8},
        },
    }

    with patch("backend.service.factor_mining.factor_mining_repo.ensure_tables"), \
         patch("backend.service.factor_mining.task_manager.is_active", return_value=False), \
         patch("backend.service.factor_mining.factor_mining_repo.claim_checkpoint_resume", return_value=claimed), \
         patch("backend.service.factor_mining.task_manager.submit_with_id") as submit:
        result = service.resume_run("run-checkpoint")

    assert result["run_id"] == "run-checkpoint"
    assert result["status"] == "resuming"
    assert submit.call_args.args[0] == "run-checkpoint"


def test_factor_mining_input_fingerprint_changes_when_snapshot_data_changes():
    service = FactorMiningService()
    frame = pd.DataFrame(
        {"close": [10.0, 11.0], "volume": [100.0, 120.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    config = {"pool": "csi300", "seed": 4}
    first = service._mining_input_fingerprint(
        config, {"2026-01-02": {"000001.SZ": (frame, "平安银行", 11.0, 120.0)}}, {}, {}, {},
    )
    changed = frame.copy()
    changed.loc[pd.Timestamp("2026-01-02"), "close"] = 11.1
    second = service._mining_input_fingerprint(
        config, {"2026-01-02": {"000001.SZ": (changed, "平安银行", 11.1, 120.0)}}, {}, {}, {},
    )

    assert first != second
