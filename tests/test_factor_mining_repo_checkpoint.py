from unittest.mock import patch

from core.data.repos.factor_mining import FactorMiningRepo


class Cursor:
    def __init__(self, row):
        self.row = row
        self.statements = []

    def execute(self, sql, params):
        self.statements.append((sql, params))

    def fetchone(self):
        return self.row


def test_checkpoint_resume_claim_refuses_recent_running_lease():
    repo = FactorMiningRepo()
    cursor = Cursor({
        "status": "running",
        "config_json": {"candidate_sources": ["gp"]},
        "checkpoint_json": {"version": 1, "stage": "gp_generation"},
        "idle_seconds": 15,
    })
    with patch("core.data.repos.factor_mining.Database.transaction", side_effect=lambda op: op(cursor)):
        result = repo.claim_checkpoint_resume("run-active")

    assert result is None
    assert len(cursor.statements) == 1


def test_checkpoint_resume_claim_atomically_marks_stale_run_resuming():
    repo = FactorMiningRepo()
    cursor = Cursor({
        "status": "running",
        "config_json": {"candidate_sources": ["gp"]},
        "checkpoint_json": {"version": 1, "stage": "gp_generation"},
        "idle_seconds": 120,
    })
    with patch("core.data.repos.factor_mining.Database.transaction", side_effect=lambda op: op(cursor)):
        result = repo.claim_checkpoint_resume("run-stale")

    assert result["checkpoint"]["stage"] == "gp_generation"
    assert result["config"]["candidate_sources"] == ["gp"]
    assert cursor.statements[-1][0].startswith("UPDATE factor_mining_runs SET status='resuming'")
