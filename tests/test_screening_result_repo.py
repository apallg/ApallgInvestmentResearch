import json

import pytest

from core.data.connection import Database
from core.data.repos.screening_result import ScreeningResultRepo


class Cursor:
    def __init__(self, rowcount=1):
        self.rowcount = rowcount
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))


def test_mark_running_is_compare_and_set_on_pending(monkeypatch):
    monkeypatch.setattr(ScreeningResultRepo, "_ready", True)
    cursor = Cursor()
    monkeypatch.setattr(Database, "transaction", lambda operation: operation(cursor))

    ScreeningResultRepo.mark_running("job-1", "worker-1")

    sql, params = cursor.calls[0]
    assert "status='pending'" in sql
    assert params[0] == "worker-1"


def test_mark_running_rejects_changed_task_state(monkeypatch):
    monkeypatch.setattr(ScreeningResultRepo, "_ready", True)
    cursor = Cursor(rowcount=0)
    monkeypatch.setattr(Database, "transaction", lambda operation: operation(cursor))

    with pytest.raises(RuntimeError, match="任务状态已变化"):
        ScreeningResultRepo.mark_running("job-1", "worker-1")


def test_complete_persists_strict_final_result_only_for_own_running_task(monkeypatch):
    monkeypatch.setattr(ScreeningResultRepo, "_ready", True)
    cursor = Cursor()
    monkeypatch.setattr(Database, "transaction", lambda operation: operation(cursor))
    result = {
        "job_id": "job-1", "result_id": "job-1", "status": "done",
        "candidates": [{"code": "000001"}], "provenance": {"snapshot_fingerprint": "abc"},
    }

    ScreeningResultRepo.complete("job-1", "worker-1", result)

    sql, params = cursor.calls[0]
    assert "status='running' AND worker_id=%s" in sql
    assert json.loads(params[0]) == result
    assert params[-2:] == ("job-1", "worker-1")


def test_complete_rejects_non_json_result_without_writing(monkeypatch):
    monkeypatch.setattr(ScreeningResultRepo, "_ready", True)
    monkeypatch.setattr(Database, "transaction", lambda operation: pytest.fail("write must not occur"))

    with pytest.raises(ValueError):
        ScreeningResultRepo.complete("job-1", "worker-1", {"score": float("nan")})
