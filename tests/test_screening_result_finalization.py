import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.api import screening
from backend.service.rebalance import RebalanceService


def test_result_endpoint_does_not_expose_screener_done_before_api_finalization(monkeypatch):
    screener = SimpleNamespace(_jobs={
        "job-123": {"job_id": "job-123", "status": "done", "candidates": [{"code": "000001"}]},
    })
    monkeypatch.setattr("core.data.ScreeningResultRepo.get", lambda job_id: {
        "job_id": job_id, "status": "running", "worker_id": screening._worker_id,
        "progress": {"stage": "running"},
    })
    monkeypatch.setattr("core.data.ScreeningResultRepo.interrupt_if_stale", lambda job_id: {
        "job_id": job_id, "status": "running", "worker_id": screening._worker_id,
        "progress": {"stage": "running"},
    })
    monkeypatch.setattr("core.domain.screening._screener.get_screener", lambda: screener)

    response = asyncio.run(screening.get_result("job-123"))

    assert response["data"]["status"] == "finalizing"
    assert "candidates" not in response["data"]


def test_save_endpoint_rejects_result_before_finalization(monkeypatch):
    monkeypatch.setattr("core.data.ScreeningResultRepo.get", lambda job_id: {
        "job_id": job_id, "status": "pending", "result_json": None,
    })
    save = Mock()
    monkeypatch.setattr(RebalanceService, "save_screening", save)

    with pytest.raises(HTTPException) as error:
        asyncio.run(screening.save_screening_result("job-123", "snapshot", ""))

    assert error.value.status_code == 409
    save.assert_not_called()


def test_completed_result_is_recovered_from_persistent_store_after_restart(monkeypatch):
    result = {
        "job_id": "job-123", "result_id": "job-123", "status": "done",
        "candidates": [{"code": "000001"}], "provenance": {"snapshot_fingerprint": "abc"},
    }
    monkeypatch.setattr(screening, "_results", {})
    monkeypatch.setattr("core.data.ScreeningResultRepo.get", lambda job_id: {
        "job_id": job_id, "status": "done", "result_json": result,
    })

    response = asyncio.run(screening.get_result("job-123"))

    assert response["data"] == result


def test_save_endpoint_uses_completed_persistent_result_after_restart(monkeypatch):
    result = {
        "job_id": "job-123", "result_id": "job-123", "status": "done",
        "candidates": [{"code": "000001"}], "provenance": {"snapshot_fingerprint": "abc"},
    }
    monkeypatch.setattr("core.data.ScreeningResultRepo.get", lambda job_id: {
        "job_id": job_id, "status": "done", "result_json": result,
    })
    save = Mock(return_value={"save_id": 7})
    monkeypatch.setattr(RebalanceService, "save_screening", save)

    response = asyncio.run(screening.save_screening_result("job-123", "snapshot", "note"))

    assert response["data"] == {"save_id": 7}
    assert save.call_args.kwargs["screening_result"] == result


def test_stale_task_is_returned_as_interrupted_after_restart(monkeypatch):
    monkeypatch.setattr("core.data.ScreeningResultRepo.get", lambda job_id: {
        "job_id": job_id, "status": "running", "worker_id": "previous-process",
    })
    monkeypatch.setattr("core.data.ScreeningResultRepo.interrupt_if_stale", lambda job_id: {
        "job_id": job_id, "status": "interrupted", "error": "worker lost",
        "progress": {"stage": "interrupted"},
    })

    response = asyncio.run(screening.get_result("job-123"))

    assert response["data"]["status"] == "interrupted"
    assert "candidates" not in response["data"]


def test_rebalance_service_rejects_unfinalized_or_unbound_results(monkeypatch):
    from core.data import ScreeningSaveRepo

    save = Mock()
    monkeypatch.setattr(ScreeningSaveRepo, "save", save)

    result = RebalanceService.save_screening(
        "job-123", "snapshot", screening_result={"status": "done", "job_id": "job-123"},
    )

    assert "error" in result
    save.assert_not_called()
