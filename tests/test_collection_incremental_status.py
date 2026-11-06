import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.unit
@pytest.mark.p1
def test_incremental_collection_reports_partial_when_qlib_sync_fails(monkeypatch):
    from backend.service import collection
    from core.adapters.qlib import bridge

    fast_collect = ModuleType("scripts.fast_collect")
    fast_collect.get_all_codes = lambda pool: ["000001.SZ", "000002.SZ", "000003.SZ"]
    fast_collect.collect_incremental = lambda *args, **kwargs: {
        "expected": 3,
        "success": 1,
        "failed": 1,
        "skipped": 1,
        "successful_codes": ["000001.SZ"],
        "failed_codes": [{"code": "000002.SZ", "stage": "fetch", "error": "timeout"}],
        "actual_date_range": {"start": "2026-10-29", "end": "2026-10-29"},
    }
    monkeypatch.setitem(sys.modules, "scripts.fast_collect", fast_collect)
    monkeypatch.setattr(collection.CollectionService, "_exclude_delisted", staticmethod(lambda codes: codes))

    jobs = {}
    monkeypatch.setattr(collection, "_set_job", lambda job_id, data: jobs.update({job_id: data}))

    class FailingBridge:
        def __init__(self, data_dir):
            pass

        def sync_incremental(self):
            raise RuntimeError("publish failed")

    monkeypatch.setattr(bridge, "QlibDataBridge", FailingBridge)

    response = collection.CollectionService.collect_incremental("all", 3, 2)
    job = jobs[response["data"]["job_id"]]

    assert job["status"] == "partial"
    assert job["collection_status"] == "partial"
    assert job["sync_status"] == "error"
    assert job["collected"] == 1
    assert job["collection_result"]["expected"] == 3
    assert job["collection_result"]["failed_codes"][0]["code"] == "000002.SZ"
    assert "publish failed" in job["sync_error"]
    assert "Qlib同步失败" in job["message"]


@pytest.mark.unit
@pytest.mark.p1
def test_tencent_collection_result_tracks_success_failures_and_actual_dates(monkeypatch):
    import pandas as pd

    from scripts import fast_collect

    class FakeFuture:
        def __init__(self, function, args, kwargs):
            self.function = function
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.function(*self.args, **self.kwargs)

    class FakePool:
        def submit(self, function, *args, **kwargs):
            return FakeFuture(function, args, kwargs)

    class Borrow:
        def __enter__(self):
            return FakePool()

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(fast_collect.thread_pool, "_default", SimpleNamespace(borrow=lambda: Borrow()))
    monkeypatch.setattr(fast_collect, "as_completed", lambda futures: list(futures))
    frames = {
        "000001.SZ": ("000001.SZ", pd.DataFrame({"date": pd.to_datetime(["2026-10-28", "2026-10-29"])})),
        "000002.SZ": {"code": "000002.SZ", "stage": "fetch", "error": "timeout"},
        "000003.SZ": ("000003.SZ", pd.DataFrame({"date": pd.to_datetime(["2026-10-30"])})),
    }
    monkeypatch.setattr(fast_collect, "fetch_one_tencent", lambda code, timeout, return_error=False: frames[code])

    def save_batch(code, frame, source):
        if code == "000003.SZ":
            raise RuntimeError("write failed")

    monkeypatch.setattr(fast_collect, "save_batch", save_batch)

    result = fast_collect.run_tencent(
        list(frames), SimpleNamespace(workers=2, timeout=1), return_details=True,
    )

    assert result["expected"] == 3
    assert result["success"] == 1
    assert result["failed"] == 2
    assert result["skipped"] == 0
    assert result["successful_codes"] == ["000001.SZ"]
    assert {item["code"] for item in result["failed_codes"]} == {"000002.SZ", "000003.SZ"}
    assert {item["stage"] for item in result["failed_codes"]} == {"fetch", "write"}
    assert result["actual_date_range"] == {"start": "2026-10-28", "end": "2026-10-29"}


@pytest.mark.unit
@pytest.mark.p1
def test_retry_incremental_only_retries_unique_failed_codes(monkeypatch):
    import sys

    from backend.service import collection
    from core.adapters.qlib import bridge

    calls = {}
    fast_collect = ModuleType("scripts.fast_collect")

    def retry(codes, **kwargs):
        calls["codes"] = codes
        calls["kwargs"] = kwargs
        return {
            "expected": 2,
            "success": 1,
            "failed": 1,
            "skipped": 0,
            "successful_codes": ["000002.SZ"],
            "failed_codes": [{"code": "000003.SZ", "stage": "fetch", "error": "timeout"}],
            "actual_date_range": {"start": "2026-10-29", "end": "2026-10-30"},
        }

    fast_collect.collect_incremental = retry
    monkeypatch.setitem(sys.modules, "scripts.fast_collect", fast_collect)
    monkeypatch.setattr(collection, "get_job", lambda job_id: {
        "status": "partial",
        "collection_status": "partial",
        "collection_result": {
            "lookback_days": 4,
            "failed_codes": [
                {"code": "000002.SZ", "stage": "fetch", "error": "timeout"},
                {"code": "000002.SZ", "stage": "write", "error": "db"},
                {"code": "000003.SZ", "stage": "fetch", "error": "timeout"},
            ],
        },
    })
    jobs = {}
    monkeypatch.setattr(collection, "_set_job", lambda job_id, data: jobs.update({job_id: data}))

    class SuccessfulBridge:
        def __init__(self, data_dir):
            pass

        def sync_incremental(self):
            return {"new_dates": 1}

    monkeypatch.setattr(bridge, "QlibDataBridge", SuccessfulBridge)

    response = collection.CollectionService.retry_failed_incremental("parent-1", threads=3)
    job_id = response["data"]["job_id"]
    job = jobs[job_id]

    assert calls["codes"] == ["000002.SZ", "000003.SZ"]
    assert calls["kwargs"] == {
        "source": "tencent", "threads": 3, "lookback_days": 4, "return_details": True,
    }
    assert response["data"]["parent_job_id"] == "parent-1"
    assert job["parent_job_id"] == "parent-1"
    assert job["status"] == "partial"
    assert job["collection_status"] == "partial"
    assert job["collection_result"]["failed_codes"][0]["code"] == "000003.SZ"
    assert job["sync_status"] == "completed"


@pytest.mark.unit
@pytest.mark.p1
def test_incremental_collection_rejects_inconsistent_stage_counts(monkeypatch):
    import sys

    from backend.service import collection
    from core.adapters.qlib import bridge

    fast_collect = ModuleType("scripts.fast_collect")
    fast_collect.collect_incremental = lambda *args, **kwargs: {
        "expected": 2,
        "success": 1,
        "failed": 0,
        "skipped": 0,
        "successful_codes": ["000001.SZ"],
        "failed_codes": [],
        "actual_date_range": None,
    }
    monkeypatch.setitem(sys.modules, "scripts.fast_collect", fast_collect)
    jobs = {}
    monkeypatch.setattr(collection, "_set_job", lambda job_id, data: jobs.update({job_id: data}))

    class UnexpectedBridge:
        def __init__(self, data_dir):
            pytest.fail("计数不一致时不应启动Qlib同步")

    monkeypatch.setattr(bridge, "QlibDataBridge", UnexpectedBridge)

    collection.CollectionService._run_incremental_job("job-1", ["000001.SZ", "000002.SZ"], 5, 2)

    assert jobs["job-1"]["status"] == "error"
    assert jobs["job-1"]["collection_status"] == "error"
    assert "期望=成功+失败+跳过" in jobs["job-1"]["message"]


@pytest.mark.unit
@pytest.mark.p1
def test_retry_failed_incremental_api_accepts_object_body(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.qlib import collect_v2
    from backend.service.collection import CollectionService

    monkeypatch.setattr(CollectionService, "retry_failed_incremental", lambda job_id, threads, background_tasks: {
        "data": {"job_id": "retry-1", "parent_job_id": job_id, "threads": threads}
    })
    app = FastAPI()
    app.include_router(collect_v2.router, prefix="/api/qlib")

    response = TestClient(app).post(
        "/api/qlib/data/collect-incremental/parent-1/retry-failed",
        json={"threads": 7},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {"job_id": "retry-1", "parent_job_id": "parent-1", "threads": 7}
