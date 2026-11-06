import pandas as pd
import datetime
import pytest


def _run_collection(monkeypatch, rows):
    from backend.service import fundamental_collection as collection
    requested_fields = []

    class Provider:
        def fina_indicator(self, **kwargs):
            requested_fields.append(kwargs["fields"])
            return rows(kwargs) if callable(rows) else pd.DataFrame(rows)

    saved = []
    monkeypatch.setattr(collection.Database, "is_available", lambda: True)
    monkeypatch.setattr(collection.Database, "fetchall", lambda query: [] if "stock_identity" in query else [])
    monkeypatch.setattr(collection, "ensure_fundamental_versions_table", lambda: None)
    monkeypatch.setattr(collection, "save_fundamental_batch", lambda *args: saved.append(args) or {"current_rows": 1, "version_rows": len(args[2])})
    monkeypatch.setattr("core.data.sources.tushare.get_pro", lambda: Provider())
    monkeypatch.setattr("core.data.sources.tushare._rate_limit", lambda: None)
    return collection.fetch_and_save_fundamentals(codes=["600000.SH"]), saved, requested_fields


def test_fundamental_collection_rejects_rows_outside_requested_security(monkeypatch):
    result, saved, requested_fields = _run_collection(monkeypatch, [{
        "ts_code": "000001.SZ", "ann_date": "20241030", "end_date": "20240930",
        "update_flag": "1", "grossprofit_margin": 30.0,
    }])

    assert result["failed_batches"] == 1
    assert result["failed_codes"] == ["600000.SH"]
    assert result["saved"] == 0
    assert saved == []
    assert "q_gsprofit_margin" in requested_fields[0]


def test_fundamental_collection_rejects_missing_announcement_date(monkeypatch):
    result, saved, _ = _run_collection(monkeypatch, [{
        "ts_code": "600000.SH", "ann_date": None, "end_date": "20240930",
        "update_flag": "1", "grossprofit_margin": 30.0,
    }])

    assert result["failed_batches"] == 1
    assert result["failed_codes"] == ["600000.SH"]
    assert result["saved"] == 0
    assert saved == []


def test_fundamental_collection_accepts_scoped_rows_with_announcement_dates(monkeypatch):
    result, saved, _ = _run_collection(monkeypatch, [{
        "ts_code": "600000.SH", "ann_date": "20241030", "end_date": "20240930",
        "update_flag": "1", "grossprofit_margin": 30.0,
    }])

    assert result["failed_batches"] == 0
    assert result["saved"] == 1
    assert len(saved) == 1
    assert saved[0][3][0][0:3] == ("600000", pd.Timestamp("2024-09-30").date(), pd.Timestamp("2024-10-30").date())


def test_fundamental_field_contract_drives_collection_and_current_projection(monkeypatch):
    from core.shared.fundamental_contract import FUNDAMENTAL_CURRENT_PROJECTION, FUNDAMENTAL_SOURCE_FIELDS

    row = {
        "ts_code": "600000.SH",
        "ann_date": "20241030",
        "end_date": "20240930",
        "update_flag": "1",
        **{field: index + 1 for index, field in enumerate(FUNDAMENTAL_SOURCE_FIELDS)},
    }
    result, saved, requested_fields = _run_collection(monkeypatch, [row])

    assert result["failed_batches"] == 0
    assert set(FUNDAMENTAL_SOURCE_FIELDS).issubset(set(requested_fields[0].split(",")))
    assert saved[0][2][0]["q_gsprofit_margin"] == FUNDAMENTAL_SOURCE_FIELDS.index("q_gsprofit_margin") + 1
    assert saved[0][3][0][3:] == tuple(
        FUNDAMENTAL_SOURCE_FIELDS.index(source_field) + 1
        for source_field, _ in FUNDAMENTAL_CURRENT_PROJECTION
    )


def test_fundamental_repository_builds_projection_from_shared_field_contract(monkeypatch):
    from core.data.repos import fundamental_versions
    from core.shared.fundamental_contract import FUNDAMENTAL_CURRENT_PROJECTION

    class Cursor:
        def __init__(self):
            self.statements = []
            self.rows = []
            self.official_rows = []

        def execute(self, statement, params=None):
            self.statements.append(statement)

        def fetchall(self):
            return self.official_rows

        def executemany(self, statement, rows):
            self.statements.append(statement)
            self.rows.extend(rows)

    cursor = Cursor()
    cursor.official_rows = [{
        "code": "600000",
        "report_date": "2024-09-30",
        "version_kind": "official_field_reconciliation",
        "payload_json": '{"official_values":{"grossprofit_margin":31.5}}',
    }]
    monkeypatch.setattr(fundamental_versions.Database, "transaction", lambda operation: operation(cursor))
    current_row = ("600000", "2024-09-30", "2024-10-30", *range(len(FUNDAMENTAL_CURRENT_PROJECTION)))

    result = fundamental_versions.save_fundamental_batch("capture", "2024-11-01", [], [current_row])

    insert_sql = next(statement for statement in cursor.statements if statement.startswith("INSERT INTO fundamental_data"))
    columns = insert_sql.split("(", 1)[1].split(")", 1)[0].split(",")
    assert columns == ["code", "report_date", "ann_date", *[column for _, column in FUNDAMENTAL_CURRENT_PROJECTION]]
    assert len(cursor.rows) == 1
    assert len(cursor.rows[0]) == len(columns)
    assert cursor.rows[0][columns.index("grossprofit_margin")] == 31.5
    assert result == {"version_rows": 0, "current_rows": 1}


def test_fundamental_cache_rebuild_submits_persisted_contract_scoped_task(monkeypatch):
    from backend.service.collection import CollectionService
    from core.shared.infrastructure.async_task import task_manager

    submitted = {}
    monkeypatch.setattr(task_manager, "submit_with_id", lambda *args, **kwargs: submitted.update(args=args, kwargs=kwargs) or args[0])
    monkeypatch.setattr("core.shared.fundamental_contract.fundamental_field_contract_sha256", lambda: "a" * 64)
    monkeypatch.setattr("core.data.fundamental.preload_cache", lambda force=False: force)
    monkeypatch.setattr("core.data.fundamental.get_cache_identity", lambda: {
        "generation": "generation-v6",
        "field_contract_sha256": "a" * 64,
    })

    response = CollectionService.rebuild_fundamental_cache("admin")
    repeated_response = CollectionService.rebuild_fundamental_cache("admin")

    job_id, task, description = submitted["args"][:3]
    assert response["data"]["job_id"] == job_id
    assert job_id.startswith("fundcache-")
    assert len(job_id) == len("fundcache-") + 12
    assert repeated_response["data"]["status"] == "running"
    assert repeated_response["data"]["job_id"] == job_id
    assert description == "重建FundVal缓存 v6 by admin"
    progress = []
    result = task(progress.append)
    assert progress[0]["stage"] == "rebuilding"
    assert result == {
        "generation": "generation-v6",
        "field_contract_sha256": "a" * 64,
        "actor": "admin",
    }


def test_fundamental_cache_rebuild_failure_is_task_failure(monkeypatch):
    from backend.service.collection import CollectionService
    from core.shared.infrastructure.async_task import task_manager

    submitted = {}
    monkeypatch.setattr(task_manager, "submit_with_id", lambda *args, **kwargs: submitted.update(args=args, kwargs=kwargs) or args[0])
    monkeypatch.setattr("core.shared.fundamental_contract.fundamental_field_contract_sha256", lambda: "b" * 64)
    monkeypatch.setattr("core.data.fundamental.preload_cache", lambda force=False: False)

    CollectionService.rebuild_fundamental_cache("admin")

    with pytest.raises(RuntimeError, match="重建失败"):
        submitted["args"][1](lambda progress: None)


def test_fundamental_cache_rebuild_route_requires_admin_actor(monkeypatch):
    from backend.api.qlib import collect_v2
    from backend.service.collection import CollectionService

    calls = []
    monkeypatch.setattr("backend.api.auth.require_admin_actor", lambda authorization: calls.append(authorization) or "operator")
    monkeypatch.setattr(CollectionService, "rebuild_fundamental_cache", lambda actor: {"actor": actor})

    result = collect_v2.rebuild_fundamental_cache("Bearer token")

    assert calls == ["Bearer token"]
    assert result == {"actor": "operator"}


def test_fundamental_bridge_keeps_single_quarter_gross_margin_separate(monkeypatch):
    from core.data import Database
    from core.data import fundamental

    monkeypatch.setattr(Database, "is_available", lambda: True)
    monkeypatch.setattr(Database, "fetchall", lambda query: [
        {
            "code": "600000",
            "ann_date": datetime.date(2024, 7, 30),
            "report_date": datetime.date(2024, 6, 30),
            "source_observed_at": datetime.datetime(2024, 7, 31, 12),
            "source_name": "tushare_fina_indicator",
            "version_kind": "provider_observation",
            "payload_json": '{"grossprofit_margin": 41.0, "q_gsprofit_margin": 23.0}',
        },
        {
            "code": "600000",
            "ann_date": datetime.date(2024, 10, 30),
            "report_date": datetime.date(2024, 9, 30),
            "source_observed_at": datetime.datetime(2024, 10, 31, 12),
            "source_name": "tushare_fina_indicator",
            "version_kind": "provider_observation",
            "payload_json": '{"grossprofit_margin": 39.0}',
        },
    ])

    frame = fundamental._load_fundamental_data()

    assert fundamental.FUND_FIELDS[10][1] == "q_gsprofit_margin"
    assert fundamental.FIELD_ID_TO_NAME[10] == "real_fund_q_gsprofit_margin"
    assert frame.loc[0, "grossprofit_margin"] == 41.0
    assert frame.loc[0, "q_gsprofit_margin"] == 23.0
    assert frame.attrs["field_coverage"]["q_gsprofit_margin"] == {
        "version_rows": 2, "available": 1, "missing": 1, "coverage": 0.5,
    }


def test_fundamental_collection_splits_capped_history_into_quarter_ranges(monkeypatch):
    row = {
        "ts_code": "600000.SH", "ann_date": "20240630", "end_date": "20240630",
        "update_flag": "1", "grossprofit_margin": 30.0, "q_gsprofit_margin": 23.0,
    }
    requests = []

    def response(kwargs):
        requests.append((kwargs["start_date"], kwargs["end_date"]))
        if kwargs["start_date"] == "20180101" and kwargs["end_date"] > "20181231":
            return pd.DataFrame([row] * 100)
        if kwargs["start_date"] == "20240101" and kwargs["end_date"] == "20241231":
            return pd.DataFrame([row] * 100)
        if kwargs["start_date"] == "20240401":
            return pd.DataFrame([row])
        return pd.DataFrame()

    result, saved, _ = _run_collection(monkeypatch, response)

    assert result["failed_batches"] == 0
    assert result["saved"] == 1
    assert ("20240401", "20240630") in requests
    assert len(saved[0][2]) == 1


def test_fundamental_collection_rejects_still_capped_quarter_without_partial_write(monkeypatch):
    row = {
        "ts_code": "600000.SH", "ann_date": "20240630", "end_date": "20240630",
        "update_flag": "1", "grossprofit_margin": 30.0, "q_gsprofit_margin": 23.0,
    }

    def response(kwargs):
        if kwargs["start_date"] == "20180101" and kwargs["end_date"] > "20181231":
            return pd.DataFrame([row] * 100)
        if kwargs["start_date"] == "20240101" and kwargs["end_date"] == "20241231":
            return pd.DataFrame([row] * 100)
        if kwargs["start_date"] == "20240401":
            return pd.DataFrame([row] * 100)
        return pd.DataFrame()

    result, saved, _ = _run_collection(monkeypatch, response)

    assert result["failed_batches"] == 1
    assert result["saved"] == 0
    assert saved == []
