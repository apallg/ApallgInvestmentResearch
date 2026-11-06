import asyncio
import datetime
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = [pytest.mark.unit, pytest.mark.p0]


@pytest.fixture
def changed_generation(tmp_path, monkeypatch):
    from core.adapters.qlib import generation
    from core.shared import config

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2026-11-04\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sh600000\n", encoding="utf-8")
    stage = generation.stage_qlib_generation(data_root=root)
    published = generation.publish_qlib_generation(stage, expected_generation_id=None)
    monkeypatch.setattr(generation, "_configured_qlib_root_at_import", root.resolve())
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", root)
    monkeypatch.setattr(config, "QLIB_DATA_DIR", root)
    monkeypatch.setattr(config, "QLIB_DATA_GENERATION_ID", None)
    monkeypatch.setattr(config, "QLIB_PROVIDER_URI", str(root))
    return root, published["generation_id"]


def test_history_quality_returns_restart_response_without_reading_database(changed_generation):
    from backend.api import data_manager
    from backend.middleware import register_middleware

    app = FastAPI()
    app.include_router(data_manager.router, prefix="/api/data")
    register_middleware(app)
    with patch("core.data.repos.historical_quality.HistoricalQuoteQualityRepo.latest_trade_date") as read:
        response = TestClient(app).get("/api/data/history-quality")

    assert response.status_code == 503
    assert response.json()["code"] == "qlib_restart_required"
    assert response.json()["restart_required"] is True
    assert response.json()["current_generation_id"] == changed_generation[1]
    assert "请重启" in response.json()["detail"]
    read.assert_not_called()


def test_overview_exposes_restart_requirement_and_disables_repair(changed_generation):
    from backend.api import data_manager

    with patch("core.data.collection_gate.RULES", []), \
         patch("core.data.collection_gate.GateCheckService.check_all", return_value={}), \
         patch.object(data_manager.data_repair_service, "latest_completed_trading_day", return_value=datetime.date(2026, 11, 4)):
        result = asyncio.run(data_manager.data_overview())["data"]

    assert result["qlib"]["restart_required"] is True
    assert result["qlib"]["fresh"] is False
    assert result["qlib"]["current_generation_id"] == changed_generation[1]
    table = next(cat for cat in result["categories"] if cat["key"] == "qlib")["tables"][0]
    assert table["status"] == "error"
    assert table["repairable"] is False
    assert "请重启" in table["issues"][0]


def test_repair_preview_remains_available_when_restart_is_required(changed_generation):
    from backend.service.data_repair import DataRepairService

    service = DataRepairService()
    with patch("core.data.collection_gate.RULES", []), \
         patch("core.data.collection_gate.GateCheckService.check_all", return_value={}), \
         patch.object(service, "latest_completed_trading_day", return_value=datetime.date(2026, 11, 4)):
        preview = service.preview_all()

    item = preview["items"][0]
    assert item["table"] == "qlib_binary"
    assert item["status"] == "restart_required"
    assert item["latest_date"] is None
    assert "不执行重复同步" in item["repair_scope"]
    assert "请重启" in item["reason"]


@pytest.mark.parametrize("already_changed", [False, True])
def test_repair_preserves_published_result_and_waits_for_restart(changed_generation, already_changed):
    from backend.service.data_repair import DataRepairService

    service = DataRepairService()
    latest_td = datetime.date(2026, 11, 4)
    after = service._snapshot("qlib_binary", latest_td)
    assert after["restart_required"] is True
    before = after if already_changed else {"status": "stale", "row_count": 1}
    saved = {"generation_id": changed_generation[1], "restart_required": True, "new_dates": 1}
    with patch.object(service, "latest_completed_trading_day", return_value=latest_td), \
         patch.object(service, "_snapshot", side_effect=[before, after]), \
         patch.object(service, "resolve_repair_func", return_value=lambda _date: saved), \
         patch.object(service, "_call_with_heartbeat", return_value=saved) as repair, \
         patch("backend.service.data_repair.task_manager.is_cancel_requested", return_value=False), \
         patch("backend.service.data_repair.task_manager.update_progress"):
        result = service._run("restart-test", "qlib_binary")

    assert result["restart_required"] is True
    assert result["all_pass"] is False
    assert result["_task_status"] == "partial"
    item = result["results"][0]
    assert item["status"] == "partial"
    assert item["restart_required"] is True
    assert "error" not in item
    if already_changed:
        repair.assert_not_called()
    else:
        assert item["saved"] == saved
        repair.assert_called_once()
