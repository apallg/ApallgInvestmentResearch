import datetime
from unittest.mock import patch

import pytest


@pytest.mark.unit
@pytest.mark.p0
def test_repair_all_defaults_to_latest_trading_day():
    from core.data import integrity

    latest_td = datetime.date(2026, 8, 24)
    with patch.object(integrity, "CHECKS", []), \
         patch("core.shared.trading_time.get_latest_trading_day", return_value=latest_td), \
         patch("core.shared.trading_time.is_trading_day", return_value=True):
        result = integrity.check_and_repair(auto_repair=False, dry_run=True)

    assert result["date"] == "2026-08-24"
    assert result["all_pass"] is True


@pytest.mark.unit
@pytest.mark.p0
def test_use_latest_rechecks_the_latest_report_period_after_repair():
    from core.data import integrity

    count_dates = []

    def count_rows(_table, _date_col, target_date, _where=""):
        count_dates.append(str(target_date)[:10])
        return 50 if len(count_dates) == 1 else 120

    checks = [{
        "table": "fundamental_data",
        "date_col": "report_date",
        "min_rows": 100,
        "desc": "基本面财务数据",
        "repair": lambda _date: 70,
        "use_latest": True,
        "where_clause": "",
    }]
    with patch.object(integrity, "CHECKS", checks), \
         patch.object(integrity, "_count", side_effect=count_rows), \
         patch("core.shared.trading_time.is_trading_day", return_value=True), \
         patch("core.data.Database.fetchone", side_effect=[
             {"latest": datetime.date(2026, 6, 30)},
             {"latest": datetime.date(2026, 6, 30)},
         ]):
        result = integrity.check_and_repair(
            trade_date="2026-08-24", auto_repair=True, dry_run=False,
        )

    assert count_dates == ["2026-06-30", "2026-06-30"]
    assert result["all_pass"] is True
    assert result["checks"][0]["count"] == 120


@pytest.mark.unit
@pytest.mark.p0
def test_ths_daily_expected_rows_follow_the_actual_monitor_pool():
    from core.data.collection_gate import GateCheckService, GateStatus, RULES

    rule = next(r for r in RULES if r.table == "ths_daily_kline")
    latest_td = datetime.date(2026, 8, 24)

    def fetchone(sql, _params=()):
        if "FROM `holdings`" in sql:
            return {"cnt": 2}
        if "MAX(`trade_date`)" in sql:
            return {"latest": latest_td}
        return {"cnt": 2}

    service = GateCheckService()
    with patch("core.data.Database.is_available", return_value=True), \
         patch("core.data.Database.fetchone", side_effect=fetchone), \
         patch.object(service, "_check_null_ratios", return_value={"close": 0.0}):
        result = service._check_single(rule, latest_td)

    assert result.status == GateStatus.FRESH
    assert result.expected_min_rows == 2


@pytest.mark.unit
@pytest.mark.p0
def test_data_repair_service_covers_all_operational_tables():
    from backend.service.data_repair import data_repair_service

    for table in (
        "daily_quotes", "raw_news", "stock_notices", "stock_sectors",
        "ths_daily_kline", "qlib_binary",
    ):
        assert callable(data_repair_service.resolve_repair_func(table))


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_data_repair_snapshot_uses_generation_read_access():
    from backend.service.data_repair import DataRepairService

    service = DataRepairService()
    latest_td = datetime.date(2026, 8, 24)
    status = {"synced": True, "last_date": "2026-08-24", "stocks": 5000}
    lock_calls = []

    def run_with_read_access(callback, *args, **kwargs):
        lock_calls.append(True)
        return callback(*args, **kwargs)

    with patch("core.adapters.qlib.bridge.QlibDataBridge") as bridge, \
         patch("backend.service.qlib_access.run_with_qlib_read_access", side_effect=run_with_read_access):
        bridge.return_value.get_status.return_value = status
        result = service._snapshot("qlib_binary", latest_td)

    assert lock_calls == [True]
    assert result == {
        "status": "fresh", "latest_date": "2026-08-24",
        "row_count": 5000, "expected_min": 4000,
    }


@pytest.mark.unit
@pytest.mark.p0
def test_data_overview_qlib_status_uses_generation_read_access():
    import asyncio

    from backend.api import data_manager

    status = {
        "synced": True, "stocks": 5000, "dates": 200,
        "first_date": "2025-01-01", "last_date": "2026-08-24",
    }
    lock_calls = []

    def run_with_read_access(callback, *args, **kwargs):
        lock_calls.append(True)
        return callback(*args, **kwargs)

    with patch("core.data.collection_gate.RULES", []), \
         patch("core.data.collection_gate.GateCheckService.check_all", return_value={}), \
         patch.object(data_manager.data_repair_service, "latest_completed_trading_day", return_value=datetime.date(2026, 8, 24)), \
         patch("core.adapters.qlib.bridge.QlibDataBridge") as bridge, \
         patch("backend.service.qlib_access.run_with_qlib_read_access", side_effect=run_with_read_access):
        bridge.return_value.get_status.return_value = status
        result = asyncio.run(data_manager.data_overview())

    assert lock_calls == [True]
    assert result["data"]["qlib"]["fresh"] is True
    assert result["data"]["qlib"]["stocks"] == 5000


@pytest.mark.unit
@pytest.mark.p0
def test_global_news_index_recovers_date_from_source_url():
    from core.adapters.nlp.collector import SentimentCollector

    collector = SentimentCollector()
    cache = {"000001": [{
        "title": "测试新闻", "url": "https://example.com/2026-08-24/article.html",
        "date": "", "source_tag": "cailian", "source_confidence": 0.9,
    }]}
    with patch.object(collector, "_fetch_cailian_all", return_value=cache):
        items = collector.collect_global_index()

    assert items[0]["date"] == "2026-08-24"


@pytest.mark.unit
@pytest.mark.p0
def test_market_news_snapshot_limits_the_fallback_stock_pool():
    from core.adapters.nlp.collector import SentimentCollector

    collector = SentimentCollector()
    liquid = [{"code": f"600{i:03d}"} for i in range(30)]
    captured = []
    with patch.object(collector, "collect_global_index", return_value=[]), \
         patch("core.data.Database.fetchall", side_effect=[[{"code": "000001"}], liquid]), \
         patch.object(collector, "collect", side_effect=lambda codes: captured.extend(codes) or []):
        collector.collect_market_snapshot(stock_limit=20)

    assert len(captured) == 20
    assert captured[0] == "000001"


@pytest.mark.unit
@pytest.mark.p0
def test_data_repair_is_submitted_as_background_task():
    from backend.service.data_repair import DataRepairService

    service = DataRepairService()
    with patch("backend.service.data_repair.task_manager.is_active", return_value=False), \
         patch("backend.service.data_repair.task_manager.submit_with_id") as submit:
        result = service.start_all()

    assert result["status"] == "pending"
    assert result["job_id"].startswith("repair")
    assert submit.call_args.kwargs["pool"] == "default"


@pytest.mark.unit
@pytest.mark.p0
def test_data_repair_reports_partial_when_source_stays_incomplete():
    from backend.service.data_repair import DataRepairService

    service = DataRepairService()
    snapshots = [
        {"status": "incomplete", "row_count": 86, "expected_min": 150},
        {"status": "incomplete", "row_count": 90, "expected_min": 150},
    ]
    with patch("core.shared.trading_time.get_latest_trading_day", return_value=datetime.date(2026, 8, 24)), \
         patch.object(service, "_snapshot", side_effect=snapshots), \
         patch.object(service, "resolve_repair_func", return_value=lambda _date: 4), \
         patch("backend.service.data_repair.task_manager.is_cancel_requested", return_value=False), \
         patch("backend.service.data_repair.task_manager.update_progress"):
        result = service._run("repair-test", "north_flow")

    assert result["all_pass"] is False
    assert result["_task_status"] == "partial"
    assert result["results"][0]["saved"] == 4
    assert result["results"][0]["status"] == "partial"
