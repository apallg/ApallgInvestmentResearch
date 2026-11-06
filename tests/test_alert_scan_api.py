import asyncio
from types import SimpleNamespace

import pandas as pd

from backend.api.alerts import resolve_notification_outbox, risk_scan_runs, trigger_scan
from fastapi import HTTPException
from core.domain.portfolio.manager import PortfolioScanner


def test_batch_scan_api_separates_triggered_unknown_failed_and_disabled(monkeypatch):
    results = [
        {"stock_code": "000001", "risk_score": 8, "risk_status": "available", "alert_triggered": True, "scan_id": "scan-1"},
        {"stock_code": "000002", "risk_score": None, "risk_status": "unknown", "alert_triggered": False, "scan_id": "scan-1"},
        {"stock_code": "000003", "risk_score": None, "risk_status": "failed", "alert_triggered": False, "scan_id": "scan-1"},
        {"stock_code": "000004", "risk_score": None, "risk_status": "disabled", "alert_triggered": False, "scan_id": "scan-1"},
    ]

    class Scanner:
        def scan_all(self, threshold, include):
            return results

    monkeypatch.setattr("core.domain.portfolio.manager.PortfolioScanner", Scanner)

    response = asyncio.run(trigger_scan(threshold=5, include=None))

    assert response["data"] == {
        "scan_id": "scan-1",
        "total_count": 4,
        "failed_count": 1,
        "skipped_count": 1,
        "count": 1,
        "unscored_count": 1,
        "results": results,
    }


def test_scan_runs_endpoint_returns_persisted_assessment_records(monkeypatch):
    rows = [{"scan_id": "scan-1", "expected_count": 2, "unknown_count": 1}]
    monkeypatch.setattr("core.data.RiskAlertRepo.get_recent_scans", lambda limit: rows)

    response = asyncio.run(risk_scan_runs(limit=10))

    assert response == {"data": rows}


def test_portfolio_scanner_rejects_fresh_quote_without_shared_data_contract():
    scanner = PortfolioScanner.__new__(PortfolioScanner)
    scanner.fetcher = SimpleNamespace(get_stock_daily=lambda _code: pd.DataFrame({"close": [10.0]}))
    scanner.analyzer = SimpleNamespace(
        add_indicators=lambda frame: frame,
        detect_cross_signals=lambda _frame: [],
    )
    scanner.realtime = SimpleNamespace(get_realtime_price=lambda _code: {
        "price": 10.0,
        "observed_at": "2026-10-30T10:00:00+08:00",
        "received_at": "2026-10-30T10:00:01+08:00",
        "session_date": "2026-10-30",
        "price_basis": "raw",
    })

    try:
        scanner.scan_holding(
            {"code": "000001", "name": "测试", "shares": 100, "cost_price": 10.0},
            include={"technical": False, "financial": False, "patterns": False, "realtime": True},
        )
    except ValueError as error:
        assert str(error) == "实时行情不可用或数据契约未验证"
    else:
        raise AssertionError("source-less fresh quote must be rejected")


def test_resolve_outbox_requires_admin_before_repository_call(monkeypatch):
    async def call():
        return await resolve_notification_outbox(
            5,
            {"resolution": "confirmed_not_sent_requeue", "reason": "核对渠道后台确认没有接收", "evidence_ref": "case-5"},
            authorization=None,
        )

    monkeypatch.setattr("core.data.RiskAlertRepo.resolve_unknown_outbox", lambda *args: (_ for _ in ()).throw(AssertionError()))
    try:
        asyncio.run(call())
    except HTTPException as error:
        assert error.status_code == 403
    else:
        raise AssertionError("admin authorization must be required")


def test_resolve_outbox_records_admin_resolution(monkeypatch):
    captured = {}
    monkeypatch.setattr("backend.api.auth.require_admin_actor", lambda authorization: "admin-actor")

    def resolve(*args):
        captured["args"] = args
        return {"id": 5, "status": "pending", "resolution_status": "operator_confirmed_not_sent_requeued"}

    monkeypatch.setattr("core.data.RiskAlertRepo.resolve_unknown_outbox", resolve)

    response = asyncio.run(resolve_notification_outbox(
        5,
        {"resolution": "confirmed_not_sent_requeue", "reason": "核对渠道后台确认没有接收", "evidence_ref": "case-5"},
        authorization="Bearer admin",
    ))

    assert captured["args"] == (5, "confirmed_not_sent_requeue", "admin-actor", "核对渠道后台确认没有接收", "case-5")
    assert response["data"]["status"] == "pending"
