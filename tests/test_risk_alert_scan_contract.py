from core.domain.portfolio import risk_alert


def _holdings():
    return [
        {"code": "600519", "name": "贵州茅台", "shares": 100, "cost_price": "100.00"},
        {"code": "000001", "name": "平安银行", "shares": 100, "cost_price": "100.00"},
    ]


def _quote(price):
    return {
        "price": price,
        "source": "test",
        "observed_at": "2026-10-30T10:00:00",
        "received_at": "2026-10-30T10:00:01",
        "session_date": "2026-10-30",
        "price_basis": "raw",
    }


def _install_scan_dependencies(monkeypatch, quotes, captured):
    from core.data import HoldingsRepo, RiskAlertRepo
    from core.data.sources.realtime import RealtimeEngine

    monkeypatch.setattr(HoldingsRepo, "get_all", lambda: _holdings())
    monkeypatch.setattr(RealtimeEngine, "get_quotes", lambda self, codes: quotes)
    monkeypatch.setattr("core.shared.persistence.now_iso", lambda: "2026-10-30T10:00:00")
    monkeypatch.setattr(risk_alert, "_get_current_regime", lambda: "steady_flat")
    monkeypatch.setattr(risk_alert, "_load_regime_stops", lambda: risk_alert._DEFAULT_REGIME_STOPS)
    monkeypatch.setattr(risk_alert, "_get_kline_df", lambda code: None)
    monkeypatch.setattr(risk_alert, "_build_alert_via_llm", lambda *args: "风险告警")
    monkeypatch.setattr(RiskAlertRepo, "record_scan", lambda scan, notifications: captured.update(
        scan=scan, notifications=notifications,
    ) or [])


def test_stop_scan_marks_missing_holding_unknown_and_keeps_other_codes_scanned(monkeypatch):
    captured = {}
    _install_scan_dependencies(monkeypatch, {"600519": _quote(89.0)}, captured)

    result = risk_alert.run_stop_alert_scan("手动扫描")

    assert result["expected"] == 2
    assert result["scanned"] == 1
    assert result["status"] == "degraded"
    assert result["unknown"] == [{"code": "000001", "reason": "missing_quote"}]
    assert [item["code"] for item in result["triggered"]] == ["600519"]
    assert captured["scan"]["expected_count"] == 2
    assert captured["scan"]["scanned_count"] == 1
    assert captured["scan"]["unknown_count"] == 1
    assert any(item["idempotency_key"].endswith(":incomplete") for item in captured["notifications"])


def test_stop_scan_with_all_quotes_missing_is_degraded_not_healthy(monkeypatch):
    captured = {}
    _install_scan_dependencies(monkeypatch, {}, captured)

    result = risk_alert.run_stop_alert_scan("手动扫描")

    assert result["expected"] == 2
    assert result["scanned"] == 0
    assert result["status"] == "degraded"
    assert {item["code"] for item in result["unknown"]} == {"600519", "000001"}
    assert result["triggered"] == []
    assert captured["scan"]["status"] == "degraded"
    assert captured["scan"]["unknown_count"] == 2
    assert len(captured["notifications"]) == 1
    assert captured["notifications"][0]["idempotency_key"].endswith(":incomplete")


def test_stop_scan_rejects_fresh_quote_without_data_provenance(monkeypatch):
    captured = {}
    quote = _quote(89.0)
    del quote["source"]
    _install_scan_dependencies(monkeypatch, {"600519": quote}, captured)

    result = risk_alert.run_stop_alert_scan("手动扫描")

    assert result["scanned"] == 0
    assert result["status"] == "degraded"
    assert result["unknown"] == [{"code": "600519", "reason": "source_missing"},
                                  {"code": "000001", "reason": "missing_quote"}]
    assert captured["scan"]["assessment"]["quote_data"]["data_envelope"]["coverage"] == {
        "requested": 2, "accepted": 0, "missing": 2,
    }


def test_stop_scan_keeps_valid_quote_and_marks_invalid_contract_quote_unknown(monkeypatch):
    captured = {}
    invalid_quote = _quote(89.0)
    invalid_quote["price_basis"] = "unknown"
    _install_scan_dependencies(monkeypatch, {
        "600519": _quote(89.0), "000001": invalid_quote,
    }, captured)

    result = risk_alert.run_stop_alert_scan("手动扫描")

    assert result["scanned"] == 1
    assert result["status"] == "degraded"
    assert result["unknown"] == [{"code": "000001", "reason": "price_basis_unknown"}]
    assert [item["code"] for item in result["triggered"]] == ["600519"]
