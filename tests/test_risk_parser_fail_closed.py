from core.domain.portfolio.manager import PortfolioScanner


def test_unparseable_ai_risk_is_unknown_not_low():
    scanner = PortfolioScanner.__new__(PortfolioScanner)

    assert scanner._parse_risk("") == ("未知", None)
    assert scanner._parse_risk(None) == ("未知", None)
    assert scanner._parse_risk("本次分析未返回风险评分") == ("未知", None)
    assert scanner._parse_risk("风险等级：低风险") == ("未知", None)
    assert scanner._parse_risk("风险评分: 12/10") == ("未知", None)
    assert scanner._parse_risk("风险评分: -1/10") == ("未知", None)
    assert scanner._parse_risk("风险评分: 5.5/10") == ("未知", None)


def test_parse_risk_accepts_valid_score_and_conservative_level_only_text():
    scanner = PortfolioScanner.__new__(PortfolioScanner)

    assert scanner._parse_risk("风险评分: 3/10\n风险等级: 低风险") == ("低风险", 3)
    assert scanner._parse_risk("结论：高风险，需要立即复核") == ("高风险", 7)


def test_batch_scan_returns_unscored_results_without_persisting_or_notifying(monkeypatch):
    scanner = PortfolioScanner.__new__(PortfolioScanner)
    item = {
        "stock_code": "000001",
        "risk_level": "未知",
        "risk_score": None,
        "risk_status": "unknown",
    }
    monkeypatch.setattr("core.data.HoldingsRepo.get_all", lambda: [{"code": "000001"}])
    monkeypatch.setattr(scanner, "scan_holding", lambda holding, include=None: item)
    monkeypatch.setattr(scanner, "persist_alert", lambda result: (_ for _ in ()).throw(AssertionError("must not persist")))

    results = scanner.scan_all(threshold=0)

    assert len(results) == 1
    assert results[0]["risk_status"] == "unknown"
    assert results[0]["scan_id"]
    assert results[0]["alert_triggered"] is False


def test_batch_scan_keeps_per_holding_failure_and_disabled_state(monkeypatch):
    scanner = PortfolioScanner.__new__(PortfolioScanner)
    holdings = [
        {"code": "000001", "name": "平安银行"},
        {"code": "000002", "name": "万科A"},
        {"code": "000003", "name": "测试", "alerts_enabled": False},
    ]
    monkeypatch.setattr("core.data.HoldingsRepo.get_all", lambda: holdings)

    def scan(holding, include=None):
        if holding["code"] == "000001":
            return {"stock_code": "000001", "stock_name": "平安银行", "risk_level": "低风险", "risk_score": 2}
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(scanner, "scan_holding", scan)
    monkeypatch.setattr(scanner, "persist_alert", lambda result: (_ for _ in ()).throw(AssertionError("must not persist")))

    results = scanner.scan_all(threshold=5)

    assert [item["risk_status"] for item in results] == ["available", "failed", "disabled"]
    assert results[0]["alert_triggered"] is False
    assert results[1]["failure_code"] == "TimeoutError"
    assert len({item["scan_id"] for item in results}) == 1
