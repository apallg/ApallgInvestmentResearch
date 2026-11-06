from core.data.repos.risk import RiskAlertRepo
from core.data.risk_notification_migration import ensure_risk_notification_outbox
from backend.service.risk_notification_outbox import dispatch_risk_notification_outbox
from core.domain.portfolio.manager import PortfolioScanner
from core.shared.infrastructure.notify import WeComNotifier


def test_assessment_and_notification_are_written_in_one_transaction(monkeypatch):
    calls = []

    class Cursor:
        lastrowid = 23

        def execute(self, sql, params=()):
            calls.append((sql, params))

    transaction_calls = []
    monkeypatch.setattr(
        "core.data.repos.risk.Database.transaction",
        lambda operation: transaction_calls.append(operation) or operation(Cursor()),
    )

    alert_id = RiskAlertRepo.add(
        "000001", "测试", "高风险", 8, "详情", "建议",
        notification_payload={"stock_code": "000001", "risk_score": 8},
        idempotency_key="scan-1:000001",
    )

    assert alert_id == 23
    assert len(transaction_calls) == 1
    assert len(calls) == 2
    assert "INSERT INTO risk_alerts" in calls[0][0]
    assert "INSERT INTO risk_notification_outbox" in calls[1][0]
    assert calls[1][1][0] == 23
    assert calls[1][1][1] is None
    assert calls[1][1][2] == "scan-1:000001"


def test_stop_scan_and_notifications_are_persisted_in_one_transaction(monkeypatch):
    calls = []

    class Cursor:
        lastrowid = 77

        def execute(self, sql, params=()):
            calls.append((sql, params))

    transactions = []
    monkeypatch.setattr(
        "core.data.repos.risk.Database.transaction",
        lambda operation: transactions.append(operation) or operation(Cursor()),
    )

    outbox_ids = RiskAlertRepo.record_scan(
        {
            "scan_id": "scan-2", "source": "stop_loss", "session_name": "午盘",
            "regime_name": "steady_flat", "status": "degraded", "expected_count": 2,
            "scanned_count": 1, "unknown_count": 1, "dynamic_unknown_count": 0,
            "triggered_count": 1, "assessment": {"unknown": [{"code": "000002"}]},
        },
        [{"idempotency_key": "scan-2:triggered", "payload": {"message_type": "markdown", "content": "mock"}}],
    )

    assert outbox_ids == [77]
    assert len(transactions) == 1
    assert len(calls) == 2
    assert "INSERT INTO risk_scan_runs" in calls[0][0]
    assert "INSERT INTO risk_notification_outbox" in calls[1][0]
    assert calls[1][1][0] == "scan-2"


def test_outbox_dispatch_records_success_and_retry_without_losing_payload(monkeypatch):
    rows = [
        {"id": 1, "alert_id": 11, "attempt_count": 1, "payload": {"stock_code": "000001"}},
        {"id": 2, "alert_id": 12, "attempt_count": 1, "payload": {"stock_code": "000002"}},
    ]
    finished = []
    sent = []
    monkeypatch.setattr(RiskAlertRepo, "claim_outbox", lambda limit: rows)
    monkeypatch.setattr(
        "core.shared.infrastructure.notify.send_risk_alert_result",
        lambda **payload: sent.append(payload) or {
            "status": "sent" if payload["stock_code"] == "000001" else "failed",
            "error": None if payload["stock_code"] == "000001" else "provider_rejected",
        },
    )
    monkeypatch.setattr(
        RiskAlertRepo,
        "finish_outbox",
        lambda row, outcome: finished.append((row["id"], outcome["status"], outcome["error"])) or ("sent" if outcome["status"] == "sent" else "retry"),
    )

    result = dispatch_risk_notification_outbox()

    assert sent == [{"stock_code": "000001"}, {"stock_code": "000002"}]
    assert finished == [(1, "sent", None), (2, "failed", "provider_rejected")]
    assert result == {"claimed": 2, "sent": 1, "retry": 1, "failed": 0, "delivery_unknown": 0, "state_update_failed": 0}


def test_outbox_dispatch_sends_scheduled_markdown_with_wecom_outcome(monkeypatch):
    row = {"id": 3, "alert_id": None, "attempt_count": 1, "payload": {"message_type": "markdown", "content": "mock"}}
    sent = []
    finished = []
    monkeypatch.setattr(RiskAlertRepo, "claim_outbox", lambda limit: [row])
    monkeypatch.setattr("core.shared.infrastructure.notify.get_wecom", lambda: type("Notifier", (), {
        "send_markdown_result": lambda self, content: sent.append(content) or {"status": "sent", "error": None}
    })())
    monkeypatch.setattr(RiskAlertRepo, "finish_outbox", lambda r, outcome: finished.append(outcome) or "sent")

    result = dispatch_risk_notification_outbox()

    assert sent == ["mock"]
    assert finished == [{"status": "sent", "error": None}]
    assert result["sent"] == 1


def test_unknown_delivery_result_is_not_automatically_retried(monkeypatch):
    calls = []

    class Cursor:
        rowcount = 1

        def execute(self, sql, params=()):
            calls.append((sql, params))

    monkeypatch.setattr("core.data.repos.risk.Database.transaction", lambda operation: operation(Cursor()))

    status = RiskAlertRepo.finish_outbox(
        {"id": 9, "alert_id": 19, "attempt_count": 1},
        {"status": "unknown", "error": "network timeout"},
    )

    assert status == "delivery_unknown"
    assert calls[0][1][0] == "delivery_unknown"
    assert len(calls) == 1


def test_outbox_persists_delivery_channel_and_provider_receipt(monkeypatch):
    calls = []

    class Cursor:
        rowcount = 1

        def execute(self, sql, params=()):
            calls.append((sql, params))

    monkeypatch.setattr("core.data.repos.risk.Database.transaction", lambda operation: operation(Cursor()))

    status = RiskAlertRepo.finish_outbox(
        {"id": 10, "alert_id": None, "attempt_count": 1},
        {"status": "sent", "channel": "wecom", "provider_response": {"errcode": 0}, "error": None},
    )

    assert status == "sent"
    assert calls[0][1][2] == "wecom"
    assert calls[0][1][3] == '{"errcode": 0}'


def test_resolve_unknown_outbox_requires_current_unknown_state_and_audits_in_transaction(monkeypatch):
    calls = []

    class Cursor:
        rowcount = 1

        def execute(self, sql, params=()):
            calls.append((sql, params))

        def fetchone(self):
            return {"id": 10, "alert_id": 22, "status": "delivery_unknown"}

    transactions = []
    monkeypatch.setattr(
        "core.data.repos.risk.Database.transaction",
        lambda operation: transactions.append(operation) or operation(Cursor()),
    )

    result = RiskAlertRepo.resolve_unknown_outbox(
        10, "confirmed_sent", "admin-token", "已在企业微信会话核对该消息已显示", "wecom-log:2026-10-30",
    )

    assert result == {"id": 10, "status": "confirmed_sent", "resolution_status": "operator_confirmed_sent"}
    assert len(transactions) == 1
    assert "INSERT INTO risk_notification_outbox_resolutions" in calls[1][0]
    assert "UPDATE risk_notification_outbox" in calls[2][0]
    assert calls[2][1][0] == "confirmed_sent"
    assert "UPDATE risk_alerts SET notified=1" in calls[3][0]


def test_wecom_delivery_outcome_distinguishes_sent_rejected_and_unknown(monkeypatch):
    class Response:
        ok = True

        def __init__(self, result):
            self.result = result

        def json(self):
            return self.result

    notifier = WeComNotifier(key="test-key")
    monkeypatch.setattr("core.shared.infrastructure.notify.requests.post", lambda *args, **kwargs: Response({"errcode": 0}))
    assert notifier.send_markdown_result("mock") == {
        "status": "sent", "channel": "wecom", "provider_response": {"errcode": 0}, "error": None,
    }

    monkeypatch.setattr("core.shared.infrastructure.notify.requests.post", lambda *args, **kwargs: Response({"errcode": 1, "errmsg": "rejected"}))
    rejected = notifier.send_markdown_result("mock")
    assert rejected["status"] == "failed"
    assert rejected["channel"] == "wecom"
    assert rejected["provider_response"] == {"errcode": 1, "errmsg": "rejected"}
    assert "rejected" in rejected["error"]

    def timeout(*args, **kwargs):
        raise TimeoutError("response lost")

    monkeypatch.setattr("core.shared.infrastructure.notify.requests.post", timeout)
    unknown = notifier.send_markdown_result("mock")
    assert unknown["status"] == "unknown"
    assert "response lost" in unknown["error"]


def test_notification_outbox_rejects_missing_idempotency_key(monkeypatch):
    def fail_if_transaction_called(_):
        raise AssertionError("invalid outbox item must fail before transaction")

    monkeypatch.setattr("core.data.repos.risk.Database.transaction", fail_if_transaction_called)

    try:
        RiskAlertRepo.add(
            "000001", "测试", "高风险", 8,
            notification_payload={"stock_code": "000001"},
        )
    except ValueError as exc:
        assert "幂等键" in str(exc)
    else:
        raise AssertionError("expected missing idempotency key to be rejected")


def test_outbox_migration_upgrades_legacy_columns_and_missing_scan_index(monkeypatch):
    executed = []
    table = {"alert_id": "NO", "idempotency_key": "NO", "payload_json": "NO", "scan_id": "YES"}
    monkeypatch.setattr("core.data.risk_notification_migration.Database.execute", lambda sql: executed.append(sql))
    monkeypatch.setattr(
        "core.data.risk_notification_migration.Database.fetchall",
        lambda sql, params=(): (
            [{"COLUMN_NAME": key, "IS_NULLABLE": nullable} for key, nullable in table.items()]
            if "information_schema.COLUMNS" in sql
            else [{"INDEX_NAME": "PRIMARY"}, {"INDEX_NAME": "uq_risk_notification_idempotency"}]
            if "information_schema.STATISTICS" in sql
            else [],
        )[0],
    )

    ensure_risk_notification_outbox()

    assert any("MODIFY alert_id INT DEFAULT NULL" in sql for sql in executed)
    assert any("ADD KEY idx_risk_notification_scan" in sql for sql in executed)
    assert any("CREATE TABLE IF NOT EXISTS `risk_scan_runs`" in sql for sql in executed)


def test_outbox_migration_does_not_recreate_existing_scan_index(monkeypatch):
    executed = []
    monkeypatch.setattr("core.data.risk_notification_migration.Database.execute", lambda sql: executed.append(sql))
    monkeypatch.setattr(
        "core.data.risk_notification_migration.Database.fetchall",
        lambda sql, params=(): (
            [{"COLUMN_NAME": "alert_id", "IS_NULLABLE": "YES"}, {"COLUMN_NAME": "scan_id", "IS_NULLABLE": "YES"}]
            if "information_schema.COLUMNS" in sql
            else [{"INDEX_NAME": "PRIMARY"}, {"INDEX_NAME": "idx_risk_notification_scan"}]
        ),
    )

    ensure_risk_notification_outbox()

    assert not any("ADD KEY idx_risk_notification_scan" in sql for sql in executed)


def test_scanner_persists_assessment_with_notification_payload_atomically(monkeypatch):
    scanner = PortfolioScanner.__new__(PortfolioScanner)
    captured = {}
    monkeypatch.setattr("core.data.Database.is_available", lambda: True)

    def add(**kwargs):
        captured.update(kwargs)
        return 41

    monkeypatch.setattr("core.data.RiskAlertRepo.add", add)
    result = {
        "stock_code": "000001",
        "stock_name": "测试",
        "risk_level": "高风险",
        "risk_score": 8,
        "risk_detail": "风险详情",
        "suggestion": "复核",
        "shares": 100,
        "cost_price": 10.5,
        "scan_id": "scan-7",
    }

    alert_id = scanner.persist_alert(result, notify=True)

    assert alert_id == 41
    assert captured["idempotency_key"] == "ai-risk:scan-7:000001"
    assert captured["notification_payload"] == {
        "stock_name": "测试",
        "stock_code": "000001",
        "risk_level": "高风险",
        "risk_score": 8,
        "risk_detail": "风险详情",
        "suggestion": "复核",
        "shares": 100,
        "cost_price": 10.5,
    }


def test_batch_scan_enqueues_notification_and_returns_pending_state(monkeypatch):
    scanner = PortfolioScanner.__new__(PortfolioScanner)
    holding = {"code": "000001", "name": "测试", "risk_threshold": 7}
    monkeypatch.setattr("core.data.HoldingsRepo.get_all", lambda: [holding])
    monkeypatch.setattr(scanner, "scan_holding", lambda *args, **kwargs: {
        "stock_code": "000001", "stock_name": "测试", "risk_level": "高风险", "risk_score": 8,
        "risk_detail": "详情", "suggestion": "复核", "shares": 100, "cost_price": 10,
    })
    monkeypatch.setattr(scanner, "persist_alert", lambda result, notify=False: 56 if notify else None)

    results = scanner.scan_all(threshold=5)

    assert results[0]["alert_id"] == 56
    assert results[0]["alert_triggered"] is True
    assert results[0]["notification_status"] == "pending"
