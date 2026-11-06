import os
import uuid

import pymysql
import pytest


@pytest.fixture
def isolated_risk_schema():
    if os.environ.get("RISK_NOTIFICATION_TEST_MYSQL") != "1":
        pytest.skip("set RISK_NOTIFICATION_TEST_MYSQL=1 to run isolated risk notification MySQL integration")

    from core.shared import config
    import core.data.connection as connection_module

    if config.MYSQL_HOST not in {"127.0.0.1", "localhost", "::1"}:
        pytest.skip("risk notification integration is restricted to local MySQL")

    database_name = "codex_risk_" + uuid.uuid4().hex[:12]
    original_database = config.MYSQL_DATABASE
    original_pool = connection_module._pool
    connection_module._pool = None
    admin = pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
    )
    pool = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE `{database_name}`")
            cursor.execute("CREATE TABLE schema_migrations (version VARCHAR(80) PRIMARY KEY, applied_at DATETIME DEFAULT CURRENT_TIMESTAMP) ENGINE=InnoDB")
            cursor.execute("""CREATE TABLE risk_notification_outbox (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                alert_id INT NOT NULL,
                idempotency_key VARCHAR(160) NOT NULL,
                payload_json LONGTEXT NOT NULL,
                status VARCHAR(24) NOT NULL DEFAULT 'pending',
                attempt_count INT NOT NULL DEFAULT 0,
                available_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                lease_until DATETIME DEFAULT NULL,
                last_error VARCHAR(1000) DEFAULT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                sent_at DATETIME DEFAULT NULL,
                UNIQUE KEY uq_risk_notification_idempotency (idempotency_key),
                KEY idx_risk_notification_ready (status, available_at, id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        config.MYSQL_DATABASE = database_name
        pool = connection_module._get_pool()
        yield database_name
    finally:
        if pool is not None:
            pool.shutdown()
        config.MYSQL_DATABASE = original_database
        connection_module._pool = original_pool
        with admin.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


def test_legacy_migration_atomic_scan_record_and_outbox_delivery(isolated_risk_schema):
    from core.data.connection import Database
    from core.data.repos.risk import RiskAlertRepo
    from core.data.risk_notification_migration import ensure_risk_notification_outbox

    ensure_risk_notification_outbox()
    ensure_risk_notification_outbox()

    columns = {row["COLUMN_NAME"]: row["IS_NULLABLE"] for row in Database.fetchall(
        "SELECT COLUMN_NAME,IS_NULLABLE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='risk_notification_outbox'"
    )}
    indexes = {row["INDEX_NAME"] for row in Database.fetchall(
        "SELECT INDEX_NAME FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='risk_notification_outbox'"
    )}
    assert columns["alert_id"] == "YES"
    assert "scan_id" in columns
    assert "delivery_channel" in columns
    assert "provider_response_json" in columns
    assert "idx_risk_notification_scan" in indexes

    scan = {
        "scan_id": str(uuid.uuid4()), "source": "stop_loss", "session_name": "隔离测试",
        "regime_name": "steady_flat", "status": "degraded", "expected_count": 2,
        "scanned_count": 1, "unknown_count": 1, "dynamic_unknown_count": 0,
        "triggered_count": 1, "assessment": {"unknown": [{"code": "000001"}]},
    }
    notification = {
        "idempotency_key": "risk-test:" + str(uuid.uuid4()),
        "payload": {"message_type": "markdown", "content": "isolated only"},
    }
    ids = RiskAlertRepo.record_scan(scan, [notification])
    assert len(ids) == 1
    assert RiskAlertRepo.get_recent_scans()[0]["assessment"]["unknown"][0]["code"] == "000001"

    rollback_scan_id = str(uuid.uuid4())
    with pytest.raises(pymysql.err.IntegrityError):
        RiskAlertRepo.record_scan(
            {**scan, "scan_id": rollback_scan_id},
            [{**notification, "payload": {"message_type": "markdown", "content": "duplicate"}}],
        )
    assert Database.fetchone("SELECT scan_id FROM risk_scan_runs WHERE scan_id=%s", (rollback_scan_id,)) is None

    claimed = RiskAlertRepo.claim_outbox()
    assert len(claimed) == 1
    assert RiskAlertRepo.finish_outbox(
        claimed[0],
        {"status": "sent", "channel": "wecom", "provider_response": {"errcode": 0}, "error": None},
    ) == "sent"
    delivered = RiskAlertRepo.get_outbox_recent()[0]
    assert delivered["status"] == "sent"
    assert delivered["delivery_channel"] == "wecom"
    assert delivered["provider_response_json"] == '{"errcode": 0}'

    unknown_scan = {**scan, "scan_id": str(uuid.uuid4())}
    unknown_notification = {**notification, "idempotency_key": "risk-test:" + str(uuid.uuid4())}
    unknown_id = RiskAlertRepo.record_scan(unknown_scan, [unknown_notification])[0]
    unknown_row = next(row for row in RiskAlertRepo.claim_outbox() if row["id"] == unknown_id)
    assert RiskAlertRepo.finish_outbox(unknown_row, {"status": "unknown", "channel": "wecom", "error": "timeout"}) == "delivery_unknown"
    resolved = RiskAlertRepo.resolve_unknown_outbox(
        unknown_id, "confirmed_not_sent_requeue", "isolated-admin", "渠道后台记录确认该请求未接收", "wecom-audit:1",
    )
    assert resolved["status"] == "pending"
    assert RiskAlertRepo.get_outbox_resolutions(unknown_id)[0]["actor"] == "isolated-admin"
    requeued = next(row for row in RiskAlertRepo.claim_outbox() if row["id"] == unknown_id)
    assert requeued["attempt_count"] == 2
    assert RiskAlertRepo.finish_outbox(requeued, {"status": "unknown", "channel": "wecom", "error": "timeout"}) == "delivery_unknown"
    resolved = RiskAlertRepo.resolve_unknown_outbox(
        unknown_id, "confirmed_sent", "isolated-admin", "渠道后台记录确认该消息已被接收", "wecom-audit:2",
    )
    assert resolved["status"] == "confirmed_sent"
    with pytest.raises(ValueError, match="只有发送结果未知"):
        RiskAlertRepo.resolve_unknown_outbox(
            unknown_id, "confirmed_not_sent_requeue", "isolated-admin", "再次核对后确认没有接收", "wecom-audit:3",
        )
