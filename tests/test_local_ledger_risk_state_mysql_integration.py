import re
import uuid
import datetime
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from threading import Barrier

import pymysql
import pytest
from unittest.mock import patch

import core.data.connection as connection
from core.data.connection import Database
from core.data.repos.local_ledger import LocalLedgerRepo, account_half_sold_setting_key
from core.data.repos.local_order_intent import LocalOrderIntentRepo
from core.data.repos.rebalance import RebalanceSignalRepo
from core.domain.portfolio.decision import decision_fingerprint, holdings_fingerprint_rows
from core.shared import config


@pytest.mark.integration
@pytest.mark.parametrize("failure_table,failure_operation", [
    ("holdings", "UPDATE"),
    ("local_ledger_accounts", "UPDATE"),
    ("rebalance_signals", "UPDATE"),
    ("settings", "INSERT"),
    ("local_ledger_events", "INSERT"),
])
def test_manual_fill_risk_state_commits_and_rolls_back_with_local_ledger(failure_table, failure_operation):
    database_name = "codex_risk_state_" + uuid.uuid4().hex[:12]
    try:
        admin = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            autocommit=True, connect_timeout=3,
        )
    except Exception as error:
        pytest.skip(f"隔离 MySQL 不可用: {type(error).__name__}")

    original_database = config.MYSQL_DATABASE
    created = False
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cursor:
                for table in ("settings", "schema_migrations", "holdings", "rebalance_signals",
                              "local_ledger_accounts", "local_ledger_events", "local_order_intents"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cursor.execute(match.group())
        finally:
            isolated.close()

        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name

        Database.execute(
            "INSERT INTO local_ledger_accounts (account_id,environment,status,cash,frozen,version) "
            "VALUES ('LOCAL-LIVE','live','active',10000,0,1)"
        )
        Database.execute(
            "INSERT INTO holdings (account_id,code,name,shares,cost_price,buy_date) "
            "VALUES ('LOCAL-LIVE','000001','测试股',200,10,DATE_SUB(CURRENT_DATE, INTERVAL 1 DAY))"
        )
        Database.execute(
            "INSERT INTO settings (setting_key,setting_value) VALUES ('rebalance_half_sold','[\"600001\"]')"
        )
        Database.execute(
            "INSERT INTO settings (setting_key,setting_value) VALUES (%s,%s)",
            (account_half_sold_setting_key("LOCAL-OTHER"), '["600002"]'),
        )
        from core.data.local_ledger_migration import ensure_account_scoped_half_sold
        ensure_account_scoped_half_sold()
        live_risk_key = account_half_sold_setting_key("LOCAL-LIVE")
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key=%s",
                                 (live_risk_key,))["setting_value"] == '["600001"]'
        snapshot = Database.fetchall(
            "SELECT code,shares,cost_price,buy_date FROM holdings WHERE account_id='LOCAL-LIVE' ORDER BY code"
        )
        fingerprint = decision_fingerprint(holdings_fingerprint_rows(snapshot))
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,planned_price,amount,"
            "account_id,account_version,input_fingerprint) VALUES (%s,'000001','测试股','reduce','止盈减半',"
            "100,11,1100,'LOCAL-LIVE',1,%s)",
            ("HALF-1", fingerprint),
        )
        signal_id = Database.fetchone("SELECT id FROM rebalance_signals WHERE batch_id='HALF-1'")["id"]
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,planned_price,amount,"
            "account_id,account_version,input_fingerprint,execution_status) "
            "VALUES ('STALE-BATCH','000002','旧批次信号','buy','建议买入',100,10,1000,'LOCAL-LIVE',1,%s,'pending')",
            (fingerprint,),
        )
        stale_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='STALE-BATCH'"
        )["id"]
        request_id = str(uuid.uuid4())
        detail = {
            "request_id": request_id, "shares": 100, "price": "11.000",
            "commission": "0.30", "stamp_duty": "0.20",
            "expected_account_version": 1, "actor": "admin",
        }

        Database.execute(
            f"CREATE TRIGGER reject_manual_fill_write BEFORE {failure_operation} ON {failure_table} "
            "FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' "
            "SET MESSAGE_TEXT='reject manual fill for rollback test'; END"
        )
        with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
            with pytest.raises(pymysql.err.Error):
                RebalanceSignalRepo.record_execution(signal_id, detail, decision_fingerprint)
        assert LocalLedgerRepo.get_account()["cash"] == 10000
        assert Database.fetchone("SELECT shares FROM holdings WHERE code='000001'")["shares"] == 200
        assert Database.fetchone("SELECT filled_shares,execution_status FROM rebalance_signals WHERE id=%s",
                                 (signal_id,)) == {"filled_shares": 0, "execution_status": "pending"}
        assert Database.fetchone("SELECT account_version,execution_status FROM rebalance_signals WHERE id=%s",
                                 (stale_signal_id,)) == {"account_version": 1, "execution_status": "pending"}
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key=%s",
                                 (live_risk_key,))["setting_value"] == '["600001"]'

        Database.execute("DROP TRIGGER reject_manual_fill_write")
        start_together = Barrier(2)

        def submit_duplicate(_):
            start_together.wait(timeout=5)
            return RebalanceSignalRepo.record_execution(signal_id, detail, decision_fingerprint)

        with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                concurrent_results = list(pool.map(
                    submit_duplicate,
                    range(2),
                ))
        result = next(item for item in concurrent_results if not item["already_recorded"])
        retry = next(item for item in concurrent_results if item["already_recorded"])
        Database.execute(
            "UPDATE local_ledger_accounts SET status='blocked' WHERE account_id='LOCAL-LIVE'"
        )
        with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
            blocked_retry = RebalanceSignalRepo.record_execution(signal_id, detail, decision_fingerprint)
        Database.execute("UPDATE local_ledger_accounts SET status='active' WHERE account_id='LOCAL-LIVE'")
        assert result["execution_status"] == "executed"
        assert retry["already_recorded"] is True
        assert blocked_retry["already_recorded"] is True
        assert blocked_retry["execution_status"] == "executed"
        assert result["invalidated_signal_ids"] == [stale_signal_id]
        assert retry["invalidated_signal_ids"] == [stale_signal_id]
        assert Database.fetchone("SELECT account_version,execution_status FROM rebalance_signals WHERE id=%s",
                                 (stale_signal_id,)) == {"account_version": 1, "execution_status": "cancelled"}
        invalidated_event = Database.fetchone(
            "SELECT JSON_EXTRACT(payload_json,'$.invalidated_signal_details[0].invalidated_remaining_shares') "
            "AS remaining FROM local_ledger_events WHERE event_key=%s",
            (f"manual:LOCAL-LIVE:{request_id}",),
        )
        assert int(invalidated_event["remaining"]) == 100
        assert Database.fetchone("SELECT shares FROM holdings WHERE code='000001'")["shares"] == 100
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key=%s",
                                 (live_risk_key,))["setting_value"] == '["000001", "600001"]'
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key=%s",
                                 (account_half_sold_setting_key("LOCAL-OTHER"),))["setting_value"] == '["600002"]'
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key='rebalance_half_sold'")[
            "setting_value"] == '["600001"]'
        ensure_account_scoped_half_sold()
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key=%s",
                                 (live_risk_key,))["setting_value"] == '["000001", "600001"]'
        assert Database.fetchone("SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='manual_fill'")[
            "n"] == 1
        assert Database.fetchone(
            "SELECT JSON_UNQUOTE(JSON_EXTRACT(payload_json,'$.risk_state_change')) AS state_change "
            "FROM local_ledger_events WHERE event_key=%s",
            (f"manual:LOCAL-LIVE:{request_id}",),
        )["state_change"] == "half_sold"

        account = LocalLedgerRepo.get_account()
        snapshot = Database.fetchall(
            "SELECT code,shares,cost_price,buy_date FROM holdings WHERE account_id='LOCAL-LIVE' ORDER BY code"
        )
        fingerprint = decision_fingerprint(holdings_fingerprint_rows(snapshot))
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,planned_price,amount,"
            "account_id,account_version,input_fingerprint) VALUES (%s,'000001','测试股','sell','止盈卖出',"
            "100,11,1100,'LOCAL-LIVE',%s,%s)",
            ("HALF-2", account["version"], fingerprint),
        )
        full_sell_id = Database.fetchone("SELECT id FROM rebalance_signals WHERE batch_id='HALF-2'")["id"]
        with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
            full_sell = RebalanceSignalRepo.record_execution(full_sell_id, {
                "request_id": str(uuid.uuid4()), "shares": 100, "price": "11.000",
                "commission": "0.30", "stamp_duty": "0.20",
                "expected_account_version": account["version"], "actor": "admin",
            }, decision_fingerprint)
        assert full_sell["execution_status"] == "executed"
        assert Database.fetchone("SELECT shares FROM holdings WHERE code='000001'") is None
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key=%s",
                                 (live_risk_key,))["setting_value"] == '["600001"]'
        assert Database.fetchone("SELECT setting_value FROM settings WHERE setting_key=%s",
                                 (account_half_sold_setting_key("LOCAL-OTHER"),))["setting_value"] == '["600002"]'
    finally:
        config.MYSQL_DATABASE = original_database
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        if created:
            with admin.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_manual_buy_fill_holding_insert_failure_rolls_back_account_and_signal():
    database_name = "codex_buy_fill_" + uuid.uuid4().hex[:12]
    try:
        admin = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            autocommit=True, connect_timeout=3,
        )
    except Exception as error:
        pytest.skip(f"隔离 MySQL 不可用: {type(error).__name__}")

    original_database = config.MYSQL_DATABASE
    created = False
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cursor:
                for table in ("settings", "schema_migrations", "holdings", "rebalance_signals",
                              "local_ledger_accounts", "local_ledger_events", "local_order_intents"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cursor.execute(match.group())
        finally:
            isolated.close()

        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name

        Database.execute(
            "INSERT INTO local_ledger_accounts (account_id,environment,status,cash,frozen,version) "
            "VALUES ('LOCAL-LIVE','live','active',10000,0,1)"
        )
        fingerprint = decision_fingerprint([])
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,planned_price,amount," 
            "account_id,account_version,input_fingerprint) "
            "VALUES ('BUY-INSERT-1','000002','买入测试股','buy','建议买入',100,10,1000,'LOCAL-LIVE',1,%s)",
            (fingerprint,),
        )
        signal_id = Database.fetchone("SELECT id FROM rebalance_signals WHERE batch_id='BUY-INSERT-1'")["id"]
        detail = {
            "request_id": str(uuid.uuid4()), "shares": 100, "price": "10.000",
            "commission": "0.30", "stamp_duty": "0.00",
            "expected_account_version": 1, "actor": "admin",
        }
        Database.execute(
            "CREATE TRIGGER reject_manual_buy_holding BEFORE INSERT ON holdings "
            "FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' "
            "SET MESSAGE_TEXT='reject buy holding insert for rollback test'; END"
        )
        with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
            with pytest.raises(pymysql.err.Error):
                RebalanceSignalRepo.record_execution(signal_id, detail, decision_fingerprint)

        assert Database.fetchone("SELECT cash,version FROM local_ledger_accounts WHERE account_id='LOCAL-LIVE'") == {
            "cash": Decimal("10000.00"), "version": 1,
        }
        assert Database.fetchone("SELECT COUNT(*) AS n FROM holdings WHERE account_id='LOCAL-LIVE'")["n"] == 0
        assert Database.fetchone("SELECT filled_shares,execution_status FROM rebalance_signals WHERE id=%s",
                                 (signal_id,)) == {"filled_shares": 0, "execution_status": "pending"}
        assert Database.fetchone("SELECT COUNT(*) AS n FROM local_ledger_events")["n"] == 0

        Database.execute("DROP TRIGGER reject_manual_buy_holding")
        with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
            result = RebalanceSignalRepo.record_execution(signal_id, detail, decision_fingerprint)
            retry = RebalanceSignalRepo.record_execution(signal_id, detail, decision_fingerprint)
        assert result["execution_status"] == "executed"
        assert retry["already_recorded"] is True
        assert Database.fetchone("SELECT cash,version FROM local_ledger_accounts WHERE account_id='LOCAL-LIVE'") == {
            "cash": Decimal("8999.70"), "version": 2,
        }
        assert Database.fetchone("SELECT shares FROM holdings WHERE account_id='LOCAL-LIVE' AND code='000002'")[
            "shares"] == 100
        assert Database.fetchone("SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='manual_fill'")[
            "n"] == 1
    finally:
        config.MYSQL_DATABASE = original_database
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        if created:
            with admin.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_reserve_and_release_invalidate_old_batch_in_isolated_mysql():
    database_name = "codex_intent_version_" + uuid.uuid4().hex[:12]
    try:
        admin = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            autocommit=True, connect_timeout=3,
        )
    except Exception as error:
        pytest.skip(f"隔离 MySQL 不可用: {type(error).__name__}")

    original_database = config.MYSQL_DATABASE
    created = False
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cursor:
                for table in ("settings", "schema_migrations", "holdings", "rebalance_signals",
                              "local_ledger_accounts", "local_ledger_events", "local_order_intents"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cursor.execute(match.group())
        finally:
            isolated.close()

        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        Database.execute(
            "INSERT INTO local_ledger_accounts (account_id,environment,status) "
            "VALUES ('LOCAL-LIVE','live','uninitialized')"
        )
        fingerprint = decision_fingerprint([])
        LocalLedgerRepo.activate("10000.00", fingerprint, "admin", "隔离测试期初核准并校验账本")

        now = datetime.datetime.now()
        bind_snapshot = {
            "can_bind": True, "broker_account_id": "QMT-INT-1", "local_version": 1,
            "broker_status": "fresh", "broker_identity_verified": True,
            "missing_items": [], "position_differences": [], "sellable_missing_codes": [],
            "pending_orders": [], "pending_journal_orders": [], "unresolved_order_reports": [],
            "pending_local_intent": False, "broker_positions": {}, "broker_sellable": {},
            "local_positions": {}, "cash_difference": 0, "cash_balance_difference": 0,
            "frozen_difference": 0, "broker_cash": 10000, "broker_frozen": 0,
            "broker_available": 10000,
            "broker_trade_report_health": {
                "conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0,
                "legacy_quarantined": 0,
            },
            "broker_received_at": now.isoformat(), "broker_as_of": now.isoformat(),
            "broker_snapshot_started_at": (now - datetime.timedelta(seconds=1)).isoformat(),
            "broker_snapshot_completed_at": now.isoformat(),
        }
        LocalOrderIntentRepo.bind_broker("QMT-INT-1", 1, "admin", bind_snapshot)

        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,planned_price,amount,"
            "account_id,account_version,input_fingerprint,execution_status) "
            "VALUES ('CURRENT-BATCH','000001','当前计划','buy','建议买入',100,10,1000,'LOCAL-LIVE',2,%s,'pending')",
            (fingerprint,),
        )
        current_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='CURRENT-BATCH'"
        )["id"]
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,planned_price,amount,"
            "account_id,account_version,input_fingerprint,execution_status) "
            "VALUES ('OLD-BATCH','000002','旧批次计划','buy','建议买入',200,10,2000,'LOCAL-LIVE',2,%s,'pending')",
            (fingerprint,),
        )
        old_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='OLD-BATCH'"
        )["id"]

        intent_id = str(uuid.uuid4())
        prepare_snapshot = {**bind_snapshot, "can_execute": True, "local_version": 2,
                            "broker_account_id": "QMT-INT-1"}
        prepare_detail = {
            "intent_id": intent_id, "signal_id": current_signal_id,
            "broker_account_id": "QMT-INT-1", "expected_account_version": 2,
            "quantity": 100, "limit_price": "10.000", "fee_cap": "1.00",
            "actor": "admin", "reconciliation": prepare_snapshot,
        }
        Database.execute(
            "CREATE TRIGGER reject_reserve_event BEFORE INSERT ON local_ledger_events "
            "FOR EACH ROW BEGIN IF NEW.event_type='reserve' THEN SIGNAL SQLSTATE '45000' "
            "SET MESSAGE_TEXT='reject reserve event'; END IF; END"
        )
        with pytest.raises(pymysql.err.Error):
            LocalOrderIntentRepo.prepare(prepare_detail, decision_fingerprint)
        assert LocalLedgerRepo.get_account()["version"] == 2
        assert LocalLedgerRepo.get_account()["frozen"] == Decimal("0.00")
        assert Database.fetchone(
            "SELECT execution_status FROM rebalance_signals WHERE id=%s", (old_signal_id,),
        )["execution_status"] == "pending"
        assert Database.fetchone("SELECT COUNT(*) AS n FROM local_order_intents")["n"] == 0
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        Database.execute("DROP TRIGGER reject_reserve_event")

        prepared = LocalOrderIntentRepo.prepare(prepare_detail, decision_fingerprint)
        assert prepared["invalidated_signal_ids"] == [old_signal_id]
        assert prepared["invalidated_signal_details"][0]["invalidated_remaining_shares"] == 200
        assert Database.fetchone(
            "SELECT account_version,execution_status FROM rebalance_signals WHERE id=%s", (old_signal_id,),
        ) == {"account_version": 2, "execution_status": "cancelled"}
        assert Database.fetchone(
            "SELECT account_version,execution_status FROM rebalance_signals WHERE id=%s", (current_signal_id,),
        ) == {"account_version": 3, "execution_status": "pending"}
        assert LocalLedgerRepo.get_account()["frozen"] == Decimal("1001.00")
        assert LocalLedgerRepo.audit_integrity()["ok"] is True

        released = LocalOrderIntentRepo.cancel_prepared(intent_id, "admin")
        duplicate_release = LocalOrderIntentRepo.cancel_prepared(intent_id, "admin")
        assert released["state"] == "cancelled" and released["already_cancelled"] is False
        assert duplicate_release["state"] == "cancelled" and duplicate_release["already_cancelled"] is True
        assert duplicate_release["invalidated_signal_details"] == released["invalidated_signal_details"]
        account = LocalLedgerRepo.get_account()
        assert account["cash"] == Decimal("10000.00")
        assert account["frozen"] == Decimal("0.00")
        assert account["version"] == 4
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type IN ('reserve','reserve_release')"
        )["n"] == 2
    finally:
        config.MYSQL_DATABASE = original_database
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        if created:
            with admin.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()
