import datetime
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymysql
import pytest

import core.data.connection as connection
from core.shared import config
from core.data.connection import Database
from core.data.repos.broker_order_report import BrokerOrderReportRepo
from core.data.repos.broker_trade_report import BrokerTradeReportRepo
from core.data.repos.local_ledger import LocalLedgerRepo
from core.data.repos.local_order_intent import LocalOrderIntentRepo
from core.data.repos.rebalance import RebalanceSignalRepo
from core.data.repos.session import RebalanceSessionRepo
from core.data.repos.holdings import HoldingsRepo, StaleHoldingError
from core.domain.portfolio.decision import decision_fingerprint, holdings_fingerprint_rows
from core.data.local_ledger_migration import ensure_local_ledger_schema
from core.adapters.broker.fake_broker import FakeBroker
from core.adapters.broker.easytrader_broker import EasytraderBroker
from core.adapters.broker.qmt_broker import QmtBroker
from core.adapters.broker.base import BaseBroker, BrokerPersistenceConflict


def _ready_reconciliation(broker_account_id, local_version):
    account = LocalLedgerRepo.get_account()
    positions = {
        row["code"]: int(row["shares"])
        for row in Database.fetchall(
            "SELECT code,shares FROM holdings WHERE account_id=%s ORDER BY code",
            (account["account_id"],),
        )
    }
    now = datetime.datetime.now()
    return {
        "can_bind": True,
        "can_execute": True,
        "broker_account_id": broker_account_id,
        "local_version": local_version,
        "broker_status": "fresh",
        "broker_identity_verified": True,
        "missing_items": [],
        "position_differences": [],
        "sellable_missing_codes": [],
        "pending_orders": [],
        "pending_journal_orders": [],
        "unresolved_order_reports": [],
        "pending_local_intent": False,
        "broker_positions": positions,
        "broker_sellable": positions.copy(),
        "local_positions": positions.copy(),
        "cash_difference": 0,
        "cash_balance_difference": 0,
        "frozen_difference": 0,
        "broker_cash": account["cash"],
        "broker_frozen": account["frozen"],
        "broker_available": account["available"],
        "broker_trade_report_health": {
            "conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0,
        },
        "broker_received_at": now.isoformat(),
        "broker_as_of": now.isoformat(),
        "broker_snapshot_started_at": (now - datetime.timedelta(seconds=1)).isoformat(),
        "broker_snapshot_completed_at": now.isoformat(),
    }


def _live_plan_fingerprint():
    account = LocalLedgerRepo.get_account()
    rows = Database.fetchall(
        "SELECT code,shares,cost_price,buy_date FROM holdings WHERE account_id=%s ORDER BY code",
        (account["account_id"],),
    )
    return decision_fingerprint(holdings_fingerprint_rows(rows))


@pytest.mark.integration
def test_order_reports_replay_partial_cancel_and_conflict_in_isolated_mysql():
    database_name = "codex_w1_" + uuid.uuid4().hex[:12]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS `broker_report_resolution_events`" in schema
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in ("holdings", "rebalance_signals", "rebalance_sessions",
                              "holdings_adjustments",
                              "accounts", "orders", "trades", "settings", "schema_migrations", "local_ledger_accounts",
                              "local_ledger_events", "local_order_intents",
                              "broker_order_states", "broker_order_submissions", "broker_order_reports",
                              "broker_trade_reports", "broker_trade_report_conflicts"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
                cur.execute("ALTER TABLE holdings DROP COLUMN account_id")
                cur.execute(
                    "INSERT INTO holdings (code,name,shares,cost_price,buy_date) "
                    "VALUES ('699999','legacy holding',100,10,'2026-10-28')"
                )
                cur.execute("ALTER TABLE holdings_adjustments DROP INDEX "
                            "idx_holdings_adjustments_account_code_time")
                cur.execute("ALTER TABLE holdings_adjustments DROP COLUMN account_id")
                cur.execute(
                    "INSERT INTO holdings_adjustments (code,action,before_json,after_json,reason) "
                    "VALUES ('000001','update',NULL,NULL,'legacy adjustment')"
                )
                for field in ("resolved_at", "resolved_by", "resolution_reason"):
                    cur.execute(f"ALTER TABLE broker_order_reports DROP COLUMN {field}")
                for field in ("resolved_at", "resolved_by", "resolution_reason", "resolution_evidence_json"):
                    cur.execute(f"ALTER TABLE broker_trade_report_conflicts DROP COLUMN {field}")
                cur.execute("ALTER TABLE local_order_intents DROP INDEX uk_local_intent_broker_order")
                for field in ("remaining_reserved_cash", "broker_order_id", "filled_quantity"):
                    cur.execute(f"ALTER TABLE local_order_intents DROP COLUMN {field}")
                for field in ("price_type", "limit_price"):
                    cur.execute(f"ALTER TABLE broker_order_states DROP COLUMN {field}")
                cur.execute("DROP TABLE broker_order_submissions")
                cur.execute(
                    "INSERT INTO local_order_intents (intent_id,account_id,broker_account_id,signal_id,"
                    "code,side,quantity,limit_price,fee_cap,reserved_cash,state,account_version) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    ("00000000-0000-4000-8000-000000000001", "LOCAL-LIVE", "QMT-LEGACY",
                     1, "000001", "buy", 100, 10, 1, 1001, "prepared", 3),
                )
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        assert Database.fetchone("SELECT @@session.time_zone AS time_zone")["time_zone"] == "+08:00"
        ensure_local_ledger_schema()
        assert Database.fetchone("SHOW TABLES LIKE 'broker_report_resolution_events'")
        assert Database.fetchone(
            "SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() "
            "AND table_name='broker_order_submissions' AND column_name='submission_json'"
        )
        assert Database.fetchone(
            "SELECT 1 FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA=DATABASE() "
            "AND TRIGGER_NAME='broker_order_submissions_no_update'"
        )
        from core.shared.trading_time import A_SHARE_TIMEZONE
        qmt = QmtBroker.__new__(QmtBroker)
        qmt._connected = True
        qmt._account = "READONLY-ACCOUNT"
        qmt._account_id = "QMT-READONLY-ACCOUNT"
        qmt._stock_account = object()
        qmt._trader = Mock()
        qmt._trader.query_stock_positions.return_value = []
        qmt._trader.query_stock_orders.return_value = [SimpleNamespace(
            account_id="READONLY-ACCOUNT", order_id="READONLY-FILLED-ORDER",
            stock_code="000001.SZ", order_type=23, order_status=56,
            order_volume=100, price_type=11, price=10, traded_volume=100,
        )]
        qmt._report_persistence_error = None
        qmt._lock = threading.RLock()
        qmt.get_account = Mock(return_value={
            "status": "fresh", "account_id": qmt.account_id,
            "as_of": datetime.datetime.now(A_SHARE_TIMEZONE).isoformat(),
            "cash": 10000, "frozen": 0, "available": 10000,
        })
        with patch.object(qmt, "replay_order_reports") as replay, \
             patch.object(qmt, "_record_order_report") as record_order:
            qmt_snapshot = qmt.get_reconciliation_snapshot()
        assert qmt_snapshot["orders"] == [{
            "id": "READONLY-FILLED-ORDER", "symbol": "000001.SZ", "side": "buy",
            "status": "filled", "quantity": 100, "price_type": "limit",
            "price": 10, "filled_qty": 100,
        }]
        replay.assert_not_called()
        record_order.assert_not_called()
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_order_reports WHERE broker='qmt' "
            "AND broker_account_id=%s", (qmt.account_id,),
        )["n"] == 0
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_order_states WHERE broker='qmt' "
            "AND broker_account_id=%s", (qmt.account_id,),
        )["n"] == 0
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_trade_reports WHERE broker='qmt' "
            "AND broker_account_id=%s", (qmt.account_id,),
        )["n"] == 0
        submission_order = {
            "id": "SUBMISSION-TRACE-1", "symbol": "000001.SZ", "side": "buy",
            "quantity": 100, "price_type": "limit", "price": Decimal("10.120"),
        }
        submission = BrokerOrderReportRepo.record_submission(
            "qmt", "QMT-TRACE", submission_order,
            {"symbol": "000001.SZ", "side": "buy", "quantity": 100,
             "price_type": "limit", "price": "10.120"},
            {"method": "order_stock", "order_type": 23, "price_type": 11,
             "price": 10.12, "quantity": 100},
            {"order_id": "SUBMISSION-TRACE-1"},
        )
        assert submission["broker_request"]["price_type"] == 11
        assert BrokerOrderReportRepo.record_submission(
            "qmt", "QMT-TRACE", submission_order,
            submission["client_request"], submission["broker_request"], submission["broker_response"],
        ) == submission
        with pytest.raises(pymysql.err.Error):
            Database.execute(
                "UPDATE broker_order_submissions SET submission_json='{}' "
                "WHERE broker='qmt' AND broker_account_id='QMT-TRACE' AND order_id='SUBMISSION-TRACE-1'"
            )
        with pytest.raises(pymysql.err.Error):
            Database.execute(
                "DELETE FROM broker_order_submissions "
                "WHERE broker='qmt' AND broker_account_id='QMT-TRACE' AND order_id='SUBMISSION-TRACE-1'"
            )
        assert BrokerOrderReportRepo.ingest(
            "qmt", "QMT-TRACE", {**submission_order, "filled_qty": 0, "status": "pending"},
            "query", raw_status=48, raw_fields={"price_type": 11},
        )["disposition"] == "applied"
        restored_submission = next(
            order for order in BrokerOrderReportRepo.list_current_states("qmt", "QMT-TRACE")
            if order["id"] == "SUBMISSION-TRACE-1"
        )["submission"]
        assert restored_submission == submission
        with pytest.raises(RuntimeError, match="提交证据冲突"):
            BrokerOrderReportRepo.record_submission(
                "qmt", "QMT-TRACE", submission_order,
                submission["client_request"], {"method": "order_stock", "price_type": 5},
                submission["broker_response"],
            )
        holding_identity_column = Database.fetchone("SHOW COLUMNS FROM holdings LIKE 'account_id'")
        assert holding_identity_column["Null"] == "YES"
        assert holding_identity_column["Default"] is None
        assert Database.fetchone(
            "SELECT account_id FROM holdings WHERE code='699999'",
        )["account_id"] is None
        Database.execute("UPDATE holdings SET account_id='LOCAL-LIVE' WHERE code='699999'")
        Database.execute(
            "ALTER TABLE holdings MODIFY COLUMN account_id VARCHAR(64) NOT NULL DEFAULT 'LOCAL-LIVE'"
        )
        ensure_local_ledger_schema()
        holding_identity_column = Database.fetchone("SHOW COLUMNS FROM holdings LIKE 'account_id'")
        assert holding_identity_column["Null"] == "YES"
        assert holding_identity_column["Default"] is None
        assert Database.fetchone(
            "SELECT account_id FROM holdings WHERE code='699999'",
        )["account_id"] == "LOCAL-LIVE"
        assert Database.fetchone(
            "SELECT account_id FROM holdings_adjustments WHERE reason='legacy adjustment'"
        )["account_id"] == "LOCAL-LIVE"
        assert Database.fetchone(
            "SHOW INDEX FROM holdings_adjustments WHERE Key_name=%s",
            ("idx_holdings_adjustments_account_code_time",),
        )
        assert Database.fetchone(
            "SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() "
            "AND table_name='broker_order_reports' AND column_name='resolved_at'"
        )
        Database.execute(
            "INSERT INTO broker_order_reports (report_key,broker,broker_account_id,order_id,symbol,"
            "side,quantity,filled_qty,status,source,disposition,raw_json,resolved_at,resolved_by,"
            "resolution_reason) VALUES (%s,'qmt','QMT-LEGACY','OLD-1','000001.SZ','buy',100,0,"
            "'cancelled','query','conflict','{}','2026-10-26 10:00:00','old-admin',%s)",
            ("c" * 64, "旧委托回报已人工核对"),
        )
        Database.execute(
            "INSERT INTO broker_trade_report_conflicts (conflict_key,broker,broker_account_id,trade_id,"
            "payload_hash,raw_json,resolved_at,resolved_by,resolution_reason,resolution_evidence_json) "
            "VALUES (%s,'qmt','QMT-LEGACY','OLD-T1',%s,'{}','2026-10-26 11:00:00',"
            "'old-admin',%s,%s)",
            ("d" * 64, "e" * 64, "旧成交回报已人工核对", '{"id":"OLD-T1"}'),
        )
        ensure_local_ledger_schema()
        ensure_local_ledger_schema()
        legacy_history = Database.fetchall(
            "SELECT report_type,report_key,actor,reason,evidence_json FROM broker_report_resolution_events "
            "WHERE broker_account_id='QMT-LEGACY' ORDER BY id"
        )
        assert [(row["report_type"], row["report_key"]) for row in legacy_history] == [
            ("order", "c" * 64), ("trade", "d" * 64),
        ]
        assert [row["actor"] for row in legacy_history] == ["old-admin", "old-admin"]
        assert Database.fetchone("SELECT remaining_reserved_cash FROM local_order_intents "
                                 "WHERE intent_id=%s", ("00000000-0000-4000-8000-000000000001",))[
                                     "remaining_reserved_cash"] == 1001
        for record_id, environment, source in (
                ("q-order-a", None, "callback"), ("q-order-b", "live", "  "),
                ("q-order-good", "live", "callback")):
            Database.execute(
                "INSERT INTO orders (id,account_id,broker,environment,source,symbol,name,side,quantity,created_at) "
                "VALUES (%s,'LOCAL-LIVE','qmt',%s,%s,'000001','test','buy',100,'2026-10-27 10:00:00')",
                (record_id, environment, source),
            )
            Database.execute(
                "INSERT INTO trades (id,order_id,account_id,broker,environment,source,symbol,name,side,"
                "price,quantity,amount,trade_time) VALUES (%s,%s,'LOCAL-LIVE','qmt',%s,%s,"
                "'000001','test','buy',10,100,1000,'2026-10-27 10:00:00')",
                (record_id, record_id, environment, source),
            )
        quarantine_first = LocalLedgerRepo.get_quarantine(1, 0)
        quarantine_second = LocalLedgerRepo.get_quarantine(1, 1)
        assert quarantine_first["counts"]["orders"] == 2
        assert quarantine_first["counts"]["trades"] == 2
        assert [quarantine_first["orders"][0]["id"], quarantine_second["orders"][0]["id"]] == [
            "q-order-b", "q-order-a",
        ]
        assert [quarantine_first["trades"][0]["id"], quarantine_second["trades"][0]["id"]] == [
            "q-order-b", "q-order-a",
        ]
        assert Database.fetchone("SELECT account_id FROM orders WHERE id='q-order-a'")["account_id"] == "LOCAL-LIVE"
        Database.execute("DELETE FROM local_order_intents WHERE intent_id=%s",
                         ("00000000-0000-4000-8000-000000000001",))
        legacy_trade = {"id": "T-LEGACY", "order_id": "O-LEGACY", "symbol": "000001.SZ",
                        "side": "buy", "quantity": 100, "price": "10.000", "amount": "1000.00",
                        "commission": "0.30", "stamp_duty": "0.00"}
        BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", legacy_trade, "query")
        Database.execute("UPDATE broker_trade_reports SET observed_at='2000-01-01 00:00:00' "
                         "WHERE broker='qmt' AND broker_account_id='QMT-TEST' AND trade_id='T-LEGACY'")
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["unassigned"] == 0
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["legacy_quarantined"] == 1
        opening_fingerprint = LocalLedgerRepo.get_opening_snapshot()["fingerprint"]
        assert LocalLedgerRepo.activate("2000.00", opening_fingerprint, "admin", "核对隔离账户期初余额")["version"] == 1
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["unassigned"] == 0
        assert BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")[0]["issue_type"] == "legacy_quarantined"
        with pytest.raises(pymysql.err.Error):
            Database.execute("UPDATE local_ledger_events SET cash_after=0 WHERE event_key=%s",
                             ("opening:LOCAL-LIVE",))
        with pytest.raises(pymysql.err.Error):
            Database.execute("DELETE FROM local_ledger_events WHERE event_key=%s",
                             ("opening:LOCAL-LIVE",))
        assert Database.fetchone("SELECT cash_after FROM local_ledger_events WHERE event_key=%s",
                                 ("opening:LOCAL-LIVE",))["cash_after"] == 2000
        failure_batch_id = uuid.uuid4().hex
        failure_signal_id = Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,'000001','rollback test','buy','买入',100,10,'LOCAL-LIVE',1,%s)",
            (failure_batch_id, _live_plan_fingerprint()),
        )
        before_failed_fill_account = LocalLedgerRepo.get_account()
        before_failed_fill_events = Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE account_id='LOCAL-LIVE'"
        )["n"]
        Database.execute(
            "CREATE TRIGGER fail_manual_fill_event_insert BEFORE INSERT ON local_ledger_events "
            "FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected manual fill event failure'"
        )
        try:
            with pytest.raises(pymysql.err.Error, match="injected manual fill event failure"):
                RebalanceSignalRepo.record_execution(
                    failure_signal_id,
                    {"request_id": str(uuid.uuid4()), "shares": 100, "price": "10.000",
                     "commission": "0.30", "stamp_duty": "0.00", "expected_account_version": 1,
                     "actor": "isolated-admin"},
                    decision_fingerprint,
                )
        finally:
            Database.execute("DROP TRIGGER IF EXISTS fail_manual_fill_event_insert")
        assert LocalLedgerRepo.get_account() == before_failed_fill_account
        assert Database.fetchone(
            "SELECT shares FROM holdings WHERE account_id='LOCAL-LIVE' AND code='000001'"
        ) is None
        assert Database.fetchone(
            "SELECT filled_shares,execution_status FROM rebalance_signals WHERE id=%s",
            (failure_signal_id,),
        ) == {"filled_shares": 0, "execution_status": "pending"}
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE account_id='LOCAL-LIVE'"
        )["n"] == before_failed_fill_events
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        assert LocalOrderIntentRepo.bind_broker(
            "QMT-TEST", 1, "admin", _ready_reconciliation("QMT-TEST", 1),
        )["account_version"] == 2
        Database.execute("UPDATE local_ledger_accounts SET broker_account_id='QMT-OTHER' "
                         "WHERE account_id='LOCAL-LIVE'")
        assert "券商资金账号与绑定事件不一致" in LocalLedgerRepo.audit_integrity()["issues"]
        Database.execute("UPDATE local_ledger_accounts SET broker_account_id='QMT-TEST' "
                         "WHERE account_id='LOCAL-LIVE'")
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        fingerprint = _live_plan_fingerprint()
        with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"):
            RebalanceSessionRepo.create(
                "SESSION-LOCAL-1", "rebalance", datetime.date.today().isoformat(),
                {"cash": 2000}, [], [], account_id="LOCAL-LIVE", account_version=2,
            )
        saved_session = RebalanceSessionRepo.get_by_batch("SESSION-LOCAL-1", account_id="LOCAL-LIVE")
        assert saved_session["account_id"] == "LOCAL-LIVE"
        assert saved_session["account_version"] == 2
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("B1", "000001", "test", "buy", "买入", 100, 10, "LOCAL-LIVE", 2, fingerprint),
        )
        signal_id = Database.fetchone("SELECT id FROM rebalance_signals WHERE batch_id=%s", ("B1",))["id"]
        intent = {"intent_id": str(uuid.uuid4()), "signal_id": signal_id,
                  "broker_account_id": "QMT-TEST", "expected_account_version": 2,
                  "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin",
                  "reconciliation": _ready_reconciliation("QMT-TEST", 2)}
        assert LocalOrderIntentRepo.prepare(intent, decision_fingerprint)["reserved_cash"] == 1001
        assert LocalOrderIntentRepo.prepare(intent, decision_fingerprint)["already_prepared"] is True
        assert LocalLedgerRepo.get_account()["available"] == 999
        Database.execute(
            "INSERT INTO broker_order_states (broker,broker_account_id,order_id,symbol,side,"
            "quantity,price_type,limit_price,filled_qty,status) "
            "VALUES ('qmt','QMT-TEST','WRONG-SUFFIX','000001.SH','buy',100,'limit',10,0,'pending')"
        )
        with pytest.raises(ValueError, match="指数身份不能用作股票代码"):
            LocalOrderIntentRepo.attach_broker_order(intent["intent_id"], "WRONG-SUFFIX", "admin")
        Database.execute("DELETE FROM broker_order_states WHERE order_id='WRONG-SUFFIX'")
        Database.execute("UPDATE local_ledger_accounts SET frozen=frozen+1 WHERE account_id='LOCAL-LIVE'")
        Database.execute("UPDATE local_order_intents SET remaining_reserved_cash=remaining_reserved_cash+1 "
                         "WHERE intent_id=%s", (intent["intent_id"],))
        assert "委托意图冻结金额或成交数量与事件回放不一致" in LocalLedgerRepo.audit_integrity()["issues"]
        Database.execute("UPDATE local_order_intents SET remaining_reserved_cash=remaining_reserved_cash-1 "
                         "WHERE intent_id=%s", (intent["intent_id"],))
        Database.execute("UPDATE local_ledger_accounts SET frozen=frozen-1 WHERE account_id='LOCAL-LIVE'")
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        assert LocalOrderIntentRepo.cancel_prepared(intent["intent_id"], "admin")["account_version"] == 4
        assert LocalLedgerRepo.get_account()["available"] == 2000
        Database.execute("UPDATE rebalance_signals SET execution_status='cancelled' WHERE id=%s", (signal_id,))
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("B2", "000001", "test", "buy", "买入", 100, 10, "LOCAL-LIVE", 4, fingerprint),
        )
        fresh_signal_id = Database.fetchone("SELECT id FROM rebalance_signals WHERE batch_id=%s", ("B2",))["id"]
        fill = {"request_id": str(uuid.uuid4()), "shares": 100, "price": 9.9,
                "commission": "0.30", "stamp_duty": "0", "expected_account_version": 4,
                "actor": "admin", "broker_trade_id": "MANUAL-T1"}
        before_bound_manual_fill = LocalLedgerRepo.get_account()
        before_bound_manual_events = Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE account_id='LOCAL-LIVE'"
        )["n"]
        with pytest.raises(ValueError, match="必须通过本地委托意图关联券商订单"):
            RebalanceSignalRepo.record_execution(
                fresh_signal_id, fill, decision_fingerprint,
            )
        assert LocalLedgerRepo.get_account() == before_bound_manual_fill
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE account_id='LOCAL-LIVE'"
        )["n"] == before_bound_manual_events
        assert Database.fetchone("SELECT shares FROM holdings WHERE code=%s", ("000001",)) is None
        base = {"id": "123", "symbol": "000001.SZ", "side": "buy",
                "quantity": 100, "filled_qty": 0, "status": "pending",
                "price_type": "limit", "price": 10, "filled_price": 0}
        pending = BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", base, "callback", 48)
        partial_order = {**base, "filled_qty": 40, "status": "partial", "filled_price": 9.9}
        partial = BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", partial_order, "query", 55)
        retry = BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", partial_order, "callback", 55)
        stale = BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", base, "query", 50)
        cancelled_order = {**partial_order, "status": "cancelled"}
        cancelled = BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", cancelled_order, "query", 54)
        contradictory = BrokerOrderReportRepo.ingest(
            "qmt", "QMT-TEST", {**base, "filled_qty": 100, "status": "filled", "filled_price": 9.9},
            "callback", 56,
        )
        state = Database.fetchone(
            "SELECT status,filled_qty FROM broker_order_states WHERE broker=%s AND broker_account_id=%s AND order_id=%s",
            ("qmt", "QMT-TEST", "123"),
        )
        assert [pending["disposition"], partial["disposition"], stale["disposition"],
                cancelled["disposition"], contradictory["disposition"]] == [
            "applied", "applied", "stale", "applied", "conflict",
        ]
        assert retry["already_recorded"] is True
        assert state == {"status": "cancelled", "filled_qty": 40}
        easytrader = EasytraderBroker(broker_account_id="ACCT-TEST")
        easytrader._connected = True
        easytrader._trader = Mock()
        easytrader._trader.balance = [{
            "资金帐号": "ACCT-TEST", "可用资金": 1000, "参考市值": 0, "总资产": 1000,
        }]
        easy_order_raw = {
            "委托编号": "ET-O-1", "证券代码": "000002", "操作": "买入", "委托数量": 100,
            "成交数量": 0, "委托价格": 12, "成交价格": 0, "状态": "已报",
        }
        easytrader._trader.today_entrusts = [easy_order_raw]
        assert easytrader.get_orders()[0]["status"] == "pending"
        assert easytrader.get_orders()[0]["report_disposition"] == "applied"
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_order_reports WHERE broker=%s "
            "AND broker_account_id=%s AND order_id=%s",
            ("easytrader", "EASYT-ACCT-TEST", "ET-O-1"),
        )["n"] == 1
        easy_order_state = Database.fetchone(
            "SELECT status,filled_qty FROM broker_order_states WHERE broker=%s "
            "AND broker_account_id=%s AND order_id=%s",
            ("easytrader", "EASYT-ACCT-TEST", "ET-O-1"),
        )
        assert easy_order_state == {"status": "pending", "filled_qty": 0}
        easy_order_report = Database.fetchone(
            "SELECT source,raw_json FROM broker_order_reports WHERE broker=%s "
            "AND broker_account_id=%s AND order_id=%s",
            ("easytrader", "EASYT-ACCT-TEST", "ET-O-1"),
        )
        assert easy_order_report["source"] == "query"
        assert "委托编号" in easy_order_report["raw_json"]
        easy_trade_raw = {
            "成交编号": "ET-T-1", "委托编号": "ET-O-1", "证券代码": "000002", "操作": "买入",
            "成交数量": 100, "成交价格": 12, "成交金额": 1200, "成交时间": "10:01:00",
        }
        easytrader._trader.today_trades = [easy_trade_raw]
        easytrader.get_trades()
        easy_trade_report = Database.fetchone(
            "SELECT first_source,commission,stamp_duty,raw_json FROM broker_trade_reports "
            "WHERE broker=%s AND broker_account_id=%s AND trade_id=%s",
            ("easytrader", "EASYT-ACCT-TEST", "ET-T-1"),
        )
        assert easy_trade_report["first_source"] == "query"
        assert easy_trade_report["commission"] is None and easy_trade_report["stamp_duty"] is None
        assert "成交编号" in easy_trade_report["raw_json"]
        easytrader._trader.position = []
        easy_snapshot = easytrader.get_reconciliation_snapshot()
        assert easy_snapshot["identity_verified"] is True
        assert easy_snapshot["trade_report_health"]["unassigned"] >= 1
        assert easy_snapshot["orders"] == [{"id": "ET-O-1", "status": "pending"}]
        assert BrokerOrderReportRepo.unresolved_count("qmt", "QMT-TEST") == 1
        with pytest.raises(ValueError, match="已变化"):
            BrokerOrderReportRepo.resolve(
                contradictory["report_key"], "qmt", "QMT-TEST",
                {**base, "status": "filled", "filled_qty": 100},
                "admin", "核对券商终态与成交数量",
            )
        with pytest.raises(ValueError, match="身份已变化"):
            BrokerOrderReportRepo.resolve(
                contradictory["report_key"], "qmt", "QMT-TEST",
                {**cancelled_order, "symbol": "000002.SZ"},
                "admin", "已核对券商终态但证券代码矛盾",
            )
        resolution = BrokerOrderReportRepo.resolve(
            contradictory["report_key"], "qmt", "QMT-TEST", cancelled_order,
            "admin", "已核对券商终态及四十股成交",
        )
        assert resolution["already_resolved"] is False
        assert BrokerOrderReportRepo.unresolved_count("qmt", "QMT-TEST") == 0
        assert BrokerOrderReportRepo.resolve(
            contradictory["report_key"], "qmt", "QMT-TEST", cancelled_order,
            "admin", "已核对券商终态及四十股成交",
        )["already_resolved"] is True
        order_history = Database.fetchall(
            "SELECT id,action,actor,reason,evidence_json FROM broker_report_resolution_events "
            "WHERE report_type='order' AND report_key=%s ORDER BY id",
            (contradictory["report_key"],),
        )
        assert [row["action"] for row in order_history] == ["resolved"]
        replayed = BrokerOrderReportRepo.ingest(
            "qmt", "QMT-TEST", {**base, "filled_qty": 100, "status": "filled", "filled_price": 9.9},
            "query", 56,
        )
        assert replayed["report_key"] == contradictory["report_key"]
        assert replayed["already_recorded"] is True
        assert BrokerOrderReportRepo.unresolved_count("qmt", "QMT-TEST") == 1
        assert BrokerOrderReportRepo.resolve(
            contradictory["report_key"], "qmt", "QMT-TEST", cancelled_order,
            "admin", "再次核对券商终态及四十股成交",
        )["already_resolved"] is False
        assert BrokerOrderReportRepo.unresolved_count("qmt", "QMT-TEST") == 0
        order_history = Database.fetchall(
            "SELECT id,action,actor,reason,evidence_json FROM broker_report_resolution_events "
            "WHERE report_type='order' AND report_key=%s ORDER BY id",
            (contradictory["report_key"],),
        )
        assert [row["action"] for row in order_history] == ["resolved", "reopened", "resolved"]
        assert order_history[0]["reason"] == "已核对券商终态及四十股成交"
        with pytest.raises(pymysql.err.Error):
            Database.execute("UPDATE broker_report_resolution_events SET reason='tampered' WHERE id=%s",
                             (order_history[0]["id"],))
        with pytest.raises(pymysql.err.Error):
            Database.execute("DELETE FROM broker_report_resolution_events WHERE id=%s",
                             (order_history[0]["id"],))
        malformed_orders = [
            {**base, "id": "130", "quantity": 100.5},
            {**base, "id": "131", "filled_qty": 40.5, "status": "partial"},
            {**base, "id": "132", "quantity": True},
            {**base, "id": "133", "filled_qty": True, "status": "partial"},
            {**base, "id": "134", "quantity": 2147483648},
        ]
        malformed_results = [BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", item, "query")
                             for item in malformed_orders]
        assert [item["disposition"] for item in malformed_results] == ["conflict"] * len(malformed_orders)
        malformed_retry = BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", malformed_orders[0], "callback")
        assert malformed_retry["already_recorded"] is True
        assert malformed_retry["status"] == "unknown"
        malformed_states = Database.fetchall(
            "SELECT order_id,status,filled_qty FROM broker_order_states WHERE broker=%s "
            "AND broker_account_id=%s AND order_id IN ('130','131','132','133','134') ORDER BY order_id",
            ("qmt", "QMT-TEST"),
        )
        assert malformed_states == [
            {"order_id": item, "status": "unknown", "filled_qty": 0}
            for item in ("131", "133")
        ]
        corrected_order = {**base, "id": "130", "status": "cancelled"}
        assert BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", corrected_order, "query")[
            "disposition"] == "applied"
        BrokerOrderReportRepo.resolve(
            malformed_results[0]["report_key"], "qmt", "QMT-TEST", corrected_order,
            "admin", "复核异常股数并确认当前券商委托",
        )
        assert BrokerOrderReportRepo.unresolved_count("qmt", "QMT-TEST") == len(malformed_orders) - 1
        trade = {"id": "T-1", "order_id": "123", "symbol": "000001.SZ", "side": "buy",
                 "quantity": 40, "price": "9.900", "amount": "396.00",
                 "commission": None, "stamp_duty": None}
        with pytest.raises(ValueError, match="成交金额与价格数量不一致"):
            BrokerTradeReportRepo.ingest(
                "qmt", "QMT-TEST", {**trade, "id": "T-MISMATCH", "amount": "401.00"}, "query",
            )
        assert Database.fetchone(
            "SELECT 1 FROM broker_trade_reports WHERE broker='qmt' "
            "AND broker_account_id='QMT-TEST' AND trade_id='T-MISMATCH'"
        ) is None
        Database.execute(
            "INSERT INTO broker_trade_reports (broker,broker_account_id,trade_id,order_id,symbol,side,"
            "quantity,price,amount,commission,stamp_duty,payload_hash,raw_json,first_source) "
            "VALUES ('qmt','QMT-LEGACY-TEST','T-LEGACY-MISMATCH','O-LEGACY-MISMATCH','000001.SZ','buy',"
            "40,9.900,401.00,NULL,NULL,%s,'{}','legacy')", ("f" * 64,),
        )
        Database.execute(
            "INSERT INTO broker_trade_report_conflicts (conflict_key,broker,broker_account_id,trade_id,"
            "payload_hash,raw_json) VALUES (%s,'qmt','QMT-LEGACY-TEST','T-LEGACY-MISMATCH',%s,'{}')",
            ("a" * 64, "f" * 64),
        )
        with pytest.raises(ValueError, match="基准回报不一致"):
            BrokerTradeReportRepo.resolve_conflicts(
                "qmt", "QMT-LEGACY-TEST", "T-LEGACY-MISMATCH",
                {"id": "T-LEGACY-MISMATCH", "order_id": "O-LEGACY-MISMATCH",
                 "symbol": "000001.SZ", "side": "buy", "quantity": 40,
                 "price": "9.900", "amount": "401.00",
                 "commission": None, "stamp_duty": None},
                "admin", "验证历史金额错配不能被人工解除",
            )
        assert BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", trade, "callback")["status"] == "recorded"
        assert BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", trade, "query")["status"] == "duplicate"
        enriched_trade = {**trade, "commission": "0.30"}
        assert BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", enriched_trade, "query")["status"] == "conflict"
        with pytest.raises(ValueError, match="基准回报不一致"):
            BrokerTradeReportRepo.resolve_conflicts(
                "qmt", "QMT-TEST", "T-1", {**trade, "commission": "0.301"},
                "admin", "拒绝无效精度的券商佣金证据",
            )
        assert BrokerTradeReportRepo.resolve_conflicts(
            "qmt", "QMT-TEST", "T-1", enriched_trade,
            "admin", "已核对券商逐笔成交佣金补全",
        )["resolved_count"] == 1
        assert BrokerTradeReportRepo.ingest(
            "qmt", "QMT-TEST", enriched_trade, "query",
        )["status"] == "duplicate"
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["conflicts"] == 0
        stored_trade = Database.fetchone(
            "SELECT commission FROM broker_trade_reports WHERE broker='qmt' "
            "AND broker_account_id='QMT-TEST' AND trade_id='T-1'"
        )
        assert stored_trade["commission"] is None
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["conflicts"] == 0
        assert BrokerTradeReportRepo.ingest(
            "qmt", "QMT-TEST", {**trade, "price": "9.800", "amount": "392.00"}, "query"
        )["status"] == "conflict"
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST") == {
            "conflicts": 1, "unpriced": 0, "unbooked": 0,
            "unassigned": 0, "legacy_quarantined": 1,
        }
        assert BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")[0]["issue_type"] == "conflict"
        with pytest.raises(ValueError, match="基准回报不一致"):
            BrokerTradeReportRepo.resolve_conflicts(
                "qmt", "QMT-TEST", "T-1", {**trade, "price": "9.800", "amount": "392.00"},
                "admin", "已核对券商当前逐笔成交记录",
            )
        with pytest.raises(ValueError, match="基准回报不一致"):
            BrokerTradeReportRepo.resolve_conflicts(
                "qmt", "QMT-TEST", "T-1", {**trade, "quantity": 40.5},
                "admin", "拒绝非整股数的券商成交证据",
            )
        assert BrokerTradeReportRepo.resolve_conflicts(
            "qmt", "QMT-TEST", "T-1", trade,
            "admin", "已核对券商当前逐笔成交记录",
        )["resolved_count"] == 1
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["conflicts"] == 0
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["unassigned"] == 1
        BrokerTradeReportRepo.ingest(
            "qmt", "QMT-TEST", {**trade, "price": "9.800", "amount": "392.00"}, "query",
        )
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["conflicts"] == 1
        BrokerTradeReportRepo.resolve_conflicts(
            "qmt", "QMT-TEST", "T-1", trade,
            "admin", "再次核对券商当前逐笔成交记录",
        )
        trade_history = Database.fetchall(
            "SELECT action,reason,evidence_json FROM broker_report_resolution_events WHERE report_type='trade' "
            "AND broker_account_id='QMT-TEST' ORDER BY id"
        )
        assert [row["action"] for row in trade_history] == [
            "resolved", "resolved", "reopened", "resolved",
        ]
        assert trade_history[0]["reason"] == "已核对券商逐笔成交佣金补全"
        assert '"commission": "0.30"' in trade_history[0]["evidence_json"]
        ensure_local_ledger_schema()
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_report_resolution_events "
            "WHERE broker_account_id='QMT-TEST'"
        )["n"] == 8
        Database.execute(
            "CREATE TRIGGER reject_isolated_report BEFORE INSERT ON broker_order_reports "
            "FOR EACH ROW BEGIN IF NEW.order_id='999' THEN "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='isolated failure'; "
            "END IF; END"
        )
        with pytest.raises(pymysql.err.Error):
            BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", {**base, "id": "999"}, "query", 48)
        assert Database.fetchone(
            "SELECT order_id FROM broker_order_states WHERE broker=%s AND broker_account_id=%s AND order_id=%s",
            ("qmt", "QMT-TEST", "999"),
        ) is None
        current = LocalLedgerRepo.get_account()
        current_fingerprint = _live_plan_fingerprint()
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("B3", "000002", "test2", "buy", "买入", 200, 4,
             "LOCAL-LIVE", current["version"], current_fingerprint),
        )
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("B3", "000003", "test sibling", "buy", "买入", 100, 4,
             "LOCAL-LIVE", current["version"], current_fingerprint),
        )
        broker_signal = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='B3' AND code='000002'"
        )["id"]
        broker_sibling_signal = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='B3' AND code='000003'"
        )["id"]
        live_intent = {"intent_id": str(uuid.uuid4()), "signal_id": broker_signal,
                       "broker_account_id": "QMT-TEST", "expected_account_version": current["version"],
                       "quantity": 200, "limit_price": 4, "fee_cap": 1, "actor": "admin",
                       "reconciliation": _ready_reconciliation("QMT-TEST", current["version"])}
        assert LocalOrderIntentRepo.prepare(live_intent, decision_fingerprint)["reserved_cash"] == 801
        broker_order = {"id": "124", "symbol": "000002.SZ", "side": "buy", "quantity": 200,
                        "filled_qty": 0, "status": "pending", "price_type": "limit",
                        "price": 4, "filled_price": 0}
        assert BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", broker_order, "callback")["disposition"] == "applied"
        Database.execute("UPDATE rebalance_signals SET account_id='OTHER' WHERE id=%s", (broker_signal,))
        with pytest.raises(ValueError, match="账户版本或剩余计划量"):
            LocalOrderIntentRepo.attach_broker_order(live_intent["intent_id"], "124", "admin")
        assert Database.fetchone(
            "SELECT state,broker_order_id FROM local_order_intents WHERE intent_id=%s",
            (live_intent["intent_id"],),
        ) == {"state": "prepared", "broker_order_id": None}
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='broker_attach'",
        )["n"] == 0
        Database.execute("UPDATE rebalance_signals SET account_id='LOCAL-LIVE' WHERE id=%s", (broker_signal,))
        assert LocalOrderIntentRepo.attach_broker_order(
            live_intent["intent_id"], "124", "admin",
        )["state"] == "submitted"
        assert LocalOrderIntentRepo.attach_broker_order(
            live_intent["intent_id"], "124", "admin",
        )["already_attached"] is True
        with pytest.raises(ValueError, match="另一券商委托"):
            LocalOrderIntentRepo.attach_broker_order(live_intent["intent_id"], "123", "admin")
        BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", {**broker_order, "filled_qty": 100,
                                    "status": "partial", "filled_price": 4}, "query")
        broker_trade = {"id": "T-2", "order_id": "124", "symbol": "000002.SZ",
                        "side": "buy", "quantity": 100, "price": "4.000", "amount": "400.00",
                        "commission": None, "stamp_duty": None}
        BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", broker_trade, "query")
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["unpriced"] == 1
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["unassigned"] == 1
        unresolved_trades = BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")
        assert {row["trade_id"]: row["issue_type"] for row in unresolved_trades} == {
            "T-1": "unassigned", "T-2": "fees_missing", "T-LEGACY": "legacy_quarantined",
        }
        BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", {**broker_order, "filled_qty": 100,
                                    "status": "cancelled", "filled_price": 4}, "query")
        with pytest.raises(ValueError, match="尚未全部入账"):
            LocalOrderIntentRepo.finalize_terminal(live_intent["intent_id"], "admin")
        with pytest.raises(ValueError, match="费用不能超过成交金额"):
            LocalOrderIntentRepo.record_broker_fill(
                live_intent["intent_id"], "T-2", "900.00", "0", "admin", decision_fingerprint,
            )
        assert Database.fetchone("SELECT shares FROM holdings WHERE code='000002'") is None
        before_failed_fill = {
            "account": LocalLedgerRepo.get_account(),
            "intent": Database.fetchone(
                "SELECT filled_quantity,remaining_reserved_cash,state,account_version "
                "FROM local_order_intents WHERE intent_id=%s", (live_intent["intent_id"],),
            ),
            "signal": Database.fetchone(
                "SELECT filled_shares,execution_status,account_version FROM rebalance_signals WHERE id=%s",
                (broker_signal,),
            ),
            "sibling": Database.fetchone(
                "SELECT execution_status FROM rebalance_signals WHERE id=%s", (broker_sibling_signal,),
            ),
            "fill_events": Database.fetchone(
                "SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='broker_fill'",
            )["n"],
        }
        for failure_table, failure_operation in (
            ("holdings", "INSERT"),
            ("local_ledger_accounts", "UPDATE"),
            ("local_order_intents", "UPDATE"),
            ("rebalance_signals", "UPDATE"),
            ("local_ledger_events", "INSERT"),
        ):
            Database.execute(
                f"CREATE TRIGGER reject_isolated_fill BEFORE {failure_operation} ON {failure_table} "
                "FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' "
                "SET MESSAGE_TEXT='isolated fill failure'; END"
            )
            with pytest.raises(pymysql.err.Error):
                LocalOrderIntentRepo.record_broker_fill(
                    live_intent["intent_id"], "T-2", "0.30", "0", "admin", decision_fingerprint,
                )
            pool_state = Database.get_pool_stats()
            assert pool_state["borrowed"] == 0
            assert pool_state["created"] == pool_state["idle"]
            assert LocalLedgerRepo.get_account() == before_failed_fill["account"]
            assert Database.fetchone(
                "SELECT filled_quantity,remaining_reserved_cash,state,account_version "
                "FROM local_order_intents WHERE intent_id=%s", (live_intent["intent_id"],),
            ) == before_failed_fill["intent"]
            assert Database.fetchone(
                "SELECT filled_shares,execution_status,account_version FROM rebalance_signals WHERE id=%s",
                (broker_signal,),
            ) == before_failed_fill["signal"]
            assert Database.fetchone(
                "SELECT execution_status FROM rebalance_signals WHERE id=%s", (broker_sibling_signal,),
            ) == before_failed_fill["sibling"]
            assert Database.fetchone("SELECT shares FROM holdings WHERE code='000002'") is None
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='broker_fill'",
            )["n"] == before_failed_fill["fill_events"]
            Database.execute("DROP TRIGGER reject_isolated_fill")
        with ThreadPoolExecutor(max_workers=2) as executor:
            booked = list(executor.map(
                lambda _: LocalOrderIntentRepo.record_broker_fill(
                    live_intent["intent_id"], "T-2", "0.30", "0", "admin", decision_fingerprint,
                ), range(2),
            ))
        assert sorted(row["already_recorded"] for row in booked) == [False, True]
        assert LocalOrderIntentRepo.record_broker_fill(
            live_intent["intent_id"], "t-2", "0.30", "0", "admin", decision_fingerprint,
        )["already_recorded"] is True
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["unpriced"] == 0
        unresolved_trades = BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")
        assert [(row["trade_id"], row["issue_type"]) for row in unresolved_trades] == [
            ("T-1", "unassigned"), ("T-LEGACY", "legacy_quarantined"),
        ]
        complete_but_unbooked = {"id": "T-UNBOOKED", "order_id": "123", "symbol": "000001.SZ",
                                 "side": "sell", "quantity": 40, "price": "9.900",
                                 "amount": "396.00", "commission": "0.30", "stamp_duty": "0.20"}
        BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", complete_but_unbooked, "query")
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["unassigned"] == 2
        unresolved_trades = BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")
        assert {row["trade_id"]: row["issue_type"] for row in unresolved_trades} == {
            "T-1": "unassigned", "T-UNBOOKED": "unassigned",
            "T-LEGACY": "legacy_quarantined",
        }
        assert LocalLedgerRepo.get_account()["cash"] == Decimal("1599.70")
        assert LocalLedgerRepo.get_account()["frozen"] == Decimal("400.70")
        assert Database.fetchone("SELECT shares FROM holdings WHERE code='000002'")["shares"] == 100
        assert Database.fetchone("SELECT filled_shares,execution_status FROM rebalance_signals WHERE id=%s",
                                 (broker_signal,)) == {"filled_shares": 100, "execution_status": "partial"}
        assert Database.fetchone(
            "SELECT execution_status,unfilled_reason FROM rebalance_signals WHERE id=%s",
            (broker_sibling_signal,),
        ) == {
            "execution_status": "cancelled",
            "unfilled_reason": "账户资金或持仓已变化，旧调仓计划已失效，请重新生成",
        }
        evidence = LocalOrderIntentRepo.get_broker_evidence(live_intent["intent_id"])
        assert evidence["order"]["order_id"] == "124"
        assert [(row["trade_id"], row["booked"]) for row in evidence["trades"]] == [("T-2", True)]
        assert evidence["conflicts"] == 0
        BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", {**broker_order, "filled_qty": 100,
                                    "status": "cancelled", "filled_price": 4}, "query")
        Database.execute("UPDATE rebalance_signals SET account_id='OTHER' WHERE id=%s", (broker_signal,))
        with pytest.raises(ValueError, match="账户版本或剩余计划量"):
            LocalOrderIntentRepo.finalize_terminal(live_intent["intent_id"], "admin")
        assert Database.fetchone("SELECT frozen FROM local_ledger_accounts WHERE account_id='LOCAL-LIVE'")[
            "frozen"] == Decimal("400.70")
        assert Database.fetchone("SELECT state FROM local_order_intents WHERE intent_id=%s",
                                 (live_intent["intent_id"],))["state"] == "partial"
        Database.execute("UPDATE rebalance_signals SET account_id='LOCAL-LIVE' WHERE id=%s", (broker_signal,))
        terminal = LocalOrderIntentRepo.finalize_terminal(live_intent["intent_id"], "admin")
        assert terminal["released_cash"] == Decimal("400.70")
        signal_after_terminal = Database.fetchone(
            "SELECT account_version,execution_status FROM rebalance_signals WHERE id=%s", (broker_signal,),
        )
        assert signal_after_terminal == {
            "account_version": LocalLedgerRepo.get_account()["version"], "execution_status": "partial",
        }
        assert LocalOrderIntentRepo.finalize_terminal(live_intent["intent_id"], "admin")["already_finalized"] is True
        assert LocalLedgerRepo.get_account()["frozen"] == 0
        assert Database.fetchone("SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='broker_fill'")["n"] == 1
        commit_account = LocalLedgerRepo.get_account()
        commit_request_id = str(uuid.uuid4())
        commit_fingerprint = "a" * 64
        Database.transaction(lambda cur: RebalanceSessionRepo.create(
            batch_id=uuid.UUID(commit_request_id).hex, check_type="rebalance",
            eval_date=datetime.date.today().isoformat(), screening_save_id=1,
            account_json={"account_version": commit_account["version"]}, holdings_json=[],
            signals_json=[], constraints_json=[], account_id="LOCAL-LIVE",
            account_version=commit_account["version"], commit_request_id=commit_request_id,
            plan_fingerprint=commit_fingerprint, cursor=cur,
        ))
        committed_session = RebalanceSessionRepo.get_by_commit_request(commit_request_id, "LOCAL-LIVE")
        assert committed_session["batch_id"] == uuid.UUID(commit_request_id).hex
        assert committed_session["plan_fingerprint"] == commit_fingerprint
        with pytest.raises(pymysql.err.IntegrityError):
            Database.transaction(lambda cur: RebalanceSessionRepo.create(
                batch_id=uuid.uuid4().hex, check_type="rebalance",
                eval_date=datetime.date.today().isoformat(), screening_save_id=1,
                account_json={}, holdings_json=[], signals_json=[], account_id="LOCAL-LIVE",
                account_version=commit_account["version"], commit_request_id=commit_request_id,
                plan_fingerprint=commit_fingerprint, cursor=cur,
            ))
        race_request_id = str(uuid.uuid4())
        race_fingerprint = "b" * 64
        race_barrier = threading.Barrier(2)

        def race_session_insert():
            race_barrier.wait(timeout=10)
            try:
                return Database.transaction(lambda cur: RebalanceSessionRepo.create(
                    batch_id=uuid.UUID(race_request_id).hex, check_type="rebalance",
                    eval_date=datetime.date.today().isoformat(), screening_save_id=1,
                    account_json={}, holdings_json=[], signals_json=[], account_id="LOCAL-LIVE",
                    account_version=commit_account["version"], commit_request_id=race_request_id,
                    plan_fingerprint=race_fingerprint, cursor=cur,
                ))
            except Exception as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            race_results = list(executor.map(lambda _: race_session_insert(), range(2)))
        assert sum(not isinstance(result, Exception) for result in race_results) == 1
        assert sum(isinstance(result, pymysql.err.IntegrityError) for result in race_results) == 1
        raced_session = RebalanceSessionRepo.get_by_commit_request(race_request_id, "LOCAL-LIVE")
        assert raced_session["plan_fingerprint"] == race_fingerprint
        after_terminal = LocalLedgerRepo.get_account()
        retry_intent = {"intent_id": str(uuid.uuid4()), "signal_id": broker_signal,
                        "broker_account_id": "QMT-TEST", "expected_account_version": after_terminal["version"],
                        "quantity": 100, "limit_price": 4, "fee_cap": 1, "actor": "admin",
                        "reconciliation": _ready_reconciliation("QMT-TEST", after_terminal["version"])}
        assert LocalOrderIntentRepo.prepare(retry_intent, decision_fingerprint)["reserved_cash"] == 401
        Database.execute("UPDATE rebalance_signals SET account_id='OTHER' WHERE id=%s", (broker_signal,))
        with pytest.raises(ValueError, match="账户版本或剩余计划量"):
            LocalOrderIntentRepo.cancel_prepared(retry_intent["intent_id"], "admin")
        assert Database.fetchone("SELECT state FROM local_order_intents WHERE intent_id=%s",
                                 (retry_intent["intent_id"],))["state"] == "prepared"
        assert Database.fetchone("SELECT frozen FROM local_ledger_accounts WHERE account_id='LOCAL-LIVE'")[
            "frozen"] == Decimal("401.00")
        Database.execute("UPDATE rebalance_signals SET account_id='LOCAL-LIVE' WHERE id=%s", (broker_signal,))
        retry_cancel = LocalOrderIntentRepo.cancel_prepared(retry_intent["intent_id"], "admin")
        assert retry_cancel["state"] == "cancelled"
        signal_after_cancel = Database.fetchone(
            "SELECT account_version,execution_status FROM rebalance_signals WHERE id=%s", (broker_signal,),
        )
        assert signal_after_cancel == {
            "account_version": LocalLedgerRepo.get_account()["version"], "execution_status": "partial",
        }
        with pytest.raises(ValueError, match="必须通过本地委托意图"):
            RebalanceSignalRepo.record_execution(
                fresh_signal_id, {**fill, "request_id": str(uuid.uuid4()),
                                  "expected_account_version": LocalLedgerRepo.get_account()["version"],
                                  "broker_trade_id": "t-2"},
                decision_fingerprint,
            )
        sell_account = LocalLedgerRepo.get_account()
        sell_fingerprint = _live_plan_fingerprint()
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("B4", "000002", "test2", "sell", "卖出", 100, 4,
             "LOCAL-LIVE", sell_account["version"], sell_fingerprint),
        )
        sell_signal_id = Database.fetchone("SELECT id FROM rebalance_signals WHERE batch_id='B4'")["id"]
        sell_intent = {"intent_id": str(uuid.uuid4()), "signal_id": sell_signal_id,
                       "broker_account_id": "QMT-TEST", "expected_account_version": sell_account["version"],
                       "quantity": 100, "limit_price": 4, "fee_cap": 0, "actor": "admin",
                       "broker_sellable": 100,
                       "reconciliation": _ready_reconciliation("QMT-TEST", sell_account["version"])}
        with pytest.raises(ValueError, match="券商可卖股数"):
            LocalOrderIntentRepo.prepare({**sell_intent, "broker_sellable": 50}, decision_fingerprint)
        assert LocalOrderIntentRepo.prepare(sell_intent, decision_fingerprint)["reserved_cash"] == 0
        sell_order = {"id": "125", "symbol": "000002.SZ", "side": "sell", "quantity": 100,
                      "filled_qty": 100, "status": "filled", "price_type": "limit",
                      "price": 4, "filled_price": 4}
        BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", sell_order, "query")
        LocalOrderIntentRepo.attach_broker_order(sell_intent["intent_id"], "125", "admin")
        BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", {
            "id": "T-3", "order_id": "125", "symbol": "000002.SZ", "side": "sell",
            "quantity": 100, "price": "4.000", "amount": "400.00",
            "commission": None, "stamp_duty": None,
        }, "query")
        sell_fill = LocalOrderIntentRepo.record_broker_fill(
            sell_intent["intent_id"], "T-3", "0.30", "0.20", "admin", decision_fingerprint,
        )
        assert sell_fill["state"] == "filled"
        assert LocalLedgerRepo.get_account()["cash"] == Decimal("1999.20")
        assert Database.fetchone("SELECT shares FROM holdings WHERE code='000002'") is None
        assert Database.fetchone("SELECT execution_status FROM rebalance_signals WHERE id=%s",
                                 (sell_signal_id,))["execution_status"] == "executed"
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        assert "T-3" not in {
            row["trade_id"] for row in BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")
        }
        Database.execute(
            "UPDATE broker_trade_reports SET quantity=99,amount='396.00' "
            "WHERE broker='qmt' AND broker_account_id='QMT-TEST' AND trade_id='T-3'"
        )
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["conflicts"] == 1
        assert {row["trade_id"]: row["issue_type"] for row in
                BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")}[
                    "T-3"] == "broker_mismatch"
        Database.execute(
            "UPDATE broker_trade_reports SET quantity=100,amount='400.00' "
            "WHERE broker='qmt' AND broker_account_id='QMT-TEST' AND trade_id='T-3'"
        )
        assert BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")["conflicts"] == 0
        holding_for_audit = Database.fetchone(
            "SELECT code,shares FROM holdings WHERE account_id='LOCAL-LIVE' ORDER BY code LIMIT 1"
        )
        Database.execute("UPDATE holdings SET shares=shares+1 WHERE account_id='LOCAL-LIVE' AND code=%s",
                         (holding_for_audit["code"],))
        assert "本地持仓与成交事件累计不一致" in LocalLedgerRepo.audit_integrity()["issues"]
        Database.execute("UPDATE holdings SET shares=shares-1 WHERE account_id='LOCAL-LIVE' AND code=%s",
                         (holding_for_audit["code"],))
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        original_cost = Database.fetchone(
            "SELECT cost_price FROM holdings WHERE account_id='LOCAL-LIVE' AND code=%s",
            (holding_for_audit["code"],),
        )["cost_price"]
        Database.execute("UPDATE holdings SET cost_price=cost_price+1 WHERE account_id='LOCAL-LIVE' AND code=%s",
                         (holding_for_audit["code"],))
        assert "本地持仓成本与成交事件累计不一致" in LocalLedgerRepo.audit_integrity()["issues"]
        Database.execute("UPDATE holdings SET cost_price=%s WHERE account_id='LOCAL-LIVE' AND code=%s",
                         (original_cost, holding_for_audit["code"]))
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        manual_account = LocalLedgerRepo.get_account()
        manual_fingerprint = _live_plan_fingerprint()
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("B5", "000001", "test", "sell", "卖出", 100, 9.9,
             "LOCAL-LIVE", manual_account["version"], manual_fingerprint),
        )
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("B5", "000002", "test sibling", "buy", "买入", 100, 9.9,
             "LOCAL-LIVE", manual_account["version"], manual_fingerprint),
        )
        manual_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='B5' AND code='000001'"
        )["id"]
        sibling_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='B5' AND code='000002'"
        )["id"]
        manual_first = {"request_id": str(uuid.uuid4()), "shares": 40, "price": "9.900",
                        "commission": "0.30", "stamp_duty": "0.20",
                        "expected_account_version": manual_account["version"],
                        "actor": "admin", "broker_trade_id": "MANUAL-PART-1"}
        with pytest.raises(ValueError, match="必须通过本地委托意图"):
            RebalanceSignalRepo.record_execution(manual_signal_id, manual_first, decision_fingerprint)
        assert LocalLedgerRepo.get_account()["cash"] == manual_account["cash"]
        assert LocalLedgerRepo.get_account()["version"] == manual_account["version"]
        assert Database.fetchone("SELECT shares FROM holdings WHERE code='000001'") is None
        assert Database.fetchone("SELECT filled_shares,execution_status FROM rebalance_signals WHERE id=%s",
                                 (manual_signal_id,)) == {"filled_shares": 0, "execution_status": "pending"}
        assert Database.fetchone(
            "SELECT execution_status FROM rebalance_signals WHERE id=%s", (sibling_signal_id,),
        )["execution_status"] == "pending"
        assert Database.fetchone("SELECT event_key FROM local_ledger_events WHERE event_key=%s",
                                 (f"manual:LOCAL-LIVE:{manual_first['request_id']}",)) is None
        assert Database.fetchone("SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='manual_fill'")[
            "n"] == 0
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", {
            "id": "T-UNBOOKED-LINKED", "order_id": "125", "symbol": "000002.SZ",
            "side": "sell", "quantity": 1, "price": "4.000", "amount": "4.00",
            "commission": "0.30", "stamp_duty": "0.20",
        }, "query")
        trade_health = BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")
        assert trade_health["unbooked"] == 1
        assert trade_health["unassigned"] == 2
        assert {row["trade_id"]: row["issue_type"] for row in
                BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST")}[
                    "T-UNBOOKED-LINKED"] == "ledger_missing"
        market_order = {"id": "126", "symbol": "000003.SZ", "side": "buy",
                        "quantity": 100, "filled_qty": 0, "status": "pending",
                        "price_type": "market", "price": 4}
        assert BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", market_order, "query")["disposition"] == "applied"
        changed_type = BrokerOrderReportRepo.ingest(
            "qmt", "QMT-TEST", {**market_order, "price_type": "limit"}, "query",
        )
        assert changed_type["disposition"] == "conflict"
        assert Database.fetchone("SELECT price_type FROM broker_order_states WHERE broker='qmt' "
                                 "AND broker_account_id='QMT-TEST' AND order_id='126'")["price_type"] == "market"
        invalid_limit = {**market_order, "id": "127", "price_type": "limit",
                         "price": "10.0000000000000000000000000000001"}
        assert BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", invalid_limit, "query")["disposition"] == "conflict"
        unknown_price_type = {**market_order, "id": "128", "price_type": "unknown"}
        assert BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", unknown_price_type, "query")["disposition"] == "conflict"
        wrong_exchange = {**market_order, "id": "129", "symbol": "000001.SH"}
        assert BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", wrong_exchange, "query")["disposition"] == "conflict"
        assert BrokerOrderReportRepo.unresolved_count("qmt", "QMT-TEST") > 0
        limit_provider = lambda _: {"limit_up": 11, "limit_down": 9}
        fake = FakeBroker(initial_cash=5000, price_provider=lambda _: 10,
                          price_limit_provider=limit_provider)
        fake.connect()
        fake_order_id = fake.place_order("000001", "测试", "buy", 100, "market")
        fake_order = fake.get_order(fake_order_id)
        assert fake_order["status"] == "filled"
        account_row = Database.fetchone("SELECT cash FROM accounts WHERE id='FAKE-LOCAL'")
        assert account_row["cash"] == Decimal("3995.00")
        assert Database.fetchone("SELECT COUNT(*) AS n FROM orders WHERE id=%s", (fake_order_id,))["n"] == 1
        assert Database.fetchone("SELECT COUNT(*) AS n FROM trades WHERE order_id=%s", (fake_order_id,))["n"] == 1
        from backend.api import trading as trading_api
        Database.execute("UPDATE orders SET source='legacy' WHERE id=%s", (fake_order_id,))
        try:
            with patch.object(config, "BROKER_TYPE", config.BROKER_FAKE), \
                 patch.object(trading_api, "_broker", None), \
                 pytest.raises(RuntimeError, match="模拟账户状态恢复失败"):
                trading_api._get_broker_locked()
        finally:
            Database.execute("UPDATE orders SET source='broker' WHERE id=%s", (fake_order_id,))
        stored_order = Database.fetchone("SELECT * FROM orders WHERE id=%s", (fake_order_id,))
        other_account = FakeBroker(initial_cash=1000)
        other_account._account_id = "FAKE-OTHER"
        with pytest.raises(RuntimeError, match="状态已回滚"):
            other_account._persist_simulation(
                order={**fake_order, "status": "cancelled", "filled_qty": 0},
                previous_state=other_account._capture_state(),
            )
        assert Database.fetchone("SELECT account_id,status FROM orders WHERE id=%s", (fake_order_id,)) == {
            "account_id": "FAKE-LOCAL", "status": stored_order["status"],
        }

        class OtherLiveBroker(BaseBroker):
            @property
            def account_id(self):
                return "QMT-OTHER"

        live_broker = OtherLiveBroker()
        with pytest.raises(BrokerPersistenceConflict, match="已属于其他账户"):
            live_broker._persist_order({**fake_order, "status": "cancelled"})
        stored_trade = Database.fetchone("SELECT * FROM trades WHERE order_id=%s", (fake_order_id,))
        with pytest.raises(BrokerPersistenceConflict, match="已属于其他账户"):
            live_broker._persist_trade({
                "id": stored_trade["id"], "order_id": stored_trade["order_id"],
                "symbol": stored_trade["symbol"], "name": stored_trade["name"],
                "side": stored_trade["side"], "price": stored_trade["price"],
                "quantity": stored_trade["quantity"], "amount": stored_trade["amount"],
                "commission": stored_trade["commission"], "stamp_duty": stored_trade["stamp_duty"],
                "trade_time": stored_trade["trade_time"],
            })
        assert Database.fetchone("SELECT account_id,quantity FROM trades WHERE id=%s", (stored_trade["id"],)) == {
            "account_id": "FAKE-LOCAL", "quantity": stored_trade["quantity"],
        }
        pending_buy_id = fake.place_order("000001", "待成交", "buy", 100, "limit", 9)
        assert fake.get_order(pending_buy_id)["status"] == "pending"
        reloaded = FakeBroker(initial_cash=0, price_provider=lambda _: 10,
                              price_limit_provider=limit_provider)
        restored_account_row = Database.fetchone("SELECT cash,frozen FROM accounts WHERE id='FAKE-LOCAL'")
        reloaded._cash = float(restored_account_row["cash"])
        reloaded._frozen = float(restored_account_row["frozen"])
        reloaded.connect()
        reloaded.restore(
            Database.fetchall("SELECT * FROM orders WHERE account_id='FAKE-LOCAL' ORDER BY created_at,id"),
            Database.fetchall("SELECT * FROM trades WHERE account_id='FAKE-LOCAL' ORDER BY trade_time,id"),
        )
        assert reloaded.get_positions()[0]["shares"] == 100
        assert reloaded.get_account()["cash"] == 3995
        assert reloaded.get_account()["frozen"] == 905
        assert reloaded.get_order(pending_buy_id)["reserved_cash"] == 905
        assert reloaded.get_account()["status"] == "fresh"
        assert reloaded.cancel_order(pending_buy_id) is True
        assert reloaded.get_account()["frozen"] == 0
        before_failed_fill = fake._capture_state()
        before_order_count = Database.fetchone("SELECT COUNT(*) AS n FROM orders WHERE account_id='FAKE-LOCAL'")["n"]
        before_trade_count = Database.fetchone("SELECT COUNT(*) AS n FROM trades WHERE account_id='FAKE-LOCAL'")["n"]
        Database.execute(
            "CREATE TRIGGER reject_isolated_fake_trade BEFORE INSERT ON trades "
            "FOR EACH ROW BEGIN IF NEW.account_id='FAKE-LOCAL' THEN "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='isolated fake trade failure'; "
            "END IF; END"
        )
        try:
            with pytest.raises(RuntimeError, match="状态已回滚"):
                fake.place_order("000002", "故障注入", "buy", 100, "market")
        finally:
            Database.execute("DROP TRIGGER reject_isolated_fake_trade")
        assert fake._capture_state() == before_failed_fill
        assert Database.fetchone("SELECT COUNT(*) AS n FROM orders WHERE account_id='FAKE-LOCAL'")["n"] == before_order_count
        assert Database.fetchone("SELECT COUNT(*) AS n FROM trades WHERE account_id='FAKE-LOCAL'")["n"] == before_trade_count
        assert Database.fetchone("SELECT cash FROM accounts WHERE id='FAKE-LOCAL'")["cash"] == Decimal("3995.00")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.rebalance import router as rebalance_router

        api_app = FastAPI()
        api_app.include_router(rebalance_router)
        api_client = TestClient(api_app)
        decision_date = datetime.date.today()
        integration_code = "600008"
        integration_decision = {
            "decision_id": "isolated-ledger-api-decision",
            "screening_save_id": 1,
            "summary": {"mode": "legacy"},
            "policy": {"status": "active"},
            "ranked_candidates": [{"code": integration_code, "name": "隔离验收股",
                                   "hard_excluded": False}],
            "selected": [integration_code],
            "data_cutoff": decision_date.isoformat(),
            "excluded": [],
            "fingerprint": "d" * 64,
        }
        api_save = {"candidates": [{"code": integration_code, "name": "隔离验收股"}],
                    "pool": "all", "eval_date": decision_date.isoformat(),
                    "direction": None}

        class IntegrationAdvisor:
            def __init__(self, **kwargs):
                from types import SimpleNamespace
                self._allocator = SimpleNamespace(config=SimpleNamespace())

            def generate_advice_from_screening(self, **kwargs):
                return {"orders": [{"code": integration_code, "name": "隔离验收股",
                                    "side": "buy", "action": "买入", "amount": 1000,
                                    "urgency": "normal", "reason": "隔离端到端提交",
                                    "current_weight": 0, "target_weight": 0.1,
                                    "delta_weight": 0.1}],
                        "classification": {}, "constraints": [], "turnover": 0,
                        "observe_candidates": [], "top_n_codes": [integration_code], "skipped": []}

        initial_signal_count = Database.fetchone(
            "SELECT COUNT(*) AS n FROM rebalance_signals WHERE account_id='LOCAL-LIVE'"
        )["n"]
        initial_session_count = Database.fetchone(
            "SELECT COUNT(*) AS n FROM rebalance_sessions WHERE account_id='LOCAL-LIVE'"
        )["n"]
        simulated_reconciliation = _ready_reconciliation(
            "QMT-TEST", LocalLedgerRepo.get_account()["version"],
        )
        api_payload = {
            "screening_save_id": 1,
            "decision_id": integration_decision["decision_id"],
            "enforce_strategy_profile": False,
            "unmanaged_reason": "隔离 MySQL 端到端验收",
        }
        with patch("backend.api.auth.require_admin_actor", return_value="isolated-admin"), \
             patch("backend.service.portfolio_decision.PortfolioDecisionService.get",
                   return_value=integration_decision), \
             patch("core.shared.trading_time.get_latest_trading_day", return_value=decision_date), \
             patch("core.data.ScreeningSaveRepo.get_by_id", return_value=api_save), \
             patch("core.data.SettingsRepo.get", return_value=None), \
             patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", IntegrationAdvisor), \
             patch("backend.service.rebalance._get_prices_batch",
                   return_value=({integration_code: 10.0}, "realtime")), \
             patch("backend.service.rebalance._fetch_recent_amount", return_value={}), \
             patch("backend.service.rebalance._get_regime_stop_params", return_value={
                 "stop_loss_warn": -0.04, "stop_loss_full": -0.08,
                 "take_profit_half": 0.25, "take_profit_full": 0.5, "source": "isolated-test",
             }), \
             patch("backend.service.rebalance._load_half_sold", return_value=set()), \
             patch("core.domain.timing.model.get_timing_ratio", return_value=1.0), \
             patch("core.domain.portfolio.consistency.check_panic", return_value=False), \
             patch("core.domain.regime.classifier.get_current_regime", return_value=None), \
             patch("core.domain.screening.exclusion.prefilter_by_liquidity_capacity",
                   return_value=([integration_code], [], {integration_code: {"status": "pass"}})), \
             patch("backend.service.local_ledger.LocalLedgerService.reconcile",
                   return_value=simulated_reconciliation), \
             patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"):
            preview_response = api_client.post("/run", json=api_payload)
            assert preview_response.status_code == 200, preview_response.text
            preview = preview_response.json()["data"]
            assert preview["preview_only"] is True
            assert preview["account"]["account_id"] == "LOCAL-LIVE"
            assert preview["account"]["cash_source"] == "local_ledger"
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM rebalance_signals WHERE account_id='LOCAL-LIVE'"
            )["n"] == initial_signal_count
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM rebalance_sessions WHERE account_id='LOCAL-LIVE'"
            )["n"] == initial_session_count

            commit_request_id = str(uuid.uuid4())
            commit_payload = {
                **api_payload,
                "preview_only": False,
                "commit_request_id": commit_request_id,
                "expected_plan_fingerprint": preview["plan_fingerprint"],
            }
            committed_response = api_client.post("/run", json=commit_payload)
            assert committed_response.status_code == 200, committed_response.text
            committed = committed_response.json()["data"]
            assert committed["committed"] is True
            assert committed["signals"][0]["suggested_shares"] == 100
            saved_session = RebalanceSessionRepo.get_by_commit_request(commit_request_id, "LOCAL-LIVE")
            assert saved_session["plan_fingerprint"] == preview["plan_fingerprint"]
            legacy_snapshot = next(row for row in saved_session["holdings_json"]
                                   if row["code"] == "699999")
            assert legacy_snapshot["buy_date"] == "2026-10-28"
            stored_signals = RebalanceSignalRepo.get_by_batch(committed["batch_id"], account_id="LOCAL-LIVE")
            assert len(stored_signals) == 1
            assert stored_signals[0]["planned_shares"] == 100

            retry_response = api_client.post("/run", json=commit_payload)
            assert retry_response.status_code == 200, retry_response.text
            retry = retry_response.json()["data"]
            assert retry["already_committed"] is True
            assert retry["signals"][0]["id"] == stored_signals[0]["id"]
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM rebalance_signals WHERE account_id='LOCAL-LIVE'"
            )["n"] == initial_signal_count + 1
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM rebalance_sessions WHERE account_id='LOCAL-LIVE'"
            )["n"] == initial_session_count + 1

            signal_id = stored_signals[0]["id"]
            pre_expiry_account = LocalLedgerRepo.get_account()
            pre_expiry_event_count = Database.fetchone(
                "SELECT COUNT(*) AS n FROM local_ledger_events WHERE account_id='LOCAL-LIVE'"
            )["n"]
            Database.execute(
                "UPDATE rebalance_signals SET created_at=DATE_SUB(NOW(), INTERVAL 1 DAY) WHERE id=%s",
                (signal_id,),
            )
            stale_reconciliation_time = datetime.datetime.now().isoformat()
            stale_intent = {
                "intent_id": str(uuid.uuid4()), "signal_id": signal_id,
                "expected_account_version": saved_session["account_version"],
                "quantity": 100, "limit_price": 10, "fee_cap": 1,
                "broker_account_id": "QMT-TEST", "actor": "isolated-admin",
                "reconciliation": _ready_reconciliation("QMT-TEST", saved_session["account_version"]),
            }
            with pytest.raises(ValueError, match="信号已过期"):
                LocalOrderIntentRepo.prepare(stale_intent, decision_fingerprint)
            assert LocalLedgerRepo.get_account()["version"] == pre_expiry_account["version"]
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM local_ledger_events WHERE account_id='LOCAL-LIVE'"
            )["n"] == pre_expiry_event_count
            Database.execute("UPDATE rebalance_signals SET created_at=NOW() WHERE id=%s", (signal_id,))

            blocked_manual_response = api_client.put(
                f"/signals/{signal_id}/execute",
                json={"shares": 100, "price": "10.000",
                      "request_id": str(uuid.uuid4()),
                      "expected_account_version": saved_session["account_version"],
                      "commission": "0.30", "stamp_duty": "0.00",
                      "broker_trade_id": "UNVERIFIED-MANUAL-TRADE"},
            )
            assert blocked_manual_response.status_code == 400
            assert "必须通过本地委托意图" in blocked_manual_response.json()["detail"]
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM local_ledger_events WHERE account_id='LOCAL-LIVE'"
            )["n"] == pre_expiry_event_count

            from backend.service.local_ledger import LocalLedgerService
            reconciliation = _ready_reconciliation("QMT-TEST", saved_session["account_version"])
            intent_id = str(uuid.uuid4())
            intent_payload = {
                "intent_id": intent_id, "signal_id": signal_id,
                "expected_account_version": saved_session["account_version"],
                "quantity": 100, "limit_price": "10.000", "fee_cap": "1.00",
            }
            with patch.object(LocalLedgerService, "reconcile", return_value=reconciliation), \
                 patch("backend.api.auth.require_admin_actor", return_value="isolated-admin"), \
                 patch("backend.service.strategy_profile.StrategyProfileService.ensure_execution_allowed"):
                prepare_response = api_client.post("/account/local/intents", json=intent_payload)
                assert prepare_response.status_code == 200, prepare_response.text
                prepared = prepare_response.json()["data"]
                assert prepared["intent_id"] == intent_id
                assert prepared["state"] == "prepared"
                assert prepared["reserved_cash"] == 1001

                simulated_order = {
                    "id": "R117-INTENT-ORDER", "symbol": f"{integration_code}.SH", "side": "buy",
                    "quantity": 100, "filled_qty": 100, "status": "filled", "price_type": "limit",
                    "price": 10, "filled_price": 10,
                }
                BrokerOrderReportRepo.ingest("qmt", "QMT-TEST", simulated_order, "query")
                attach_response = api_client.post(
                    f"/account/local/intents/{intent_id}/attach-order",
                    json={"broker_order_id": "R117-INTENT-ORDER"},
                )
                assert attach_response.status_code == 200, attach_response.text
                assert attach_response.json()["data"]["state"] == "submitted"

                simulated_trade = {
                    "id": "R117-INTENT-TRADE", "order_id": "R117-INTENT-ORDER",
                    "symbol": f"{integration_code}.SH", "side": "buy", "quantity": 100,
                    "price": "10.000", "amount": "1000.00", "commission": "0.30", "stamp_duty": "0.00",
                }
                BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", simulated_trade, "query")
                broker_fill_payload = {
                    "broker_trade_id": "R117-INTENT-TRADE", "commission": "0.30", "stamp_duty": "0.00",
                }
                fill_response = api_client.post(
                    f"/account/local/intents/{intent_id}/broker-fill", json=broker_fill_payload,
                )
                assert fill_response.status_code == 200, fill_response.text
                fill_result = fill_response.json()["data"]
                assert fill_result["already_recorded"] is False
                assert fill_result["state"] == "filled"
                fill_retry = api_client.post(
                    f"/account/local/intents/{intent_id}/broker-fill", json=broker_fill_payload,
                )
                assert fill_retry.status_code == 200, fill_retry.text
                assert fill_retry.json()["data"]["already_recorded"] is True

            assert LocalLedgerRepo.audit_integrity()["ok"] is True
            new_account = LocalLedgerRepo.get_account()
            assert new_account["version"] == saved_session["account_version"] + 3
            assert new_account["cash"] == Decimal("998.90")
            assert Database.fetchone(
                "SELECT shares FROM holdings WHERE account_id='LOCAL-LIVE' AND code=%s",
                (integration_code,),
            )["shares"] == 100

            refreshed_response = api_client.post("/run", json=api_payload)
            assert refreshed_response.status_code == 200, refreshed_response.text
            refreshed_preview = refreshed_response.json()["data"]
            assert refreshed_preview["preview_only"] is True
            assert refreshed_preview["account"]["account_version"] == new_account["version"]
            assert refreshed_preview["holdings"][0]["code"] == integration_code
            assert refreshed_preview["holdings"][0]["shares"] == 100
            assert refreshed_preview["plan_fingerprint"] != preview["plan_fingerprint"]
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM rebalance_signals WHERE account_id='LOCAL-LIVE'"
            )["n"] == initial_signal_count + 1
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM rebalance_sessions WHERE account_id='LOCAL-LIVE'"
            )["n"] == initial_session_count + 1

        Database.execute("UPDATE local_ledger_accounts SET cash=cash+1 WHERE account_id='LOCAL-LIVE'")
        assert "账户现金与事件累计不一致" in LocalLedgerRepo.audit_integrity()["issues"]
        Database.execute("UPDATE local_ledger_accounts SET cash=cash-1 WHERE account_id='LOCAL-LIVE'")
        account_for_audit = LocalLedgerRepo.get_account()
        Database.execute(
            "INSERT INTO local_ledger_events (event_key,account_id,event_type,cash_delta,cash_after,"
            "account_version,actor,occurred_at) VALUES (%s,'LOCAL-LIVE','reserve',0,%s,%s,'test',NOW())",
            ("audit-tamper:" + uuid.uuid4().hex, account_for_audit["cash"] + 1,
             account_for_audit["version"] + 1),
        )
        Database.execute("UPDATE local_ledger_accounts SET version=version+1 WHERE account_id='LOCAL-LIVE'")
        assert "账户事件逐笔现金不连续" in LocalLedgerRepo.audit_integrity()["issues"]
    finally:
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created and database_name.startswith("codex_w1_"):
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_easytrader_intent_partial_fill_and_terminal_lifecycle_in_isolated_mysql():
    database_name = "codex_easytrader_w1_" + uuid.uuid4().hex[:10]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated_tables = (
            "holdings", "rebalance_signals", "rebalance_sessions", "holdings_adjustments",
            "accounts", "orders", "trades", "settings", "schema_migrations", "local_ledger_accounts",
            "local_ledger_events", "local_order_intents", "broker_order_states",
            "broker_order_reports", "broker_trade_reports", "broker_trade_report_conflicts",
        )
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in isolated_tables:
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        Database.execute("ALTER TABLE holdings DROP COLUMN version")
        ensure_local_ledger_schema()
        version_column = Database.fetchone("SHOW COLUMNS FROM holdings LIKE 'version'")
        assert version_column and str(version_column["Default"]) == "1"

        opening = LocalLedgerRepo.get_opening_snapshot()
        LocalLedgerRepo.activate("2000.00", opening["fingerprint"], "admin", "隔离Easytrader期初核准测试")
        broker_account_id = "EASYT-ACCOUNT-1"
        from backend.service.local_ledger import LocalLedgerService

        easytrader = EasytraderBroker(broker_account_id="ACCOUNT-1")
        easytrader._connected = True
        easytrader._trader = Mock()
        easytrader._trader.balance = [{
            "资金帐号": "ACCOUNT-1", "可用资金": 2000, "参考市值": 0, "总资产": 2000,
        }]
        easytrader._trader.position = []
        easytrader._trader.today_entrusts = []
        easytrader._trader.today_trades = []
        with patch("backend.api.trading.get_broker", return_value=easytrader), \
             patch("core.shared.config.BROKER_TYPE", "easytrader"), \
             patch("core.shared.config.BROKER_FAKE", "fake"):
            bind_reconciliation = LocalLedgerService.reconcile()
            assert bind_reconciliation["can_bind"] is True
            bound = LocalOrderIntentRepo.bind_broker(
                broker_account_id, 1, "admin", bind_reconciliation,
            )
        assert bound["account_version"] == 2
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("EASYT-B1", "000001", "test", "buy", "买入", 100, 10,
             "LOCAL-LIVE", 2, opening["fingerprint"]),
        )
        signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id=%s", ("EASYT-B1",),
        )["id"]
        with patch("backend.api.trading.get_broker", return_value=easytrader), \
             patch("core.shared.config.BROKER_TYPE", "easytrader"), \
             patch("core.shared.config.BROKER_FAKE", "fake"):
            execution_reconciliation = LocalLedgerService.reconcile()
            assert execution_reconciliation["can_execute"] is True
        intent = {
            "intent_id": str(uuid.uuid4()), "signal_id": signal_id,
            "broker_account_id": broker_account_id, "expected_account_version": 2,
            "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin",
            "reconciliation": execution_reconciliation,
        }
        assert LocalOrderIntentRepo.prepare(intent, decision_fingerprint)["reserved_cash"] == 1001
        order_raw = {"委托编号": "ET-O-1", "证券代码": "000001", "操作": "买入",
                     "委托数量": 100, "成交数量": 0, "委托价格": 10,
                     "成交价格": 0, "状态": "已报"}
        easytrader._trader.today_entrusts = [order_raw]
        assert easytrader.get_orders()[0]["report_disposition"] == "applied"
        assert LocalOrderIntentRepo.attach_broker_order(intent["intent_id"], "ET-O-1", "admin")[
            "state"] == "submitted"
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("EASYT-OLD-BATCH", "000002", "stale", "buy", "买入", 200, 10,
             "LOCAL-LIVE", 2, opening["fingerprint"]),
        )
        stale_fill_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id=%s", ("EASYT-OLD-BATCH",),
        )["id"]

        partial_order_raw = {**order_raw, "成交数量": 40, "成交价格": 10, "状态": "部成"}
        easytrader._trader.today_entrusts = [partial_order_raw]
        easytrader.get_orders()
        easytrader._trader.today_trades = [{
            "成交编号": "ET-T-1", "委托编号": "ET-O-1", "证券代码": "000001", "操作": "买入",
            "成交数量": 40, "成交价格": 10, "成交金额": 400,
            "手续费": 0.12, "印花税": 0, "成交时间": "10:01:00",
        }]
        easytrader.get_trades()
        Database.execute(
            "CREATE TRIGGER reject_broker_fill_event BEFORE INSERT ON local_ledger_events "
            "FOR EACH ROW BEGIN IF NEW.event_type='broker_fill' THEN SIGNAL SQLSTATE '45000' "
            "SET MESSAGE_TEXT='reject broker fill event'; END IF; END"
        )
        with pytest.raises(pymysql.err.Error):
            LocalOrderIntentRepo.record_broker_fill(
                intent["intent_id"], "ET-T-1", "0.12", "0.00", "admin", decision_fingerprint,
            )
        rolled_back_fill_account = LocalLedgerRepo.get_account()
        assert rolled_back_fill_account["cash"] == Decimal("2000.00")
        assert rolled_back_fill_account["frozen"] == Decimal("1001.00")
        assert Database.fetchone(
            "SELECT state,filled_quantity,account_version FROM local_order_intents WHERE intent_id=%s",
            (intent["intent_id"],),
        ) == {"state": "submitted", "filled_quantity": 0,
              "account_version": rolled_back_fill_account["version"]}
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM holdings WHERE account_id='LOCAL-LIVE' AND code='000001'"
        )["n"] == 0
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='broker_fill'"
        )["n"] == 0
        failed_fill_evidence = LocalOrderIntentRepo.get_broker_evidence(intent["intent_id"])
        assert [(row["trade_id"], row["booked"]) for row in failed_fill_evidence["trades"]] == [
            ("ET-T-1", False),
        ]
        Database.execute("DROP TRIGGER reject_broker_fill_event")
        fill = LocalOrderIntentRepo.record_broker_fill(
            intent["intent_id"], "ET-T-1", "0.12", "0.00", "admin", decision_fingerprint,
        )
        assert fill["state"] == "partial"
        assert fill["remaining_plan_shares"] == 60
        assert fill["invalidated_signal_ids"] == [stale_fill_signal_id]
        assert fill["invalidated_signal_details"][0]["invalidated_remaining_shares"] == 200
        assert LocalLedgerRepo.get_account()["cash"] == Decimal("1599.88")
        fill_retry = LocalOrderIntentRepo.record_broker_fill(
            intent["intent_id"], "ET-T-1", "0.12", "0.00", "admin", decision_fingerprint,
        )
        assert fill_retry["already_recorded"] is True
        assert fill_retry["invalidated_signal_ids"] == [stale_fill_signal_id]
        fill_event = Database.fetchone(
            "SELECT payload_json FROM local_ledger_events WHERE event_type='broker_fill'",
        )
        fill_event_payload = json.loads(fill_event["payload_json"])
        assert fill_event_payload["invalidated_signal_details"] == fill["invalidated_signal_details"]

        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("EASYT-B1", "000003", "same batch", "buy", "买入", 100, 10,
             "LOCAL-LIVE", fill["account_version"], opening["fingerprint"]),
        )
        same_batch_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id=%s AND code='000003'", ("EASYT-B1",),
        )["id"]
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,"
            "planned_price,account_id,account_version,input_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("EASYT-NEW-BATCH", "000004", "stale after terminal", "buy", "买入", 300, 10,
             "LOCAL-LIVE", fill["account_version"], opening["fingerprint"]),
        )
        stale_terminal_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id=%s", ("EASYT-NEW-BATCH",),
        )["id"]

        cancelled_order_raw = {**partial_order_raw, "状态": "已撤"}
        easytrader._trader.today_entrusts = [cancelled_order_raw]
        easytrader.get_orders()
        before_terminal = LocalLedgerRepo.get_account()
        Database.execute(
            "CREATE TRIGGER reject_broker_terminal_event BEFORE INSERT ON local_ledger_events "
            "FOR EACH ROW BEGIN IF NEW.event_type='broker_terminal' THEN SIGNAL SQLSTATE '45000' "
            "SET MESSAGE_TEXT='reject broker terminal event'; END IF; END"
        )
        with pytest.raises(pymysql.err.Error):
            LocalOrderIntentRepo.finalize_terminal(intent["intent_id"], "admin")
        rolled_back_terminal_account = LocalLedgerRepo.get_account()
        assert rolled_back_terminal_account["frozen"] == before_terminal["frozen"]
        assert Database.fetchone(
            "SELECT state,account_version FROM local_order_intents WHERE intent_id=%s", (intent["intent_id"],),
        ) == {"state": "partial", "account_version": rolled_back_terminal_account["version"]}
        assert Database.fetchone(
            "SELECT execution_status FROM rebalance_signals WHERE id=%s", (stale_terminal_signal_id,),
        )["execution_status"] == "pending"
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM local_ledger_events WHERE event_type='broker_terminal'"
        )["n"] == 0
        Database.execute("DROP TRIGGER reject_broker_terminal_event")
        terminal = LocalOrderIntentRepo.finalize_terminal(intent["intent_id"], "admin")
        assert terminal["state"] == "cancelled"
        assert terminal["invalidated_signal_ids"] == [stale_terminal_signal_id]
        assert terminal["invalidated_signal_details"][0]["invalidated_remaining_shares"] == 300
        assert Database.fetchone(
            "SELECT execution_status,account_version FROM rebalance_signals WHERE id=%s",
            (same_batch_signal_id,),
        ) == {"execution_status": "pending", "account_version": terminal["account_version"]}
        terminal_event = Database.fetchone(
            "SELECT payload_json FROM local_ledger_events WHERE event_type='broker_terminal'",
        )
        terminal_payload = json.loads(terminal_event["payload_json"])
        assert terminal_payload["invalidated_signal_details"] == terminal["invalidated_signal_details"]
        terminal_retry = LocalOrderIntentRepo.finalize_terminal(intent["intent_id"], "admin")
        assert terminal_retry["already_finalized"] is True
        assert terminal_retry["invalidated_signal_details"] == terminal["invalidated_signal_details"]
        assert LocalLedgerRepo.get_account()["frozen"] == 0
        projected_holding = Database.fetchone(
            "SELECT shares,version FROM holdings WHERE account_id='LOCAL-LIVE' AND code='000001'",
        )
        assert projected_holding["shares"] == 40
        assert projected_holding["version"] == 1
        evidence = LocalOrderIntentRepo.get_broker_evidence(intent["intent_id"])
        assert evidence["order"]["status"] == "cancelled"
        assert [(row["trade_id"], row["booked"]) for row in evidence["trades"]] == [("ET-T-1", True)]

        late_fill_order = {**cancelled_order_raw, "成交数量": 60, "成交价格": 10}
        easytrader._trader.today_entrusts = [late_fill_order]
        easytrader._trader.today_trades = [
            {"成交编号": "ET-T-1", "委托编号": "ET-O-1", "证券代码": "000001", "操作": "买入",
             "成交数量": 40, "成交价格": 10, "成交金额": 400,
             "手续费": 0.12, "印花税": 0, "成交时间": "10:01:00"},
            {"成交编号": "ET-T-2", "委托编号": "ET-O-1", "证券代码": "000001", "操作": "买入",
             "成交数量": 20, "成交价格": 10, "成交金额": 200,
             "手续费": 0.06, "印花税": 0, "成交时间": "10:02:00"},
        ]
        late_order_report = BrokerOrderReportRepo.ingest(
            "easytrader", broker_account_id,
            {"id": "ET-O-1", "symbol": "000001", "side": "buy", "quantity": 100,
             "price_type": "limit", "price": 10, "filled_qty": 60, "filled_price": 10,
             "status": "cancelled"},
            "query", raw_status="已撤", raw_fields=late_fill_order,
        )
        easytrader.get_trades()
        assert late_order_report["disposition"] == "conflict"
        assert BrokerOrderReportRepo.unresolved_count("easytrader", broker_account_id) == 1
        late_evidence = LocalOrderIntentRepo.get_broker_evidence(intent["intent_id"])
        assert late_evidence["order_conflicts"] == 1
        assert late_evidence["trade_conflicts"] == 0
        assert late_evidence["conflicts"] == 1
        with pytest.raises(ValueError, match="委托意图或账户版本已变化"):
            LocalOrderIntentRepo.record_broker_fill(
                intent["intent_id"], "ET-T-2", "0.06", "0.00", "admin", decision_fingerprint,
            )
        assert LocalLedgerRepo.get_account()["cash"] == Decimal("1599.88")
        assert LocalLedgerRepo.get_account()["frozen"] == Decimal("0.00")
        assert Database.fetchone(
            "SELECT shares FROM holdings WHERE account_id='LOCAL-LIVE' AND code='000001'",
        )["shares"] == 40
        assert Database.fetchone(
            "SELECT state,filled_quantity FROM local_order_intents WHERE intent_id=%s", (intent["intent_id"],),
        ) == {"state": "cancelled", "filled_quantity": 40}
        assert Database.fetchone(
            "SELECT execution_status,filled_shares FROM rebalance_signals WHERE id=%s", (signal_id,),
        ) == {"execution_status": "partial", "filled_shares": 40}
        evidence = LocalOrderIntentRepo.get_broker_evidence(intent["intent_id"])
        assert [(row["trade_id"], row["booked"]) for row in evidence["trades"]] == [
            ("ET-T-1", True), ("ET-T-2", False),
        ]
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
    finally:
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created and database_name.startswith("codex_easytrader_w1_"):
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_opening_approval_binds_legacy_holdings_and_rolls_back_as_one_mysql_transaction():
    database_name = "codex_opening_w1_" + uuid.uuid4().hex[:10]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in ("holdings", "local_ledger_accounts", "local_ledger_events", "local_order_intents"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
                cur.execute("ALTER TABLE holdings MODIFY account_id VARCHAR(64) NULL DEFAULT 'LOCAL-LIVE'")
                cur.execute(
                    "INSERT INTO local_ledger_accounts (account_id,environment) VALUES ('LOCAL-LIVE','live')"
                )
                cur.execute(
                    "INSERT INTO holdings (account_id,code,name,shares,cost_price) "
                    "VALUES (NULL,'000001','legacy',100,10)"
                )
                cur.execute(
                    "CREATE TRIGGER fail_opening_insert BEFORE INSERT ON local_ledger_events "
                    "FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='forced opening failure'"
                )
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        opening = LocalLedgerRepo.get_opening_snapshot()
        assert opening["pending_holdings"] == [
            {"code": "000001", "shares": 100, "cost_price": 10.0,
             "buy_date": None, "source_account_id": None},
        ]
        with pytest.raises(Exception, match="forced opening failure"):
            LocalLedgerRepo.activate(
                "1000.00", opening["fingerprint"], "admin", "核准旧持仓并确认期初现金",
            )
        assert Database.fetchone(
            "SELECT account_id FROM holdings WHERE code='000001'",
        )["account_id"] is None
        assert Database.fetchone(
            "SELECT status,version,cash FROM local_ledger_accounts WHERE account_id='LOCAL-LIVE'",
        ) == {"status": "uninitialized", "version": 0, "cash": None}
        assert Database.fetchone("SELECT COUNT(*) AS n FROM local_ledger_events")["n"] == 0
        Database.execute("DROP TRIGGER fail_opening_insert")

        activated = LocalLedgerRepo.activate(
            "1000.00", opening["fingerprint"], "admin", "核准旧持仓并确认期初现金",
        )
        assert activated["status"] == "active"
        assert Database.fetchone(
            "SELECT account_id FROM holdings WHERE code='000001'",
        )["account_id"] == "LOCAL-LIVE"
        assert Database.fetchone(
            "SELECT version FROM holdings WHERE code='000001'",
        )["version"] == 2
        event = Database.fetchone("SELECT payload_json FROM local_ledger_events WHERE event_type='opening'")
        payload = event["payload_json"]
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        assert payload["pending_holdings_approved"] == 1
        assert LocalLedgerRepo.audit_integrity()["ok"] is True
        Database.execute(
            "INSERT INTO holdings (account_id,code,name,shares,cost_price) "
            "VALUES (NULL,'000002','unassigned',100,5)"
        )
        integrity = LocalLedgerRepo.audit_integrity()
        assert integrity["ok"] is False
        assert "存在未归属的旧持仓" in integrity["issues"]
    finally:
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created and database_name.startswith("codex_opening_w1_"):
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_holding_crud_is_audited_atomic_and_optimistic_in_isolated_mysql():
    database_name = "codex_holdings_w1_" + uuid.uuid4().hex[:10]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in ("holdings", "holdings_adjustments", "local_ledger_accounts"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
                cur.execute(
                    "INSERT INTO local_ledger_accounts (account_id,environment,status) "
                    "VALUES ('LOCAL-LIVE','live','uninitialized')"
                )
                cur.execute(
                    "CREATE TRIGGER fail_holding_adjustment BEFORE INSERT ON holdings_adjustments "
                    "FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='forced adjustment failure'"
                )
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        with pytest.raises(Exception, match="forced adjustment failure"):
            HoldingsRepo.add("000001", "平安银行", 100, 10, datetime.date.today().isoformat())
        assert Database.fetchone("SELECT COUNT(*) AS n FROM holdings") ["n"] == 0
        assert Database.fetchone("SELECT COUNT(*) AS n FROM holdings_adjustments") ["n"] == 0
        Database.execute("DROP TRIGGER fail_holding_adjustment")

        HoldingsRepo.add("000001", "平安银行", 100, 10, datetime.date.today().isoformat())
        initial_holding = HoldingsRepo.get_by_code("000001")
        with pytest.raises(pymysql.err.IntegrityError):
            HoldingsRepo.add("000001", "覆盖尝试", 500, 99, datetime.date.today().isoformat())
        row = HoldingsRepo.get_by_code("000001")
        assert row["shares"] == 100
        assert row["cost_price"] == Decimal("10.00")

        Database.execute(
            "CREATE TRIGGER fail_holding_update_audit BEFORE INSERT ON holdings_adjustments "
            "FOR EACH ROW BEGIN IF NEW.action='update' THEN "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='forced update audit failure'; END IF; END"
        )
        with pytest.raises(Exception, match="forced update audit failure"):
            HoldingsRepo.update("000001", risk_threshold=8, expected_id=initial_holding["id"], expected_version=1)
        assert HoldingsRepo.get_by_code("000001")["risk_threshold"] == 7
        Database.execute("DROP TRIGGER fail_holding_update_audit")

        today = datetime.date.today().isoformat()
        barrier = threading.Barrier(2)

        def concurrent_update(shares):
            barrier.wait(timeout=5)
            try:
                return HoldingsRepo.update(
                    "000001", shares=shares, expected_shares=100,
                    expected_cost_price="10.00", expected_buy_date=today,
                    expected_id=initial_holding["id"], expected_version=1,
                    reason="隔离并发调整持仓数量",
                )
            except StaleHoldingError:
                return "stale"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(concurrent_update, (200, 300)))
        assert sorted(str(value) for value in outcomes) == ["1", "stale"]
        current = HoldingsRepo.get_by_code("000001")
        assert current["shares"] in (200, 300)

        with pytest.raises(StaleHoldingError, match="持仓已变化"):
            HoldingsRepo.update(
                "000001", shares=400, expected_shares=100,
                expected_cost_price="10.00", expected_buy_date=today,
                expected_id=initial_holding["id"], expected_version=1,
                reason="隔离过期快照不得覆盖",
            )
        assert HoldingsRepo.update("000002", risk_threshold=8, expected_id=1, expected_version=1) == 0
        assert HoldingsRepo.delete(
            "000002", "核对后删除不存在的本地持仓", 100, 10, today, 1, 1,
        ) == 0

        adjustments = HoldingsRepo.get_adjustments("000001")
        actions = [row["action"] for row in adjustments]
        assert actions.count("create") == 1
        assert actions.count("update") == 1
        update_event = next(row for row in adjustments if row["action"] == "update")
        assert update_event["before_json"]["shares"] == 100
        assert update_event["after_json"]["shares"] == current["shares"]

        Database.execute(
            "CREATE TRIGGER fail_holding_delete_audit BEFORE INSERT ON holdings_adjustments "
            "FOR EACH ROW BEGIN IF NEW.action='delete' THEN "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='forced delete audit failure'; END IF; END"
        )
        with pytest.raises(Exception, match="forced delete audit failure"):
            HoldingsRepo.delete(
                "000001", "隔离验证删除失败应回滚", current["shares"],
                current["cost_price"], today, current["id"], current["version"],
            )
        assert HoldingsRepo.get_by_code("000001")["shares"] == current["shares"]
        Database.execute("DROP TRIGGER fail_holding_delete_audit")

        assert HoldingsRepo.delete(
            "000001", "隔离核对后删除本地持仓", current["shares"],
            current["cost_price"], today, current["id"], current["version"],
        ) == 1
        assert HoldingsRepo.get_by_code("000001") is None
        assert [row["action"] for row in HoldingsRepo.get_adjustments("000001")].count("delete") == 1

        Database.execute(
            "UPDATE local_ledger_accounts SET status='active' WHERE account_id='LOCAL-LIVE'"
        )
        with pytest.raises(ValueError, match="本地账户已激活"):
            HoldingsRepo.add("000002", "万科A", 100, 10, today)
        with pytest.raises(ValueError, match="本地账户已激活"):
            HoldingsRepo.update(
                "000001", shares=100, expected_shares=100,
                expected_cost_price="10.00", expected_buy_date=today,
                expected_id=current["id"], expected_version=1,
                reason="不能编辑已激活账本持仓",
            )
        with pytest.raises(ValueError, match="本地账户已激活"):
            HoldingsRepo.delete("000001", "不能删除已激活账本持仓", 100, 10, today, current["id"], 1)
    finally:
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created and database_name.startswith("codex_holdings_w1_"):
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_rebalance_signal_replacement_preserves_priority_and_rolls_back_in_isolated_mysql():
    database_name = "codex_signal_w1_" + uuid.uuid4().hex[:10]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in ("rebalance_signals", "local_ledger_accounts", "local_order_intents", "holdings"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
                cur.execute(
                    "INSERT INTO local_ledger_accounts (account_id,environment,status,cash,version) "
                    "VALUES ('LOCAL-LIVE','live','active',10000,1)"
                )
                cur.execute(
                    "INSERT INTO holdings (account_id,code,name,shares,cost_price,buy_date) "
                    "VALUES ('LOCAL-LIVE','000001','测试股票',100,10,CURDATE())"
                )
                cur.execute(
                    "INSERT INTO rebalance_signals (batch_id,check_type,code,name,side,action,urgency,"
                    "execution_status,account_id,account_version,planned_shares,source_mode,input_fingerprint) "
                    "VALUES ('old-batch','rebalance','000001','测试股票','sell','止损卖出','stop_loss',"
                    "'pending','LOCAL-LIVE',1,100,'live','fp')"
                )
                cur.execute(
                    "CREATE TRIGGER fail_signal_insert BEFORE INSERT ON rebalance_signals "
                    "FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='forced signal insert failure'"
                )
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name

        def create_batch(batch_id, urgency):
            return RebalanceSignalRepo.batch_create(
                batch_id, None, "rebalance",
                [{"code": "000001", "name": "测试股票", "side": "sell", "action": "止损卖出",
                  "urgency": urgency, "planned_shares": 100, "planned_price": 10}],
                source_mode="live", input_fingerprint="fp", replace_pending=True,
                fingerprint_fn=lambda rows: "fp", account_id="LOCAL-LIVE", account_version=1,
            )

        with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
             patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
            with pytest.raises(ValueError, match="更高优先级"):
                create_batch("lower-priority", "normal")
            assert Database.fetchone(
                "SELECT execution_status FROM rebalance_signals WHERE batch_id='old-batch'",
            )["execution_status"] == "pending"
            with pytest.raises(Exception, match="forced signal insert failure"):
                create_batch("failed-replacement", "stop_loss")
            assert Database.fetchone(
                "SELECT execution_status FROM rebalance_signals WHERE batch_id='old-batch'",
            )["execution_status"] == "pending"
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM rebalance_signals WHERE batch_id='failed-replacement'",
            )["n"] == 0

            Database.execute("DROP TRIGGER fail_signal_insert")
            new_ids = create_batch("new-batch", "stop_loss")

        assert len(new_ids) == 1
        assert Database.fetchone(
            "SELECT execution_status FROM rebalance_signals WHERE batch_id='old-batch'",
        )["execution_status"] == "cancelled"
        assert Database.fetchone(
            "SELECT execution_status,planned_shares,account_version FROM rebalance_signals "
            "WHERE batch_id='new-batch'",
        ) == {"execution_status": "pending", "planned_shares": 100, "account_version": 1}
    finally:
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created and database_name.startswith("codex_signal_w1_"):
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_qmt_recovers_failed_trade_projection_from_durable_report_in_isolated_mysql():
    database_name = "codex_qmt_replay_" + uuid.uuid4().hex[:12]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in ("trades", "broker_trade_reports", "broker_trade_report_conflicts",
                              "broker_report_resolution_events"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        Database.execute(
            "CREATE TRIGGER fail_qmt_trade_projection BEFORE INSERT ON trades FOR EACH ROW "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected projection failure'"
        )
        broker = QmtBroker("unused", "test")
        broker._connected = True
        broker._trader = Mock()
        broker._stock_account = object()
        raw_trade = SimpleNamespace(
            account_id="test", traded_id="T-REPLAY", order_id="O-1", stock_code="000001.SZ",
            order_type=23, traded_price=10, traded_volume=100, traded_amount=1000,
            commission=1, traded_time="10:00:00",
        )
        broker._trader.query_stock_trades.side_effect = [[raw_trade], []]
        with patch.object(broker, "_should_sync", return_value=True):
            broker._sync_trades()
            assert broker._report_persistence_error
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM broker_trade_reports WHERE trade_id='T-REPLAY'",
            )["n"] == 1
            assert Database.fetchone("SELECT COUNT(*) AS n FROM trades")["n"] == 0
            Database.execute("DROP TRIGGER fail_qmt_trade_projection")
            broker._sync_trades()

        assert broker._report_persistence_error is None
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_trade_reports WHERE trade_id='T-REPLAY'",
        )["n"] == 1
        projected = Database.fetchone("SELECT * FROM trades WHERE id='T-REPLAY'")
        assert projected["account_id"] == "QMT-test"
        assert projected["broker"] == "qmt"
        assert projected["quantity"] == 100
        assert Decimal(str(projected["amount"])) == Decimal("1000.00")
        contradictory = SimpleNamespace(
            account_id="test", traded_id="T-REPLAY", order_id="O-1", stock_code="000001.SZ",
            order_type=23, traded_price=9.9, traded_volume=100, traded_amount=990,
            commission=1, traded_time="10:00:00",
        )
        broker._trader.query_stock_trades.side_effect = None
        broker._trader.query_stock_trades.return_value = [contradictory]
        with patch.object(broker, "_should_sync", return_value=True):
            with pytest.raises(RuntimeError, match="查询或回报持久化不可用"):
                broker.get_trades()
        assert BrokerTradeReportRepo.unresolved_conflict_count("qmt", broker.account_id) == 1
        assert "未核对的矛盾成交回报" in broker._report_persistence_error
        assert len(broker._trades) == 1

        BrokerTradeReportRepo.resolve_conflicts(
            "qmt", broker.account_id, "T-REPLAY",
            {"id": "T-REPLAY", "order_id": "O-1", "symbol": "000001.SZ", "side": "buy",
             "quantity": 100, "price": "10.000", "amount": "1000.00",
             "commission": "1.00", "stamp_duty": None},
            "isolated-admin", "管理员核对后确认本地回报基准",
        )
        broker._trades[0]["price"] = 9.9
        broker._trader.query_stock_trades.return_value = [raw_trade]
        with patch.object(broker, "_should_sync", return_value=True):
            recovered_trades = broker.get_trades()
        assert broker._report_persistence_error is None
        assert BrokerTradeReportRepo.unresolved_conflict_count("qmt", broker.account_id) == 0
        assert len(recovered_trades) == 1
        assert recovered_trades[0]["price"] == 10
        assert recovered_trades[0]["amount"] == 1000
    finally:
        if connection._pool is not None:
            if connection._pool.stats()["borrowed"] == 0:
                connection._pool.shutdown()
                connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_easytrader_recovers_failed_trade_projection_and_excludes_conflicts_in_isolated_mysql():
    database_name = "codex_easytrader_replay_" + uuid.uuid4().hex[:12]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in ("orders", "broker_order_states", "broker_order_reports", "broker_order_submissions",
                              "trades", "broker_trade_reports", "broker_trade_report_conflicts",
                              "broker_report_resolution_events"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        Database.execute(
            "CREATE TRIGGER fail_easytrader_trade_projection BEFORE INSERT ON trades FOR EACH ROW "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected projection failure'"
        )
        broker = EasytraderBroker(broker_account_id="ACCT-REPLAY")
        broker._connected = True
        broker._trader = Mock()
        raw_trade = {
            "成交编号": "ET-T-REPLAY", "委托编号": "ET-O-1", "证券代码": "000001", "操作": "买入",
            "成交数量": 100, "成交价格": 10, "成交金额": 1000, "手续费": 1, "印花税": 0,
            "成交时间": "10:00:00",
        }
        broker._trader.today_trades = [raw_trade]
        with pytest.raises(Exception, match="injected projection failure"):
            broker.get_trades()
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_trade_reports WHERE broker='easytrader' AND trade_id='ET-T-REPLAY'",
        )["n"] == 1
        assert Database.fetchone("SELECT COUNT(*) AS n FROM trades")["n"] == 0

        Database.execute("DROP TRIGGER fail_easytrader_trade_projection")
        broker._trader.today_trades = []
        recovered = broker.get_trades()
        assert [trade["id"] for trade in recovered] == ["ET-T-REPLAY"]
        projected = Database.fetchone("SELECT * FROM trades WHERE id='ET-T-REPLAY'")
        assert projected["account_id"] == "EASYT-ACCT-REPLAY"
        assert projected["broker"] == "easytrader"
        assert projected["quantity"] == 100

        Database.execute(
            "CREATE TRIGGER fail_easytrader_order_projection BEFORE INSERT ON orders FOR EACH ROW "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected order projection failure'"
        )
        raw_order = {
            "委托编号": "ET-O-REPLAY", "证券代码": "000003", "操作": "买入", "委托数量": 100,
            "成交数量": 0, "委托价格": 12, "成交价格": 0, "状态": "已报", "委托时间": "09:31:00",
        }
        broker._trader.today_entrusts = [raw_order]
        with pytest.raises(Exception, match="injected order projection failure"):
            broker.get_orders()
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_order_reports WHERE broker='easytrader' AND order_id='ET-O-REPLAY'",
        )["n"] == 1
        assert Database.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0

        Database.execute("DROP TRIGGER fail_easytrader_order_projection")
        broker._trader.today_entrusts = []
        recovered_orders = broker.get_orders()
        assert [order["id"] for order in recovered_orders] == ["ET-O-REPLAY"]
        assert recovered_orders[0]["recovered_from_persisted_report"] is True
        projected_order = Database.fetchone("SELECT * FROM orders WHERE id='ET-O-REPLAY'")
        assert projected_order["account_id"] == "EASYT-ACCT-REPLAY"
        assert projected_order["broker"] == "easytrader"
        assert projected_order["status"] == "pending"

        canonical = {
            "id": "ET-T-CONFLICT", "order_id": "ET-O-2", "symbol": "000002", "side": "buy",
            "quantity": 100, "price": 11, "amount": 1100, "commission": 1, "stamp_duty": 0,
        }
        BrokerTradeReportRepo.ingest("easytrader", broker.account_id, canonical, "query")
        contradictory = dict(canonical, amount=1090, price=10.9)
        result = BrokerTradeReportRepo.ingest("easytrader", broker.account_id, contradictory, "query")
        assert result["status"] == "conflict"
        assert BrokerTradeReportRepo.list_unprojected("easytrader", broker.account_id) == ()
        assert Database.fetchone("SELECT COUNT(*) AS n FROM trades WHERE id='ET-T-CONFLICT'")["n"] == 0
        broker._trader.today_trades = [{
            "成交编号": "ET-T-CONFLICT", "委托编号": "ET-O-2", "证券代码": "000002", "操作": "买入",
            "成交数量": 100, "成交价格": 10.9, "成交金额": 1090, "手续费": 1, "印花税": 0,
        }]
        with pytest.raises(RuntimeError, match="成交回报"):
            broker.get_trades()
        assert Database.fetchone("SELECT COUNT(*) AS n FROM trades WHERE id='ET-T-CONFLICT'")["n"] == 0
    finally:
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created and database_name.startswith("codex_easytrader_replay_"):
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_qmt_recovers_failed_order_projection_from_durable_report_in_isolated_mysql():
    database_name = "codex_qmt_order_replay_" + uuid.uuid4().hex[:12]
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
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        isolated = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=database_name, autocommit=True,
        )
        try:
            with isolated.cursor() as cur:
                for table in ("orders", "broker_order_states", "broker_order_reports",
                              "broker_order_submissions"):
                    match = re.search(
                        rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                        schema,
                    )
                    assert match is not None
                    cur.execute(match.group())
        finally:
            isolated.close()
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        Database.execute(
            "CREATE TRIGGER fail_qmt_order_projection BEFORE INSERT ON orders FOR EACH ROW "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected projection failure'"
        )
        broker = QmtBroker("unused", "test")
        broker._connected = True
        broker._trader = Mock()
        broker._stock_account = object()
        raw_order = SimpleNamespace(
            account_id="test", order_id=123, stock_code="000001.SZ", order_type=23,
            order_volume=100, order_status=52, price_type=11, price=10,
            traded_volume=40, traded_price=9.9, order_time="2026-10-30 10:00:00",
        )
        broker._trader.query_stock_orders.side_effect = [[raw_order], []]
        with patch.object(broker, "_should_sync", return_value=True):
            broker._sync_orders()
            assert broker._report_persistence_error
            assert Database.fetchone(
                "SELECT COUNT(*) AS n FROM broker_order_reports WHERE order_id='123'",
            )["n"] == 1
            assert Database.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0
            Database.execute("DROP TRIGGER fail_qmt_order_projection")
            broker._sync_orders()

        assert broker._report_persistence_error is None
        assert Database.fetchone(
            "SELECT COUNT(*) AS n FROM broker_order_reports WHERE order_id='123'",
        )["n"] == 1
        projected = Database.fetchone("SELECT * FROM orders WHERE id='123'")
        assert projected["account_id"] == "QMT-test"
        assert projected["broker"] == "qmt"
        assert projected["environment"] == "live"
        assert projected["status"] == "partial"
        assert projected["filled_qty"] == 40
        assert broker._orders["123"]["status"] == "partial"
        assert broker._orders["123"]["recovered_from_persisted_report"] is True
    finally:
        if connection._pool is not None:
            if connection._pool.stats()["borrowed"] == 0:
                connection._pool.shutdown()
                connection._pool = None
        config.MYSQL_DATABASE = original_database
        if created:
            with admin.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()
