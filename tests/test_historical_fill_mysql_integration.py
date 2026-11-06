import re
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pymysql
import pytest

import core.data.connection as connection
from core.data.connection import Database
from core.data.repos.local_ledger import ACCOUNT_ID, LocalLedgerRepo
from core.domain.portfolio.decision import decision_fingerprint
from core.shared import config
from core.shared.trading_time import market_date


@pytest.mark.integration
def test_historical_fill_is_idempotent_does_not_revive_signal_and_rolls_back_atomically():
    database_name = "codex_historical_fill_" + uuid.uuid4().hex[:12]
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
                for table in ("holdings", "rebalance_signals", "local_ledger_accounts",
                              "local_ledger_events", "local_order_intents"):
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

        trade_date = market_date() - timedelta(days=1)
        opening_date = trade_date - timedelta(days=10)
        Database.execute(
            "INSERT INTO local_ledger_accounts (account_id,environment,status) VALUES (%s,'live','uninitialized')",
            (ACCOUNT_ID,),
        )
        Database.execute(
            "INSERT INTO holdings (account_id,code,name,shares,cost_price,buy_date) VALUES (%s,'000001','测试股',200,10,%s)",
            (ACCOUNT_ID, opening_date),
        )
        opening = LocalLedgerRepo.get_opening_snapshot()
        LocalLedgerRepo.activate(10000, opening["fingerprint"], "admin", "隔离集成测试期初核准")
        Database.execute("UPDATE local_ledger_accounts SET opened_at=%s WHERE account_id=%s",
                         (datetime.combine(opening_date, datetime.min.time()), ACCOUNT_ID))
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,account_id,"
            "account_version,execution_status) VALUES ('OLD-HISTORY','000001','测试股','sell','止盈减半',"
            "100,%s,1,'pending')", (ACCOUNT_ID,),
        )
        signal_id = Database.fetchone("SELECT id FROM rebalance_signals WHERE batch_id='OLD-HISTORY'")["id"]
        Database.execute("UPDATE rebalance_signals SET created_at=%s WHERE id=%s",
                         (datetime.combine(trade_date, datetime.min.time()), signal_id))
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,account_id,"
            "account_version,execution_status) VALUES ('OLD-HISTORY','000002','其他股','buy','买入',"
            "100,%s,1,'partial')", (ACCOUNT_ID,),
        )
        sibling_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='OLD-HISTORY' AND code='000002'"
        )["id"]

        request_id = str(uuid.uuid4())
        detail = {
            "request_id": request_id, "trade_date": trade_date.isoformat(), "code": "000001",
            "side": "sell", "shares": 100, "price": "11.000", "commission": "0.30",
            "stamp_duty": "0.55", "expected_account_version": 1, "broker_trade_id": "BROKER-TRADE-1",
            "source_signal_id": signal_id, "reason": "依据券商交割单补记漏记成交", "actor": "admin",
        }
        result = LocalLedgerRepo.record_historical_fill(detail)
        retry = LocalLedgerRepo.record_historical_fill(detail)
        assert result["already_recorded"] is False
        assert result["signal_updated"] is True
        assert result["invalidated_signal_ids"] == [signal_id, sibling_id]
        assert result["remaining_signal_shares"] == 0
        assert retry["already_recorded"] is True
        assert retry["invalidated_signal_ids"] == [signal_id, sibling_id]
        assert Database.fetchone("SELECT shares FROM holdings WHERE account_id=%s AND code='000001'",
                                 (ACCOUNT_ID,))["shares"] == 100
        assert LocalLedgerRepo.get_account()["cash"] == Decimal("11099.15")
        assert Database.fetchall(
            "SELECT execution_status,filled_shares,unfilled_reason FROM rebalance_signals "
            "WHERE id IN (%s,%s) ORDER BY id", (signal_id, sibling_id),
        ) == [
            {"execution_status": "cancelled", "filled_shares": 0,
             "unfilled_reason": "历史成交已补录，账户快照变化；剩余计划已失效，请重新生成"},
            {"execution_status": "cancelled", "filled_shares": 0,
             "unfilled_reason": "历史成交已补录，账户快照变化；剩余计划已失效，请重新生成"},
        ]
        integrity = LocalLedgerRepo.audit_integrity()
        assert integrity["ok"] is True, integrity

        over_plan = {**detail, "request_id": str(uuid.uuid4()), "expected_account_version": 2,
                     "broker_trade_id": "BROKER-TRADE-OVER"}
        with pytest.raises(ValueError, match="超过来源信号剩余计划股数"):
            LocalLedgerRepo.record_historical_fill(over_plan)

        Database.execute(
            "CREATE TRIGGER reject_historical_event BEFORE INSERT ON local_ledger_events FOR EACH ROW "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='reject historical event'"
        )
        Database.execute(
            "INSERT INTO rebalance_signals (batch_id,code,name,side,action,planned_shares,account_id,"
            "account_version,execution_status) VALUES ('ROLLBACK-HISTORY','000001','测试股','buy','买入',"
            "100,%s,2,'pending')", (ACCOUNT_ID,),
        )
        rollback_signal_id = Database.fetchone(
            "SELECT id FROM rebalance_signals WHERE batch_id='ROLLBACK-HISTORY'"
        )["id"]
        failed = {**detail, "request_id": str(uuid.uuid4()), "source_signal_id": None,
                  "broker_trade_id": "BROKER-TRADE-2", "shares": 100,
                  "expected_account_version": 2}
        with pytest.raises(pymysql.err.Error, match="reject historical event"):
            LocalLedgerRepo.record_historical_fill(failed)
        assert LocalLedgerRepo.get_account()["version"] == 2
        assert LocalLedgerRepo.get_account()["cash"] == Decimal("11099.15")
        assert Database.fetchone("SELECT shares FROM holdings WHERE account_id=%s AND code='000001'",
                                 (ACCOUNT_ID,))["shares"] == 100
        assert Database.fetchone(
            "SELECT execution_status,unfilled_reason FROM rebalance_signals WHERE id=%s",
            (rollback_signal_id,),
        ) == {"execution_status": "pending", "unfilled_reason": None}
        Database.execute("DROP TRIGGER reject_historical_event")

        out_of_order = {**detail, "request_id": str(uuid.uuid4()),
                        "trade_date": (trade_date - timedelta(days=1)).isoformat(),
                        "source_signal_id": None, "broker_trade_id": "BROKER-TRADE-0",
                        "expected_account_version": 2}
        with pytest.raises(ValueError, match="账本已有较晚成交"):
            LocalLedgerRepo.record_historical_fill(out_of_order)
    finally:
        config.MYSQL_DATABASE = original_database
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        if created:
            with admin.cursor() as cursor:
                cursor.execute(f"DROP DATABASE `{database_name}`")
        admin.close()
