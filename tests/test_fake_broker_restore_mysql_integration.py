import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pymysql
import pytest

import core.data.connection as connection
from core.data.connection import Database
from core.shared import config
from core.adapters.broker.fake_broker import FakeBroker


@pytest.mark.integration
def test_fake_broker_account_orders_and_trades_restore_from_isolated_mysql():
    database_name = "codex_fake_restore_" + uuid.uuid4().hex[:12]
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
                for table in ("accounts", "orders", "trades"):
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

        quote = lambda symbol: {"price": 10, "limit_up": 11, "limit_down": 9}
        broker = FakeBroker(initial_cash=10000, price_provider=quote,
                            price_limit_provider=quote, price_limits_configured=True)
        assert broker.connect() is True
        broker._save_account()
        from core.adapters.broker import fake_broker as fake_module
        for table in ("accounts", "orders", "trades"):
            trigger = f"reject_fake_{table}"
            Database.execute(
                f"CREATE TRIGGER {trigger} BEFORE INSERT ON {table} FOR EACH ROW "
                f"SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='reject isolated fake {table}'"
            )
            with pytest.raises(RuntimeError, match="写入失败，状态已回滚"):
                broker.place_order("000001.SZ", "测试股", "buy", 100, "market")
            assert broker._cash == 10000
            assert broker._frozen == 0
            assert broker.get_orders() == []
            assert broker.get_trades() == []
            assert Database.fetchone("SELECT cash,frozen FROM accounts WHERE id=%s", (broker.account_id,)) == {
                "cash": Decimal("10000.00"), "frozen": Decimal("0.00"),
            }
            assert Database.fetchone("SELECT COUNT(*) AS n FROM orders WHERE account_id=%s",
                                     (broker.account_id,))["n"] == 0
            assert Database.fetchone("SELECT COUNT(*) AS n FROM trades WHERE account_id=%s",
                                     (broker.account_id,))["n"] == 0
            Database.execute(f"DROP TRIGGER {trigger}")
        with patch.object(fake_module, "_market_now", return_value=datetime(2026, 10, 29, 10, 0)):
            filled_order_id = broker.place_order("000001.SZ", "测试股", "buy", 100, "market")
        account = Database.fetchone("SELECT cash,frozen FROM accounts WHERE id=%s", (broker.account_id,))
        orders = Database.fetchall("SELECT * FROM orders WHERE account_id=%s ORDER BY created_at,id",
                                   (broker.account_id,))
        trades = Database.fetchall("SELECT * FROM trades WHERE account_id=%s ORDER BY trade_time,id",
                                   (broker.account_id,))
        broker.disconnect()
        broker = FakeBroker(initial_cash=0, price_provider=quote,
                            price_limit_provider=quote, price_limits_configured=True)
        broker._cash = float(account["cash"])
        broker._frozen = float(account["frozen"])
        with patch.object(fake_module, "_market_now", return_value=datetime(2026, 10, 30, 10, 0)):
            broker.restore(orders, trades)
        broker.connect()
        with patch.object(fake_module, "_market_now", return_value=datetime(2026, 10, 30, 10, 0)):
            sold_order_id = broker.place_order("000001.SZ", "测试股", "sell", 50, "market")
            pending_order_id = broker.place_order("000001.SZ", "测试股", "buy", 100, "limit", 9)
            cancelled_order_id = broker.place_order("000001.SZ", "测试股", "buy", 100, "limit", 9)
        assert broker.get_order(filled_order_id)["status"] == "filled"
        assert broker.get_order(sold_order_id)["status"] == "filled"
        assert broker.get_order(pending_order_id)["status"] == "pending"
        assert broker._persistence_available is True
        assert [
            (trade["side"], Decimal(str(trade["amount"])),
             Decimal(str(trade["commission"])), Decimal(str(trade["stamp_duty"])))
            for trade in broker.get_trades()
        ] == [
            ("buy", Decimal("1000.00"),
             round(max(Decimal("5"), Decimal("1000") * Decimal(str(config.PORTFOLIO_COMMISSION_RATE))), 2),
             Decimal("0.00")),
            ("sell", Decimal("500.00"),
             round(max(Decimal("5"), Decimal("500") * Decimal(str(config.PORTFOLIO_COMMISSION_RATE))), 2),
             round(Decimal("500") * Decimal(str(config.PORTFOLIO_STAMP_DUTY)), 2)),
        ]
        assert all(Decimal(str(trade["amount"])) == round(
            Decimal(str(trade["price"])) * trade["quantity"], 2)
                   for trade in broker.get_trades())
        assert all(Decimal(str(trade["commission"])) == round(max(
            Decimal("5"), Decimal(str(trade["amount"])) *
            Decimal(str(config.PORTFOLIO_COMMISSION_RATE))), 2)
                   for trade in broker.get_trades())
        assert all(Decimal(str(trade["stamp_duty"])) == round(
            Decimal(str(trade["amount"])) * Decimal(str(config.PORTFOLIO_STAMP_DUTY)), 2)
                   for trade in broker.get_trades() if trade["side"] == "sell")
        cash_after_trades = float(Decimal("10000") - sum(
            Decimal(str(trade["amount"])) + Decimal(str(trade["commission"])) +
            Decimal(str(trade["stamp_duty"]))
            for trade in broker.get_trades() if trade["side"] == "buy"
        ) + sum(
            Decimal(str(trade["amount"])) - Decimal(str(trade["commission"])) -
            Decimal(str(trade["stamp_duty"]))
            for trade in broker.get_trades() if trade["side"] == "sell"
        ))
        pending_reserve = broker.get_order(pending_order_id)["reserved_cash"]
        total_frozen = pending_reserve * 2
        with pytest.raises(ValueError, match="不能低于已冻结金额"):
            broker.set_cash(total_frozen - 0.01)
        assert broker._cash == cash_after_trades
        assert broker._frozen == total_frozen
        broker.set_cash(total_frozen + 190)
        assert broker._cash == total_frozen + 190
        assert broker._frozen == total_frozen
        assert broker.cancel_order(cancelled_order_id) is True
        assert broker.get_order(cancelled_order_id)["status"] == "cancelled"
        assert broker._frozen == pending_reserve
        rejected_order_id = broker.place_order("000001.SZ", "测试股", "buy", 10000, "market")
        assert broker.get_order(rejected_order_id)["status"] == "rejected"
        assert broker._cash == total_frozen + 190
        assert broker._frozen == pending_reserve

        startup = """
import json, sys
from core.shared import config
config.MYSQL_DATABASE = sys.argv[1]
config.MYSQL_ENABLED = True
config.BROKER_TYPE = 'fake'
from datetime import datetime
from core.adapters.broker import fake_broker as fake_module
fake_module._market_now = lambda: datetime(2026, 10, 30, 10, 0)
from backend.api import trading
trading._make_price_provider = lambda: lambda symbol: {'price': 10, 'limit_up': 11, 'limit_down': 9}
broker = trading.get_broker()
print(json.dumps({
    'cash': broker._cash,
    'frozen': broker._frozen,
    'available': broker._cash - broker._frozen,
    'holdings': broker._holdings,
    'orders': {order['id']: {'status': order['status'], 'reserved_cash': order.get('reserved_cash', 0)}
               for order in broker.get_orders()},
    'trades': len(broker.get_trades()),
    'trade_fees': [[float(trade['amount']), float(trade['commission']), float(trade['stamp_duty'])]
                   for trade in broker.get_trades()],
    'shares': broker.get_positions()[0]['shares'],
}, ensure_ascii=False))
broker.disconnect()
"""
        restarted = subprocess.run(
            [sys.executable, "-c", startup, database_name],
            cwd=Path(__file__).resolve().parents[1], env=os.environ.copy(),
            capture_output=True, text=True, check=True, timeout=30,
        )
        recovered = json.loads(restarted.stdout.splitlines()[-1])
        assert recovered["cash"] == broker._cash
        assert recovered["frozen"] == broker._frozen
        assert recovered["available"] == broker._cash - broker._frozen
        assert recovered["holdings"] == broker._holdings
        assert recovered["orders"][filled_order_id]["status"] == "filled"
        assert recovered["orders"][sold_order_id]["status"] == "filled"
        assert recovered["orders"][pending_order_id]["status"] == "pending"
        assert recovered["orders"][cancelled_order_id]["status"] == "cancelled"
        assert recovered["orders"][rejected_order_id]["status"] == "rejected"
        assert recovered["orders"][pending_order_id]["reserved_cash"] == broker.get_order(
            pending_order_id)["reserved_cash"]
        assert recovered["trades"] == len(broker.get_trades()) == 2
        expected_fees = [[float(trade["amount"]), float(trade["commission"]), float(trade["stamp_duty"])]
                         for trade in broker.get_trades()]
        assert recovered["trade_fees"] == expected_fees
        assert recovered["shares"] == broker.get_positions()[0]["shares"] == 50
    finally:
        config.MYSQL_DATABASE = original_database
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        if created:
            with admin.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()
