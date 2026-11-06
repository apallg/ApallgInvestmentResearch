"""
交易执行抽象接口，所有 Broker 实现必须遵循此接口
"""
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class BrokerPersistenceConflict(RuntimeError):
    pass


def _trade_values(trade):
    trade_time = str(trade.get("trade_time") or "").replace("T", " ").split(".")[0]
    return (trade.get("order_id"), trade.get("symbol"), trade.get("side"),
            Decimal(str(trade.get("price") or 0)), int(trade.get("quantity") or 0),
            Decimal(str(trade.get("amount") or 0)), Decimal(str(trade.get("commission") or 0)),
            Decimal(str(trade.get("stamp_duty") or 0)), trade_time)


class BaseBroker:
    supported_price_types = ("limit",)

    def _persist_order(self, order, required=False):
        try:
            from core.data import Database
            if not Database.is_available():
                if required:
                    raise RuntimeError("数据库不可用，委托投影未持久化")
                return
            account_id = self.account_id
            broker = getattr(self, "broker_name", account_id.split("-")[0].lower())
            environment = "simulation" if account_id.startswith("FAKE-") else "live"

            def operation(cur):
                cur.execute("SELECT account_id,broker,environment,source,symbol,side,quantity,price_type,price "
                            "FROM orders WHERE id=%s FOR UPDATE",
                            (order["id"],))
                existing = cur.fetchone()
                identity = (account_id, broker, environment, "broker")
                if existing and tuple(existing.get(key) for key in ("account_id", "broker", "environment", "source")) != identity:
                    raise BrokerPersistenceConflict(f"委托编号 {order['id']} 已属于其他账户或来源，拒绝覆盖")
                if existing and (existing.get("symbol"), existing.get("side"), int(existing.get("quantity")),
                                 existing.get("price_type"), Decimal(str(existing.get("price") or 0))) != (
                        order["symbol"], order["side"], int(order["quantity"]), order["price_type"],
                        Decimal(str(order.get("price") or 0))):
                    raise BrokerPersistenceConflict(f"委托编号 {order['id']} 委托内容冲突，拒绝覆盖")
                cur.execute(
                    "INSERT INTO orders (id,symbol,name,side,quantity,price_type,price,"
                    "filled_qty,filled_price,status,created_at,updated_at,account_id,broker,environment,source) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON DUPLICATE KEY UPDATE filled_qty=VALUES(filled_qty),"
                    "filled_price=VALUES(filled_price),status=VALUES(status),updated_at=VALUES(updated_at)",
                    (order["id"], order["symbol"], order["name"], order["side"],
                     order["quantity"], order["price_type"], order.get("price") or 0,
                     order["filled_qty"], order["filled_price"], order["status"],
                     order["created_at"], order["updated_at"], account_id, broker, environment, "broker"))

            Database.transaction(operation)
        except BrokerPersistenceConflict:
            raise
        except Exception as e:
            if required:
                raise
            logger.debug(f"order持久化跳过: {e}")

    def _persist_trade(self, trade, required=False):
        try:
            from core.data import Database
            if not Database.is_available():
                if required:
                    raise RuntimeError("数据库不可用，成交投影未持久化")
                return
            account_id = self.account_id
            broker = getattr(self, "broker_name", account_id.split("-")[0].lower())
            environment = "simulation" if account_id.startswith("FAKE-") else "live"

            def operation(cur):
                cur.execute("SELECT account_id,broker,environment,source,order_id,symbol,side,price,quantity,"
                            "amount,commission,stamp_duty,trade_time FROM trades WHERE id=%s FOR UPDATE",
                            (trade["id"],))
                existing = cur.fetchone()
                identity = (account_id, broker, environment, "broker")
                if existing:
                    stored_identity = tuple(existing.get(key) for key in
                                            ("account_id", "broker", "environment", "source"))
                    if stored_identity != identity:
                        raise BrokerPersistenceConflict(f"成交编号 {trade['id']} 已属于其他账户或来源，拒绝覆盖")
                    if _trade_values(existing) != _trade_values(trade):
                        raise BrokerPersistenceConflict(f"成交编号 {trade['id']} 内容冲突，拒绝重复入账")
                    return
                cur.execute(
                    "INSERT INTO trades (id,order_id,symbol,name,side,price,quantity,"
                    "amount,commission,stamp_duty,trade_time,account_id,broker,environment,source) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (trade["id"], trade["order_id"], trade["symbol"], trade["name"],
                     trade["side"], trade["price"], trade["quantity"],
                     trade["amount"], trade["commission"], trade["stamp_duty"],
                     trade["trade_time"], account_id, broker, environment, "broker"))

            Database.transaction(operation)
        except BrokerPersistenceConflict:
            raise
        except Exception as e:
            if required:
                raise
            logger.debug(f"trade持久化跳过: {e}")
    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    @property
    def connected(self) -> bool:
        raise NotImplementedError

    @property
    def account_id(self) -> str:
        raise NotImplementedError

    def place_order(self, symbol, name, side, quantity, price_type, price=None):
        """
        下单，返回 order_id。
        side: "buy" | "sell"
        price_type: "market" | "limit"
        """
        raise NotImplementedError

    def cancel_order(self, order_id) -> bool:
        raise NotImplementedError

    def get_order(self, order_id):
        raise NotImplementedError

    def get_orders(self, status=None):
        raise NotImplementedError

    def get_trades(self, symbol=None):
        raise NotImplementedError

    def get_positions(self):
        raise NotImplementedError

    def get_account(self):
        raise NotImplementedError
