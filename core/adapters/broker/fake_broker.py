"""
FakeBroker — 本地模拟撮合，无需 QMT 权限，用于开发调试。
接口与 BaseBroker 完全一致，后续切换 QmtBroker 时策略层零改动。
"""
import logging
import math
import uuid
import copy
from datetime import datetime
from collections import OrderedDict
from decimal import Decimal, InvalidOperation
from core.adapters.broker.base import BaseBroker, BrokerPersistenceConflict, _trade_values
from core.shared.order_validation import normalize_a_share_symbol
from core.shared.trading_time import A_SHARE_TIMEZONE, market_date

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_FILLED = "filled"
STATUS_CANCELLED = "cancelled"
STATUS_REJECTED = "rejected"


def _market_now():
    return datetime.now(A_SHARE_TIMEZONE).replace(tzinfo=None)


def _read_share_count(value, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError("模拟账户存在无效股数，拒绝恢复")
    try:
        quantity = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("模拟账户存在无效股数，拒绝恢复") from None
    if (not quantity.is_finite() or quantity != quantity.to_integral_value() or
            quantity < 0 or (quantity == 0 and not allow_zero)):
        raise ValueError("模拟账户存在无效股数，拒绝恢复")
    return int(quantity)


class FakeBroker(BaseBroker):
    supported_price_types = ("market", "limit")

    def __init__(self, initial_cash=1000000, commission_rate=None,
                 stamp_duty=None, price_provider=None, price_limit_provider=None,
                 price_limits_configured=False):
        from core.shared import config

        self._cash = initial_cash
        self._frozen = 0  # 冻结资金（限价单）
        self._commission_rate = config.PORTFOLIO_COMMISSION_RATE if commission_rate is None else commission_rate
        self._stamp_duty = config.PORTFOLIO_STAMP_DUTY if stamp_duty is None else stamp_duty
        self._price_provider = price_provider or (lambda s: 0)
        self._price_limit_provider = price_limit_provider
        self._price_limits_configured = price_limit_provider is not None or price_limits_configured

        self._orders = OrderedDict()
        self._trades = []
        self._holdings = {}
        self._today_bought = {}
        self._frozen_shares = {}

        self._connected = False
        self._account_id = "FAKE-LOCAL"
        self._persistence_available = False

    def _capture_state(self):
        return copy.deepcopy((self._cash, self._frozen, self._orders, self._trades,
                              self._holdings, self._today_bought, self._frozen_shares))

    def _restore_state(self, state):
        (self._cash, self._frozen, self._orders, self._trades,
         self._holdings, self._today_bought, self._frozen_shares) = state

    def _write_account(self, cur):
        market_value = round(sum(row["market_value"] for row in self.get_positions()), 2)
        cur.execute(
            "INSERT INTO accounts (id,cash,frozen,market_value,total_assets) VALUES (%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE cash=VALUES(cash),frozen=VALUES(frozen),"
            "market_value=VALUES(market_value),total_assets=VALUES(total_assets)",
            (self.account_id, self._cash, self._frozen, market_value, round(self._cash + market_value, 2)),
        )

    def _persist_simulation(self, order=None, trade=None, previous_state=None):
        from core.data import Database

        if not Database.is_available():
            self._persistence_available = False
            return

        def operation(cur):
            self._write_account(cur)
            if order:
                cur.execute("SELECT account_id,broker,environment,source,symbol,side,quantity,price_type,price "
                            "FROM orders WHERE id=%s FOR UPDATE",
                            (order["id"],))
                existing = cur.fetchone()
                identity = (self.account_id, "fake", "simulation", "broker")
                if existing and tuple(existing.get(key) for key in
                                      ("account_id", "broker", "environment", "source")) != identity:
                    raise BrokerPersistenceConflict(f"模拟委托编号 {order['id']} 已属于其他账户或来源，拒绝覆盖")
                if existing and (existing.get("symbol"), existing.get("side"), int(existing.get("quantity")),
                                 existing.get("price_type"), Decimal(str(existing.get("price") or 0))) != (
                        order["symbol"], order["side"], int(order["quantity"]), order["price_type"],
                        Decimal(str(order.get("price") or 0))):
                    raise BrokerPersistenceConflict(f"模拟委托编号 {order['id']} 委托内容冲突，拒绝覆盖")
                cur.execute(
                    "INSERT INTO orders (id,symbol,name,side,quantity,price_type,price,filled_qty,filled_price,"
                    "status,created_at,updated_at,account_id,broker,environment,source) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON DUPLICATE KEY UPDATE filled_qty=VALUES(filled_qty),filled_price=VALUES(filled_price),"
                    "status=VALUES(status),updated_at=VALUES(updated_at)",
                    (order["id"], order["symbol"], order["name"], order["side"], order["quantity"],
                     order["price_type"], order.get("price") or 0, order["filled_qty"], order["filled_price"],
                     order["status"], order["created_at"], order["updated_at"], self.account_id,
                     "fake", "simulation", "broker"),
                )
            if trade:
                cur.execute("SELECT account_id,broker,environment,source,order_id,symbol,side,price,quantity,"
                            "amount,commission,stamp_duty,trade_time FROM trades WHERE id=%s FOR UPDATE",
                            (trade["id"],))
                existing = cur.fetchone()
                identity = (self.account_id, "fake", "simulation", "broker")
                if existing:
                    stored_identity = tuple(existing.get(key) for key in
                                            ("account_id", "broker", "environment", "source"))
                    if stored_identity != identity:
                        raise BrokerPersistenceConflict(f"模拟成交编号 {trade['id']} 已属于其他账户或来源，拒绝覆盖")
                    if _trade_values(existing) != _trade_values(trade):
                        raise BrokerPersistenceConflict(f"模拟成交编号 {trade['id']} 内容冲突，拒绝重复入账")
                    return
                cur.execute(
                    "INSERT INTO trades (id,order_id,symbol,name,side,price,quantity,amount,commission,stamp_duty,"
                    "trade_time,account_id,broker,environment,source) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    "%s,%s,%s,%s,%s)",
                    (trade["id"], trade["order_id"], trade["symbol"], trade["name"], trade["side"],
                     trade["price"], trade["quantity"], trade["amount"], trade["commission"],
                     trade["stamp_duty"], trade["trade_time"], self.account_id, "fake", "simulation", "broker"),
                )

        try:
            Database.transaction(operation)
            self._persistence_available = True
        except Exception as error:
            if previous_state is not None:
                self._restore_state(previous_state)
            self._persistence_available = False
            raise RuntimeError("模拟账户写入失败，状态已回滚") from error

    def restore(self, orders, trades):
        if isinstance(self._cash, bool) or isinstance(self._frozen, bool):
            raise ValueError("模拟账户资金快照无效，拒绝恢复")
        try:
            cash = Decimal(str(self._cash))
            frozen = Decimal(str(self._frozen))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("模拟账户资金快照无效，拒绝恢复") from None
        if (not cash.is_finite() or not frozen.is_finite() or cash < 0 or frozen < 0 or
                frozen > cash):
            raise ValueError("模拟账户现金或冻结资金快照无效，拒绝恢复")
        self._cash = float(cash)
        self._frozen = float(frozen)
        order_ids = set()
        order_filled = {}
        for row in orders:
            order_id = str(row.get("id") or "")
            if not order_id or order_id in order_ids:
                raise ValueError("模拟账户委托编号缺失或重复，拒绝恢复")
            order_ids.add(order_id)
            if row.get("status") not in (STATUS_PENDING, STATUS_FILLED, STATUS_CANCELLED, STATUS_REJECTED):
                raise ValueError("模拟账户委托状态无效，拒绝恢复")
            if row.get("side") not in ("buy", "sell"):
                raise ValueError("模拟账户委托方向无效，拒绝恢复")
            quantity = _read_share_count(row.get("quantity"))
            raw_filled_qty = row.get("filled_qty")
            filled_qty = _read_share_count(0 if raw_filled_qty is None else raw_filled_qty,
                                           allow_zero=True)
            if filled_qty > quantity:
                raise ValueError("模拟账户委托成交股数超过委托股数，拒绝恢复")
            if ((row["status"] == STATUS_FILLED and filled_qty != quantity) or
                    (row["status"] == STATUS_PENDING and filled_qty >= quantity)):
                raise ValueError("模拟账户委托状态与成交股数不一致，拒绝恢复")
            order_filled[order_id] = {"quantity": filled_qty, "symbol": row.get("symbol"),
                                      "side": row["side"], "trade_quantity": 0,
                                      "trade_amount": Decimal("0")}
        self._orders = OrderedDict((row["id"], row) for row in orders)
        self._trades = list(trades)
        self._holdings = {}
        self._today_bought = {}
        self._frozen_shares = {}
        today = market_date(_market_now())
        trade_ids = set()
        for trade in self._trades:
            if trade.get("side") not in ("buy", "sell"):
                raise ValueError("模拟账户成交方向无效，拒绝恢复")
            trade_id = str(trade.get("id") or "")
            if not trade_id or trade_id in trade_ids:
                raise ValueError("模拟账户成交编号缺失或重复，拒绝恢复")
            trade_ids.add(trade_id)
            symbol = trade["symbol"]
            holding = self._holdings.setdefault(symbol, {"shares": 0, "avg_cost": 0})
            quantity = _read_share_count(trade.get("quantity"))
            order_id = str(trade.get("order_id") or "")
            order = order_filled.get(order_id)
            if not order:
                raise ValueError("模拟账户成交缺少对应委托，拒绝恢复")
            if order["symbol"] != symbol or order["side"] != trade["side"]:
                raise ValueError("模拟账户成交与委托证券或方向不一致，拒绝恢复")
            try:
                price = Decimal(str(trade.get("price")))
                amount = Decimal(str(trade.get("amount")))
                commission = Decimal(str(trade.get("commission") or 0))
                stamp_duty = Decimal(str(trade.get("stamp_duty") or 0))
            except (InvalidOperation, TypeError, ValueError):
                raise ValueError("模拟账户成交金额或费用无效，拒绝恢复") from None
            if (not price.is_finite() or price <= 0 or not amount.is_finite() or amount < 0 or
                    not commission.is_finite() or commission < 0 or not stamp_duty.is_finite() or
                    stamp_duty < 0 or round(float(price) * quantity, 2) != round(float(amount), 2) or
                    (trade["side"] == "buy" and stamp_duty != 0)):
                raise ValueError("模拟账户成交金额或费用无效，拒绝恢复")
            order["trade_quantity"] += quantity
            order["trade_amount"] += price * quantity
            if trade["side"] == "buy":
                old_cost = holding["shares"] * holding["avg_cost"]
                total_cost = float(trade["amount"]) + float(trade.get("commission") or 0)
                holding["shares"] += quantity
                holding["avg_cost"] = (old_cost + total_cost) / holding["shares"]
                trade_time = trade["trade_time"]
                trade_at = trade_time if isinstance(trade_time, datetime) else datetime.fromisoformat(str(trade_time))
                trade_date = market_date(trade_at)
                if trade_date == today:
                    self._today_bought[symbol] = self._today_bought.get(symbol, 0) + quantity
            else:
                holding["shares"] -= quantity
                if holding["shares"] <= 0:
                    self._holdings.pop(symbol, None)
        for order_id, order in order_filled.items():
            if order["trade_quantity"] != order["quantity"]:
                raise ValueError("模拟账户订单成交股数与成交事件不一致，拒绝恢复")
            if order["quantity"]:
                row = self._orders[order_id]
                try:
                    filled_price = Decimal(str(row.get("filled_price") or 0))
                except (InvalidOperation, TypeError, ValueError):
                    raise ValueError("模拟账户委托成交价格无效，拒绝恢复") from None
                average_price = order["trade_amount"] / order["trade_quantity"]
                if (not filled_price.is_finite() or filled_price <= 0 or
                        round(float(filled_price), 3) != round(float(average_price), 3)):
                    raise ValueError("模拟账户委托成交价格与成交事件不一致，拒绝恢复")
        restored_frozen = 0.0
        for order in self._orders.values():
            if order["status"] != STATUS_PENDING:
                continue
            if order.get("side") not in ("buy", "sell"):
                raise ValueError("模拟账户待处理委托方向无效，拒绝恢复")
            quantity = _read_share_count(order.get("quantity"))
            raw_filled_qty = order.get("filled_qty")
            filled_qty = _read_share_count(0 if raw_filled_qty is None else raw_filled_qty, allow_zero=True)
            remaining = quantity - filled_qty
            if filled_qty > quantity or remaining <= 0:
                raise ValueError("模拟账户存在无效的待处理委托数量，拒绝恢复")
            if order["side"] == "buy":
                try:
                    price = Decimal(str(order.get("price")))
                except (InvalidOperation, TypeError, ValueError):
                    raise ValueError("模拟账户待处理买单价格无效，拒绝恢复") from None
                if not price.is_finite() or price <= 0:
                    raise ValueError("模拟账户待处理买单价格无效，拒绝恢复")
                amount = float(price) * remaining
                order["reserved_cash"] = round(amount + max(5, amount * self._commission_rate), 2)
                restored_frozen += order["reserved_cash"]
            else:
                symbol = order["symbol"]
                order["reserved_shares"] = remaining
                self._frozen_shares[symbol] = self._frozen_shares.get(symbol, 0) + remaining
        if round(restored_frozen, 2) != round(self._frozen, 2):
            raise ValueError("模拟账户冻结资金与待处理买单预留不一致，拒绝恢复")
        for symbol, reserved_shares in self._frozen_shares.items():
            holding = self._holdings.get(symbol)
            sellable = ((holding["shares"] if holding else 0) - self._today_bought.get(symbol, 0))
            if reserved_shares > sellable:
                raise ValueError("模拟账户待处理卖单超过可卖持仓，拒绝恢复")
        self._persistence_available = True

    def connect(self):
        self._connected = True
        logger.info(f"FakeBroker 已连接, 账户: {self._account_id}")
        return True

    def disconnect(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    @property
    def account_id(self):
        return self._account_id

    def place_order(self, symbol, name, side, quantity, price_type, price=None):
        if not self._connected:
            raise RuntimeError("Broker 未连接")
        symbol = normalize_a_share_symbol(symbol)
        if side not in ("buy", "sell") or price_type not in ("market", "limit"):
            raise ValueError("委托方向或价格类型无效")
        if isinstance(quantity, bool):
            raise ValueError("委托股数无效")
        try:
            quantity_value = float(quantity)
            price = float(price or 0)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("委托股数或价格无效") from None
        if (not math.isfinite(quantity_value) or quantity_value != int(quantity_value) or
                quantity_value <= 0 or quantity_value > 100000000):
            raise ValueError("委托股数无效")
        quantity = int(quantity_value)
        if side == "buy" and quantity % 100:
            raise ValueError("A股买入股数必须为100股整数倍")
        if (price_type == "limit" and
                (not math.isfinite(price) or price <= 0 or price >= 1000000000 or
                 round(price, 3) != price)):
            raise ValueError("限价须大于0且最多三位小数")
        if price_type == "market" and price != 0:
            raise ValueError("市价单不能带入限价")

        previous_state = self._capture_state()
        order_id = _market_now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6].upper()

        order = {
            "id": order_id,
            "symbol": symbol,
            "name": name,
            "side": side,
            "quantity": int(quantity),
            "price_type": price_type,
            "price": price,
            "filled_qty": 0,
            "filled_price": 0,
            "status": STATUS_PENDING,
            "created_at": _market_now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": _market_now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        quote = self._price_provider(symbol)
        quote_bounds = quote if isinstance(quote, dict) else None
        current_price = quote.get("price", 0) if quote_bounds is not None else quote
        try:
            current_price = float(current_price)
        except (TypeError, ValueError, OverflowError):
            current_price = 0
        if not math.isfinite(current_price) or current_price <= 0:
            raise RuntimeError("模拟委托缺少有效实时价格，未创建委托")
        est_price = current_price if price_type == "market" else price
        est_cost = round(est_price * quantity, 2)
        estimated_commission = round(max(5, est_cost * self._commission_rate), 2)
        estimated_stamp = round(est_cost * self._stamp_duty, 2) if side == "sell" else 0
        estimated_total = round(est_cost + estimated_commission + estimated_stamp, 2)
        if side == "buy" and estimated_total > self._cash - self._frozen:
            order["status"] = STATUS_REJECTED
            self._orders[order_id] = order
            logger.warning(f"资金不足拒绝: {order_id}")
            self._persist_simulation(order, previous_state=previous_state)
            return order_id
        if side == "sell":
            holding = self._holdings.get(symbol)
            sellable = ((holding["shares"] if holding else 0) - self._today_bought.get(symbol, 0)
                        - self._frozen_shares.get(symbol, 0))
            if quantity > sellable:
                order["status"] = STATUS_REJECTED
                self._orders[order_id] = order
                self._persist_simulation(order, previous_state=previous_state)
                return order_id

        limit_bounds = None
        limits_available = False
        if self._price_limit_provider or quote_bounds is not None:
            try:
                bounds = self._price_limit_provider(symbol) if self._price_limit_provider else quote_bounds
                upper = Decimal(str(bounds.get("limit_up"))) if isinstance(bounds, dict) else Decimal("NaN")
                lower = Decimal(str(bounds.get("limit_down"))) if isinstance(bounds, dict) else Decimal("NaN")
                if upper.is_finite() and lower.is_finite() and lower > 0 and upper >= lower:
                    limit_bounds = (lower, upper)
                    limits_available = True
            except (InvalidOperation, TypeError, ValueError, AttributeError, OverflowError):
                pass

        trade = None
        limit_price_outside = (limit_bounds is not None and price_type == "limit" and
                               (Decimal(str(price)) > limit_bounds[1] or Decimal(str(price)) < limit_bounds[0]))
        if limit_price_outside:
            order["status"] = STATUS_REJECTED
        else:
            marketable = (price_type == "market" or
                          (side == "buy" and current_price <= price) or
                          (side == "sell" and current_price >= price))
            at_unfillable_limit = (limit_bounds is not None and
                                   ((side == "buy" and Decimal(str(current_price)) >= limit_bounds[1]) or
                                    (side == "sell" and Decimal(str(current_price)) <= limit_bounds[0])))
            if marketable and limits_available and not at_unfillable_limit:
                trade = self._fill_order(order, current_price)
            elif price_type == "market":
                order["status"] = STATUS_REJECTED
            else:
                order["reserved_cash"] = estimated_total if side == "buy" else 0
                self._frozen += order["reserved_cash"]
                if side == "sell":
                    order["reserved_shares"] = quantity
                    self._frozen_shares[symbol] = self._frozen_shares.get(symbol, 0) + quantity

        self._orders[order_id] = order
        if trade:
            self._trades.append(trade)
        self._persist_simulation(order, trade, previous_state)
        logger.info(f"下单: {symbol} {side} {quantity}股 @ {price or '市价'} → {order['status']}")

        return order_id

    def cancel_order(self, order_id):
        order = self._orders.get(order_id)
        if not order or order["status"] != STATUS_PENDING:
            return False
        previous_state = self._capture_state()
        self._frozen = max(0, self._frozen - float(order.get("reserved_cash") or 0))
        reserved_shares = int(order.get("reserved_shares") or 0)
        if reserved_shares:
            symbol = order["symbol"]
            self._frozen_shares[symbol] = max(0, self._frozen_shares.get(symbol, 0) - reserved_shares)
        order["status"] = STATUS_CANCELLED
        order["updated_at"] = _market_now().strftime("%Y-%m-%d %H:%M:%S")
        self._persist_simulation(order, previous_state=previous_state)
        return True

    def get_order(self, order_id):
        return self._orders.get(order_id)

    def get_orders(self, status=None):
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o["status"] == status]
        return orders

    def get_trades(self, symbol=None):
        if symbol:
            return [t for t in self._trades if t["symbol"] == symbol]
        return list(self._trades)

    def get_positions(self):
        result = []
        for symbol, h in self._holdings.items():
            if h["shares"] <= 0:
                continue
            quote = self._price_provider(symbol)
            price = quote.get("price", 0) if isinstance(quote, dict) else quote
            try:
                price = float(price or 0)
            except (TypeError, ValueError, OverflowError):
                price = 0
            market_value = h["shares"] * price
            cost = h["shares"] * h["avg_cost"]
            result.append({
                "symbol": symbol, "shares": h["shares"],
                "avg_cost": round(h["avg_cost"], 3),
                "current_price": round(price, 3),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(market_value - cost, 2),
            })
        return result

    def get_account(self):
        positions = self.get_positions()
        missing_codes = [p["symbol"] for p in positions if p["current_price"] <= 0]
        total_market = sum(p["market_value"] for p in positions)
        total_pnl = sum(p["unrealized_pnl"] for p in positions)
        observed = _market_now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "account_id": self._account_id,
            "connected": self._connected,
            "status": "unavailable" if not self._connected else (
                "degraded" if (missing_codes or not self._persistence_available or
                               not self._price_limits_configured) else "fresh"),
            "source": "fake",
            "persistence": "available" if self._persistence_available else "memory_only",
            "price_limit_status": "configured" if self._price_limits_configured else "unavailable",
            "as_of": observed if self._connected else None,
            "missing_codes": missing_codes,
            "cash": round(self._cash, 2) if self._connected else None,
            "frozen": round(self._frozen, 2) if self._connected else None,
            "available": round(self._cash - self._frozen, 2) if self._connected else None,
            "market_value": round(total_market, 2) if self._connected and not missing_codes else None,
            "total_assets": round(self._cash + total_market, 2) if self._connected and not missing_codes else None,
            "unrealized_pnl": round(total_pnl, 2) if self._connected and not missing_codes else None,
            "updated_at": observed if self._connected else None,
        }

    def add_cash(self, amount):
        amount = self._cash_amount(amount)
        if amount <= 0:
            raise ValueError("充值金额必须大于0")
        previous_state = self._capture_state()
        self._cash += amount
        self._save_account(previous_state)
        return self.get_account()

    def set_cash(self, amount):
        amount = self._cash_amount(amount)
        if amount < self._frozen:
            raise ValueError("资金不能低于已冻结金额")
        if amount < 0:
            raise ValueError("资金不能为负")
        previous_state = self._capture_state()
        self._cash = amount
        self._save_account(previous_state)
        return self.get_account()

    def withdraw_cash(self, amount):
        amount = self._cash_amount(amount)
        if amount <= 0:
            raise ValueError("出金金额必须大于0")
        if amount > self._cash - self._frozen:
            raise ValueError(f"可用资金不足 (可用: {self._cash - self._frozen:.2f})")
        previous_state = self._capture_state()
        self._cash -= amount
        self._save_account(previous_state)
        return self.get_account()

    @staticmethod
    def _cash_amount(amount):
        if isinstance(amount, bool):
            raise ValueError("资金金额无效")
        try:
            value = float(amount)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("资金金额无效") from None
        if not math.isfinite(value):
            raise ValueError("资金金额无效")
        return round(value, 2)

    def _fill_order(self, order, fill_price):
        if not math.isfinite(float(fill_price)) or float(fill_price) <= 0:
            order["status"] = STATUS_REJECTED
            return
        qty = order["quantity"]
        cost = round(float(fill_price) * qty, 2)
        commission = round(max(5, cost * self._commission_rate), 2)
        stamp = round(cost * self._stamp_duty, 2) if order["side"] == "sell" else 0
        total_cost = round(cost + commission + stamp, 2)

        if order["side"] == "buy":
            if total_cost > self._cash - self._frozen:
                order["status"] = STATUS_REJECTED
                return
            self._cash = round(self._cash - total_cost, 2)
            h = self._holdings.get(order["symbol"], {"shares": 0, "avg_cost": 0})
            old_total = h["shares"] * h["avg_cost"]
            h["shares"] += qty
            h["avg_cost"] = (old_total + total_cost) / h["shares"] if h["shares"] > 0 else fill_price
            self._holdings[order["symbol"]] = h
            self._today_bought[order["symbol"]] = self._today_bought.get(order["symbol"], 0) + qty
        else:
            h = self._holdings.get(order["symbol"])
            sellable = ((h["shares"] if h else 0) - self._today_bought.get(order["symbol"], 0)
                        - self._frozen_shares.get(order["symbol"], 0))
            if not h or sellable < qty:
                order["status"] = STATUS_REJECTED
                return
            h["shares"] -= qty
            self._cash = round(self._cash + cost - commission - stamp, 2)

        order["filled_qty"] = qty
        order["filled_price"] = fill_price
        order["status"] = STATUS_FILLED
        market_now = _market_now()
        order["updated_at"] = market_now.strftime("%Y-%m-%d %H:%M:%S")

        trade = {
            "id": "T" + order["id"][:14] + uuid.uuid4().hex[:4].upper(),
            "order_id": order["id"],
            "symbol": order["symbol"],
            "name": order["name"],
            "side": order["side"],
            "price": round(fill_price, 3),
            "quantity": qty,
            "amount": round(cost, 2),
            "commission": round(commission, 2),
            "stamp_duty": round(stamp, 2),
            "trade_time": market_now.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return trade

    def _save_account(self, previous_state=None):
        self._persist_simulation(previous_state=previous_state)
