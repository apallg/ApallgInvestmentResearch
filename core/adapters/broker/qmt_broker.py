"""
QmtBroker — 对接迅投 miniQMT 真实交易接口。
封装 xtquant.XtQuantTrader，实现 BaseBroker 接口。
需要：miniQMT 已启动、xtquant 已安装、账户已登录。
"""
import logging
import math
import time
import threading
from datetime import datetime
from decimal import Decimal, InvalidOperation
from core.adapters.broker.base import BaseBroker
from core.shared.order_validation import normalize_a_share_symbol

logger = logging.getLogger(__name__)

STATUS_MAP = {48: "pending", 49: "pending", 50: "pending", 51: "pending",
              52: "partial", 53: "cancelled", 54: "cancelled",
              55: "partial", 56: "filled", 57: "rejected"}

BUY_ORDER_TYPES = (23,)
SELL_ORDER_TYPES = (24,)
FIX_PRICE_TYPE = 11
MARKET_PRICE_TYPES = frozenset((5, 12, 18, 19, 20, 21, 22, 23, 24, 42, 43, 44, 45, 46, 47, 48))


def _price_type(raw):
    if raw == FIX_PRICE_TYPE:
        return "limit"
    if raw in MARKET_PRICE_TYPES:
        return "market"
    return "unknown"


class QmtBroker(BaseBroker):
    def __init__(self, userdata_dir, account, session_id=123456, commission_rate=0.0003, stamp_duty=0.001):
        self._userdata_dir = userdata_dir
        self._account = str(account)
        self._session_id = int(session_id)
        self._commission_rate = commission_rate
        self._stamp_duty = stamp_duty
        self._stock_account = None

        self._trader = None
        self._callback = None
        self._connected = False
        self._account_id = f"QMT-{self._account}"

        self._orders = {}
        self._trades = []
        self._positions = {}
        self._asset = None
        self._asset_observed_at = None
        self._asset_sync_ok = False
        self._positions_observed_at = None

        self._lock = threading.Lock()
        self._last_sync = {}  # method → timestamp
        self._event_thread = None
        self._report_persistence_error = None
        self._report_failed_orders = set()
        self._report_query_errors = set()
        self._last_order_query_count = None
        self._last_trade_query_count = None

    def _record_order_report(self, order, source, raw_status, raw_order=None):
        try:
            if str(self._attr(raw_order, "account_id", "")) != self._account:
                raise RuntimeError("QMT 委托回报资金账号不一致或缺失")
            from core.data.repos.broker_order_report import BrokerOrderReportRepo
            fields = ("account_id", "order_id", "stock_code", "order_type", "order_status", "order_volume",
                      "price_type", "price", "traded_volume", "traded_price", "order_time",
                      "order_sysid", "status_msg")
            raw_fields = {key: self._attr(raw_order, key) for key in fields}
            result = BrokerOrderReportRepo.ingest(
                "qmt", self.account_id, order, source, raw_status, raw_fields,
            )
            if result["disposition"] in ("conflict", "unknown"):
                logger.error("QMT 回报需人工核对: 委托 %s, 状态 %s", order.get("id"), result["disposition"])
            with self._lock:
                self._report_failed_orders.discard(str(order.get("id") or ""))
            return result
        except Exception as error:
            with self._lock:
                self._report_failed_orders.add(str(order.get("id") or ""))
                self._report_persistence_error = f"{type(error).__name__}: {error}"
            logger.error("QMT 回报写入失败，需从券商重新查询委托: %s", error)
            return None

    def _persist_qmt_order(self, order):
        key = str(order.get("id") or "")
        try:
            self._persist_order(order, required=True)
        except Exception as error:
            with self._lock:
                self._report_failed_orders.add(key)
                self._report_persistence_error = f"{type(error).__name__}: {error}"
            logger.error("QMT 委托投影写入失败，等待持久化回报重放: %s", error)
            return False
        with self._lock:
            self._report_failed_orders.discard(key)
            if not self._report_failed_orders and not self._report_query_errors:
                self._report_persistence_error = None
        return True

    def _record_trade_report(self, trade, source, raw_trade=None):
        key = f"trade:{trade.get('id') or ''}"
        try:
            if str(self._attr(raw_trade, "account_id", "")) != self._account:
                raise RuntimeError("QMT 成交回报资金账号不一致或缺失")
            from core.data.repos.broker_trade_report import BrokerTradeReportRepo
            fields = ("account_id", "traded_id", "order_id", "stock_code", "order_type", "traded_price",
                      "traded_volume", "traded_amount", "traded_time", "commission")
            raw_fields = {field: self._attr(raw_trade, field) for field in fields}
            result = BrokerTradeReportRepo.ingest("qmt", self.account_id, trade, source, raw_fields)
            if result["status"] == "conflict":
                logger.error("QMT 成交回报内容冲突: %s", trade.get("id"))
            return result
        except Exception as error:
            with self._lock:
                self._report_failed_orders.add(key)
                self._report_persistence_error = f"{type(error).__name__}: {error}"
            logger.error("QMT 成交回报写入失败，需从券商重新查询成交: %s", error)
            return None

    def _persist_qmt_trade(self, trade):
        key = f"trade:{trade.get('id') or ''}"
        try:
            self._persist_trade(trade, required=True)
        except Exception as error:
            with self._lock:
                self._report_failed_orders.add(key)
                self._report_persistence_error = f"{type(error).__name__}: {error}"
            logger.error("QMT 成交投影写入失败，等待券商查询重放: %s", error)
            return False
        with self._lock:
            self._report_failed_orders.discard(key)
            if not self._report_failed_orders and not self._report_query_errors:
                self._report_persistence_error = None
        return True

    def connect(self):
        import os
        if not self._account.strip():
            raise RuntimeError("QMT 资金账号未配置")
        if not isinstance(self._userdata_dir, (str, os.PathLike)) or not os.path.isdir(self._userdata_dir):
            raise RuntimeError("QMT userdata_mini 目录不存在或不可用")
        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError:
            raise RuntimeError(
                "xtquant 未安装，请从 QMT 安装目录安装：\n"
                "pip install {QMT_DIR}/bin.x64/xtquant-*.whl\n"
                "或参考 https://dict.thinktrader.net"
            )

        # 清理可能残留的会话锁文件（避免 connect 返回 -1）
        self._cleanup_stale_locks()

        self._stock_account = StockAccount(self._account, 'STOCK')
        self._trader = XtQuantTrader(self._userdata_dir, self._session_id)
        self._callback = _TraderCallback(self)
        self._trader.register_callback(self._callback)
        self._trader.start()
        connect_result = self._trader.connect()

        if connect_result == 0:
            self._connected = True
            self._sync_positions()
            self._sync_asset()
            self._sync_orders()
            self._sync_trades()
            # 启动后台线程处理 QMT 事件循环（心跳 + 回调）
            self._event_thread = threading.Thread(
                target=self._trader.run_forever, daemon=True, name="qmt-event"
            )
            self._event_thread.start()
            logger.info(f"QmtBroker 已连接, 账户: {self._stock_account}, 资产: {self._asset}")
        else:
            logger.error(f"QmtBroker 连接失败, 返回码: {connect_result}")
            return False

        return True

    def replay_order_reports(self):
        if not self._connected or not self._trader:
            raise RuntimeError("QMT 未连接")
        self._last_sync.pop("orders", None)
        self._last_sync.pop("trades", None)
        self._sync_orders()
        self._sync_trades()
        with self._lock:
            report_error = self._report_persistence_error
        if report_error:
            raise RuntimeError("QMT 委托查询或回报持久化失败")
        from core.data.repos.broker_order_report import BrokerOrderReportRepo
        from core.data.repos.broker_trade_report import BrokerTradeReportRepo
        return {"observed_orders": self._last_order_query_count,
                "observed_trades": self._last_trade_query_count,
                "unresolved_reports": BrokerOrderReportRepo.unresolved_count("qmt", self.account_id),
                "trade_report_health": BrokerTradeReportRepo.get_health("qmt", self.account_id)}

    def get_trade_reconciliation_evidence(self, trade_id):
        if not self._connected or not self._trader:
            raise RuntimeError("QMT 未连接")
        trades = self._trader.query_stock_trades(self._stock_account)
        if trades is None:
            raise RuntimeError("QMT 成交查询未返回数据")
        trades = list(trades)
        if any(str(self._attr(trade, "account_id", "")) != self._account for trade in trades):
            raise RuntimeError("QMT 成交资金账号不一致或缺失")
        matches = [trade for trade in trades if str(self._attr(trade, "traded_id", "")) == str(trade_id)]
        if len(matches) != 1:
            raise ValueError("券商当前逐笔成交清单缺少唯一对应记录")
        trade = matches[0]
        order_type = self._attr(trade, "order_type", 0)
        return {
            "id": str(self._attr(trade, "traded_id", "")),
            "order_id": str(self._attr(trade, "order_id", "")),
            "symbol": self._attr(trade, "stock_code", ""),
            "side": "buy" if order_type in BUY_ORDER_TYPES else "sell" if order_type in SELL_ORDER_TYPES else "unknown",
            "quantity": self._attr(trade, "traded_volume", 0),
            "price": round(float(self._attr(trade, "traded_price", 0)), 3),
            "amount": round(float(self._attr(trade, "traded_amount", 0)), 2),
            "commission": self._attr(trade, "commission", None),
            "stamp_duty": None,
        }

    def _cleanup_stale_locks(self):
        """清理上次会话可能残留的锁文件，避免 session_id 被锁定"""
        import os
        prefixes = (
            f"down_queue_win_{self._session_id}",
            f"up_queue_win_{self._session_id}",
            f"lock_down_queue_win_{self._session_id}",
            f"lock_up_queue_win_{self._session_id}",
        )
        try:
            for name in os.listdir(self._userdata_dir):
                if name.startswith(prefixes):
                    try:
                        os.remove(os.path.join(self._userdata_dir, name))
                        logger.debug(f"清理残留锁文件: {name}")
                    except Exception:
                        pass
        except Exception:
            pass

    def disconnect(self):
        self._connected = False
        if self._trader:
            self._trader.stop()
            self._trader = None
        self._event_thread = None

    @property
    def connected(self):
        return self._connected

    @property
    def account_id(self):
        return self._account_id

    @staticmethod
    def _normalize_symbol(symbol):
        """补齐交易所后缀：600879 → 600879.SH, 000001 → 000001.SZ"""
        symbol = symbol.strip()
        if "." in symbol:
            return symbol
        from core.data.sources.miniqmt import MiniQmtSource
        return MiniQmtSource.to_xtcode(symbol)

    def place_order(self, symbol, name, side, quantity, price_type, price=None):
        from core.shared.order_validation import validate_real_limit_order
        symbol, quantity, price = validate_real_limit_order(symbol, side, quantity, price_type, price)
        if not self._connected:
            raise RuntimeError("QMT Broker 未连接")

        try:
            from xtquant import xtconstant
        except ImportError:
            raise RuntimeError("xtquant 未安装")

        symbol = self._normalize_symbol(symbol)

        if side == "buy":
            order_type = xtconstant.STOCK_BUY
        elif side == "sell":
            order_type = xtconstant.STOCK_SELL
        else:
            raise ValueError(f"不支持的交易方向: {side}")

        price_type_xt = xtconstant.FIX_PRICE

        try:
            order_id = self._trader.order_stock(
                self._stock_account, symbol, order_type, int(quantity),
                price_type_xt, float(price), "Apallg投研", name or symbol
            )
        except Exception as error:
            logger.exception("QMT委托调用异常，券商接单状态不明确")
            raise RuntimeError("QMT委托调用异常，结果不明确；须先查询券商委托，禁止直接重试") from error

        if isinstance(order_id, bool) or not isinstance(order_id, int):
            raise RuntimeError("QMT返回无效委托编号，结果不明确；须先查询券商委托，禁止直接重试")

        if order_id < 0:
            logger.error(f"QMT 下单失败: {symbol} {side} {quantity}股, 返回码: {order_id}")
            raise RuntimeError(f"QMT 下单失败, 返回码: {order_id}")

        order = {
            "id": str(order_id),
            "symbol": symbol,
            "name": name,
            "side": side,
            "quantity": int(quantity),
            "price_type": price_type,
            "price": price,
            "broker_request": {"symbol": symbol, "side": side, "quantity": quantity,
                               "price_type": "FIX_PRICE", "price": price},
            "broker_response": {"order_id": str(order_id)},
            "filled_qty": 0,
            "filled_price": 0,
            "status": "pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        submission = {
            "client_request": {"symbol": symbol, "side": side, "quantity": int(quantity),
                               "price_type": price_type, "price": price},
            "broker_request": {"method": "order_stock", "account": self._account, "symbol": symbol,
                               "order_type": order_type, "quantity": int(quantity),
                               "price_type": price_type_xt, "price_type_name": "FIX_PRICE",
                               "price": float(price), "strategy_name": "Apallg投研", "order_remark": name or symbol},
            "broker_response": {"order_id": str(order_id)},
        }
        try:
            from core.data.repos.broker_order_report import BrokerOrderReportRepo
            submission = BrokerOrderReportRepo.record_submission(
                "qmt", self.account_id, order, submission["client_request"],
                submission["broker_request"], submission["broker_response"],
            )
        except Exception as error:
            with self._lock:
                self._report_persistence_error = f"{type(error).__name__}: {error}"
                self._report_failed_orders.add(str(order_id))
            raise RuntimeError("QMT已返回委托编号，但请求/响应证据未持久化；请先查询券商，禁止直接重试") from error
        order["submission"] = submission

        with self._lock:
            self._orders[str(order_id)] = order

        self._persist_order(order)
        logger.info(f"QMT下单: {symbol} {side} {quantity}股 @ {price if price else '市价'} → {order_id}")
        return str(order_id)

    def cancel_order(self, order_id):
        if not self._connected:
            return False
        result = self._trader.cancel_order(int(order_id))
        return result == 0

    def get_order(self, order_id):
        self._sync_orders()
        with self._lock:
            if not self._connected or not self._trader or self._report_persistence_error:
                raise RuntimeError("QMT 委托查询不可用")
            return self._orders.get(str(order_id))

    def get_orders(self, status=None):
        self._sync_orders()
        with self._lock:
            if not self._connected or not self._trader or self._report_persistence_error:
                raise RuntimeError("QMT 委托查询或回报持久化不可用")
            orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o["status"] == status]
        return orders

    def get_trades(self, symbol=None):
        self._sync_trades()
        with self._lock:
            if not self._connected or not self._trader or self._report_persistence_error:
                raise RuntimeError("QMT 成交查询或回报持久化不可用")
            trades = list(self._trades)
        if symbol:
            trades = [t for t in trades if t["symbol"] == symbol]
        return trades

    def get_positions(self):
        self._sync_positions()
        with self._lock:
            if (not self._connected or not self._trader or self._positions_observed_at is None or
                    not 0 <= (datetime.now() - self._positions_observed_at).total_seconds() <= 5):
                raise RuntimeError("QMT 持仓查询不可用")
            result = []
            for symbol, p in self._positions.items():
                result.append({
                    "symbol": self._attr(p, "stock_code", symbol),
                    "shares": int(Decimal(str(self._attr(p, "volume")))),
                    "avg_cost": round(float(self._attr(p, "avg_price", 0)), 3),
                    "current_price": round(float(self._attr(p, "last_price", 0)), 3),
                    "market_value": round(float(self._attr(p, "market_value", 0)), 2),
                    "unrealized_pnl": round(float(self._attr(p, "income", 0)), 2),
                })
            return result

    def get_account(self):
        self._sync_asset()
        self._sync_positions()
        with self._lock:
            asset = self._asset
            positions_snapshot = dict(self._positions)
            as_of = self._asset_observed_at
            asset_ready = self._asset_sync_ok and as_of is not None and (datetime.now() - as_of).total_seconds() <= 5
            positions_ready = self._positions_observed_at is not None and (datetime.now() - self._positions_observed_at).total_seconds() <= 5
        if not self._connected or not asset_ready:
            observed = as_of.strftime("%Y-%m-%d %H:%M:%S") if as_of else None
            return {
                "account_id": self._account_id,
                "connected": self._connected,
                "status": "stale" if self._connected and as_of else "unavailable",
                "source": "qmt",
                "as_of": observed,
                "cash": None, "frozen": None, "available": None,
                "market_value": None, "total_assets": None,
                "unrealized_pnl": None,
                "updated_at": observed,
            }
        available = float(self._attr(asset, "cash", 0))
        frozen = float(self._attr(asset, "frozen_cash", 0))
        cash = available + frozen
        market_value = float(self._attr(asset, "market_value", 0))
        total = float(self._attr(asset, "total_asset", cash + market_value))
        unrealized = None
        if positions_ready:
            values = []
            for position in positions_snapshot.values():
                try:
                    market_value = float(self._attr(position, "market_value"))
                    avg_price = float(self._attr(position, "avg_price"))
                    volume = float(self._attr(position, "volume"))
                    value = market_value - avg_price * volume
                    if not all(math.isfinite(number) for number in (market_value, avg_price, volume, value)):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    values = None
                    break
                values.append(value)
            if values is not None:
                unrealized = sum(values)
        observed = as_of.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "account_id": self._account_id,
            "connected": True,
            "status": "fresh",
            "source": "qmt",
            "as_of": observed,
            "cash": round(cash, 2),
            "frozen": round(frozen, 2),
            "available": round(available, 2),
            "market_value": round(market_value, 2),
            "total_assets": round(total, 2),
            "unrealized_pnl": round(unrealized, 2) if unrealized is not None else None,
            "updated_at": observed,
        }

    def get_reconciliation_snapshot(self, check_report_integrity=True):
        if not self._connected or not self._trader:
            raise RuntimeError("QMT 未连接")
        with self._lock:
            report_error = self._report_persistence_error
        if report_error:
            raise RuntimeError("QMT 回报持久化失败，须先重放委托与成交查询")
        from core.data.repos.broker_order_report import BrokerOrderReportRepo
        from core.data.repos.broker_trade_report import BrokerTradeReportRepo
        if check_report_integrity:
            if BrokerOrderReportRepo.unresolved_count("qmt", self.account_id):
                raise RuntimeError("QMT 存在未核对的异常委托回报")
        trade_health = BrokerTradeReportRepo.get_health("qmt", self.account_id)
        if check_report_integrity and trade_health["conflicts"]:
            raise RuntimeError("QMT 存在同一成交编号的矛盾回报")
        from core.shared.trading_time import A_SHARE_TIMEZONE, normalize_a_share_datetime

        snapshot_started_at = datetime.now(A_SHARE_TIMEZONE)
        account = self.get_account()
        if account.get("status") != "fresh":
            raise RuntimeError("QMT 资产快照不可用")
        positions = self._trader.query_stock_positions(self._stock_account)
        orders = self._trader.query_stock_orders(self._stock_account)
        if positions is None or orders is None:
            raise RuntimeError("QMT 持仓或委托查询不可用")
        positions = list(positions)
        orders = list(orders)
        if any(str(self._attr(item, "account_id", "")) != self._account
               for item in (*positions, *orders)):
            raise RuntimeError("QMT 持仓或委托资金账号不一致或缺失")
        normalized_orders = [
            {"id": str(self._attr(o, "order_id", "")),
             "symbol": self._attr(o, "stock_code", ""),
             "side": ("buy" if self._attr(o, "order_type", 0) in BUY_ORDER_TYPES else
                      "sell" if self._attr(o, "order_type", 0) in SELL_ORDER_TYPES else "unknown"),
             "status": STATUS_MAP.get(self._attr(o, "order_status", 0), "unknown"),
             "quantity": self._attr(o, "order_volume", 0),
             "price_type": _price_type(self._attr(o, "price_type", 0)),
             "price": self._attr(o, "price", None),
             "filled_qty": self._attr(o, "traded_volume", 0)}
            for o in orders
        ]
        snapshot_completed_at = datetime.now(A_SHARE_TIMEZONE)
        try:
            asset_observed_at = normalize_a_share_datetime(datetime.fromisoformat(str(account["as_of"])))
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("QMT 资产快照时间不可核验") from None
        if (snapshot_completed_at - snapshot_started_at).total_seconds() > 5 or (
                snapshot_completed_at - asset_observed_at).total_seconds() > 5 or (
                snapshot_completed_at - asset_observed_at).total_seconds() < -1:
            raise RuntimeError("QMT 对账快照跨时或资产已过期")
        return {
            "account": account,
            "snapshot_started_at": snapshot_started_at.isoformat(timespec="seconds"),
            "snapshot_completed_at": snapshot_completed_at.isoformat(timespec="seconds"),
            "identity_verified": True,
            "trade_report_health": trade_health,
            "positions": [
                {"symbol": self._attr(p, "stock_code", ""),
                 "shares": self._attr(p, "volume", None),
                 "sellable": self._attr(p, "can_use_volume", None)}
                for p in positions
            ],
            "orders": normalized_orders,
        }

    @staticmethod
    def _attr(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _should_sync(self, method, min_interval=1.0):
        now = time.time()
        last = self._last_sync.get(method, 0)
        if now - last < min_interval:
            return False
        self._last_sync[method] = now
        return True

    def _sync_orders(self):
        if not self._trader or not self._connected:
            return
        if not self._should_sync("orders", 1.0):
            return
        try:
            orders = self._trader.query_stock_orders(self._stock_account)
            if orders is None:
                raise RuntimeError("QMT 委托查询未返回数据")
            self._last_order_query_count = len(orders)
            from core.data.repos.broker_order_report import BrokerOrderReportRepo
            submission_metadata = BrokerOrderReportRepo.list_submission_metadata(
                "qmt", self.account_id,
                [str(self._attr(order, "order_id", "")) for order in orders],
            )
            observations = []
            with self._lock:
                for o in orders:
                    oid = str(self._attr(o, "order_id", ""))
                    order_type = self._attr(o, "order_type", 0)
                    order_status = self._attr(o, "order_status", 0)
                    existing = self._orders.get(oid, {})
                    candidate = {
                        "id": oid,
                        "symbol": self._attr(o, "stock_code", existing.get("symbol", "")),
                        "name": existing.get("name", ""),
                        "side": "buy" if order_type in BUY_ORDER_TYPES else "sell" if order_type in SELL_ORDER_TYPES else "unknown",
                        "quantity": self._attr(o, "order_volume", 0),
                        "price_type": _price_type(self._attr(o, "price_type", 0)),
                        "price": float(self._attr(o, "price", 0)),
                        "filled_qty": self._attr(o, "traded_volume", 0),
                        "filled_price": float(self._attr(o, "traded_price", 0)),
                        "status": STATUS_MAP.get(order_status, "unknown"),
                        "created_at": existing.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "submission": submission_metadata.get(oid) or existing.get("submission"),
                    }
                    observations.append((candidate, order_status, o))
            results = [(item, self._record_order_report(item, "query", raw_status, raw_order))
                       for item, raw_status, raw_order in observations]
            for item, result in results:
                if not result or result["disposition"] in ("conflict", "unknown"):
                    continue
                item["status"] = result["status"]
                item["filled_qty"] = result["filled_qty"]
                if not self._persist_qmt_order(item):
                    raise RuntimeError("QMT 委托投影未持久化")
                with self._lock:
                    self._orders[item["id"]] = item
            from core.data.repos.broker_order_report import BrokerOrderReportRepo
            durable_orders = BrokerOrderReportRepo.list_current_states("qmt", self.account_id)
            observed_order_ids = {
                str(item["id"]) for item, result in results
                if result and result.get("disposition") in ("applied", "duplicate")
            }
            unprojected_ids = {
                str(row["order_id"])
                for row in BrokerOrderReportRepo.list_unprojected("qmt", self.account_id)
            }
            for item in durable_orders:
                item["recovered_from_persisted_report"] = str(item["id"]) not in observed_order_ids
                if str(item["id"]) in unprojected_ids and not self._persist_qmt_order(item):
                    raise RuntimeError("QMT 持久化委托状态投影未恢复")
            with self._lock:
                for item in durable_orders:
                    self._orders[str(item["id"])] = item
            unresolved_reports = BrokerOrderReportRepo.unresolved_count("qmt", self.account_id)
            if unresolved_reports:
                raise RuntimeError(f"QMT 存在 {unresolved_reports} 笔未核对的异常委托回报")
            with self._lock:
                self._report_query_errors.discard("orders")
                if not self._report_failed_orders and not self._report_query_errors:
                    self._report_persistence_error = None
        except Exception as e:
            with self._lock:
                self._report_query_errors.add("orders")
                self._report_persistence_error = f"{type(e).__name__}: {e}"
            logger.error("同步券商委托失败: %s", e)

    def _sync_trades(self):
        if not self._trader or not self._connected:
            return
        if not self._should_sync("trades", 1.0):
            return
        try:
            trades = self._trader.query_stock_trades(self._stock_account)
            if trades is None:
                raise RuntimeError("QMT 成交查询未返回数据")
            self._last_trade_query_count = len(trades)
            results = []
            for t in trades:
                entry = {
                    "id": str(self._attr(t, "traded_id", "")),
                    "order_id": str(self._attr(t, "order_id", "")),
                    "symbol": self._attr(t, "stock_code", ""),
                    "name": "",
                    "side": ("buy" if self._attr(t, "order_type", 0) in BUY_ORDER_TYPES else
                             "sell" if self._attr(t, "order_type", 0) in SELL_ORDER_TYPES else "unknown"),
                    "price": round(float(self._attr(t, "traded_price", 0)), 3),
                    "quantity": self._attr(t, "traded_volume", 0),
                    "amount": round(float(self._attr(t, "traded_amount", 0)), 2),
                    "commission": self._attr(t, "commission", None),
                    "stamp_duty": None,
                    "trade_time": None,
                }
                results.append((entry, self._record_trade_report(entry, "query", t)))
            from core.data.repos.broker_trade_report import BrokerTradeReportRepo
            unprojected = BrokerTradeReportRepo.list_unprojected("qmt", self.account_id)
            to_persist = {}
            with self._lock:
                trades_by_id = {entry["id"]: entry for entry in self._trades}
                for entry, result in results:
                    if result and result["status"] != "conflict":
                        trades_by_id[entry["id"]] = entry
                        to_persist[entry["id"]] = entry
                for entry in unprojected:
                    entry["name"] = ""
                    trades_by_id[entry["id"]] = entry
                    to_persist[entry["id"]] = entry
                self._trades = list(trades_by_id.values())
            for entry in to_persist.values():
                if not self._persist_qmt_trade(entry):
                    raise RuntimeError("QMT 成交投影未持久化")
            unresolved_conflicts = BrokerTradeReportRepo.unresolved_conflict_count("qmt", self.account_id)
            if unresolved_conflicts:
                raise RuntimeError(f"QMT 存在 {unresolved_conflicts} 笔未核对的矛盾成交回报")
            with self._lock:
                self._report_query_errors.discard("trades")
                if not self._report_failed_orders and not self._report_query_errors:
                    self._report_persistence_error = None
        except Exception as e:
            with self._lock:
                self._report_query_errors.add("trades")
                self._report_persistence_error = f"{type(e).__name__}: {e}"
            logger.error("同步券商成交失败: %s", e)

    def _sync_positions(self):
        if not self._trader or not self._connected:
            return
        if not self._should_sync("positions", 1.0):
            return
        try:
            positions = self._trader.query_stock_positions(self._stock_account)
            if positions is None:
                raise RuntimeError("QMT 持仓查询未返回数据")
            current = {}
            for position in positions:
                if str(self._attr(position, "account_id", "")) != self._account:
                    raise RuntimeError("QMT 持仓资金账号不一致或缺失")
                symbol = self._attr(position, "stock_code")
                code = normalize_a_share_symbol(symbol)
                try:
                    shares = Decimal(str(self._attr(position, "volume")))
                except (InvalidOperation, TypeError, ValueError):
                    raise RuntimeError("QMT 持仓股数无效") from None
                if (not shares.is_finite() or shares < 0 or
                        shares != shares.to_integral_value() or code in current):
                    raise RuntimeError("QMT 持仓股数或证券身份无效")
                current[code] = position
            with self._lock:
                self._positions = current
                self._positions_observed_at = datetime.now()
        except Exception as e:
            with self._lock:
                self._positions_observed_at = None
            logger.debug(f"同步持仓失败: {e}")

    def _sync_asset(self):
        if not self._trader or not self._connected:
            return
        if not self._should_sync("asset", 1.0):
            return
        try:
            asset = self._trader.query_stock_asset(self._stock_account)
            if (asset is not None and str(self._attr(asset, "account_id", "")) == self._account and all(
                math.isfinite(float(self._attr(asset, field))) and
                float(self._attr(asset, field)) >= 0
                for field in ("cash", "frozen_cash", "market_value", "total_asset")
            )):
                with self._lock:
                    self._asset = asset
                    self._asset_observed_at = datetime.now()
                    self._asset_sync_ok = True
            else:
                with self._lock:
                    self._asset_sync_ok = False
        except Exception as e:
            with self._lock:
                self._asset_sync_ok = False
            logger.debug(f"同步资产失败: {e}")

class _TraderCallback:
    def __init__(self, broker):
        self._broker = broker

    def on_stock_order(self, order):
        oid = str(self._broker._attr(order, "order_id", ""))
        status = self._broker._attr(order, "order_status", 48)
        order_type = self._broker._attr(order, "order_type", 0)
        with self._broker._lock:
            existing = self._broker._orders.get(oid, {})
            candidate = {
                "id": oid,
                "symbol": self._broker._attr(order, "stock_code", existing.get("symbol", "")),
                "name": existing.get("name", ""),
                "side": "buy" if order_type in BUY_ORDER_TYPES else "sell" if order_type in SELL_ORDER_TYPES else "unknown",
                "quantity": self._broker._attr(order, "order_volume", 0),
                "price_type": _price_type(self._broker._attr(order, "price_type", 0)),
                "price": float(self._broker._attr(order, "price", 0)),
                "filled_qty": self._broker._attr(order, "traded_volume", 0),
                "filled_price": float(self._broker._attr(order, "traded_price", 0)),
                "status": STATUS_MAP.get(status, "unknown"),
                "submission": existing.get("submission"),
                "created_at": existing.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        result = self._broker._record_order_report(candidate, "callback", status, order)
        if not result or result["disposition"] in ("conflict", "unknown"):
            return
        candidate["status"] = result["status"]
        candidate["filled_qty"] = result["filled_qty"]
        if not self._broker._persist_qmt_order(candidate):
            return
        with self._broker._lock:
            self._broker._orders[oid] = candidate
        self._broker._sync_positions()

    def on_stock_trade(self, trade):
        tid = str(self._broker._attr(trade, "traded_id", ""))
        entry = {
            "id": tid,
            "order_id": str(self._broker._attr(trade, "order_id", "")),
            "symbol": self._broker._attr(trade, "stock_code", ""),
            "name": "",
            "side": ("buy" if self._broker._attr(trade, "order_type", 0) in BUY_ORDER_TYPES else
                     "sell" if self._broker._attr(trade, "order_type", 0) in SELL_ORDER_TYPES else "unknown"),
            "price": round(float(self._broker._attr(trade, "traded_price", 0)), 3),
            "quantity": self._broker._attr(trade, "traded_volume", 0),
            "amount": round(float(self._broker._attr(trade, "traded_amount", 0)), 2),
            "commission": self._broker._attr(trade, "commission", None),
            "stamp_duty": None,
            "trade_time": None,
        }
        result = self._broker._record_trade_report(entry, "callback", trade)
        if not result or result["status"] == "conflict":
            return
        with self._broker._lock:
            new = not any(t.get("id") == tid for t in self._broker._trades)
            if new:
                self._broker._trades.append(entry)
            retry = f"trade:{tid}" in self._broker._report_failed_orders
        if (new or retry) and not self._broker._persist_qmt_trade(entry):
            return

    def on_stock_asset(self, asset):
        with self._broker._lock:
            self._broker._asset = asset

    def on_stock_position(self, position):
        code = self._broker._attr(position, "stock_code", "")
        if code:
            with self._broker._lock:
                self._broker._positions[code] = position

    def on_connected(self):
        logger.info("QMT 连接已建立")

    def on_disconnected(self):
        logger.warning("QMT 连接已断开")
        self._broker._connected = False

    def on_cancel_error(self, order_id, msg):
        logger.error(f"QMT 撤单失败 [{order_id}]: {msg}")

    def on_order_error(self, order_id, msg):
        logger.error(f"QMT 下单失败 [{order_id}]: {msg}")
