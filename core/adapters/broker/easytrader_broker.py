"""
EasytraderBroker — 通过 easytrader 操控同花顺客户端实现实盘交易。
基于 pywinauto GUI 自动化，对接银河证券 (yh_client)。
需要：同花顺客户端已打开并登录，easytrader + pywin32 已安装。
"""
import logging
import re
import threading
from datetime import datetime
from decimal import Decimal, InvalidOperation
from core.adapters.broker.base import BaseBroker
from core.shared.order_validation import normalize_a_share_symbol

logger = logging.getLogger(__name__)

STATUS_MAP = {"已报": "pending", "部成": "partial", "已成": "filled",
              "已撤": "cancelled", "废单": "rejected"}


def _read_share_count(value, allow_zero=False):
    if value is None or isinstance(value, bool):
        raise ValueError("股数缺失或无效")
    try:
        shares = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("股数缺失或无效") from None
    if (not shares.is_finite() or shares != shares.to_integral_value() or
            shares < (0 if allow_zero else 1) or shares > Decimal("2147483647")):
        raise ValueError("股数缺失或无效")
    return int(shares)


class EasytraderBroker(BaseBroker):
    broker_name = "easytrader"

    def __init__(self, user="", password="", exe_path="", commission_rate=0.0003, stamp_duty=0.001,
                 broker_account_id=""):
        self._user = user
        self._password = password
        self._exe_path = exe_path or r"C:\双子星-中国银河证券\Binarystar.exe"
        self._commission_rate = commission_rate
        self._stamp_duty = stamp_duty

        self._trader = None
        self._connected = False
        account_id = str(broker_account_id or "").strip()
        if account_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", account_id):
            raise ValueError("EASYTRAIDER_ACCOUNT_ID 格式无效")
        self._configured_account_id = account_id
        self._account_id = f"EASYT-{account_id}" if account_id else "EASYT-UNVERIFIED"

        self._lock = threading.Lock()

    def connect(self):
        if self._account_id == "EASYT-UNVERIFIED":
            raise RuntimeError("EASYTRAIDER_ACCOUNT_ID 未配置，拒绝连接未标识的券商账户")
        try:
            import easytrader
        except ImportError:
            raise RuntimeError(
                "easytrader 未安装，请执行: pip install easytrader pywin32"
            )

        try:
            self._trader = easytrader.use("yh_client")
        except Exception as e:
            raise RuntimeError(f"easytrader 初始化失败: {e}")

        try:
            self._trader.connect(self._exe_path)
            logger.info("EasytraderBroker 已连接到同花顺客户端 (银河证券)")
        except Exception as e:
            raise RuntimeError(
                f"连接同花顺客户端失败，请确认客户端已打开: {e}\n"
                f"exe_path: {self._exe_path}"
            )

        self._connected = True
        logger.info("EasytraderBroker 已连接 (客户端已登录)")

        # 切换为 WMCopy 网格策略，避开验证码弹窗
        try:
            from easytrader.grid_strategies import WMCopy
            self._trader.grid_strategy = WMCopy()
            logger.info("EasytraderBroker 已启用 WMCopy 网格策略")
        except Exception as e:
            logger.warning(f"WMCopy 策略设置失败: {e}")

        return True

    def disconnect(self):
        self._connected = False
        if self._trader:
            try:
                self._trader.exit()
            except Exception:
                pass
            self._trader = None

    @property
    def connected(self):
        return self._connected

    @property
    def account_id(self):
        return self._account_id

    def place_order(self, symbol, name, side, quantity, price_type, price=None):
        from core.shared.order_validation import validate_real_limit_order
        symbol, qty, price = validate_real_limit_order(symbol, side, quantity, price_type, price)
        symbol = symbol.split(".")[0]
        if not self._connected or self._trader is None:
            raise RuntimeError("EasytraderBroker 未连接")

        try:
            if side == "buy":
                order_id = self._trader.buy(symbol, price, qty)
            else:
                order_id = self._trader.sell(symbol, price, qty)

        except Exception as e:
            logger.error(f"easytrader 委托结果不明确: {symbol} {side} {qty}股, {e}")
            raise RuntimeError("券商委托结果不明确，须查询券商委托后人工核对") from e
        if not order_id:
            raise RuntimeError("券商未返回委托编号，结果不明确，须查询券商委托后人工核对")
        order_id_str = str(order_id)

        order = {
            "id": order_id_str,
            "symbol": symbol,
            "name": name,
            "side": side,
            "quantity": qty,
            "price_type": price_type,
            "price": price,
            "broker_request": {"symbol": symbol, "side": side, "quantity": qty,
                               "price_type": "limit", "price": price},
            "broker_response": {"order_id": order_id_str},
            "filled_qty": 0,
            "filled_price": 0,
            "status": "pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        submission = {
            "client_request": {"symbol": symbol, "side": side, "quantity": qty,
                               "price_type": price_type, "price": price},
            "broker_request": {"method": "buy" if side == "buy" else "sell",
                               "args": [symbol, price, qty], "price_type": "limit"},
            "broker_response": {"order_id": order_id_str},
        }
        try:
            from core.data.repos.broker_order_report import BrokerOrderReportRepo
            submission = BrokerOrderReportRepo.record_submission(
                "easytrader", self.account_id, order, submission["client_request"],
                submission["broker_request"], submission["broker_response"],
            )
        except Exception as error:
            raise RuntimeError("Easytrader已返回委托编号，但请求/响应证据未持久化；请先查询券商，禁止直接重试") from error
        order["submission"] = submission

        self._persist_order(order)
        logger.info(f"easytrader下单: {symbol} {side} {qty}股 → {order_id_str}")
        return order_id_str

    def cancel_order(self, order_id):
        if not self._connected or self._trader is None:
            return False
        try:
            self._trader.cancel_entrust(str(order_id))
            logger.info(f"easytrader 撤单: {order_id}")
            return True
        except Exception as e:
            logger.error(f"easytrader 撤单失败 [{order_id}]: {e}")
            return False

    def get_order(self, order_id):
        orders = self.get_orders()
        for o in orders:
            if o["id"] == str(order_id):
                return o
        return None

    def get_orders(self, status=None, persist_reports=True):
        if not self._connected or self._trader is None:
            raise RuntimeError("Easytrader 未连接，委托查询不可用")
        try:
            raw = self._trader.today_entrusts
            if raw is None:
                raise RuntimeError("券商未返回委托清单")
        except Exception as e:
            logger.error(f"easytrader 查询委托失败: {e}")
            self._dismiss_captcha()
            raise RuntimeError("Easytrader 委托查询不可用") from e

        from core.data.repos.broker_order_report import BrokerOrderReportRepo

        submission_metadata = BrokerOrderReportRepo.list_submission_metadata(
            "easytrader", self.account_id,
            [str(r.get("合同编号") or r.get("委托编号") or "") for r in raw],
        )
        orders = []
        for r in raw:
            try:
                oid = str(r.get("合同编号") or r.get("委托编号") or "")
                symbol = str(r.get("证券代码") or r.get("股票代码") or "")
                normalize_a_share_symbol(symbol)
                operation = str(r.get("操作") or r.get("买卖") or "")
                if "买" in operation and "卖" not in operation:
                    side = "buy"
                elif "卖" in operation and "买" not in operation:
                    side = "sell"
                else:
                    raise ValueError
                quantity = _read_share_count(r.get("委托数量"))
                filled_qty = _read_share_count(r.get("成交数量"), allow_zero=True)
                if not oid or filled_qty > quantity:
                    raise ValueError
                price = Decimal(str(r.get("委托价格")))
                filled_price = Decimal(str(r.get("成交价格") or 0))
                if (not price.is_finite() or price <= 0 or price >= Decimal("1000000000") or
                        price != price.quantize(Decimal("0.001")) or
                        not filled_price.is_finite() or filled_price < 0 or
                        (filled_qty and filled_price <= 0)):
                    raise ValueError
                raw_status = str(r.get("状态") or r.get("委托状态") or "")
                status = STATUS_MAP.get(raw_status, "unknown")
                if ((status == "filled" and filled_qty != quantity) or
                        (status == "partial" and not 0 < filled_qty < quantity)):
                    raise ValueError
                o = {
                    "id": oid,
                    "symbol": symbol,
                    "name": str(r.get("证券名称") or ""),
                    "side": side,
                    "quantity": quantity,
                    "price_type": "limit",
                    "price": float(price),
                    "filled_qty": filled_qty,
                    "filled_price": float(filled_price),
                    "status": status,
                    "created_at": str(r.get("委托时间") or r.get("报单时间") or ""),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except (AttributeError, InvalidOperation, TypeError, ValueError):
                raise RuntimeError("Easytrader 委托证券、方向、股数或价格无效") from None
            report = (BrokerOrderReportRepo.ingest(
                "easytrader", self.account_id, o, "query", raw_status=raw_status, raw_fields=r,
            ) if persist_reports else None)
            if report is not None:
                o["report_disposition"] = report["disposition"]
                o["recovered_from_persisted_report"] = report["disposition"] == "stale"
            if oid in submission_metadata:
                o["submission"] = submission_metadata[oid]
            if report and report["disposition"] in ("applied", "duplicate", "stale"):
                o["status"] = report["status"]
                o["filled_qty"] = report["filled_qty"]
            elif report:
                o["status"] = "unknown"
            orders.append(o)
        if persist_reports:
            if BrokerOrderReportRepo.unresolved_count("easytrader", self.account_id):
                raise RuntimeError("Easytrader 存在未核对的异常委托回报，委托查询不可用")
            durable_orders = BrokerOrderReportRepo.list_current_states("easytrader", self.account_id)
            current_order_ids = {
                str(order["id"]) for order in orders
                if order.get("report_disposition") in ("applied", "duplicate")
            }
            unprojected_ids = {
                str(row["order_id"])
                for row in BrokerOrderReportRepo.list_unprojected("easytrader", self.account_id)
            }
            for order in durable_orders:
                order["recovered_from_persisted_report"] = str(order["id"]) not in current_order_ids
                if str(order["id"]) in unprojected_ids:
                    self._persist_order(order, required=True)
            by_id = {order["id"]: order for order in orders}
            for order in durable_orders:
                by_id.setdefault(order["id"], order)
            orders = list(by_id.values())
        if status:
            orders = [o for o in orders if o["status"] == status]
        return orders

    def get_trades(self, symbol=None, persist_reports=True):
        if not self._connected or self._trader is None:
            raise RuntimeError("Easytrader 未连接，成交查询不可用")
        try:
            raw = self._trader.today_trades
            if raw is None:
                raise RuntimeError("券商未返回成交清单")
        except Exception as e:
            logger.error(f"easytrader 查询成交失败: {e}")
            self._dismiss_captcha()
            raise RuntimeError("Easytrader 成交查询不可用") from e

        requested_symbol = symbol
        from core.data.repos.broker_trade_report import BrokerTradeReportRepo

        trades = []
        for r in raw:
            try:
                trade_id = str(r.get("合同编号") or r.get("成交编号") or "")
                order_id = str(r.get("委托编号") or "")
                symbol = str(r.get("证券代码") or "")
                normalize_a_share_symbol(symbol)
                operation = str(r.get("操作") or "")
                if "买" in operation and "卖" not in operation:
                    side = "buy"
                elif "卖" in operation and "买" not in operation:
                    side = "sell"
                else:
                    raise ValueError
                quantity = _read_share_count(r.get("成交数量"))
                price = Decimal(str(r.get("成交价格")))
                amount = Decimal(str(r.get("成交金额")))
                if (not trade_id or not order_id or not price.is_finite() or price <= 0 or
                        not amount.is_finite() or amount <= 0):
                    raise ValueError
                commission = (Decimal(str(r["手续费"])) if r.get("手续费") is not None else None)
                stamp_duty = (Decimal(str(r["印花税"])) if r.get("印花税") is not None else None)
                if any(fee is not None and (not fee.is_finite() or fee < 0)
                       for fee in (commission, stamp_duty)):
                    raise ValueError
                t = {
                    "id": trade_id,
                    "order_id": order_id,
                    "symbol": symbol,
                    "name": str(r.get("证券名称") or ""),
                    "side": side,
                    "price": float(price),
                    "quantity": quantity,
                    "amount": float(amount),
                    "commission": float(commission) if commission is not None else None,
                    "stamp_duty": float(stamp_duty) if stamp_duty is not None else None,
                    "trade_time": str(r.get("成交时间") or ""),
                }
            except (AttributeError, InvalidOperation, TypeError, ValueError):
                raise RuntimeError("Easytrader 成交证券、方向、股数、金额或费用无效") from None
            if persist_reports:
                report = BrokerTradeReportRepo.ingest(
                    "easytrader", self.account_id, t, "query", raw_fields=r,
                )
                if report.get("status") not in ("recorded", "duplicate"):
                    raise RuntimeError("Easytrader 存在未核对或未知状态的成交回报，成交查询不可用")
            trades.append(t)
        if persist_reports:
            if BrokerTradeReportRepo.unresolved_conflict_count("easytrader", self.account_id):
                raise RuntimeError("Easytrader 存在未核对的矛盾成交回报，成交查询不可用")
            known_names = {trade["id"]: trade.get("name", "") for trade in trades}
            unprojected = BrokerTradeReportRepo.list_unprojected("easytrader", self.account_id)
            for trade in unprojected:
                trade["name"] = known_names.get(trade["id"], "")
                trade["price"] = float(trade["price"])
                trade["quantity"] = int(trade["quantity"])
                trade["amount"] = float(trade["amount"])
                trade["commission"] = float(trade["commission"]) if trade["commission"] is not None else None
                trade["stamp_duty"] = float(trade["stamp_duty"]) if trade["stamp_duty"] is not None else None
                self._persist_trade(trade, required=True)
            by_id = {trade["id"]: trade for trade in trades}
            for trade in unprojected:
                by_id.setdefault(trade["id"], trade)
            trades = list(by_id.values())
        if requested_symbol:
            trades = [t for t in trades if t["symbol"] == requested_symbol]
        return trades

    def get_trade_reconciliation_evidence(self, trade_id):
        snapshot = self.get_reconciliation_snapshot()
        if (snapshot.get("identity_verified") is not True or
                snapshot.get("account", {}).get("account_id") != self.account_id):
            raise RuntimeError("Easytrader 成交核对账户身份未通过")
        matches = [trade for trade in self.get_trades(persist_reports=False) if trade["id"] == str(trade_id)]
        verified_balance = self._read_balance_snapshot()
        if (verified_balance["account_id"] != self._configured_account_id or
                any(Decimal(str(round(verified_balance[key], 2))) != Decimal(str(snapshot["account"][key]))
                    for key in ("cash", "frozen", "available"))):
            raise RuntimeError("Easytrader 成交核对期间账户或资金快照发生变化")
        if len(matches) != 1:
            raise ValueError("券商当前逐笔成交清单缺少唯一对应记录")
        return matches[0]

    def get_order_reconciliation_evidence(self, order_id):
        snapshot = self.get_reconciliation_snapshot()
        if (snapshot.get("identity_verified") is not True or
                snapshot.get("account", {}).get("account_id") != self.account_id):
            raise RuntimeError("Easytrader 委托核对账户身份未通过")
        matches = [order for order in self.get_orders(persist_reports=False) if order["id"] == str(order_id)]
        verified_balance = self._read_balance_snapshot()
        if (verified_balance["account_id"] != self._configured_account_id or
                any(Decimal(str(round(verified_balance[key], 2))) != Decimal(str(snapshot["account"][key]))
                    for key in ("cash", "frozen", "available"))):
            raise RuntimeError("Easytrader 委托核对期间账户或资金快照发生变化")
        if len(matches) != 1:
            raise ValueError("券商当前委托清单缺少唯一对应记录")
        return matches[0]

    def _dismiss_captcha(self):
        """关闭同花顺新版验证码弹窗"""
        try:
            top = self._trader.app.top_window()
            if top.class_name() == "#32770" and top.is_visible():
                # 按取消按钮关闭
                top.child_window(control_id=2, class_name="Button").click()
                return True
        except Exception:
            pass
        return False

    def get_positions(self):
        if not self._connected or self._trader is None:
            raise RuntimeError("Easytrader 未连接，持仓查询不可用")
        try:
            raw = self._trader.position
            if raw is None:
                raise RuntimeError("券商未返回持仓清单")
        except Exception as e:
            logger.error(f"easytrader 查询持仓失败: {e}")
            self._dismiss_captcha()
            raise RuntimeError("Easytrader 持仓查询不可用") from e

        positions = []
        for r in raw:
            try:
                code = str(r.get("证券代码") or r.get("股票代码") or "")
                normalize_a_share_symbol(code)
                raw_shares = next((r[key] for key in ("股份余额", "当前持仓", "股票余额", "当前拥股")
                                   if r.get(key) is not None), None)
                shares = _read_share_count(raw_shares, allow_zero=True)
                raw_sellable = r.get("股份可用")
                try:
                    sellable = _read_share_count(raw_sellable, allow_zero=True)
                    if sellable > shares:
                        sellable = None
                except (InvalidOperation, TypeError, ValueError):
                    sellable = None
            except (AttributeError, InvalidOperation, TypeError, ValueError):
                raise RuntimeError("Easytrader 持仓证券代码或股数无效") from None
            if shares == 0:
                continue
            positions.append({
                "symbol": code,
                "shares": shares,
                "sellable": sellable,
                "avg_cost": float(r.get("成本价") or r.get("买入均价") or r.get("参考成本价") or 0),
                "current_price": float(r.get("最新价") or r.get("市价") or 0),
                "market_value": float(r.get("市值") or r.get("参考市值") or 0),
                "unrealized_pnl": float(r.get("浮动盈亏") or r.get("盈亏") or 0),
            })
        return positions

    def _read_balance_snapshot(self):
        if not self._configured_account_id:
            raise RuntimeError("Easytrader 未配置可核验资金账号")
        try:
            rows = self._trader.balance
        except Exception as error:
            raise RuntimeError("Easytrader 资金快照不可用") from error
        if not isinstance(rows, (list, tuple)) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise RuntimeError("Easytrader 资金账号快照行数或格式无法核验")
        row = rows[0]
        account_id = str(row.get("资金帐号") or "").strip()
        if not account_id:
            raise RuntimeError("Easytrader 资金快照缺少资金账号")
        if account_id != self._configured_account_id:
            raise RuntimeError("Easytrader 返回资金账号与配置账号不一致")
        try:
            available = Decimal(str(row.get("可用资金")))
            market_value = Decimal(str(row.get("参考市值")))
            total_assets = Decimal(str(row.get("总资产")))
            unrealized = (Decimal(str(row["股份参考盈亏"]))
                          if row.get("股份参考盈亏") is not None else None)
        except (InvalidOperation, TypeError, ValueError):
            raise RuntimeError("Easytrader 资金快照金额字段无效") from None
        values = (available, market_value, total_assets)
        if any(not value.is_finite() or value < 0 or value >= Decimal("1000000000000") for value in values):
            raise RuntimeError("Easytrader 资金快照金额字段无效")
        if unrealized is not None and (not unrealized.is_finite() or abs(unrealized) >= Decimal("1000000000000")):
            unrealized = None
        cash = total_assets - market_value
        frozen = cash - available
        if cash < 0 or frozen < 0:
            raise RuntimeError("Easytrader 资金快照分项不一致")
        return {
            "account_id": account_id,
            "cash": cash,
            "frozen": frozen,
            "available": available,
            "market_value": market_value,
            "total_assets": total_assets,
            "unrealized_pnl": unrealized,
        }

    def get_account(self):
        if not self._connected or self._trader is None:
            return {
                "account_id": self._account_id,
                "connected": False, "status": "unavailable", "source": "easytrader", "as_of": None,
                "cash": None, "frozen": None, "available": None,
                "market_value": None, "total_assets": None,
                "unrealized_pnl": None, "updated_at": None,
            }

        try:
            snapshot = self._read_balance_snapshot()

            logger.info(
                "Easytrader 余额读取成功: 总资产=%s, 可用=%s, 市值=%s",
                snapshot["total_assets"], snapshot["available"], snapshot["market_value"],
            )
        except Exception as e:
            logger.error(f"easytrader 查询资金失败: {e}")
            return {
                "account_id": self._account_id,
                "connected": True, "status": "unavailable", "source": "easytrader", "as_of": None,
                "cash": None, "frozen": None, "available": None,
                "market_value": None, "total_assets": None,
                "unrealized_pnl": None, "updated_at": None,
            }

        observed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "account_id": self._account_id,
            "connected": True, "status": "fresh", "source": "easytrader", "as_of": observed,
            "cash": round(snapshot["cash"], 2),
            "frozen": round(snapshot["frozen"], 2),
            "available": round(snapshot["available"], 2),
            "market_value": round(snapshot["market_value"], 2),
            "total_assets": round(snapshot["total_assets"], 2),
            "unrealized_pnl": round(snapshot["unrealized_pnl"], 2) if snapshot["unrealized_pnl"] is not None else None,
            "updated_at": observed,
        }

    def get_reconciliation_snapshot(self):
        if not self._connected or self._trader is None:
            raise RuntimeError("Easytrader 未连接")
        from core.shared.trading_time import A_SHARE_TIMEZONE, normalize_a_share_datetime

        snapshot_started_at = datetime.now(A_SHARE_TIMEZONE)
        account = self.get_account()
        if account.get("status") != "fresh":
            raise RuntimeError("Easytrader 资产快照不可用")
        positions = self.get_positions()
        orders = self.get_orders(persist_reports=False)
        self.get_trades(persist_reports=False)
        verified_balance = self._read_balance_snapshot()
        verified_account_id = verified_balance["account_id"]
        if any(Decimal(str(round(verified_balance[key], 2))) != Decimal(str(account[key]))
               for key in ("cash", "frozen", "available")):
            raise RuntimeError("Easytrader 对账期间资金余额发生变化")
        from core.data.repos.broker_trade_report import BrokerTradeReportRepo
        trade_report_health = BrokerTradeReportRepo.get_health("easytrader", self.account_id)
        snapshot_completed_at = datetime.now(A_SHARE_TIMEZONE)
        try:
            asset_observed_at = normalize_a_share_datetime(datetime.fromisoformat(str(account["as_of"])))
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Easytrader 资产快照时间不可核验") from None
        if ((snapshot_completed_at - snapshot_started_at).total_seconds() > 5 or
                (snapshot_completed_at - asset_observed_at).total_seconds() > 5 or
                (snapshot_completed_at - asset_observed_at).total_seconds() < -1):
            raise RuntimeError("Easytrader 对账快照跨时或资产已过期")
        return {
            "account": account,
            "snapshot_started_at": snapshot_started_at.isoformat(timespec="seconds"),
            "snapshot_completed_at": snapshot_completed_at.isoformat(timespec="seconds"),
            "identity_verified": verified_account_id == self._configured_account_id,
            "trade_report_health": trade_report_health,
            "positions": [
                {"symbol": p["symbol"], "shares": p["shares"], "sellable": p["sellable"]}
                for p in positions
            ],
            "orders": [
                {"id": o["id"], "status": o["status"]}
                for o in orders
            ],
        }
