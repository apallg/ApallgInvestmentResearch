from datetime import datetime
from decimal import Decimal, InvalidOperation
import math
import re
import uuid

from core.data.connection import Database
from core.data.repos.local_ledger import ACCOUNT_ID, LocalLedgerRepo, _read_ledger_share_count
from core.data.repos.local_order_intent import LocalOrderIntentRepo
from core.shared.order_validation import normalize_a_share_symbol


class LocalLedgerService:
    @staticmethod
    def record_historical_fill(detail):
        return LocalLedgerRepo.record_historical_fill(detail)

    @staticmethod
    def get_account():
        for _ in range(2):
            account = LocalLedgerRepo.get_account()
            integrity = LocalLedgerRepo.audit_integrity()
            opening = LocalLedgerRepo.get_opening_snapshot()
            latest = LocalLedgerRepo.get_account()
            same_account = all(account.get(key) == latest.get(key) for key in (
                "status", "cash", "frozen", "version", "broker_account_id",
            ))
            if (same_account and account.get("version") == integrity.get("account_version")):
                account["integrity"] = integrity
                account["holdings_fingerprint"] = opening["fingerprint"]
                account["holdings_count"] = len(opening["holdings"])
                account["pending_holdings"] = opening.get("pending_holdings", [])
                account["legacy_unknown"] = LocalLedgerRepo.get_legacy_counts()
                return account
        raise RuntimeError("本地账本在读取期间持续变化，请刷新后重试")

    @staticmethod
    def get_advice_snapshot():
        from backend.service.market_data import MarketDataService

        local = LocalLedgerRepo.get_account()
        if not LocalLedgerRepo.audit_integrity()["ok"]:
            raise RuntimeError("本地账本校验失败，停止生成调仓建议")
        holdings = Database.fetchall(
            "SELECT code,name,shares,cost_price,buy_date FROM holdings WHERE account_id=%s ORDER BY code",
            (ACCOUNT_ID,),
        )
        codes = [row["code"] for row in holdings if int(row["shares"] or 0) > 0]
        quotes = {}
        if codes:
            response = MarketDataService.get_live_quotes(codes, use_case="execution")
            if response.get("meta", {}).get("freshness") != "fresh":
                raise RuntimeError("本地持仓行情不完整，停止生成调仓建议")
            quotes = response.get("data") or {}
        positions = []
        market_value = 0.0
        for row in holdings:
            code = row["code"]
            shares = int(row["shares"] or 0)
            if shares < 0:
                raise RuntimeError("本地持仓数量无效，停止生成调仓建议")
            if shares == 0:
                continue
            quote = quotes.get(code) or {}
            try:
                price = float(quote.get("price"))
            except (TypeError, ValueError, OverflowError):
                price = 0.0
            if not math.isfinite(price) or price <= 0:
                raise RuntimeError("本地持仓行情不完整，停止生成调仓建议")
            cost_price = float(row["cost_price"] or 0)
            value = price * shares
            if not math.isfinite(value):
                raise RuntimeError("本地持仓市值无效，停止生成调仓建议")
            cost = cost_price * shares
            market_value += value
            positions.append({
                "symbol": code, "code": code, "name": row["name"], "shares": shares,
                "cost_price": cost_price, "current_price": price, "market_value": value,
                "unrealized_pnl": value - cost if cost > 0 else 0,
                "pnl_pct": (value - cost) / cost if cost > 0 else 0,
                "pct_chg": float(quote.get("pct_chg") or 0),
                "buy_date": str(row["buy_date"] or ""),
            })
        total_assets = float(local["cash"] or 0) + market_value
        latest = LocalLedgerRepo.get_account()
        if (latest["version"] != local["version"] or latest["cash"] != local["cash"] or
                latest["frozen"] != local["frozen"]):
            raise RuntimeError("本地账本在读取建议期间已变化，请重新生成")
        for position in positions:
            position["weight"] = position["market_value"] / total_assets if total_assets > 0 else 0
        return {
            "account": {"account_id": ACCOUNT_ID, "cash": float(local["available"] or 0),
                        "available_cash": float(local["available"] or 0),
                        "total_cash": float(local["cash"] or 0),
                        "frozen_cash": float(local["frozen"] or 0),
                        "market_value": market_value, "total_assets": total_assets,
                        "status": local["status"], "quote_status": "fresh",
                        "cash_source": "local_ledger",
                        "version": local["version"],
                        "broker_account_id": local.get("broker_account_id")},
            "holdings": positions,
        }

    @staticmethod
    def get_t0_account_snapshot():
        from datetime import date

        snapshot = LocalLedgerService.get_advice_snapshot()
        account = snapshot["account"]
        if account["status"] != "active":
            raise ValueError("本地账户期初尚未核准，不能生成做T建议")
        try:
            available_cash = Decimal(str(account.get("available_cash")))
        except (InvalidOperation, TypeError, ValueError):
            available_cash = Decimal("NaN")
        if not available_cash.is_finite() or available_cash < 0:
            raise RuntimeError("本地可用现金缺失或无效，不能生成做T建议")
        if account.get("broker_account_id"):
            reconciliation = LocalLedgerService.reconcile()
            if (reconciliation.get("can_execute") is not True or
                    reconciliation.get("local_version") != account["version"] or
                    reconciliation.get("broker_account_id") != account["broker_account_id"]):
                raise RuntimeError(reconciliation.get("reason") or "券商对账状态不可用，不能生成做T建议")
            sellable = reconciliation.get("broker_sellable") or {}
            for holding in snapshot["holdings"]:
                shares = sellable.get(holding["code"])
                if isinstance(shares, bool) or not isinstance(shares, int) or not 0 <= shares <= holding["shares"]:
                    raise RuntimeError("券商可卖股数缺失或无效，不能生成做T建议")
                holding["sellable_shares"] = shares
            snapshot["sellable_source"] = "broker_reconciliation"
        else:
            from core.shared.trading_time import market_date

            rows = Database.fetchall(
                "SELECT code,COALESCE(SUM(shares),0) AS bought_today FROM local_ledger_events "
                "WHERE account_id=%s AND event_type IN ('manual_fill','broker_fill') "
                "AND side IN ('buy','add') AND occurred_at>=CURDATE() "
                "AND occurred_at<CURDATE()+INTERVAL 1 DAY GROUP BY code",
                (ACCOUNT_ID,),
            )
            bought_today = {}
            for row in rows:
                bought_today[row["code"]] = _read_ledger_share_count(row["bought_today"], allow_zero=True)
            today = market_date()
            for holding in snapshot["holdings"]:
                shares = holding["shares"]
                bought = bought_today.get(holding["code"], 0)
                if bought > shares:
                    raise RuntimeError("本地当日买入成交超过当前持仓，不能生成做T建议")
                try:
                    buy_date = date.fromisoformat(str(holding.get("buy_date") or "")[:10])
                except ValueError:
                    buy_date = None
                holding["sellable_shares"] = shares - bought if buy_date and buy_date < today else 0
            snapshot["sellable_source"] = "local_ledger_t1"
            latest = LocalLedgerRepo.get_account()
            if any(latest.get(key) != account.get(key)
                   for key in ("version", "cash", "frozen", "broker_account_id")):
                raise RuntimeError("本地账本在读取做T快照期间已变化，请重试")
        snapshot["account"]["available_cash"] = float(available_cash)
        return snapshot

    @staticmethod
    def reconcile():
        from core.adapters.broker import EasytraderBroker, QmtBroker, is_fake_broker
        from core.data.repos.broker_order_report import BrokerOrderReportRepo
        from core.shared.config import BROKER_TYPE, BROKER_FAKE

        local = LocalLedgerRepo.get_account()
        integrity = LocalLedgerRepo.audit_integrity()
        missing_items = []

        rows = Database.fetchall(
            "SELECT code,shares FROM holdings WHERE account_id=%s ORDER BY code", (ACCOUNT_ID,)
        )
        local_positions = {}
        for row in rows:
            try:
                local_positions[row["code"]] = _read_ledger_share_count(row.get("shares"), allow_zero=True)
            except ValueError:
                local_positions[row["code"]] = None
                missing_items.append("local.position.shares")
        result = {
            "account_id": ACCOUNT_ID,
            "local_status": local["status"],
            "local_version": local["version"],
            "local_as_of": local.get("updated_at"),
            "local_cash": local.get("cash"),
            "local_frozen": local.get("frozen"),
            "local_available": local["available"],
            "ledger_integrity": integrity,
            "local_positions": local_positions,
            "legacy_unknown": LocalLedgerRepo.get_legacy_counts(),
            "broker_status": "unavailable",
            "broker_account_id": None,
            "bound_broker_account_id": local.get("broker_account_id"),
            "broker_identity_verified": False,
            "broker_as_of": None,
            "broker_received_at": None,
            "broker_snapshot_started_at": None,
            "broker_snapshot_completed_at": None,
            "broker_available": None,
            "broker_cash": None,
            "broker_frozen": None,
            "cash_difference": None,
            "cash_balance_difference": None,
            "frozen_difference": None,
            "position_differences": [],
            "broker_sellable": {},
            "sellable_missing_codes": [],
            "pending_orders": [],
            "pending_journal_orders": [],
            "unresolved_order_reports": [],
            "pending_local_intent": False,
            "broker_trade_report_health": None,
            "missing_items": list(dict.fromkeys(missing_items)),
            "can_bind": False,
            "can_execute": False,
            "reason": "券商数据不可用",
        }
        def add_missing(*items):
            result["missing_items"] = list(dict.fromkeys([*result["missing_items"], *items]))

        if local["status"] != "active":
            result["reason"] = "本地账户期初尚未核准"
        if BROKER_TYPE == BROKER_FAKE:
            result["broker_status"] = "simulation"
            result["reason"] = "当前为模拟券商，不能核验实盘账户"
            return result
        try:
            from backend.api.trading import get_broker
            broker = get_broker()
            if is_fake_broker(broker):
                result["broker_status"] = "simulation"
                result["broker_account_id"] = broker.account_id
                result["reason"] = "当前为模拟券商，不能核验实盘账户"
                return result
            broker_key = ("qmt" if isinstance(broker, QmtBroker) else
                          "easytrader" if isinstance(broker, EasytraderBroker) else None)
            if broker_key is None:
                add_missing("broker.report_namespace")
                result["reason"] = "券商适配器缺少受支持的回报命名空间"
                return result
            if isinstance(broker, QmtBroker):
                snapshot = broker.get_reconciliation_snapshot(check_report_integrity=False)
            else:
                snapshot = broker.get_reconciliation_snapshot()
            account = snapshot["account"]
            identity_verified = snapshot.get("identity_verified") is True
            required = ("account_id", "as_of", "cash", "frozen", "available")
            add_missing(*(f"account.{key}" for key in required if account.get(key) is None))
            result["broker_account_id"] = account.get("account_id")
            result["broker_identity_verified"] = identity_verified
            if result["missing_items"]:
                raise RuntimeError("券商账户快照缺少身份、时间或资金分项")
            if account["account_id"] != broker.account_id:
                add_missing("account.identity")
                result["reason"] = "券商返回资金账号与当前连接不一致"
                raise RuntimeError("券商账户身份与当前连接不一致")
            try:
                broker_cash = Decimal(str(account["cash"]))
                broker_frozen = Decimal(str(account["frozen"]))
                available = Decimal(str(account["available"]))
            except (InvalidOperation, TypeError, ValueError):
                add_missing("account.fund_values")
                raise RuntimeError("券商账户资金分项无效") from None
            if (any(not value.is_finite() or value < 0 for value in
                    (broker_cash, broker_frozen, available)) or
                    broker_cash - broker_frozen != available):
                add_missing("account.fund_consistency")
                raise RuntimeError("券商账户资金分项不一致")
            try:
                broker_time = datetime.fromisoformat(str(account["as_of"]))
            except ValueError:
                add_missing("account.as_of")
                raise RuntimeError("券商账户快照时间无效") from None
            from core.shared.trading_time import A_SHARE_TIMEZONE, normalize_a_share_datetime

            broker_time = normalize_a_share_datetime(broker_time)
            broker_received_at = datetime.now(A_SHARE_TIMEZONE)
            broker_age = (broker_received_at - broker_time).total_seconds()
            difference = available - Decimal(str(local["available"])) if local["available"] is not None else None
            cash_balance_difference = (broker_cash - Decimal(str(local["cash"]))
                                       if local.get("cash") is not None else None)
            frozen_difference = (broker_frozen - Decimal(str(local["frozen"]))
                                 if local.get("frozen") is not None else None)
            result.update({
                "broker_account_id": account["account_id"],
                "broker_identity_verified": identity_verified,
                "broker_as_of": account["as_of"],
                "broker_received_at": broker_received_at,
                "broker_snapshot_started_at": snapshot.get("snapshot_started_at"),
                "broker_snapshot_completed_at": snapshot.get("snapshot_completed_at"),
                "broker_age_seconds": round(broker_age, 1),
                "broker_available": available,
                "broker_cash": broker_cash,
                "broker_frozen": broker_frozen,
                "cash_difference": difference,
                "cash_balance_difference": cash_balance_difference,
                "frozen_difference": frozen_difference,
            })
            broker_positions = {}
            broker_sellable = {}
            sellable_missing_codes = []
            for item in snapshot["positions"]:
                try:
                    code = normalize_a_share_symbol(item["symbol"])
                except (KeyError, ValueError):
                    add_missing("position.symbol")
                    raise RuntimeError("券商持仓证券代码或交易所无法核验") from None
                try:
                    quantity = Decimal(str(item.get("shares")))
                except Exception:
                    quantity = Decimal("NaN")
                if (not quantity.is_finite() or
                        quantity < 0 or quantity != quantity.to_integral_value()):
                    add_missing("position.symbol", "position.shares")
                    raise RuntimeError("券商持仓数据缺少代码或数量")
                if code in broker_positions:
                    add_missing("position.duplicate_code")
                    raise RuntimeError("券商持仓同一证券出现多行，无法核验")
                broker_positions[code] = int(quantity)
                result["broker_positions"] = dict(broker_positions)
                try:
                    sellable = Decimal(str(item.get("sellable")))
                except Exception:
                    sellable = Decimal("NaN")
                if (not sellable.is_finite() or sellable < 0 or
                        sellable != sellable.to_integral_value() or sellable > quantity):
                    broker_sellable[code] = None
                    sellable_missing_codes.append(code)
                    result["sellable_missing_codes"] = sorted(set(sellable_missing_codes))
                    result["broker_sellable"] = dict(broker_sellable)
                    add_missing("position.sellable")
                elif broker_sellable.get(code, 0) is not None:
                    broker_sellable[code] = broker_sellable.get(code, 0) + int(sellable)
                    result["broker_sellable"] = dict(broker_sellable)
            differences = [
                {"code": code, "local_shares": local_positions.get(code, 0),
                 "broker_shares": broker_positions.get(code, 0)}
                for code in sorted(set(local_positions) | set(broker_positions))
                if local_positions.get(code, 0) != broker_positions.get(code, 0)
            ]
            result["position_differences"] = differences
            if any(not item.get("id") or not item.get("status") for item in snapshot["orders"]):
                add_missing("order.id", "order.status")
                raise RuntimeError("券商委托数据缺少编号或状态")
            pending = [item for item in snapshot["orders"]
                       if item["status"] not in ("filled", "cancelled", "rejected")]
            pending_journal = []
            if broker_key:
                pending_journal = list(BrokerOrderReportRepo.list_unfinished(broker_key, broker.account_id))
            unresolved_order_reports = BrokerOrderReportRepo.list_unresolved(
                broker_key, broker.account_id,
            ) if broker_key else []
            trade_report_health = snapshot.get("trade_report_health")
            trade_report_health_error = None
            if broker_key:
                if (not isinstance(trade_report_health, dict) or
                        any(isinstance(trade_report_health.get(key), bool) or
                            not isinstance(trade_report_health.get(key), int) or
                            trade_report_health[key] < 0 for key in
                            ("conflicts", "unpriced", "unbooked", "unassigned")) or
                        ("legacy_quarantined" in trade_report_health and
                         (isinstance(trade_report_health["legacy_quarantined"], bool) or
                          not isinstance(trade_report_health["legacy_quarantined"], int) or
                          trade_report_health["legacy_quarantined"] < 0))):
                    trade_report_health_error = "券商成交回报完整性状态不可用"
                    add_missing("broker.trade_report_health")
                elif trade_report_health["conflicts"]:
                    trade_report_health_error = "券商成交回报存在未解决冲突"
                elif trade_report_health["unpriced"]:
                    trade_report_health_error = "券商成交回报费用不完整，无法核准对账"
                    add_missing("broker.trade_fees")
                elif trade_report_health["unbooked"]:
                    trade_report_health_error = "存在尚未入账的券商成交回报"
                    add_missing("broker.trade_ledger_events")
                elif trade_report_health["unassigned"]:
                    trade_report_health_error = "期初核准后出现无法归属的券商成交回报"
                    add_missing("broker.trade_unassigned")
            pending_local_intent = LocalOrderIntentRepo.has_unresolved()
            result.update({
                "broker_status": "stale" if broker_age > 60 or broker_age < -5 else "fresh",
                "broker_account_id": account["account_id"],
                "broker_identity_verified": identity_verified,
                "broker_as_of": account["as_of"],
                "broker_received_at": broker_received_at.isoformat(timespec="seconds"),
                "broker_snapshot_started_at": snapshot.get("snapshot_started_at"),
                "broker_snapshot_completed_at": snapshot.get("snapshot_completed_at"),
                "broker_age_seconds": round(broker_age, 1),
                "broker_available": available,
                "broker_cash": broker_cash,
                "broker_frozen": broker_frozen,
                "cash_difference": difference,
                "cash_balance_difference": cash_balance_difference,
                "frozen_difference": frozen_difference,
                "broker_positions": broker_positions,
                "broker_sellable": broker_sellable,
                "sellable_missing_codes": sorted(set(sellable_missing_codes)),
                "position_differences": differences,
                "pending_orders": pending,
                "pending_journal_orders": pending_journal,
                "unresolved_order_reports": [
                    {key: row.get(key) for key in (
                        "report_key", "order_id", "symbol", "side", "quantity", "filled_qty",
                        "status", "source", "disposition", "observed_at",
                    )}
                    for row in unresolved_order_reports
                ],
                "pending_local_intent": pending_local_intent,
                "broker_trade_report_health": trade_report_health,
            })
            if not identity_verified:
                add_missing("broker.verified_account_id")
            if local["status"] != "active":
                result["reason"] = "本地账户期初尚未核准"
            elif not integrity["ok"] or integrity["account_version"] != local["version"]:
                result["reason"] = "本地账本现金、持仓、冻结额或事件链不一致"
            elif differences:
                result["reason"] = "券商与本地持仓不一致"
            elif sellable_missing_codes:
                result["reason"] = "券商持仓可卖股数缺失，无法核验实盘委托"
            elif local.get("broker_account_id") and local["broker_account_id"] != account["account_id"]:
                result["reason"] = "券商资金账号与本地绑定不一致"
            elif not identity_verified:
                result["reason"] = "券商接口未提供可核验的资金账号"
            elif trade_report_health_error:
                result["reason"] = trade_report_health_error
            elif pending_local_intent:
                result["reason"] = "存在未完成的本地委托意图"
            elif (difference is None or cash_balance_difference is None or frozen_difference is None or
                  any(value != 0 for value in
                      (difference, cash_balance_difference, frozen_difference))):
                result["reason"] = "券商与本地现金、冻结额或可用资金不一致"
            elif pending:
                result["reason"] = "券商存在未确认委托"
            elif unresolved_order_reports:
                result["reason"] = "券商回报日志中存在未核对委托"
            elif pending_journal:
                result["reason"] = "券商回报日志中存在未终结委托"
            elif broker_age > 60 or broker_age < -5:
                result["reason"] = "券商账户快照已过期或时间异常"
            else:
                latest = LocalLedgerRepo.get_account()
                latest_integrity = LocalLedgerRepo.audit_integrity()
                if any(latest.get(key) != local.get(key) for key in (
                        "status", "cash", "frozen", "version", "broker_account_id")):
                    result["reason"] = "对账期间本地账户已变化，请重新核对"
                elif (not latest_integrity.get("ok") or
                      latest_integrity.get("account_version") != latest.get("version")):
                    result["reason"] = "对账期间本地账本完整性校验失败"
                elif not local.get("broker_account_id"):
                    result["reason"] = "券商与本地数据一致，可核对后绑定资金账号"
                    result["can_bind"] = True
                else:
                    result["reason"] = None
                    result["can_execute"] = True
        except Exception as error:
            result["broker_error"] = str(error)
            configuration_issues = {
                "QMT_ACCOUNT 未配置，请在 .env 中设置 QMT 资金账号":
                    ("QMT 资金账号未配置", "broker.configuration.account"),
                "QMT userdata_mini 目录不存在或不可用":
                    ("QMT userdata_mini 目录不可用", "broker.configuration.userdata_dir"),
                "EASYTRAIDER_USER 未配置，请在 .env 中设置同花顺账号":
                    ("Easytrader 登录账号未配置", "broker.configuration.user"),
                "EASYTRAIDER_ACCOUNT_ID 未配置，拒绝连接未标识的券商账户":
                    ("Easytrader 资金账号未配置", "broker.configuration.easytrader_account"),
                "EASYTRAIDER_ACCOUNT_ID 格式无效":
                    ("Easytrader 资金账号格式无效", "broker.configuration.easytrader_account"),
            }
            configuration_issue = configuration_issues.get(str(error))
            if configuration_issue:
                result["reason"] = configuration_issue[0]
                add_missing(configuration_issue[1])
            if not result["missing_items"]:
                result["missing_items"] = ["broker_snapshot"]
        return result

    @staticmethod
    def bind_broker(expected_version, actor):
        reconciliation = LocalLedgerService.reconcile()
        can_bind = reconciliation.get("can_bind") is True
        already_bound = (reconciliation.get("can_execute") is True and
                         reconciliation.get("bound_broker_account_id") == reconciliation.get("broker_account_id"))
        if not can_bind and not already_bound:
            raise ValueError(reconciliation["reason"] or "券商对账未通过")
        return LocalOrderIntentRepo.bind_broker(
            reconciliation["broker_account_id"], expected_version, actor, reconciliation
        )

    @staticmethod
    def close_signal_remainder(signal_id, expected_version, actor, reason):
        from core.data.repos.rebalance import RebalanceSignalRepo
        from core.shared.config import BROKER_TYPE, BROKER_FAKE

        account = LocalLedgerRepo.get_account()
        reconciliation = None
        prior = Database.fetchone(
            "SELECT event_key FROM local_ledger_events WHERE event_key=%s",
            (f"signal-close:{ACCOUNT_ID}:{signal_id}",),
        )
        if not prior and (BROKER_TYPE != BROKER_FAKE or account.get("broker_account_id")):
            if not account.get("broker_account_id"):
                raise ValueError("真实券商账户尚未绑定，不能关闭部分成交计划")
            reconciliation = LocalLedgerService.reconcile()
            if not reconciliation["can_execute"]:
                raise ValueError(reconciliation["reason"] or "券商对账未通过")
        return RebalanceSignalRepo.close_remainder(
            signal_id, expected_version, actor, reason, reconciliation
        )

    @staticmethod
    def prepare_intent(detail):
        try:
            intent_id = str(uuid.UUID(str(detail["intent_id"])))
            integer_values = []
            for key in ("signal_id", "expected_account_version", "quantity"):
                raw_value = detail[key]
                if isinstance(raw_value, bool):
                    raise ValueError
                exact_value = Decimal(str(raw_value))
                if not exact_value.is_finite() or exact_value != exact_value.to_integral_value():
                    raise ValueError
                integer_values.append(int(exact_value))
            signal_id, expected_version, quantity = integer_values
            price = Decimal(str(detail["limit_price"]))
            fee_cap = Decimal(str(detail["fee_cap"]))
        except (KeyError, TypeError, ValueError, InvalidOperation):
            raise ValueError("委托参数无效") from None
        if signal_id <= 0 or expected_version < 1 or quantity <= 0 or quantity > 100000000:
            raise ValueError("委托信号、版本或数量无效")
        if not price.is_finite() or price <= 0 or price >= Decimal("1000000000") or price != price.quantize(Decimal("0.001")):
            raise ValueError("委托限价须为正数且最多三位小数")
        if not fee_cap.is_finite() or fee_cap < 0 or fee_cap >= Decimal("10000000000") or fee_cap != fee_cap.quantize(Decimal("0.01")):
            raise ValueError("费用上限须为非负且最多两位小数")
        previous = LocalOrderIntentRepo.get_by_id(intent_id)
        if previous:
            broker_account_id = previous["broker_account_id"]
        else:
            from core.data.repos.rebalance import RebalanceSignalRepo
            signal = RebalanceSignalRepo.get_by_id(signal_id, account_id=ACCOUNT_ID)
            if not signal:
                raise ValueError("调仓信号不存在")
            if signal.get("side") in ("buy", "add"):
                from backend.service.strategy_profile import StrategyProfileService
                StrategyProfileService.ensure_execution_allowed(
                    signal.get("profile_id"), signal.get("profile_version"),
                    signal.get("profile_fingerprint"),
                    enforce=signal.get("source_mode") != "live_unmanaged",
                )
            reconciliation = LocalLedgerService.reconcile()
            if not reconciliation["can_execute"]:
                raise ValueError(reconciliation["reason"] or "券商对账未通过")
            broker_account_id = reconciliation["broker_account_id"]
            if signal.get("side") in ("sell", "reduce"):
                sellable = reconciliation["broker_sellable"].get(signal["code"])
                if sellable is None:
                    raise ValueError("券商可卖股数不可核验，不能准备卖出委托")
                if quantity > sellable:
                    raise ValueError("委托股数超过券商当前可卖股数")
        from core.domain.portfolio.decision import decision_fingerprint
        return LocalOrderIntentRepo.prepare({
            "intent_id": intent_id, "signal_id": signal_id,
            "expected_account_version": expected_version, "quantity": quantity,
            "limit_price": price, "fee_cap": fee_cap,
            "broker_account_id": broker_account_id, "actor": detail["actor"],
            "reconciliation": reconciliation if not previous else None,
            "broker_sellable": (reconciliation["broker_sellable"].get(signal["code"])
                                if not previous and signal.get("side") in ("sell", "reduce") else None),
        }, decision_fingerprint)

    @staticmethod
    def _get_report_broker():
        from backend.api.trading import get_broker
        from core.adapters.broker import EasytraderBroker, QmtBroker

        broker = get_broker()
        if isinstance(broker, QmtBroker):
            return broker, "qmt"
        if isinstance(broker, EasytraderBroker):
            return broker, "easytrader"
        raise ValueError("当前券商类型不支持回报核对")

    @staticmethod
    def _get_report_scope():
        from core.shared.config import BROKER_EASYT, BROKER_QMT, BROKER_TYPE, EASYTRAIDER_ACCOUNT_ID, QMT_ACCOUNT

        account_id = LocalLedgerRepo.get_account().get("broker_account_id")
        if account_id:
            if account_id.startswith("QMT-"):
                return "qmt", account_id
            if account_id.startswith("EASYT-") and account_id != "EASYT-UNVERIFIED":
                return "easytrader", account_id
            raise ValueError("本地绑定券商账号标识无法解析，拒绝查询其他账号回报")
        if BROKER_TYPE == BROKER_QMT and str(QMT_ACCOUNT or "").strip():
            return "qmt", f"QMT-{str(QMT_ACCOUNT).strip()}"
        if BROKER_TYPE == BROKER_EASYT and str(EASYTRAIDER_ACCOUNT_ID or "").strip():
            return "easytrader", f"EASYT-{str(EASYTRAIDER_ACCOUNT_ID).strip()}"
        raise ValueError("本地账户尚未绑定，且没有明确券商资金账号，无法查询回报隔离记录")

    @staticmethod
    def list_unresolved_broker_reports(limit=100):
        from core.data.repos.broker_order_report import BrokerOrderReportRepo

        broker_key, broker_account_id = LocalLedgerService._get_report_scope()
        return BrokerOrderReportRepo.list_unresolved(broker_key, broker_account_id, limit)

    @staticmethod
    def list_unresolved_broker_trades(limit=100):
        from core.data.repos.broker_trade_report import BrokerTradeReportRepo

        broker_key, broker_account_id = LocalLedgerService._get_report_scope()
        return BrokerTradeReportRepo.list_unresolved(broker_key, broker_account_id, limit)

    @staticmethod
    def list_broker_report_history(report_type, report_key, limit=100):
        if report_type not in ("order", "trade") or not re.fullmatch(r"[0-9a-f]{64}", report_key):
            raise ValueError("券商回报类型或编号无效")
        broker_key, broker_account_id = LocalLedgerService._get_report_scope()
        return Database.fetchall(
            "SELECT id,report_type,report_key,action,actor,reason,evidence_json,occurred_at "
            "FROM broker_report_resolution_events WHERE broker=%s AND broker_account_id=%s "
            "AND report_type=%s AND report_key=%s ORDER BY id DESC LIMIT %s",
            (broker_key, broker_account_id, report_type, report_key, limit),
        )

    @staticmethod
    def resolve_broker_trade_conflicts(trade_id, actor, reason):
        from core.data.repos.broker_trade_report import BrokerTradeReportRepo

        broker, broker_key = LocalLedgerService._get_report_broker()
        if LocalLedgerService._get_report_scope() != (broker_key, broker.account_id):
            raise ValueError("当前券商账号与本地绑定账号不一致，拒绝解除成交回报冲突")
        evidence_reader = getattr(broker, "get_trade_reconciliation_evidence", None)
        if not callable(evidence_reader):
            raise ValueError("当前券商适配器不支持独立读取成交核对证据")
        evidence = evidence_reader(trade_id)
        return BrokerTradeReportRepo.resolve_conflicts(
            broker_key, broker.account_id, trade_id, evidence, actor, reason,
        )

    @staticmethod
    def replay_broker_reports():
        from core.adapters.broker import QmtBroker

        broker, broker_key = LocalLedgerService._get_report_broker()
        if isinstance(broker, QmtBroker):
            return broker.replay_order_reports()
        orders = broker.get_orders()
        trades = broker.get_trades()
        from core.data.repos.broker_order_report import BrokerOrderReportRepo
        from core.data.repos.broker_trade_report import BrokerTradeReportRepo
        return {
            "observed_orders": len(orders),
            "observed_trades": len(trades),
            "unresolved_reports": BrokerOrderReportRepo.unresolved_count(broker_key, broker.account_id),
            "trade_report_health": BrokerTradeReportRepo.get_health(broker_key, broker.account_id),
        }

    @staticmethod
    def resolve_broker_report(report_key, actor, reason):
        from core.adapters.broker import QmtBroker
        from core.data.repos.broker_order_report import BrokerOrderReportRepo

        broker, broker_key = LocalLedgerService._get_report_broker()
        if LocalLedgerService._get_report_scope() != (broker_key, broker.account_id):
            raise ValueError("当前券商账号与本地绑定账号不一致，拒绝核对委托回报")
        report = BrokerOrderReportRepo.get_by_key(report_key)
        if not report or report["broker"] != broker_key or report["broker_account_id"] != broker.account_id:
            raise ValueError("券商回报不存在或账户不匹配")
        if report.get("resolved_at"):
            return {"report_key": report_key, "already_resolved": True}
        if isinstance(broker, QmtBroker):
            snapshot = broker.get_reconciliation_snapshot(check_report_integrity=False)
            orders = snapshot["orders"]
        else:
            snapshot = broker.get_reconciliation_snapshot()
            evidence_reader = getattr(broker, "get_order_reconciliation_evidence", None)
            if not callable(evidence_reader):
                raise ValueError("当前券商适配器不支持独立读取委托核对证据")
            orders = [evidence_reader(report["order_id"])]
        if (snapshot.get("identity_verified") is not True or
                snapshot["account"]["account_id"] != broker.account_id):
            raise ValueError("券商资金账号与回报不一致")
        matching = [order for order in orders if order["id"] == report["order_id"]]
        if len(matching) != 1:
            raise ValueError("券商当前委托清单缺少唯一对应记录")
        current = matching[0]
        if current["status"] not in ("filled", "cancelled", "rejected"):
            raise ValueError("券商委托尚未终结")
        return BrokerOrderReportRepo.resolve(
            report_key, broker_key, broker.account_id, current, actor, reason,
        )
