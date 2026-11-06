"""
调仓服务 — 选股保存 + 每日止盈止损 + 频率调仓 + 信号执行追踪
"""
import datetime
import json
import logging
import math
import uuid
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

logger = logging.getLogger(__name__)


def _make_batch_id():
    return uuid.uuid4().hex[:12]


def _serialize_holding_date(value):
    return value.isoformat() if isinstance(value, (datetime.date, datetime.datetime)) else value


# 市场状态 → 止盈止损参数 (索引顺序与 STATE_NAMES 一致)
_STATE_NAMES = ["volatile_up", "volatile_down", "volatile_flat", "steady_up", "steady_down", "steady_flat"]
_DEFAULT_REGIME_STOPS = {
    "volatile_up":    {"warn": -0.04, "full": -0.08, "profitHalf": 0.12, "profitFull": 0.25},
    "volatile_down":  {"warn": -0.05, "full": -0.10, "profitHalf": 0.15, "profitFull": 0.25},
    "volatile_flat":  {"warn": -0.05, "full": -0.10, "profitHalf": 0.15, "profitFull": 0.30},
    "steady_up":      {"warn": -0.07, "full": -0.14, "profitHalf": 0.20, "profitFull": 0.40},
    "steady_down":    {"warn": -0.06, "full": -0.12, "profitHalf": 0.18, "profitFull": 0.35},
    "steady_flat":    {"warn": -0.05, "full": -0.10, "profitHalf": 0.15, "profitFull": 0.30},
}


def _load_regime_stops():
    """从 config/regime_ratio.json 加载每状态的止盈止损参数, 回退到默认值"""
    import json
    from pathlib import Path
    stops = dict(_DEFAULT_REGIME_STOPS)
    try:
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "regime_ratio.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            raw_stops = cfg.get("stops", {})
            for k, v in raw_stops.items():
                idx = int(k)
                if 0 <= idx < len(_STATE_NAMES):
                    name = _STATE_NAMES[idx]
                    _w = float(v.get("warn", 5))
                    _f = float(v.get("full", 10))
                    if _w == 0 or _f == 0:
                        logger.warning(f"[RegimeStops] 止损参数含0值 (idx={k}), 使用默认值")
                        continue
                    stops[name] = {
                        "warn": -abs(_w) / 100,
                        "full": -abs(_f) / 100,
                        "profitHalf": float(v.get("profitHalf", 15)) / 100,
                        "profitFull": float(v.get("profitFull", 30)) / 100,
                    }
    except Exception as e:
        logger.warning(f"[RegimeStops] 加载止盈止损配置失败, 使用默认值: {e}")
        pass
    return stops


def _get_regime_stop_params(regime_info, user_stop_warn, user_stop_full,
                             user_take_half, user_take_full):
    """根据当前市场状态获取止盈止损参数 (用户设置优先, 否则用 regime 预配置)"""
    if regime_info:
        regime_name = regime_info.get("regime", "")
        defaults = _load_regime_stops().get(regime_name, _DEFAULT_REGIME_STOPS.get("steady_flat", {}))
    else:
        defaults = _DEFAULT_REGIME_STOPS["steady_flat"]
    return {
        "stop_loss_warn": user_stop_warn if user_stop_warn is not None else defaults.get("warn", -0.05),
        "stop_loss_full": user_stop_full if user_stop_full is not None else defaults.get("full", -0.10),
        "take_profit_half": user_take_half if user_take_half is not None else defaults.get("profitHalf", 0.15),
        "take_profit_full": user_take_full if user_take_full is not None else defaults.get("profitFull", 0.30),
        "source": "user" if user_stop_warn is not None else ("regime:" + regime_info.get("regime", "unknown") if regime_info else "default"),
    }


def _get_prices_batch(codes):
    """获取可执行的实时价格，不允许回退到历史收盘价。"""
    from backend.service.market_data import MarketDataService

    prices, meta = MarketDataService.get_live_prices(codes, use_case="execution")
    if meta.get("freshness") != "fresh":
        logger.warning(
            "[Prices] 实时行情不可用，禁止使用历史价格: missing=%s freshness=%s",
            meta.get("missing_codes", []), meta.get("freshness"),
        )
        return {}, "unavailable"
    return prices, "realtime"


def _fetch_fundamental_quality(codes, as_of=None):
    """取候选最近一期 (PIT ann_date) 基本面质量指标 {code: {profit_dedt, debt_to_assets}}"""
    from backend.service.portfolio_decision import load_fundamental_quality
    from core.shared.trading_time import get_latest_trading_day
    return load_fundamental_quality(codes, as_of or get_latest_trading_day())


def _fetch_recent_amount(codes, as_of=None):
    """取候选20日成交额中位数 {code: 元}"""
    from backend.service.portfolio_decision import load_amount_median_20d
    from core.shared.trading_time import get_latest_trading_day
    return load_amount_median_20d(codes, as_of or get_latest_trading_day())


def _resolve_position_ratio(explicit_ratio, fallback_ratio, regime_info):
    regime_ratio = None
    if isinstance(regime_info, dict):
        regime_ratio = regime_info.get("position")
        if regime_ratio is None:
            regime_ratio = (regime_info.get("details") or {}).get("position_ratio")
    if explicit_ratio is not None:
        ratio, source = explicit_ratio, "explicit"
    elif regime_ratio is not None:
        ratio, source = regime_ratio, "unified_regime"
    else:
        ratio, source = fallback_ratio, "timing_fallback"
    return max(0.0, min(0.95, float(ratio or 0.0))), source


def _round_buy_order(amount, price):
    price = float(price or 0)
    shares = int(float(amount or 0) / price / 100) * 100 if price > 0 else 0
    return shares, round(shares * price, 2)


def _load_half_sold(account_id=None):
    """加载已减半持仓集合 {code} — 跨日持久化, 避免 warn 档每天重复减半。"""
    try:
        from core.data.repos.settings import SettingsRepo
        from core.data.repos.local_ledger import ACCOUNT_ID, account_half_sold_setting_key
        import json as _json
        _raw = SettingsRepo.get(account_half_sold_setting_key(ACCOUNT_ID if account_id is None else account_id), None)
        if not _raw:
            return set()
        parsed = _json.loads(_raw) if isinstance(_raw, str) else _raw
        return set(parsed)
    except Exception as e:
        logger.debug(f"[减半] 加载已减半状态失败: {e}")
        return set()


def _check_risk_signal(code, name, shares, cost_price, current_price, market_value,
                       stop_loss_warn, stop_loss_full, take_profit_half, take_profit_full,
                       already_half=False):
    """阶梯止盈止损判断 — 委托 classify_risk_action 纯函数, 构造 signal dict。
    already_half=True (已减半) 时不再触发减半, 只判清仓档。"""
    from core.domain.portfolio.allocator import classify_risk_action
    if shares <= 0 or cost_price <= 0 or current_price <= 0:
        return None
    pnl_pct = (current_price - cost_price) / cost_price
    action, reason = classify_risk_action(
        pnl_pct, stop_loss_warn, stop_loss_full,
        take_profit_half, take_profit_full,
        already_half=already_half,
    )
    if action is None:
        return None

    is_full = action == "full"
    is_stop = pnl_pct <= stop_loss_full or (not is_full and pnl_pct <= stop_loss_warn)
    planned_shares = shares if is_full else min(shares, max(100, int(shares * 0.5 / 100) * 100))
    return {
        "code": code, "name": name,
        "side": "sell" if is_full else "reduce",
        "action": reason.split(" (")[0],  # 止损清仓/止盈清仓/止损减半/止盈减半
        "current_weight": 0, "target_weight": 0, "delta_weight": 0,
        "amount": round(planned_shares * current_price, 2),
        "urgency": "stop_loss" if is_stop else "take_profit",
        "reason": reason,
        "current_price": round(current_price, 3), "cost_price": round(cost_price, 3),
        "pnl_pct": round(pnl_pct, 4), "current_shares": shares,
        "suggested_shares": planned_shares,
        "planned_shares": planned_shares,
        "current_market_value": round(market_value, 2),
    }


class RebalanceService:

    # ─── 选股保存 ───

    @classmethod
    def save_screening(cls, job_id, save_name, note="", screening_result=None):
        from core.data import ScreeningSaveRepo

        result = screening_result
        if (not isinstance(result, dict) or result.get("status") != "done"
                or result.get("result_id") != str(job_id)):
            return {"error": "只有服务端最终完成且绑定不可变 result_id 的选股结果可保存"}
        prediction = result.get("prediction") or {}
        provenance = result.get("provenance") or {}
        snapshot_fingerprint = result.get("snapshot_fingerprint")
        if (prediction.get("fresh") is not True
                or prediction.get("decision_date") != prediction.get("prediction_date")
                or result.get("eval_date") != prediction.get("decision_date")
                or not isinstance(snapshot_fingerprint, str)
                or provenance.get("snapshot_fingerprint") != snapshot_fingerprint):
            return {"error": "选股结果缺少匹配的日期或快照来源证明，不能保存"}

        candidates = result.get("candidates", [])
        if not candidates:
            return {"error": "选股结果中没有 candidates"}

        pool = result.get("pool", "all")
        from core.shared.trading_time import get_latest_trading_day
        eval_date = result.get("eval_date") or get_latest_trading_day().isoformat()
        direction = result.get("direction")
        recommended = result.get("recommended")
        params_json = dict(result.get("provenance") or {})
        params_json.update({
            "request_params": result.get("request_params") or params_json.get("request_params"),
            "model_routing": result.get("model_routing") or params_json.get("model_routing"),
            "data_cutoff": result.get("data_cutoff") or params_json.get("data_cutoff") or eval_date,
            "direction": direction,
            "snapshot_fingerprint": snapshot_fingerprint,
        })
        profile = result.get("strategy_profile") or {}

        save_id = ScreeningSaveRepo.save(
            save_name=save_name,
            job_id=job_id,
            pool=pool,
            eval_date=eval_date,
            candidates=candidates,
            direction=direction,
            recommended=recommended,
            params_json=params_json,
            note=note,
            profile_id=profile.get("profile_id"),
            profile_version=profile.get("version"),
            profile_fingerprint=profile.get("fingerprint"),
            data_fingerprint=profile.get("data_fingerprint"),
            model_fingerprint=profile.get("model_fingerprint"),
            factor_fingerprint=profile.get("factor_fingerprint"),
        )

        logger.info(f"[调仓] 已保存选股结果: #{save_id} '{save_name}', {len(candidates)} 只候选")
        return {"save_id": save_id, "save_name": save_name, "candidate_count": len(candidates)}

    @classmethod
    def list_saves(cls, days=30, pool=None, limit=20):
        from core.data import ScreeningSaveRepo
        rows = ScreeningSaveRepo.get_all(days=days, pool=pool, limit=limit)
        for r in rows:
            if isinstance(r.get("eval_date"), datetime.date):
                r["eval_date"] = r["eval_date"].isoformat()
        return rows

    @classmethod
    def get_save(cls, save_id):
        from core.data import ScreeningSaveRepo
        r = ScreeningSaveRepo.get_by_id(save_id)
        if r and isinstance(r.get("eval_date"), datetime.date):
            r["eval_date"] = r["eval_date"].isoformat()
        return r

    @classmethod
    def delete_save(cls, save_id):
        from core.data import ScreeningSaveRepo
        ScreeningSaveRepo.delete(save_id)

    @classmethod
    def get_cooldown(cls):
        from core.data import Database, SettingsRepo
        import json as _json

        raw = SettingsRepo.get("rebalance_cooldown", None)
        if not raw:
            return {"as_of": None, "items": []}
        parsed = _json.loads(raw) if isinstance(raw, str) else raw
        as_of = parsed.get("as_of") if isinstance(parsed, dict) else None
        periods = parsed.get("periods", parsed) if isinstance(parsed, dict) else {}
        periods = {str(code): int(value) for code, value in periods.items() if int(value) > 0}
        if not periods:
            return {"as_of": as_of, "items": []}

        placeholders = ",".join(["%s"] * len(periods))
        rows = Database.fetchall(
            f"SELECT code, name FROM rebalance_signals WHERE code IN ({placeholders}) "
            "AND name<>'' ORDER BY id DESC",
            tuple(periods.keys()),
        )
        names = {}
        for row in rows:
            names.setdefault(row["code"], row.get("name", ""))
        return {
            "as_of": as_of,
            "items": [
                {"code": code, "name": names.get(code, ""), "remaining_periods": remaining}
                for code, remaining in sorted(periods.items())
            ],
        }

    # ─── 每日止盈止损检查 ───

    @classmethod
    def daily_risk_check(cls, stop_loss_warn=None, stop_loss_full=None,
                         take_profit_half=None, take_profit_full=None, regime_info=None):
        """阶梯止盈止损检查 — 两档止损 + 两档止盈; 参数为None时按当前市场状态自动选"""
        from core.data import HoldingsRepo, RebalanceSignalRepo
        from core.data.repos.local_ledger import ACCOUNT_ID, LocalLedgerRepo

        # 按市场状态解析止盈止损参数
        params = _get_regime_stop_params(regime_info, stop_loss_warn, stop_loss_full,
                                          take_profit_half, take_profit_full)
        _sw = params["stop_loss_warn"]
        _sf = params["stop_loss_full"]
        _th = params["take_profit_half"]
        _tf = params["take_profit_full"]

        account = LocalLedgerRepo.get_account()
        if account["status"] == "active":
            try:
                integrity = LocalLedgerRepo.audit_integrity()
            except Exception as error:
                logger.error("[调仓] 本地账本完整性校验异常，停止每日风险判断: %s", error)
                integrity = None
            if (not isinstance(integrity, dict) or not integrity.get("ok") or
                    integrity.get("account_version") != account.get("version")):
                return {
                    "signals": [], "message": "本地账本完整性校验失败，已跳过风险判断",
                    "data_status": "unavailable", "execution_blocked": True,
                    "execution_block_reason": "本地账本不一致，禁止生成可执行风险信号",
                }
        holdings = HoldingsRepo.get_all(account_id=ACCOUNT_ID)
        if not holdings:
            return {"signals": [], "message": "无持仓数据"}

        codes = [h["code"] for h in holdings]
        prices, price_source = _get_prices_batch(codes)
        if price_source != "realtime":
            return {
                "signals": [],
                "message": "实时行情不可用，已跳过风险判断",
                "data_status": "unavailable",
            }

        batch_id = _make_batch_id()
        signals = []
        from core.shared.trading_time import market_date
        today = market_date()
        half_sold = _load_half_sold()

        for h in holdings:
            code = h["code"]
            shares = int(h.get("shares", 0))
            cost_price = float(h.get("cost_price", 0) or 0)
            if shares <= 0 or cost_price <= 0:
                continue
            current_price = prices.get(code, 0)
            market_value = shares * current_price

            signal = _check_risk_signal(
                code, h.get("name", ""), shares, cost_price, current_price, market_value,
                _sw, _sf, _th, _tf, already_half=(code in half_sold))
            if signal:
                signals.append(signal)

        from core.data import RebalanceSessionRepo
        _mv = sum(int(h.get("shares", 0)) * float(prices.get(h["code"], 0) or 0) for h in holdings)
        available_cash = float(account["available"]) if account["available"] is not None else None
        total_cash = float(account["cash"]) if account.get("cash") is not None else None
        frozen_cash = float(account["frozen"]) if account.get("frozen") is not None else None
        session_payload = {
            "batch_id": batch_id, "check_type": "daily_risk", "eval_date": today.isoformat(),
            "account_json": {"total_assets": round(_mv + total_cash, 2) if total_cash is not None else None,
                             "cash": available_cash, "market_value": round(_mv, 2),
                             "total_cash": total_cash, "frozen_cash": frozen_cash,
                             "cash_source": "local_ledger" if available_cash is not None else "unapproved"},
            "holdings_json": [{"code": h["code"], "name": h.get("name", ""), "shares": h["shares"],
                               "cost_price": float(h.get("cost_price", 0) or 0),
                               "buy_date": _serialize_holding_date(h.get("buy_date")),
                               "current_price": prices.get(h["code"], 0)} for h in holdings],
            "signals_json": signals,
            "stop_loss_pct": _sf, "take_profit_half": _th, "take_profit_full": _tf,
            "regime_json": regime_info,
            "account_id": ACCOUNT_ID, "account_version": account["version"],
        }
        result = {
            "batch_id": batch_id,
            "check_date": today.isoformat(),
            "signals": signals,
            "total_holdings": len(holdings),
            "triggered": len(signals),
            "regime_stop_params": {"stop_loss_warn": _sw, "stop_loss_full": _sf,
                                   "take_profit_half": _th, "take_profit_full": _tf,
                                   "source": params["source"]},
        }
        if account["status"] != "active":
            result["execution_blocked"] = True
            result["execution_block_reason"] = "本地账户期初尚未核准，当前仅能预览"
            return result
        try:
            from core.domain.portfolio.decision import decision_fingerprint, holdings_fingerprint_rows
            holdings_fingerprint = decision_fingerprint(holdings_fingerprint_rows(holdings))
            RebalanceSignalRepo.batch_create(
                batch_id=batch_id, screening_save_id=None,
                check_type="daily_risk", signals=signals, replace_pending=True,
                input_fingerprint=holdings_fingerprint,
                fingerprint_fn=decision_fingerprint,
                session_payload=session_payload,
                account_id=ACCOUNT_ID, account_version=account["version"])
            if signals:
                logger.info(f"[调仓] 每日检查生成 {len(signals)} 条风险信号, batch={batch_id}")
        except ValueError as e:
            return {"signals": [], "message": str(e),
                    "data_status": "unavailable", "triggered": 0}
        except Exception as e:
            logger.error(f"[调仓] 风险信号持久化失败 (batch={batch_id}): {e}")
            return {"signals": [], "message": "风险信号保存失败，请稍后重试",
                    "data_status": "unavailable", "triggered": 0}

        return result

    # ─── 调仓检查 ───

    @classmethod
    def _saved_commit_result(cls, previous, plan_fingerprint):
        if not previous:
            return None
        if previous.get("plan_fingerprint") != plan_fingerprint.lower():
            return {"error": "提交请求 ID 已用于另一份调仓方案"}
        from core.data import RebalanceSignalRepo
        signals = RebalanceSignalRepo.get_by_batch(previous["batch_id"], account_id="LOCAL-LIVE")
        return {"batch_id": previous["batch_id"], "check_date": str(previous["eval_date"]),
                "screening_save_id": previous.get("screening_save_id"),
                "account": previous.get("account_json"), "holdings": previous.get("holdings_json"),
                "signals": signals, "constraints": previous.get("constraints_json") or [],
                "price_source": "saved_commit", "plan_fingerprint": plan_fingerprint.lower(),
                "preview_only": False, "committed": True, "already_committed": True}

    @classmethod
    def run_rebalance(cls, screening_save_id, top_n=5, sizing="equal",
                      frequency="weekly", code_prefixes=None,
                      stop_loss_pct=None, take_profit_half=None, take_profit_full=None,
                      cooldown_stop_weeks=None, cooldown_profit_weeks=None,
                      cash=None, timing_ratio=None, panic=None, regime_info=None,
                      decision_id=None, enforce_strategy_profile=True,
                      unmanaged_reason=None, unmanaged_actor=None, preview_only=True,
                      commit_request_id=None, expected_plan_fingerprint=None):
        from core.data import HoldingsRepo, ScreeningSaveRepo, RebalanceSignalRepo
        from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor
        from core.shared.trading_time import get_latest_trading_day

        unmanaged_reason = str(unmanaged_reason or "").strip()
        if not enforce_strategy_profile and (len(unmanaged_reason) < 8 or not unmanaged_actor):
            return {"error": "人工接管必须由管理员发起并填写至少8个字符的理由"}
        if not preview_only:
            if not commit_request_id or not expected_plan_fingerprint:
                return {"error": "调仓提交必须携带请求 ID 和预览方案指纹"}
            try:
                normalized_request_id = str(uuid.UUID(str(commit_request_id)))
            except (TypeError, ValueError, AttributeError):
                return {"error": "提交请求 ID 必须是有效 UUID"}
            if (not isinstance(expected_plan_fingerprint, str) or len(expected_plan_fingerprint) != 64 or
                    any(char not in "0123456789abcdef" for char in expected_plan_fingerprint.lower())):
                return {"error": "提交必须携带有效的预览方案指纹"}
            from core.data.repos.session import RebalanceSessionRepo
            previous = RebalanceSessionRepo.get_by_commit_request(normalized_request_id, "LOCAL-LIVE")
            if previous:
                return cls._saved_commit_result(previous, expected_plan_fingerprint)
            commit_request_id = normalized_request_id

        decision_as_of = get_latest_trading_day()
        decision = None

        from core.data.repos.local_ledger import LocalLedgerRepo, ACCOUNT_ID
        local_account = LocalLedgerRepo.get_account()
        ledger_integrity = None
        if local_account["status"] == "active":
            try:
                ledger_integrity = LocalLedgerRepo.audit_integrity()
            except Exception as error:
                logger.error("[调仓] 本地账本完整性校验异常，仅允许预览: %s", error)
        manual_budget = None
        try:
            from core.data import SettingsRepo
            saved_budget = SettingsRepo.get("rebalance_cash")
            manual_budget = float(saved_budget) if saved_budget not in (None, "") else None
            if manual_budget is not None and (not math.isfinite(manual_budget) or manual_budget < 0):
                logger.warning("[调仓] 保存的人工预算无效，未用于预览")
                manual_budget = None
        except Exception:
            logger.warning("[调仓] 读取人工预算失败，未用于预览", exc_info=True)
        explicit_budget = cash is not None
        if explicit_budget:
            try:
                cash = float(cash)
            except (TypeError, ValueError):
                return {"error": "预览现金预算必须为有效非负金额"}
            if not math.isfinite(cash) or cash < 0:
                return {"error": "预览现金预算必须为有效非负金额"}
        account_block_reason = None
        ledger_cash_valid = (
            local_account["status"] == "active" and
            isinstance(ledger_integrity, dict) and ledger_integrity.get("ok") and
            ledger_integrity.get("account_version") == local_account.get("version") and
            local_account.get("available") is not None
        )
        if local_account["status"] != "active":
            account_block_reason = "本地账户期初尚未核准，当前仅能预览"
        elif not ledger_cash_valid:
            account_block_reason = "本地账本完整性校验失败，当前仅能预览"
        elif explicit_budget:
            account_block_reason = "显式现金仅用于预览，当前方案不可执行"
        cash = (cash if explicit_budget else
                float(local_account["available"]) if ledger_cash_valid else
                max(manual_budget or 0.0, 0.0))

        save = ScreeningSaveRepo.get_by_id(screening_save_id)
        if not save:
            return {"error": f"选股保存 #{screening_save_id} 不存在"}

        profile = None
        profile_info = None
        execution_block_reason = None
        if save.get("profile_id"):
            try:
                from backend.service.strategy_profile import StrategyProfileService
                profile = StrategyProfileService.resolve(save["profile_id"], save["profile_version"])
                if profile["fingerprint"] != save.get("profile_fingerprint"):
                    return {"error": "选股保存的Profile指纹已失配，拒绝生成调仓方案"}
                locked = StrategyProfileService.bind_operation(profile, "rebalance")
                top_n = locked["top_n"]
                sizing = locked["sizing"]
                frequency = locked["frequency"]
                code_prefixes = locked["code_prefixes"]
                cooldown_stop_weeks = locked["cooldown_stop_weeks"]
                cooldown_profit_weeks = locked["cooldown_profit_weeks"]
                profile_info = {
                    "profile_id": profile["profile_id"], "version": profile["version"],
                    "status": profile["status"], "fingerprint": profile["fingerprint"],
                    "data_fingerprint": save.get("data_fingerprint"),
                    "model_fingerprint": save.get("model_fingerprint"),
                    "factor_fingerprint": save.get("factor_fingerprint"),
                }
                if enforce_strategy_profile:
                    try:
                        StrategyProfileService.ensure_execution_allowed(
                            profile["profile_id"], profile["version"], profile["fingerprint"],
                        )
                    except ValueError as error:
                        execution_block_reason = str(error)
                if str(save.get("eval_date"))[:10] != decision_as_of.isoformat():
                    execution_block_reason = "选股快照不是最新有效交易日，禁止新增买入"
                from core.data import StrategyProfileRepo
                data_snapshot = StrategyProfileRepo.get_snapshot(save.get("data_fingerprint"))
                if (not StrategyProfileService.snapshot_has_verified_pit(data_snapshot)
                        or data_snapshot.get("freshness_status") != "fresh"
                        or data_snapshot.get("security_reference_status") != "pass"):
                    execution_block_reason = "选股快照的数据指纹不可追溯，或PIT/历史证券状态校验失败"
            except ValueError as error:
                return {"error": str(error)}
        elif enforce_strategy_profile:
            try:
                from backend.service.strategy_profile import StrategyProfileService
                StrategyProfileService.ensure_execution_allowed(None, None, None)
            except ValueError as error:
                execution_block_reason = str(error)

        candidates = save.get("candidates", [])
        if decision_id:
            from backend.service.portfolio_decision import PortfolioDecisionService
            decision = PortfolioDecisionService.get(decision_id)
            if not decision:
                return {"error": f"决策快照 {decision_id} 不存在"}
            if int(decision.get("screening_save_id") or 0) != int(screening_save_id):
                return {"error": "决策快照与选股保存不匹配"}
            from core.domain.portfolio.decision import decision_can_execute
            if not decision_can_execute(decision):
                return {"error": "未激活的决策策略仅用于预览和评估，不能生成实盘调仓信号"}
            candidates = [
                row for row in decision.get("ranked_candidates", [])
                if not row.get("hard_excluded")
            ]
            top_n = len(decision.get("selected", [])) or top_n
            try:
                decision_as_of = datetime.date.fromisoformat(decision["data_cutoff"])
            except (KeyError, TypeError, ValueError):
                return {"error": "决策快照缺少有效的数据截止日"}
            if decision_as_of != get_latest_trading_day():
                return {"error": "决策快照已过期，请重新生成后再执行调仓"}
        if not candidates:
            return {"error": "保存的选股结果中没有 candidates"}

        pool = save.get("pool", "all")

        # 用户勾选过滤
        if not decision and code_prefixes and isinstance(code_prefixes, list) and len(code_prefixes) > 0:
            prefix_set = set(code_prefixes)
            candidates = [c for c in candidates if c.get("code") in prefix_set]
            if not candidates:
                return {"error": "勾选的股票不在候选列表中"}

        # 兼容非快照调用；风险规则始终生效，候选不足时减少持仓数而不是跳过过滤。
        filtered_out = []
        if decision:
            filtered_out = [
                {
                    "code": row.get("code"),
                    "name": row.get("name", ""),
                    "reason": row.get("exclusion_reason", "硬风险排除"),
                }
                for row in decision.get("excluded", [])
            ]
        if not decision:
            from core.domain.screening.exclusion import is_excluded, prefilter_by_liquidity_capacity, prefilter_by_quality
            _name_map = {c.get("code"): c.get("name", "") for c in candidates}
            _codes = []
            reason_labels = {"st_stock": "ST股票", "beijing_exchange": "北交所股票不在当前交易范围"}
            for candidate in candidates:
                code = candidate.get("code")
                excluded, reason = is_excluded(code, candidate.get("name", ""))
                if excluded:
                    filtered_out.append({
                        "code": code,
                        "name": candidate.get("name", ""),
                        "reason": reason_labels.get(reason, reason),
                    })
                else:
                    _codes.append(code)
            try:
                quality_data = _fetch_fundamental_quality(_codes, decision_as_of)
                for _code, _values in quality_data.items():
                    _values["name"] = _name_map.get(_code) or _values.get("name", "")
                amount_data = _fetch_recent_amount(_codes, decision_as_of)
                _kept, _dropped = prefilter_by_quality(_codes, quality_data, require_complete=True)
                _kept, _dropped2, _ = prefilter_by_liquidity_capacity(_kept, amount_data)
                _dropped = _dropped + _dropped2
                _kept_set = set(_kept)
                _filtered = [c for c in candidates if c.get("code") in _kept_set]
                filtered_out.extend({"code": cd, "name": _name_map.get(cd, ""), "reason": rs}
                                    for cd, rs in _dropped)
                candidates = _filtered
            except Exception as e:
                logger.warning(f"[调仓] 质量/流动性过滤不可用，阻断新增候选: {e}")
                filtered_out.extend({
                    "code": candidate.get("code"),
                    "name": candidate.get("name", ""),
                    "reason": "质量/流动性数据不可用，禁止新增风险",
                } for candidate in candidates if candidate.get("code") in _codes)
                candidates = []

        # 按频率调整参数
        freq_config = {
            "daily": {"stop_loss": -0.03, "take_profit_half": 0.10, "take_profit_full": 0.20,
                       "max_turnover": 0.20},
            "weekly": {"stop_loss": -0.08, "take_profit_half": 0.25, "take_profit_full": 0.50,
                        "max_turnover": 0.30},
            "biweekly": {"stop_loss": -0.12, "take_profit_half": 0.30, "take_profit_full": 0.60,
                          "max_turnover": 0.60},
            "monthly": {"stop_loss": -0.15, "take_profit_half": 0.35, "take_profit_full": 0.70,
                         "max_turnover": 0.70},
        }
        fc = freq_config.get(frequency, freq_config["weekly"])

        _sl = stop_loss_pct if stop_loss_pct is not None else fc["stop_loss"]
        _tp_half = take_profit_half if take_profit_half is not None else fc["take_profit_half"]
        _tp_full = take_profit_full if take_profit_full is not None else fc["take_profit_full"]

        # 按市场状态解析止盈止损 (regime_info 在后面的 async fetch 中获取, 此处先用频率默认)
        # 实际应用时 regime_info 已可用, 在止盈止损检查前重新解析

        # 构建 advisor
        _sl_warn = _sl / 2 if _sl else -0.05
        _advisor_kwargs = dict(
            top_n=top_n,
            pool=pool,
            sizing=sizing,
            code_prefixes=code_prefixes,
            freq=frequency,
            stop_loss_warn=_sl_warn,
            stop_loss_full=_sl,
            max_turnover=fc["max_turnover"],
        )
        if cooldown_stop_weeks is not None:
            _advisor_kwargs["cooldown_stop_weeks"] = cooldown_stop_weeks
        if cooldown_profit_weeks is not None:
            _advisor_kwargs["cooldown_profit_weeks"] = cooldown_profit_weeks
        advisor = WeeklyRebalanceAdvisor(**_advisor_kwargs)
        advisor.take_profit_half = _tp_half
        advisor.take_profit_full = _tp_full
        advisor._allocator.config.take_profit_half = _tp_half
        advisor._allocator.config.take_profit_full = _tp_full

        # 获取持仓和目标候选的可执行实时价。
        holdings_rows = HoldingsRepo.get_all(account_id=ACCOUNT_ID)
        codes = list(dict.fromkeys(
            [h["code"] for h in holdings_rows]
            + [c.get("code") for c in candidates[:top_n] if c.get("code")]
        ))
        prices, price_source = _get_prices_batch(codes)
        if codes and price_source != "realtime":
            return {
                "error": "实时行情不可用，无法生成可执行调仓建议",
                "data_status": "unavailable",
                "price_source": price_source,
                "orders": [],
                "classification": {},
                "holdings": [],
            }

        holdings_data = []
        total_market_value = 0
        for h in holdings_rows:
            code = h["code"]
            shares = int(h.get("shares", 0))
            cost_price = float(h.get("cost_price", 0) or 0)
            current_price = prices.get(code, 0)
            market_value = shares * current_price
            total_market_value += market_value
            pnl_pct = (current_price - cost_price) / cost_price if cost_price > 0 else 0
            holdings_data.append({
                "code": code,
                "name": h.get("name", ""),
                "shares": shares,
                "cost_price": cost_price,
                "buy_date": _serialize_holding_date(h.get("buy_date")),
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": market_value - (shares * cost_price),
                "pnl_pct": pnl_pct,
            })

        total_cash = (float(local_account["cash"])
                      if ledger_cash_valid and not explicit_budget else cash)
        frozen_cash = (float(local_account["frozen"])
                       if ledger_cash_valid and not explicit_budget else None)
        total_assets = total_market_value + (total_cash or 0)
        account_data = {
            "total_assets": round(total_assets, 2),
            "cash": cash or 0,
            "available_cash": (float(local_account["available"])
                               if ledger_cash_valid and not explicit_budget else None),
            "total_cash": total_cash,
            "frozen_cash": frozen_cash,
            "market_value": round(total_market_value, 2),
            "account_id": ACCOUNT_ID,
            "account_version": local_account["version"],
            "local_status": local_account["status"],
            "local_cash": float(local_account["available"]) if ledger_cash_valid else None,
            "manual_budget": manual_budget,
            "cash_source": "preview_budget" if account_block_reason else "local_ledger",
        }
        if profile_info and not account_block_reason and not preview_only:
            try:
                from core.data import StrategyProfileRepo
                from backend.service.strategy_profile import StrategyProfileService
                StrategyProfileRepo.record_runtime_equity(
                    profile_info["profile_id"], profile_info["version"], total_assets,
                    decision_as_of.isoformat(),
                )
                StrategyProfileService.ensure_execution_allowed(
                    profile_info["profile_id"], profile_info["version"], profile_info["fingerprint"],
                    enforce=enforce_strategy_profile,
                )
            except Exception as error:
                execution_block_reason = f"实盘净值风控校验失败，禁止新增买入：{error}"

        # 获取择时/恐慌/市场状态 (已有则跳过; 未传则8秒超时拿, 超时用默认值)
        _ctx = {"timing_ratio": timing_ratio, "panic": panic, "regime_info": regime_info}
        if timing_ratio is None or panic is None or regime_info is None:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
            def _fetch_slow():
                try:
                    from core.domain.timing.model import get_timing_ratio
                    tr = get_timing_ratio()
                    _ctx["timing_ratio"] = tr if tr else 1.0
                except Exception as e:
                    logger.warning(f"[Rebalance] 获取择时比例失败: {e}")
                try:
                    from core.domain.portfolio.consistency import check_panic
                    _ctx["panic"] = check_panic()
                except Exception as e:
                    logger.warning(f"[Rebalance] 恐慌检测失败: {e}")
                try:
                    from core.domain.regime.classifier import get_current_regime
                    raw = get_current_regime()
                    if raw is not None:
                        d = raw.to_dict() if hasattr(raw, 'to_dict') else raw
                        det = d.get("details", {})
                        d["position"] = det.get("position_ratio", 0.5)
                        d["vol_scalar"] = det.get("vol_scalar", 1.0)
                        _ctx["regime_info"] = d
                except Exception as e:
                    logger.warning(f"[Rebalance] 获取市场状态失败: {e}")
            _pool = ThreadPoolExecutor(max_workers=1)
            try:
                _fut = _pool.submit(_fetch_slow)
                try:
                    _fut.result(timeout=8)
                except FutTimeout:
                    pass
            finally:
                _pool.shutdown(wait=False)
        fallback_timing_ratio = _ctx.get("timing_ratio", 1.0) or 1.0
        panic = _ctx.get("panic", False) or False
        regime_info = _ctx.get("regime_info")
        timing_ratio, position_source = _resolve_position_ratio(
            timing_ratio, fallback_timing_ratio, regime_info,
        )
        if panic:
            timing_ratio = 0.0
            position_source = "panic_override"

        # 统一止盈止损阈值: 与下方补充检查共用 regime 配置 (对齐回测 stop_source=regime),
        # 用户显式参数优先 — 覆盖 advisor 创建时用的频率默认阈值, 避免两套阈值并存"先触发者赢"
        _rsp = _get_regime_stop_params(regime_info, None, None, None, None)
        _rsw, _rsf, _rth, _rtf = _rsp["stop_loss_warn"], _rsp["stop_loss_full"], _rsp["take_profit_half"], _rsp["take_profit_full"]
        _final_sl = stop_loss_pct if stop_loss_pct is not None else _rsf
        _final_th = take_profit_half if take_profit_half is not None else _rth
        _final_tf = take_profit_full if take_profit_full is not None else _rtf
        advisor._allocator.config.stop_loss_warn = _rsw
        advisor._allocator.config.stop_loss_full = _final_sl
        advisor._allocator.config.take_profit_half = _final_th
        advisor._allocator.config.take_profit_full = _final_tf
        advisor.take_profit_half = _final_th
        advisor.take_profit_full = _final_tf

        # 调用 advisor 生成调仓建议
        execution_candidates = []
        for candidate in candidates:
            row = dict(candidate)
            if prices.get(row.get("code"), 0) > 0:
                row["close"] = prices[row["code"]]
            execution_candidates.append(row)
        screening_result = {
            "candidates": execution_candidates,
            "regime": regime_info,  # 与补充检查用同一市场状态来源, 避免阈值按不同状态取值
            "direction": save.get("direction"),
        }

        advice = advisor.generate_advice_from_screening(
            screening_result=screening_result,
            holdings_data=holdings_data,
            account_data=account_data,
            timing_ratio=timing_ratio,
            panic=panic,
        )

        if "error" in advice:
            return advice

        orders = advice.get("orders", [])
        classification = advice.get("classification", {})

        if execution_block_reason:
            blocked_buys = [order for order in orders if order.get("side") in ("buy", "add")]
            orders = [order for order in orders if order.get("side") not in ("buy", "add")]
            for order in blocked_buys:
                filtered_out.append({
                    "code": order.get("code"), "name": order.get("name", ""),
                    "reason": execution_block_reason,
                })
            advice.setdefault("constraints", []).append({
                "name": "Champion生产准入", "value": "禁止新增买入", "limit": "仅当前Champion",
                "pass": False, "note": execution_block_reason,
            })
        elif not enforce_strategy_profile:
            advice.setdefault("constraints", []).append({
                "name": "实盘策略版本管控", "value": "已关闭（人工接管）", "limit": "默认仅当前Champion",
                "pass": True, "note": f"操作者={unmanaged_actor}；理由={unmanaged_reason}",
            })

        planned_amounts = {}
        lot_blocked = []
        lot_checked_orders = []
        for order in orders:
            code = order.get("code")
            if order.get("side") not in ("buy", "add"):
                lot_checked_orders.append(order)
                continue
            price = float(prices.get(code) or 0)
            shares, actual_amount = _round_buy_order(order.get("amount"), price)
            if shares < 100:
                lot_blocked.append(code)
                filtered_out.append({
                    "code": code,
                    "name": order.get("name", ""),
                    "reason": "目标金额不足买入100股",
                })
                continue
            order["current_price"] = round(price, 3)
            order["suggested_shares"] = shares
            order["amount"] = actual_amount
            planned_amounts[code] = order["amount"]
            lot_checked_orders.append(order)
        orders = lot_checked_orders
        amount_medians = _fetch_recent_amount(list(planned_amounts), decision_as_of)
        from core.domain.screening.exclusion import prefilter_by_liquidity_capacity
        _, capacity_dropped, capacity_checks = prefilter_by_liquidity_capacity(
            list(planned_amounts), amount_medians, planned_amounts=planned_amounts,
        )
        capacity_dropped_map = dict(capacity_dropped)
        capacity_blocked = list(lot_blocked)
        capacity_warnings = []
        capacity_orders = []
        for order in orders:
            check = capacity_checks.get(order.get("code")) if order.get("side") in ("buy", "add") else None
            if check:
                order["liquidity_capacity"] = check
                if check["status"] == "warning":
                    capacity_warnings.append(order.get("code"))
                if check["status"] == "excluded":
                    capacity_blocked.append(order.get("code"))
                    filtered_out.append({
                        "code": order.get("code"),
                        "name": order.get("name", ""),
                        "reason": capacity_dropped_map.get(order.get("code"), "下单容量超限"),
                    })
                    continue
            capacity_orders.append(order)
        orders = capacity_orders
        advice["constraints"].append({
            "name": "下单容量",
            "value": f"{len(capacity_warnings)}只预警 / {len(capacity_blocked)}只阻断",
            "limit": "预警>1.5%，阻断>2%",
            "pass": not capacity_blocked,
            "note": "按20日成交额中位数计算；有效数据不足15日时阻断买入",
        })
        if decision:
            advice["constraints"].append({
                "name": "决策快照",
                "value": decision["decision_id"],
                "limit": "预览、执行、审计一致",
                "pass": True,
                "note": decision.get("fingerprint", ""),
            })

        # 检查止盈止损（补充 advisor 未覆盖的持仓）— 复用上方统一的 regime 阈值
        # (_rsp/_final_sl/_final_th/_final_tf 已在 advisor 生成前解析, 与 advisor 同一来源)

        half_sold = _load_half_sold()
        for h in holdings_data:
            code = h["code"]
            already_stop = any(o["code"] == code and o.get("urgency") in ("stop_loss","take_profit") for o in orders)
            if already_stop:
                continue
            new_signal = _check_risk_signal(
                code, h["name"], h["shares"], h["cost_price"], h["current_price"],
                h["market_value"], _rsw, _final_sl, _final_th, _final_tf,
                already_half=(code in half_sold))
            if new_signal:
                orders.insert(0, new_signal)

        # 补充每条 order 的持仓信息 + 建议股数
        for o in orders:
            code = o["code"]
            h = next((x for x in holdings_data if x["code"] == code), None)
            price = o.get("current_price", 0) or (h["current_price"] if h else 0)
            amount = o.get("amount", 0)
            if h:
                o["current_price"] = round(h["current_price"], 3)
                o["cost_price"] = round(h["cost_price"], 3)
                o["current_shares"] = h["shares"]
                o["pnl_pct"] = round(h.get("pnl_pct", 0), 4)
                o["current_market_value"] = round(h["market_value"], 2)
            # 计算建议股数 (A股100股整数手, 向下取整)
            side = o.get("side", "")
            urgency = o.get("urgency", "")
            if side == "sell" and urgency == "stop_loss":
                o["suggested_shares"] = h["shares"] if h else 0
            elif side == "sell":
                o["suggested_shares"] = h["shares"] if h else 0
            elif side == "reduce":
                o["suggested_shares"] = min(h["shares"], max(100, int(amount / price / 100) * 100)) if h and price > 0 else 0
                o["amount"] = round(o["suggested_shares"] * price, 2)
            elif side in ("buy", "add"):
                o["suggested_shares"] = int(o.get("suggested_shares") or 0)
            else:
                o["suggested_shares"] = 0
            if "suggested_shares" not in o:
                o["suggested_shares"] = 0

        # 持久化信号
        batch_id = _make_batch_id()
        signals_for_db = []
        for o in orders:
            signals_for_db.append({
                "code": o["code"],
                "name": o.get("name", ""),
                "side": o["side"],
                "action": o.get("action", o["side"]),
                "current_weight": o.get("current_weight", 0),
                "target_weight": o.get("target_weight", 0),
                "delta_weight": o.get("delta_weight", 0),
                "amount": o.get("amount", 0),
                "urgency": o.get("urgency", "normal"),
                "reason": o.get("reason", ""),
                "planned_price": o.get("current_price"),
                "planned_shares": o.get("suggested_shares"),
            })

        from core.domain.portfolio.decision import decision_fingerprint, holdings_fingerprint_rows
        holdings_fingerprint = decision_fingerprint(holdings_fingerprint_rows(holdings_data))
        source_mode = "live" if enforce_strategy_profile else "live_unmanaged"
        session_payload = {
            "batch_id": batch_id, "check_type": "rebalance",
            "screening_save_id": screening_save_id, "eval_date": decision_as_of.isoformat(),
            "account_json": account_data, "holdings_json": holdings_data,
            "signals_json": [{"code": o["code"], "side": o["side"],
                              "action": o.get("action", ""), "urgency": o.get("urgency", ""),
                              "amount": o.get("amount", 0)} for o in orders],
            "constraints_json": advice.get("constraints", []), "regime_json": regime_info,
            "timing_ratio": timing_ratio, "turnover": advice.get("turnover", 0),
            "stop_loss_pct": _final_sl, "take_profit_half": _final_th,
            "take_profit_full": _final_tf,
            "profile_id": (profile_info or {}).get("profile_id"),
            "profile_version": (profile_info or {}).get("version"),
            "profile_fingerprint": (profile_info or {}).get("fingerprint"),
            "data_fingerprint": (profile_info or {}).get("data_fingerprint"),
            "model_fingerprint": (profile_info or {}).get("model_fingerprint"),
            "factor_fingerprint": (profile_info or {}).get("factor_fingerprint"),
            "source_mode": source_mode,
            "account_id": ACCOUNT_ID,
            "account_version": local_account["version"],
        }
        from core.domain.portfolio.decision import decision_fingerprint
        plan_fingerprint = decision_fingerprint({
            "screening_save_id": screening_save_id,
            "decision_id": decision.get("decision_id") if decision else None,
            "account_id": ACCOUNT_ID,
            "account_version": local_account["version"],
            "holdings_fingerprint": holdings_fingerprint,
            "parameters": {"top_n": top_n, "sizing": sizing, "frequency": frequency,
                           "code_prefixes": code_prefixes, "stop_loss_pct": stop_loss_pct,
                           "take_profit_half": take_profit_half, "take_profit_full": take_profit_full,
                           "cooldown_stop_weeks": cooldown_stop_weeks,
                           "cooldown_profit_weeks": cooldown_profit_weeks,
                           "cash": cash, "enforce_strategy_profile": enforce_strategy_profile,
                           "unmanaged_reason": unmanaged_reason, "unmanaged_actor": unmanaged_actor},
            "signals": signals_for_db,
            "session": {key: value for key, value in session_payload.items() if key != "batch_id"},
            "constraints": advice.get("constraints", []),
            "filtered_out": filtered_out,
        })
        if not preview_only and commit_request_id and expected_plan_fingerprint.lower() != plan_fingerprint:
            return {"error": "账户、行情或方案内容已变化，请重新预览后提交"}
        if not preview_only and commit_request_id and account_block_reason:
            return {"error": account_block_reason}
        if not preview_only and local_account.get("broker_account_id"):
            try:
                from backend.service.local_ledger import LocalLedgerService
                reconciliation = LocalLedgerService.reconcile()
            except Exception as error:
                logger.warning("[调仓] 提交前券商只读对账失败，阻止保存实盘计划: %s", error)
                return {"error": "券商对账不可用，未保存实盘计划", "commit_outcome": "not_committed"}
            if not isinstance(reconciliation, dict):
                return {"error": "券商对账结果无效，未保存实盘计划", "commit_outcome": "not_committed"}
            if (reconciliation.get("can_execute") is not True or
                    reconciliation.get("broker_account_id") != local_account.get("broker_account_id") or
                    reconciliation.get("local_version") != local_account.get("version")):
                reason = reconciliation.get("reason") or "券商对账状态或本地账户版本已变化"
                return {"error": f"{reason}，未保存实盘计划", "commit_outcome": "not_committed"}
        batch_id = uuid.UUID(commit_request_id).hex if not preview_only and commit_request_id else _make_batch_id()
        session_payload["batch_id"] = batch_id
        session_payload["commit_request_id"] = commit_request_id if not preview_only else None
        session_payload["plan_fingerprint"] = plan_fingerprint if not preview_only else None
        signal_ids = []
        if not preview_only and not account_block_reason:
            try:
                if not account_block_reason:
                    from core.data.repos.session import RebalanceSessionRepo
                    RebalanceSessionRepo.ensure_commit_schema()
                signal_ids = RebalanceSignalRepo.batch_create(
                    batch_id=batch_id, screening_save_id=screening_save_id,
                    check_type="rebalance", signals=signals_for_db,
                    profile_id=(profile_info or {}).get("profile_id"),
                    profile_version=(profile_info or {}).get("version"),
                    profile_fingerprint=(profile_info or {}).get("fingerprint"),
                    data_fingerprint=(profile_info or {}).get("data_fingerprint"),
                    model_fingerprint=(profile_info or {}).get("model_fingerprint"),
                    factor_fingerprint=(profile_info or {}).get("factor_fingerprint"),
                    source_mode=source_mode, input_fingerprint=holdings_fingerprint,
                    replace_pending=True, fingerprint_fn=decision_fingerprint,
                    session_payload=session_payload,
                    account_id=ACCOUNT_ID, account_version=local_account["version"],
                )
            except ValueError as e:
                return {"error": str(e), "commit_outcome": "not_committed"}
            except Exception as e:
                if commit_request_id:
                    try:
                        from core.data.repos.session import RebalanceSessionRepo
                        previous = RebalanceSessionRepo.get_by_commit_request(commit_request_id, ACCOUNT_ID)
                        saved_result = cls._saved_commit_result(previous, plan_fingerprint)
                        if saved_result:
                            return saved_result
                    except Exception as lookup_error:
                        logger.warning(f"[调仓] 提交竞争后回读失败 (request={commit_request_id}): {lookup_error}")
                logger.error(f"[调仓] 信号批次保存失败 (batch={batch_id}): {e}")
                return {"error": "调仓信号保存失败，请稍后重试", "commit_outcome": "unknown"}

        # 将 signal_id 挂回 order
        for i, o in enumerate(orders):
            if i < len(signal_ids):
                o["signal_id"] = signal_ids[i]

        today = decision_as_of
        logger.info(f"[调仓] 调仓检查完成: batch={batch_id}, {len(orders)} 条信号, "
                    f"止盈止损={sum(1 for o in orders if o.get('urgency') in ('stop_loss','take_profit'))}")

        return {
            "batch_id": batch_id if not preview_only else None,
            "check_date": today.isoformat(),
            "screening_save_id": screening_save_id,
            "save_name": save.get("save_name", ""),
            "frequency": frequency,
            "pool": pool,
            "top_n": top_n,
            "sizing": sizing,
            "timing_ratio": round(timing_ratio, 4),
            "position_source": position_source,
            "panic": panic,
            "regime": regime_info,
            "account": account_data,
            "holdings": holdings_data,
            "signals": orders,
            "plan_fingerprint": plan_fingerprint,
            "preview_only": bool(preview_only),
            "committed": not preview_only and not account_block_reason,
            "classification": classification,
            "observe_candidates": advice.get("observe_candidates", []),
            "top_n_codes": advice.get("top_n_codes", []),
            "skipped": advice.get("skipped", []),
            "filtered_out": filtered_out,
            "liquidity_capacity": {
                "amount_metric": "median_20d",
                "min_median_amount": 10000000,
                "warn_participation": 0.015,
                "max_participation": 0.02,
                "checks": capacity_checks,
            },
            "constraints": advice.get("constraints", []),
            "turnover": advice.get("turnover", 0),
            "stop_loss_pct": _final_sl,
            "take_profit_half": _final_th,
            "take_profit_full": _final_tf,
            "regime_stop_params": {"stop_loss_warn": _rsw, "stop_loss_full": _final_sl,
                                    "take_profit_half": _final_th, "take_profit_full": _final_tf,
                                    "source": _rsp["source"]},
            "price_source": price_source,
            "strategy_profile": profile_info or {"legacy": True, "promotable": False},
            "execution_blocked": bool(execution_block_reason or account_block_reason),
            "execution_block_reason": account_block_reason or execution_block_reason,
            "commit_blocked": bool(account_block_reason),
            "strategy_profile_control_enabled": bool(enforce_strategy_profile),
            "source_mode": source_mode,
            "manual_takeover": ({
                "actor": unmanaged_actor,
                "reason": unmanaged_reason,
            } if not enforce_strategy_profile else None),
            "decision": {
                "decision_id": decision.get("decision_id"),
                "policy": decision.get("policy"),
                "fingerprint": decision.get("fingerprint"),
                "data_cutoff": decision.get("data_cutoff"),
            } if decision else None,
        }

    # ─── 执行追踪 ───

    @classmethod
    def mark_executed(cls, signal_id, detail=None):
        from core.data import RebalanceSignalRepo
        from core.data.repos.local_ledger import ACCOUNT_ID

        signal = RebalanceSignalRepo.get_by_id(signal_id, account_id=ACCOUNT_ID)
        if not signal:
            return {"error": f"信号 #{signal_id} 不存在"}
        if signal.get("account_id") != ACCOUNT_ID:
            return {"error": "仅可确认本地实盘账户的信号"}
        try:
            planned_shares = Decimal(str(signal.get("planned_shares")))
        except (InvalidOperation, TypeError, ValueError):
            planned_shares = Decimal("0")
        if (not planned_shares.is_finite() or planned_shares <= 0 or
                planned_shares != planned_shares.to_integral_value()):
            return {"error": "旧信号缺少账户或明确计划股数，请重新生成计划"}
        fresh_error = None
        if signal.get("execution_status") not in ("pending", "partial"):
            fresh_error = f"信号 #{signal_id} 当前状态为 {signal.get('execution_status')}，不能记录成交"
        exec_detail = detail or {}
        import uuid
        try:
            request_id = str(uuid.UUID(str(exec_detail.get("request_id"))))
            raw_version = exec_detail.get("expected_account_version")
            if isinstance(raw_version, bool):
                raise ValueError
            version_value = Decimal(str(raw_version))
            if (not version_value.is_finite() or
                    version_value != version_value.to_integral_value() or version_value < 1):
                raise ValueError
            expected_version = int(version_value)
        except (TypeError, ValueError, AttributeError, InvalidOperation):
            return {"error": "成交确认必须提供 UUID 请求 ID 和账户版本"}
        try:
            commission = Decimal(str(exec_detail.get("commission")))
            stamp_duty = Decimal(str(exec_detail.get("stamp_duty")))
        except (InvalidOperation, TypeError, ValueError):
            return {"error": "实际费用必须为有效金额"}
        if any(not amount.is_finite() or amount < 0 or amount >= Decimal("10000000000") or
               amount != amount.quantize(Decimal("0.01"))
               for amount in (commission, stamp_duty)):
            return {"error": "实际费用必须为非负且最多两位小数"}
        if len(str(exec_detail.get("actor") or "")) == 0:
            return {"error": "成交确认缺少管理员身份"}
        if "broker_trade_id" in exec_detail:
            broker_trade_id = str(exec_detail["broker_trade_id"] or "").strip()
            if not broker_trade_id or len(broker_trade_id) > 64:
                return {"error": "券商成交编号无效"}
            exec_detail["broker_trade_id"] = broker_trade_id
        shares = exec_detail.get("shares", 0)
        price = exec_detail.get("price", 0)
        date = exec_detail.get("date")
        if date:
            try:
                trade_date = datetime.date.fromisoformat(str(date))
            except ValueError:
                return {"error": "成交日期必须为YYYY-MM-DD"}

        try:
            quantity = Decimal(str(shares))
            decimal_price = Decimal(str(price))
        except (InvalidOperation, TypeError, ValueError):
            return {"error": "成交股数和价格必须为有效数字"}
        if (not quantity.is_finite() or not decimal_price.is_finite() or
                quantity != quantity.to_integral_value()):
            return {"error": "成交股数和价格必须为有效数字"}
        if quantity <= 0 or decimal_price <= 0:
            return {"error": "成交股数和价格必须大于0"}
        if quantity > 2147483647:
            return {"error": "成交股数超过数据库范围"}
        if decimal_price >= Decimal("1000000000") or decimal_price != decimal_price.quantize(Decimal("0.001")):
            return {"error": "成交价格最多三位小数且须在数据库范围内"}
        shares = int(quantity)
        if commission + stamp_duty > decimal_price * shares:
            return {"error": "实际费用不能超过成交金额"}
        if shares % 100 != 0 and signal.get("side") in ("buy", "add"):
            return {"error": "A股买入股数必须为100股整数倍"}
        from core.data import HoldingsRepo
        if signal.get("side") in ("buy", "add"):
            try:
                from backend.service.strategy_profile import StrategyProfileService
                StrategyProfileService.ensure_execution_allowed(
                    signal.get("profile_id"), signal.get("profile_version"), signal.get("profile_fingerprint"),
                    enforce=signal.get("source_mode") != "live_unmanaged",
                )
            except ValueError as error:
                fresh_error = fresh_error or str(error)
            from core.domain.portfolio.decision import decision_fingerprint, holdings_fingerprint_rows
            current_holdings = HoldingsRepo.get_all(account_id=signal["account_id"])
            current_fingerprint = decision_fingerprint(holdings_fingerprint_rows(current_holdings))
            if signal.get("input_fingerprint") and current_fingerprint != signal["input_fingerprint"]:
                fresh_error = fresh_error or "持仓已变更，旧调仓方案已过期"
            planned_price = Decimal(str(signal.get("planned_price") or 0))
            if planned_price > 0 and abs(decimal_price / planned_price - 1) > Decimal("0.02"):
                fresh_error = fresh_error or "实际价格偏离计划价超过2%，请刷新行情并重新生成方案"
            planned_amount = Decimal(str(signal.get("amount") or 0))
            if planned_amount > 0 and shares * decimal_price > planned_amount * Decimal("1.05"):
                fresh_error = fresh_error or "实际成交金额超过计划金额5%"
        exec_detail = {**exec_detail, "request_id": request_id,
                       "expected_account_version": expected_version,
                       "commission": str(commission), "stamp_duty": str(stamp_duty),
                       "shares": shares, "price": str(decimal_price)}
        from core.domain.portfolio.decision import decision_fingerprint
        try:
            recorded = RebalanceSignalRepo.record_execution(signal_id, exec_detail, decision_fingerprint,
                                                              fresh_error=fresh_error)
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"[调仓] 信号#{signal_id} 成交记账失败: {e}")
            return {"error": f"信号#{signal_id} 成交记账失败，持仓与信号未更新"}
        if recorded["already_recorded"]:
            return {"signal_id": signal_id, **recorded}
        logger.info(f"[调仓] 信号#{signal_id} 已执行: {signal['side']} {signal['code']}")
        return {"signal_id": signal_id, **recorded}

    @classmethod
    def mark_skipped(cls, signal_id, reason=""):
        from core.data import RebalanceSignalRepo
        from core.data.repos.local_ledger import ACCOUNT_ID

        signal = RebalanceSignalRepo.get_by_id(signal_id, account_id=ACCOUNT_ID)
        if not signal:
            return {"error": f"信号 #{signal_id} 不存在"}
        if signal.get("account_id") != ACCOUNT_ID:
            return {"error": "仅可更改本地实盘账户的信号"}
        if signal.get("execution_status") == "skipped":
            return {"signal_id": signal_id, "execution_status": "skipped"}
        if signal.get("execution_status") != "pending":
            return {"error": f"信号 #{signal_id} 当前状态为 {signal.get('execution_status')}，不能跳过"}

        reason = str(reason or "").strip() or "用户确认本次未成交"
        try:
            if RebalanceSignalRepo.mark_skipped(signal_id, reason) != 1:
                return {"error": f"信号 #{signal_id} 状态已变化，未执行跳过"}
        except ValueError as error:
            return {"error": str(error)}
        logger.info(f"[调仓] 信号#{signal_id} 已跳过: {reason}")
        return {"signal_id": signal_id, "execution_status": "skipped"}

    @classmethod
    def batch_execute(cls, signal_details):
        """批量执行 — signal_details: [{signal_id, shares?, price?, date?}] (C2修复: per-signal详情)"""
        if not signal_details:
            return {"executed": 0}

        executed = 0
        failures = []
        for item in signal_details:
            sid = item.get("signal_id") if isinstance(item, dict) else item
            if not sid:
                continue
            detail = item if isinstance(item, dict) else {}
            result = cls.mark_executed(sid, {
                "shares": detail.get("shares", 0),
                "price": detail.get("price", 0),
                "date": detail.get("date"),
                "request_id": detail.get("request_id"),
                "expected_account_version": detail.get("expected_account_version"),
                "commission": detail.get("commission"),
                "stamp_duty": detail.get("stamp_duty"),
                "actor": detail.get("actor"),
                **({"broker_trade_id": detail["broker_trade_id"]}
                   if "broker_trade_id" in detail else {}),
            })
            if not result.get("error"):
                executed += 1
            else:
                failures.append({"signal_id": sid, "error": result["error"]})

        logger.info(f"[调仓] 批量执行 {executed}/{len(signal_details)} 条信号")
        return {"executed": executed, "failures": failures}

    @classmethod
    def get_pending_signals(cls, check_type=None, limit=200, offset=0):
        from core.data import RebalanceSignalRepo
        return RebalanceSignalRepo.get_pending(check_type=check_type, limit=limit, offset=offset)

    @classmethod
    def get_history(cls, days=30, limit=50):
        from core.data import RebalanceSignalRepo
        from core.data.repos.local_ledger import ACCOUNT_ID
        return RebalanceSignalRepo.get_history(days=days, limit=limit, account_id=ACCOUNT_ID)

    @classmethod
    def get_batch_signals(cls, batch_id, execution_status=None):
        from core.data import RebalanceSignalRepo
        from core.data.repos.local_ledger import ACCOUNT_ID
        return RebalanceSignalRepo.get_by_batch(
            batch_id, execution_status=execution_status, account_id=ACCOUNT_ID)

    @classmethod
    def analyze_intraday_entry(cls, codes, reference_date=None, ma_data=None, sell_signals=None):
        """
        日内执行建议 — 买入目标做回调检测, 卖出信号直接给执行决策。

        Args:
            codes: 买入候选股票代码列表
            reference_date: 参考日期 "YYYY-MM-DD", None 则为今天
            ma_data: {code: {ma5, ma10, ma20}, ...} (买入侧支撑位用)
            sell_signals: [{code, side, urgency, name, pnl_pct, suggested_shares}, ...]
                          (卖出侧调仓信号, 只取日内状态算执行决策, 不做回调检测)

        Returns:
            {results: [IntradayResult dict], sell_results: [dict], source: str,
             timestamp: str, unavailable: [str]}
        """
        from datetime import datetime as dt
        from core.domain.portfolio.intraday import IntradayPullbackDetector
        from core.shared.trading_time import A_SHARE_TIMEZONE, is_trading_time, market_date

        now = dt.now(A_SHARE_TIMEZONE)
        if reference_date is None:
            reference_date = market_date(now).isoformat()
        reference_date = str(reference_date)[:10]
        timestamp = now.strftime("%H:%M:%S")
        try:
            market_open = is_trading_time(now) and reference_date == market_date(now).isoformat()
        except Exception as e:
            logger.warning("[Intraday] 交易时段核验不可用: %s", e)
            market_open = False
        from core.domain.portfolio.execution import resolve_phase
        phase = resolve_phase(now.strftime("%H:%M")) if market_open else "closed"
        if phase == "closed":
            market_open = False
        detector = IntradayPullbackDetector()
        results = []
        sell_results = []
        unavailable = []
        source = "fallback"

        sell_signals = sell_signals or []
        sell_codes = [s.get("code") for s in sell_signals if s.get("code")]
        all_codes = list(dict.fromkeys(list(codes) + sell_codes))

        per_code_results = _fetch_intraday_data(all_codes, reference_date, source)
        from backend.service.market_data import MarketDataService
        try:
            quote_response = MarketDataService.get_live_quotes(all_codes, use_case="execution")
            quote_meta = quote_response.get("meta") or {}
            quotes = (quote_response.get("data") or {}) if quote_meta.get("freshness") == "fresh" else {}
        except Exception as e:
            logger.warning("[Intraday] 执行行情读取失败: %s", e)
            quote_meta = {"freshness": "unavailable"}
            quotes = {}
        realtime_prices = {}
        for code, quote in quotes.items():
            try:
                price = float(quote.get("price") or quote.get("last_price") or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(price) and price > 0:
                realtime_prices[code] = price
        realtime_source = "realtime" if quote_meta.get("freshness") == "fresh" else "unavailable"
        if realtime_source == "realtime" and realtime_prices:
            source = "realtime"

        sellable_by_code = {}
        sellable_error = None
        if sell_signals:
            try:
                from backend.service.local_ledger import LocalLedgerService
                account_snapshot = LocalLedgerService.get_t0_account_snapshot()
                sellable_by_code = {
                    str(item.get("code")): item.get("sellable_shares")
                    for item in account_snapshot.get("holdings", [])
                }
            except Exception as e:
                sellable_error = str(e) or "本地可卖持仓不可用"

        for code in codes:
            fetch_result = per_code_results.get(code)
            df_5min = None
            prev_day = {}

            if fetch_result and fetch_result.get("df") is not None:
                df_5min = fetch_result["df"]
                source = fetch_result.get("source", source)
            if fetch_result and fetch_result.get("prev_day"):
                prev_day = fetch_result["prev_day"]

            if df_5min is None and not prev_day:
                unavailable.append(code)
                continue

            mas = (ma_data or {}).get(code, {})
            result = detector.analyze(
                df_5min=df_5min,
                prev_day_low=prev_day.get("low", 0),
                prev_day_high=prev_day.get("high", 0),
                prev_day_close=prev_day.get("close", 0),
                reference_date=reference_date,
                ma5=mas.get("ma5"),
                ma10=mas.get("ma10"),
                ma20=mas.get("ma20"),
            )
            result.code = code
            if realtime_source == "realtime" and realtime_prices.get(code, 0) > 0:
                result.current_price = realtime_prices[code]
            d = _intraday_result_to_dict(result)
            quantity_verified = bool(fetch_result and fetch_result.get("quantity_status") == "verified")
            semantics_verified = bool(fetch_result and fetch_result.get("semantic_status") == "verified")
            analysis_verified = bool(df_5min is not None and quantity_verified and semantics_verified)
            d["analysis_status"] = "verified" if analysis_verified else "unverified"
            d["data_provenance"] = {
                "source": fetch_result.get("source") if fetch_result else None,
                "quantity_status": fetch_result.get("quantity_status", "unknown") if fetch_result else "unknown",
                "semantic_status": fetch_result.get("semantic_status", "unverified") if fetch_result else "unverified",
                "data_envelope": fetch_result.get("data_envelope") if fetch_result else None,
            }
            if not analysis_verified:
                d.update({
                    "entry_quality_score": None,
                    "entry_quality_label": "数据未认证",
                    "open_price": None,
                    "day_high": None,
                    "day_low": None,
                    "vwap": None,
                    "supports": [],
                    "nearest_support": None,
                    "signals": [],
                    "recent_bars": [],
                    "volume_ma": None,
                    "risk_warning": "分钟来源语义未核验，量额派生评分和技术位不可用",
                })
                if not (realtime_source == "realtime" and realtime_prices.get(code, 0) > 0):
                    d["current_price"] = None
                _invalidate_unverified_intraday_analysis(
                    result,
                    realtime_prices.get(code) if realtime_source == "realtime" else None,
                )
            blockers = _intraday_execution_blockers(
                code, df_5min, quotes.get(code), reference_date, now, market_open, "buy",
                bars_verified=quantity_verified,
                semantics_verified=semantics_verified,
            )
            d["execution"] = _execution_advice(
                result, prev_day, phase=phase, blockers=blockers,
                quote=quotes.get(code), security_name=quotes.get(code, {}).get("name"),
            )
            results.append(d)

        for sig in sell_signals:
            code = sig.get("code")
            if not code:
                continue
            fetch_result = per_code_results.get(code)
            df_5min = None
            prev_day = {}
            if fetch_result and fetch_result.get("df") is not None:
                df_5min = fetch_result["df"]
            if fetch_result and fetch_result.get("prev_day"):
                prev_day = fetch_result["prev_day"]
            if df_5min is None and not prev_day:
                continue

            side = sig.get("side") or "sell"
            urgency = sig.get("urgency") or "normal"
            try:
                result = detector.analyze(
                    df_5min=df_5min,
                    prev_day_low=prev_day.get("low", 0),
                    prev_day_high=prev_day.get("high", 0),
                    prev_day_close=prev_day.get("close", 0),
                    reference_date=reference_date,
                )
                result.code = code
                if realtime_source == "realtime" and realtime_prices.get(code, 0) > 0:
                    result.current_price = realtime_prices[code]
                elif not (fetch_result and fetch_result.get("quantity_status") == "verified"
                          and fetch_result.get("semantic_status") == "verified"):
                    result.current_price = None
                analysis_verified = bool(
                    df_5min is not None
                    and fetch_result
                    and fetch_result.get("quantity_status") == "verified"
                    and fetch_result.get("semantic_status") == "verified"
                )
                if not analysis_verified:
                    _invalidate_unverified_intraday_analysis(
                        result,
                        realtime_prices.get(code) if realtime_source == "realtime" else None,
                    )
            except Exception as e:
                logger.warning(f"[Rebalance] 卖出侧 {code} 日内分析失败: {e}")
                continue

            sell_results.append({
                "code": code,
                "name": sig.get("name", ""),
                "side": side,
                "urgency": urgency,
                "pnl_pct": float(sig["pnl_pct"]) if sig.get("pnl_pct") is not None else None,
                "suggested_shares": int(sig["suggested_shares"]) if sig.get("suggested_shares") is not None else None,
                "current_price": float(result.current_price) if result.current_price is not None else None,
                "analysis_status": "verified" if analysis_verified else "unverified",
                "data_provenance": {
                    "source": fetch_result.get("source") if fetch_result else None,
                    "quantity_status": fetch_result.get("quantity_status", "unknown") if fetch_result else "unknown",
                    "semantic_status": fetch_result.get("semantic_status", "unverified") if fetch_result else "unverified",
                    "data_envelope": fetch_result.get("data_envelope") if fetch_result else None,
                },
                "execution": _execution_advice(
                    result, prev_day, side=side, urgency=urgency, phase=phase,
                    quote=quotes.get(code), security_name=quotes.get(code, {}).get("name") or sig.get("name"),
                    blockers=_intraday_execution_blockers(
                        code, df_5min, quotes.get(code), reference_date, now, market_open, side,
                        requested_shares=sig.get("suggested_shares"),
                        sellable_shares=sellable_by_code.get(code),
                        sellable_error=sellable_error,
                        security_name=sig.get("name"),
                        bars_verified=bool(fetch_result and fetch_result.get("quantity_status") == "verified"),
                        semantics_verified=bool(fetch_result and fetch_result.get("semantic_status") == "verified"),
                    ),
                ),
            })

        return {
            "results": results,
            "sell_results": sell_results,
            "source": source,
            "timestamp": timestamp,
            "market_open": market_open,
            "market_phase": phase,
            "unavailable": unavailable,
            "message": f"买入 {len(codes)} 只, 可用 {len(results)}; 卖出 {len(sell_codes)} 只, 可用 {len(sell_results)}"
        }

    @classmethod
    def analyze_t0(cls, codes, reference_date=None, holdings=None, available_cash=None):
        """T+0 做T建议 — I(激进)/N(保守) 双档，供调仓页展示。
        实时路径从本地权威账户读取持仓、可用现金和可卖股数。"""
        from datetime import date, timedelta, datetime as dt
        from core.domain.portfolio.t0 import T0Engine
        from core.data.repos.intraday import IntradayDataRepo
        from core.shared.data_contract import make_data_envelope
        import pandas as pd

        is_live = reference_date is None
        if is_live:
            from backend.service.local_ledger import LocalLedgerService

            account_snapshot = LocalLedgerService.get_t0_account_snapshot()
            account = account_snapshot["account"]
            holdings = account_snapshot["holdings"]
            available_cash = account["available_cash"]
            from core.shared.trading_time import market_date
            reference_date = market_date().isoformat()
        ref = str(reference_date)[:10]
        start = (date.fromisoformat(ref) - timedelta(days=400)).isoformat()
        from core.shared.trading_time import A_SHARE_TIMEZONE

        hold_map = {}
        if isinstance(holdings, dict):
            hold_map = holdings
        elif isinstance(holdings, list):
            for h in holdings:
                if isinstance(h, dict) and h.get("code"):
                    hold_map[h["code"]] = h

        engine = T0Engine()
        results = []
        semantic_statuses = []
        semantic_envelopes = []
        unavailable = []
        unavailable_details = {}
        codes = list(dict.fromkeys(str(code).strip() for code in codes))
        requested_count = len(codes)
        if is_live:
            for code in codes:
                if code not in hold_map:
                    unavailable.append(code)
                    unavailable_details[code] = {
                        "reason_code": "local_holding_missing",
                        "reason": "本地权威账本中没有该证券持仓",
                    }
            codes = [code for code in codes if code in hold_map]
        cash_per_code = None
        if available_cash is not None and len(codes) > 0:
            cash_per_code = max(float(available_cash), 0.0) / len(codes)
        for code in codes:
            try:
                minute_envelope = None
                if is_live:
                    df = _try_ths_intraday_live(code, ref)
                    quantity_status = "verified" if df is not None and not df.empty else "unknown"
                    minute_envelope = getattr(df, "attrs", {}).get("data_envelope") if df is not None else None
                else:
                    history = IntradayDataRepo.get_history_bars(code, ref, ref)
                    df = pd.DataFrame(history) if history else None
                    if df is not None and not df.empty and "bar_time" in df.columns:
                        df = df.set_index("bar_time")
                    quantity_status = (
                        "verified" if history and all(row.get("quantity_status") == "verified" for row in history)
                        else "unknown"
                    )
                    envelopes = [row.get("data_envelope") for row in history or []]
                    if envelopes and all(envelope == envelopes[0] for envelope in envelopes):
                        minute_envelope = envelopes[0]
                from core.shared.minute_data import has_verified_minute_semantics
                semantic_status = "verified" if has_verified_minute_semantics(minute_envelope) else "unverified"
                semantic_envelopes.append(minute_envelope)
                daily = IntradayDataRepo.get_daily_kline(code, start, ref)
                if df is None or getattr(df, "empty", True):
                    unavailable.append(code)
                    unavailable_details[code] = {
                        "reason_code": "minute_data_unavailable",
                        "reason": "没有可用的分钟行情；实时数据可能缺失、过期、非交易时段或量额未核验",
                    }
                    continue
                if quantity_status != "verified":
                    unavailable.append(code)
                    unavailable_details[code] = {
                        "reason_code": "minute_quantity_unverified",
                        "reason": "分钟成交量额状态未核验，不能生成做T建议",
                    }
                    continue
                hold = hold_map.get(code, {})
                total_shares = hold.get("total_shares")
                if total_shares is None:
                    total_shares = hold.get("shares")
                sellable_shares = hold.get("sellable_shares")
                res = engine.compute(
                    code=code,
                    df_5min=df,
                    daily_bars=daily,
                    current_price=None,
                    total_shares=total_shares,
                    available_cash=cash_per_code,
                    sellable_shares=sellable_shares,
                    quantity_status=quantity_status,
                    semantic_status=semantic_status,
                )
                semantic_statuses.append(semantic_status)
                latest_bar = df.index[-1] if len(df.index) else None
                if hasattr(latest_bar, "to_pydatetime"):
                    latest_bar = latest_bar.to_pydatetime()
                elif isinstance(latest_bar, str):
                    try:
                        latest_bar = dt.fromisoformat(latest_bar)
                    except ValueError:
                        latest_bar = None
                if isinstance(latest_bar, dt):
                    if latest_bar.tzinfo is None:
                        latest_bar = latest_bar.replace(tzinfo=A_SHARE_TIMEZONE)
                    else:
                        latest_bar = latest_bar.astimezone(A_SHARE_TIMEZONE)
                    observed_at = latest_bar.isoformat(timespec="seconds")
                else:
                    observed_at = None
                if isinstance(res, dict):
                    semantic_missing_fields = [
                        "price_basis",
                        "volume_unit",
                        "amount_unit",
                        "source_quantity_semantics",
                    ] if semantic_status != "verified" else []
                    provenance_envelope = dict(minute_envelope or {})
                    if not provenance_envelope:
                        provenance_envelope = make_data_envelope(
                            source="ths_live_minute" if is_live else "ths_minute_history",
                            observed_at=observed_at,
                            as_of=ref,
                            unit={"price": "CNY/share", "volume": "unknown", "amount": "unknown"},
                            price_basis="unknown",
                            coverage={"requested": 1, "accepted": 1, "missing": 0},
                            status="partial",
                            missing_reason=semantic_missing_fields,
                        )
                    provenance_envelope["coverage"] = {"requested": 1, "accepted": 1, "missing": 0}
                    provenance_envelope["quantity_status"] = quantity_status
                    provenance_envelope["semantic_status"] = semantic_status
                    provenance_envelope["local_gate_status"] = "passed"
                    res["data_provenance"] = {
                        **provenance_envelope,
                    }
                    if semantic_status != "verified":
                        for variant_name in ("aggressive", "conservative"):
                            variant = res.get(variant_name)
                            if isinstance(variant, dict):
                                variant["allowed"] = False
                                variant["suggested_shares"] = 0
                                variant["suggested_sell_shares"] = 0
                                variant.setdefault("flags", []).append("分钟数据源语义未核验，不可执行")
                results.append(res)
            except Exception as e:
                logger.warning(f"[T0] {code} 计算失败: {e}")
                unavailable.append(code)
                unavailable_details[code] = {
                    "reason_code": "calculation_failed",
                    "reason": "读取或计算异常，未生成建议",
                }
        if is_live:
            latest_account = LocalLedgerService.get_account()
            expected_account = account_snapshot["account"]
            if (latest_account.get("status") != "active" or
                    latest_account.get("version") != expected_account.get("version") or
                    latest_account.get("broker_account_id") != expected_account.get("broker_account_id")):
                raise RuntimeError("本地账户在T+0计算期间已变化，请刷新后重新生成建议")
            if expected_account.get("broker_account_id"):
                latest_reconciliation = LocalLedgerService.reconcile()
                if (latest_reconciliation.get("can_execute") is not True or
                        latest_reconciliation.get("local_version") != expected_account.get("version") or
                        latest_reconciliation.get("broker_account_id") != expected_account.get("broker_account_id")):
                    raise RuntimeError(latest_reconciliation.get("reason") or "T+0计算期间券商对账状态变化，请刷新后重新生成建议")
                latest_sellable = latest_reconciliation.get("broker_sellable") or {}
                if any(latest_sellable.get(code) != holding.get("sellable_shares")
                       for code, holding in hold_map.items()):
                    raise RuntimeError("券商可卖数量在T+0计算期间已变化，请刷新后重新生成建议")
        if not results:
            data_status = "unavailable"
        elif (unavailable or not semantic_statuses
              or not all(status == "verified" for status in semantic_statuses)
              or not semantic_envelopes or any(envelope is None for envelope in semantic_envelopes)
              or any(envelope != semantic_envelopes[0] for envelope in semantic_envelopes[1:])):
            data_status = "partial"
        else:
            data_status = "verified"
        response_time = dt.now(A_SHARE_TIMEZONE)
        available_at = response_time.isoformat(timespec="seconds")
        for result in results:
            provenance = result.get("data_provenance") if isinstance(result, dict) else None
            if provenance is not None:
                provenance["available_at"] = available_at
        base_envelope = semantic_envelopes[0] if data_status == "verified" else None
        data_envelope = make_data_envelope(
            source=(base_envelope or {}).get("source") or ("ths_live_minute" if is_live else "ths_minute_history"),
            observed_at=max(
                (result["data_provenance"]["observed_at"] for result in results
                 if isinstance(result, dict) and result.get("data_provenance", {}).get("observed_at")),
                default=None,
            ),
            available_at=available_at,
            as_of=ref,
            version=(base_envelope or {}).get("version"),
            unit=(base_envelope or {}).get("unit") or {"price": "CNY/share", "volume": "unknown", "amount": "unknown"},
            price_basis=(base_envelope or {}).get("price_basis") or "unknown",
            coverage={"requested": requested_count, "accepted": len(results), "missing": len(set(unavailable))},
            status=data_status,
            missing_reason=(["price_basis", "volume_unit", "amount_unit", "source_quantity_semantics"]
                            if results and data_status != "verified" else unavailable_details if not results else []),
        )
        if base_envelope:
            data_envelope["source_quantity_semantics"] = base_envelope.get("source_quantity_semantics")
        return {
            "results": results,
            "unavailable": list(dict.fromkeys(unavailable)),
            "unavailable_details": unavailable_details,
            "source": "ths",
            "mode": "live" if is_live else "historical_preview",
            "data_source": "ths_live_minute" if is_live else "ths_minute_history",
            "data_status": data_status,
            "data_envelope": data_envelope,
            "semantic_status": "verified" if semantic_statuses and all(
                status == "verified" for status in semantic_statuses
            ) else "unverified",
            "semantic_missing_fields": [
                "price_basis",
                "volume_unit",
                "amount_unit",
                "source_quantity_semantics",
            ],
            "coverage": {
                "requested": requested_count,
                "available": len(results),
                "missing": len(set(unavailable)),
            },
            "as_of": ref,
            "available_at": available_at,
            "sellable_source": account_snapshot.get("sellable_source") if is_live else "historical_input",
            "account_id": account_snapshot["account"]["account_id"] if is_live else None,
            "account_version": account_snapshot["account"]["version"] if is_live else None,
            "timestamp": response_time.strftime("%H:%M:%S"),
            "message": f"T+0建议 {len(results)} 只, 不可用 {len(unavailable)} 只",
        }

    @classmethod
    def preview_stocks(cls, codes, reference_date=None, stop_mode="fixed"):
        if not codes:
            return {"results": {}, "regime_stops": _get_regime_stop_dict()}

        import os
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        regime_stops = _get_regime_stop_dict()
        results = {}

        with ThreadPoolExecutor(max_workers=min(8, len(codes))) as pool:
            futures = {pool.submit(_preview_single_stock, c, reference_date, regime_stops, stop_mode): c for c in codes}
            for f in as_completed(futures):
                try:
                    r = f.result()
                    results[r["code"]] = r
                except Exception:
                    pass

        return {"results": results, "regime_stops": regime_stops}


def _fetch_intraday_data(codes, reference_date, source_flag):
    """
    L2 辅助: 为每只股票获取 5min DataFrame + 前日日K。
    只走 THS 实时爬虫, 不用 miniQMT / 其它 API / 历史缓存。拿不到退化为仅日K。
    """
    from datetime import date, timedelta
    from core.shared.minute_data import has_verified_minute_semantics

    result = {}

    for code in codes:
        df = _try_ths_intraday_live(code, reference_date)
        fetch_source = "ths" if df is not None else "none"

        prev_day = _get_prev_day_kline(code, reference_date)
        envelope = getattr(df, "attrs", {}).get("data_envelope") if df is not None else None
        result[code] = {
            "df": df, "prev_day": prev_day, "source": fetch_source,
            "quantity_status": "verified" if df is not None else "unknown",
            "data_envelope": envelope,
            "semantic_status": "verified" if has_verified_minute_semantics(envelope) else "unverified",
        }

    return result


def _try_ths_intraday_live(code, trade_date, now=None):
    """实时爬取同花顺分时 (1分钟) → 重采样 5min OHLC DataFrame。"""
    import pandas as pd
    try:
        from core.shared.trading_time import A_SHARE_TIMEZONE, is_trading_time, market_date
        now = now or datetime.datetime.now(A_SHARE_TIMEZONE)
        if now.tzinfo is None:
            now = now.replace(tzinfo=A_SHARE_TIMEZONE)
        else:
            now = now.astimezone(A_SHARE_TIMEZONE)
        requested_date = str(trade_date)[:10]
        if requested_date != market_date(now).isoformat() or not is_trading_time(now):
            return None

        from backend.service.market_data import MarketDataService
        from core.data.ths_collector import _resample_to_ohlc
        rows, _ = MarketDataService.get_live_intraday(code)
        if not rows:
            return None
        current_rows = []
        now_naive = now.replace(tzinfo=None)
        for row in rows:
            try:
                row_time = datetime.datetime.fromisoformat(str(row["bar_time"]))
                if row_time.tzinfo is not None:
                    row_time = row_time.astimezone(A_SHARE_TIMEZONE).replace(tzinfo=None)
                if row_time.date().isoformat() != requested_date:
                    continue
                age_seconds = (now_naive - row_time).total_seconds()
                if age_seconds >= -5:
                    current_rows.append((row, row_time))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if not current_rows:
            return None
        latest_time = max(row_time for _, row_time in current_rows)
        if (now_naive - latest_time).total_seconds() > 600:
            return None
        from core.shared.minute_data import normalize_trade_quantity
        rows_for_resample = [row for row, _ in current_rows]
        quantity_statuses = [row.get("quantity_status") for row in rows_for_resample]
        if all(status == "unknown" for status in quantity_statuses):
            try:
                from core.data.ths_collector import THSCollector
                THSCollector()._save_intraday(code, rows_for_resample)
            except Exception:
                pass
            return None
        if any(status != "verified" for status in quantity_statuses):
            return None
        for row, _ in current_rows:
            if normalize_trade_quantity(row.get("volume"), row.get("amount")) is None:
                return None
        ohlc_rows = []
        for row in rows_for_resample:
            ohlc = _resample_to_ohlc(row, rows_for_resample)
            if ohlc:
                if ohlc.get("quantity_status") != "verified":
                    return None
                ohlc_rows.append({
                    "time": row["bar_time"],
                    "open": ohlc["open"],
                    "high": ohlc["high"],
                    "low": ohlc["low"],
                    "close": ohlc["close"],
                    "volume": ohlc["volume"],
                    "amount": ohlc["amount"],
                })
        if not ohlc_rows:
            return None
        try:
            from core.data.ths_collector import THSCollector
            THSCollector()._save_intraday(code, rows_for_resample)
        except Exception:
            pass
        df = pd.DataFrame(ohlc_rows)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        from core.shared.data_contract import make_data_envelope
        df.attrs["data_envelope"] = make_data_envelope(
            source="ths_live_minute", observed_at=str(df.index[-1]),
            available_at=now.isoformat(timespec="seconds"), as_of=requested_date,
            version="unverified", unit={"price": "CNY/share", "volume": "unknown", "amount": "unknown"},
            price_basis="unknown", status="partial",
            missing_reason=["price_basis", "volume_unit", "amount_unit", "source_quantity_semantics"],
        ) | {"source_quantity_semantics": "unknown"}
        return df
    except Exception:
        return None


def _get_intraday_vwap(code, reference_date):
    """取当日分钟数据算 VWAP (THS 实时爬虫), 非当日/拿不到返回 None"""
    from datetime import datetime
    from core.domain.portfolio.intraday import IntradayPullbackDetector
    from core.shared.minute_data import has_verified_minute_semantics

    from core.shared.persistence import today_iso
    ref = str(reference_date)[:10] if reference_date else today_iso()
    df = _try_ths_intraday_live(code, ref)
    if df is None or df.empty:
        return None
    envelope = getattr(df, "attrs", {}).get("data_envelope")
    if not has_verified_minute_semantics(envelope):
        return None
    try:
        # 最后 bar 必须属于参考日(今天), 否则是历史数据, 不算盘中 VWAP
        last_day = str(df.index[-1])[:10]
        if last_day != ref:
            return None
        age_seconds = (datetime.now() - df.index[-1].to_pydatetime()).total_seconds()
        if age_seconds < -5 or age_seconds > 600:
            return None
        vwap = IntradayPullbackDetector()._calc_vwap(df)
        return float(vwap) if vwap and vwap > 0 else None
    except Exception:
        return None


def _get_prev_day_kline(code, reference_date):
    """获取参考日前最近一个已入库交易日 (不含参考日, 避免前视) 的日K高低收"""
    try:
        from core.data.connection import Database
        ref = str(reference_date)[:10]
        row = Database.fetchone(
            "SELECT trade_date, open, high, low, close FROM daily_quotes "
            "WHERE code=%s AND trade_date<%s ORDER BY trade_date DESC LIMIT 1",
            (code, ref))
        if row:
            return {"low": float(row.get("low", 0)), "high": float(row.get("high", 0)),
                    "close": float(row.get("close", 0))}
    except Exception:
        pass
    return {}


def _intraday_result_to_dict(result):
    def _f(v):
        return float(v) if v is not None else None

    return {
        "code": result.code,
        "current_price": _f(result.current_price),
        "open_price": _f(result.open_price),
        "day_high": _f(result.day_high),
        "day_low": _f(result.day_low),
        "vwap": _f(result.vwap),
        "supports": [{"name": s.name, "price": _f(s.price), "distance_pct": _f(s.distance_pct)}
                     for s in result.supports],
        "nearest_support": {"name": result.nearest_support.name,
                            "price": _f(result.nearest_support.price),
                            "distance_pct": _f(result.nearest_support.distance_pct)}
        if result.nearest_support else None,
        "signals": [{"type": s.type, "description": s.description, "strength": int(s.strength)}
                    for s in result.signals],
        "entry_quality_score": _f(result.entry_quality_score),
        "entry_quality_label": result.entry_quality_label,
        "risk_warning": result.risk_warning,
        "recent_bars": result.recent_bars,
        "volume_ma": _f(result.volume_ma),
    }


def _intraday_price_limit_bounds(code, quote, security_name=None):
    from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
    from core.shared.market_rules import limit_pct_for_code

    blockers = []
    if not quote:
        return None, None, ["缺少涨跌停核验报价"]
    try:
        previous_close = Decimal(str(quote.get("prev_close")))
        if not previous_close.is_finite() or previous_close <= 0:
            raise InvalidOperation
    except (InvalidOperation, TypeError, ValueError):
        return None, None, ["报价缺少有效前收盘价，不能核验涨跌停"]

    name = str(quote.get("name") or security_name or "").strip()
    if not name:
        return None, None, ["证券名称缺失，不能确认ST涨跌幅规则"]
    if len(str(code)) != 6 or not str(code).isdigit():
        return None, None, ["证券代码无效，不能核验涨跌停规则"]
    if name[:1].upper() in ("N", "C"):
        return None, None, ["新股上市初期可能不适用常规涨跌停，需专项核验"]

    is_st = "ST" in name.upper()
    limit_pct = Decimal(str(limit_pct_for_code(code, is_st=is_st)))
    limit_up = (previous_close * (Decimal("1") + limit_pct)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP,
    )
    limit_down = (previous_close * (Decimal("1") - limit_pct)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP,
    )
    return limit_up, limit_down, blockers


def _intraday_execution_blockers(code, df, quote, reference_date, now, market_open, side,
                                 requested_shares=None, sellable_shares=None,
                                 sellable_error=None, security_name=None, bars_verified=False,
                                 semantics_verified=False):
    from decimal import Decimal, InvalidOperation
    import pandas as pd
    from core.domain.market_data.freshness import assess_quote
    from core.shared import config

    blockers = []
    if side not in ("buy", "add", "sell", "reduce"):
        blockers.append("交易方向无效")
    is_sell = side in ("sell", "reduce")
    if not market_open:
        blockers.append("当前不是有效交易时段")
    if not quote:
        blockers.append("缺少新鲜实时执行报价")
    else:
        quote_status = assess_quote(
            code,
            quote,
            now.replace(tzinfo=None),
            config.LIVE_QUOTE_MAX_AGE,
            expected_session_date=now.date().isoformat(),
        )
        if not quote_status["usable"]:
            blockers.append("实时报价时间缺失、过期或不属于当前交易日")
        if quote.get("session_date") != now.date().isoformat():
            blockers.append("实时报价不属于当前交易日")
        try:
            current_price = Decimal(str(quote.get("price") or quote.get("last_price")))
            if not current_price.is_finite() or current_price <= 0:
                raise InvalidOperation
        except (InvalidOperation, TypeError, ValueError):
            current_price = None
            blockers.append("报价缺少有效现价，不能核验涨跌停")

        limit_up, limit_down, limit_blockers = _intraday_price_limit_bounds(
            code, quote, security_name,
        )
        blockers.extend(limit_blockers)
        if current_price is not None and limit_up is not None and limit_down is not None:
            if is_sell and current_price <= limit_down:
                blockers.append("现价触及或低于跌停价，卖出不可执行")
            if not is_sell and current_price >= limit_up:
                blockers.append("现价触及或高于涨停价，买入不可执行")

    if df is None or getattr(df, "empty", True):
        blockers.append("缺少当日已验证分钟线")
    elif not bars_verified:
        blockers.append("分钟线未通过verified状态校验")
    else:
        try:
            last_at = pd.Timestamp(df.index[-1])
            if last_at.tzinfo is not None:
                last_at = last_at.tz_convert(now.tzinfo).tz_localize(None)
            now_naive = now.replace(tzinfo=None)
            age_seconds = (now_naive - last_at.to_pydatetime()).total_seconds()
            if last_at.strftime("%Y-%m-%d") != reference_date:
                blockers.append("分钟线不属于请求的当前交易日")
            if age_seconds < -5 or age_seconds > 600:
                blockers.append("最新分钟线超过10分钟或时间异常")
        except Exception:
            blockers.append("分钟线时间戳无效")

    if not semantics_verified:
        blockers.append("分钟数据单位、累计口径或价格基准未完成语义核验")

    if is_sell:
        if sellable_error:
            blockers.append("本地可卖持仓不可用")
        elif isinstance(sellable_shares, bool) or not isinstance(sellable_shares, int) or sellable_shares <= 0:
            blockers.append("可卖股数缺失或为零")
        if isinstance(requested_shares, bool) or not isinstance(requested_shares, int) or requested_shares <= 0:
            blockers.append("卖出计划股数缺失或无效")
        elif isinstance(sellable_shares, int) and requested_shares > sellable_shares:
            blockers.append("卖出计划超过本地核验的可卖股数")
    return blockers


def _invalidate_unverified_intraday_analysis(result, current_price):
    result.current_price = current_price
    result.vwap = 0.0
    result.open_price = 0.0
    result.entry_quality_score = None
    result.nearest_support = None
    result.risk_warning = "分钟来源语义未核验，量额派生评分和技术位不可用"


def _execution_advice(result, prev_day, side="buy", urgency="normal", phase=None, blockers=None,
                      quote=None, security_name=None):
    """日内执行决策 — 把日内状态翻译成挂单动作 (L2, 供前端显示建议)。

    side/urgency 透传调仓信号 (buy/add/sell/reduce × normal/stop_loss/take_profit),
    买入侧由 analyze_intraday_entry 默认调用, 卖出侧由 sell_signals 逐条传入。
    """
    from datetime import datetime as dt
    from core.domain.portfolio.execution import ACTION_MARKET, decide_execution, resolve_phase

    phase = phase or resolve_phase(dt.now().strftime("%H:%M"))
    blockers = list(blockers) if blockers is not None else ["执行条件未核验"]
    decision = decide_execution(
        side=side, urgency=urgency, phase=phase,
        current_price=result.current_price, vwap=result.vwap,
        prev_close=prev_day.get("close", 0) or 0.0,
        open_price=result.open_price or None,
        entry_score=getattr(result, "entry_quality_score", None) or None,
        risk_warning=getattr(result, "risk_warning", None),
        support_price=getattr(result, "nearest_support", None).price if getattr(result, "nearest_support", None) else None,
        must_fill=True,
    )
    current_price = float(result.current_price or 0)
    limit_price = float(decision.limit_price) if decision.limit_price is not None else None
    is_sell = side in ("sell", "reduce")
    if decision.action == ACTION_MARKET:
        blockers.append("真实券商执行链仅支持限价单；保留止损意图，须核验并改为合法限价委托")
    if limit_price is not None:
        limit_up, limit_down, limit_blockers = _intraday_price_limit_bounds(
            getattr(result, "code", ""), quote, security_name,
        )
        blockers.extend(limit_blockers)
        if limit_up is not None and limit_down is not None:
            from decimal import Decimal, InvalidOperation
            try:
                proposed_price = Decimal(str(limit_price))
                if not proposed_price.is_finite() or proposed_price < limit_down or proposed_price > limit_up:
                    blockers.append("建议限价超出当日涨跌停范围，不能作为可执行委托")
            except (InvalidOperation, TypeError, ValueError):
                blockers.append("建议限价无效，不能作为可执行委托")
    blockers = list(dict.fromkeys(blockers))
    if blockers:
        status = "blocked"
        can_execute = False
        trigger_text = "执行资格阻断：" + "；".join(blockers)
        distance_pct = abs(current_price - limit_price) / current_price if limit_price and current_price > 0 else None
    elif decision.action in ("market", "fill"):
        status = "ready"
        can_execute = True
        trigger_text = "已进入可成交状态，可立即执行"
        distance_pct = 0.0
    elif decision.action == "limit" and limit_price and current_price > 0:
        can_execute = bool(current_price >= limit_price if is_sell else current_price <= limit_price)
        status = "ready" if can_execute else "waiting"
        distance_pct = abs(current_price - limit_price) / current_price
        if can_execute:
            trigger_text = "价格已进入成交范围，可挂单成交"
        elif is_sell:
            trigger_text = f"等待价格升至 {limit_price:.2f}"
        else:
            trigger_text = f"等待价格回落至 {limit_price:.2f}"
    else:
        status = "blocked"
        can_execute = False
        trigger_text = "当前不执行"
        distance_pct = None
    return {
        "action": decision.action,
        "intent_action": ACTION_MARKET if side in ("sell", "reduce") and urgency == "stop_loss" else decision.action,
        "limit_price": limit_price,
        "reason": decision.reason,
        "phase": phase,
        "current_price": current_price,
        "status": status,
        "can_execute": bool(can_execute),
        "blockers": blockers,
        "trigger_text": trigger_text,
        "distance_pct": float(round(distance_pct, 4)) if distance_pct is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Preview Stocks — 选股预览: 日K + 支撑压力 + 建议价位
# ═══════════════════════════════════════════════════════════════════════════

def _ths_float(v):
    try:
        return round(float(v), 3)
    except (ValueError, TypeError):
        return None


def _preview_single_stock(code, reference_date, regime_stops, stop_mode="fixed"):
    """对单只股票计算预览数据（K线、支撑压力、建议价位）"""
    import numpy as np
    from datetime import date, timedelta
    import pandas as pd
    from core.shared.trading_time import get_latest_trading_day

    result = {
        "code": code,
        "name": "",
        "realtime": None,
        "last_close": None,
        "execution_allowed": False,
        "price_basis": "qfq",
        "kline_price_basis": "qfq",
        "price_status": "unverified_basis",
        "adjustment_scale": None,
        "adjustment_anchor": None,
        "kline": [],
        "supports": [],
        "resistances": [],
        "suggested_buy": None,
        "stop_loss_ref": None,
        "take_profit_ref": None,
        "entry_quality": None,
        "technical_analysis": None,
    }

    # 1. 日K线 — 统一使用 daily_quotes 前复权口径
    try:
        from core.data.connection import Database
        ref = date.fromisoformat(reference_date[:10]) if reference_date else date.today()
        ref_end = ref.isoformat()
        ref_start = (ref - timedelta(days=180)).isoformat()
        rows = Database.fetchall(
            "SELECT trade_date, open, high, low, close, volume, amount, pct_chg, adj_factor "
            "FROM daily_quotes WHERE code=%s AND trade_date>=%s AND trade_date<=%s "
            "ORDER BY trade_date ASC",
            (code, ref_start, ref_end))
        if rows:
            df = pd.DataFrame(rows)
            for col in ('open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'adj_factor'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df = None
        if df is None or df.empty:
            return result
    except Exception:
        return result

    # 名称 — 从实时行情或 DB 取
    try:
        from core.data.repos.holdings import HoldingsRepo
        h = HoldingsRepo.get_by_code(code)
        if h and h.get("name"):
            result["name"] = h["name"]
    except Exception:
        pass

    # 2. 技术指标
    try:
        from core.domain.screening.indicators import Analyzer
        df = Analyzer.add_indicators(df)
    except Exception:
        pass

    quote = None
    if ref == date.today():
        try:
            from backend.service.market_data import MarketDataService
            live = MarketDataService.get_live_quotes([code], use_case="execution")
            quote = live["data"].get(code)
            if quote:
                if not result["name"]:
                    result["name"] = quote.get("name", "")
                result["realtime"] = {
                    "price": quote.get("price"),
                    "prev_close": quote.get("prev_close"),
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "pct_chg": quote.get("pct_chg"),
                    "volume": quote.get("volume", 0),
                    "source": quote.get("source"),
                    "observed_at": quote.get("observed_at"),
                    "age_seconds": live["meta"].get("age_seconds"),
                }
        except Exception as e:
            logger.debug(f"[Preview] 实时行情不可用 {code}: {e}")
    else:
        result["price_status"] = "historical_preview"

    last = df.iloc[-1]
    last_close = float(last.get("close") or 0)
    if not np.isfinite(last_close) or last_close <= 0:
        result["price_status"] = "invalid_daily_close"
        return result
    result["last_close"] = {
        "price": last_close,
        "date": str(last.get("trade_date") or last.get("date") or "")[:10],
        "source": "historical",
        "price_basis": "qfq",
    }
    current_price = result["last_close"]["price"]
    if quote:
        from core.domain.market_data.price_basis import qfq_to_live_raw_scale
        previous_date = get_latest_trading_day(date.today() - timedelta(days=1))
        scale = qfq_to_live_raw_scale(
            last.get("close"), last.get("adj_factor"), last.get("trade_date"),
            previous_date, quote.get("prev_close"),
            df["adj_factor"].tail(60) if "adj_factor" in df.columns else [],
        )
        if scale is not None:
            result["execution_allowed"] = True
            result["price_basis"] = "raw"
            result["price_status"] = "verified"
            result["adjustment_scale"] = scale
            result["adjustment_anchor"] = str(last.get("trade_date"))[:10]
            current_price = float(quote.get("price"))

    if ref == date.today():
        try:
            from core.domain.screening.indicators import Analyzer
            analysis_price = current_price / result["adjustment_scale"] if result["execution_allowed"] else current_price
            result["technical_analysis"] = Analyzer.technical_signal_summary(df, analysis_price)
            if result["technical_analysis"]:
                result["technical_analysis"]["price_source"] = (
                    (result["realtime"] or {}).get("source") if result["execution_allowed"] else "daily_close_qfq"
                )
        except Exception as error:
            logger.warning("[Preview] 技术指标判读失败 %s: %s", code, error)

    # 4. K线序列 (最近90天, 跳过含空值的行)
    tail = df.tail(120)
    for _, row in tail.iterrows():
        o = row.get("open")
        h = row.get("high")
        l = row.get("low")
        c = row.get("close")
        v = row.get("volume")
        if any(value is None or not np.isfinite(float(value)) for value in (o, h, l, c)):
            continue
        d = row.get("trade_date") or row.name
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        else:
            d = str(d)[:10]
        result["kline"].append({
            "date": d,
            "open": round(float(o), 3),
            "high": round(float(h), 3),
            "low": round(float(l), 3),
            "close": round(float(c), 3),
            "volume": int(v) if v is not None and np.isfinite(float(v)) else 0,
        })
    result["kline"] = result["kline"][-90:]

    if not result["execution_allowed"]:
        return result

    # 5. 支撑位/压力位（从日K指标取）
    last_row = df.iloc[-1]
    supports = []
    resistances = []

    for label, col in [("MA20", "MA20"), ("MA60", "MA60"), ("MA5", "MA5"), ("MA10", "MA10")]:
        if col in df.columns:
            val = float(last_row[col]) * result["adjustment_scale"]
            if np.isfinite(val) and val > 0:
                if val < current_price:
                    supports.append({"name": label, "price": round(val, 3)})
                else:
                    resistances.append({"name": label, "price": round(val, 3)})
    if "BB_lower" in df.columns:
        val = float(last_row["BB_lower"]) * result["adjustment_scale"]
        if np.isfinite(val) and val > 0 and val < current_price:
            supports.append({"name": "布林下轨", "price": round(val, 3)})
    if "BB_upper" in df.columns:
        val = float(last_row["BB_upper"]) * result["adjustment_scale"]
        if np.isfinite(val) and val > 0 and val > current_price:
            resistances.append({"name": "布林上轨", "price": round(val, 3)})

    # 前日高低
    try:
        prev_d = get_latest_trading_day(ref - timedelta(days=1)).isoformat()
        from core.data.repos.quotes import DailyQuotesRepo
        prev_df = DailyQuotesRepo.get_range(code, prev_d, prev_d)
        if prev_df is not None and not prev_df.empty:
            prev_row = prev_df.sort_values("trade_date").iloc[-1]
            prev_low = float(prev_row.get("low", 0)) * result["adjustment_scale"]
            prev_high = float(prev_row.get("high", 0)) * result["adjustment_scale"]
            if prev_low > 0 and prev_low < current_price:
                supports.append({"name": "前日低点", "price": round(prev_low, 3)})
            if prev_high > 0 and prev_high > current_price:
                resistances.append({"name": "前日高点", "price": round(prev_high, 3)})
    except Exception:
        pass

    # 排序：支撑位价高在前，压力位价低在前
    supports.sort(key=lambda x: -x["price"])
    resistances.sort(key=lambda x: x["price"])
    result["supports"] = supports[:5]
    result["resistances"] = resistances[:5]

    # 5b. MA 值 (供日内检测面板展示支撑位)
    result["ma"] = {}
    for col, key in [("MA5", "ma5"), ("MA10", "ma10"), ("MA20", "ma20")]:
        if col in df.columns:
            val = float(last_row[col])
            if np.isfinite(val) and val > 0:
                result["ma"][key] = round(val * result["adjustment_scale"], 3)

    # 6. 建议价位 — 盘中动态 VWAP 锚, 无当日分钟数据回退静态支撑
    vwap = _get_intraday_vwap(code, ref.isoformat())
    if vwap and vwap > 0:
        result["suggested_buy"] = {
            "low": round(vwap * 0.997, 3),
            "high": round(vwap, 3),
        }
        result["buy_anchor"] = "vwap"
    else:
        buy_low = supports[0]["price"] if supports else current_price * 0.98
        buy_high = supports[1]["price"] if len(supports) > 1 else current_price * 0.99
        if buy_low > buy_high:
            buy_low, buy_high = buy_high, buy_low
        result["suggested_buy"] = {
            "low": round(buy_low, 3),
            "high": round(buy_high, 3),
        }
        result["buy_anchor"] = "support"

    # 7. 止损/止盈参考 (支持 fixed / dynamic / combined 三种模式)
    # 先算固定百分比基准
    if regime_stops:
        fixed_stop = round(current_price * (1 + regime_stops.get("stop_loss_full", -0.10)), 3)
        fixed_profit = round(current_price * (1 + regime_stops.get("take_profit_full", 0.30)), 3)
    else:
        fixed_stop = round(current_price * 0.90, 3)
        fixed_profit = round(current_price * 1.30, 3)

    nearest_support = supports[0]["price"] if supports else None
    nearest_resistance = resistances[0]["price"] if resistances else None
    dyn_stop = round(nearest_support * 0.98, 3) if nearest_support else fixed_stop
    dyn_profit = round(nearest_resistance, 3) if nearest_resistance else fixed_profit

    if stop_mode == "dynamic":
        result["stop_loss_ref"] = dyn_stop
        result["take_profit_ref"] = dyn_profit
    elif stop_mode == "combined":
        result["stop_loss_ref"] = max(dyn_stop, fixed_stop)   # 取更紧的止损
        result["take_profit_ref"] = min(dyn_profit, fixed_profit)  # 取更近的止盈
    else:
        result["stop_loss_ref"] = fixed_stop
        result["take_profit_ref"] = fixed_profit
    result["stop_mode"] = stop_mode

    # 8. 入场质量（日K级别，不依赖分钟数据）
    p_low = supports[0]["price"] if supports else current_price
    dist = (current_price - p_low) / p_low if p_low > 0 else 0.05
    if dist <= 0.01:
        score, label = 70.0, "优秀"
    elif dist <= 0.02:
        score, label = 55.0, "良好"
    elif dist <= 0.03:
        score, label = 40.0, "一般"
    else:
        score, label = 25.0, "观望"
    result["entry_quality"] = {"score": round(score, 1), "label": label, "signals": []}

    return result


def _get_regime_stop_dict():
    """获取当前市场状态下的止盈止损参数 (dict)"""
    try:
        from core.domain.regime.classifier import get_current_regime
        raw = get_current_regime()
        if raw:
            info = raw.to_dict() if hasattr(raw, 'to_dict') else raw
            regime = info.get("regime", "")
            stops = _load_regime_stops()
            for name, params in stops.items():
                if name == regime:
                    return {
                        "stop_loss_warn": params.get("warn", -0.05),
                        "stop_loss_full": params.get("full", -0.10),
                        "take_profit_half": params.get("profitHalf", 0.15),
                        "take_profit_full": params.get("profitFull", 0.30),
                        "source": f"regime:{regime}"
                    }
    except Exception:
        pass
    return {"stop_loss_full": -0.10, "take_profit_full": 0.30, "source": "default"}
