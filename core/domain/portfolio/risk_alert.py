"""
持仓止盈止损监控告警

定时扫描持仓盈亏，检查固定止盈止损线和动态支撑压力线。
只在触发时通过 LLM 组织语言推送企微通知，避免噪音。
"""
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

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
    stops = dict(_DEFAULT_REGIME_STOPS)
    try:
        from core.shared.persistence import load_json
        cfg_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "regime_ratio.json"
        cfg = load_json(str(cfg_path))
        if cfg:
            raw_stops = cfg.get("stops", {})
            for k, v in raw_stops.items():
                idx = int(k)
                if 0 <= idx < len(_STATE_NAMES):
                    name = _STATE_NAMES[idx]
                    _w = float(v.get("warn", 5))
                    _f = float(v.get("full", 10))
                    if _w == 0 or _f == 0:
                        continue
                    stops[name] = {
                        "warn": -abs(_w) / 100,
                        "full": -abs(_f) / 100,
                        "profitHalf": float(v.get("profitHalf", 15)) / 100,
                        "profitFull": float(v.get("profitFull", 30)) / 100,
                    }
    except Exception as e:
        logger.warning(f"[RiskAlert] 加载止盈止损配置失败, 使用默认值: {e}")
    return stops


def _get_current_regime():
    try:
        from core.domain.regime.classifier import get_current_regime
        raw = get_current_regime()
        if raw:
            info = raw.to_dict() if hasattr(raw, 'to_dict') else raw
            return info.get("regime", "steady_flat")
    except Exception as e:
        logger.warning(f"[RiskAlert] 获取市场状态失败: {e}")
    return "steady_flat"


def _get_prices(codes):
    from core.shared import config
    from core.shared.data_contract import data_envelope_allows_use, make_data_envelope, make_live_quote_envelope
    from core.shared.trading_time import A_SHARE_TIMEZONE, is_trading_day
    from core.shared.persistence import now_iso

    now = datetime.fromisoformat(now_iso())
    now = now.replace(tzinfo=A_SHARE_TIMEZONE) if now.tzinfo is None else now.astimezone(A_SHARE_TIMEZONE)
    try:
        expected_session_date = now.date().isoformat() if codes and is_trading_day(now) else None
    except Exception:
        expected_session_date = None
    try:
        from core.data.sources.realtime import RealtimeEngine
        from core.domain.market_data.freshness import assess_batch
        if not codes:
            checked = {
                "accepted": {},
                "statuses": {},
                "missing_codes": [],
                "coverage": {"requested": 0, "accepted": 0, "missing": 0},
                "max_age_seconds": None,
                "freshness": "unavailable",
            }
            quotes = {}
        else:
            quotes = RealtimeEngine().get_quotes(codes) or {}
            checked = assess_batch(
                quotes,
                codes,
                now,
                config.LIVE_QUOTE_MAX_AGE,
                expected_session_date=expected_session_date,
                require_complete=False,
            )
        accepted = dict(checked["accepted"])
        quote_statuses = checked["statuses"]
        row_contract_failures = {}
        for code, quote in list(accepted.items()):
            row_envelope = make_live_quote_envelope(quote)
            if not data_envelope_allows_use(row_envelope, "risk"):
                errors = row_envelope.get("validation_errors") or [
                    row_envelope.get("missing_reason") or "data_contract_unverified"
                ]
                row_contract_failures[code] = errors
                accepted.pop(code)
                quote_statuses[code] = {
                    **quote_statuses[code],
                    "usable": False,
                    "freshness": "unavailable",
                    "missing_reason": errors[0],
                    "data_contract_failures": errors,
                }
        checked["accepted"] = accepted
        checked["coverage"] = {
            "requested": len(codes),
            "accepted": len(accepted),
            "missing": len(codes) - len(accepted),
        }
        checked["freshness"] = (
            "fresh" if len(accepted) == len(codes) and codes else
            "degraded" if accepted else "unavailable"
        )
        checked["partial"] = bool(accepted) and len(accepted) < len(codes)
        sources = sorted({quote.get("source") for quote in accepted.values() if quote.get("source")})
        source = sources[0] if len(sources) == 1 else ("mixed" if sources else None)
        observed_at = max((quote.get("observed_at", "") for quote in accepted.values()), default=None)
        available_at = max((quote.get("received_at", "") for quote in accepted.values()), default=None)
        as_of = expected_session_date or max(
            (str(quote.get("session_date")) for quote in accepted.values() if quote.get("session_date")),
            default=None,
        )
        quote_data = {
            "source": source,
            "source_chain": list(config.REALTIME_QUOTE_CHAIN),
            "observed_at": observed_at,
            "available_at": available_at,
            "as_of": as_of,
            "freshness": checked["freshness"],
            "max_age_seconds": config.LIVE_QUOTE_MAX_AGE,
            "coverage": checked["coverage"],
            "quote_statuses": quote_statuses,
            "data_envelope": make_data_envelope(
                source=source,
                observed_at=observed_at,
                available_at=available_at,
                as_of=as_of,
                unit={"price": "CNY/share", "volume": "share", "amount": "yuan"},
                price_basis="raw",
                coverage=checked["coverage"],
                status={
                    "fresh": "available", "degraded": "partial", "unavailable": "unavailable",
                }[checked["freshness"]],
                missing_reason={
                    code: status["missing_reason"]
                    for code, status in quote_statuses.items()
                    if status.get("missing_reason")
                } or row_contract_failures or ("no_holdings" if not codes else None),
            ),
        }
        return (
            {
                code: float(quote.get("price") or quote.get("last_price") or 0)
                for code, quote in accepted.items()
            },
            quote_statuses,
            accepted,
            quote_data,
        )
    except Exception as e:
        logger.error(f"[RiskAlert] 实时行情获取失败: {e}")
        statuses = {
            code: {"freshness": "unavailable", "missing_reason": "quote_fetch_failed"}
            for code in codes
        }
        coverage = {"requested": len(codes), "accepted": 0, "missing": len(codes)}
        return {}, statuses, {}, {
            "source": None,
            "source_chain": list(config.REALTIME_QUOTE_CHAIN),
            "observed_at": None,
            "available_at": None,
            "as_of": expected_session_date,
            "freshness": "unavailable",
            "max_age_seconds": config.LIVE_QUOTE_MAX_AGE,
            "coverage": coverage,
            "quote_statuses": statuses,
            "data_envelope": make_data_envelope(
                source=None,
                as_of=expected_session_date,
                unit={"price": "CNY/share", "volume": "share", "amount": "yuan"},
                price_basis="raw",
                coverage=coverage,
                status="unavailable",
                missing_reason={code: "quote_fetch_failed" for code in codes} or "no_holdings",
            ),
        }


def _check_fixed_stops(shares, cost_price, current_price, stops):
    if shares <= 0 or cost_price <= 0 or current_price <= 0:
        return []

    pnl_pct = (current_price - cost_price) / cost_price
    signals = []

    stop_warn = stops.get("warn", -0.05)
    stop_full = stops.get("full", -0.10)
    profit_half = stops.get("profitHalf", 0.15)
    profit_full = stops.get("profitFull", 0.30)

    if pnl_pct <= stop_full:
        signals.append({"type": "fixed_stop_loss_full", "action": "止损清仓",
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "threshold_pct": round(stop_full * 100, 2), "severity": "high"})
    elif pnl_pct >= profit_full:
        signals.append({"type": "fixed_take_profit_full", "action": "止盈清仓",
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "threshold_pct": round(profit_full * 100, 2), "severity": "high"})
    elif pnl_pct <= stop_warn:
        signals.append({"type": "fixed_stop_loss_warn", "action": "止损减半",
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "threshold_pct": round(stop_warn * 100, 2), "severity": "medium"})
    elif pnl_pct >= profit_half:
        signals.append({"type": "fixed_take_profit_half", "action": "止盈减半",
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "threshold_pct": round(profit_half * 100, 2), "severity": "medium"})

    return signals


def _get_kline_df(code):
    try:
        from core.data.repos.quotes import DailyQuotesRepo
        from core.shared.persistence import today_date
        end = today_date().isoformat()
        start = (today_date() - timedelta(days=120)).isoformat()
        df = DailyQuotesRepo.get_range(code, start, end)
        if df is None or df.empty or len(df) < 20:
            return None
        df = df.sort_values("trade_date")
        close = df["close"].astype(float)
        df["MA5"] = close.rolling(5).mean()
        df["MA10"] = close.rolling(10).mean()
        df["MA20"] = close.rolling(20).mean()
        df["MA60"] = close.rolling(60).mean()
        ma20 = df["MA20"]
        std20 = close.rolling(20).std()
        df["BB_upper"] = ma20 + 2 * std20
        df["BB_lower"] = ma20 - 2 * std20
        return df
    except Exception as e:
        logger.debug(f"[RiskAlert] 获取 {code} K线失败: {e}")
        return None


def _check_dynamic_levels(current_price, df, scale):
    if df is None or df.empty:
        return []

    last = df.iloc[-1]
    levels = []
    for label, col in [("MA5", "MA5"), ("MA10", "MA10"), ("MA20", "MA20"), ("MA60", "MA60"),
                        ("布林下轨", "BB_lower"), ("布林上轨", "BB_upper")]:
        val = last.get(col)
        if val is None or not np.isfinite(float(val)) or float(val) <= 0:
            continue
        val = float(val) * scale
        dist_pct = (current_price - val) / val * 100
        levels.append({"name": label, "price": round(val, 3), "dist_pct": round(dist_pct, 2)})

    supports = [l for l in levels if l["price"] < current_price]
    resistances = [l for l in levels if l["price"] > current_price]
    nearest_support = max(supports, key=lambda x: x["price"]) if supports else None
    nearest_resistance = min(resistances, key=lambda x: x["price"]) if resistances else None

    signals = []
    if nearest_support and nearest_support["dist_pct"] <= 2.0:
        severity = "high" if nearest_support["dist_pct"] <= 0 else "medium"
        signals.append({"type": "dynamic_near_support", "action": "接近支撑位",
                        "level_name": nearest_support["name"],
                        "level_price": nearest_support["price"],
                        "distance_pct": nearest_support["dist_pct"], "severity": severity,
                        "price_basis": "raw", "reference_date": str(last.get("trade_date"))[:10],
                        "conversion_scale": round(scale, 8)})
    if nearest_resistance and nearest_resistance["dist_pct"] >= -2.0:
        signals.append({"type": "dynamic_near_resistance", "action": "接近压力位",
                        "level_name": nearest_resistance["name"],
                        "level_price": nearest_resistance["price"],
                        "distance_pct": nearest_resistance["dist_pct"], "severity": "medium",
                        "price_basis": "raw", "reference_date": str(last.get("trade_date"))[:10],
                        "conversion_scale": round(scale, 8)})

    return signals


def _build_llm_prompt(triggered, session_name, regime_name):
    lines = []
    for t in triggered:
        code = t["code"]
        name = t["name"]
        cost = t["cost_price"]
        price = t["current_price"]
        pnl = t["pnl_pct"]
        shares = t["shares"]
        mv = t["market_value"]

        lines.append(f"- {name}({code}): 成本{cost:.2f} 现价{price:.2f} "
                     f"盈亏{pnl:+.2f}% 持仓{shares}股 市值约{mv:.1f}万")
        for sig in t.get("signals", []):
            if sig["type"].startswith("fixed"):
                lines.append(f"  [{sig['severity']}] {sig['action']}: 盈亏{sig['pnl_pct']:+.2f}% "
                           f"(触发线 {sig['threshold_pct']:+.2f}%)")
            elif "support" in sig["type"]:
                lines.append(f"  [{sig['severity']}] {sig['action']}: "
                           f"距{sig['level_name']}({sig['level_price']:.2f}) {sig['distance_pct']:+.2f}%")
            else:
                lines.append(f"  [{sig['severity']}] {sig['action']}: "
                           f"距{sig['level_name']}({sig['level_price']:.2f}) {sig['distance_pct']:+.2f}%")

    holdings_text = "\n".join(lines)

    return f"""你是A股交易风险监控助手。当前**{session_name}**扫描，市场状态 **{regime_name}**。

以下持仓触发了止盈止损线或动态支撑压力位，请用简洁专业的语言汇总告警，生成一条企微 markdown 消息。

## 触发持仓

{holdings_text}

## 要求
1. 标题 "## 🚨 {session_name}止盈止损告警"
2. 每只触发股票简要说明：触发原因、盈亏状况、建议操作
3. 优先报告止损信号（风险优先），再报告止盈信号
4. 动态支撑/压力信号作为补充提示
5. 总字数控制在 1000 字以内，简洁有力
6. 使用 markdown 格式，关键数字用 **加粗**
7. 末尾注明扫描时间和市场状态"""


def _build_alert_via_llm(triggered, session_name, regime_name):
    prompt = _build_llm_prompt(triggered, session_name, regime_name)

    try:
        from core.shared.infrastructure.llm import LLMClient
        llm = LLMClient()
        content = llm.chat([
            {"role": "system", "content": "你是交易风险监控助手，用简洁专业的语言汇总止盈止损告警。只输出 markdown 消息，不要额外说明。"},
            {"role": "user", "content": prompt},
        ], temperature=0.3, max_tokens=1024)
    except Exception as e:
        logger.error(f"[RiskAlert] LLM 调用失败，使用简化告警: {e}")
        content = _build_fallback_alert(triggered, session_name, regime_name)

    return content or _build_fallback_alert(triggered, session_name, regime_name)


def _build_fallback_alert(triggered, session_name, regime_name):
    from core.shared.persistence import now_iso
    now = now_iso()[11:19]
    lines = [
        f"## 🚨 {session_name}止盈止损告警",
        f"> 时间: {now} | 市场状态: {regime_name}",
        f"> 触发数量: **{len(triggered)}只**",
        "",
    ]

    for t in sorted(triggered, key=lambda x: (
        0 if any(s["severity"] == "high" for s in x["signals"]) else 1
    )):
        name = t["name"]
        code = t["code"]
        cost = t["cost_price"]
        price = t["current_price"]
        pnl = t["pnl_pct"]

        lines.append(f"### {name}({code})")
        lines.append(f"> 成本: **{cost:.2f}** | 现价: **{price:.2f}** | 盈亏: **{pnl:+.2f}%**")
        for sig in t["signals"]:
            emoji = "🔴" if sig["severity"] == "high" else "🟡"
            lines.append(f"> {emoji} {sig['action']}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 主入口 — 由调度器调用
# ═══════════════════════════════════════════════════════════════════════════

def run_stop_alert_scan(session_name="定时扫描", threshold=None):
    """
    扫描所有持仓的止盈止损状态并持久化本轮评估与通知任务。

    Args:
        session_name: 扫描时段名称（早盘/午盘/尾盘/手动）
        threshold: 兼容旧接口，实际不使用

    Returns:
        dict: 扫描覆盖和触发状态
    """
    try:
        from core.data import HoldingsRepo
        holdings = HoldingsRepo.get_all()
    except Exception as e:
        logger.error(f"[RiskAlert] 获取持仓失败: {e}")
        raise

    import uuid
    from core.shared.persistence import now_iso
    scan_id = str(uuid.uuid4())
    codes = [h["code"] for h in holdings]
    prices, quote_statuses, accepted_quotes, quote_data = _get_prices(codes)
    if not holdings:
        logger.info(f"[RiskAlert] {session_name}: 无持仓数据")
        from core.data import RiskAlertRepo
        RiskAlertRepo.record_scan({
            "scan_id": scan_id, "source": "stop_loss", "session_name": session_name,
            "regime_name": None, "status": "empty", "expected_count": 0,
            "scanned_count": 0, "unknown_count": 0, "dynamic_unknown_count": 0,
            "triggered_count": 0, "assessment": {
                "scan_at": now_iso(), "triggered": [], "unknown": [], "quote_data": quote_data,
            },
        }, [])
        return {"scan_id": scan_id, "expected": 0, "scanned": 0, "unknown": [], "triggered": [], "status": "empty"}

    regime_name = _get_current_regime()
    stops = _load_regime_stops().get(regime_name, _DEFAULT_REGIME_STOPS["steady_flat"])

    triggered = []
    unknown = []
    dynamic_unknown = []
    scanned = 0

    for h in holdings:
        code = h["code"]
        try:
            shares = int(h.get("shares", 0))
            cost_price = float(h.get("cost_price", 0) or 0)
        except (TypeError, ValueError):
            shares, cost_price = 0, 0
        if shares <= 0 or not np.isfinite(cost_price) or cost_price <= 0:
            unknown.append({"code": code, "reason": "invalid_holding"})
            continue

        current_price = prices.get(code, 0)
        if current_price <= 0:
            status = quote_statuses.get(code) or {}
            unknown.append({
                "code": code,
                "reason": status.get("missing_reason") or status.get("reason") or status.get("freshness") or "unavailable",
            })
            continue
        scanned += 1

        pnl_pct = (current_price - cost_price) / cost_price * 100
        market_value = shares * current_price / 10000

        all_signals = []

        fixed_signals = _check_fixed_stops(shares, cost_price, current_price, stops)
        all_signals.extend(fixed_signals)

        df = _get_kline_df(code)
        if df is not None:
            from core.domain.market_data.price_basis import qfq_to_live_raw_scale
            from core.shared.persistence import today_date
            from core.shared.trading_time import get_latest_trading_day

            last = df.iloc[-1]
            previous_date = get_latest_trading_day(today_date() - timedelta(days=1))
            quote = accepted_quotes.get(code) or {}
            scale = qfq_to_live_raw_scale(
                last.get("close"), last.get("adj_factor"), last.get("trade_date"),
                previous_date, quote.get("prev_close"),
                df["adj_factor"].tail(60) if "adj_factor" in df.columns else [],
            )
            if scale is not None:
                dynamic_signals = _check_dynamic_levels(current_price, df, scale)
                all_signals.extend(dynamic_signals)
            else:
                dynamic_unknown.append(code)
        else:
            dynamic_unknown.append(code)

        if all_signals:
            triggered.append({
                "code": code,
                "name": h.get("name", ""),
                "cost_price": round(cost_price, 3),
                "current_price": round(current_price, 3),
                "pnl_pct": round(pnl_pct, 2),
                "shares": shares,
                "market_value": round(market_value, 2),
                "signals": all_signals,
            })

    notifications = []
    if unknown or dynamic_unknown:
        logger.error(f"[RiskAlert] {session_name}: 应扫{len(holdings)}只，已扫{scanned}只，未知{len(unknown)}只: {unknown}")
        from core.shared.infrastructure.notify import _safe_truncate_bytes, WECOM_MARKDOWN_MAX_BYTES
        details = "、".join(f"{item['code']}({item['reason']})" for item in unknown)
        dynamic_details = "、".join(dynamic_unknown)
        content = f"## ⚠️ {session_name}持仓风控扫描不完整\n> 应扫{len(holdings)}只，已扫{scanned}只，未知行情{len(unknown)}只，动态价位不可用{len(dynamic_unknown)}只\n> 未知行情: {details or '--'}\n> 动态价位不可用: {dynamic_details or '--'}\n请检查行情与持仓数据，不要将未触发视为安全。"
        notifications.append({
            "idempotency_key": f"stop-alert:{scan_id}:incomplete",
            "payload": {"message_type": "markdown", "content": _safe_truncate_bytes(content, WECOM_MARKDOWN_MAX_BYTES)},
        })
    if dynamic_unknown:
        logger.warning(f"[RiskAlert] {session_name}: {len(dynamic_unknown)}只动态价位缺少可核验raw锚点: {dynamic_unknown}")
    if triggered:
        logger.info(f"[RiskAlert] {session_name}: {len(triggered)}只触发止盈止损线")
        from core.shared.infrastructure.notify import _safe_truncate_bytes, WECOM_MARKDOWN_MAX_BYTES
        content = _build_alert_via_llm(triggered, session_name, regime_name)
        notifications.append({
            "idempotency_key": f"stop-alert:{scan_id}:triggered",
            "payload": {"message_type": "markdown", "content": _safe_truncate_bytes(content, WECOM_MARKDOWN_MAX_BYTES)},
        })
    elif not unknown:
        logger.info(f"[RiskAlert] {session_name}: 无触发，{scanned}只持仓已扫描")

    status = "degraded" if unknown else ("partial" if dynamic_unknown else "complete")
    scan = {
        "scan_id": scan_id,
        "source": "stop_loss",
        "session_name": session_name,
        "regime_name": regime_name,
        "status": status,
        "expected_count": len(holdings),
        "scanned_count": scanned,
        "unknown_count": len(unknown),
        "dynamic_unknown_count": len(dynamic_unknown),
        "triggered_count": len(triggered),
        "assessment": {
            "scan_at": now_iso(),
            "triggered": triggered,
            "unknown": unknown,
            "dynamic_unknown": dynamic_unknown,
            "quote_data": quote_data,
        },
    }
    from core.data import RiskAlertRepo
    notification_ids = RiskAlertRepo.record_scan(scan, notifications)
    return {"scan_id": scan_id, "expected": len(holdings), "scanned": scanned, "unknown": unknown,
            "dynamic_unknown": dynamic_unknown, "triggered": triggered,
            "notification_ids": notification_ids, "status": status}
