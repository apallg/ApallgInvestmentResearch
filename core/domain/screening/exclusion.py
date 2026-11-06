"""
统一排除规则
过滤 ETF/LOF/B 股/ST 股/退市股
"""

import datetime
import math

import pandas as pd

ETF_LOF_PREFIXES = ("51", "52", "56", "58", "15", "16")
B_SHARE_PREFIXES = ("920", "900", "200")

EXCLUDED_PREFIXES = ETF_LOF_PREFIXES + B_SHARE_PREFIXES

EXCLUDED_SUFFIXES = (".BJ",)   # 北交所 (可选排除)


def is_excluded(code: str, name: str = "") -> (bool, str):
    """返回 (是否排除, 原因)"""
    if not code or len(code) < 3:
        return True, "invalid_code"

    code_str = str(code).split(".")[0]
    for prefix in EXCLUDED_PREFIXES:
        if code_str.startswith(prefix):
            return True, f"excluded_prefix:{prefix}"

    if any(code_str.endswith(suffix) or str(code).endswith(suffix)
           for suffix in EXCLUDED_SUFFIXES):
        return True, "beijing_exchange"

    if name and ("ST" in name or "*ST" in name):
        return True, "st_stock"

    return False, ""


def filter_codes(codes_with_names: list) -> list:
    """输入 [(code, name), ...]，返回过滤后的 [(code, name), ...]"""
    return [(c, n) for c, n in codes_with_names if not is_excluded(c, n)[0]]


def evaluate_decision_day_filter(code, df, decision_date, name="", min_turnover=10000000,
                                 min_price=2.0, min_listing_days=60,
                                 latest_adj_factor=None, limit_pct=0.10,
                                 min_history_observations=None, listing_date=None):
    result = {
        "reason": "data_error",
        "close": 0.0,
        "volume": 0.0,
        "pct_chg": 0.0,
        "avg_turnover": 0.0,
        "raw_close": 0.0,
    }
    if df is None or df.empty:
        return result

    if "trade_date" in df.columns:
        dates = pd.to_datetime(df["trade_date"], errors="coerce")
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = pd.Series(df.index, index=df.index)
    else:
        return result

    decision_ts = pd.Timestamp(decision_date).normalize()
    history = df.loc[dates <= decision_ts]
    history_dates = dates.loc[dates <= decision_ts]
    if history.empty:
        return result
    if (min_history_observations is not None
            and int((dates < decision_ts).sum()) < int(min_history_observations)):
        result["reason"] = "insufficient_history"
        return result

    exact_mask = history_dates.dt.normalize() == decision_ts
    exact = history.loc[exact_mask]
    row = exact.iloc[-1] if not exact.empty else history.iloc[-1]

    try:
        close = float(row.get("close", 0) or 0)
    except (TypeError, ValueError):
        close = 0.0
    try:
        volume = float(row.get("volume", 0) or 0)
    except (TypeError, ValueError):
        volume = 0.0
    try:
        pct_chg = float(row.get("pct_chg", 0) or 0)
    except (TypeError, ValueError):
        pct_chg = 0.0
    if pd.isna(close):
        close = 0.0
    if pd.isna(volume):
        volume = 0.0
    if pd.isna(pct_chg):
        pct_chg = 0.0
    if pct_chg == 0 and len(history) >= 2:
        try:
            prev_close = float(history.iloc[-2].get("close", 0) or 0)
            if prev_close > 0:
                pct_chg = (close / prev_close - 1) * 100
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    avg_turnover = 0.0
    if "amount" in history.columns:
        amount = pd.to_numeric(history.tail(20)["amount"], errors="coerce")
        if amount.notna().any():
            avg_turnover = float(amount.fillna(0).mean())

    raw_close = close
    try:
        row_adj = float(row.get("adj_factor", 0) or 0)
        latest_adj = float(latest_adj_factor or 0)
        if row_adj > 0 and latest_adj > 0:
            raw_close = close * latest_adj / row_adj
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    result.update({
        "close": close,
        "volume": volume,
        "pct_chg": pct_chg,
        "avg_turnover": avg_turnover,
        "raw_close": raw_close,
    })

    upper_name = str(name or "").upper()
    if upper_name.startswith("ST") or upper_name.startswith("*ST"):
        result["reason"] = "st"
    elif exact.empty or close <= 0 or volume <= 0:
        result["reason"] = "suspended"
    elif pct_chg >= float(limit_pct) * 100 - 0.5:
        result["reason"] = "limit_up"
    elif "amount" not in history.columns:
        result["reason"] = "data_error"
    elif avg_turnover < float(min_turnover):
        result["reason"] = "low_liquidity"
    elif raw_close < float(min_price):
        result["reason"] = "low_price"
    else:
        cutoff = decision_ts.date() - datetime.timedelta(days=int(min_listing_days))
        if listing_date is None:
            result["reason"] = "identity_unknown"
        else:
            try:
                listing_day = pd.Timestamp(listing_date).date()
            except (TypeError, ValueError):
                result["reason"] = "identity_unknown"
            else:
                result["reason"] = "new_listing" if listing_day > cutoff else None
    return result


def prefilter_by_turnover(codes, top_n, turnover_data):
    """按成交额预筛选，保留流动性最好的 top_n 只。
    纯 L3 函数——turnover_data 需由调用方预先从 DB 获取。

    Args:
        codes: 候选代码列表
        top_n: 保留数量
        turnover_data: {code: avg_amount} dict, 由 L2/L4 层预先查询

    Returns:
        按成交额降序排列的 top_n 代码列表
    """
    if not codes or top_n <= 0:
        return codes
    if not turnover_data:
        return codes[:top_n]
    ranked = sorted(turnover_data.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n] if c in codes]


def prefilter_by_index(codes, valid_members):
    """按指数成分预筛选，只保留属于指定指数的代码。
    纯 L3 函数——valid_members 需由调用方预先从 Qlib/数据源获取。

    Args:
        codes: 候选代码列表
        valid_members: set of valid codes (裸代码，如 '600519')

    Returns:
        过滤后的代码列表
    """
    if not valid_members:
        return codes
    return [c for c in codes if c in valid_members]


def prefilter_by_quality(codes, quality_data, require_complete=False):
    """宽阈值基本面硬过滤 — 剔除明显问题股, 不配置化、不误杀。

    纯 L3 函数——quality_data 需由 L2 从 fundamental_data 取最近一期 (PIT ann_date)。

    Args:
        codes: 候选代码列表
        quality_data: {code: {"profit_dedt": 扣非净利润, "debt_to_assets": 资产负债率,
                              "industry": 行业, "name": 证券名称}}

    Returns:
        (kept_codes, dropped): dropped = [(code, reason), ...]
    """
    if not quality_data and not require_complete:
        return codes, []
    if not isinstance(quality_data, dict):
        quality_data = {}
    kept, dropped = [], []
    for code in codes:
        q = quality_data.get(code)
        if not q or (require_complete and not isinstance(q, dict)):
            if require_complete:
                dropped.append((code, "基本面质量数据缺失，禁止新增风险"))
                continue
            kept.append(code)
            continue
        if require_complete:
            profit = q.get("profit_dedt")
            if isinstance(profit, bool):
                profit = None
            else:
                try:
                    profit = float(profit)
                except (TypeError, ValueError):
                    profit = None
            if profit is None or not math.isfinite(profit):
                dropped.append((code, "扣非净利润质量状态未知，禁止新增风险"))
                continue
            industry = str(q.get("industry") or "").strip()
            name = str(q.get("name") or "").strip()
            is_financial = (
                industry.startswith(("J66", "J67", "J68", "J69"))
                or any(label in industry for label in ("银行", "货币金融", "证券", "保险", "多元金融"))
                or name.endswith("银行")
            )
            debt = q.get("debt_to_assets")
            if isinstance(debt, bool):
                debt = None
            if debt is None and not is_financial:
                dropped.append((code, "资产负债率质量状态未知，禁止新增风险"))
                continue
            if debt is not None:
                try:
                    debt = float(debt)
                except (TypeError, ValueError):
                    debt = None
                if (debt is None or not math.isfinite(debt)) and not is_financial:
                    dropped.append((code, "资产负债率质量状态未知，禁止新增风险"))
                    continue
            if profit < 0:
                dropped.append((code, "扣非净利润为负"))
                continue
            if not is_financial and debt is not None and debt > 85:
                dropped.append((code, f"资产负债率 {debt:.0f}%"))
                continue
            kept.append(code)
            continue
        if q.get("profit_dedt") is not None and q["profit_dedt"] < 0:
            dropped.append((code, "扣非净利润为负"))
            continue
        industry = str(q.get("industry") or "").strip()
        name = str(q.get("name") or "").strip()
        is_financial = (
            industry.startswith(("J66", "J67", "J68", "J69"))
            or any(label in industry for label in ("银行", "货币金融", "证券", "保险", "多元金融"))
            or name.endswith("银行")
        )
        if (not is_financial and q.get("debt_to_assets") is not None
                and q["debt_to_assets"] > 85):
            dropped.append((code, f"资产负债率 {q['debt_to_assets']:.0f}%"))
            continue
        kept.append(code)
    return kept, dropped


def prefilter_by_liquidity_capacity(
        codes, amount_data, planned_amounts=None, min_amount=10000000,
        warn_participation=0.015, max_participation=0.02):
    """按20日成交额中位数和计划下单容量过滤候选。纯 L3 函数。

    Args:
        codes: 候选代码列表
        amount_data: {code: 20日成交额中位数(元)}
        planned_amounts: {code: 计划买入/加仓金额(元)}
        min_amount: 最低20日成交额中位数, 默认 1000 万元

    Returns:
        (kept_codes, dropped, checks)
    """
    kept, dropped = [], []
    checks = {}
    amount_data = amount_data or {}
    planned_amounts = planned_amounts or {}
    for code in codes:
        amt = amount_data.get(code)
        planned = planned_amounts.get(code)
        participation = planned / amt if planned is not None and amt and amt > 0 else None
        status = "normal"
        reason = ""
        if amt is None:
            status = "excluded"
            reason = "近20日有效成交额不足15日，无法计算下单容量"
        elif amt < min_amount:
            status = "excluded"
            reason = f"20日成交额中位数 {amt / 1e4:.0f}万 < {min_amount / 1e4:.0f}万"
        elif participation is not None and participation > max_participation:
            status = "excluded"
            reason = f"计划下单占20日成交额中位数 {participation:.1%} > {max_participation:.0%}"
        elif participation is not None and participation > warn_participation:
            status = "warning"
        checks[code] = {
            "median_amount_20d": amt,
            "planned_amount": planned,
            "participation_rate": participation,
            "status": status,
        }
        if reason:
            dropped.append((code, reason))
        else:
            kept.append(code)
    return kept, dropped, checks


def prefilter_by_min_amount(codes, amount_data, min_amount=10000000):
    kept, dropped, _ = prefilter_by_liquidity_capacity(
        codes, amount_data, min_amount=min_amount)
    return kept, dropped
