"""
一致性风控 — 基于微盘股涨跌幅分布的踩踏检测

当市场最小的5%股票出现涨跌幅异常聚集时，判定为恐慌性踩踏，清仓避险。
- 仅在熊市（上证<240日均线）启用，牛市自动关闭
- 有状态机机制：触发清仓后需明确的恢复信号才会重新入场

参考: 聚宽策略 2026-28 (韶华不负)
"""
import json
import logging
import datetime as dt
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["check_panic", "reset_state"]

_STATE_FILE = Path(__file__).parent.parent / "data" / "consistency_state.json"
_LOOKBACK = 120  # BOLL带计算窗口
_MIN_STOCKS = 100  # 单日最少微盘股数
_MIN_HISTORY = 30  # 最少需要多少天数据才能计算


def _load_state():
    from core.shared.persistence import load_json
    if _STATE_FILE.exists():
        try:
            return load_json(str(_STATE_FILE))
        except Exception:
            pass
    return {"panic": False, "last_updated": None}


def _save_state(state):
    from core.shared.persistence import save_json
    save_json(state, str(_STATE_FILE))


def _latest_trade_date():
    """最新可用交易日"""
    from core.data import Database
    from core.shared.persistence import today_date
    if not Database.is_available():
        return today_date()
    rows = Database.fetchall("SELECT MAX(trade_date) AS d FROM daily_quotes")
    if rows and rows[0]["d"]:
        return rows[0]["d"]
    return today_date()


def _is_bear_market(date):
    """上证收盘 < 240日均线 → 熊市"""
    try:
        import pandas as pd
        from core.data.fetcher import DataFetcher
        df = DataFetcher().get_index_daily("上证指数", days=365)
        if df is None or df.empty:
            logger.warning("[一致性风控] 无法获取上证指数数据，默认按熊市处理 (与其他异常路径保持一致)")
            return True
        df = df.sort_values("date")
        df = df[df["date"] <= pd.Timestamp(date)]
        if len(df) < 240:
            logger.warning(f"[一致性风控] 上证数据不足240天({len(df)}天)，默认按熊市处理")
            return True
        close = df["close"].astype(float)
        ma240 = close.tail(240).mean()
        return float(close.iloc[-1]) < float(ma240)
    except Exception as e:
        logger.warning(f"[一致性风控] 判断牛熊失败: {e}，默认按熊市处理")
        return True


def _get_merged_data(start_date, end_date):
    """获取 daily_quotes (pct_chg+close) + daily_basic (total_mv)，宽松匹配

    daily_basic 可能比 daily_quotes 滞后1-2天，用 ±15天窗口就近匹配。
    pct_chg 可能为 NULL (腾讯源)，用 close/prev_close 自动补算。
    """
    from core.data import Database
    if not Database.is_available():
        return None

    # daily_basic 往前多取15天，覆盖滞后
    basic_start = (dt.date.fromisoformat(str(start_date)) - dt.timedelta(days=15)).isoformat()
    basic_rows = Database.fetchall(
        "SELECT code, trade_date, total_mv FROM daily_basic "
        "WHERE trade_date BETWEEN %s AND %s AND total_mv IS NOT NULL AND total_mv > 0 "
        "ORDER BY code, trade_date",
        (basic_start, str(end_date))
    )
    quote_rows = Database.fetchall(
        "SELECT code, trade_date, pct_chg, close FROM daily_quotes "
        "WHERE trade_date BETWEEN %s AND %s "
        "ORDER BY code, trade_date",
        (str(start_date), str(end_date))
    )
    if not basic_rows or not quote_rows:
        return None

    import pandas as pd
    basic = pd.DataFrame(basic_rows)
    quote = pd.DataFrame(quote_rows)
    basic["trade_date"] = pd.to_datetime(basic["trade_date"])
    quote["trade_date"] = pd.to_datetime(quote["trade_date"])

    # 补算 pct_chg: 当 NULL 时用 close / prev_close
    quote["close"] = pd.to_numeric(quote["close"], errors="coerce")
    quote["pct_chg"] = pd.to_numeric(quote["pct_chg"], errors="coerce")
    null_mask = quote["pct_chg"].isna()
    if null_mask.any():
        quote["prev_close"] = quote.groupby("code")["close"].shift(1)
        quote.loc[null_mask, "pct_chg"] = (
            (quote.loc[null_mask, "close"] / quote.loc[null_mask, "prev_close"] - 1) * 100
        )
        quote.drop(columns=["prev_close"], inplace=True)
    quote = quote.dropna(subset=["pct_chg"])

    # merge_ordered: 对每个 (code, trade_date) 找最近的历史 total_mv
    merged = pd.merge_asof(
        quote.sort_values("trade_date"),
        basic.sort_values("trade_date"),
        by="code",
        on="trade_date",
        direction="backward",
        tolerance=pd.Timedelta(days=15),
    )
    merged["total_mv"] = pd.to_numeric(merged["total_mv"], errors="coerce")
    return merged.dropna(subset=["total_mv"])


def _compute_consistency(pct_values):
    """单日一致性 = 落在(中位数±1σ)内的比例"""
    if len(pct_values) < _MIN_STOCKS:
        return None, None
    arr = np.array(pct_values, dtype=float)
    med = np.median(arr)
    std = np.std(arr)
    if std == 0:
        return float(med), 1.0
    in_range = ((arr >= med - std) & (arr <= med + std)).sum()
    return float(med), float(in_range / len(arr))


def _get_consistency_series(end_date, lookback=_LOOKBACK):
    """计算过去 lookback 个交易日的一致性序列 [(date_str, median, consistency), ...]"""
    start = end_date - dt.timedelta(days=lookback * 2)
    df = _get_merged_data(start.isoformat(), end_date.isoformat())
    if df is None or df.empty:
        return []

    dates = sorted(df["trade_date"].unique())
    if len(dates) < _MIN_HISTORY:
        return []

    results = []
    for d in dates[-lookback:]:
        day_data = df[df["trade_date"] == d]
        if len(day_data) < 2000:
            continue
        # 市值最小的 5%
        n = max(1, int(len(day_data) * 0.05))
        bottom = day_data.nsmallest(n, "total_mv")
        med, cons = _compute_consistency(bottom["pct_chg"].values)
        if med is not None:
            results.append((d.strftime("%Y-%m-%d"), med, cons))

    return results


def check_panic(date=None):
    """检查是否需要因踩踏恐慌而清仓

    Returns:
        True  → 触发清仓信号（仓位系数应设为 0）
        False → 正常持仓
    """
    if date is None:
        date = _latest_trade_date()
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)

    # 1. 仅熊市启用
    if not _is_bear_market(date):
        state = _load_state()
        if state.get("panic"):
            state["panic"] = False
            state["last_updated"] = date.isoformat()
            _save_state(state)
            logger.info("[一致性风控] 牛市，关闭风控")
        return False

    # 2. 计算一致性序列
    series = _get_consistency_series(date)
    if len(series) < _MIN_HISTORY:
        logger.info(f"[一致性风控] 历史数据不足({len(series)}天)，跳过")
        return _load_state().get("panic", False)

    cons_list = [s[2] for s in series]
    cons_mean = np.mean(cons_list)
    cons_std = np.std(cons_list)
    cons_upper = cons_mean + cons_std

    today_date, today_med, today_cons = series[-1]

    # 3. 读状态 + 判断
    state = _load_state()
    was_panic = state.get("panic", False)

    if today_med < -2 and today_cons >= cons_upper:
        logger.info(
            f"[一致性风控] 清仓 — {today_date} 微盘涨跌幅中位数={today_med:.2f}%, "
            f"一致性={today_cons:.3f} >= 上轨={cons_upper:.3f}"
        )
        state["panic"] = True
        state["last_updated"] = date.isoformat()
        _save_state(state)
        return True

    if today_med > 2 and today_cons >= cons_mean:
        logger.info(
            f"[一致性风控] 恢复 — {today_date} 微盘涨跌幅中位数={today_med:.2f}%, "
            f"一致性={today_cons:.3f} >= 均值={cons_mean:.3f}"
        )
        state["panic"] = False
        state["last_updated"] = date.isoformat()
        _save_state(state)
        return False

    if was_panic:
        logger.info(
            f"[一致性风控] 维持清仓 — {today_date} "
            f"med={today_med:.2f}% cons={today_cons:.3f}(upper={cons_upper:.3f})"
        )

    return was_panic


def reset_state():
    """重置风控状态（测试/手动恢复）"""
    from core.shared.persistence import today_iso
    _save_state({"panic": False, "last_updated": today_iso()})
    logger.info("[一致性风控] 状态已重置")
