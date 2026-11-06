"""
统一评估指标 — 替代 overfitting_checker.py 和 config_tuner.py 中各自实现的 IC/夏普/回撤
"""
import numpy as np
from typing import List, Dict, Optional
from core.shared import get_freq_meta


def compute_trade_diagnostics(trades_log):
    buys = [t for t in trades_log if t.get("action") == "buy"]
    sells = [t for t in trades_log if t.get("action") == "sell"]
    stops = [t for t in sells if "止损" in t.get("reason", "")]
    profits = [t for t in sells if "止盈" in t.get("reason", "")]
    stop_returns = [float(t["realized_return"]) for t in stops
                    if t.get("realized_return") is not None]
    gap_slippage = [float(t["gap_slippage"]) for t in stops
                    if t.get("gap_slippage") is not None]
    return {
        "buy_count": len(buys),
        "sell_count": len(sells),
        "stop_loss_count": len(stops),
        "stop_loss_full_count": sum(1 for t in stops if "清仓" in t.get("reason", "")),
        "stop_loss_half_count": sum(1 for t in stops if "减半" in t.get("reason", "")),
        "atr_stop_count": sum(1 for t in stops if "ATR" in t.get("reason", "")),
        "take_profit_count": len(profits),
        "gap_through_count": sum(1 for v in gap_slippage if v < 0),
        "avg_stop_return": round(sum(stop_returns) / len(stop_returns), 6)
        if stop_returns else None,
        "avg_gap_slippage": round(sum(gap_slippage) / len(gap_slippage), 6)
        if gap_slippage else None,
        "traded_notional": round(sum(float(t.get("value", 0) or 0) for t in trades_log), 6),
        "transaction_cost": round(sum(float(t.get("cost", 0) or 0) for t in trades_log), 6),
    }
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


def rank_ic(scores: List[float], returns: List[float]) -> float:
    if len(scores) < 10 or len(returns) < 10:
        return 0.0
    try:
        if spearmanr:
            corr, _ = spearmanr(scores, returns)
            return float(corr if not np.isnan(corr) else 0.0)
        s_rank = _rank(scores)
        r_rank = _rank(returns)
        return float(np.corrcoef(s_rank, r_rank)[0, 1])
    except Exception:
        return 0.0


def _rank(arr):
    from collections import defaultdict
    n = len(arr)
    sorted_pairs = sorted(enumerate(arr), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and sorted_pairs[j][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j
    return ranks


def sharpe_ratio(returns: List[float], annualize: bool = False) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std < 1e-9:
        return 0.0
    sr = mean / std
    if annualize:
        sr *= np.sqrt(252)
    return round(float(sr), 4)


def max_drawdown(equity: List[float]) -> float:
    if len(equity) < 2:
        return 0.0
    arr = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak > 0, peak, 1.0)
    return round(float(abs(dd.min())), 4)


def win_rate(returns: List[float]) -> float:
    if not returns:
        return 0.0
    arr = np.array(returns, dtype=float)
    return round(float((arr > 0).mean()), 3)


def hit_rate(scores: List[float], returns: List[float], threshold: float = 0) -> float:
    if len(scores) < 5:
        return 0.0
    top_n = max(5, len(scores) // 5)
    top_idx = np.argsort(scores)[-top_n:]
    top_rets = np.array(returns)[top_idx]
    return round(float((top_rets > threshold).mean()), 3)


def aggregate_validation(date_results: List[Dict], periods=(5, 10, 20)) -> Dict:
    valid = [r for r in date_results if r.get(f"forward_{periods[0]}d") is not None]
    if not valid:
        return {"fitness": -999, "error": "无有效验证数据"}

    n = len(valid)
    result = {"n_dates": n}
    for p in periods:
        key = f"{p}d"
        fkey = f"forward_{key}"
        rets = [r.get(fkey, 0) or 0 for r in valid]
        result[f"mean_{key}"] = round(float(np.mean(rets)), 4)
        result[f"std_{key}"] = round(float(np.std(rets)), 4)
        result[f"sharpe_{key}"] = sharpe_ratio(rets)
        result[f"hit_rate_{key}"] = win_rate(rets)

    all_scores = []
    all_rets = []
    for r in valid:
        scores = r.get("top_scores", [])
        codes = r.get("top_codes", [])
        per_code = r.get("per_code_returns", {})
        for score, code in zip(scores, codes):
            code_ret = per_code.get(code, {}).get("20d")
            if code_ret is not None:
                all_scores.append(score)
                all_rets.append(code_ret)
    result["rank_ic"] = rank_ic(all_scores, all_rets)

    mean_20 = result.get("mean_20d", 0) or 0
    hit_20 = result.get("hit_rate_20d", 0) or 0
    std_20 = result.get("std_20d", 0.3) or 0.3
    ic = result.get("rank_ic", 0) or 0
    result["fitness"] = round(
        mean_20 * 0.4 + hit_20 * 0.3 + ic * 0.2 + (1.0 - min(std_20, 0.3) / 0.3) * 0.1, 4)
    return result


def compute_backtest_metrics(equity, benchmark_equity, equity_dates,
                              periods, trades_log, freq,
                              profit_factor_cap=None):
    """统一回测绩效计算。纯 L3 函数——不依赖 DB/文件/时间函数。

    合并自 DirectionBacktestRunner 和 ScreeningBacktestRunner 中 ~85% 重复的实现。

    Args:
        equity: 净值序列 list[float]
        benchmark_equity: 基准净值序列 list[float]
        equity_dates: 日期标签 list[str]
        periods: 每期详情 list[dict]，需含 regime/direction/period_return/timing_ratio
        trades_log: 完整交易日志 list[dict]
        freq: 频率 "day"/"2d"/"week"/"biweek"/"month"
        profit_factor_cap: 盈亏比上限，None 不截断。screening 端用 999.0

    Returns:
        {"summary": {...}, "equity_curve": [...], "benchmark_curve": [...],
         "regime_breakdown": {...}, "direction_distribution": {...}}
    """
    eq = np.array(equity)
    bm = np.array(benchmark_equity)
    n = len(eq)
    if n == 0:
        return {
            "summary": {"n_periods": 0, "n_trades": len(trades_log), "timing": None},
            "equity_curve": [], "benchmark_curve": [],
            "periods": periods, "trades": trades_log[-100:],
            "regime_breakdown": {}, "direction_distribution": {},
            "trade_diagnostics": compute_trade_diagnostics(trades_log),
        }
    weekly_ret = np.diff(eq) / eq[:-1] if n > 1 else np.array([0])
    weekly_bench = np.diff(bm) / bm[:-1] if n > 1 else np.array([0])
    weekly_excess = weekly_ret - weekly_bench

    ann_factor = get_freq_meta(freq).periods_per_year
    annual_return = float((eq[-1] ** (ann_factor / max(n - 1, 1)) - 1)) if n > 1 else 0
    annual_bench = float((bm[-1] ** (ann_factor / max(n - 1, 1)) - 1)) if n > 1 else 0
    annual_excess = annual_return - annual_bench

    ret_std = float(np.std(weekly_ret))
    sharpe = float(np.mean(weekly_ret) / max(ret_std, 1e-8) * np.sqrt(ann_factor))

    downside = weekly_ret[weekly_ret < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else ret_std
    sortino = float(np.mean(weekly_ret) / max(downside_std, 1e-8) * np.sqrt(ann_factor))

    max_dd = 0.0
    peak = eq[0]
    for v in eq:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = float(dd)

    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0
    excess_calmar = annual_excess / abs(max_dd) if abs(max_dd) > 1e-8 else 0
    wins = sum(1 for r in weekly_ret if r > 0)
    win_rate_val = wins / max(len(weekly_ret), 1)
    excess_wins = sum(1 for r in weekly_excess if r > 0)
    excess_win_rate_val = excess_wins / max(len(weekly_excess), 1)
    best_week = float(np.max(weekly_ret)) if len(weekly_ret) > 0 else 0
    worst_week = float(np.min(weekly_ret)) if len(weekly_ret) > 0 else 0
    avg_win = float(np.mean(weekly_ret[weekly_ret > 0])) if wins > 0 else 0
    avg_loss = float(abs(np.mean(weekly_ret[weekly_ret < 0]))) if wins < len(weekly_ret) else 0
    pf = avg_win / max(avg_loss, 1e-8) if avg_win > 0 else 0
    if profit_factor_cap is not None:
        pf = min(pf, profit_factor_cap)

    turnover_by_date = {}
    for trade in trades_log:
        date_key = str(trade.get("date") or "")[:10]
        action = trade.get("action")
        if not date_key or action not in ("buy", "sell"):
            continue
        side = "buy" if action == "buy" else "sell"
        turnover_by_date.setdefault(date_key, {"buy": 0.0, "sell": 0.0})[side] += abs(float(trade.get("value") or 0))
    one_way_turnovers = [
        (row["buy"] + row["sell"]) / 2 for row in turnover_by_date.values()
    ]

    regime_breakdown = {}
    direction_counts = {}
    for p in periods:
        r = p.get("regime", "unknown")
        if r not in regime_breakdown:
            regime_breakdown[r] = {"count": 0, "total_return": 0.0}
        regime_breakdown[r]["count"] += 1
        regime_breakdown[r]["total_return"] += p.get("period_return", 0)
        if p.get("period_return", 0) > 0:
            regime_breakdown[r].setdefault("wins", 0)
            regime_breakdown[r]["wins"] += 1
        d = p.get("direction", "")
        if d:
            direction_counts[d] = direction_counts.get(d, 0) + 1

    for r in regime_breakdown:
        c = regime_breakdown[r]["count"]
        if c > 0:
            regime_breakdown[r]["avg_return"] = round(regime_breakdown[r]["total_return"] / c, 6)
            w = regime_breakdown[r].get("wins", 0)
            regime_breakdown[r]["win_rate"] = round(w / c, 4)

    timing_summary = None
    if any(p.get("timing_ratio", 1.0) < 1.0 for p in periods):
        ratios = [p.get("timing_ratio", 1.0) for p in periods]
        timing_summary = {
            "avg": round(sum(ratios) / len(ratios), 3),
            "min": round(min(ratios), 3),
            "n_reduced": sum(1 for r in ratios if r < 1.0),
        }

    return {
        "summary": {
            "annual_return": round(annual_return, 4),
            "annual_benchmark": round(annual_bench, 4),
            "annual_excess": round(annual_excess, 4),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "calmar": round(calmar, 2),
            "excess_calmar": round(excess_calmar, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(float(eq[-1] - 1), 4),
            "benchmark_return": round(float(bm[-1] - 1), 4),
            "win_rate": round(win_rate_val, 4),
            "excess_win_rate": round(excess_win_rate_val, 4),
            "profit_factor": round(pf, 2),
            "best_week": round(best_week, 6),
            "worst_week": round(worst_week, 6),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "n_periods": len(periods),
            "n_trades": len(trades_log),
            "avg_one_way_turnover": round(float(np.mean(one_way_turnovers)), 4) if one_way_turnovers else 0.0,
            "max_one_way_turnover": round(float(np.max(one_way_turnovers)), 4) if one_way_turnovers else 0.0,
            "timing": timing_summary,
        },
        "equity_curve": [{"date": d, "value": round(float(v), 6)} for d, v in zip(equity_dates, eq)],
        "benchmark_curve": [{"date": d, "value": round(float(v), 6)} for d, v in zip(equity_dates, bm)],
        "periods": periods,
        "trades": trades_log[-100:],
        "trade_diagnostics": compute_trade_diagnostics(trades_log),
        "regime_breakdown": regime_breakdown,
        "direction_distribution": direction_counts,
    }
