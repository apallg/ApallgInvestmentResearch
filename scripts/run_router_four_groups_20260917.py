import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/experiments/contextual_top5_weekly_corrected_20260916.json"
OUT = ROOT / "data/experiments/router_four_groups_20260917.json"
TREND_RUN = "7e5e67ba9b714c76b0808a2626d0f7d1"
REBOUND_RUN = "1882d6477fed4ec8950b8ba166c5d16d"
EXTRA_REBOUND = ROOT / "data/experiments/rebound_q3_pred_20260916.pkl"
TOP_N = 5
BUY_COST = 0.0004
SELL_COST = 0.0009


def _find_artifact(run, name):
    paths = list(ROOT.glob(f"mlruns/*/{run}/artifacts/{name}"))
    if not paths:
        raise FileNotFoundError(f"missing artifact {run}/{name}")
    return paths[0]


def _load_prediction(run):
    obj = pickle.loads(_find_artifact(run, "pred.pkl").read_bytes())
    if run == REBOUND_RUN and EXTRA_REBOUND.exists():
        extra = pickle.loads(EXTRA_REBOUND.read_bytes())
        obj = pd.concat([obj, extra])
        obj = obj[~obj.index.duplicated(keep="first")]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    obj = obj.copy()
    obj.index = pd.MultiIndex.from_tuples(obj.index, names=["date", "code"])
    obj.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(obj.index.get_level_values(0)),
        obj.index.get_level_values(1),
    ])
    return obj.rename("score").astype(float)


def _weekly_dates(dates):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame.date.dt.isocalendar().year.astype(str) + "-W" + frame.date.dt.isocalendar().week.astype(str).str.zfill(2)
    return frame.groupby("week", sort=True).date.max().tolist()


def _load_open(codes, dates):
    sys.path.insert(0, str(ROOT))
    from core.adapters.qlib.config import init_qlib
    init_qlib()
    from qlib.data import D
    data = D.features(sorted(codes), ["$open"], start_time=str(min(dates).date()), end_time=str(max(dates).date()), freq="day")
    out = data.iloc[:, 0].unstack("instrument")
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _load_trade_data(codes, dates):
    sys.path.insert(0, str(ROOT))
    from core.adapters.qlib.config import init_qlib
    init_qlib()
    from qlib.data import D
    start = (min(dates) - pd.Timedelta(days=35)).date()
    end = max(dates).date()
    data = D.features(sorted(codes), ["$open", "$close", "$factor"], start_time=str(start), end_time=str(end), freq="day")
    frames = {}
    for field in ("$open", "$close", "$factor"):
        frame = data[field].unstack("instrument")
        frame.index = pd.to_datetime(frame.index)
        frames[field] = frame.sort_index()
    return frames


def _norm_code(code):
    value = str(code).upper().replace("SH", "").replace("SZ", "").replace("BJ", "")
    return value.replace(".", "")[-6:]


def _load_industry_map():
    try:
        sys.path.insert(0, str(ROOT))
        from core.data.fetcher import DataFetcher
        raw = DataFetcher().get_industry_map() or {}
        return {_norm_code(k): (v.get("industry") or v.get("name") or "UNKNOWN") if isinstance(v, dict) else str(v) for k, v in raw.items()}
    except Exception:
        return {}


def _exact_value(frame, date, code):
    if frame is None or code not in frame.columns or pd.Timestamp(date) not in frame.index:
        return np.nan
    value = frame.loc[pd.Timestamp(date), code]
    return float(value) if np.isfinite(value) else np.nan


def _latest_value(frame, date, code):
    if frame is None or code not in frame.columns:
        return np.nan
    prior = frame.loc[frame.index <= pd.Timestamp(date), code].dropna()
    return float(prior.iloc[-1]) if len(prior) else np.nan


def _limit_pct(code):
    pure = _norm_code(code)
    if pure.startswith(("300", "301", "688")):
        return 0.20
    if pure.startswith(("8", "4")):
        return 0.30
    return 0.10


def _opening_limit(open_df, close_df, date, code, direction):
    px = _exact_value(open_df, date, code)
    prev = _latest_value(close_df, pd.Timestamp(date) - pd.Timedelta(days=1), code)
    if not np.isfinite(px) or px <= 0 or not np.isfinite(prev) or prev <= 0:
        return False
    change = px / prev - 1.0
    limit = _limit_pct(code)
    return bool(change >= limit * 0.98) if direction == "up" else bool(change <= -limit * 0.98)


def _expert_weekly(pred, opens, decision_dates, next_exec):
    baskets = {}
    rows = []
    for date, execution, following in zip(decision_dates, next_exec, next_exec[1:] + [None]):
        if following is None or date not in pred.index.get_level_values("date") or execution not in opens.index or following not in opens.index:
            continue
        scores = pred.xs(date, level="date").dropna().sort_values(ascending=False)
        scores = scores[scores.index.isin(opens.columns)]
        if len(scores) < TOP_N:
            continue
        codes = list(scores.head(TOP_N).index)
        entry = opens.loc[execution, codes]
        exit_ = opens.loc[following, codes]
        valid = entry.notna() & exit_.notna() & (entry > 0)
        if valid.sum() < TOP_N:
            continue
        baskets[str(date.date())] = codes
        rows.append({"date": date, "execution": execution, "following_execution": following})
    return pd.DataFrame(rows), baskets


def _fusion_baskets(trend, rebound, opens, decision_dates, next_exec):
    baskets = {}
    valid_dates = []
    for date, execution, following in zip(decision_dates, next_exec, next_exec[1:] + [None]):
        if following is None or date not in trend.index.get_level_values("date") or date not in rebound.index.get_level_values("date"):
            continue
        if execution not in opens.index or following not in opens.index:
            continue
        ta = trend.xs(date, level="date").dropna()
        rb = rebound.xs(date, level="date").dropna()
        common = ta.index.intersection(rb.index).intersection(opens.columns)
        if len(common) < TOP_N:
            continue
        combined = ta.loc[common].rank(method="average", pct=True) + rb.loc[common].rank(method="average", pct=True)
        codes = list(combined.sort_values(ascending=False).head(TOP_N).index)
        entry = opens.loc[execution, codes]
        exit_ = opens.loc[following, codes]
        if int((entry.notna() & exit_.notna() & (entry > 0)).sum()) < TOP_N:
            continue
        baskets[str(date.date())] = codes
        valid_dates.append(date)
    return baskets, valid_dates


def _account_constrained(routes, baskets, trade_data, industry_map, buy_cost=BUY_COST, sell_cost=SELL_COST):
    open_df, close_df = trade_data["$open"], trade_data["$close"]
    initial_capital = 1_000_000.0
    cash = initial_capital
    holdings = {}
    buy_dates = {}
    daily = []
    blocked_up = blocked_down = suspended = rounded = 0
    max_buy_turnover = max_sell_turnover = 0.0
    for index, row in routes.iterrows():
        date = pd.Timestamp(index)
        execution = pd.Timestamp(row["execution"])
        following = pd.Timestamp(row["following_execution"])
        target = list(baskets.get(str(date.date()), []))
        target = [c for c in target if np.isfinite(_exact_value(open_df, execution, c)) and _exact_value(open_df, execution, c) > 0]
        prices = {code: _exact_value(open_df, execution, code) for code in set(target) | set(holdings)}
        pre_value = cash + sum(qty * (prices.get(code) if np.isfinite(prices.get(code, np.nan)) else _latest_value(close_df, execution, code)) for code, qty in holdings.items())
        if not np.isfinite(pre_value) or pre_value <= 0:
            pre_value = cash
        target_weights = {}
        if target:
            base = 0.95 / len(target)
            by_industry = {}
            for code in target:
                key = industry_map.get(_norm_code(code), "UNKNOWN")
                by_industry[key] = by_industry.get(key, 0) + 1
            for code in target:
                key = industry_map.get(_norm_code(code), "UNKNOWN")
                target_weights[code] = min(base, 0.30 / by_industry[key])
        target_values = {code: weight * pre_value for code, weight in target_weights.items()}
        sell_orders = []
        for code, qty in list(holdings.items()):
            px = prices.get(code, np.nan)
            if not np.isfinite(px) or px <= 0:
                suspended += 1
                continue
            current_value = qty * px
            excess = max(0.0, current_value - target_values.get(code, 0.0))
            if excess <= 0 or buy_dates.get(code) == execution:
                continue
            if _opening_limit(open_df, close_df, execution, code, "down"):
                blocked_down += 1
                continue
            sell_orders.append((code, min(excess, current_value)))
        total_sell = sum(amount for _, amount in sell_orders)
        sell_scale = min(1.0, 0.30 * pre_value / total_sell) if total_sell > 0 else 0.0
        turnover = buy_turnover = sell_turnover = trade_cost = 0.0
        period_orders = []
        for code, amount in sell_orders:
            px = prices[code]
            shares = int(np.floor((amount * sell_scale) / (px * 100)) * 100)
            if shares <= 0:
                rounded += 1
                continue
            shares = min(shares, holdings[code])
            proceeds = shares * px
            holdings[code] -= shares
            cash += proceeds * (1.0 - sell_cost)
            turnover += proceeds / pre_value
            sell_turnover += proceeds / pre_value
            trade_cost += proceeds / pre_value * sell_cost
            period_orders.append({"side": "sell", "code": code, "shares": int(shares), "price": float(px), "gross": float(proceeds), "cost": float(proceeds * sell_cost)})
            if holdings[code] <= 0:
                holdings.pop(code, None)
                buy_dates.pop(code, None)
        buy_orders = []
        for code, target_value in target_values.items():
            px = prices.get(code, np.nan)
            if not np.isfinite(px) or px <= 0:
                suspended += 1
                continue
            deficit = max(0.0, target_value - holdings.get(code, 0) * px)
            if deficit > 0:
                if _opening_limit(open_df, close_df, execution, code, "up"):
                    blocked_up += 1
                    continue
                buy_orders.append((code, deficit))
        total_buy = sum(amount for _, amount in buy_orders)
        buy_scale = min(1.0, 0.30 * pre_value / total_buy) if total_buy > 0 else 0.0
        for code, amount in buy_orders:
            px = prices[code]
            amount = min(amount * buy_scale, cash / (1.0 + buy_cost))
            shares = int(np.floor(amount / (px * 100)) * 100)
            if shares <= 0:
                rounded += 1
                continue
            cost = shares * px
            total = cost * (1.0 + buy_cost)
            if total > cash:
                continue
            cash -= total
            holdings[code] = holdings.get(code, 0) + shares
            buy_dates[code] = execution
            turnover += cost / pre_value
            buy_turnover += cost / pre_value
            trade_cost += cost / pre_value * buy_cost
            period_orders.append({"side": "buy", "code": code, "shares": int(shares), "price": float(px), "gross": float(cost), "cost": float(cost * buy_cost)})
        valuation = cash
        for code, qty in holdings.items():
            px = _latest_value(open_df, following, code)
            if not np.isfinite(px) or px <= 0:
                px = _latest_value(close_df, following, code)
            if np.isfinite(px) and px > 0:
                valuation += qty * px
        nav = float(valuation / initial_capital)
        exposure = float((valuation - cash) / valuation) if valuation > 0 else 0.0
        max_buy_turnover = max(max_buy_turnover, buy_turnover)
        max_sell_turnover = max(max_sell_turnover, sell_turnover)
        daily.append({"date": date, "execution": execution, "following_execution": following, "nav": nav, "cash": float(cash), "holdings": {str(code): int(qty) for code, qty in holdings.items()}, "orders": period_orders, "turnover": turnover, "buy_turnover": buy_turnover, "sell_turnover": sell_turnover, "trade_cost": trade_cost, "exposure": exposure, "blocked_limit_up": blocked_up, "blocked_limit_down": blocked_down, "suspended": suspended, "lot_rounding": rounded})
    frame = pd.DataFrame(daily)
    nav = frame.nav.to_numpy(dtype=float)
    peak = np.maximum.accumulate(nav)
    return {"n": int(len(frame)), "final_nav": float(nav[-1]), "return": float(nav[-1] - 1.0), "max_drawdown": float(np.min(nav / peak - 1.0)), "mean_turnover": float(frame.turnover.mean()), "mean_exposure": float(frame.exposure.mean()), "max_buy_turnover": float(max_buy_turnover), "max_sell_turnover": float(max_sell_turnover), "blocked_limit_up": int(blocked_up), "blocked_limit_down": int(blocked_down), "suspended_trades": int(suspended), "lot_rounding": int(rounded), "daily": frame.to_dict(orient="records")}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    trend = _load_prediction(TREND_RUN)
    rebound = _load_prediction(REBOUND_RUN)
    common_dates = sorted(set(trend.index.get_level_values("date")) & set(rebound.index.get_level_values("date")))
    common_dates = [d for d in common_dates if pd.Timestamp("2025-07-01") <= d <= pd.Timestamp("2026-08-31")]
    weekly = _weekly_dates(common_dates)
    execution_map = {d: common_dates[common_dates.index(d) + 1] for d in weekly if d in common_dates and common_dates.index(d) + 1 < len(common_dates)}
    decision_dates = [d for d in weekly if d in execution_map]
    next_exec = [execution_map[d] for d in decision_dates]
    all_codes = set(trend.index.get_level_values("code")) & set(rebound.index.get_level_values("code"))
    dates_for_data = next_exec + next_exec[1:]
    opens = _load_open(all_codes, dates_for_data)
    trade_data = _load_trade_data(all_codes, dates_for_data)
    industry_map = _load_industry_map()
    t_week, t_baskets = _expert_weekly(trend, opens, decision_dates, next_exec)
    r_week, r_baskets = _expert_weekly(rebound, opens, decision_dates, next_exec)
    merged = t_week.merge(r_week, on=["date", "execution", "following_execution"], suffixes=("_t", "_r"))
    fusion_baskets, fusion_dates = _fusion_baskets(trend, rebound, opens, decision_dates, next_exec)
    trace = {str(pd.Timestamp(x["date"]).date()): x.get("route_G0_volatility", "trend") for x in source.get("weekly_trace", [])}
    x_baskets = {key: (r_baskets if trace.get(key) == "rebound" else t_baskets).get(key, []) for key in trace}
    route_index = merged.set_index("date")
    baskets = {"T_fixed_trend": t_baskets, "R_fixed_rebound": r_baskets, "F_static_fusion": fusion_baskets, "X_g0_volatility": x_baskets}
    accounts = {}
    stress = {}
    for name, basket in baskets.items():
        print(f"running {name}", flush=True)
        accounts[name] = _account_constrained(route_index, basket, trade_data, industry_map)
        stress[name] = _account_constrained(route_index, basket, trade_data, industry_map, BUY_COST * 2.0, SELL_COST * 2.0)
    summary = {}
    for name, account in accounts.items():
        summary[name] = {key: value for key, value in account.items() if key != "daily"}
        summary[name]["stress_final_nav"] = stress[name]["final_nav"]
    increments = {}
    frames = {name: pd.DataFrame(account["daily"]).set_index("date") for name, account in accounts.items()}
    for left, right in [("T_fixed_trend", "R_fixed_rebound"), ("T_fixed_trend", "F_static_fusion"), ("T_fixed_trend", "X_g0_volatility"), ("F_static_fusion", "X_g0_volatility")]:
        common = frames[left].index.intersection(frames[right].index)
        delta = (frames[right].nav.loc[common].pct_change(fill_method=None) - frames[left].nav.loc[common].pct_change(fill_method=None)).dropna()
        increments[f"{right}_minus_{left}"] = {"n": int(len(delta)), "mean_weekly_delta": float(delta.mean()), "sum_weekly_delta": float(delta.sum()), "right_wins": int((delta > 0).sum()), "left_wins": int((delta < 0).sum())}
    payload = {"status": "router_four_groups_same_contract_replay", "source": str(SOURCE.relative_to(ROOT)), "source_sha256": _sha(SOURCE), "contract": {"period": ["2025-07-01", "2026-08-31"], "top_n": TOP_N, "buy_cost": BUY_COST, "sell_cost": SELL_COST, "execution": "weekly_last_day_to_next_open", "target_exposure": 0.95, "max_one_way_turnover": 0.30, "max_industry_weight": 0.30, "lot_size": 100, "limit_up_down": True, "suspension_block": True}, "decision_weeks": int(len(route_index)), "fusion_weeks": int(len(fusion_dates)), "route_source": "G0 route from maturity-corrected source, same date index", "summary": summary, "increments": increments, "routes": [{"date": str(pd.Timestamp(date).date()), "execution": str(pd.Timestamp(row.execution).date()), "following_execution": str(pd.Timestamp(row.following_execution).date()), "route_X": trace.get(str(pd.Timestamp(date).date()), "trend")} for date, row in route_index.iterrows()], "baskets": baskets, "accounts": {name: {"daily": value["daily"]} for name, value in accounts.items()}}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"decision_weeks": len(route_index), "fusion_weeks": len(fusion_dates), "summary": summary, "increments": increments}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
