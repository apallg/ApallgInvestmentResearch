import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
TREND_RUN = "7e5e67ba9b714c76b0808a2626d0f7d1"
REBOUND_RUN = "1882d6477fed4ec8950b8ba166c5d16d"
EXPERIMENT = "656749665990832865"
START = pd.Timestamp("2025-07-01")
END = pd.Timestamp("2026-08-31")
TOP_N = 5
BUY_COST = 0.0004
SELL_COST = 0.0009
SEED = 20260916
MIN_TRAIN = 26
MIN_LEAF = 8
KAPPA = 0.001
OUT = ROOT / "data" / "experiments" / "contextual_top5_weekly_constrained_20260916.json"
REPORT = ROOT / "docs" / "contextual_top5_weekly_constrained_20260916.md"
CONTEXT_PATH = ROOT / "data" / "experiments" / "pit_market_context_extended_20260916.parquet"
EXTRA_REBOUND = ROOT / "data" / "experiments" / "rebound_q3_pred_20260916.pkl"
REGIME_PREDICTION_PATH = ROOT / "data" / "experiments" / "regime_predictions_2025_07_2026_08_20260916.json"

GROUPS = {
    "G0_volatility": ["vol15", "dvol5"],
    "G1_market_structure": ["vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3"],
    "G2_strength_continuation": ["vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3", "continuation3"],
    "G3_plus_disagreement": ["vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3", "continuation3", "rank_disagreement", "top5_overlap"],
}


def _find_artifact(run, name):
    paths = list(ROOT.glob(f"mlruns/*/{run}/artifacts/{name}"))
    if not paths:
        raise FileNotFoundError(f"missing artifact {run}/{name}")
    return paths[0]


def _read_metadata(run):
    path = _find_artifact(run, "factor_keys.json")
    return json.loads(pickle.loads(path.read_bytes()))


def _load_prediction(run):
    obj = pickle.load(open(_find_artifact(run, "pred.pkl"), "rb"))
    if run == REBOUND_RUN and EXTRA_REBOUND.exists():
        extra = pickle.loads(EXTRA_REBOUND.read_bytes())
        obj = pd.concat([obj, extra])
        obj = obj[~obj.index.duplicated(keep="first")]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    obj = obj.copy()
    obj.index = pd.MultiIndex.from_tuples(obj.index, names=["date", "code"])
    obj.index = pd.MultiIndex.from_arrays([pd.to_datetime(obj.index.get_level_values(0)), obj.index.get_level_values(1)])
    return obj.rename("score").astype(float)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(values, block_size, draws=5000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    blocks = [values[i:i + block_size] for i in range(0, len(values), block_size)]
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])[:len(values)].mean() for _ in range(draws)]
    q = np.quantile(samples, [0.025, 0.975])
    return {"n": int(len(values)), "mean": float(values.mean()), "lower": float(q[0]), "upper": float(q[1])}


def _weekly_dates(dates):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame.date.dt.isocalendar().year.astype(str) + "-W" + frame.date.dt.isocalendar().week.astype(str).str.zfill(2)
    return frame.groupby("week", sort=True).date.max().tolist()


def _rank_disagreement(a, b):
    common = a.index.intersection(b.index)
    if len(common) < 20:
        return np.nan
    return float(1.0 - a.loc[common].rank().corr(b.loc[common].rank(), method="pearson"))


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


def _limit_pct(code, name=""):
    pure = _norm_code(code)
    if "ST" in str(name).upper():
        return 0.05
    if pure.startswith(("300", "301", "688")):
        return 0.20
    if pure.startswith(("8", "4")):
        return 0.30
    return 0.10


def _latest_value(frame, date, code):
    if frame is None:
        return np.nan
    if code not in frame.columns:
        return np.nan
    prior = frame.loc[frame.index <= pd.Timestamp(date), code].dropna()
    return float(prior.iloc[-1]) if len(prior) else np.nan


def _exact_value(frame, date, code):
    if frame is None or code not in frame.columns or pd.Timestamp(date) not in frame.index:
        return np.nan
    value = frame.loc[pd.Timestamp(date), code]
    return float(value) if np.isfinite(value) else np.nan


def _opening_limit(open_df, close_df, date, code, direction, industry_map):
    px = _exact_value(open_df, date, code)
    prev = _latest_value(close_df, pd.Timestamp(date) - pd.Timedelta(days=1), code)
    if not np.isfinite(px) or px <= 0 or not np.isfinite(prev) or prev <= 0:
        return False, px
    name = ""
    limit = _limit_pct(code, name)
    change = px / prev - 1.0
    if direction == "up":
        return bool(change >= limit * 0.98), px
    return bool(change <= -limit * 0.98), px


def _load_rule_regime(decision_dates):
    from core.data.connection import Database
    rows = Database.fetchall(
        "SELECT trade_date, regime FROM market_regime_history WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date",
        (str((min(decision_dates) - pd.Timedelta(days=14)).date()), str(max(decision_dates).date())),
    )
    history = [(pd.Timestamp(r["trade_date"]), str(r["regime"])) for r in rows]
    mapping = {}
    fallback = 0
    for date in decision_dates:
        prior = [item for item in history if item[0] <= date]
        regime = prior[-1][1] if prior else ""
        if regime in ("trending_down", "ranging"):
            mapping[date] = "rebound"
        elif regime == "trending_up":
            mapping[date] = "trend"
        else:
            mapping[date] = "trend"
            fallback += 1
    return mapping, {"rows": len(rows), "fallback_decisions": fallback, "regimes": sorted(set(r for _, r in history))}


def _load_supervised_regime(decision_dates, merged):
    payload = json.loads(REGIME_PREDICTION_PATH.read_text(encoding="utf-8"))
    by_date = {pd.Timestamp(row["date"]): row["state"] for row in payload.get("rows", [])}
    direct = {
        "volatile_up": "trend", "steady_up": "trend",
        "volatile_down": "rebound", "steady_flat": "rebound",
    }
    mapping = {}
    fallback = 0
    for i, date in enumerate(decision_dates):
        state = by_date.get(pd.Timestamp(date))
        if state in direct:
            mapping[date] = direct[state]
            continue
        prior = merged.iloc[:i]
        prior = prior[prior["label_matured_at"] <= pd.Timestamp(date)]["advantage"].dropna()
        mapping[date] = "rebound" if len(prior) >= MIN_TRAIN and float(prior.mean()) > 0 else "trend"
        fallback += 1
    return mapping, {"rows": len(payload.get("rows", [])), "fallback_decisions": fallback, "model": payload.get("model"), "retrospective": True}


def _expert_weekly(pred, opens, decision_dates, next_exec):
    rows = []
    baskets = {}
    for date, execution, following in zip(decision_dates, next_exec, next_exec[1:] + [None]):
        if following is None:
            continue
        if date not in pred.index.get_level_values("date") or execution not in opens.index or following not in opens.index:
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
        gross = float((exit_[valid] / entry[valid] - 1.0).mean())
        net = float((1.0 + gross) * (1.0 - BUY_COST) * (1.0 - SELL_COST) - 1.0)
        baskets[str(date.date())] = codes
        rows.append({"date": date, "execution": execution, "following_execution": following, "gross": gross, "net": net, "n": int(valid.sum())})
    return pd.DataFrame(rows), baskets


def _account(routes, baskets, opens, executions, buy_cost=BUY_COST, sell_cost=SELL_COST):
    nav = 1.0
    previous_target = []
    daily = []
    for i, date in enumerate(routes.index):
        execution = pd.Timestamp(routes.loc[date, "execution"])
        following = pd.Timestamp(routes.loc[date, "following_execution"])
        if execution not in opens.index or following not in opens.index:
            continue
        target = baskets.get(str(date.date()), [])
        entry = opens.loc[execution]
        exit_ = opens.loc[following]
        target = [c for c in target if pd.notna(entry.get(c, np.nan)) and pd.notna(exit_.get(c, np.nan)) and float(entry[c]) > 0]
        if not target:
            daily.append({"date": date, "execution": execution, "nav": nav, "turnover": 0.0, "trade_cost": 0.0})
            continue
        previous = set(previous_target)
        current = set(target)
        buy_weight = sum(1.0 / len(current) for code in current - previous)
        sell_weight = sum(1.0 / len(previous) for code in previous - current) if previous else 0.0
        trade_cost = buy_weight * buy_cost + sell_weight * sell_cost
        gross = float((exit_[target] / entry[target] - 1.0).mean())
        nav = nav * (1.0 - trade_cost) * (1.0 + gross)
        daily.append({"date": date, "execution": execution, "following_execution": following, "nav": nav, "turnover": buy_weight + sell_weight, "trade_cost": trade_cost, "gross": gross})
        previous_target = target
    frame = pd.DataFrame(daily)
    if frame.empty:
        return {"n": 0, "final_nav": None, "return": None, "max_drawdown": None, "mean_turnover": None, "daily": []}
    nav = frame.nav.to_numpy(dtype=float)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / peak - 1.0))
    return {"n": int(len(frame)), "final_nav": float(nav[-1]), "return": float(nav[-1] - 1.0), "max_drawdown": mdd, "mean_turnover": float(frame.turnover.mean()), "daily": frame.to_dict(orient="records")}


def _account_constrained(routes, baskets, trade_data, industry_map, buy_cost=BUY_COST, sell_cost=SELL_COST):
    open_df, close_df = trade_data["$open"], trade_data["$close"]
    initial_capital = 1000000.0
    cash = initial_capital
    holdings = {}
    buy_dates = {}
    daily = []
    blocked_up = 0
    blocked_down = 0
    suspended = 0
    rounded = 0
    max_buy_turnover = 0.0
    max_sell_turnover = 0.0
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
            blocked, _ = _opening_limit(open_df, close_df, execution, code, "down", industry_map)
            if blocked:
                blocked_down += 1
                continue
            sell_orders.append((code, min(excess, current_value)))
        total_sell = sum(amount for _, amount in sell_orders)
        sell_scale = min(1.0, 0.30 * pre_value / total_sell) if total_sell > 0 else 0.0
        turnover = 0.0
        buy_turnover = 0.0
        sell_turnover = 0.0
        trade_cost = 0.0
        for code, amount in sell_orders:
            px = prices[code]
            shares = int(np.floor((amount * sell_scale) / (px * 100.0)) * 100)
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
            if holdings[code] <= 0:
                holdings.pop(code, None)
                buy_dates.pop(code, None)
        buy_orders = []
        post_value = cash + sum(qty * prices.get(code, _latest_value(close_df, execution, code)) for code, qty in holdings.items())
        for code, target_value in target_values.items():
            px = prices.get(code, np.nan)
            if not np.isfinite(px) or px <= 0:
                suspended += 1
                continue
            current_value = holdings.get(code, 0) * px
            deficit = max(0.0, target_value - current_value)
            if deficit <= 0:
                continue
            blocked, _ = _opening_limit(open_df, close_df, execution, code, "up", industry_map)
            if blocked:
                blocked_up += 1
                continue
            buy_orders.append((code, deficit))
        total_buy = sum(amount for _, amount in buy_orders)
        buy_scale = min(1.0, 0.30 * pre_value / total_buy) if total_buy > 0 else 0.0
        for code, amount in buy_orders:
            px = prices[code]
            affordable = cash / (1.0 + buy_cost)
            amount = min(amount * buy_scale, affordable)
            shares = int(np.floor(amount / (px * 100.0)) * 100)
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
        daily.append({"date": date, "execution": execution, "following_execution": following, "nav": nav, "turnover": turnover, "buy_turnover": buy_turnover, "sell_turnover": sell_turnover, "trade_cost": trade_cost, "exposure": exposure, "blocked_limit_up": blocked_up, "blocked_limit_down": blocked_down, "suspended": suspended, "lot_rounding": rounded})
    frame = pd.DataFrame(daily)
    if frame.empty:
        return {"n": 0, "final_nav": None, "return": None, "max_drawdown": None, "mean_turnover": None, "mean_exposure": None, "max_buy_turnover": 0.0, "max_sell_turnover": 0.0, "blocked_limit_up": 0, "blocked_limit_down": 0, "suspended_trades": 0, "lot_rounding": 0, "daily": []}
    nav = frame.nav.to_numpy(dtype=float)
    peak = np.maximum.accumulate(nav)
    return {"n": int(len(frame)), "final_nav": float(nav[-1]), "return": float(nav[-1] - 1.0), "max_drawdown": float(np.min(nav / peak - 1.0)), "mean_turnover": float(frame.turnover.mean()), "mean_exposure": float(frame.exposure.mean()), "max_buy_turnover": float(max_buy_turnover), "max_sell_turnover": float(max_sell_turnover), "blocked_limit_up": int(blocked_up), "blocked_limit_down": int(blocked_down), "suspended_trades": int(suspended), "lot_rounding": int(rounded), "daily": frame.to_dict(orient="records")}


def main():
    trend = _load_prediction(TREND_RUN)
    rebound = _load_prediction(REBOUND_RUN)
    common_dates = sorted(set(trend.index.get_level_values("date")) & set(rebound.index.get_level_values("date")))
    common_dates = [d for d in common_dates if START <= d <= END]
    weekly = _weekly_dates(common_dates)
    executions = [d for d in common_dates if d > weekly[0]]
    execution_map = {d: common_dates[common_dates.index(d) + 1] for d in weekly if d in common_dates and common_dates.index(d) + 1 < len(common_dates)}
    decision_dates = [d for d in weekly if d in execution_map]
    next_exec = [execution_map[d] for d in decision_dates]
    all_codes = set(trend.index.get_level_values("code")) & set(rebound.index.get_level_values("code"))
    opens = _load_open(all_codes, next_exec + next_exec[1:])
    trade_data = _load_trade_data(all_codes, next_exec + next_exec[1:])
    industry_map = _load_industry_map()
    t_week, t_baskets = _expert_weekly(trend, opens, decision_dates, next_exec)
    r_week, r_baskets = _expert_weekly(rebound, opens, decision_dates, next_exec)
    merged = t_week.merge(r_week, on=["date", "execution", "following_execution"], suffixes=("_trend", "_rebound"))
    merged["label_matured_at"] = pd.to_datetime(merged["following_execution"])
    merged["advantage"] = merged.net_rebound - merged.net_trend
    merged["target"] = (merged.advantage > 0).astype(int)
    context = pd.read_parquet(CONTEXT_PATH)
    context["date"] = pd.to_datetime(context["date"])
    context["dvol5"] = context.vol15 - context.vol15.shift(5)
    context = context.set_index("date")
    feature_rows = []
    for date in merged.date:
        base = context.loc[date].to_dict() if date in context.index else {}
        ta = trend.xs(date, level="date") if date in trend.index.get_level_values("date") else pd.Series(dtype=float)
        rb = rebound.xs(date, level="date") if date in rebound.index.get_level_values("date") else pd.Series(dtype=float)
        overlap = len(set(t_baskets.get(str(date.date()), [])) & set(r_baskets.get(str(date.date()), []))) / TOP_N
        base.update({"date": date, "rank_disagreement": _rank_disagreement(ta, rb), "top5_overlap": overlap})
        feature_rows.append(base)
    feature = pd.DataFrame(feature_rows).set_index("date")
    merged = merged.set_index("date").join(feature, how="left").reset_index()
    routes = {"fixed_trend": pd.Series("trend", index=merged.index), "fixed_rebound": pd.Series("rebound", index=merged.index)}
    rule_map, rule_meta = _load_rule_regime(list(merged.date))
    routes["B6_rule_history"] = pd.Series([rule_map.get(d, "trend") for d in merged.date], index=merged.index)
    supervised_map, supervised_meta = _load_supervised_regime(list(merged.date), merged)
    routes["B6_supervised_retrospective"] = pd.Series([supervised_map.get(d, "trend") for d in merged.date], index=merged.index)
    test_start = MIN_TRAIN
    for name, columns in GROUPS.items():
        selected = []
        for i in range(test_start, len(merged)):
            row = merged.iloc[i]
            prior = merged.iloc[:i]
            prior = prior[prior["label_matured_at"] <= pd.Timestamp(row["date"])]
            train = prior.dropna(subset=columns + ["advantage"])
            if len(train) < MIN_TRAIN or row[columns].isna().any():
                selected.append("trend")
                continue
            model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=MIN_LEAF, random_state=SEED)
            model.fit(train[columns].to_numpy(), train.advantage.to_numpy())
            selected.append("rebound" if float(model.predict(row[columns].to_numpy().reshape(1, -1))[0]) > KAPPA else "trend")
        routes[name] = pd.Series(["trend"] * test_start + selected, index=merged.index)
    vol_route = ["trend"] * test_start
    for i in range(test_start, len(merged)):
        prior = merged.iloc[:i].vol15.dropna()
        vol_route.append("rebound" if len(prior) >= MIN_TRAIN and merged.iloc[i].vol15 > prior.quantile(0.7) else "trend")
    routes["Bvol_quantile"] = pd.Series(vol_route, index=merged.index)
    for name, route in routes.items():
        merged[f"route_{name}"] = route.to_numpy()
        merged[f"selected_net_{name}"] = [float(merged.iloc[i].net_rebound if route.iloc[i] == "rebound" else merged.iloc[i].net_trend) for i in range(len(merged))]
    outcomes = {}
    accounts = {}
    stress_accounts = {}
    for name, route in routes.items():
        chosen = []
        values = []
        if name == "fixed_trend":
            chosen = [t_baskets.get(str(d.date()), []) for d in merged.date]
        elif name == "fixed_rebound":
            chosen = [r_baskets.get(str(d.date()), []) for d in merged.date]
        else:
            chosen = [((r_baskets if route.iloc[i] == "rebound" else t_baskets).get(str(d.date()), [])) for i, d in enumerate(merged.date)]
        for i, route_name in enumerate(route):
            values.append(float(merged.iloc[i].net_rebound if route_name == "rebound" else merged.iloc[i].net_trend))
        values = np.asarray(values, dtype=float)
        model_values = values[test_start:]
        model_target = merged.target.to_numpy(dtype=bool)[test_start:]
        outcomes[name] = {
            "n": int(len(values)),
            "warmup_n": int(min(test_start, len(values))),
            "model_oos_n": int(len(model_values)),
            "route_rebound_count": int((route == "rebound").sum()),
            "model_oos_rebound_count": int((route.iloc[test_start:] == "rebound").sum()),
            "mean_net": float(values.mean()),
            "sum_net": float(values.sum()),
            "ci_4week": _bootstrap(values, 4),
            "route_accuracy": float(((route == "rebound") == merged.target.astype(bool)).mean()),
            "model_oos_mean_net": float(model_values.mean()) if len(model_values) else None,
            "model_oos_sum_net": float(model_values.sum()) if len(model_values) else None,
            "model_oos_ci_4week": _bootstrap(model_values, 4),
            "model_oos_route_accuracy": float(((route.iloc[test_start:] == "rebound") == model_target).mean()) if len(model_values) else None,
        }
        baskets = {str(d.date()): chosen[i] for i, d in enumerate(merged.date)}
        route_index = merged.set_index("date")
        accounts[name] = _account_constrained(route_index, baskets, trade_data, industry_map)
        stress_accounts[name] = _account_constrained(route_index, baskets, trade_data, industry_map, BUY_COST * 2.0, SELL_COST * 2.0)
    for name, item in outcomes.items():
        item["account"] = {k: v for k, v in accounts[name].items() if k != "daily"}
        item["stress_account"] = {k: v for k, v in stress_accounts[name].items() if k != "daily"}
    for name, item in outcomes.items():
        item["account_relative_to_fixed_trend"] = float(item["account"]["final_nav"] / outcomes["fixed_trend"]["account"]["final_nav"] - 1.0)
        item["account_relative_to_g0"] = float(item["account"]["final_nav"] / outcomes["G0_volatility"]["account"]["final_nav"] - 1.0)
    account_increments = {}
    account_by_name = {name: pd.DataFrame(item["daily"]) for name, item in accounts.items()}
    for left, right in [("G0_volatility", "G1_market_structure"), ("G1_market_structure", "G2_strength_continuation"), ("G2_strength_continuation", "G3_plus_disagreement")]:
        left_frame = account_by_name[left].set_index("date")
        right_frame = account_by_name[right].set_index("date")
        common = left_frame.index.intersection(right_frame.index)
        left_ret = left_frame.nav.loc[common].pct_change(fill_method=None).dropna()
        right_ret = right_frame.nav.loc[common].pct_change(fill_method=None).dropna()
        delta = (right_ret - left_ret).dropna().to_numpy(dtype=float)
        account_increments[f"{right}_minus_{left}"] = {"mean": float(delta.mean()), "sum": float(delta.sum()), "ci_4week": _bootstrap(delta, 4), "ci_8week": _bootstrap(delta, 8), "n": int(len(delta)), "right_wins": int((delta > 0).sum()), "left_wins": int((delta < 0).sum())}
    increments = {}
    for left, right in [("G0_volatility", "G1_market_structure"), ("G1_market_structure", "G2_strength_continuation"), ("G2_strength_continuation", "G3_plus_disagreement")]:
        delta = merged[f"selected_net_{right}"] - merged[f"selected_net_{left}"]
        ci = _bootstrap(delta.to_numpy(), 4)
        increments[f"{right}_minus_{left}"] = {"mean": float(delta.mean()), "sum": float(delta.sum()), "ci_4week": ci, "ci_8week": _bootstrap(delta.to_numpy(), 8), "right_wins": int((delta > 0).sum()), "left_wins": int((delta < 0).sum())}
    source_files = [_find_artifact(TREND_RUN, "pred.pkl"), _find_artifact(REBOUND_RUN, "pred.pkl"), EXTRA_REBOUND, CONTEXT_PATH]
    payload = {
        "schema_version": 1,
        "status": "top5_weekly_expert_pair_constrained_diagnostic",
        "experiment": EXPERIMENT,
        "period": [str(START.date()), str(END.date())],
        "decision_dates": int(len(merged)),
        "candidate_week_dates": int(len(decision_dates)),
        "dropped_week_dates": [str(d.date()) for d in decision_dates if d not in set(merged.date)],
        "trend_run": TREND_RUN,
        "rebound_run": REBOUND_RUN,
        "rule_regime_source": rule_meta,
        "supervised_regime_source": supervised_meta,
        "trend_metadata": _read_metadata(TREND_RUN),
        "rebound_metadata": _read_metadata(REBOUND_RUN),
        "source_sha256": {str(p.relative_to(ROOT)): _hash(p) for p in source_files},
        "contract": {"top_n": TOP_N, "buy_cost": BUY_COST, "sell_cost": SELL_COST, "weekly_last_day_to_next_open": True, "model_label_horizon": 10, "target_exposure": 0.95, "max_one_way_turnover": 0.30, "max_industry_weight": 0.30, "lot_size": 100, "limit_up_down": True, "suspension_block": True},
        "router": {"min_train": MIN_TRAIN, "min_leaf": MIN_LEAF, "max_depth": 2, "kappa": KAPPA, "seed": SEED, "groups": GROUPS},
        "outcomes": outcomes,
        "increments": increments,
        "account_increments": account_increments,
        "weekly_trace": merged.to_dict(orient="records"),
        "limitations": [
            "两专家预测覆盖同一 2025-07 至 2026-08 测试期；反弹模型 2026Q3 预测由固定 MLflow 模型按原训练合同外推生成，但因子集合在更早任务中选择，不能称端到端因子选择 OOS。",
            "账户回放已接入共同股票池、开盘价、涨跌停近似判定、停牌缺失开盘阻断、100股整手、95%目标仓位、单票20%上限、行业30%上限和单边30%换手上限；未模拟券商最低佣金和逐笔成交队列，因此仍是约束诊断，不是生产验收。",
            "没有可核实的同期 B6 六状态连续账户，因此无法判定 H4 新路由是否超过当前六状态基准。",
            "B6_rule_history 仅使用 market_regime_history 中的旧 trending_up/trending_down/ranging 记录，按决策日前最后状态映射 T/R；它不是同期监督六状态/HMM 的复现，仅作历史规则参照。",
            "B6_supervised_retrospective 使用当前 data/timing_model/regime_h10.pkl 对历史日期回算，模型训练截止晚于部分测试日期，属于回顾性诊断，不是点时样本外 HMM/六状态基线。",
            "样本约一年、有效周数不足协议要求的五个季度和独立确认期；所有结果不能直接发布或改生产路由。",
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Top5 周频双专家路由约束账户回放（2026-09-16）", "", "状态：`top5_weekly_expert_pair_constrained_diagnostic`。账户按开盘成交与组合约束回放，简化账户结果保留在 extended JSON，尚非完整生产验收。", "", f"测试期：{START.date()} 至 {END.date()}；候选周数：{len(decision_dates)}；有效决策周数：{len(merged)}；丢弃周：{len(decision_dates) - len(merged)}；趋势 run：`{TREND_RUN}`；反弹 run：`{REBOUND_RUN}`。反弹 2026Q3 预测来自同一固定 MLflow 模型的原合同外推，文件哈希写入 JSON。", "", f"树路由采用 expanding walk-forward：前26个有效周为 warm-up，后{max(0, len(merged) - test_start)}个周才是树的样本外决策；固定基准和账户从全部{len(merged)}周开始，因此不能把全期均值当作树路由的独立样本外成绩。账户约束为目标仓位95%、单票20%、行业30%、单边换手30%、100股整手、涨跌停开盘阻断、无开盘视为停牌。", "", "## 结果", "", "| 策略 | 全期周数 | warm-up周数 | 树OOS周数 | 选反弹次数 | OOS路由准确率 | 全期净优势均值 | OOS净优势均值 | 约束账户终值 | 约束账户回撤 | 加倍成本终值 | 平均仓位 | 平均换手 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in outcomes.items():
        ci, acc = item["ci_4week"], item["account"]
        lines.append(f"| {name} | {item['n']} | {item['warmup_n']} | {item['model_oos_n']} | {item['route_rebound_count']} | {item['model_oos_route_accuracy']:.3f} | {item['mean_net']:.5f} | {item['model_oos_mean_net']:.5f} | {acc['final_nav']:.5f} | {acc['max_drawdown']:.5f} | {item['stress_account']['final_nav']:.5f} | {acc['mean_exposure']:.3f} | {acc['mean_turnover']:.3f} |")
    lines += ["", "## 逐组配对增量", "", "| 增量 | 平均周收益差 | 4周块95%区间 | 8周块95%区间 | 右组胜出周 | 左组胜出周 |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in increments.items():
        ci, ci8 = item["ci_4week"], item["ci_8week"]
        lines.append(f"| {name} | {item['mean']:.5f} | [{ci['lower']:.5f}, {ci['upper']:.5f}] | [{ci8['lower']:.5f}, {ci8['upper']:.5f}] | {item['right_wins']} | {item['left_wins']} |")
    lines += ["", "## 约束账户配对增量", "", "| 增量 | 平均周净值差 | 4周块95%区间 | 8周块95%区间 | 右组胜出周 | 左组胜出周 |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in account_increments.items():
        ci, ci8 = item["ci_4week"], item["ci_8week"]
        lines.append(f"| {name} | {item['mean']:.5f} | [{ci['lower']:.5f}, {ci['upper']:.5f}] | [{ci8['lower']:.5f}, {ci8['upper']:.5f}] | {item['right_wins']} | {item['left_wins']} |")
    lines += ["", "## 逐组判定", "", "- G0 是波动率上下文的 Top5 周频配对诊断；不能与项目含空仓的原波动率策略直接等同。", "- G1、G2、G3 逐组加入市场结构、强弱延续、模型分歧；本期点估计和区间只能说明该测试期是否有线索，不能代表未来。", "- 只有完整成交约束、同期 B6、至少五个开发季度及冻结后的独立确认期全部完成，才允许形成生产候选。", "", "## 数据边界", "", "- 因子集合和模型训练配方来自已有 MLflow 资产；因子筛选曾使用历史研究，早期回放不构成端到端因子 OOS。", "- 约束账户使用 Qlib 复权日线开盘/收盘，按缺失开盘阻断交易、100股整手、涨跌停幅度近似、行业30%和单边30%执行；行业映射来自当前缓存，非历史 PIT 快照，未模拟最低佣金和队列成交。", "- 丢弃周及其原因已写入 JSON；不能将有效周数当作完整自然周覆盖。", "- 本报告不修改现有模型、因子或生产配置。"]
    g0, g1, g2, g3 = (outcomes[k] for k in ("G0_volatility", "G1_market_structure", "G2_strength_continuation", "G3_plus_disagreement"))
    g1_vs_g0 = g1["account"]["final_nav"] / g0["account"]["final_nav"] - 1.0
    lines += ["", "## 当前确定性结论", "", f"- 在这个共同测试期的约束 Top5 周频账户中，G1 终值 {g1['account']['final_nav']:.4f}，G0 终值 {g0['account']['final_nav']:.4f}，相对差额 {g1_vs_g0:.2%}；加倍买卖费率后 G1 终值 {g1['stress_account']['final_nav']:.4f}。", f"- G1 树路由在 {g1['model_oos_n']} 个 OOS 周的净优势均值为 {g1['model_oos_mean_net']:.5f}，但样本仍短，不能替代跨季度确认。", f"- G2 和 G3 的账户终值分别为 {g2['account']['final_nav']:.4f} 和 {g3['account']['final_nav']:.4f}，本期没有证据支持继续加入强弱延续或模型分歧。", f"- B6_rule_history 作为历史规则参照的结果写入 JSON；它只含旧三类状态的按日映射，不能宣称是 HMM/监督六状态复现。最终问题 H4 仍未判定，不能宣称 G1 超过现有 HMM/六状态策略。", "", "结论状态：G1 进入下一阶段完整实验；G2/G3 暂不扩展；不修改生产路由。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: {x: v for x, v in item.items() if x != "account"} for k, item in outcomes.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
