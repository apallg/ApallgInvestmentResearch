import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/experiments/horizon_multiscale_research_20261104"


def historical_industry(index):
    path = OUTPUT / "stock_index_membership_snapshot.json"
    if not path.exists():
        path = ROOT / "data/experiments/monthly_group_research_v4_20261103/stock_index_membership_snapshot.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    codes = sorted({str(row["index_code"]) for row in rows if row.get("index_code")})
    ids = {code: number for number, code in enumerate(codes)}
    intervals = {}
    for row in rows:
        ts_code = str(row.get("ts_code") or "")
        if "." not in ts_code or not row.get("index_code"):
            continue
        symbol, exchange = ts_code.split(".", 1)
        prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(exchange.upper())
        if prefix:
            intervals.setdefault(prefix + symbol, []).append((pd.Timestamp(row["in_date"]),
                pd.Timestamp(row["out_date"]) if row.get("out_date") else pd.Timestamp.max,
                ids[str(row["index_code"])]))
    result = np.full(len(index), -1, dtype=np.int16)
    positions = pd.Series(np.arange(len(index)), index=index)
    for instrument, rows_for_stock in positions.groupby(level="instrument", sort=False):
        offsets = rows_for_stock.to_numpy()
        dates = pd.DatetimeIndex(rows_for_stock.index.get_level_values("datetime"))
        for start, end, group_id in intervals.get(instrument, []):
            mask = (dates >= start) & (dates <= end)
            if (result[offsets[mask]] >= 0).any():
                raise RuntimeError(f"历史行业归属冲突: {instrument}")
            result[offsets[mask]] = group_id
    return pd.Series(result, index=index, name="industry_id"), codes


def build_monthly_bars(quotes, calendar):
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    periods = calendar.to_period("M").unique()
    market_days = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).size().reindex(periods)
    output = []
    for instrument, part in quotes.groupby(level="instrument", sort=False):
        stock = part.droplevel("instrument").sort_index()
        stock_period = stock.index.to_period("M")
        bars = stock.groupby(stock_period).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), valid_days=("close", "count"))
        bars = bars.reindex(periods)
        previous_close = bars["close"].shift(1).where(bars["close"].shift(1) > 0)
        previous_volume = bars["volume"].shift(1).where(bars["volume"].shift(1) > 0)
        values = pd.DataFrame(index=periods)
        values["open_gap"] = np.log(bars["open"].div(previous_close))
        values["high_gap"] = np.log(bars["high"].div(previous_close))
        values["low_gap"] = np.log(bars["low"].div(previous_close))
        values["close_gap"] = np.log(bars["close"].div(previous_close))
        values["volume_change"] = np.log(bars["volume"].div(previous_volume))
        values["valid_day_ratio"] = bars["valid_days"].div(market_days)
        values["instrument"] = instrument
        output.append(values.reset_index(names="month").set_index(["month", "instrument"]))
    return pd.concat(output).sort_index().replace([np.inf, -np.inf], np.nan).astype({
        "open_gap": np.float32, "high_gap": np.float32, "low_gap": np.float32,
        "close_gap": np.float32, "volume_change": np.float32, "valid_day_ratio": np.float32})


def monthly_history_for_date(bars, instrument, as_of, calendar, months=12):
    as_of = pd.Timestamp(as_of)
    calendar = pd.DatetimeIndex(calendar)
    month_ends = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).last()
    complete = month_ends.loc[month_ends <= as_of]
    if complete.empty:
        periods = pd.PeriodIndex([], freq="M")
    else:
        last_period = complete.index[-1]
        periods = pd.period_range(last_period - (months - 1), last_period, freq="M")
    values = bars.xs(instrument, level="instrument").reindex(periods) if instrument in bars.index.get_level_values("instrument") else pd.DataFrame(index=periods, columns=bars.columns)
    values = values.reindex(columns=["open_gap", "high_gap", "low_gap", "close_gap", "volume_change", "valid_day_ratio"])
    missing = values.isna().astype(np.float32).rename(columns=lambda name: f"missing_{name}")
    values = values.fillna(0).astype(np.float32)
    return pd.concat([values, missing], axis=1).reindex(periods)


def build_groups(quotes, context, calendar, listing_dates=None):
    index = quotes.index
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    instruments = index.get_level_values("instrument")
    month_ends = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).last()
    last_dates = dates.to_period("M").map(month_ends)
    snapshots = index[dates == last_dates]
    groups = pd.DataFrame(-1, index=index, columns=["B3", "A2", "R4"], dtype=np.int16)
    continuous = pd.DataFrame(np.nan, index=index, columns=["r4_industry_strength60", "r4_stock_strength20"], dtype=np.float32)
    industry, industry_codes = historical_industry(index)
    close = quotes["close"].where(quotes["close"] > 0)
    stock_return = pd.Series(np.nan, index=index, dtype=np.float32)
    for instrument, part in close.groupby(level="instrument", sort=False):
        prices = part.droplevel("instrument").reindex(calendar)
        daily_return = prices.div(prices.shift(1)).sub(1).where(prices.notna() & prices.shift(1).notna())
        stock_return.loc[part.index] = daily_return.reindex(part.index.get_level_values("datetime")).to_numpy(dtype=np.float32)
    valid_industry = industry >= 0
    source = pd.DataFrame({"industry": industry, "ret": stock_return}, index=index)
    daily_industry = source.loc[valid_industry].groupby([pd.Grouper(level="datetime"), "industry"])["ret"].agg(["mean", "count"])
    daily_industry.loc[daily_industry["count"] < 20, "mean"] = np.nan
    industry_ret = daily_industry["mean"].rename("industry_ret")
    industry_loo = pd.Series(np.nan, index=index, dtype=np.float32)
    key = pd.MultiIndex.from_arrays([dates, industry])
    total = industry_ret.reindex(key).to_numpy(dtype=float)
    count = daily_industry["count"].reindex(key).to_numpy(dtype=float)
    own = stock_return.to_numpy(dtype=float)
    industry_loo.iloc[:] = np.where((count >= 20) & np.isfinite(own), (total * count - own) / (count - 1), np.nan).astype(np.float32)
    market_ret = stock_return.groupby(level="datetime").mean().reindex(calendar)
    industry_wide = daily_industry["mean"].unstack("industry").reindex(calendar)
    industry_market_log = np.log1p(industry_wide).subtract(np.log1p(market_ret), axis=0)
    r4_industry = industry_market_log.rolling(60, min_periods=60).sum().stack(future_stack=True)
    aligned_industry_strength = r4_industry.reindex(key)
    industry_strength = aligned_industry_strength.to_numpy(dtype=float)
    continuous["r4_industry_strength60"] = industry_strength.astype(np.float32)
    stock_strength = pd.Series(np.nan, index=index, dtype=np.float32)
    for instrument, part in close.groupby(level="instrument", sort=False):
        stock_dates = part.index.get_level_values("datetime")
        relative = pd.Series(np.nan, index=calendar, dtype=float)
        target = part.index
        relative.loc[stock_dates] = (np.log1p(stock_return.reindex(target).to_numpy(dtype=float))
                                     - np.log1p(industry_loo.reindex(target).to_numpy(dtype=float)))
        rolling_strength = relative.rolling(20, min_periods=20).sum()
        stock_strength.loc[target] = rolling_strength.reindex(stock_dates).to_numpy(dtype=np.float32)
    continuous["r4_stock_strength20"] = stock_strength
    raw_continuous = continuous.copy()
    context_board = context[["context_board_main", "context_board_star", "context_board_chinext", "context_board_beijing"]]
    b3 = pd.Series(-1, index=index, dtype=np.int16)
    board = context_board.to_numpy(dtype=float)
    known_board = np.isfinite(board).all(axis=1) & (np.isclose(board.sum(axis=1), 1.0))
    b3.iloc[np.flatnonzero(known_board)] = np.select(
        [board[known_board, 0] == 1, (board[known_board, 1] + board[known_board, 2]) == 1, board[known_board, 3] == 1], [0, 1, 2], default=-1).astype(np.int16)
    a2 = pd.Series(-1, index=index, dtype=np.int16)
    if listing_dates:
        list_date = pd.to_datetime(pd.Series(instruments.map(listing_dates), index=index), errors="coerce")
        known_age = list_date.notna()
        anniversary_map = {code: pd.Timestamp(value) + pd.DateOffset(years=3) for code, value in listing_dates.items()}
        anniversary = pd.Series(pd.to_datetime(pd.Index(instruments.map(anniversary_map))), index=index)
        a2.loc[known_age] = (dates[known_age.to_numpy()] >= anniversary.loc[known_age].to_numpy()).astype(np.int16)
    month_periods = dates.to_period("M")
    month_positions = {period: np.flatnonzero(month_periods == period) for period in month_periods.unique()}
    r4_snapshot_valid = 0
    r4_route_valid = 0
    for month_end in snapshots.get_level_values("datetime").unique().sort_values():
        snap_index = snapshots[snapshots.get_level_values("datetime") == month_end]
        snap_symbols = snap_index.get_level_values("instrument")
        next_month = month_end.to_period("M") + 1
        target_positions = month_positions.get(next_month, np.array([], dtype=np.int64))
        route_symbols = instruments[target_positions]
        route_b3 = pd.Series(b3.loc[snap_index].to_numpy(), index=snap_symbols).reindex(route_symbols).to_numpy(dtype=np.int16)
        route_a2 = pd.Series(a2.loc[snap_index].to_numpy(), index=snap_symbols).reindex(route_symbols).to_numpy(dtype=np.int16)
        groups.iloc[target_positions, groups.columns.get_loc("B3")] = route_b3
        groups.iloc[target_positions, groups.columns.get_loc("A2")] = route_a2
        routes = pd.DataFrame({"industry": raw_continuous.loc[snap_index, "r4_industry_strength60"].to_numpy(),
                               "stock": raw_continuous.loc[snap_index, "r4_stock_strength20"].to_numpy()}, index=snap_symbols).reindex(route_symbols)
        r4_snapshot_valid += int(raw_continuous.loc[snap_index, ["r4_industry_strength60", "r4_stock_strength20"]].notna().all(axis=1).sum())
        r4_route_valid += int(routes.notna().all(axis=1).sum())
        continuous.iloc[target_positions, continuous.columns.get_indexer(["r4_industry_strength60", "r4_stock_strength20"])] = routes.to_numpy(dtype=np.float32)
        known = routes.notna().all(axis=1)
        r4 = pd.Series(-1, index=routes.index, dtype=np.int16)
        r4.loc[known] = ((routes.loc[known, "industry"] >= 0).astype(np.int16)
                         + 2 * (routes.loc[known, "stock"] >= 0).astype(np.int16))
        groups.iloc[target_positions, groups.columns.get_loc("R4")] = r4.to_numpy(dtype=np.int16)
    return groups, continuous, {"industry_codes": industry_codes, "unknown_share": groups.eq(-1).mean().to_dict(),
        "r4_values_available_before_routing": {column: float(continuous[column].notna().mean()) for column in continuous},
        "r4_values_available_after_routing": {column: float(continuous[column].notna().mean()) for column in continuous},
        "r4_intermediate": {"industry_daily_valid": int(daily_industry["mean"].notna().sum()),
            "industry_window_valid": int(r4_industry.notna().sum()), "aligned_valid": int(aligned_industry_strength.notna().sum()),
            "stock_window_valid": int(stock_strength.notna().sum()), "snapshot_valid": r4_snapshot_valid,
            "route_valid": r4_route_valid}}
