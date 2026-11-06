import numpy as np
import pandas as pd

from scripts.horizon_multiscale_features import build_monthly_bars, monthly_history_for_date


MONTH_VALUE_COLUMNS = ["open_gap", "high_gap", "low_gap", "close_gap", "volume_change", "valid_day_ratio"]


def build_daily_long_features(quotes, calendar):
    calendar = pd.DatetimeIndex(calendar)
    close = quotes["close"].where(quotes["close"] > 0)
    volume = quotes["volume"].where(quotes["volume"] > 0)
    out = pd.DataFrame(index=quotes.index)
    for instrument, rows in close.groupby(level="instrument", sort=False):
        prices = rows.droplevel("instrument").reindex(calendar)
        returns = prices.pct_change(fill_method=None)
        features = pd.DataFrame(index=calendar)
        for window in (20, 60, 120, 250):
            features[f"return_{window}"] = prices.div(prices.shift(window)).sub(1)
            features[f"ma_deviation_{window}"] = prices.div(prices.rolling(window, min_periods=window).mean()).sub(1)
        for window in (20, 60):
            features[f"volatility_{window}"] = returns.rolling(window, min_periods=window).std()
        stock_volume = volume.xs(instrument, level="instrument").reindex(calendar)
        for window in (5, 20):
            features[f"volume_ratio_{window}"] = stock_volume.div(
                stock_volume.shift(1).rolling(window, min_periods=window).mean())
        out.loc[rows.index, features.columns] = features.reindex(rows.index.get_level_values("datetime")).to_numpy()
    out = out.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return pd.concat([out, out.isna().astype(np.float32).rename(columns=lambda x: f"missing_{x}")], axis=1).fillna(0)


def build_monthly_sequence_table(quotes, calendar):
    bars = build_monthly_bars(quotes, calendar)
    values = bars[MONTH_VALUE_COLUMNS]
    masks = values.isna().astype(np.float32)
    return pd.concat([values.fillna(0), masks.rename(columns=lambda name: f"missing_{name}")], axis=1).astype(np.float32)


def daily_window_positions(index, calendar, endpoints, length=20):
    calendar = pd.DatetimeIndex(calendar)
    calendar_positions = calendar.get_indexer(index.get_level_values("datetime"))
    if (calendar_positions < 0).any():
        raise ValueError("序列日期未映射到统一交易日历")
    by_instrument = {}
    for instrument, location in pd.Series(np.arange(len(index)), index=index).groupby(level="instrument", sort=False):
        rows = location.to_numpy(dtype=np.int64)
        positions = calendar_positions[rows]
        lookup = np.full(len(calendar), -1, dtype=np.int64)
        lookup[positions] = rows
        by_instrument[instrument] = lookup
    dates = endpoints.get_level_values("datetime")
    instruments = endpoints.get_level_values("instrument")
    result = np.full((len(endpoints), length), -1, dtype=np.int64)
    for row, (date, instrument) in enumerate(zip(dates, instruments)):
        end = calendar.get_loc(date)
        lookup = by_instrument.get(instrument)
        if lookup is not None:
            offsets = np.arange(end - length + 1, end + 1)
            valid = (offsets >= 0) & (offsets < len(calendar))
            result[row, valid] = lookup[offsets[valid]]
    return result


def monthly_sequence_for_endpoint(bars, instrument, as_of, calendar, months=12):
    return monthly_history_for_date(bars, instrument, as_of, calendar, months)[
        MONTH_VALUE_COLUMNS + [f"missing_{name}" for name in MONTH_VALUE_COLUMNS]
    ].to_numpy(dtype=np.float32)
