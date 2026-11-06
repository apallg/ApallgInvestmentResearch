import numpy as np
import pandas as pd


def build_multiscale_features(quotes, calendar):
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(quotes.columns):
        raise ValueError(f"日线缺少字段: {sorted(required - set(quotes.columns))}")
    if quotes.index.names != ["datetime", "instrument"] or quotes.index.has_duplicates:
        raise ValueError("日线必须使用无重复的 datetime/instrument 索引")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    if len(calendar) == 0 or not quotes.index.get_level_values("datetime").isin(calendar).all():
        raise ValueError("日线日期未完整包含于冻结交易日历")
    output = []
    for instrument, stock in quotes.groupby(level="instrument", sort=False):
        stock = stock.droplevel("instrument").sort_index()
        close = stock["close"].where(stock["close"] > 0)
        volume = stock["volume"].where(stock["volume"] >= 0)
        daily = pd.DataFrame(index=stock.index)
        for n in (20, 60, 120, 250):
            daily[f"daily_return_{n}"] = close.div(close.shift(n)).sub(1)
            daily[f"daily_ma_gap_{n}"] = close.div(close.rolling(n, min_periods=n).mean()).sub(1)
        returns = close.pct_change(fill_method=None)
        for n in (20, 60):
            daily[f"daily_vol_{n}"] = returns.rolling(n, min_periods=n).std()
        for n in (5, 20):
            daily[f"daily_volume_ratio_{n}"] = volume.div(volume.shift(1).rolling(n, min_periods=n).mean().replace(0, np.nan))
        for frequency, prefix in (("W-SUN", "week"), ("M", "month")):
            day_periods = stock.index.to_period(frequency)
            calendar_periods = calendar.to_period(frequency)
            final_days = pd.Series(calendar, index=calendar_periods).groupby(level=0).last()
            bars = stock.groupby(day_periods).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            bars.index = pd.DatetimeIndex(final_days.reindex(bars.index))
            bars = bars.loc[bars.index.notna()].sort_index()
            completed = final_days.loc[final_days.index.to_timestamp(how="end").normalize() <= calendar.max()]
            completed = completed.loc[(completed >= stock.index.min()) & (completed <= stock.index.max())]
            bars = bars.reindex(pd.DatetimeIndex(completed))
            features = pd.DataFrame(index=bars.index)
            bar_close = bars["close"].where(bars["close"] > 0)
            if prefix == "week":
                for n in (1, 4, 12):
                    features[f"week_return_{n}"] = bar_close.div(bar_close.shift(n)).sub(1)
                features["week_ma_ratio_4_12"] = bar_close.rolling(4, min_periods=4).mean().div(bar_close.rolling(12, min_periods=12).mean()).sub(1)
                features["week_vol_12"] = bar_close.pct_change(fill_method=None).rolling(12, min_periods=12).std()
                features["week_range"] = bars["high"].sub(bars["low"]).div(bar_close.shift(1))
                features["week_volume_ratio_4"] = bars["volume"].div(bars["volume"].shift(1).rolling(4, min_periods=4).mean().replace(0, np.nan))
                low = bars["low"].rolling(12, min_periods=12).min()
                high = bars["high"].rolling(12, min_periods=12).max()
                features["week_position_12"] = bar_close.sub(low).div(high.sub(low).replace(0, np.nan))
            else:
                for n in (1, 3, 6):
                    features[f"month_return_{n}"] = bar_close.div(bar_close.shift(n)).sub(1)
                features["month_ma_gap_6"] = bar_close.div(bar_close.rolling(6, min_periods=6).mean()).sub(1)
                features["month_vol_6"] = bar_close.pct_change(fill_method=None).rolling(6, min_periods=6).std()
                high = bars["high"].rolling(12, min_periods=12).max()
                low = bars["low"].rolling(12, min_periods=12).min()
                features["month_drawdown_12"] = bar_close.div(high).sub(1)
                features["month_volume_ratio_3"] = bars["volume"].div(bars["volume"].shift(1).rolling(3, min_periods=3).mean().replace(0, np.nan))
                features["month_position_12"] = bar_close.sub(low).div(high.sub(low).replace(0, np.nan))
            features = features.reindex(stock.index, method="ffill")
            daily = pd.concat([daily, features], axis=1)
        daily["instrument"] = instrument
        output.append(daily.reset_index().set_index(["datetime", "instrument"]))
    return pd.concat(output).reindex(quotes.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def build_style_groups(context, daily_vol_60, calendar):
    required = {"context_log_circ_mv", "context_log_amount_mean60"}
    if not required.issubset(context.columns):
        raise ValueError(f"风格分组缺少字段: {sorted(required - set(context.columns))}")
    data = context[list(required)].copy()
    if "context_missing_size" in context:
        data.loc[context["context_missing_size"] > 0, "context_log_circ_mv"] = np.nan
    if "context_missing_liquidity" in context:
        data.loc[context["context_missing_liquidity"] > 0, "context_log_amount_mean60"] = np.nan
    data["daily_vol_60"] = daily_vol_60.reindex(data.index)
    dates = pd.DatetimeIndex(data.index.get_level_values("datetime"))
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    month_ends = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).last()
    end_dates = dates.to_period("M").map(month_ends)
    month_end_rows = data.loc[dates == end_dates]
    medians = month_end_rows.groupby(level="datetime").median()
    bits = pd.DataFrame(index=month_end_rows.index)
    for offset, field in enumerate(("context_log_circ_mv", "context_log_amount_mean60", "daily_vol_60")):
        values = month_end_rows[field]
        threshold = pd.Series(medians[field].reindex(month_end_rows.index.get_level_values("datetime")).to_numpy(), index=month_end_rows.index)
        bits[field] = np.where(values.notna() & threshold.notna(), (values > threshold).astype(int) * (2 ** offset), -1)
    groups = bits.where(bits >= 0).sum(axis=1, min_count=3).astype("Int8")
    month = groups.index.get_level_values("datetime").to_period("M") + 1
    codes = groups.index.get_level_values("instrument")
    routed = pd.Series(groups.to_numpy(), index=pd.MultiIndex.from_arrays([month, codes], names=["month", "instrument"]))
    target = pd.MultiIndex.from_arrays([dates.to_period("M"), data.index.get_level_values("instrument")], names=["month", "instrument"])
    return pd.Series(routed.reindex(target).to_numpy(), index=context.index, name="style_group").astype("Int8")
