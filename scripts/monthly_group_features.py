import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


MONTH_COLUMNS = (
    "month_return_1", "month_return_3", "month_return_6", "month_ma_gap_6",
    "month_vol_6", "month_drawdown_12", "month_volume_ratio_3", "month_position_12",
)


def build_daily_features(quotes):
    output = []
    for instrument, part in quotes.groupby(level="instrument", sort=False):
        stock = part.droplevel("instrument").sort_index()
        close = stock["close"].where(stock["close"] > 0)
        volume = stock["volume"].where(stock["volume"] >= 0)
        features = pd.DataFrame(index=stock.index)
        for n in (20, 60, 120, 250):
            features[f"daily_return_{n}"] = close.div(close.shift(n)).sub(1)
            features[f"daily_ma_gap_{n}"] = close.div(close.rolling(n, min_periods=n).mean()).sub(1)
        returns = close.pct_change(fill_method=None)
        for n in (20, 60):
            features[f"daily_vol_{n}"] = returns.rolling(n, min_periods=n).std()
        for n in (5, 20):
            features[f"daily_volume_ratio_{n}"] = volume.div(volume.shift(1).rolling(n, min_periods=n).mean().replace(0, np.nan))
        features["instrument"] = instrument
        output.append(features.reset_index().set_index(["datetime", "instrument"]))
    return pd.concat(output).reindex(quotes.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def build_monthly_features(quotes, calendar):
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    month_ends = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).last()
    completed = month_ends.loc[month_ends.index.to_timestamp(how="end").normalize() <= calendar.max()]
    output = []
    for instrument, part in quotes.groupby(level="instrument", sort=False):
        stock = part.droplevel("instrument").sort_index()
        periods = stock.index.to_period("M")
        bars = stock.groupby(periods).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        bars.index = pd.DatetimeIndex(month_ends.reindex(bars.index))
        ends = completed.loc[(completed >= stock.index.min()) & (completed <= stock.index.max())]
        bars = bars.reindex(pd.DatetimeIndex(ends))
        close = bars["close"].where(bars["close"] > 0)
        features = pd.DataFrame(index=bars.index)
        for n in (1, 3, 6):
            features[f"month_return_{n}"] = close.div(close.shift(n)).sub(1)
        features["month_ma_gap_6"] = close.div(close.rolling(6, min_periods=6).mean()).sub(1)
        features["month_vol_6"] = close.pct_change(fill_method=None).rolling(6, min_periods=6).std()
        high = bars["high"].rolling(12, min_periods=12).max()
        low = bars["low"].rolling(12, min_periods=12).min()
        features["month_drawdown_12"] = close.div(high).sub(1)
        features["month_volume_ratio_3"] = bars["volume"].div(bars["volume"].shift(1).rolling(3, min_periods=3).mean().replace(0, np.nan))
        features["month_position_12"] = close.sub(low).div(high.sub(low).replace(0, np.nan))
        features = features.reindex(stock.index, method="ffill")
        features["instrument"] = instrument
        output.append(features.reset_index().set_index(["datetime", "instrument"]))
    return pd.concat(output).reindex(quotes.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def build_classifications(context, monthly, quotes, industry, calendar):
    dates = pd.DatetimeIndex(context.index.get_level_values("datetime"))
    symbols = context.index.get_level_values("instrument")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    month_ends = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).last()
    current_ends = pd.DatetimeIndex(dates.to_period("M").map(month_ends))
    is_end = dates == current_ends
    month_index = context.index[is_end]
    groups = pd.DataFrame(-1, index=context.index, columns=["SL4", "PB4", "IC6", "GS8"], dtype=np.int16)
    audits = {name: [] for name in groups}
    size = context["context_log_circ_mv"].where(context["context_missing_size"] == 0)
    liquidity = context["context_log_amount_mean60"].where(context["context_missing_liquidity"] == 0)
    close = quotes["close"].where(quotes["close"] > 0)
    stock_return_20 = close.div(close.groupby(level="instrument", sort=False).shift(20)).sub(1)
    stock_return_60 = close.div(close.groupby(level="instrument", sort=False).shift(60)).sub(1)
    volatility = close.groupby(level="instrument", sort=False).pct_change(fill_method=None).groupby(level="instrument", sort=False).transform(lambda values: values.rolling(60, min_periods=60).std())
    previous_centers = None
    previous_industry = None
    previous_routes = {name: {} for name in groups}
    history = pd.DataFrame({"industry": industry, "return": close.groupby(level="instrument", sort=False).pct_change(fill_method=None)})
    for month_end in month_index.get_level_values("datetime").unique().sort_values():
        index = month_index[month_index.get_level_values("datetime") == month_end]
        snapshot = context.loc[index]
        values = pd.DataFrame({"size": size.loc[index], "liquidity": liquidity.loc[index], "volatility": volatility.loc[index]})
        medians = values.median()
        valid_sl = values[["size", "liquidity"]].notna().all(axis=1)
        sl = pd.Series(-1, index=index, dtype=np.int16)
        sl.loc[valid_sl] = ((values.loc[valid_sl, "size"] > medians["size"]).astype(np.int16)
                            + 2 * (values.loc[valid_sl, "liquidity"] > medians["liquidity"]).astype(np.int16))
        valid_gs = values.notna().all(axis=1)
        gs = pd.Series(-1, index=index, dtype=np.int16)
        gs.loc[valid_gs] = (sl.loc[valid_gs] + 4 * (values.loc[valid_gs, "volatility"] > medians["volatility"]).astype(np.int16))
        behavior = pd.DataFrame({"ret20": stock_return_20.loc[index], "ret60": stock_return_60.loc[index],
                                 "vol60": volatility.loc[index], "beta120": snapshot["context_beta120"],
                                 "corr120": snapshot["context_corr120"],
                                 "residvol120": snapshot["context_residvol120"]}).replace([np.inf, -np.inf], np.nan)
        valid_pb = behavior.notna().all(axis=1) & (snapshot["context_missing_market_relation"] == 0)
        pb = pd.Series(-1, index=index, dtype=np.int16)
        center_audit = None
        if valid_pb.sum() >= 400:
            ranks = behavior.loc[valid_pb].rank(pct=True).to_numpy(dtype=float)
            model = KMeans(n_clusters=4, random_state=42, n_init=10).fit(ranks)
            centers = model.cluster_centers_
            if previous_centers is None:
                order = np.lexsort(centers.T[::-1])
                mapping = np.empty(4, dtype=np.int16)
                mapping[order] = np.arange(4)
            else:
                row, col = linear_sum_assignment(((previous_centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
                mapping = np.empty(4, dtype=np.int16)
                mapping[col] = row
            pb.loc[valid_pb] = mapping[model.labels_]
            previous_centers = centers[np.argsort(mapping)]
            center_audit = previous_centers.tolist()
        ic = pd.Series(-1, index=index, dtype=np.int16)
        industry_map = {}
        market_days = calendar[(calendar <= month_end)][-252:]
        previous = history.loc[(slice(market_days[0], month_end), slice(None)), :].dropna()
        if len(market_days) == 252 and not previous.empty:
            industry_day = previous.groupby([pd.Grouper(level="datetime"), "industry"])["return"].agg(["mean", "count"])
            industry_day = industry_day.loc[(industry_day.index.get_level_values("industry") >= 0) & (industry_day["count"] >= 20)]
            returns = industry_day["mean"].unstack("industry")
            eligible = returns.columns[returns.count() >= 200]
            correlation = returns[eligible].corr(min_periods=200)
            while len(correlation) >= 6 and correlation.isna().any().any():
                worst = correlation.isna().sum().sort_values(ascending=False).index[0]
                correlation = correlation.drop(index=worst, columns=worst)
            if len(correlation) >= 6:
                distance = (1 - correlation.clip(-1, 1)).clip(lower=0)
                np.fill_diagonal(distance.values, 0)
                labels = cut_tree(linkage(squareform(distance.to_numpy(), checks=False), method="average"), n_clusters=6).ravel()
                industry_map = {int(code): int(label) for code, label in zip(correlation.index, labels)}
                if previous_industry:
                    overlap = np.zeros((6, 6), dtype=int)
                    for code, label in industry_map.items():
                        if code in previous_industry:
                            overlap[previous_industry[code], label] += 1
                    row, col = linear_sum_assignment(-overlap)
                    remap = dict(zip(col, row))
                    industry_map = {code: remap[label] for code, label in industry_map.items()}
                else:
                    ordered = sorted(set(industry_map.values()), key=lambda group: min(code for code, value in industry_map.items() if value == group))
                    remap = {value: number for number, value in enumerate(ordered)}
                    industry_map = {code: remap[value] for code, value in industry_map.items()}
                previous_industry = industry_map
                ic = industry.loc[index].map(industry_map).fillna(-1).astype(np.int16)
        snapshots = {"SL4": sl, "PB4": pb, "IC6": ic, "GS8": gs}
        for name, values in snapshots.items():
            target_month = month_end.to_period("M") + 1
            target = (dates.to_period("M") == target_month) & symbols.isin(index.get_level_values("instrument"))
            route = pd.Series(values.to_numpy(), index=index.get_level_values("instrument"))
            groups.loc[target, name] = route.reindex(symbols[target]).to_numpy()
            prior = previous_routes[name]
            comparable = [(symbol, int(value)) for symbol, value in route.items()
                          if symbol in prior and value >= 0 and prior[symbol] >= 0]
            migration = (sum(prior[symbol] != value for symbol, value in comparable) / len(comparable)
                         if comparable else None)
            previous_routes[name] = route.to_dict()
            audits[name].append({"as_of": str(month_end.date()), "valid": int((values >= 0).sum()),
                                 "unknown": int((values < 0).sum()), "sizes": {int(key): int(count) for key, count in values.value_counts().sort_index().items()},
                                 "migration_rate": migration,
                                 "centers": center_audit if name == "PB4" else None,
                                 "industry_members": industry_map if name == "IC6" else None})
    return groups, audits
