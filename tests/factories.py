import pandas as pd
import numpy as np
from datetime import date, timedelta


def _trading_dates(start, n):
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def make_kline_df(prices=None, volumes=None, amounts=None, n_rows=120,
                  with_indicators=False, start_date=None):
    if start_date is None:
        start_date = date.today() - timedelta(days=int(n_rows * 7 / 5 * 2))
    dates = _trading_dates(start_date, n_rows)

    rng = np.random.default_rng(42)
    if prices is None:
        rets = rng.normal(0.0005, 0.02, n_rows)
        prices = 10.0 * np.cumprod(1 + rets)

    closes = np.array(prices[-n_rows:]) if len(prices) > n_rows else np.array(prices)
    if len(closes) < n_rows:
        closes = np.pad(closes, (n_rows - len(closes), 0), 'edge')

    highs = closes * (1 + np.abs(rng.normal(0, 0.015, n_rows)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.015, n_rows)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * (1 + rng.normal(0, 0.005))

    if volumes is None:
        volumes = rng.lognormal(mean=14, sigma=0.8, size=n_rows).astype(float)
    if amounts is None:
        amounts = volumes * closes

    turnover = volumes / 1e7 * (1 + rng.uniform(-0.3, 0.3, n_rows))
    pct_chg = np.zeros(n_rows)
    if n_rows >= 2:
        pct_chg[1:] = (closes[1:] - closes[:-1]) / closes[:-1] * 100

    df = pd.DataFrame({
        "trade_date": dates,
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "amount": amounts,
        "turnover": np.abs(turnover),
        "pct_chg": pct_chg,
    })

    if with_indicators:
        df = _add_indicators(df, rng)

    return df


def _add_indicators(df, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(df)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MA
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()

    # KDJ
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100
    rsv = rsv.fillna(50.0)
    k_vals = [50.0]
    d_vals = [50.0]
    for i in range(1, n):
        kv = 2/3 * k_vals[-1] + 1/3 * rsv.iloc[i]
        if np.isnan(kv):
            kv = k_vals[-1]
        k_vals.append(kv)
        d_vals.append(2/3 * d_vals[-1] + 1/3 * k_vals[-1])
    df["k"] = k_vals
    df["d"] = d_vals
    df["j"] = [3 * k_vals[i] - 2 * d_vals[i] for i in range(n)]

    # Bollinger
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    # Volume MA
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    return df


def make_kline_df_uptrend(n_rows=120):
    closes = 10.0 * (1.0 + np.linspace(0.0, 0.80, n_rows))
    return make_kline_df(prices=closes.tolist(), n_rows=n_rows, with_indicators=True)


def make_kline_df_downtrend(n_rows=120):
    closes = 10.0 * (1.0 - np.linspace(0.0, 0.55, n_rows))
    return make_kline_df(prices=closes.tolist(), n_rows=n_rows, with_indicators=True)


def make_kline_df_ranging(n_rows=120):
    rng = np.random.default_rng(99)
    noise = rng.normal(0, 0.08, n_rows).cumsum() * 0.3
    closes = 10.0 + np.clip(noise, -1.5, 1.5)
    return make_kline_df(prices=closes.tolist(), n_rows=n_rows, with_indicators=True)


def make_kline_df_short(n_rows=10):
    return make_kline_df(n_rows=n_rows, with_indicators=True)


def make_kline_df_with_price(prices, n_rows=None):
    if n_rows is None:
        n_rows = len(prices)
    return make_kline_df(prices=list(prices), n_rows=n_rows, with_indicators=True)
