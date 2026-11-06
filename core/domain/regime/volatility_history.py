import numpy as np
import pandas as pd


def build_volatility_history(index_df, start=None, end=None):
    if index_df is None or index_df.empty:
        return []
    date_col = "trade_date" if "trade_date" in index_df.columns else "date"
    if date_col not in index_df.columns or "close" not in index_df.columns:
        return []
    frame = index_df[[date_col, "close"]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().sort_values(date_col).drop_duplicates(date_col, keep="last")
    frame["ret"] = frame["close"].pct_change(fill_method=None)
    frame["ret_20d"] = frame["close"].pct_change(20, fill_method=None)
    frame["vol_20d"] = frame["ret"].rolling(20).std() * np.sqrt(252)
    frame["vol_percentile"] = frame["vol_20d"].rolling(252).rank(pct=True)
    frame = frame.dropna(subset=["ret_20d", "vol_20d", "vol_percentile"])
    if start is not None:
        frame = frame[frame[date_col] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame[date_col] <= pd.Timestamp(end)]
    rows = []
    for row in frame.itertuples(index=False):
        trade_date = getattr(row, date_col)
        ret_20d = float(row.ret_20d)
        vol_20d = float(row.vol_20d)
        vol_percentile = float(row.vol_percentile)
        vol_prefix = "volatile" if vol_percentile > 0.7 else "steady"
        direction = "up" if ret_20d > 0.02 else ("down" if ret_20d < -0.02 else "flat")
        rows.append({
            "trade_date": trade_date.date().isoformat(),
            "regime": f"{vol_prefix}_{direction}",
            "confidence": round(min(max(abs(ret_20d) * 5, abs(vol_percentile - 0.5) * 2), 0.99), 4),
            "vol_20d": vol_20d,
            "vol_percentile": vol_percentile,
            "vol_state": "high_vol" if vol_prefix == "volatile" else "low_vol",
            "method": "rule_based_pit_v1",
        })
    return rows
