import numpy as np
import pandas as pd


def rolling_last_rank(series, window):
    if window <= 0:
        raise ValueError("隔离研究只支持固定正窗口")
    values = series.to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    count = np.zeros(len(values), dtype=float)
    for lag in range(1, min(window, len(values))):
        count[lag:] += values[lag:] >= values[:-lag]
    denominator = np.minimum(np.arange(len(values)), window - 1)
    result = np.divide(count, denominator, out=np.full(len(values), np.nan), where=denominator > 0)
    valid = np.concatenate(([0], np.cumsum(~np.isnan(values))))
    positions = np.arange(len(values))
    observations = valid[positions + 1] - valid[np.maximum(0, positions - window + 1)]
    result[observations == 0] = np.nan
    return pd.Series(result, index=series.index)


def aligned_correlation(left, right, window):
    result = left.rolling(window, min_periods=1).corr(right)
    left_constant = pd.Series(np.isclose(left.rolling(window, min_periods=1).std(), 0, atol=2e-5), index=left.index)
    right_constant = pd.Series(np.isclose(right.rolling(window, min_periods=1).std(), 0, atol=2e-5), index=right.index)
    mask = left_constant.reindex(result.index, fill_value=False) | right_constant.reindex(result.index, fill_value=False)
    result.loc[mask] = np.nan
    return result
