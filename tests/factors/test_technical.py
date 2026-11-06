import pytest
import pandas as pd
import numpy as np
from tests.factories import make_kline_df
from core.domain.factors.stock_factors import (
    factor_macd, factor_rsi, factor_ma_alignment,
    factor_kdj, factor_bollinger, factor_volume,
)


class TestFactorMacd:
    def test_missing_column_returns_zero(self):
        df = make_kline_df(n_rows=50)
        assert factor_macd(df) == 0.0

    def test_golden_cross_returns_one(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-2], "macd"] = 0.05
        df.at[df.index[-2], "macd_signal"] = 0.10
        df.at[df.index[-1], "macd"] = 0.15
        df.at[df.index[-1], "macd_signal"] = 0.12
        assert factor_macd(df) == 1.0

    def test_death_cross_returns_negative_one(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-2], "macd"] = 0.15
        df.at[df.index[-2], "macd_signal"] = 0.12
        df.at[df.index[-1], "macd"] = 0.05
        df.at[df.index[-1], "macd_signal"] = 0.10
        assert factor_macd(df) == -1.0

    def test_no_cross_returns_zero(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-2], "macd"] = 0.10
        df.at[df.index[-2], "macd_signal"] = 0.12
        df.at[df.index[-1], "macd"] = 0.09
        df.at[df.index[-1], "macd_signal"] = 0.11
        assert factor_macd(df) == 0.0

    def test_single_row_returns_zero(self):
        df = make_kline_df(n_rows=1, with_indicators=True)
        assert factor_macd(df) == 0.0

    def test_nan_macd_returns_zero(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "macd"] = np.nan
        assert factor_macd(df) == 0.0


class TestFactorRsi:
    def test_missing_column_returns_zero(self):
        df = make_kline_df(n_rows=50)
        assert factor_rsi(df) == 0.0

    def test_oversold_returns_one(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "rsi14"] = 25.0
        assert factor_rsi(df) == 1.0

    def test_overbought_returns_negative_one(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "rsi14"] = 75.0
        assert factor_rsi(df) == -1.0

    def test_mid_range_returns_small(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "rsi14"] = 50.0
        result = factor_rsi(df)
        assert -0.25 < result < 0.25

    def test_nan_rsi_returns_zero(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "rsi14"] = np.nan
        assert factor_rsi(df) == 0.0


class TestFactorMaAlignment:
    def test_bull_alignment_returns_one(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "ma5"] = 15.0
        df.at[df.index[-1], "ma10"] = 14.0
        df.at[df.index[-1], "ma20"] = 13.0
        df.at[df.index[-1], "ma60"] = 12.0
        assert factor_ma_alignment(df) == 1.0

    def test_bear_alignment_returns_negative_one(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "ma5"] = 12.0
        df.at[df.index[-1], "ma10"] = 13.0
        df.at[df.index[-1], "ma20"] = 14.0
        df.at[df.index[-1], "ma60"] = 15.0
        assert factor_ma_alignment(df) == -1.0

    def test_mixed_returns_zero(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "ma5"] = 14.0
        df.at[df.index[-1], "ma10"] = 13.0
        df.at[df.index[-1], "ma20"] = 15.0
        df.at[df.index[-1], "ma60"] = 12.0
        assert factor_ma_alignment(df) == 0.0

    def test_partial_columns_returns_zero(self):
        df = make_kline_df(n_rows=60)
        df["ma5"] = [12.0] * 60
        df["ma10"] = [11.0] * 60
        # 只有2条均线，不够3条
        assert factor_ma_alignment(df) == 0.0


class TestFactorKdj:
    def test_j_oversold_returns_positive(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "j"] = 15.0
        assert factor_kdj(df) == 0.8

    def test_j_overbought_returns_negative(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "j"] = 85.0
        assert factor_kdj(df) == -0.5

    def test_golden_cross_low_zone(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-2], "k"] = 30.0
        df.at[df.index[-2], "d"] = 35.0
        df.at[df.index[-1], "k"] = 36.0
        df.at[df.index[-1], "d"] = 34.0
        assert factor_kdj(df) == 0.6

    def test_death_cross_high_zone(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-2], "k"] = 70.0
        df.at[df.index[-2], "d"] = 65.0
        df.at[df.index[-1], "k"] = 63.0
        df.at[df.index[-1], "d"] = 68.0
        assert factor_kdj(df) == -0.4

    def test_missing_column_returns_zero(self):
        df = make_kline_df(n_rows=50)
        df["k"] = [50.0] * 50
        assert factor_kdj(df) == 0.0


class TestFactorBollinger:
    def test_at_lower_band_returns_one(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        lower = df["bb_lower"].iloc[-1]
        df.at[df.index[-1], "close"] = lower
        assert factor_bollinger(df) == 1.0

    def test_at_upper_band_returns_negative(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        upper = df["bb_upper"].iloc[-1]
        df.at[df.index[-1], "close"] = upper
        assert factor_bollinger(df) == -0.8

    def test_above_middle_returns_mild_negative(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        middle = df["bb_middle"].iloc[-1]
        upper = df["bb_upper"].iloc[-1]
        df.at[df.index[-1], "close"] = (middle + upper) / 2
        assert factor_bollinger(df) == -0.2

    def test_below_middle_returns_mild_positive(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        middle = df["bb_middle"].iloc[-1]
        lower = df["bb_lower"].iloc[-1]
        df.at[df.index[-1], "close"] = (lower + middle) / 2
        assert factor_bollinger(df) == 0.2

    def test_missing_column_returns_zero(self):
        df = make_kline_df(n_rows=50)
        assert factor_bollinger(df) == 0.0


class TestFactorVolume:
    def test_price_up_high_volume_ratio(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "close"] = 11.0
        df.at[df.index[-2], "close"] = 10.0
        df.at[df.index[-1], "volume"] = 2000000.0
        df.at[df.index[-1], "vol_ma5"] = 1000000.0
        df.at[df.index[-1], "vol_ma20"] = 900000.0
        assert factor_volume(df) == 0.7

    def test_price_down_high_volume_ratio(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df.at[df.index[-1], "close"] = 9.0
        df.at[df.index[-2], "close"] = 10.0
        df.at[df.index[-1], "volume"] = 2000000.0
        df.at[df.index[-1], "vol_ma5"] = 1000000.0
        df.at[df.index[-1], "vol_ma20"] = 900000.0
        assert factor_volume(df) == -0.5

    def test_missing_columns_returns_zero(self):
        df = make_kline_df(n_rows=50)
        assert factor_volume(df) == 0.0

    def test_zero_vol_ma5_returns_zero(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df["vol_ma5"] = 0.0
        assert factor_volume(df) == 0.0
