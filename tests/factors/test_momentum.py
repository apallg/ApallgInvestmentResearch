import pytest
import numpy as np
from tests.factories import make_kline_df_uptrend, make_kline_df_downtrend, make_kline_df
from core.domain.factors.stock_factors import factor_momentum, factor_reversal, factor_trend_momentum


class TestFactorMomentum:
    def test_strong_uptrend_positive(self):
        prices = [10.0] * 100 + list(np.linspace(10.0, 16.0, 20))
        df = make_kline_df(n_rows=120, with_indicators=True)
        for i, p in enumerate(prices):
            df.at[df.index[i], "close"] = p
        result = factor_momentum(df, n=20)
        assert result > 0.3

    def test_strong_downtrend_negative(self):
        prices = [15.0] * 100 + list(np.linspace(15.0, 8.0, 20))
        df = make_kline_df(n_rows=120, with_indicators=True)
        for i, p in enumerate(prices):
            df.at[df.index[i], "close"] = p
        result = factor_momentum(df, n=20)
        assert result < -0.3

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        result = factor_momentum(df, n=20)
        assert result == 0.0

    def test_output_in_range(self):
        df = make_kline_df_uptrend(120)
        result = factor_momentum(df, n=20)
        assert -1.0 <= result <= 1.0


class TestFactorReversal:
    def test_after_drop_positive(self):
        prices = [10.0] * 110 + [10.0, 9.8, 9.5, 9.2, 9.0]
        df = make_kline_df(n_rows=115, with_indicators=True)
        for i, p in enumerate(prices[-len(df):]):
            df.at[df.index[i], "close"] = p
        result = factor_reversal(df, n=5)
        assert result > 0

    def test_after_rally_negative(self):
        prices = [10.0] * 110 + [10.0, 10.2, 10.5, 10.8, 11.0]
        df = make_kline_df(n_rows=115, with_indicators=True)
        for i, p in enumerate(prices[-len(df):]):
            df.at[df.index[i], "close"] = p
        result = factor_reversal(df, n=5)
        assert result < 0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=3)
        result = factor_reversal(df, n=5)
        assert result == 0.0

    def test_output_in_range(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        result = factor_reversal(df, n=5)
        assert -1.0 <= result <= 1.0


class TestFactorTrendMomentum:
    def test_long_uptrend(self):
        df = make_kline_df_uptrend(120)
        result = factor_trend_momentum(df, n=60)
        assert result > 0

    def test_short_df_falls_back_to_momentum(self):
        df = make_kline_df(n_rows=30)
        fallback = factor_momentum(df, n=20)
        result = factor_trend_momentum(df, n=60)
        assert result == fallback

    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = factor_trend_momentum(df, n=60)
        assert -1.0 <= result <= 1.0
