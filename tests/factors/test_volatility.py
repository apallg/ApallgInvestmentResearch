import pytest
import numpy as np
from tests.factories import make_kline_df, make_kline_df_ranging
from core.domain.factors.stock_factors import factor_volatility, factor_vol_regime, factor_max_drawdown


class TestFactorVolatility:
    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        assert factor_volatility(df, n=20) == 0.0

    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = factor_volatility(df, n=20)
        assert -1.0 <= result <= 1.0


class TestFactorVolRegime:
    def test_spike_high(self):
        df = make_kline_df(n_rows=80)
        # 制造波动率突变：近期高波动，历史低波动
        closes = np.array(df["close"], dtype=float)
        closes[-20:] = closes[-20] + np.random.default_rng(1).normal(0, 0.5, 20).cumsum()
        df["close"] = closes
        result = factor_vol_regime(df, n=20)
        assert result == -1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=20)
        assert factor_vol_regime(df, n=20) == 0.0

    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = factor_vol_regime(df, n=20)
        assert -1.0 <= result <= 1.0


class TestFactorMaxDrawdown:
    def test_deep_drawdown(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        closes = list(df["close"])
        closes[-5:] = [8.0, 7.8, 7.5, 7.3, 7.0]
        df["close"] = closes
        result = factor_max_drawdown(df, n=20)
        assert result < -0.5

    def test_no_drawdown(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        closes = np.linspace(10.0, 12.0, 60)
        df["close"] = closes
        result = factor_max_drawdown(df, n=20)
        assert result >= -0.01

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        assert factor_max_drawdown(df, n=20) == 0.0

    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = factor_max_drawdown(df, n=20)
        assert -1.0 <= result <= 1.0
