import pytest
from tests.factories import make_kline_df
from core.domain.factors.stock_factors import _score_size, _score_cap_median


class TestScoreSize:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        result = _score_size(df, n=20)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        assert _score_size(df, n=20) == 0.0

    def test_zero_amount_returns_zero(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df["amount"] = 0.0
        assert _score_size(df, n=20) == 0.0


class TestScoreCapMedian:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = _score_cap_median(df, n=60)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=30)
        assert _score_cap_median(df, n=60) == 0.0
