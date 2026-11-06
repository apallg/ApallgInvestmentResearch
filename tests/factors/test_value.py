import pytest
from tests.factories import make_kline_df
from core.domain.factors.stock_factors import _score_ep, _score_bp, _score_dp


class TestScoreEp:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        result = _score_ep(df, n=5)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=3)
        assert _score_ep(df, n=5) == 0.0

    def test_zero_amount_returns_zero(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        df["amount"] = 0.0
        assert _score_ep(df, n=5) == 0.0


class TestScoreBp:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        result = _score_bp(df, n=20)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        assert _score_bp(df, n=20) == 0.0


class TestScoreDp:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = _score_dp(df, n=60)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=30)
        assert _score_dp(df, n=60) == 0.0
