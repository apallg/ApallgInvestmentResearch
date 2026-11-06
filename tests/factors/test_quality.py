import pytest
from tests.factories import make_kline_df
from core.domain.factors.stock_factors import (
    _score_roe, _score_gross_margin, _score_debt_ratio, _score_accruals
)


class TestScoreRoe:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = _score_roe(df, n=20)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        assert _score_roe(df, n=20) == 0.0


class TestScoreGrossMargin:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        result = _score_gross_margin(df, n=20)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        assert _score_gross_margin(df, n=20) == 0.0


class TestScoreDebtRatio:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=60, with_indicators=True)
        result = _score_debt_ratio(df, n=20)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=10)
        assert _score_debt_ratio(df, n=20) == 0.0


class TestScoreAccruals:
    def test_output_in_range(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = _score_accruals(df, n=60)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_zero(self):
        df = make_kline_df(n_rows=30)
        assert _score_accruals(df, n=60) == 0.0
