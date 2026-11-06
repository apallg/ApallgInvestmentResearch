"""市场宽度纯函数测试 (从 timing/breadth.py 提取)"""
import pytest
import numpy as np
import pandas as pd

from core.domain.timing.breadth import compute_vol_turnover_kernel, compute_rsrs_momentum
from core.domain.timing.breadth import MarketBreadth
from backend.api.timing import _present_breadth_metrics


@pytest.mark.p0
def test_breadth_display_uses_all_observed_rows_without_changing_model_features():
    rows = [
        {
            "pct_chg": 1 if index < 50 else -1 if index < 90 else None,
            "amount": 1000 if index < 110 else None,
            "high": 11 if index < 110 else None,
            "low": 9 if index < 110 else None,
            "close": 10 if index < 110 else None,
            "volume": 100 if index < 110 else None,
            "turnover": 1 if index < 110 else None,
        }
        for index in range(120)
    ]

    internal = MarketBreadth().compute_from_rows(rows)
    result = _present_breadth_metrics(internal)

    assert internal["total_stocks"] == 110
    assert internal["up_ratio"] == result["up_ratio"]
    assert result["total_stocks"] == 120
    assert result["up_count"] == 50
    assert result["down_count"] == 40
    assert result["unknown_count"] == 30
    assert result["change_coverage"] == 0.75
    assert result["amplitude_median"] == 20.0
    assert result["coverage_basis"] == "observed_daily_quotes_rows"


@pytest.mark.p0
def test_breadth_does_not_infer_limit_counts_from_one_universal_threshold():
    rows = [{
        "pct_chg": 9.6,
        "volume": 100,
        "amount": 1000,
        "high": 11,
        "low": 9,
        "close": 10,
        "turnover": 1,
    } for _ in range(100)]

    internal = MarketBreadth().compute_from_rows(rows)
    result = _present_breadth_metrics(internal)

    assert internal["limit_up"] == 100
    assert result["limit_up"] is None
    assert result["limit_down"] is None
    assert result["limit_count_missing_reason"] == "security_specific_limit_price_unavailable"


@pytest.mark.p0
def test_breadth_api_masks_unverified_limit_counts():
    import asyncio
    from unittest.mock import patch
    from backend.api.timing import get_breadth_api

    internal = {
        "up_count": 50, "down_count": 40, "flat_count": 0, "unknown_count": 10,
        "valid_change_count": 90, "change_coverage": 0.9, "up_ratio": 1.25,
        "limit_up": 80, "limit_down": 0, "total_stocks": 95,
        "observed_row_count": 100, "observed_up_count": 50,
        "observed_down_count": 40, "observed_flat_count": 0,
        "observed_unknown_count": 10, "observed_valid_change_count": 90,
        "observed_change_coverage": 0.9, "observed_up_ratio": 1.25,
    }
    with patch("backend.api.timing._get_breadth", return_value=internal):
        result = asyncio.run(get_breadth_api())["data"]

    assert result["total_stocks"] == 100
    assert result["up_count"] == 50
    assert result["limit_up"] is None
    assert result["limit_down"] is None
    assert result["limit_count_status"] == "unavailable"


@pytest.mark.p1
class TestComputeRsrsMomentum:
    def test_upward_trend_positive(self):
        lows = np.array([10.0 + i * 0.02 for i in range(100)])
        score, ann_ret, r2 = compute_rsrs_momentum(lows, 89)
        assert score is not None
        assert ann_ret > 0
        assert 0 <= r2 <= 1

    def test_insufficient_data(self):
        lows = np.array([10.0] * 50)
        assert compute_rsrs_momentum(lows, 89) == (None, None, None)

    def test_none_input(self):
        assert compute_rsrs_momentum(None, 89) == (None, None, None)

    def test_r2_bounds(self):
        lows = np.array([10.0 + i * 0.01 + np.sin(i / 5) * 0.05 for i in range(100)])
        _, _, r2 = compute_rsrs_momentum(lows, 89)
        assert 0 <= r2 <= 1


@pytest.mark.p1
class TestComputeVolTurnoverKernel:
    def _df(self, n=250):
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        turnover = np.abs(rng.normal(1.0, 0.2, n))
        return pd.DataFrame({"close": close, "turnover": turnover})

    def test_returns_kernel(self):
        df = self._df()
        latest, series = compute_vol_turnover_kernel(df, lookback=200)
        assert latest is not None
        assert isinstance(latest, float)

    def test_empty_df(self):
        assert compute_vol_turnover_kernel(pd.DataFrame(), lookback=200) == (None, None)

    def test_insufficient_data(self):
        df = self._df(n=50)
        assert compute_vol_turnover_kernel(df, lookback=200) == (None, None)

    def test_missing_turnover(self):
        df = pd.DataFrame({"close": np.arange(250) + 100.0})
        assert compute_vol_turnover_kernel(df, lookback=200) == (None, None)
