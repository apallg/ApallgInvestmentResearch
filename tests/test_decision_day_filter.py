import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from core.domain.backtest.engine import DirectionBacktestRunner
from core.domain.regime.types import Regime
from core.domain.screening.exclusion import evaluate_decision_day_filter


def _prices(close=10.0, volume=1000000.0, amount=50000000.0):
    dates = pd.date_range(end="2026-08-03", periods=90, freq="D")
    return pd.DataFrame({
        "trade_date": dates,
        "close": [close] * len(dates),
        "open": [close] * len(dates),
        "volume": [volume] * len(dates),
        "amount": [amount] * len(dates),
        "pct_chg": [0.0] * len(dates),
        "adj_factor": [1.0] * len(dates),
    })


@pytest.mark.p0
def test_decision_day_filter_requires_exact_quote_and_checks_liquidity():
    suspended = _prices().iloc[:-1]
    illiquid = _prices(amount=9000000.0)

    suspended_result = evaluate_decision_day_filter(
        "000001", suspended, datetime.date(2026, 8, 3))
    illiquid_result = evaluate_decision_day_filter(
        "000002", illiquid, datetime.date(2026, 8, 3))

    assert suspended_result["reason"] == "suspended"
    assert illiquid_result["reason"] == "low_liquidity"


@pytest.mark.p0
def test_stable_history_requires_500_observations_before_decision_date():
    insufficient = _prices().iloc[:0]
    insufficient = pd.DataFrame({
        "trade_date": pd.bdate_range(end="2026-08-03", periods=500),
        "close": 10.0,
        "open": 10.0,
        "volume": 1000000.0,
        "amount": 50000000.0,
        "pct_chg": 0.0,
        "adj_factor": 1.0,
    })
    sufficient = pd.concat([
        pd.DataFrame({
            "trade_date": [insufficient["trade_date"].iloc[0] - pd.offsets.BDay(1)],
            "close": [10.0], "open": [10.0], "volume": [1000000.0],
            "amount": [50000000.0], "pct_chg": [0.0], "adj_factor": [1.0],
        }),
        insufficient,
    ], ignore_index=True)

    rejected = evaluate_decision_day_filter(
        "000001", insufficient, datetime.date(2026, 8, 3),
        min_history_observations=500,
        listing_date=datetime.date(2020, 1, 1),
    )
    accepted = evaluate_decision_day_filter(
        "000001", sufficient, datetime.date(2026, 8, 3),
        min_history_observations=500,
        listing_date=datetime.date(2020, 1, 1),
    )

    assert rejected["reason"] == "insufficient_history"
    assert accepted["reason"] is None


@pytest.mark.p0
def test_decision_day_filter_uses_identity_listing_date_not_cached_history_start():
    result = evaluate_decision_day_filter(
        "000001", _prices(), datetime.date(2026, 8, 3),
        listing_date=datetime.date(2026, 7, 1),
    )

    assert result["reason"] == "new_listing"


@pytest.mark.p0
def test_decision_day_filter_rejects_unknown_listing_date():
    result = evaluate_decision_day_filter(
        "000001", _prices(), datetime.date(2026, 8, 3),
    )

    assert result["reason"] == "identity_unknown"


@pytest.mark.p0
def test_direction_backtest_uses_rank_order_and_refills_target_count(monkeypatch):
    from core.data.repos.stock_identity import StockIdentityRepo

    monkeypatch.setattr(
        StockIdentityRepo, "get_listing_dates",
        classmethod(lambda cls, codes: {
            "000002": datetime.date(2020, 1, 1),
            "000003": datetime.date(2020, 1, 1),
        }),
    )
    runner = DirectionBacktestRunner.__new__(DirectionBacktestRunner)
    runner._latest_adj_factors = {}
    price_cache = {
        "000001": _prices(),
        "000002": _prices(),
        "000003": _prices(),
    }
    sector_map = {
        "000001": {"name": "ST甲"},
        "000002": {"name": "乙"},
        "000003": {"name": "丙"},
    }

    eligible, excluded, examined = runner._filter_decision_day_ranking(
        ["000001", "000002", "000003"], datetime.date(2026, 8, 3), 2,
        price_cache, sector_map,
    )

    assert eligible == ["000002", "000003"]
    assert excluded["st"] == 1
    assert examined == 3


@pytest.mark.p0
def test_formal_backtest_requires_point_in_time_context_for_every_decision_date():
    dates = [datetime.date(2026, 8, 3), datetime.date(2026, 8, 10)]
    memberships = {"2026-08-03": ["000001"]}
    references = {
        "2026-08-03": {"000001": {"name": "甲", "industry": "银行", "source": "pit_import"}},
        "2026-08-10": {"000001": {"name": "甲", "industry": "银行", "source": "pit_import"}},
    }

    with pytest.raises(RuntimeError, match="PIT股票池成员缺少决策日"):
        DirectionBacktestRunner._require_point_in_time_context(
            dates, memberships, references,
        )


@pytest.mark.p0
def test_direction_backtest_materializes_latest_prior_pit_snapshot():
    dates = [datetime.date(2026, 8, 3), datetime.date(2026, 8, 10)]
    memberships = {"2026-08-01": ["000001"]}
    references = {
        "2026-08-01": {
            "000001": {"name": "甲", "industry": "银行", "source": "pit_import"},
        },
    }

    resolved_memberships, resolved_references = (
        DirectionBacktestRunner._materialize_point_in_time_context(
            dates, memberships, references,
        )
    )

    assert list(resolved_memberships) == ["2026-08-03", "2026-08-10"]
    DirectionBacktestRunner._require_point_in_time_context(
        dates, resolved_memberships, resolved_references,
    )


@pytest.mark.p0
def test_direction_backtest_rejects_partial_security_reference_source():
    dates = [datetime.date(2026, 8, 3)]
    memberships = {"2026-08-03": ["000001"]}
    references = {
        "2026-08-03": {
            "000001": {"source": "pit_market_snapshot_partial"},
        },
    }

    with pytest.raises(RuntimeError, match="来源不权威"):
        DirectionBacktestRunner._require_point_in_time_context(
            dates, memberships, references,
        )


@pytest.mark.p0
def test_direction_backtest_forwards_macro_regime_switch(monkeypatch):
    from core.domain.regime import unified

    captured = {}

    def fake_regime(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            regime=Regime.STEADY_UP,
            confidence=0.7,
            sub_scores={},
            details={},
        )

    monkeypatch.setattr(unified, "get_unified_regime", fake_regime)
    runner = DirectionBacktestRunner.__new__(DirectionBacktestRunner)
    runner._crf_trans = None

    runner._classify_regime(
        datetime.date(2026, 8, 3), None, None,
        macro_regime=False,
    )

    assert captured["apply_macro"] is False
