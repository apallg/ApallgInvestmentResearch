from datetime import date

import pandas as pd
import pytest

from core.domain.backtest.engine import DirectionBacktestRunner
from core.domain.portfolio.allocator import AllocationConfig, PositionAllocator


def make_runner(period=2, mult=2.0):
    runner = DirectionBacktestRunner.__new__(DirectionBacktestRunner)
    runner._atr_period = period
    runner._atr_mult = mult
    runner._cooldown_stop = 2
    runner._cooldown_profit = 1
    runner._take_profit_half = 0.15
    runner._take_profit_full = 0.30
    runner._allocator = PositionAllocator(AllocationConfig())
    return runner


def make_prices(low=91.0, high=106.0):
    return pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "open": [100.0, 102.0, 104.0, 95.0],
        "high": [102.0, 104.0, 106.0, high],
        "low": [98.0, 100.0, 102.0, low],
        "close": [100.0, 102.0, 104.0, 100.0],
    })


@pytest.mark.p0
def test_atr_action_exposes_trigger_price():
    runner = make_runner()
    actions = runner._atr_risk_actions(
        {"000001": 1.0}, {"000001": 100.0}, "2026-01-04",
        {"000001": make_prices()}, set())
    assert actions["000001"].action == "full"
    assert actions["000001"].trigger_price == pytest.approx(92.0)


@pytest.mark.p0
def test_atr_mode_keeps_take_profit():
    runner = make_runner()
    actions = runner._atr_risk_actions(
        {"000001": 1.0}, {"000001": 100.0}, "2026-01-04",
        {"000001": make_prices(low=99.0, high=131.0)}, set())
    assert actions["000001"].action == "full"
    assert "止盈" in actions["000001"].reason
    assert actions["000001"].trigger_price is None


@pytest.mark.p0
def test_atr_cold_start_falls_back_to_fixed():
    runner = make_runner(period=14)
    actions = runner._atr_risk_actions(
        {"000001": 1.0}, {"000001": 100.0}, "2026-01-04",
        {"000001": make_prices(low=89.0)}, set())
    assert actions["000001"].action == "full"
    assert "止损" in actions["000001"].reason
    assert actions["000001"].trigger_price is None


@pytest.mark.p0
@pytest.mark.parametrize(("trigger_open", "expected_cash"), [
    (95.0, 0.92),
    (90.0, 0.90),
])
def test_daily_atr_fill_price(trigger_open, expected_cash):
    runner = make_runner()
    runner._stop_source = "atr"
    runner._sell_cost_rate = 0.0
    runner._limit_down_cache = {}
    runner._index_df = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })
    prices = pd.DataFrame({
        "trade_date": [
            "2025-12-29", "2025-12-30", "2025-12-31",
            "2026-01-01", "2026-01-02", "2026-01-03",
        ],
        "open": [100.0, 102.0, 104.0, 104.0, trigger_open, 91.0],
        "high": [102.0, 104.0, 106.0, 106.0, 93.0, 92.0],
        "low": [98.0, 100.0, 102.0, 102.0, 89.0, 90.0],
        "close": [100.0, 102.0, 104.0, 104.0, 91.0, 91.0],
    })
    holdings, _, _, cash, _, _, trades = runner._daily_risk_check(
        holdings={"000001": 1.0}, entry_dates={"000001": "2026-01-01"},
        entry_prices={"000001": 100.0}, cash=0.0, cooldown={}, half_sold=set(),
        prev_date=date(2026, 1, 1), curr_date=date(2026, 1, 3),
        price_cache={"000001": prices})
    assert holdings == {}
    assert cash == pytest.approx(expected_cash)
    assert len(trades) == 1
    assert trades[0]["realized_return"] == pytest.approx(expected_cash - 1)
    assert trades[0]["gap_slippage"] == pytest.approx(
        min(0.0, expected_cash / 0.92 - 1))


@pytest.mark.p0
def test_daily_atr_does_not_sell_at_limit_down():
    runner = make_runner()
    runner._stop_source = "atr"
    runner._sell_cost_rate = 0.0
    runner._limit_down_cache = {"000001": {"20260102": True}}
    runner._index_df = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })
    prices = pd.DataFrame({
        "trade_date": [
            "2025-12-29", "2025-12-30", "2025-12-31",
            "2026-01-01", "2026-01-02", "2026-01-03",
        ],
        "open": [100.0, 102.0, 104.0, 104.0, 90.0, 91.0],
        "high": [102.0, 104.0, 106.0, 106.0, 93.0, 92.0],
        "low": [98.0, 100.0, 102.0, 102.0, 89.0, 90.0],
        "close": [100.0, 102.0, 104.0, 104.0, 91.0, 91.0],
    })
    holdings, _, _, cash, _, _, trades = runner._daily_risk_check(
        holdings={"000001": 1.0}, entry_dates={"000001": "2026-01-01"},
        entry_prices={"000001": 100.0}, cash=0.0, cooldown={}, half_sold=set(),
        prev_date=date(2026, 1, 1), curr_date=date(2026, 1, 3),
        price_cache={"000001": prices})
    assert holdings == {"000001": 1.0}
    assert cash == 0.0
    assert trades == []


@pytest.mark.p0
def test_daily_risk_check_keeps_entry_dates_when_no_trading_days():
    runner = make_runner()
    runner._index_df = pd.DataFrame({"trade_date": ["2026-01-01"]})

    entry_dates = {"000001": "2026-01-01"}
    result = runner._daily_risk_check(
        holdings={"000001": 1.0}, entry_dates=entry_dates,
        entry_prices={"000001": 100.0}, cash=0.0, cooldown={}, half_sold=set(),
        prev_date=date(2026, 1, 1), curr_date=date(2026, 1, 1),
        price_cache={},
    )

    assert len(result) == 7
    assert result[1] is entry_dates
    assert result[-1] == []
