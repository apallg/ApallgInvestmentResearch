"""方向回测主循环核心单元集成测试 — mock 数据驱动 _simulate_period_fast。

覆盖目标: 方向回测主循环的"模拟当期"核心步骤 (止损止盈撮合/换手约束/买入调仓/净值迭代)。
主循环 run() 需真实 qlib/DB/网络 (全 mock 成本过高), 此测试用内存数据驱动
主循环最重的 _simulate_period_fast, 抓撮合与调仓逻辑回归。
"""
from datetime import date

import pandas as pd
import pytest

from core.domain.backtest.engine import DirectionBacktestRunner
from core.domain.portfolio.allocator import AllocationConfig, PositionAllocator


def _make_runner():
    r = DirectionBacktestRunner.__new__(DirectionBacktestRunner)
    r._stop_source = "fixed"
    r._sell_cost_rate = 0.0
    r._buy_cost_rate = 0.0003
    r._cash_yield = False
    r._cooldown_stop = 2
    r._cooldown_profit = 1
    r._take_profit_half = 0.15
    r._take_profit_full = 0.30
    r._dynamic_lookback = 60
    r._dynamic_trail = 0.10
    r._limit_up_cache = {}
    r._limit_down_cache = {}
    cfg = AllocationConfig()
    cfg.stop_loss_warn = -0.05
    cfg.stop_loss_full = -0.10
    cfg.take_profit_half = 0.15
    cfg.take_profit_full = 0.30
    r._allocator = PositionAllocator(cfg)
    return r


@pytest.mark.p0
def test_simulate_period_executes_stop_loss_and_buy():
    runner = _make_runner()
    # A 深跌触发 full 止损; B 正常持有; C 新买入
    price_map = {
        "A": pd.DataFrame([
            {"trade_date": "2025-01-02", "open": 10.0, "high": 10.2, "low": 8.5, "close": 9.0,
             "volume": 1e6, "amount": 1e8},
            {"trade_date": "2025-01-03", "open": 9.0, "high": 9.2, "low": 8.7, "close": 8.8,
             "volume": 1e6, "amount": 1e8},
        ]),
        "B": pd.DataFrame([
            {"trade_date": "2025-01-02", "open": 20.0, "high": 20.4, "low": 19.8, "close": 20.1,
             "volume": 1e6, "amount": 1e8},
            {"trade_date": "2025-01-03", "open": 20.1, "high": 20.5, "low": 19.9, "close": 20.2,
             "volume": 1e6, "amount": 1e8},
        ]),
        "C": pd.DataFrame([
            {"trade_date": "2025-01-02", "open": 30.0, "high": 30.4, "low": 29.8, "close": 30.1,
             "volume": 1e6, "amount": 1e8},
            {"trade_date": "2025-01-03", "open": 30.1, "high": 30.5, "low": 29.9, "close": 30.2,
             "volume": 1e6, "amount": 1e8},
        ]),
    }
    holdings = {"A": 1000.0, "B": 2000.0}
    entry_prices = {"A": 10.0, "B": 20.0}
    entry_dates = {"A": "2025-01-02", "B": "2025-01-02"}

    total_end, new_holdings, _, _, cash, _, _, trades = runner._simulate_period_fast(
        holdings=holdings, entry_dates=entry_dates, entry_prices=entry_prices,
        cash=0.0, cooldown={}, half_sold=set(),
        buy_codes=["B", "C"], buy_scores={"B": 0.8, "C": 0.7},
        sell_threshold={"B", "C"},
        curr_date=date(2025, 1, 2), top_n=2, price_cache=price_map,
        timing_ratio=1.0, sector_map={}, next_str="2025-01-03")

    # A 触发止损被清仓 (p_low -15% < -10% full)
    assert "A" not in new_holdings
    # C 是新买入标的
    assert "C" in new_holdings
    # 止损有成交记录
    sell_trades = [t for t in trades if t["action"] == "sell"]
    assert len(sell_trades) >= 1
    # 净值为正且组合未清空
    assert total_end > 0
    assert len(new_holdings) >= 1


@pytest.mark.p0
def test_simulate_period_keeps_holdings_without_triggers():
    """无止损/止盈触发时持仓保留 (仅 B 一只, 无新买入 → 维持)"""
    runner = _make_runner()
    price_map = {
        "B": pd.DataFrame([
            {"trade_date": "2025-01-02", "open": 20.0, "high": 20.4, "low": 19.8, "close": 20.1,
             "volume": 1e6, "amount": 1e8},
            {"trade_date": "2025-01-03", "open": 20.1, "high": 20.5, "low": 19.9, "close": 20.2,
             "volume": 1e6, "amount": 1e8},
        ]),
    }
    total_end, new_holdings, _, _, _, _, _, trades = runner._simulate_period_fast(
        holdings={"B": 2000.0}, entry_dates={"B": "2025-01-02"}, entry_prices={"B": 20.0},
        cash=0.0, cooldown={}, half_sold=set(),
        buy_codes=["B"], buy_scores={"B": 0.8}, sell_threshold={"B"},
        curr_date=date(2025, 1, 2), top_n=1, price_cache=price_map,
        timing_ratio=1.0, sector_map={}, next_str="2025-01-03")

    assert "B" in new_holdings
    assert total_end > 0


@pytest.mark.p0
def test_simulate_period_does_not_buy_without_t1_open():
    runner = _make_runner()
    price_map = {
        "C": pd.DataFrame([{
            "trade_date": "2025-01-02", "open": 30.0, "high": 30.4,
            "low": 29.8, "close": 30.1, "volume": 1e6, "amount": 1e8,
        }]),
    }

    total_end, holdings, entry_dates, entry_prices, cash, _, _, trades = runner._simulate_period_fast(
        holdings={}, entry_dates={}, entry_prices={}, cash=1.0, cooldown={}, half_sold=set(),
        buy_codes=["C"], buy_scores={"C": 0.8}, sell_threshold={"C"},
        curr_date=date(2025, 1, 2), top_n=1, price_cache=price_map,
        timing_ratio=1.0, sector_map={}, next_str="2025-01-03")

    assert holdings == {}
    assert entry_dates == {}
    assert entry_prices == {}
    assert trades == []
    assert cash == pytest.approx(1.0)
    assert total_end == pytest.approx(1.0)
