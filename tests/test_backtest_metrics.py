"""compute_backtest_metrics 统一回测绩效纯函数测试"""
import pytest

from core.domain.backtest.metrics import compute_backtest_metrics, compute_trade_diagnostics


def _basic_inputs():
    equity = [1.0, 1.02, 1.01, 1.05, 1.03]
    benchmark = [1.0, 1.01, 1.01, 1.03, 1.02]
    dates = ["2025-01-03", "2025-01-10", "2025-01-17", "2025-01-24", "2025-01-31"]
    periods = [{"regime": "steady_up", "direction": "trend_momentum",
                "period_return": 0.02, "timing_ratio": 1.0}] * 4
    return equity, benchmark, dates, periods, []


@pytest.mark.p0
class TestComputeMetrics:
    def test_summary_keys(self):
        result = compute_backtest_metrics(*_basic_inputs(), freq="week")
        s = result["summary"]
        for key in ("annual_return", "sharpe", "sortino", "max_drawdown",
                    "win_rate", "profit_factor", "n_periods", "n_trades"):
            assert key in s

    def test_equity_curve_length(self):
        result = compute_backtest_metrics(*_basic_inputs(), freq="week")
        assert len(result["equity_curve"]) == 5
        assert len(result["benchmark_curve"]) == 5

    def test_max_drawdown_negative(self):
        result = compute_backtest_metrics(*_basic_inputs(), freq="week")
        # 回撤应该是负值 (净值从 1.05 回落到 1.03)
        assert result["summary"]["max_drawdown"] <= 0

    def test_regime_breakdown(self):
        result = compute_backtest_metrics(*_basic_inputs(), freq="week")
        assert "steady_up" in result["regime_breakdown"]
        assert result["regime_breakdown"]["steady_up"]["count"] == 4

    def test_direction_distribution(self):
        result = compute_backtest_metrics(*_basic_inputs(), freq="week")
        assert result["direction_distribution"]["trend_momentum"] == 4

    def test_empty_equity(self):
        result = compute_backtest_metrics([], [], [], [], [], freq="week")
        # 空输入不崩溃, 返回结构完整
        assert "summary" in result

    def test_profit_factor_cap(self):
        # 无 cap vs 有 cap 的行为
        equity = [1.0, 1.5, 2.0, 3.0]  # 高盈亏比
        benchmark = [1.0, 1.0, 1.0, 1.0]
        dates = ["2025-01-03", "2025-01-10", "2025-01-17", "2025-01-24"]
        periods = [{"regime": "steady_up", "direction": "trend",
                    "period_return": 0.1, "timing_ratio": 1.0}] * 3
        no_cap = compute_backtest_metrics(equity, benchmark, dates, periods, [], "week")
        capped = compute_backtest_metrics(equity, benchmark, dates, periods, [], "week",
                                          profit_factor_cap=999.0)
        # cap 会限制 profit_factor 到上限, 所以 capped <= no_cap
        assert capped["summary"]["profit_factor"] <= no_cap["summary"]["profit_factor"]

    def test_trade_diagnostics(self):
        trades = [
            {"action": "buy", "value": 0.2, "cost": 0.00008, "reason": "新进"},
            {"action": "sell", "value": 0.18, "cost": 0.00016,
             "reason": "ATR止损清仓", "realized_return": -0.1,
             "gap_slippage": -0.02},
            {"action": "sell", "value": 0.3, "cost": 0.00027,
             "reason": "止盈清仓", "realized_return": 0.3},
        ]
        result = compute_trade_diagnostics(trades)
        assert result["buy_count"] == 1
        assert result["sell_count"] == 2
        assert result["stop_loss_count"] == 1
        assert result["stop_loss_full_count"] == 1
        assert result["stop_loss_half_count"] == 0
        assert result["atr_stop_count"] == 1
        assert result["take_profit_count"] == 1
        assert result["gap_through_count"] == 1
        assert result["avg_stop_return"] == pytest.approx(-0.1)
        assert result["avg_gap_slippage"] == pytest.approx(-0.02)
        assert result["traded_notional"] == pytest.approx(0.68)
        assert result["transaction_cost"] == pytest.approx(0.00051)
