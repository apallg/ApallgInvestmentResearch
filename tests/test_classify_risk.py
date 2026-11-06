"""
classify_risk_action 阶梯止盈止损纯函数测试。

回归重点: 之前 L2 _check_risk_signal 和 L3 _classify_by_pnl 是两套实现,
统一后必须保证四档判断 + already_half 分支一致。
"""
import pytest

from core.domain.portfolio.allocator import (
    classify_risk_action, classify_dynamic_trailing, compute_atr, classify_atr_stop,
)


D = (-0.05, -0.10, 0.15, 0.30)  # warn, full, half, full


@pytest.mark.p0
class TestClassifyRisk:
    def test_stop_loss_full(self):
        action, reason = classify_risk_action(-0.15, *D)
        assert action == "full"
        assert "止损清仓" in reason

    def test_take_profit_full(self):
        action, reason = classify_risk_action(0.35, *D)
        assert action == "full"
        assert "止盈清仓" in reason

    def test_stop_loss_half(self):
        action, _ = classify_risk_action(-0.07, *D)
        assert action == "half"

    def test_take_profit_half(self):
        action, _ = classify_risk_action(0.20, *D)
        assert action == "half"

    def test_no_action(self):
        action, _ = classify_risk_action(0.01, *D)
        assert action is None

    def test_boundary_exact_stop_full(self):
        # 恰好触及 full 阈值
        action, _ = classify_risk_action(-0.10, *D)
        assert action == "full"

    def test_boundary_just_above_stop_full(self):
        # 略高于 full 阈值, 落入 warn 档
        action, _ = classify_risk_action(-0.09, *D)
        assert action == "half"


@pytest.mark.p0
class TestAlreadyHalf:
    """already_half 回归: 已减半后不再触发减半, 只判断清仓档"""

    def test_already_half_stop_full(self):
        action, _ = classify_risk_action(-0.15, *D, already_half=True)
        assert action == "full"

    def test_already_half_no_reduce(self):
        # 已减半, 浮亏在 warn 档(-0.07), 不应再触发减半
        action, _ = classify_risk_action(-0.07, *D, already_half=True)
        assert action is None

    def test_already_half_take_profit_full(self):
        action, _ = classify_risk_action(0.35, *D, already_half=True)
        assert action == "full"

    def test_already_half_no_take_profit_half(self):
        # 已减半, 浮盈在 half 档(0.20), 不应再触发减半
        action, _ = classify_risk_action(0.20, *D, already_half=True)
        assert action is None


@pytest.mark.p0
class TestDynamicTrailing:
    """classify_dynamic_trailing 动态回撤止损纯函数测试"""

    def test_full_stop(self):
        action, reason = classify_dynamic_trailing([10.0, 11.0, 12.0], 11.3, trail_full=0.05)
        assert action == "full"
        assert "清仓" in reason

    def test_half_stop(self):
        action, _ = classify_dynamic_trailing([10.0, 11.0, 12.0], 11.5, trail_full=0.05)
        assert action == "half"

    def test_no_action(self):
        action, _ = classify_dynamic_trailing([10.0, 11.0, 12.0], 11.8, trail_full=0.05)
        assert action is None

    def test_already_half_no_reduce(self):
        action, _ = classify_dynamic_trailing([10.0, 11.0, 12.0], 11.5, trail_full=0.05, already_half=True)
        assert action is None

    def test_lookback_window(self):
        # 前一个 100 在 lookback=3 窗口外, 最高应取 3 而非 100
        action, _ = classify_dynamic_trailing([100.0, 1.0, 2.0, 3.0], 2.8, lookback=3, trail_full=0.05)
        assert action == "full"

    def test_empty_series(self):
        assert classify_dynamic_trailing([], 10.0)[0] is None
        assert classify_dynamic_trailing(None, 10.0)[0] is None


@pytest.mark.p0
class TestComputeAtr:
    def test_constant_range(self):
        high = list(range(12, 27))
        low = list(range(10, 25))
        close = list(range(11, 26))
        assert compute_atr(high, low, close, period=14) == pytest.approx(2.0)

    def test_insufficient_data(self):
        assert compute_atr([10], [8], [9], period=14) is None
        assert compute_atr([10], [8], [9], period=0) is None

    def test_ignores_nan(self):
        high = [10, 12, float("nan"), 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
        low = [8, 10, 12, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
        close = [9, 11, 13, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]
        assert compute_atr(high, low, close, period=14) is not None

    def test_wilder_smoothing(self):
        high = [11, 12, 13, 17]
        low = [9, 10, 11, 13]
        close = [10, 11, 12, 15]
        assert compute_atr(high, low, close, period=2) == pytest.approx(3.5)


@pytest.mark.p0
class TestClassifyAtrStop:
    def test_trigger(self):
        action, reason = classify_atr_stop(100.0, 5.0, 2.0, 89.5)
        assert action == "full"
        assert "ATR止损" in reason

    def test_no_trigger(self):
        action, _ = classify_atr_stop(100.0, 5.0, 2.0, 90.5)
        assert action is None

    def test_invalid_entry(self):
        assert classify_atr_stop(0, 5.0, 2.0, 90.0)[0] is None
        assert classify_atr_stop(None, 5.0, 2.0, 90.0)[0] is None

    def test_invalid_atr(self):
        assert classify_atr_stop(100.0, 0, 2.0, 90.0)[0] is None
        assert classify_atr_stop(100.0, None, 2.0, 90.0)[0] is None
        assert classify_atr_stop(100.0, 5.0, 0, 90.0)[0] is None

    def test_invalid_low(self):
        assert classify_atr_stop(100.0, 5.0, 2.0, None)[0] is None

    def test_boundary(self):
        action, _ = classify_atr_stop(100.0, 5.0, 2.0, 90.0)
        assert action == "full"
