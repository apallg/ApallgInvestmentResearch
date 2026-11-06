"""
一致性风控单元测试 — 踩踏检测状态机、一致性计算、边界条件
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from core.domain.portfolio.consistency import (
    _compute_consistency, check_panic, reset_state,
    _load_state, _save_state, _MIN_STOCKS, _MIN_HISTORY, _STATE_FILE,
)


class TestComputeConsistency:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        values = rng.normal(-1.5, 1.0, 500)
        med, cons = _compute_consistency(values)
        assert med is not None
        assert -3 < med < 0
        assert 0.5 < cons < 0.8

    @pytest.mark.unit
    @pytest.mark.p0
    def test_all_same_values(self):
        values = [-2.0] * 200
        med, cons = _compute_consistency(values)
        assert med == -2.0
        assert cons == 1.0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_uniform_distribution(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(-3, 3, 500).tolist()
        med, cons = _compute_consistency(values)
        assert med is not None
        assert 0.55 < cons < 0.65

    @pytest.mark.unit
    @pytest.mark.p0
    def test_too_few_stocks_returns_none(self):
        values = [1.0] * (_MIN_STOCKS - 1)
        med, cons = _compute_consistency(values)
        assert med is None
        assert cons is None

    @pytest.mark.unit
    @pytest.mark.p1
    def test_exactly_min_stocks(self):
        rng = np.random.default_rng(42)
        values = rng.normal(-1, 1, _MIN_STOCKS).tolist()
        med, cons = _compute_consistency(values)
        assert med is not None
        assert cons is not None

    @pytest.mark.unit
    @pytest.mark.p1
    def test_extreme_outliers(self):
        values = [-10.0] * 50 + [-9.8] * 50 + [-0.5] * 50 + [3.0] * 50 + [9.9] * 50
        med, cons = _compute_consistency(values)
        assert med is not None
        assert cons < 0.8

    @pytest.mark.unit
    @pytest.mark.p2
    def test_nan_values_filtered(self):
        arr = np.array([float('nan'), -2.0, float('nan'), -1.5, -2.1] + [-1.8] * 150)
        med, cons = _compute_consistency(arr.tolist())
        assert med is not None
        assert cons is not None

    @pytest.mark.unit
    @pytest.mark.p2
    def test_zero_std(self):
        med, cons = _compute_consistency([-2.0] * 200)
        assert med == -2.0
        assert cons == 1.0


class TestPanicDetection:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_no_panic_when_bull_market(self):
        reset_state()
        with patch("core.domain.portfolio.consistency._is_bear_market", return_value=False):
            result = check_panic(date="2025-06-15")
            assert result is False
            state = _load_state()
            assert state.get("panic") is False

    @pytest.mark.unit
    @pytest.mark.p0
    def test_reset_panic_to_normal_in_bull(self):
        _save_state({"panic": True, "last_updated": "2025-06-14"})
        with patch("core.domain.portfolio.consistency._is_bear_market", return_value=False):
            result = check_panic(date="2025-06-15")
            assert result is False
            state = _load_state()
            assert state.get("panic") is False

    @pytest.mark.unit
    @pytest.mark.p0
    def test_insufficient_history_returns_current_state(self):
        _save_state({"panic": False, "last_updated": "2025-06-01"})
        with patch("core.domain.portfolio.consistency._is_bear_market", return_value=True):
            with patch("core.domain.portfolio.consistency._get_consistency_series", return_value=[]):
                result = check_panic(date="2025-06-15")
                assert result is False

    @pytest.mark.unit
    @pytest.mark.p0
    def test_insufficient_history_stays_panic_if_was_panic(self):
        _save_state({"panic": True, "last_updated": "2025-06-01"})
        with patch("core.domain.portfolio.consistency._is_bear_market", return_value=True):
            with patch("core.domain.portfolio.consistency._get_consistency_series", return_value=[]):
                result = check_panic(date="2025-06-15")
                assert result is True

    @pytest.mark.unit
    @pytest.mark.p1
    def test_panic_maintained_if_no_recovery(self):
        _save_state({"panic": True, "last_updated": "2025-06-14"})
        series = [
            ("2025-06-01", -1.0, 0.60), ("2025-06-02", -0.8, 0.58),
            ("2025-06-03", -0.5, 0.62), ("2025-06-04", -1.2, 0.59),
        ] * 10  # > MIN_HISTORY
        # last entry: med=-1.2, cons=0.59, cons_mean≈0.60
        # Not panic trigger (med > -2) and not recovery (med < 2 but cons not >= mean)
        with patch("core.domain.portfolio.consistency._is_bear_market", return_value=True):
            with patch("core.domain.portfolio.consistency._get_consistency_series", return_value=series):
                result = check_panic(date="2025-06-15")
                assert result is True  # was_panic stays

    @pytest.mark.unit
    @pytest.mark.p1
    def test_recovery_signal(self):
        _save_state({"panic": True, "last_updated": "2025-06-14"})
        # 构建 30+ 条记录, 最后一条触发恢复: med > 2, cons >= mean
        series = [
            (f"2025-06-{i:02d}", 0.5, 0.50) for i in range(1, 30)
        ]
        series.append(("2025-06-30", 3.0, 0.95))
        with patch("core.domain.portfolio.consistency._is_bear_market", return_value=True):
            with patch("core.domain.portfolio.consistency._get_consistency_series", return_value=series):
                result = check_panic(date="2025-06-30")
                assert result is False
                state = _load_state()
                assert state.get("panic") is False

    @pytest.mark.unit
    @pytest.mark.p0
    def test_reset_state_function(self):
        _save_state({"panic": True, "last_updated": "2025-01-01"})
        reset_state()
        state = _load_state()
        assert state.get("panic") is False


class TestStateFile:

    @pytest.mark.unit
    @pytest.mark.p2
    def test_load_default_when_no_file(self):
        with patch.object(type(_STATE_FILE), "exists", return_value=False):
            state = _load_state()
            assert state["panic"] is False
            assert state["last_updated"] is None
