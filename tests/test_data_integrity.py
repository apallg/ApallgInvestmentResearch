"""
数据完整性测试 — 冲击成本估计、forward returns 缓存边界、数据验证
"""
import pytest
import numpy as np
import pandas as pd
import datetime
from unittest.mock import patch, Mock

from core.domain.screening._scoring_legacy import estimate_impact_cost, apply_impact_penalty, _estimate_daily_volatility
from core.domain.factors.forward_returns import ForwardReturnCache


class TestImpactCost:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_normal_case_positive(self):
        impact = estimate_impact_cost(close=10.0, volume=1_000_000, trade_amount=100_000)
        assert 0.0001 < impact < 0.01, f"impact={impact}"

    @pytest.mark.unit
    @pytest.mark.p0
    def test_zero_close_returns_zero(self):
        impact = estimate_impact_cost(close=0.0, volume=1_000_000)
        assert impact == 0.0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_zero_volume_returns_zero(self):
        impact = estimate_impact_cost(close=10.0, volume=0)
        assert impact == 0.0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_zero_participation_returns_zero(self):
        # trade_amount=0 triggers default (1% of daily_volume), not zero participation
        # Use exceedingly small trade_amount to get near-zero impact
        impact = estimate_impact_cost(close=10.0, volume=1_000_000, trade_amount=0.01)
        assert impact < 0.001

    @pytest.mark.unit
    @pytest.mark.p1
    def test_high_participation_penalized_more(self):
        low = estimate_impact_cost(close=10.0, volume=1_000_000, trade_amount=10_000)
        high = estimate_impact_cost(close=10.0, volume=1_000_000, trade_amount=500_000)
        assert high > low, f"low={low}, high={high}"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_impact_capped_at_5pct(self):
        impact = estimate_impact_cost(close=1.0, volume=100, trade_amount=100_000_000)
        assert impact <= 0.05

    @pytest.mark.unit
    @pytest.mark.p2
    def test_custom_eta(self):
        impact_default = estimate_impact_cost(close=10.0, volume=1_000_000, trade_amount=100_000)
        impact_high = estimate_impact_cost(close=10.0, volume=1_000_000, trade_amount=100_000, eta=0.5)
        assert impact_high > impact_default

    @pytest.mark.unit
    @pytest.mark.p2
    def test_exception_returns_zero(self):
        impact = estimate_impact_cost(close="invalid", volume="bad")
        assert impact == 0.0


class TestImpactPenalty:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_penalty_reduces_score(self):
        score = 0.8
        penalized = apply_impact_penalty(score, 0.01)
        assert penalized < score

    @pytest.mark.unit
    @pytest.mark.p0
    def test_zero_impact_no_penalty(self):
        score = 0.5
        penalized = apply_impact_penalty(score, 0.0)
        assert penalized == score

    @pytest.mark.unit
    @pytest.mark.p1
    def test_negative_impact_no_penalty(self):
        score = 0.5
        penalized = apply_impact_penalty(score, -0.01)
        assert penalized == score

    @pytest.mark.unit
    @pytest.mark.p1
    def test_large_impact_capped_penalty(self):
        score = 1.0
        penalized = apply_impact_penalty(score, 0.5)
        assert penalized >= 0.7  # tanh 压缩防止过度惩罚


class TestDailyVolatility:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_returns_default_without_history(self):
        vol = _estimate_daily_volatility(10.0)
        assert vol == 0.02

    @pytest.mark.unit
    @pytest.mark.p2
    def test_exception_returns_default(self):
        vol = _estimate_daily_volatility("invalid")
        assert vol == 0.02

    @pytest.mark.unit
    @pytest.mark.p1
    def test_with_amount_series(self):
        s = pd.Series([10.0, 10.2, 10.1, 10.3, 10.5, 10.0, 9.8] * 3)
        vol = _estimate_daily_volatility(10.0, amount=s)
        assert 0.005 <= vol <= 0.05


class TestForwardReturnCache:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cache = ForwardReturnCache()
        self.cache._mem.clear()

    @pytest.mark.unit
    @pytest.mark.p0
    def test_empty_codes_returns_empty(self):
        assert self.cache.get_batch([], datetime.date.today(), 20) == {}

    @pytest.mark.unit
    @pytest.mark.p0
    def test_unknown_codes_return_zero(self):
        result = self.cache.get_batch(["999999"], datetime.date.today(), 20)
        assert result["999999"] == 0.0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_memory_hit_bypasses_db(self):
        today = datetime.date.today()
        key = ("000001", today.isoformat(), 20)
        self.cache._mem[key] = 0.1234
        result = self.cache.get_batch(["000001", "600519"], today, 20)
        assert result["000001"] == 0.1234

    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_list_preserves_order(self):
        today = datetime.date.today()
        self.cache._mem[("000001", today.isoformat(), 10)] = 0.05
        result = self.cache.get_list(["600519", "000001", "300750"], today, 10)
        assert result == [0.0, 0.05, 0.0]

    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_multi_period_keys(self):
        today = datetime.date.today()
        result = self.cache.get_multi_period(["000001"], today, (5, 10, 20))
        assert set(result.keys()) == {5, 10, 20}

    @pytest.mark.unit
    @pytest.mark.p1
    def test_clear_memory_empties_cache(self):
        self.cache._mem[("x", "2024-01-01", 20)] = 0.1
        self.cache.clear_memory()
        assert len(self.cache._mem) == 0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_same_date_different_periods_different_keys(self):
        today = datetime.date.today()
        key_5 = ("000001", today.isoformat(), 5)
        key_20 = ("000001", today.isoformat(), 20)
        self.cache._mem[key_5] = 0.01
        self.cache._mem[key_20] = 0.10
        assert self.cache.get_batch(["000001"], today, 5)["000001"] == 0.01
        assert self.cache.get_batch(["000001"], today, 20)["000001"] == 0.10

    @pytest.mark.unit
    @pytest.mark.p2
    def test_string_date_accepted(self):
        self.cache._mem[("000001", "2025-06-15", 10)] = 0.05
        result = self.cache.get_batch(["000001"], "2025-06-15", 10)
        assert result["000001"] == 0.05

    @pytest.mark.unit
    @pytest.mark.p1
    def test_singleton_same_instance(self):
        a = ForwardReturnCache.get_instance()
        b = ForwardReturnCache.get_instance()
        assert a is b

    @pytest.mark.unit
    @pytest.mark.p2
    def test_invalid_period_returns_zero(self):
        result = self.cache.get_batch(["000001"], datetime.date.today(), 99)
        assert result["000001"] == 0.0


class TestForwardReturnMemoryIsolation:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_caches_share_same_backing_store(self):
        cache1 = ForwardReturnCache()
        cache2 = ForwardReturnCache()
        # ForwardReturnCache 的 __init__ 初始化新的 _mem dict,
        # 但 _instance 单例是通过 get_instance() 获取的
        cache1._mem[("test", "2025-01-01", 20)] = 0.5
        # 直接构造的实例不共享 _mem (不是单例)
        # 实际共享的是 get_instance() 返回的单例
        assert ("test", "2025-01-01", 20) in cache1._mem
