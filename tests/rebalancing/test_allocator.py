"""
统一 PositionAllocator 单元测试.

覆盖: 权重计算、仓位约束、换手约束、止盈止损、Bug修复验证、配置
"""
import pytest
from core.domain.portfolio.allocator import (
    AllocationConfig,
    PositionAllocator,
    build_allocation_config,
    get_regime_stop_params,
    load_regime_stops,
)


def _c(code, score):
    return {"code": code, "score": score}


def _sector_map(mapping):
    return {code: {"industry": ind} for code, ind in mapping.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 权重计算
# ═══════════════════════════════════════════════════════════════════════════

class TestRawWeights:
    def test_score_weighted_basic(self):
        cfg = AllocationConfig(sizing="score_weighted", target_sum=1.0)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.9), _c("B", 0.6), _c("C", 0.3)]
        w = alloc._compute_raw_weights(cands)
        assert len(w) == 3
        assert w["A"] > w["B"] > w["C"]
        assert abs(sum(w.values()) - 1.0) < 0.001

    def test_score_weighted_ratio(self):
        cfg = AllocationConfig(sizing="score_weighted", score_power=2.0, target_sum=1.0)
        alloc = PositionAllocator(cfg)
        cands = [_c("X", 0.8), _c("Y", 0.4)]
        w = alloc._compute_raw_weights(cands)
        ratio = w["X"] / w["Y"]
        assert abs(ratio - 4.0) < 0.01

    def test_equal_weight(self):
        cfg = AllocationConfig(sizing="equal", target_sum=0.95)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.9), _c("B", 0.5), _c("C", 0.2), _c("D", 0.1)]
        w = alloc._compute_raw_weights(cands)
        assert len(w) == 4
        for v in w.values():
            assert abs(v - 0.2375) < 0.01

    def test_proportional_weight(self):
        cfg = AllocationConfig(sizing="proportional", target_sum=1.0)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.6), _c("B", 0.3), _c("C", 0.1)]
        w = alloc._compute_raw_weights(cands)
        assert abs(w["A"] - 0.6) < 0.001
        assert abs(w["B"] - 0.3) < 0.001
        assert abs(w["C"] - 0.1) < 0.001

    def test_zero_scores_fallback(self):
        cfg = AllocationConfig(sizing="score_weighted", target_sum=1.0)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0), _c("B", 0), _c("C", 0)]
        w = alloc._compute_raw_weights(cands)
        assert len(w) == 3
        assert abs(sum(w.values()) - 1.0) < 0.001

    def test_negative_scores_clamped(self):
        cfg = AllocationConfig(sizing="score_weighted", target_sum=1.0)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", -0.5), _c("B", 0.5)]
        w = alloc._compute_raw_weights(cands)
        assert w["A"] > 0
        assert w["B"] > w["A"]


# ═══════════════════════════════════════════════════════════════════════════
# 仓位约束
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionCaps:
    def test_single_cap_reduces_large_weight(self):
        cfg = AllocationConfig(max_single_pct=0.20, max_sector_pct=0.40,
                               target_sum=1.0, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        weights = {"A": 0.50, "B": 0.12, "C": 0.10, "D": 0.10,
                   "E": 0.08, "F": 0.05, "G": 0.05}
        sm = _sector_map({"A": "电子", "B": "医药", "C": "银行",
                          "D": "食品", "E": "地产", "F": "能源", "G": "化工"})
        result = alloc._apply_position_caps(weights, sm)
        assert result["A"] <= 0.20 + 0.005
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_sector_cap_reduces_concentrated_industry(self):
        cfg = AllocationConfig(max_single_pct=0.20, max_sector_pct=0.30,
                               target_sum=1.0, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        weights = {"A": 0.15, "B": 0.15, "C": 0.20, "D": 0.15,
                   "E": 0.13, "F": 0.12, "G": 0.10}
        sm = _sector_map({"A": "电子", "B": "电子", "C": "电子", "D": "电子",
                          "E": "医药", "F": "银行", "G": "食品"})
        result = alloc._apply_position_caps(weights, sm)
        sector_e = sum(v for c, v in result.items()
                       if c in ("A", "B", "C", "D"))
        assert sector_e <= 0.30 + 0.04
        assert abs(sum(result.values()) - 1.0) < 0.01
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_double_cap_with_sufficient_diversity(self):
        cfg = AllocationConfig(max_single_pct=0.20, max_sector_pct=0.35,
                               apply_double_cap=True, target_sum=1.0)
        alloc = PositionAllocator(cfg)
        weights = {"A": 0.40, "B": 0.18, "C": 0.15, "D": 0.12,
                   "E": 0.08, "F": 0.07}
        sm = _sector_map({"A": "电子", "B": "电子", "C": "医药",
                          "D": "银行", "E": "食品", "F": "地产"})
        result = alloc._apply_position_caps(weights, sm)
        result = alloc._normalize_with_timing(result, 1.0)
        result = alloc._apply_position_caps(result, sm)
        assert result["A"] <= 0.20
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_single_floor(self):
        cfg = AllocationConfig(single_floor_pct=0.10, max_single_pct=0.25,
                               target_sum=0.95, max_redistribution_rounds=30)
        alloc = PositionAllocator(cfg)
        weights = {"A": 0.40, "B": 0.20, "C": 0.15, "D": 0.12, "E": 0.08}
        result = alloc._apply_clamp_and_redistribute(weights)
        for v in result.values():
            assert v >= 0.10 - 0.005
            assert v <= 0.25 + 0.005
        assert abs(sum(result.values()) - 0.95) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# 换手约束
# ═══════════════════════════════════════════════════════════════════════════

class TestTurnoverLimit:
    def test_within_limit_unchanged(self):
        cfg = AllocationConfig(max_turnover=0.50)
        alloc = PositionAllocator(cfg)
        target = {"A": 0.30, "B": 0.30, "C": 0.40}
        current = {"A": 0.25, "B": 0.35, "C": 0.40}
        result = alloc._apply_turnover_limit(target, current)
        assert result == target

    def test_exceeds_limit_scales(self):
        cfg = AllocationConfig(max_turnover=0.10)
        alloc = PositionAllocator(cfg)
        target = {"A": 0.60, "B": 0.40}
        current = {"A": 0.10, "B": 0.40}
        result = alloc._apply_turnover_limit(target, current)
        assert result["A"] < 0.60
        assert result["A"] > 0.10

    def test_turnover_scaling_does_not_renormalize_past_budget(self):
        cfg = AllocationConfig(max_single_pct=0.20, max_turnover=0.10,
                               turnover_cap_single_at_end=True)
        alloc = PositionAllocator(cfg)
        target = {"A": 0.80, "B": 0.20}
        current = {"A": 0.20, "B": 0.80}
        result = alloc._apply_turnover_limit(target, current)
        assert alloc.compute_turnover(result, current) == pytest.approx(0.10)

    def test_cash_funded_buys_respect_one_way_budget(self):
        cfg = AllocationConfig(max_turnover=0.30)
        alloc = PositionAllocator(cfg)
        target = {f"S{i}": 0.19 for i in range(5)}
        current = {"OLD": 0.01}
        result = alloc._apply_turnover_limit(target, current)
        buys = sum(max(0.0, result.get(code, 0) - current.get(code, 0)) for code in result)
        assert buys == pytest.approx(0.30)
        assert alloc.compute_turnover(result, current) == pytest.approx(0.30)

    def test_filters_below_threshold(self):
        cfg = AllocationConfig(min_weight_threshold=0.01, max_turnover=0.10)
        alloc = PositionAllocator(cfg)
        target = {"A": 0.80, "B": 0.20}
        current = {"A": 0.70, "B": 0.30}
        result = alloc._apply_turnover_limit(target, current)
        for v in result.values():
            assert v > 0.005

    def test_short_circuit_empty_current(self):
        cfg = AllocationConfig(max_turnover=0.30)
        alloc = PositionAllocator(cfg)
        target = {"A": 1.0}
        result = alloc._apply_turnover_limit(target, {})
        assert result == target

    def test_short_circuit_empty_target(self):
        cfg = AllocationConfig(max_turnover=0.30)
        alloc = PositionAllocator(cfg)
        result = alloc._apply_turnover_limit({}, {"A": 0.5})
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# 止盈止损
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyRisk:
    def _cfg(self):
        return AllocationConfig(
            stop_loss_warn=-0.05, stop_loss_full=-0.10,
            take_profit_half=0.15, take_profit_full=0.30,
            cooldown_stop_periods=2, cooldown_profit_periods=1,
        )

    def test_stop_loss_full_intraday(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "A", "entry_price": 100, "p_low": 88, "p_high": 102}]
        actions = alloc.classify_risk(info, use_intraday_low_high=True)
        assert "A" in actions
        assert actions["A"].action == "full"
        assert "止损" in actions["A"].reason

    def test_stop_loss_warn_intraday(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "B", "entry_price": 100, "p_low": 94, "p_high": 101}]
        actions = alloc.classify_risk(info, use_intraday_low_high=True)
        assert "B" in actions
        assert actions["B"].action == "half"

    def test_take_profit_full_intraday(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "C", "entry_price": 100, "p_low": 98, "p_high": 132}]
        actions = alloc.classify_risk(info, use_intraday_low_high=True)
        assert "C" in actions
        assert actions["C"].action == "full"
        assert "止盈" in actions["C"].reason

    def test_take_profit_half_intraday(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "D", "entry_price": 100, "p_low": 98, "p_high": 118}]
        actions = alloc.classify_risk(info, use_intraday_low_high=True)
        assert "D" in actions
        assert actions["D"].action == "half"

    def test_half_sold_skips_warn(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "E", "entry_price": 100, "p_low": 94, "p_high": 101}]
        actions = alloc.classify_risk(info, half_sold={"E"}, use_intraday_low_high=True)
        assert "E" not in actions

    def test_half_sold_triggers_full(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "F", "entry_price": 100, "p_low": 88, "p_high": 102}]
        actions = alloc.classify_risk(info, half_sold={"F"}, use_intraday_low_high=True)
        assert "F" in actions
        assert actions["F"].action == "full"

    def test_no_action_normal(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "G", "entry_price": 100, "p_low": 97, "p_high": 105}]
        actions = alloc.classify_risk(info, use_intraday_low_high=True)
        assert "G" not in actions

    def test_classify_by_pnl_stop_loss(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "H", "pnl_pct": -0.12}]
        actions = alloc.classify_risk(info, use_intraday_low_high=False)
        assert "H" in actions
        assert actions["H"].action == "full"

    def test_classify_by_pnl_take_profit_half(self):
        alloc = PositionAllocator(self._cfg())
        info = [{"code": "I", "pnl_pct": 0.20}]
        actions = alloc.classify_risk(info, use_intraday_low_high=False)
        assert "I" in actions
        assert actions["I"].action == "half"


# ═══════════════════════════════════════════════════════════════════════════
# Bug修复验证
# ═══════════════════════════════════════════════════════════════════════════

class TestBugFixes:
    def test_timing_ratio_normalized_correctly(self):
        cfg = AllocationConfig(target_sum=1.0, max_single_pct=0.40,
                               max_sector_pct=0.60, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.9), _c("B", 0.6), _c("C", 0.3)]
        w = alloc._compute_raw_weights(cands)
        w = alloc._apply_position_caps(w, None)
        w = alloc._normalize_with_timing(w, 0.5)
        assert abs(sum(w.values()) - 0.5) < 0.001

    def test_allocate_does_not_double_apply_timing(self):
        cfg = AllocationConfig(target_sum=1.0, max_single_pct=0.40,
                               max_sector_pct=0.60, max_turnover=1.0,
                               apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.9), _c("B", 0.6), _c("C", 0.3)]
        result = alloc.allocate(cands, {}, timing_ratio=0.5)
        assert abs(sum(result.target_weights.values()) - 0.5) < 0.001

    def test_turnover_cap_is_exact_when_single_cap_flag_enabled(self):
        cfg = AllocationConfig(max_single_pct=0.20, max_turnover=0.10,
                               turnover_cap_single_at_end=True)
        alloc = PositionAllocator(cfg)
        target = {"A": 0.80, "B": 0.20}
        current = {"A": 0.20, "B": 0.80}
        result = alloc._apply_turnover_limit(target, current)
        assert alloc.compute_turnover(result, current) == pytest.approx(0.10)

    def test_no_cap_when_disabled(self):
        cfg = AllocationConfig(max_single_pct=0.20, max_turnover=0.10,
                               turnover_cap_single_at_end=False)
        alloc = PositionAllocator(cfg)
        target = {"A": 0.80, "B": 0.20}
        current = {"A": 0.20, "B": 0.80}
        result = alloc._apply_turnover_limit(target, current)
        assert alloc.compute_turnover(result, current) == pytest.approx(0.10)


# ═══════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_build_allocation_config_defaults(self):
        cfg = build_allocation_config("week")
        assert cfg.max_single_pct == 0.20
        assert cfg.max_sector_pct == 0.30
        assert cfg.max_turnover == 0.30
        assert cfg.frequency_days == 7
        assert cfg.apply_double_cap is True

    def test_build_allocation_config_override(self):
        cfg = build_allocation_config("week", max_single_pct=0.15, max_turnover=0.40)
        assert cfg.max_single_pct == 0.15
        assert cfg.max_turnover == 0.40

    def test_build_allocation_config_freq_day(self):
        cfg = build_allocation_config("day")
        assert cfg.frequency_days == 1
        assert cfg.cooldown_stop_periods == 10

    def test_allocation_config_buy_cost_rate(self):
        cfg = AllocationConfig(commission_rate=0.0003, slippage_rate=0.0001)
        assert cfg.buy_cost_rate == pytest.approx(0.0004)

    def test_allocation_config_sell_cost_rate(self):
        cfg = AllocationConfig(commission_rate=0.0003, stamp_duty=0.0005,
                               slippage_rate=0.0001)
        assert cfg.sell_cost_rate == pytest.approx(0.0009)


class TestRegimeStops:
    def test_load_regime_stops(self):
        stops = load_regime_stops()
        assert isinstance(stops, dict)
        assert 0 in stops
        assert "warn" in stops[0]

    def test_get_regime_stop_params_default(self):
        params = get_regime_stop_params()
        assert params["stop_loss_warn"] < 0
        assert params["stop_loss_full"] < params["stop_loss_warn"]
        assert params["take_profit_half"] > 0
        assert params["take_profit_full"] > params["take_profit_half"]

    def test_get_regime_stop_params_user_override(self):
        overrides = {"stop_loss_warn": -0.03, "take_profit_full": 0.20}
        params = get_regime_stop_params(user_overrides=overrides)
        assert params["stop_loss_warn"] == -0.03
        assert params["take_profit_full"] == 0.20
        assert params["source"] == "user"


# ═══════════════════════════════════════════════════════════════════════════
# 集成: allocate() 完整流程
# ═══════════════════════════════════════════════════════════════════════════

class TestAllocate:
    def test_full_pipeline_backtest_style(self):
        cfg = AllocationConfig(
            sizing="score_weighted", score_key="score",
            max_single_pct=0.20, max_sector_pct=0.30,
            max_turnover=0.30, target_sum=1.0,
            apply_double_cap=True, turnover_cap_single_at_end=True,
        )
        alloc = PositionAllocator(cfg)
        cands = [
            {"code": "A", "score": 0.9},
            {"code": "B", "score": 0.8},
            {"code": "C", "score": 0.7},
            {"code": "D", "score": 0.6},
            {"code": "E", "score": 0.5},
        ]
        sm = _sector_map({"A": "电子", "B": "电子", "C": "电子",
                          "D": "医药", "E": "医药"})
        current = {}
        result = alloc.allocate(cands, current, sm, timing_ratio=1.0)
        assert len(result.target_weights) > 0
        assert abs(sum(result.target_weights.values()) - 1.0) < 0.01

    def test_full_pipeline_live_style(self):
        cfg = AllocationConfig(
            sizing="score_weighted", score_key="qlib_score",
            max_single_pct=0.20, max_sector_pct=0.30,
            single_floor_pct=0.10, target_sum=0.95,
            impact_cost_lambda=0.3, max_redistribution_rounds=30,
        )
        alloc = PositionAllocator(cfg)
        cands = [
            {"code": "A", "qlib_score": 0.9},
            {"code": "B", "qlib_score": 0.7},
            {"code": "C", "qlib_score": 0.5},
            {"code": "D", "qlib_score": 0.3},
            {"code": "E", "qlib_score": 0.2},
        ]
        w = alloc.allocate_weights_only(cands)
        assert len(w) > 0
        for v in w.values():
            assert v >= 0.10 - 0.005
            assert v <= 0.20 + 0.005
        assert abs(sum(w.values()) - 0.95) < 0.01

    def test_cooldown_excludes_candidates(self):
        cfg = AllocationConfig(target_sum=1.0, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.9), _c("B", 0.8), _c("C", 0.7)]
        cooldown = {"B": 3}
        w = alloc.allocate_weights_only(cands, cooldown=cooldown)
        assert "B" not in w
        assert "A" in w
        assert "C" in w

    def test_panic_timing_zero_returns_empty(self):
        cfg = AllocationConfig(target_sum=1.0)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.9), _c("B", 0.8)]
        result = alloc.allocate(cands, {}, timing_ratio=0.0)
        assert result.target_weights == {}

    def test_allocate_with_turnover_limit_applied(self):
        cfg = AllocationConfig(max_turnover=0.10, max_single_pct=0.30,
                               apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [_c("A", 0.9), _c("B", 0.6)]
        current = {"A": 0.50, "B": 0.50}
        result = alloc.allocate(cands, current, timing_ratio=1.0)
        assert len(result.target_weights) > 0
        assert result.turnover <= 0.10 + 0.005
