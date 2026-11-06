"""
市场状态权重验证 — 权重归一化、所有状态全覆盖、零权重项一致性
"""
import pytest
import numpy as np

from core.domain.regime.weights import REGIME_WEIGHTS, REGIME_TECH_PRESETS, get_vol_direction
from core.domain.regime.classifier import Regime, HMM_TO_LEGACY


EXPECTED_SOURCES = {
    "qlib", "tech", "sentiment", "fundamental", "flow",
    "north_flow", "size_turnover", "volatility",
    "illiq", "overnight_gap", "ht_factors",
    "wb_industry", "wb_divergence", "wb_moneyflow",
    "momentum",
}


class TestRegimeWeights:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_all_regimes_covered(self):
        for regime in Regime:
            assert regime in REGIME_WEIGHTS, f"Missing {regime}"

    @pytest.mark.unit
    @pytest.mark.p0
    def test_weights_sum_to_one(self):
        for regime, weights in REGIME_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.05, \
                f"{regime.value}: sum={total:.4f}"

    @pytest.mark.unit
    @pytest.mark.p0
    def test_qlib_dominant(self):
        for regime, weights in REGIME_WEIGHTS.items():
            qlib_w = weights.get("qlib", 0)
            assert qlib_w >= 0.30, \
                f"{regime.value}: qlib={qlib_w} < 0.30, not dominant"
            assert qlib_w <= 0.38, \
                f"{regime.value}: qlib={qlib_w} > 0.38, over-concentrated"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_disabled_sources_are_zero(self):
        disabled = {"illiq", "overnight_gap", "ht_factors", "momentum"}
        for regime, weights in REGIME_WEIGHTS.items():
            for src in disabled:
                assert weights.get(src, 0) == 0.0, \
                    f"{regime.value}: {src}={weights.get(src)} should be 0"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_no_negative_weights(self):
        for regime, weights in REGIME_WEIGHTS.items():
            for src, w in weights.items():
                assert w >= 0, f"{regime.value}: {src}={w} < 0"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_fundamental_higher_in_down_markets(self):
        w_up = REGIME_WEIGHTS[Regime.VOLATILE_UP]["fundamental"]
        w_down = REGIME_WEIGHTS[Regime.VOLATILE_DOWN]["fundamental"]
        assert w_down >= w_up, \
            f"Fundamental in down ({w_down}) should be >= up ({w_up})"

    @pytest.mark.unit
    @pytest.mark.p2
    def test_volatility_higher_in_volatile(self):
        w_vol = REGIME_WEIGHTS[Regime.VOLATILE_FLAT]["volatility"]
        w_up = REGIME_WEIGHTS[Regime.STEADY_UP]["volatility"]
        assert w_vol >= w_up, \
            f"Vol in volatile ({w_vol}) should be >= up ({w_up})"

    @pytest.mark.unit
    @pytest.mark.p2
    def test_bull_steady_highest_qlib(self):
        w_steady = REGIME_WEIGHTS[Regime.STEADY_UP]["qlib"]
        for regime, weights in REGIME_WEIGHTS.items():
            assert w_steady >= weights["qlib"] - 0.01, (
                f"STEADY_UP qlib ({w_steady}) should be highest, "
                f"{regime.value}={weights['qlib']}"
            )

    @pytest.mark.unit
    @pytest.mark.p1
    def test_all_known_sources_present(self):
        for regime, weights in REGIME_WEIGHTS.items():
            for src in EXPECTED_SOURCES:
                assert src in weights, \
                    f"{regime.value}: missing source '{src}'"

    @pytest.mark.unit
    @pytest.mark.p2
    def test_set_table_consistency(self):
        """TECH_PRESETS 中各状态的 ADX 阈值有合理区间"""
        if REGIME_TECH_PRESETS:
            for regime, preset in REGIME_TECH_PRESETS.items():
                if "adx_threshold" in preset:
                    t = preset["adx_threshold"]
                    assert 10 <= t <= 40, \
                        f"{regime}: adx_threshold={t} out of [10,40]"


class TestHMMToLegacy:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_all_regime_states_mapped(self):
        for regime in Regime:
            assert regime in HMM_TO_LEGACY, f"Missing HMM_TO_LEGACY[{regime}]"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_identity_mapping(self):
        for regime in Regime:
            assert HMM_TO_LEGACY[regime] == regime, \
                f"HMM_TO_LEGACY[{regime}] should be identity"


class TestGetVolDirection:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_high_vol_uses_oversold_rebound(self):
        assert get_vol_direction(0.8) == "oversold_rebound"

    @pytest.mark.unit
    @pytest.mark.p0
    def test_low_vol_uses_trend_momentum(self):
        assert get_vol_direction(0.2) == "trend_momentum"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_none_falls_back_to_trend_momentum(self):
        assert get_vol_direction(None) == "trend_momentum"

    @pytest.mark.unit
    @pytest.mark.p2
    def test_threshold_boundary_not_high_vol(self):
        assert get_vol_direction(0.5) == "trend_momentum"
