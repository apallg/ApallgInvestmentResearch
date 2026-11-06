import pytest
import numpy as np
from core.domain.factors.neutralize import (
    time_weighted_mean, orthogonalize_residual,
    cross_sectional_neutralize, neutralized_daily_ic,
)


class TestTimeWeightedMean:
    def test_equal_weight(self):
        result = time_weighted_mean([1, 2, 3, 4, 5], decay=1.0)
        assert result == 3.0

    def test_recent_heavy(self):
        result = time_weighted_mean([1, 2, 3, 4, 5], decay=0.5)
        assert result > 3.0

    def test_empty_returns_zero(self):
        assert time_weighted_mean([], decay=0.5) == 0.0

    def test_single_value(self):
        assert time_weighted_mean([7.0], decay=0.5) == 7.0

    def test_none_filtered_out(self):
        result = time_weighted_mean([None, 1.0, None, 3.0], decay=0.5)
        assert result == pytest.approx(time_weighted_mean([1.0, 3.0], decay=0.5))


class TestOrthogonalizeResidual:
    def test_empty_factor_matrix_returns_demeaned(self):
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = orthogonalize_residual(target, [])
        assert result == pytest.approx(target - target.mean())

    def test_single_factor_residual_uncorrelated(self):
        n = 100
        factor = np.random.default_rng(0).normal(0, 1, n)
        target = factor * 0.7 + np.random.default_rng(1).normal(0, 0.3, n)
        resid = orthogonalize_residual(target, [factor])
        assert np.abs(np.corrcoef(resid, factor)[0, 1]) < 1e-12

    def test_constant_factor_dropped(self):
        target = np.array([1.0, 2.0, 3.0])
        factor = np.array([5.0, 5.0, 5.0])  # std = 0
        resid = orthogonalize_residual(target, [factor])
        assert resid == pytest.approx(target - target.mean())

    def test_none_factor_matrix(self):
        target = np.array([1.0, 2.0, 3.0])
        result = orthogonalize_residual(target, None)
        assert result == pytest.approx(target - target.mean())


class TestCrossSectionalNeutralize:
    def test_few_codes_returns_unchanged(self):
        codes = ["A", "B", "C"]
        scores = [0.1, 0.2, 0.3]
        result = cross_sectional_neutralize(codes, scores, {}, {})
        for c, s in zip(codes, scores):
            assert result[c] == pytest.approx(s)

    def test_removes_size_effect(self):
        n = 30
        codes = [f"{i:06d}" for i in range(n)]
        scores = np.linspace(0, 1, n).tolist()
        mktcap_map = {c: (i + 1) * 1e9 for i, c in enumerate(codes)}
        # 分数和市值正相关
        result = cross_sectional_neutralize(codes, scores, mktcap_map, {})
        residuals = [result[c] for c in codes]
        assert len(residuals) == n

    def test_without_market_cap_not_crashing(self):
        n = 30
        codes = [f"{i:06d}" for i in range(n)]
        scores = np.random.default_rng(0).normal(0, 0.1, n).tolist()
        result = cross_sectional_neutralize(codes, scores, {}, {})
        assert len(result) == n
        for c in codes:
            assert result[c] is not None

    def test_with_industry(self):
        n = 30
        codes = [f"{i:06d}" for i in range(n)]
        scores = np.random.default_rng(0).normal(0, 0.1, n).tolist()
        industry_map = {c: {"industry": "A" if i < 15 else "B"} for i, c in enumerate(codes)}
        result = cross_sectional_neutralize(codes, scores, {}, industry_map)
        assert len(result) == n

    def test_details_report_no_effect_when_exposures_are_missing(self):
        n = 30
        codes = [f"{i:06d}" for i in range(n)]
        scores = np.linspace(0, 1, n).tolist()

        result, audit = cross_sectional_neutralize(
            codes, scores, {}, {}, return_details=True)

        assert result == dict(zip(codes, scores))
        assert audit["status"] == "no_effect_missing_exposures"
        assert audit["active_exposures"] == []

    def test_details_report_partial_for_market_cap_only(self):
        n = 30
        codes = [f"{i:06d}" for i in range(n)]
        scores = np.linspace(0, 1, n).tolist()
        mktcap_map = {c: (i + 1) * 1e9 for i, c in enumerate(codes)}

        _, audit = cross_sectional_neutralize(
            codes, scores, mktcap_map, {}, return_details=True)

        assert audit["status"] == "partial"
        assert audit["active_exposures"] == ["market_cap"]
        assert audit["market_cap_coverage"] == 1.0


class TestNeutralizedDailyIc:
    def test_perfect_correlation(self):
        n = 30
        codes = [f"{i:06d}" for i in range(n)]
        scores = np.linspace(0, 1, n).tolist()
        fwd = {c: s for c, s in zip(codes, scores)}
        ic = neutralized_daily_ic(codes, scores, fwd, {}, {})
        assert ic is not None
        assert ic > 0.9

    def test_perfect_inverse_correlation(self):
        n = 30
        codes = [f"{i:06d}" for i in range(n)]
        scores = np.linspace(0, 1, n).tolist()
        fwd = {c: -s for c, s in zip(codes, scores)}
        ic = neutralized_daily_ic(codes, scores, fwd, {}, {})
        assert ic is not None
        assert ic < -0.9

    def test_too_few_matches_returns_none(self):
        codes = ["A", "B", "C"]
        scores = [0.1, 0.2, 0.3]
        fwd = {"A": 0.01}
        ic = neutralized_daily_ic(codes, scores, fwd, {}, {})
        assert ic is None
