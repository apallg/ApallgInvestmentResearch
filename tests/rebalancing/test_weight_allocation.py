import pytest
from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor


class TestAllocateWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.advisor = WeeklyRebalanceAdvisor(top_n=5, max_position_pct=0.25, min_cash_reserve=0.05)

    def _candidates(self, codes_scores):
        return [{"code": c, "qlib_score": s, "name": f"股票{c}"} for c, s in codes_scores]

    def test_equal_weight(self):
        c = self._candidates([("A", 0.8), ("B", 0.7), ("C", 0.6), ("D", 0.5), ("E", 0.4)])
        w = self.advisor._allocate_weights(c, "equal")
        assert len(w) == 5
        for v in w.values():
            assert v == pytest.approx(0.19)

    def test_equal_weight_sums_below_one(self):
        c = self._candidates([("A", 0.8), ("B", 0.7)])
        w = self.advisor._allocate_weights(c, "equal")
        assert sum(w.values()) == pytest.approx(0.95)

    def test_score_weighted_top_score_gets_max(self):
        c = self._candidates([("A", 0.9), ("B", 0.5), ("C", 0.3), ("D", 0.2), ("E", 0.1)])
        w = self.advisor._allocate_weights(c, "score_weighted")
        assert w["A"] == pytest.approx(0.25)  # 最高分触及上限
        assert w["E"] >= 0.10  # 最低分不低于下限
        assert sum(w.values()) <= 0.95  # 受上限约束可能达不到 target_sum

    def test_zero_scores_fallback_to_equal(self):
        c = self._candidates([("A", 0), ("B", 0), ("C", 0)])
        w = self.advisor._allocate_weights(c, "score_weighted")
        for v in w.values():
            assert v >= 0.10  # 不低于下限
            assert v <= 0.25  # 不超上限

    def test_clips_to_max_position(self):
        c = self._candidates([("A", 0.99), ("B", 0.01)])
        w = self.advisor._allocate_weights(c, "score_weighted")
        assert w["A"] <= 0.25

    def test_empty_candidates_returns_empty(self):
        w = self.advisor._allocate_weights([], "equal")
        assert w == {}

    def test_negative_scores_clipped_to_zero(self):
        c = self._candidates([("A", -0.5), ("B", 0.5)])
        w = self.advisor._allocate_weights(c, "score_weighted")
        # -0.5 被 clip 到 0, 和 0.5 的加权
        assert w["A"] == pytest.approx(0.10)  # floor
        assert w["B"] == pytest.approx(0.25)  # ceiling
