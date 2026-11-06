"""选股打分/过滤纯函数测试 (从 _screener.py 提取)"""
import pytest

from core.domain.screening.scoring import rank_l1
from core.domain.screening.exclusion import prefilter_by_turnover, prefilter_by_index


@pytest.mark.p0
class TestRankL1:
    def test_sort_descending(self):
        candidates = [
            {"code": "a", "l0_score": 0.5},
            {"code": "b", "l0_score": 0.9},
            {"code": "c", "l0_score": 0.7},
        ]
        ranked = rank_l1(candidates)
        assert [c["code"] for c in ranked] == ["b", "c", "a"]

    def test_sets_l1_score(self):
        candidates = [{"code": "a", "l0_score": 0.81234}]
        ranked = rank_l1(candidates)
        assert ranked[0]["l1_score"] == 0.812
        assert ranked[0]["strategy_score"] == 0

    def test_empty(self):
        assert rank_l1([]) == []


@pytest.mark.p0
class TestPrefilterByTurnover:
    def test_top_n_by_amount(self):
        codes = ["a", "b", "c", "d"]
        turnover = {"a": 100, "b": 300, "c": 50, "d": 200}
        result = prefilter_by_turnover(codes, 2, turnover)
        assert result == ["b", "d"]

    def test_empty_turnover_data(self):
        codes = ["a", "b", "c"]
        assert prefilter_by_turnover(codes, 2, {}) == ["a", "b"]

    def test_empty_codes(self):
        assert prefilter_by_turnover([], 5, {"a": 100}) == []

    def test_zero_top_n(self):
        assert prefilter_by_turnover(["a", "b"], 0, {"a": 100}) == ["a", "b"]


@pytest.mark.p0
class TestPrefilterByIndex:
    def test_filter(self):
        assert prefilter_by_index(["a", "b", "c"], {"a", "c"}) == ["a", "c"]

    def test_empty_members(self):
        assert prefilter_by_index(["a", "b"], set()) == ["a", "b"]

    def test_keeps_order(self):
        assert prefilter_by_index(["c", "a", "b"], {"a", "b", "c"}) == ["c", "a", "b"]
