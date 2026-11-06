import pytest
from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor


class TestClassifyHoldings:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.advisor = WeeklyRebalanceAdvisor(top_n=5, top_k=10, stop_loss_pct=-0.08)

    def _make_holding(self, code, mv=50000, pnl=-1000, cost=None):
        if cost is None:
            cost = mv - pnl
        return {
            "symbol": code,
            "market_value": mv,
            "unrealized_pnl": pnl,
        }

    def test_stop_loss(self):
        holdings = {"000001": self._make_holding("000001", pnl=-10000, cost=100000)}
        candidate_map = {}
        result = self.advisor._classify_holdings(holdings, candidate_map, [], 200000)
        assert len(result["stop_loss"]) == 1
        assert result["stop_loss"][0]["code"] == "000001"

    def test_not_stop_loss_when_pnl_above_threshold(self):
        holdings = {"000001": self._make_holding("000001", pnl=-1500)}
        candidate_map = {}
        result = self.advisor._classify_holdings(holdings, candidate_map, [], 200000)
        assert len(result["stop_loss"]) == 0
        assert len(result["sell"]) == 1

    def test_sell_when_not_in_candidates(self):
        holdings = {"000001": self._make_holding("000001")}
        candidate_map = {}
        result = self.advisor._classify_holdings(holdings, candidate_map, [], 200000)
        assert len(result["sell"]) == 1
        assert result["sell"][0]["reason"] == "不在本周 Top-10 推荐中"

    def test_hold_when_in_top_n(self):
        holdings = {"000001": self._make_holding("000001")}
        candidate_map = {"000001": {"rank": 2, "name": "测试", "qlib_score": 0.8}}
        result = self.advisor._classify_holdings(holdings, candidate_map, ["000001"], 200000)
        assert len(result["hold"]) == 1
        assert result["hold"][0]["rank"] == 2

    def test_observe_when_in_top_k(self):
        holdings = {"000001": self._make_holding("000001")}
        candidate_map = {"000001": {"rank": 7, "name": "测试", "qlib_score": 0.5}}
        result = self.advisor._classify_holdings(holdings, candidate_map, ["A", "B", "C", "D", "E"], 200000)
        assert len(result["observe"]) == 1
        assert result["observe"][0]["rank"] == 7

    def test_stop_loss_priority_over_sell(self):
        holdings = {"000001": self._make_holding("000001", pnl=-10000, cost=100000)}
        candidate_map = {}
        result = self.advisor._classify_holdings(holdings, candidate_map, [], 200000)
        assert len(result["stop_loss"]) == 1
        assert len(result["sell"]) == 0
