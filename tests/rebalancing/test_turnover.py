import pytest
from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor


class TestComputeTurnover:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.advisor = WeeklyRebalanceAdvisor()

    def test_no_change_zero_turnover(self):
        w = {"A": 0.3, "B": 0.3, "C": 0.35}
        result = self.advisor._compute_turnover(w, w)
        assert result == 0.0

    def test_initial_build_turnover_includes_cash_leg(self):
        cur = {}
        tgt = {"A": 0.3, "B": 0.3, "C": 0.35}
        result = self.advisor._compute_turnover(tgt, cur)
        assert result == pytest.approx(0.95)

    def test_partial_rotation(self):
        cur = {"A": 0.4, "B": 0.55}
        tgt = {"A": 0.3, "B": 0.4, "C": 0.25}
        result = self.advisor._compute_turnover(tgt, cur)
        stock_leg = abs(0.3-0.4) + abs(0.4-0.55) + abs(0.25-0)
        cash_leg = abs(sum(tgt.values()) - sum(cur.values()))
        expected = (stock_leg + cash_leg) / 2
        assert result == pytest.approx(expected)

    def test_new_stock_only(self):
        cur = {"A": 0.5, "B": 0.45}
        tgt = {"A": 0.5, "B": 0.45, "C": 0}
        result = self.advisor._compute_turnover(tgt, cur)
        assert result == 0.0


class TestCheckConstraints:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.advisor = WeeklyRebalanceAdvisor(max_turnover=0.50, max_position_pct=0.25,
                                               min_cash_reserve=0.05, max_sector_pct=0.30)

    def test_turnover_pass(self):
        results = self.advisor._check_constraints(0.30, {"A": 0.3}, [{"code": "A"}])
        turnover_check = next(r for r in results if r["name"] == "换手率")
        assert turnover_check["pass"] is True

    def test_turnover_fail(self):
        results = self.advisor._check_constraints(0.70, {"A": 0.5}, [{"code": "A"}])
        turnover_check = next(r for r in results if r["name"] == "换手率")
        assert turnover_check["pass"] is False

    def test_initial_build_bypasses_turnover_limit(self):
        results = self.advisor._check_constraints(
            0.6, {"A": 0.2, "B": 0.2, "C": 0.2},
            [{"code": "A"}, {"code": "B"}, {"code": "C"}],
            initial_build=True,
        )
        turnover_check = next(r for r in results if r["name"] == "换手率")
        assert turnover_check["pass"] is True
        assert turnover_check["note"] == "空仓首次建仓，不适用调仓换手上限"

    def test_position_pass(self):
        results = self.advisor._check_constraints(0.20, {"A": 0.20}, [{"code": "A"}])
        pos_check = next(r for r in results if r["name"] == "单只最大仓位")
        assert pos_check["pass"] is True

    def test_position_fail(self):
        results = self.advisor._check_constraints(0.30, {"A": 0.30}, [{"code": "A"}])
        pos_check = next(r for r in results if r["name"] == "单只最大仓位")
        assert pos_check["pass"] is False

    def test_cash_reserve_pass(self):
        results = self.advisor._check_constraints(0.10, {"A": 0.90}, [{"code": "A"}])
        cash_check = next(r for r in results if r["name"] == "现金保留")
        assert cash_check["pass"] is True

    def test_cash_reserve_fail(self):
        results = self.advisor._check_constraints(0.10, {"A": 1.0}, [{"code": "A"}])
        cash_check = next(r for r in results if r["name"] == "现金保留")
        assert cash_check["pass"] is False
