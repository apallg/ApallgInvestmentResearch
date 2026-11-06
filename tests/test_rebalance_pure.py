"""调仓引擎纯函数测试 (从 WeeklyRebalanceAdvisor 提取到模块级)"""
import pytest

from core.domain.portfolio.rebalance import (
    compute_turnover, classify_limit_codes, compute_prediction_correlation,
    build_combo_table, format_text_report,
)


@pytest.mark.p0
class TestComputeTurnover:
    def test_normal(self):
        assert compute_turnover({"a": 0.5, "b": 0.3}, {"a": 0.3, "b": 0.5}) == 0.2

    def test_symmetric(self):
        # 换手率双向对称
        t1 = compute_turnover({"a": 0.5}, {"b": 0.5})
        t2 = compute_turnover({"b": 0.5}, {"a": 0.5})
        assert t1 == t2 == 0.5

    def test_empty(self):
        assert compute_turnover({}, {}) == 0.0

    def test_new_position(self):
        assert compute_turnover({"a": 0.6}, {}) == 0.6


@pytest.mark.p0
class TestClassifyLimitCodes:
    def test_limit_up(self):
        up, down = classify_limit_codes([{"code": "a", "pct_chg": 10.0}])
        assert up == {"a"}
        assert down == set()

    def test_limit_down(self):
        up, down = classify_limit_codes([{"code": "a", "pct_chg": -10.0}])
        assert down == {"a"}
        assert up == set()

    def test_normal(self):
        up, down = classify_limit_codes([{"code": "a", "pct_chg": 1.0}])
        assert up == set()
        assert down == set()

    def test_missing_pct(self):
        up, down = classify_limit_codes([{"code": "a"}])
        assert up == set()
        assert down == set()


@pytest.mark.p1
class TestPredictionCorrelation:
    def test_negative_correlation(self):
        cur = {f"c{i}": float(i) for i in range(30)}
        prev = {f"c{i}": float(30 - i) for i in range(30)}
        corr = compute_prediction_correlation(cur, prev)
        assert corr is not None
        assert corr < -0.9

    def test_insufficient_common(self):
        cur = {f"a{i}": 1.0 for i in range(10)}
        prev = {f"b{i}": 1.0 for i in range(10)}
        assert compute_prediction_correlation(cur, prev) is None

    def test_empty_prev(self):
        assert compute_prediction_correlation({"a": 1.0}, {}) is None


@pytest.mark.p1
class TestBuildComboTable:
    def test_hold(self):
        candidates = [{"code": "a", "name": "A", "qlib_score": 0.8}]
        table = build_combo_table(candidates, {"a": 0.2}, {"a": 0.2}, {"a": {}}, 0.12)
        assert table[0]["action"] == "持有"

    def test_buy(self):
        candidates = [{"code": "a", "name": "A", "qlib_score": 0.8}]
        table = build_combo_table(candidates, {"a": 0.2}, {}, {}, 0.12)
        assert table[0]["action"] == "★买入"

    def test_add(self):
        candidates = [{"code": "a", "name": "A", "qlib_score": 0.8}]
        table = build_combo_table(candidates, {"a": 0.3}, {"a": 0.1}, {"a": {}}, 0.12)
        assert table[0]["action"] == "加仓"

    def test_reduce(self):
        candidates = [{"code": "a", "name": "A", "qlib_score": 0.8}]
        table = build_combo_table(candidates, {"a": 0.1}, {"a": 0.3}, {"a": {}}, 0.12)
        assert table[0]["action"] == "减仓"


@pytest.mark.p1
class TestFormatTextReport:
    def test_returns_str(self):
        report = {
            "date": "2025-01-06", "pool": "csi300", "sectors": "不限", "top_n": 5,
            "account": {"total_assets": 100000, "cash": 40000, "available_cash": 40000,
                        "total_cash": 50000, "frozen_cash": 10000, "market_value": 50000},
            "target_combo": [], "constraints": [], "auxiliary": {}, "risk": {},
        }
        text = format_text_report(report)
        assert isinstance(text, str)
        assert "周频调仓建议" in text
        assert "总现金 ¥50,000" in text
        assert "可用 ¥40,000" in text
        assert "冻结 ¥10,000" in text
