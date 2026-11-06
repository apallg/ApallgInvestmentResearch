"""PIT 选股链路测试 — 基本面评分只能看到 as_of 时点已披露的财报。"""
import datetime as dt

from core.domain.screening._scoring_legacy import _score_fundamentals
from core.domain.screening import _screener


def _period(report_date, avail_date, roe):
    return (report_date, avail_date, {
        "净资产收益率": roe,
        "净利润增长率": 20,
        "营业收入增长率": 15,
        "资产负债率": 40,
    })


def test_score_fundamentals_uses_only_visible_reports():
    cache = {
        "000001": [
            _period(dt.date(2023, 12, 31), dt.date(2024, 4, 30), 5.0),
            _period(dt.date(2024, 6, 30), dt.date(2024, 8, 31), 12.0),
        ]
    }
    # 2024-08-01 时，半年报（2024-06-30）尚未到法定披露截止 2024-08-31，不应可见
    score_before = _score_fundamentals("000001", as_of=dt.date(2024, 8, 1), financial_cache=cache)
    # 2024-09-01 时，半年报已披露，应使用 ROE=12
    score_after = _score_fundamentals("000001", as_of=dt.date(2024, 9, 1), financial_cache=cache)

    assert score_before != score_after
    # 前一时点只能看到 2023 年报，ROE 更低，分数应更小
    assert score_before < score_after


def test_stable_history_pool_uses_only_observations_before_decision_date(monkeypatch):
    captured = {}

    def fetchall(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return [{"code": "000001", "cnt": 500}]

    monkeypatch.setattr("core.data.Database.is_available", lambda: True)
    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    _screener._stable_history_cache.clear()
    result = _screener.filter_pool_codes(["000001", "000002"], "stable_history_500d", dt.date(2024, 8, 1))
    assert result == ["000001"]
    assert "trade_date < %s" in captured["sql"]
    assert captured["params"][-2:] == ["2024-08-01", 500]


def test_pit_pool_fails_closed_when_database_is_unavailable(monkeypatch):
    monkeypatch.setattr("core.data.Database.is_available", lambda: False)
    _screener._stable_history_cache.clear()
    assert _screener.filter_pool_codes(["000001"], "stable_history_500d", dt.date(2024, 8, 1)) == []
