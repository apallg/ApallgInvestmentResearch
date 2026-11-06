"""实盘调仓 generate_advice_from_screening 回归测试 — mock 外部依赖, 验证主流程跑通。

覆盖目标: 实盘赚钱主路径。历史上 observe 区持仓非空时抛 NameError
(current_weights 未赋值即引用), 此测试复现该场景防回归, 并验证观察区持仓
目标权重保持当前权重。
"""
from datetime import date
from unittest.mock import patch

import pytest

from core.domain.portfolio.allocator import AllocationConfig, PositionAllocator
from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor, classify_limit_codes


def _make_advisor():
    advisor = WeeklyRebalanceAdvisor.__new__(WeeklyRebalanceAdvisor)
    advisor.top_n = 2
    advisor.top_k = 5
    advisor.max_turnover = 0.3
    advisor.max_position_pct = 0.20
    advisor.max_sector_pct = 0.30
    advisor.min_cash_reserve = 0.05
    advisor.pool = "a500"
    advisor._cooldown = {}
    advisor._cooldown_as_of = date.today().isoformat()
    advisor._regime_stops = {}
    advisor._allocator = PositionAllocator(AllocationConfig())
    return advisor


@pytest.mark.p0
def test_advice_with_observe_holding_does_not_crash():
    """观察区持仓非空 (排名 3~5 的股票仍持有) 时不抛 NameError, 且观察区保留当前权重"""
    advisor = _make_advisor()

    screening_result = {"candidates": [
        {"code": "600001", "name": "A", "l0_score": 0.9, "qlib_score": 0.8},
        {"code": "600002", "name": "B", "l0_score": 0.8, "qlib_score": 0.7},
        {"code": "600003", "name": "C", "l0_score": 0.7, "qlib_score": 0.6},
        {"code": "600004", "name": "D", "l0_score": 0.6, "qlib_score": 0.5},
        {"code": "600005", "name": "E", "l0_score": 0.5, "qlib_score": 0.4},
    ], "regime": {}}
    holdings_data = [
        {"code": "600003", "market_value": 20000, "current_price": 10.0,
         "shares": 2000, "cost_price": 9.0, "unrealized_pnl": 2000.0},
    ]
    account_data = {"total_assets": 100000}

    classification = {
        "observe": [{"code": "600003", "name": ""}],
        "sell": [], "stop_loss": [], "reduce": [], "buy": [], "hold": [],
    }

    with patch("core.domain.portfolio.rebalance._get_industry_map", return_value={}), \
         patch.object(advisor, "_classify_holdings", return_value=classification), \
         patch.object(advisor, "_lookup_names", return_value={}), \
         patch.object(advisor, "_detect_limit_codes", return_value=(set(), set())), \
         patch.object(advisor, "_generate_orders", return_value=[]):
        result = advisor.generate_advice_from_screening(
            screening_result, holdings_data, account_data)

    assert "error" not in result
    assert len(result["observe_candidates"]) == 3
    # 观察区持仓 600003 目标权重保持当前权重 (20000/100000 = 0.2)
    assert abs(result["target_weights"]["600003"] - 0.2) < 1e-6
    assert sum(result["target_weights"].values()) <= 0.95 + 1e-9
    assert all(result["target_weights"][c] <= 0.20 for c in ("600001", "600002"))


@pytest.mark.p0
def test_advice_empty_candidates_returns_error():
    advisor = _make_advisor()
    result = advisor.generate_advice_from_screening(
        {"candidates": [], "regime": {}}, [], {"total_assets": 100000})
    assert "error" in result


@pytest.mark.p0
def test_turnover_clipped_orders_are_reported_as_skipped():
    advisor = _make_advisor()
    advisor.top_n = 5
    target_weights = {f"60000{i}": 0.19 for i in range(1, 6)}
    candidate_map = {
        code: {"rank": index, "name": f"股票{index}", "qlib_score": 1.0}
        for index, code in enumerate(target_weights, 1)
    }
    classification = {"hold": [], "observe": [], "sell": [], "stop_loss": [], "reduce": []}
    skipped = []

    with patch.object(advisor, "_persist_cooldown"):
        orders = advisor._generate_orders(
            classification, target_weights, {"600099": 0.01}, 100000, {"cash": 99000},
            {code: 10.0 for code in target_weights}, candidate_map,
            effective_max_turnover=0.30,
            limit_up_codes=set(), limit_down_codes=set(),
            keep_observe=True, skipped_out=skipped,
        )

    assert [order["code"] for order in orders] == ["600001"]
    assert [item["code"] for item in skipped] == ["600002", "600003", "600004", "600005"]
    assert all(item["reason"] == "换手预算不足（单边上限30%），本轮延后" for item in skipped)


@pytest.mark.p0
def test_initial_build_is_not_clipped_by_turnover_limit():
    advisor = _make_advisor()
    advisor.top_n = 5
    target_weights = {f"60000{i}": 0.19 for i in range(1, 6)}
    candidate_map = {
        code: {"rank": index, "name": f"股票{index}", "qlib_score": 1.0}
        for index, code in enumerate(target_weights, 1)
    }
    classification = {"hold": [], "observe": [], "sell": [], "stop_loss": [], "reduce": []}
    skipped = []

    with patch.object(advisor, "_persist_cooldown"):
        orders = advisor._generate_orders(
            classification, target_weights, {}, 100000, {"cash": 100000},
            {code: 10.0 for code in target_weights}, candidate_map,
            effective_max_turnover=0.30,
            limit_up_codes=set(), limit_down_codes=set(),
            keep_observe=True, skipped_out=skipped,
        )

    assert [order["code"] for order in orders] == list(target_weights)
    assert skipped == []


@pytest.mark.p0
def test_same_day_does_not_consume_cooldown():
    advisor = _make_advisor()
    advisor._cooldown = {"600001": 2}
    advisor._cooldown_as_of = "2026-08-18"
    with patch.object(advisor, "_persist_cooldown"):
        advisor._cooldown_cleanup(date(2026, 8, 18))
        advisor._cooldown_cleanup(date(2026, 8, 18))
    assert advisor._cooldown == {"600001": 2}


@pytest.mark.p0
def test_cooldown_advances_by_rebalance_period():
    advisor = _make_advisor()
    advisor.freq = "week"
    advisor._cooldown = {"600001": 2}
    advisor._cooldown_as_of = "2026-08-11"
    with patch.object(advisor, "_persist_cooldown"):
        advisor._cooldown_cleanup(date(2026, 8, 18))
    assert advisor._cooldown == {"600001": 1}


@pytest.mark.p0
def test_load_cooldown_accepts_repo_deserialized_payload():
    advisor = WeeklyRebalanceAdvisor.__new__(WeeklyRebalanceAdvisor)
    payload = {"as_of": "2026-08-18", "periods": {"600001": 2}}
    with patch("core.data.repos.settings.SettingsRepo.get", return_value=payload):
        assert advisor._load_persisted_cooldown() == {"600001": 2}
    assert advisor._cooldown_as_of == "2026-08-18"


@pytest.mark.p0
def test_main_board_st_uses_five_percent_limit():
    up, down = classify_limit_codes([
        {"code": "600001", "name": "*ST测试", "pct_chg": 4.8},
        {"code": "600002", "name": "正常股", "pct_chg": 4.8},
        {"code": "600003", "name": "ST测试", "pct_chg": -4.8},
    ])
    assert up == {"600001"}
    assert down == {"600003"}
