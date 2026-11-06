"""L3 纯函数测试 — 基本面质量 + 流动性过滤"""
import pytest
from core.domain.screening.exclusion import (
    prefilter_by_quality,
    prefilter_by_liquidity_capacity,
    prefilter_by_min_amount,
    is_excluded,
)


def test_is_excluded_st():
    assert is_excluded("600001", "ST海投")[0] is True
    assert is_excluded("600001", "*ST海投")[0] is True
    assert is_excluded("600001", "正常股")[0] is False


def test_is_excluded_etf_bj():
    assert is_excluded("510300", "沪深300ETF")[0] is True
    assert is_excluded("830001.BJ", "北交所股")[0] is True


def test_prefilter_by_quality_keeps_good_drops_bad():
    codes = ["A", "B", "C", "D"]
    data = {
        "A": {"profit_dedt": 1.0, "debt_to_assets": 40},
        "B": {"profit_dedt": -0.5, "debt_to_assets": 50},   # 扣非为负
        "C": {"profit_dedt": 2.0, "debt_to_assets": 90},    # 负债率过高
        "D": {"profit_dedt": 3.0, "debt_to_assets": 30},
    }
    kept, dropped = prefilter_by_quality(codes, data)
    assert kept == ["A", "D"]
    assert [d[0] for d in dropped] == ["B", "C"]


def test_prefilter_by_quality_does_not_apply_generic_leverage_limit_to_financials():
    codes = ["601166", "601328", "600926", "601111", "001287"]
    data = {
        "601166": {"profit_dedt": 1, "debt_to_assets": 92, "industry": "J66货币金融服务"},
        "601328": {"profit_dedt": 1, "debt_to_assets": 92, "industry": "银行"},
        "600926": {"profit_dedt": 1, "debt_to_assets": 93, "name": "杭州银行"},
        "601111": {"profit_dedt": 1, "debt_to_assets": 88, "industry": "G56航空运输业"},
        "001287": {"profit_dedt": 1, "debt_to_assets": 88, "industry": "F51批发业"},
    }

    kept, dropped = prefilter_by_quality(codes, data)

    assert kept == ["601166", "601328", "600926"]
    assert [row[0] for row in dropped] == ["601111", "001287"]


def test_prefilter_by_quality_no_data_not_blocked():
    kept, dropped = prefilter_by_quality(["X"], {"Y": {"profit_dedt": -1}})
    assert kept == ["X"]
    assert dropped == []


def test_prefilter_by_quality_empty():
    kept, dropped = prefilter_by_quality(["A"], None)
    assert kept == ["A"]
    assert dropped == []


def test_prefilter_by_quality_execution_gate_blocks_missing_rows_and_fields():
    kept, dropped = prefilter_by_quality(
        ["A", "B", "C"],
        {
            "B": {"debt_to_assets": 30},
            "C": {"profit_dedt": 1},
        },
        require_complete=True,
    )

    assert kept == []
    assert dropped == [
        ("A", "基本面质量数据缺失，禁止新增风险"),
        ("B", "扣非净利润质量状态未知，禁止新增风险"),
        ("C", "资产负债率质量状态未知，禁止新增风险"),
    ]


def test_prefilter_by_quality_execution_gate_allows_complete_quality_rows():
    kept, dropped = prefilter_by_quality(
        ["A", "B"],
        {
            "A": {"profit_dedt": 1, "debt_to_assets": 30},
            "B": {"profit_dedt": 1, "debt_to_assets": 92, "industry": "J66货币金融服务"},
        },
        require_complete=True,
    )

    assert kept == ["A", "B"]
    assert dropped == []


def test_prefilter_by_quality_execution_gate_rejects_malformed_and_boolean_rows():
    kept, dropped = prefilter_by_quality(
        ["A", "B", "C"],
        {
            "A": "invalid-row",
            "B": {"profit_dedt": True, "debt_to_assets": 30},
            "C": {"profit_dedt": 1, "debt_to_assets": True},
        },
        require_complete=True,
    )

    assert kept == []
    assert dropped == [
        ("A", "基本面质量数据缺失，禁止新增风险"),
        ("B", "扣非净利润质量状态未知，禁止新增风险"),
        ("C", "资产负债率质量状态未知，禁止新增风险"),
    ]


def test_prefilter_by_min_amount_threshold():
    kept, dropped = prefilter_by_min_amount(
        ["A", "B", "C"],
        {"A": 50000000, "B": 9999999, "C": 10000000})
    assert kept == ["A", "C"]
    assert [d[0] for d in dropped] == ["B"]


def test_prefilter_by_min_amount_missing_data_is_blocked():
    kept, dropped = prefilter_by_min_amount(["A", "B"], {"A": 50000000})
    assert kept == ["A"]
    assert dropped == [("B", "近20日有效成交额不足15日，无法计算下单容量")]


def test_prefilter_by_min_amount_empty_data():
    kept, dropped = prefilter_by_min_amount(["A", "B"], None)
    assert kept == []
    assert [row[0] for row in dropped] == ["A", "B"]


def test_prefilter_by_liquidity_capacity_warns_and_blocks_large_orders():
    kept, dropped, checks = prefilter_by_liquidity_capacity(
        ["A", "B", "C"],
        {"A": 20000000, "B": 20000000, "C": 20000000},
        planned_amounts={"A": 200000, "B": 350000, "C": 500000},
    )

    assert kept == ["A", "B"]
    assert [row[0] for row in dropped] == ["C"]
    assert checks["A"]["status"] == "normal"
    assert checks["B"]["status"] == "warning"
    assert checks["C"]["status"] == "excluded"
