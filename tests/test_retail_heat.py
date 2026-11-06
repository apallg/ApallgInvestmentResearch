"""
compute_retail_heat 散户热门板块热度纯函数测试。

验证点:
- 散户流入强度 = 散户净额/流通市值, 大票不因净额大而虚高
- 板块按热度分降序, 成分股不足 min_stocks 的板块被剔除
- 空输入安全返回
"""
import numpy as np
import pandas as pd
import pytest

from core.domain.screening.retail_heat import compute_retail_heat


def _row(code, sm, md, lg, elg, main, turnover, pct_chg, mv):
    return {
        "code": code, "sm_net_flow": sm, "md_net_flow": md, "lg_net_flow": lg,
        "elg_net_flow": elg, "main_net_flow": main, "turnover_rate": turnover,
        "pct_chg": pct_chg, "circ_mv": mv,
    }


def _two_sectors():
    rows = []
    sector_map = {}
    # A: 散户净额大但市值小 -> 强度高
    for i in range(4):
        c = f"00000{i}"
        rows.append(_row(c, 100, 10, 20, 30, 50, 5.0, 2.0, 1000))
        sector_map[c] = ["A"]
    # B: 散户净额同 A, 但市值大 10 倍 -> 强度低
    for i in range(4, 8):
        c = f"00000{i}"
        rows.append(_row(c, 100, 10, 20, 30, 50, 5.0, 2.0, 10000))
        sector_map[c] = ["B"]
    return pd.DataFrame(rows), sector_map


@pytest.mark.unit
class TestRetailHeat:
    def test_ranking_by_retail_strength(self):
        df, sector_map = _two_sectors()
        result = compute_retail_heat(df, sector_map)
        # 换手/涨幅/上涨占比全相同, 排序完全由散户流入强度决定
        assert list(result["sector"]) == ["A", "B"]

    def test_strength_normalized_by_mv(self):
        df, sector_map = _two_sectors()
        result = compute_retail_heat(df, sector_map).set_index("sector")
        # A 的散户净额合计 = B, 但市值小 -> 强度 (净额/市值) 更高
        assert result.loc["A", "retail_strength"] > result.loc["B", "retail_strength"]

    def test_min_stocks_filter(self):
        df, sector_map = _two_sectors()
        # 追加 C 板块仅 2 只, 应被 min_stocks=3 剔除
        for i in range(8, 10):
            c = f"00000{i}"
            df = pd.concat([df, pd.DataFrame([_row(c, 500, 0, 0, 0, 0, 5.0, 2.0, 100)])], ignore_index=True)
            sector_map[c] = ["C"]
        result = compute_retail_heat(df, sector_map)
        assert set(result["sector"]) == {"A", "B"}

    def test_empty_input(self):
        empty_df = pd.DataFrame(columns=[
            "code", "sm_net_flow", "md_net_flow", "lg_net_flow", "elg_net_flow",
            "main_net_flow", "turnover_rate", "pct_chg", "circ_mv"])
        assert compute_retail_heat(empty_df, {}).empty
        df, _ = _two_sectors()
        assert compute_retail_heat(df, {}).empty

    def test_custom_weights(self):
        df, sector_map = _two_sectors()
        # 权重全压涨幅: A 涨幅 2%, B 涨幅 2% -> 无差别; 改 B 涨幅更高
        df.loc[df["code"].isin(["000004", "000005", "000006", "000007"]), "pct_chg"] = 5.0
        result = compute_retail_heat(
            df, sector_map, weights={"retail_inflow": 0.0, "turnover": 0.0,
                                     "pct_chg": 1.0, "advance_ratio": 0.0})
        assert list(result["sector"]) == ["B", "A"]
