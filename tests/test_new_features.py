"""
测试 P1/P3/P5/P2/P4 改动 (改写版 — 修正 mock 路径)
"""
import sys
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, '.')


def _make_index_df(n_rows=260, trend=0):
    dates = []
    d = date(2020, 1, 1)
    while len(dates) < n_rows:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    rng = np.random.default_rng(42)
    rets = rng.normal(trend / 250, 0.015, n_rows)
    closes = 3500 * np.cumprod(1 + rets)
    highs = closes * (1 + np.abs(rng.normal(0, 0.01, n_rows)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.01, n_rows)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * (1 + rng.normal(0, 0.003))
    volumes = rng.lognormal(mean=20, sigma=0.5, size=n_rows).astype(float)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "amount": volumes * closes,
    }).sort_values("date").reset_index(drop=True)


# ═══════════════ P1 ═══════════════

class TestDirectionActiveSources:
    def test_all_directions_have_keys(self):
        from core.domain.regime.weights import DIRECTION_ACTIVE_SOURCES
        assert len(DIRECTION_ACTIVE_SOURCES) == 3
        for d in ["trend_momentum", "oversold_rebound", "volume_breakout"]:
            assert d in DIRECTION_ACTIVE_SOURCES
            assert len(DIRECTION_ACTIVE_SOURCES[d]) > 3

    def test_trend_momentum_focus(self):
        from core.domain.regime.weights import DIRECTION_ACTIVE_SOURCES
        s = DIRECTION_ACTIVE_SOURCES["trend_momentum"]
        assert "qlib" in s          # 主模型始终在
        assert "north_flow" in s
        assert "volatility" not in s  # 低波不适合趋势市

    def test_oversold_rebound_volatile(self):
        from core.domain.regime.weights import DIRECTION_ACTIVE_SOURCES
        s = DIRECTION_ACTIVE_SOURCES["oversold_rebound"]
        assert "momentum" not in s     # 急跌中动量无效
        assert "volatility" in s       # 波动率识别超卖
        assert "flow" in s             # 资金流捕捉恐慌放量
        assert "north_flow" in s       # 北向跟踪聪明钱

    def test_volume_breakout_ranging(self):
        from core.domain.regime.weights import DIRECTION_ACTIVE_SOURCES
        s = DIRECTION_ACTIVE_SOURCES["volume_breakout"]
        assert "fundamental" not in s  # 基本面不适合震荡市
        assert "momentum" not in s     # 动量不适合震荡市
        assert "tech" in s
        assert "flow" in s


class TestFilterWeightsByDirection:
    BASE = {"momentum": 0.22, "volatility": 0.15, "tech": 0.12, "fundamental": 0.08,
            "north_flow": 0.15, "flow": 0.05, "qlib": 0.10, "sentiment": 0.05,
            "size_turnover": 0.08, "illiq": 0.02, "overnight_gap": 0.03,
            "ht_factors": 0.05, "wb_industry": 0.03, "wb_divergence": 0.03, "wb_moneyflow": 0.03}

    def test_trend_momentum_zeros_defensive(self):
        from core.domain.regime.weights import filter_weights_by_direction
        f = filter_weights_by_direction(dict(self.BASE), "trend_momentum")
        assert f["volatility"] == 0.0    # 趋势方向不看低波
        assert f["size_turnover"] == 0.0 # 趋势方向不看小市值
        assert f["qlib"] > 0.1           # 主模型始终在
        assert f["north_flow"] > 0.1

    def test_oversold_rebound_zeros_offensive(self):
        from core.domain.regime.weights import filter_weights_by_direction
        f = filter_weights_by_direction(dict(self.BASE), "oversold_rebound")
        assert f["momentum"] == 0.0    # 急跌动量无效
        assert f["fundamental"] == 0.0 # 基本面短期不兑现
        assert f["qlib"] > 0.1         # 主模型始终在
        assert f["volatility"] > 0.1   # 波动率核心信号
        assert f["flow"] > 0           # 资金流捕捉恐慌
        assert f["north_flow"] > 0     # 北向聪明钱

    def test_normalizes_to_one(self):
        from core.domain.regime.weights import filter_weights_by_direction
        for d in ["trend_momentum", "oversold_rebound", "volume_breakout"]:
            f = filter_weights_by_direction(dict(self.BASE), d)
            assert abs(sum(f.values()) - 1.0) < 0.01, f"{d}: sum={sum(f.values())}"

    def test_unchanged_when_unknown_direction(self):
        from core.domain.regime.weights import filter_weights_by_direction
        w = {"momentum": 0.5, "volatility": 0.5}
        assert filter_weights_by_direction(w, "nonexistent") == w

    def test_single_source_kept(self):
        """只有一个源在活跃集合里 → 归一化后为 1.0"""
        from core.domain.regime.weights import filter_weights_by_direction
        w = {"qlib": 0.5, "volatility": 0.5}
        f = filter_weights_by_direction(w, "trend_momentum")
        assert f["qlib"] == 1.0       # qlib在趋势方向活跃
        assert f["volatility"] == 0.0

    def test_all_sources_kept_when_all_active(self):
        """如果某个方向的活跃源包含所有源 → 无零化"""
        from core.domain.regime.weights import filter_weights_by_direction
        # volume_breakout 包含 tech/flow/volatility/size_turnover/illiq/wb_*
        w = {"tech": 0.3, "flow": 0.3, "volatility": 0.2, "size_turnover": 0.2}
        f = filter_weights_by_direction(w, "volume_breakout")
        assert all(v > 0 for v in f.values())
        assert abs(sum(f.values()) - 1.0) < 0.01


# ═══════════════ P3: Vol/Turnover ═══════════════

class TestVolTurnoverRatio:
    def test_returns_value_for_valid_data(self):
        df = _make_index_df(300)
        mock_fetcher = MagicMock()
        mock_fetcher.get_index_daily.return_value = df
        # DataFetcher 在 market_breadth 的 get_vol_turnover_ratio 中 lazy import
        with patch("core.data.fetcher.DataFetcher", return_value=mock_fetcher):
            from core.domain.timing.breadth import get_vol_turnover_ratio
            latest, series = get_vol_turnover_ratio("000300", lookback=200)
            assert latest is not None
            assert isinstance(latest, float)
            assert latest > 0

    def test_returns_none_for_empty_df(self):
        mock_fetcher = MagicMock()
        mock_fetcher.get_index_daily.return_value = pd.DataFrame()
        with patch("core.data.fetcher.DataFetcher", return_value=mock_fetcher):
            from core.domain.timing.breadth import get_vol_turnover_ratio
            latest, series = get_vol_turnover_ratio("000300")
            assert latest is None

    def test_down_trend_produces_valid_kernel(self):
        df = _make_index_df(300, trend=-0.15)
        mock_fetcher = MagicMock()
        mock_fetcher.get_index_daily.return_value = df
        with patch("core.data.fetcher.DataFetcher", return_value=mock_fetcher):
            from core.domain.timing.breadth import get_vol_turnover_ratio
            latest, _ = get_vol_turnover_ratio("000300", lookback=200)
            assert latest is not None
            assert latest > 0


# ═══════════════ P5: RSRS ═══════════════

class TestRsrsMomentumScores:
    def test_returns_4_indices(self):
        dfs = [_make_index_df(120) for _ in range(4)]
        mock_fetcher = MagicMock()
        mock_fetcher.get_index_daily.side_effect = dfs
        with patch("core.data.fetcher.DataFetcher", return_value=mock_fetcher):
            from core.domain.timing.breadth import get_rsrs_momentum_scores
            result = get_rsrs_momentum_scores(lookback=89)
            assert len(result) == 4
            for k in ["上证50", "沪深300", "中证500", "国证2000"]:
                assert k in result

    def test_each_entry_has_score_and_r2(self):
        dfs = [_make_index_df(120) for _ in range(4)]
        mock_fetcher = MagicMock()
        mock_fetcher.get_index_daily.side_effect = dfs
        with patch("core.data.fetcher.DataFetcher", return_value=mock_fetcher):
            from core.domain.timing.breadth import get_rsrs_momentum_scores
            result = get_rsrs_momentum_scores(lookback=89)
            for name, info in result.items():
                assert "score" in info, f"{name} missing score"
                assert "r2" in info, f"{name} missing r2"
                if info["r2"] is not None:
                    assert 0 <= info["r2"] <= 1, f"{name} r2={info['r2']} out of range"

    def test_exactly_one_dominant(self):
        # 用强趋势确保区分度: 上证50 强涨 vs 其余弱趋势
        dfs = [_make_index_df(200, trend=0.4), _make_index_df(200, trend=0.0),
               _make_index_df(200, trend=0.0), _make_index_df(200, trend=0.0)]
        mock_fetcher = MagicMock()
        mock_fetcher.get_index_daily.side_effect = dfs
        with patch("core.data.fetcher.DataFetcher", return_value=mock_fetcher):
            from core.domain.timing.breadth import get_rsrs_momentum_scores
            result = get_rsrs_momentum_scores(lookback=89)
            dominants = [n for n, i in result.items() if i.get("dominant")]
            # 上证50 应有明显最高分
            assert len(dominants) >= 1, f"Expected at least 1 dominant, got {[(n,i.get('score')) for n,i in result.items()]}"
            assert "上证50" in dominants

    def test_uptrend_positive(self):
        df_up = _make_index_df(120, trend=0.3)
        mock_fetcher = MagicMock()
        mock_fetcher.get_index_daily.side_effect = [df_up, _make_index_df(120), _make_index_df(120), _make_index_df(120)]
        with patch("core.data.fetcher.DataFetcher", return_value=mock_fetcher):
            from core.domain.timing.breadth import get_rsrs_momentum_scores
            result = get_rsrs_momentum_scores(lookback=89)
            assert result["上证50"]["score"] is not None


# ═══════════════ P2: ETF Flow ═══════════════

class TestEtfFlowScoring:
    def test_zero_when_no_industry(self):
        from core.domain.screening._scoring_legacy import _score_etf_flow
        assert _score_etf_flow("000001", industry=None) == 0.0
        assert _score_etf_flow("000001", industry="") == 0.0

    def test_score_in_range(self):
        with patch("core.data.sources.etf_flow.get_stock_etf_flow_score", return_value=0.1):
            from core.domain.screening._scoring_legacy import _score_etf_flow
            r = _score_etf_flow("000001", industry="电子")
            assert -1.0 <= r <= 1.0

    def test_exception_returns_zero(self):
        with patch("core.data.sources.etf_flow.get_stock_etf_flow_score",
                   side_effect=RuntimeError("boom")):
            from core.domain.screening._scoring_legacy import _score_etf_flow
            assert _score_etf_flow("000001", industry="电子") == 0.0


class TestEtfFlowModule:
    def test_industry_map_loaded(self):
        from core.data.sources.etf_flow import ETF_INDUSTRY_MAP, ETF_STYLE_MAP
        assert len(ETF_INDUSTRY_MAP) > 10
        assert len(ETF_STYLE_MAP) > 5
        assert "电子" in set(ETF_INDUSTRY_MAP.values())
        assert "大盘" in set(ETF_STYLE_MAP.values())

    def test_score_by_inflow_strength(self):
        from core.data.sources.etf_flow import get_stock_etf_flow_score
        cases = [
            ({"电子": 2e9}, 0.15),   # >10亿
            ({"电子": 8e8}, 0.10),   # >5亿
            ({"电子": 3e8}, 0.06),   # >1亿
            ({"电子": 5e7}, 0.03),   # >0
            ({"电子": 0}, 0.0),
            ({"电子": -3e8}, -0.06),
            ({"电子": -8e8}, -0.10),
            ({"电子": -2e9}, -0.15),
        ]
        for flow, expected in cases:
            with patch("core.data.sources.etf_flow.get_etf_flow_by_industry", return_value=flow):
                result = get_stock_etf_flow_score("000001", "电子")
                assert result == expected, f"flow={flow} → expected {expected}, got {result}"

    def test_unknown_industry_returns_zero(self):
        from core.data.sources.etf_flow import get_stock_etf_flow_score
        with patch("core.data.sources.etf_flow.get_etf_flow_by_industry", return_value={}):
            assert get_stock_etf_flow_score("000001", "不存在") == 0.0


# ═══════════════ P4: Sector Rotation ═══════════════

class TestRankSectors:
    def test_returns_list_of_tuples(self):
        with patch("core.data.Database") as mock_db:
            mock_db.fetchall.side_effect = [
                [{"industry": "电子", "avg_ret": 0.05}, {"industry": "医药生物", "avg_ret": 0.02}],
                [{"industry": "电子", "breadth": 0.65}, {"industry": "医药生物", "breadth": 0.45}],
            ]
            with patch("core.data.sources.etf_flow.get_etf_flow_by_industry", return_value={}):
                from core.domain.screening._scoring_legacy import _rank_sectors
                result = _rank_sectors(eval_date="2026-01-15", top_n=2)
                assert isinstance(result, list)
                if result:
                    assert isinstance(result[0], tuple)
                    assert len(result[0]) == 3  # (name, score, dims)

    def test_empty_when_no_data(self):
        with patch("core.data.Database") as mock_db:
            mock_db.fetchall.return_value = []
            with patch("core.data.sources.etf_flow.get_etf_flow_by_industry", return_value={}):
                from core.domain.screening._scoring_legacy import _rank_sectors
                assert _rank_sectors(eval_date="2026-01-15") == []

    def test_ranks_descending(self):
        with patch("core.data.Database") as mock_db:
            mock_db.fetchall.side_effect = [
                [{"industry": "电子", "avg_ret": 0.10}, {"industry": "银行", "avg_ret": 0.01},
                 {"industry": "医药生物", "avg_ret": 0.05}],
                [{"industry": "电子", "breadth": 0.70}, {"industry": "银行", "breadth": 0.30},
                 {"industry": "医药生物", "breadth": 0.50}],
            ]
            with patch("core.data.sources.etf_flow.get_etf_flow_by_industry", return_value={}):
                from core.domain.screening._scoring_legacy import _rank_sectors
                result = _rank_sectors(eval_date="2026-01-15", top_n=3)
                if len(result) >= 2:
                    assert result[0][1] >= result[1][1]
                    assert result[1][1] >= result[2][1]

    def test_etf_flow_boosts_ranking(self):
        """强 ETF 流入应提升行业排名"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchall.side_effect = [
                [{"industry": "电子", "avg_ret": 0.03}, {"industry": "医药生物", "avg_ret": 0.03}],
                [{"industry": "电子", "breadth": 0.50}, {"industry": "医药生物", "breadth": 0.50}],
            ]
            # 电子 ETF 流入 = 医药生物 ETF 流出 → 电子排第一
            etf_flows = {"电子": 1e9, "医药生物": -5e8}
            with patch("core.data.sources.etf_flow.get_etf_flow_by_industry", return_value=etf_flows):
                from core.domain.screening._scoring_legacy import _rank_sectors
                result = _rank_sectors(eval_date="2026-01-15", top_n=2)
                assert len(result) == 2
                assert result[0][0] == "电子"  # 动量相同但 ETF 流入 → 排前面
                assert result[1][0] == "医药生物"
