"""
板块融合 dir_scores 复用测试 — 锁住回测复用 pred_cache、选股现场预测两条路径。

回归重点: 方案 A 让回测融合复用主循环已预加载的方向模型截面,
避免每调仓日现场构建 DatasetH (150-205s)。dir_scores 给定时不触发
_find_model_by_direction / _load_qlib_predictions; 不给定时走现场预测(选股页面)。
"""
from unittest.mock import MagicMock

import pytest

from core.domain.screening._screener import BuySideScreener


def _make_screener():
    s = BuySideScreener.__new__(BuySideScreener)
    s.fetcher = MagicMock()
    s.fetcher.get_industry_map.return_value = {"600000": {"industry": "行业A"}}
    return s


def _patch_divergence(monkeypatch):
    import core.domain.regime.sector as sector_mod
    monkeypatch.setattr(
        sector_mod, "get_divergent_sectors",
        lambda as_of=None, market_regime=None, quotes_df=None: {
            "行业A": {"direction": "trend_momentum", "confidence": 0.8, "regime": 3},
        })


class TestSectorFusionDirScores:
    def test_dir_scores_reuse_skips_live_predict(self, monkeypatch):
        _patch_divergence(monkeypatch)
        s = _make_screener()
        calls = []

        def fake_find(direction):
            calls.append(("find", direction))
            return ("exp", "run")

        def fake_load(model, as_of=None, pool="csi300", **kwargs):
            calls.append(("load", model))
            return {"600000": 0.5}

        monkeypatch.setattr(BuySideScreener, "_find_model_by_direction", staticmethod(fake_find))
        monkeypatch.setattr(BuySideScreener, "_load_qlib_predictions", staticmethod(fake_load))

        fused, info = s._apply_sector_fusion(
            {"600000": 0.3}, "2024-01-01", "a500",
            dir_scores={"trend_momentum": {"600000": 0.7}})

        assert calls == [], "dir_scores 给定时不应触发现场预测"
        assert fused["600000"] != 0.3, "应被 dir_scores 融合"
        assert info["models"]["trend_momentum"] == "trend_momentum"

    def test_no_dir_scores_uses_live_predict(self, monkeypatch):
        _patch_divergence(monkeypatch)
        s = _make_screener()
        calls = []

        def fake_find(direction):
            calls.append(("find", direction))
            return ("exp", "run")

        def fake_load(model, as_of=None, pool="csi300", **kwargs):
            calls.append(("load", model))
            return {"600000": 0.5}

        monkeypatch.setattr(BuySideScreener, "_find_model_by_direction", staticmethod(fake_find))
        monkeypatch.setattr(BuySideScreener, "_load_qlib_predictions", staticmethod(fake_load))

        s._apply_sector_fusion({"600000": 0.3}, "2024-01-01", "a500")

        assert calls == [("find", "trend_momentum"), ("load", ("exp", "run"))], \
            "无 dir_scores 时应走现场预测路径"
