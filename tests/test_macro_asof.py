"""
宏观修正 as_of 贯穿测试 — 锁住禁止前视偏差。

回归重点: 回测时宏观修正必须用决策日(as_of)的历史宏观数据,
不能落到 today_date()。之前 get_sector_rotation_signals /
calculate_macro_risk_score / get_factor_macro_fitness / _trend
都没传 as_of, 导致回测用了"今天"的宏观数据(前视)。
"""
import datetime as dt

import pandas as pd
import pytest

from core.domain.macro import analyzer
from core.domain.macro.data import MacroDataAccess, MacroSnapshot


class TestMacroAsOf:
    def test_get_macro_snapshot_passes_as_of(self, monkeypatch):
        """get_macro_snapshot 必须把 as_of 传到各 get_latest_xxx 的 SQL"""
        calls = []

        def fake_query(table, columns, where=None, params=None, order_by=None, limit=None):
            calls.append((table, params))
            return []

        monkeypatch.setattr(MacroDataAccess, "_query", staticmethod(fake_query))
        as_of = dt.date(2024, 3, 1)
        MacroDataAccess.get_macro_snapshot(as_of=as_of)
        assert calls, "get_macro_snapshot 未触发任何查询"
        for table, params in calls:
            assert params == (as_of,), f"{table} 的 as_of 未贯穿"

    def test_lpr_only_snapshot_is_not_treated_as_empty(self, monkeypatch):
        as_of = dt.date(2026, 10, 30)

        def fake_query(table, columns, where=None, params=None, order_by=None, limit=None):
            if table == "macro_lpr":
                return [{
                    "trade_date": dt.date(2026, 10, 20),
                    "lpr_1y": 3.0,
                    "lpr_5y": 3.5,
                    "source_name": "tushare",
                    "source_observed_at": dt.datetime(2026, 10, 20, 9, 30),
                }]
            return []

        monkeypatch.setattr(MacroDataAccess, "_query", staticmethod(fake_query))

        snapshot = MacroDataAccess.get_macro_snapshot(as_of=as_of)

        assert snapshot is not None
        assert snapshot.lpr_5y == 3.5
        assert snapshot.raw["lpr"]["source_name"] == "tushare"

    def test_get_macro_change_passes_as_of(self, monkeypatch):
        """get_macro_change 的 end 必须是 as_of, 不能是 today"""
        captured = {}

        def fake_timeseries(start, end):
            captured["start"] = start
            captured["end"] = end
            idx = pd.date_range(start, end, freq="D")
            return pd.DataFrame({"pmi": [50.0] * len(idx)}, index=idx)

        monkeypatch.setattr(MacroDataAccess, "get_macro_timeseries", staticmethod(fake_timeseries))
        as_of = dt.date(2024, 3, 1)
        MacroDataAccess.get_macro_change("pmi", 90, as_of=as_of)
        assert captured["end"] == as_of
        assert captured["start"] == as_of - dt.timedelta(days=90)

    def test_get_macro_change_accepts_str_as_of(self, monkeypatch):
        """snapshot.as_of_date 是 str, 必须正确转成 date"""
        captured = {}

        def fake_timeseries(start, end):
            captured["end"] = end
            idx = pd.date_range(start, end, freq="D")
            return pd.DataFrame({"pmi": [50.0] * len(idx)}, index=idx)

        monkeypatch.setattr(MacroDataAccess, "get_macro_timeseries", staticmethod(fake_timeseries))
        MacroDataAccess.get_macro_change("pmi", 90, as_of="2024-03-01")
        assert captured["end"] == dt.date(2024, 3, 1)

    def test_sector_rotation_signals_passes_as_of(self, monkeypatch):
        captured = {}

        def fake_snapshot(as_of=None):
            captured["as_of"] = as_of
            return None

        monkeypatch.setattr(MacroDataAccess, "get_macro_snapshot", staticmethod(fake_snapshot))
        as_of = dt.date(2024, 3, 1)
        analyzer.MacroAnalyzer().get_sector_rotation_signals(as_of=as_of)
        assert captured["as_of"] == as_of

    def test_calculate_macro_risk_score_passes_as_of(self, monkeypatch):
        snap_captured = {}
        chg_calls = []

        def fake_snapshot(as_of=None):
            snap_captured["as_of"] = as_of
            return MacroSnapshot(as_of_date=str(as_of) if as_of else "")

        def fake_change(indicator, days=30, as_of=None):
            chg_calls.append((indicator, days, as_of))
            return None

        monkeypatch.setattr(MacroDataAccess, "get_macro_snapshot", staticmethod(fake_snapshot))
        monkeypatch.setattr(MacroDataAccess, "get_macro_change", staticmethod(fake_change))
        as_of = dt.date(2024, 3, 1)
        analyzer.MacroAnalyzer().calculate_macro_risk_score(as_of=as_of)
        assert snap_captured["as_of"] == as_of
        assert chg_calls, "calculate_macro_risk_score 未触发 get_macro_change"
        for _, _, a in chg_calls:
            assert a == as_of

    def test_trend_reads_as_of_from_snapshot(self, monkeypatch):
        """_trend 从 snapshot.as_of_date 读 as_of, 而不是 today"""
        calls = []

        def fake_change(indicator, days=30, as_of=None):
            calls.append((indicator, days, as_of))
            return 0.1

        monkeypatch.setattr(MacroDataAccess, "get_macro_change", staticmethod(fake_change))
        snap = MacroSnapshot(as_of_date="2024-03-01")
        analyzer._trend(snap, "lpr_5y", -0.15, 90)
        assert calls[0][2] == "2024-03-01"

    def test_factor_macro_fitness_passes_as_of(self, monkeypatch):
        captured = {}

        def fake_snapshot(as_of=None):
            captured["as_of"] = as_of
            return None

        monkeypatch.setattr(MacroDataAccess, "get_macro_snapshot", staticmethod(fake_snapshot))
        as_of = dt.date(2024, 3, 1)
        analyzer.MacroAnalyzer().get_factor_macro_fitness(["momentum"], as_of=as_of)
        assert captured["as_of"] == as_of
