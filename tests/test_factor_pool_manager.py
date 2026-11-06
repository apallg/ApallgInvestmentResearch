"""
因子池管理器单元测试 — CRUD、时间维度、同步、备份还原、精选集
"""
import pytest
import json
import datetime
from unittest.mock import patch, Mock, MagicMock

from core.domain.factors.pool import (
    FactorPoolManager, get_pool, reload_pool,
    DEFAULT_STATE, STATE_PATH, _lock,
)


@pytest.fixture
def mgr():
    """每次测试用独立 Manager, 不污染全局 _pool"""
    mgr = FactorPoolManager()
    mgr.state = {
        "white_box": {},
        "traditional": dict(DEFAULT_STATE["traditional"]),
        "history": {"white_box": [], "traditional": []},
        "monitoring": {},
        "updated_at": "",
    }
    # 禁用 _save 避免写文件/MySQL
    mgr._save = lambda: None
    return mgr


class TestAddRemove:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_add_factor(self, mgr):
        mgr.add_factor("test_mom_20", "测试动量20", template_key="momentum",
                       params={"n": 20}, category="white_box")
        assert "test_mom_20" in mgr.state["white_box"]
        e = mgr.state["white_box"]["test_mom_20"]
        assert e["label"] == "测试动量20"
        assert e["template_key"] == "momentum"
        assert e["params"]["n"] == 20
        assert e["active"] is True

    @pytest.mark.unit
    @pytest.mark.p0
    def test_add_factor_default_category(self, mgr):
        mgr.add_factor("f1", "因子1")
        assert "f1" in mgr.state["white_box"]

    @pytest.mark.unit
    @pytest.mark.p0
    def test_add_factor_to_named_curated_set(self, mgr):
        mgr.state["curated"] = {
            "trend_momentum": {
                "created": "2026-01-01T00:00:00", "flat_list": [],
                "factors": {}, "stats": {},
            }
        }
        target = mgr.add_factor("mined", "挖掘因子", expr="$close", curated_set="trend_momentum")
        assert target == "trend_momentum"
        assert mgr.state["curated"]["trend_momentum"]["flat_list"] == ["mined"]
        assert mgr.state["curated"]["trend_momentum"]["factors"]["挖掘因子"] == ["mined"]

    @pytest.mark.unit
    @pytest.mark.p0
    def test_remove_factor(self, mgr):
        mgr.add_factor("f1", "因子1")
        assert mgr.remove_factor("f1") is True
        assert mgr.state["white_box"]["f1"]["active"] is False


@pytest.mark.unit
def test_curated_list_exposes_qualification_and_defaults_legacy_to_unknown(mgr):
    mgr.save_curated_set("research", {"trend": [("f1", "f1", None, None)]}, {}, {})
    mgr.state["curated"]["legacy"] = {"created": "2020-01-01", "flat_list": ["old"]}

    listed = {item["name"]: item for item in mgr.list_curated_sets()}

    assert listed["research"]["qualification"] == "research_only"
    assert listed["research"]["production_eligible"] is False
    assert listed["legacy"]["qualification"] == "unknown"
    assert listed["legacy"]["production_eligible"] is False

    @pytest.mark.unit
    @pytest.mark.p1
    def test_remove_nonexistent(self, mgr):
        assert mgr.remove_factor("nonexistent") is False

    @pytest.mark.unit
    @pytest.mark.p0
    def test_delete_factor_hard(self, mgr):
        mgr.add_factor("f1", "因子1")
        assert mgr.delete_factor("f1") is True
        assert "f1" not in mgr.state["white_box"]

    @pytest.mark.unit
    @pytest.mark.p0
    def test_toggle_factor(self, mgr):
        mgr.add_factor("f1", "因子1")
        assert mgr.toggle_factor("f1", False) is True
        assert mgr.state["white_box"]["f1"]["active"] is False
        assert mgr.toggle_factor("f1", True) is True
        assert mgr.state["white_box"]["f1"]["active"] is True

    @pytest.mark.unit
    @pytest.mark.p1
    def test_toggle_nonexistent(self, mgr):
        assert mgr.toggle_factor("no", True) is False


class TestGetActiveFactors:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_empty_pool_returns_empty(self, mgr):
        mgr.state["white_box"] = {}
        for k in mgr.state["traditional"]:
            mgr.state["traditional"][k]["active"] = False
        result = mgr.get_active_factors()
        assert result == {}

    @pytest.mark.unit
    @pytest.mark.p0
    def test_returns_active_only(self, mgr):
        mgr.add_factor("f1", "活跃")
        mgr.add_factor("f2", "停用")
        mgr.state["white_box"]["f2"]["active"] = False
        result = mgr.get_active_factors()
        assert "f1" in result
        assert "f2" not in result

    @pytest.mark.unit
    @pytest.mark.p0
    def test_as_of_historical(self, mgr):
        mgr.add_factor("f_old", "旧因子")
        mgr.add_factor("f_new", "新因子")
        old_spec = {"f_old": {"label": "旧因子", "active": True}}
        both_specs = {**old_spec, "f_new": {"label": "新因子", "active": True}}
        mgr.state["history"]["white_box"] = [
            {"effective_date": "2025-01-01", "keys": ["f_old"], "factor_specs": old_spec, "provenance": "explicit_snapshot"},
            {"effective_date": "2025-06-01", "keys": ["f_old", "f_new"], "factor_specs": both_specs, "provenance": "explicit_snapshot"},
        ]
        result_jan = mgr.get_active_factors(as_of="2025-01-15")
        assert "f_old" in result_jan
        assert "f_new" not in result_jan

        result_jun = mgr.get_active_factors(as_of="2025-06-15")
        assert "f_old" in result_jun
        assert "f_new" in result_jun

    @pytest.mark.unit
    @pytest.mark.p1
    def test_as_of_before_all_history(self, mgr):
        mgr.add_factor("f1", "因子1")
        # 同时禁用 traditional 以避免干扰
        for k in mgr.state["traditional"]:
            mgr.state["traditional"][k]["active"] = False
        mgr.state["history"]["white_box"] = [
            ("2025-06-01", ["f1"]),
        ]
        # as_of 早于可信历史记录时，不得回退到可能包含未来状态的当前快照
        result = mgr.get_active_factors(as_of="2024-01-01")
        assert result == {}

    @pytest.mark.unit
    @pytest.mark.p1
    def test_as_of_ignores_legacy_anchor_history_and_current_fallback(self, mgr):
        mgr.add_factor("future_factor", "未来验证因子")
        mgr.state["history"]["white_box"] = [("2025-06-01", ["future_factor"])]

        assert mgr.get_active_factors(as_of="2025-01-01") == {}

    @pytest.mark.unit
    @pytest.mark.p1
    def test_as_of_rolling_history_starts_at_label_maturity(self, mgr):
        mgr.add_factor("rolled", "滚动因子", expr="$close")
        mgr.record_history(["rolled"], datetime.date(2025, 4, 1), source_anchor=datetime.date(2025, 1, 1))

        assert mgr.get_active_factors(as_of="2025-03-31") == {}
        historical = mgr.get_active_factors(as_of="2025-04-01")
        assert historical["rolled"]["expr"] == "$close"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_as_of_uses_definition_frozen_in_historical_snapshot(self, mgr):
        mgr.add_factor("factor_a", "Original", expr="$close", params={"period": 5})
        mgr.sync_active_set(
            ["factor_a"],
            factor_meta={"factor_a": {"label": "Original", "expr": "$close", "params": {"period": 5}}},
            anchor_date=datetime.date(2025, 1, 1),
        )
        mgr.sync_active_set(
            ["factor_a"],
            factor_meta={"factor_a": {"label": "Changed", "expr": "$open", "params": {"period": 20}}},
            anchor_date=datetime.date(2025, 3, 1),
        )

        old = mgr.get_active_factors(as_of="2025-02-01")["factor_a"]
        new = mgr.get_active_factors(as_of="2025-04-01")["factor_a"]
        assert (old["label"], old["expr"], old["params"]) == ("Original", "$close", {"period": 5})
        assert (new["label"], new["expr"], new["params"]) == ("Changed", "$open", {"period": 20})

    @pytest.mark.unit
    @pytest.mark.p1
    def test_as_of_rejects_key_only_history_without_factor_definitions(self, mgr):
        mgr.state["history"]["white_box"] = [{
            "effective_date": "2025-01-01",
            "keys": ["factor_a"],
            "provenance": "explicit_snapshot",
        }]

        assert mgr.get_active_factors(as_of="2025-02-01") == {}

    @pytest.mark.unit
    @pytest.mark.p1
    def test_category_filter(self, mgr):
        mgr.add_factor("wf", "白盒", category="white_box")
        mgr.state["traditional"]["tech"]["active"] = True
        result = mgr.get_active_factors(category="white_box")
        assert "wf" in result
        assert "tech" not in result

    @pytest.mark.unit
    @pytest.mark.p2
    def test_pool_category_in_result(self, mgr):
        mgr.add_factor("f1", "测试", category="white_box")
        result = mgr.get_active_factors()
        assert result["f1"]["pool_category"] == "white_box"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_missing_data_safety_is_persisted_as_unknown(self, mgr):
        mgr.add_factor("manual_factor", "手工因子")

        assert mgr.get_active_factors()["manual_factor"]["data_safety"] == "unknown"


class TestGetAllFactors:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_includes_inactive(self, mgr):
        mgr.add_factor("f1", "活跃")
        mgr.add_factor("f2", "停用")
        mgr.state["white_box"]["f2"]["active"] = False
        result = mgr.get_all_factors()
        assert "f1" in result
        assert "f2" in result
        assert result["f1"]["key"] == "f1"


class TestMonitoring:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_update_monitoring(self, mgr):
        mgr.update_monitoring("f1", 0.15)
        assert "f1" in mgr.state["monitoring"]
        mon = mgr.state["monitoring"]["f1"]
        assert len(mon["ir_history"]) == 1
        assert mon["ir_history"][0][1] == 0.15

    @pytest.mark.unit
    @pytest.mark.p1
    def test_monitoring_capped_12_months(self, mgr):
        for i in range(15):
            mgr.update_monitoring("f1", 0.1)
        assert len(mgr.state["monitoring"]["f1"]["ir_history"]) == 12

    @pytest.mark.unit
    @pytest.mark.p1
    def test_auto_prune(self, mgr):
        mgr.add_factor("f_weak", "弱因子")
        mgr.state["monitoring"]["f_weak"] = {
            "ir_history": [("2025-01", 0.01), ("2025-02", 0.01)],
            "last_check": "2025-02"
        }
        removed = mgr.auto_prune(ir_threshold=0.015, consecutive_months=2)
        assert "f_weak" in removed

    @pytest.mark.unit
    @pytest.mark.p1
    def test_auto_prune_not_enough_months(self, mgr):
        mgr.add_factor("f_ok", "OK因子")
        mgr.state["monitoring"]["f_ok"] = {
            "ir_history": [("2025-01", 0.01)],
            "last_check": "2025-01"
        }
        removed = mgr.auto_prune(ir_threshold=0.015, consecutive_months=2)
        assert len(removed) == 0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_auto_prune_good_factor_stays(self, mgr):
        mgr.add_factor("f_good", "好因子")
        mgr.state["monitoring"]["f_good"] = {
            "ir_history": [("2025-01", 0.05), ("2025-02", 0.04)],
            "last_check": "2025-02"
        }
        removed = mgr.auto_prune(ir_threshold=0.015, consecutive_months=2)
        assert "f_good" not in removed


class TestSyncActiveSet:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_sync_adds_new_factors(self, mgr):
        added, deactivated = mgr.sync_active_set(
            ["new_f1", "new_f2"],
            factor_meta={
                "new_f1": {"label": "新因子1"},
                "new_f2": {"label": "新因子2", "expr": "close/Ref(close,5)-1"}
            },
            category="white_box",
            anchor_date=datetime.date(2025, 6, 1),
        )
        assert set(added) == {"new_f1", "new_f2"}
        assert mgr.state["white_box"]["new_f1"]["active"]
        assert mgr.state["white_box"]["new_f2"]["expr"] == "close/Ref(close,5)-1"

    @pytest.mark.unit
    @pytest.mark.p0
    def test_sync_deactivates_old(self, mgr):
        mgr.add_factor("old_f", "旧因子")
        added, deactivated = mgr.sync_active_set(["new_f"], category="white_box")
        assert "old_f" in deactivated
        assert mgr.state["white_box"]["old_f"]["active"] is False

    @pytest.mark.unit
    @pytest.mark.p0
    def test_sync_reactivates_existing(self, mgr):
        mgr.add_factor("f1", "因子1")
        mgr.state["white_box"]["f1"]["active"] = False
        added, deactivated = mgr.sync_active_set(["f1"], category="white_box")
        assert "f1" not in added
        assert mgr.state["white_box"]["f1"]["active"] is True

    @pytest.mark.unit
    @pytest.mark.p0
    def test_sync_none_keys_skipped(self, mgr):
        added, deactivated = mgr.sync_active_set(None)
        assert added == []
        assert deactivated == []

    @pytest.mark.unit
    @pytest.mark.p0
    def test_sync_records_history(self, mgr):
        anchor = datetime.date(2025, 7, 1)
        added, _ = mgr.sync_active_set(
            ["f_a", "f_b"], anchor_date=anchor
        )
        hist = mgr.state["history"]["white_box"]
        assert len(hist) == 1
        assert hist[0]["effective_date"] == "2025-07-01"
        assert set(hist[0]["keys"]) == {"f_a", "f_b"}

    @pytest.mark.unit
    @pytest.mark.p1
    def test_sync_overwrites_same_anchor(self, mgr):
        anchor = datetime.date(2025, 7, 1)
        mgr.sync_active_set(["f_a"], anchor_date=anchor)
        mgr.sync_active_set(["f_b", "f_c"], anchor_date=anchor)
        hist = mgr.state["history"]["white_box"]
        assert len(hist) == 1  # 覆盖, 不是追加
        assert set(hist[0]["keys"]) == {"f_b", "f_c"}

    @pytest.mark.unit
    @pytest.mark.p1
    def test_sync_hybrid_category(self, mgr):
        added, _ = mgr.sync_active_set(["h1", "h2"], category="hybrid",
                                       anchor_date=datetime.date(2025, 8, 1))
        assert "h1" in mgr.state.get("hybrid", {})


class TestRecordHistory:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_record_history_does_not_change_active(self, mgr):
        mgr.add_factor("f1", "因子1")
        mgr.record_history(
            ["f1", "f2"], datetime.date(2025, 6, 1),
            factor_specs={"f1": {"label": "因子1"}, "f2": {"label": "因子2"}},
        )
        assert mgr.state["white_box"]["f1"]["active"] is True
        assert "f2" not in mgr.state["white_box"]  # 只记录, 不入库

    @pytest.mark.unit
    @pytest.mark.p1
    def test_record_history_sorted(self, mgr):
        mgr.record_history(["b"], datetime.date(2025, 3, 1), factor_specs={"b": {"expr": "$close"}})
        mgr.record_history(["a"], datetime.date(2025, 1, 1), factor_specs={"a": {"expr": "$open"}})
        dates = [h["effective_date"] for h in mgr.state["history"]["white_box"]]
        assert dates == sorted(dates)


class TestCuratedSets:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_save_and_get_curated(self, mgr):
        factors = {"momentum": [("m1", 0.8, 0.9, 0.7)]}
        mgr.save_curated_set("v1", factors, {"description": "测试"}, {"n": 1})
        result = mgr.get_curated_set("v1")
        assert result is not None
        assert result["description"] == "测试"
        assert result["flat_list"] == ["m1"]

    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_latest_curated(self, mgr):
        mgr.save_curated_set("v1", {}, {}, {})
        mgr.save_curated_set("v2", {}, {}, {})
        latest = mgr.get_curated_factors("latest")
        curated = mgr.list_curated_sets()
        assert len(curated) == 2

    @pytest.mark.unit
    @pytest.mark.p1
    def test_delete_curated(self, mgr):
        mgr.save_curated_set("v1", {}, {}, {})
        assert mgr.delete_curated_set("v1") is True
        assert mgr.get_curated_set("v1") is None

    @pytest.mark.unit
    @pytest.mark.p2
    def test_auto_cleanup_old_curated(self, mgr):
        for i in range(7):
            mgr.save_curated_set(f"v{i}", {}, {}, {})
        # 清理在 delete_curated_set 中触发, 不在 save 时触发
        assert mgr.delete_curated_set("v0") is True
        curated = mgr.list_curated_sets()
        assert len(curated) <= 5


class TestSourceWeights:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_source_weights_normalizes(self, mgr):
        mgr.add_factor("tech_1", "技术1", category="white_box")
        mgr.state["traditional"]["tech"]["active"] = True
        weights = mgr.get_source_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert "tech" in weights

    @pytest.mark.unit
    @pytest.mark.p0
    def test_get_source_weights_empty(self, mgr):
        mgr.state["white_box"] = {}
        for k in mgr.state["traditional"]:
            mgr.state["traditional"][k]["active"] = False
        weights = mgr.get_source_weights()
        assert weights == {}


class TestSingleton:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_pool_returns_same_instance(self):
        import core.domain.factors.pool as fpm
        fpm._pool = None
        p1 = get_pool()
        p2 = get_pool()
        assert p1 is p2

    @pytest.mark.unit
    @pytest.mark.p1
    def test_reload_creates_new_instance(self):
        import core.domain.factors.pool as fpm
        p1 = get_pool()
        p2 = reload_pool()
        assert p1 is not p2


class TestDeepMerge:

    @pytest.mark.unit
    @pytest.mark.p2
    def test_adds_missing_keys(self):
        base = {"a": 1}
        defaults = {"a": 99, "b": 2}
        merged = FactorPoolManager._deep_merge(base, defaults)
        assert merged["a"] == 1  # 不覆盖
        assert merged["b"] == 2  # 补缺失

    @pytest.mark.unit
    @pytest.mark.p2
    def test_nested_merge(self):
        base = {"x": {"y": 1}}
        defaults = {"x": {"y": 99, "z": 2}}
        merged = FactorPoolManager._deep_merge(base, defaults)
        assert merged["x"]["y"] == 1
        assert merged["x"]["z"] == 2
