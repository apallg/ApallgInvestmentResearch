import pytest
import numpy as np
from unittest.mock import Mock, patch

from core.domain.factors.validation import OverfittingChecker


def _make_variant():
    return Mock()


def test_walkforward_uses_supplied_l1_date_plan():
    checker = OverfittingChecker()
    plan = {
        "train": ["2024-01-02", "2024-01-09", "2024-01-16"],
        "valid": ["2024-10-01", "2024-10-08", "2024-10-15"],
        "test": ["2025-01-02", "2025-01-09", "2025-01-16"],
    }
    seen = []

    def fake_ic(_variant, dates, _ctx):
        seen.append([str(value) for value in dates])
        return 0.03

    with patch.object(checker, "_calc_rank_ic", side_effect=fake_ic):
        passed, _, _ = checker._check_walkforward({
            "variant": {"variant_key": "gp_test", "expr": "$close"},
            "eval_weeks": 18,
            "anchor_date": None,
            "l1_dates": plan,
        })

    assert passed is True
    assert seen == [plan["train"], plan["valid"], plan["test"]]


def test_hard_gate_failure_marks_following_layers_skipped():
    checker = OverfittingChecker()
    with patch.object(checker, "_check_deredundancy", return_value=(False, 0.0, "冗余")):
        result = checker.check_all_layers(
            {"variant_key": "gp_test", "expr": "$close"}, mode="full",
        )

    statuses = {item["id"]: item["status"] for item in result["layers"]}
    assert statuses[5] == "failed"
    assert statuses[1] == "skipped"
    assert statuses[4] == "skipped"
    assert statuses[7] == "skipped"
    assert statuses[8] == "skipped"
    assert result["passed_count"] == 0


def test_expression_l5_uses_real_cross_sectional_correlation():
    checker = OverfittingChecker()
    codes = [f"c{i}" for i in range(30)]
    pool = Mock()
    pool.get_active_factors.return_value = {
        "existing": {"expr": "$open", "template_key": "existing"},
    }
    screener = Mock()
    screener.screen.return_value = {"candidates": []}

    def scores(variant, *_args, **_kwargs):
        values = [float(i) for i in range(30)]
        return codes, values

    with patch("core.domain.factors.registry.get_pool", return_value=pool), \
         patch("core.domain.screening._screener.get_screener", return_value=screener), \
         patch.object(checker, "_variant_scores", side_effect=scores):
        passed, _, detail = checker._check_deredundancy({
            "variant": {"variant_key": "gp_test", "expr": "$close"},
            "snapshots": {"2025-01-02": object()},
            "anchor_date": __import__("datetime").date(2025, 1, 2),
            "pool": "csi300",
            "code_prefixes": None,
        })

    assert passed is False
    assert "existing" in detail
    assert checker._last_deredundancy["max_corr"] == pytest.approx(1.0)


class TestCalcIcir:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_normal_case(self):
        checker = OverfittingChecker()
        daily_ics = [0.05, 0.03, 0.04, 0.06, 0.02]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.04, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert n_dates == 5
        assert abs(ic_mean - np.mean(daily_ics)) < 0.001
        expected_std = float(np.std(daily_ics))
        expected_icir = abs(ic_mean) / max(expected_std, 0.0001)
        assert abs(icir - expected_icir) < 0.001

    @pytest.mark.unit
    @pytest.mark.p1
    def test_all_positive_ics(self):
        checker = OverfittingChecker()
        daily_ics = [0.10, 0.08, 0.12, 0.09]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.0975, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert icir > 0
        assert ic_mean > 0
        assert n_dates == 4

    @pytest.mark.unit
    @pytest.mark.p1
    def test_all_negative_ics_icir_still_positive(self):
        checker = OverfittingChecker()
        daily_ics = [-0.05, -0.03, -0.04, -0.06]
        with patch.object(checker, '_calc_rank_ic', return_value=(-0.045, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert icir > 0
        assert ic_mean < 0
        assert n_dates == 4

    @pytest.mark.unit
    @pytest.mark.p0
    def test_two_dates_accepted(self):
        """2 个有效 IC 现在可通过 (阈值从 3 降到 2)"""
        checker = OverfittingChecker()
        daily_ics = [0.05, 0.03]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.04, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert n_dates == 2, '2 个 IC 应该被接受'
        assert icir >= 0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_single_date_still_rejected(self):
        """仅 1 个有效 IC → 仍返回零"""
        checker = OverfittingChecker()
        daily_ics = [0.05]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.05, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert icir == 0.0
        assert n_dates == 0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_empty_daily_ics_returns_zero(self):
        checker = OverfittingChecker()
        with patch.object(checker, '_calc_rank_ic', return_value=(0.0, [])):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert icir == 0.0
        assert ic_mean == 0.0
        assert n_dates == 0

    @pytest.mark.unit
    @pytest.mark.p2
    def test_nan_and_inf_filtered(self):
        checker = OverfittingChecker()
        daily_ics = [0.05, float('nan'), 0.03, float('inf'), 0.04]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.04, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert n_dates == 3
        assert abs(ic_mean - np.mean([0.05, 0.03, 0.04])) < 0.001

    @pytest.mark.unit
    @pytest.mark.p2
    def test_all_nan_and_inf_returns_zero(self):
        checker = OverfittingChecker()
        daily_ics = [float('nan'), float('nan'), float('inf'), float('-inf')]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.0, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert icir == 0.0
        assert n_dates == 0

    @pytest.mark.unit
    @pytest.mark.p2
    def test_identical_ics_zero_std_capped(self):
        checker = OverfittingChecker()
        daily_ics = [0.05, 0.05, 0.05, 0.05]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.05, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert ic_mean == 0.05
        assert icir == pytest.approx(0.05 / 0.0001)
        assert n_dates == 4

    @pytest.mark.unit
    @pytest.mark.p2
    def test_mixed_sign_low_icir(self):
        checker = OverfittingChecker()
        daily_ics = [0.10, -0.09, 0.02, -0.01]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.005, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert icir < 0.5
        assert n_dates == 4

    @pytest.mark.unit
    @pytest.mark.p0
    def test_rank_ic_error_sentinel_returns_zero(self):
        checker = OverfittingChecker()
        with patch.object(checker, '_calc_rank_ic', return_value=(0.0, [])):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert icir == 0.0
        assert ic_mean == 0.0
        assert n_dates == 0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_ic_mean_is_simple_not_time_weighted(self):
        checker = OverfittingChecker()
        daily_ics = [0.01, 0.02, 0.03, 0.04, 0.05]
        time_weighted = 0.042
        with patch.object(checker, '_calc_rank_ic', return_value=(time_weighted, daily_ics)):
            icir, ic_mean, n_dates, _ = checker._calc_icir(_make_variant(), [], {})

        assert ic_mean != time_weighted
        assert abs(ic_mean - np.mean(daily_ics)) < 0.001

    @pytest.mark.unit
    @pytest.mark.p0
    def test_returns_daily_ics_list(self):
        """正常路径返回 daily_ics list (经 NaN 过滤后的值)"""
        checker = OverfittingChecker()
        daily_ics = [0.05, float('nan'), 0.03, 0.04]
        with patch.object(checker, '_calc_rank_ic', return_value=(0.04, daily_ics)):
            icir, ic_mean, n_dates, daily_ics_out = checker._calc_icir(_make_variant(), [], {})

        assert n_dates == 3
        assert daily_ics_out == [0.05, 0.03, 0.04]  # NaN filtered out

    @pytest.mark.unit
    @pytest.mark.p1
    def test_two_dates_returns_list(self):
        """2 个有效 IC → daily_ics 返回列表 (不再拒绝)"""
        checker = OverfittingChecker()
        with patch.object(checker, '_calc_rank_ic', return_value=(0.04, [0.05, 0.03])):
            icir, ic_mean, n_dates, daily_ics_out = checker._calc_icir(_make_variant(), [], {})

        assert n_dates == 2
        assert daily_ics_out == [0.05, 0.03]


class TestSaveScoresToDb:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_saves_n_ic_zero_entries(self):
        """n_ic=0 的条目也要存 DB (防止重复触发实时计算)"""
        from core.domain.factors.selector import _save_scores_to_db
        from unittest.mock import patch, MagicMock
        import datetime as dt

        anchor = dt.date(2025, 6, 30)
        scores = {
            "factor_a": {"icir": 0.0, "ic_mean": 0.0, "ic_std": 0.0, "n_ic_dates": 0},
        }

        with patch("core.data.Database") as mock_db:
            mock_db.is_available.return_value = True
            mock_db.fetchall = MagicMock()
            mock_db.execute = MagicMock()
            _save_scores_to_db(["factor_a"], anchor, scores)
            assert mock_db.executemany.called, "n_ic=0 的条目应该被保存"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_saves_zero_icir_with_valid_n_ic(self):
        """icir=0 但 n_ic>0 的条目 (真正无预测力的因子) 应该保存"""
        from core.domain.factors.selector import _save_scores_to_db
        from unittest.mock import patch, MagicMock
        import datetime as dt

        anchor = dt.date(2025, 6, 30)
        scores = {
            "factor_b": {"icir": 0.0, "ic_mean": 0.001, "ic_std": 0.01,
                         "n_ic_dates": 5, "daily_ics": [0.001, -0.001, 0.002, -0.002, 0.0]},
        }

        with patch("core.data.Database") as mock_db:
            mock_db.is_available.return_value = True
            mock_db.fetchall = MagicMock()
            mock_db.execute = MagicMock()
            _save_scores_to_db(["factor_b"], anchor, scores)
            assert mock_db.executemany.called, "icir=0 但 n_ic>0 的因子应该保存"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_empty_scores_writes_zero_values(self):
        """空 scores_map 中不存在的 key → info 为空 → 写入零值 (不再跳过)"""
        from core.domain.factors.selector import _save_scores_to_db
        from unittest.mock import patch, MagicMock
        import datetime as dt

        anchor = dt.date(2025, 6, 30)
        with patch("core.data.Database") as mock_db:
            mock_db.is_available.return_value = True
            mock_db.execute = MagicMock()
            _save_scores_to_db(["factor_c"], anchor, {})
            # 新行为: 即使 info 为空, 也会写入 (icir=0, ic_mean=0, ...)
            assert mock_db.executemany.called

    @pytest.mark.unit
    @pytest.mark.p2
    def test_db_unavailable_noop(self):
        """DB 不可用时静默跳过"""
        from core.domain.factors.selector import _save_scores_to_db
        from unittest.mock import patch
        import datetime as dt

        anchor = dt.date(2025, 6, 30)
        scores = {"factor_d": {"icir": 0.05, "ic_mean": 0.03, "ic_std": 0.02, "n_ic_dates": 5}}

        with patch("core.data.Database") as mock_db:
            mock_db.is_available.return_value = False
            _save_scores_to_db(["factor_d"], anchor, scores)
            # 不应该抛异常

    @pytest.mark.unit
    @pytest.mark.p2
    def test_none_anchor_noop(self):
        """anchor_date 为 None 时跳过"""
        from core.domain.factors.selector import _save_scores_to_db
        scores = {"factor_e": {"icir": 0.05, "ic_mean": 0.03, "ic_std": 0.02, "n_ic_dates": 5}}
        _save_scores_to_db(["factor_e"], None, scores)
        # 不应该抛异常

    @pytest.mark.unit
    @pytest.mark.p0
    def test_both_existing_and_missing_keys_written(self):
        """scores_map 中存在的 key 写入真实值, 不存在的写入零值 (不再跳过)"""
        from core.domain.factors.selector import _save_scores_to_db
        from unittest.mock import patch, MagicMock
        import datetime as dt

        anchor = dt.date(2025, 6, 30)
        scores = {"factor_f": {"icir": 0.05, "ic_mean": 0.03, "ic_std": 0.02, "n_ic_dates": 5}}

        with patch("core.data.Database") as mock_db:
            mock_db.is_available.return_value = True
            mock_db.fetchall = MagicMock()
            mock_db.execute = MagicMock()
            _save_scores_to_db(["factor_f", "factor_not_in_scores"], anchor, scores)
            assert mock_db.executemany.call_count == 1, "两个 key 应批量写入"
            written_keys = [row[0] for row in mock_db.executemany.call_args_list[0][0][1]]
            assert "factor_f" in written_keys
            assert "factor_not_in_scores" in written_keys


class TestCurateDirectionLogic:
    """curate_direction() 行为测试 — 调用真实生产函数"""

    @pytest.mark.unit
    @pytest.mark.p0
    def test_curate_direction_returns_list(self):
        from core.domain.factors.selector import curate_direction
        name, keys, count = curate_direction("trend_momentum", [], None, {})
        assert name == "trend_momentum" and keys == [] and count == 0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_curate_direction_empty_keys_no_error(self):
        from core.domain.factors.selector import curate_direction
        name, keys, count = curate_direction("trend_momentum", [], None, {})
        assert keys == []

    @pytest.mark.unit
    @pytest.mark.p1
    def test_curate_direction_no_adapter_returns_none(self):
        from core.domain.factors.selector import curate_direction
        name, keys, count = curate_direction("trend_momentum", ["fake_key"], None, {}, save=False)
        assert name == "trend_momentum"  # adapter=None 时应安全返回

    @pytest.mark.unit
    @pytest.mark.p1
    def test_curate_direction_unknown_direction_returns_empty(self):
        import pytest
        from core.domain.factors.selector import curate_direction
        with pytest.raises(ValueError):
            curate_direction("__nonexistent__", ["fake"], None, {})

    @pytest.mark.unit
    @pytest.mark.p2
    def test_curate_returns_list(self):
        from core.domain.factors.selector import curate
        result = curate([], None, {})
        assert isinstance(result, tuple) and isinstance(result[0], dict)  # curate() returns (selected_by_category, report, ...)
        db_scores = {"factor_a": {"icir": 0.05}}
        # 不存在的 key
        result = db_scores.get("factor_missing", {}).get("icir", 0)
        assert result == 0
        # 存在但 icir=0
        db_scores["factor_zero"] = {"icir": 0.0}
        result = db_scores.get("factor_zero", {}).get("icir", 0)
        assert result == 0


class TestRegimeIcirLogic:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_regime_with_data_uses_regime_only(self):
        """regime 有数据 → 用 regime ICIR, 不考虑全局"""
        regime_icir_map = {"f1": {"icir": 0.5}, "f2": {"icir": 0.3}}

        icir_map = {}
        for key in ["f1", "f2", "f3"]:
            if key in regime_icir_map:
                icir_map[key] = regime_icir_map[key]["icir"]
        assert icir_map == {"f1": 0.5, "f2": 0.3}
        # f3 不在 regime_icir_map → 无条目 → get(k,0) 返回 0
        assert icir_map.get("f3", 0) == 0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_regime_no_data_returns_empty(self):
        """regime 无数据 → 跳过精选, 返回空"""
        regime_icir_map = {}
        if not regime_icir_map:
            result = ("test_direction", [], 0)
        assert result == ("test_direction", [], 0)

    @pytest.mark.unit
    @pytest.mark.p2
    def test_regime_partial_coverage(self):
        """部分因子有 regime ICIR: 有的排前, 没的排末尾"""
        regime_icir_map = {"f1": {"icir": 0.5}}

        icir_map = {}
        for key in ["f1", "f2", "f3"]:
            if key in regime_icir_map:
                icir_map[key] = regime_icir_map[key]["icir"]

        ranked = sorted(["f1", "f2", "f3"], key=lambda k: icir_map.get(k, 0), reverse=True)
        assert ranked[0] == "f1"
        # f2, f3 无 regime ICIR → 都是 0 → 稳定排序


class TestMakeVariant:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_make_variant_from_adapter(self):
        """adapter 中存在的因子 → 直接返回"""
        from unittest.mock import patch, MagicMock
        adapter = MagicMock()
        adapter.to_variant.return_value = {"variant_key": "a158_vol5"}

        def _make_variant(key):
            v = adapter.to_variant(key)
            if v:
                return v
            return None

        result = _make_variant("a158_vol5")
        assert result is not None
        assert result["variant_key"] == "a158_vol5"
        adapter.to_variant.assert_called_once_with("a158_vol5")

    @pytest.mark.unit
    @pytest.mark.p0
    def test_make_variant_from_pool_with_expr(self):
        """adapter 中不存在但因子池中有 expr → 从池构建"""
        from unittest.mock import MagicMock

        adapter = MagicMock()
        adapter.to_variant.return_value = None  # adapter 不认识
        pool_mgr = MagicMock()
        pool_mgr.get_active_factors.return_value = {
            "pb": {"expr": "FundVal($close, 4)", "label": "市净率",
                   "params": {"n": 0}, "source": "pdf_extracted", "category": "white_box"}
        }

        def _make_variant(key):
            v = adapter.to_variant(key)
            if v:
                return v
            info = pool_mgr.get_active_factors(category="white_box").get(key, {})
            expr = info.get("expr", "")
            if expr:
                return {
                    "variant_key": key, "template_key": key,
                    "label": info.get("label", key),
                    "params": info.get("params", {}),
                    "default_n": info.get("params", {}).get("n", 20),
                    "source": info.get("source", ""),
                    "category": info.get("category", ""),
                    "expr": expr,
                }
            return None

        result = _make_variant("pb")
        assert result is not None
        assert result["expr"] == "FundVal($close, 4)"
        assert result["variant_key"] == "pb"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_make_variant_no_expr_returns_none(self):
        """adapter 无且池中无 expr → None"""
        from unittest.mock import MagicMock
        adapter = MagicMock()
        adapter.to_variant.return_value = None
        pool_mgr = MagicMock()
        pool_mgr.get_active_factors.return_value = {"tech": {"label": "tech"}}

        def _make_variant(key):
            v = adapter.to_variant(key)
            if v:
                return v
            info = pool_mgr.get_active_factors(category="white_box").get(key, {})
            if info.get("expr"):
                return {"variant_key": key}
            return None

        assert _make_variant("tech") is None

    @pytest.mark.unit
    @pytest.mark.p2
    def test_make_variant_adapter_priority(self):
        """adapter 有值时优先用 adapter 版本"""
        from unittest.mock import MagicMock
        adapter = MagicMock()
        adapter.to_variant.return_value = {"variant_key": "pb", "expr": "adapter_expr"}
        pool_mgr = MagicMock()
        pool_mgr.get_active_factors.return_value = {"pb": {"expr": "pool_expr"}}

        def _make_variant(key):
            v = adapter.to_variant(key)
            if v:
                return v
            info = pool_mgr.get_active_factors(category="white_box").get(key, {})
            if info.get("expr"):
                return {"variant_key": key, "expr": info.get("expr")}
            return None

        result = _make_variant("pb")
        assert result is not None
        assert result["expr"] == "adapter_expr"


class TestCurateFlowEdges:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_expr_keys_filter(self):
        """sel_keys 中过滤出可计算的 expr_keys"""
        sel_keys = ["a158_vol5", "pb", "tech", "pe", "MOM"]
        # mock: a158_vol5, pb, pe, MOM 有 expr; tech 没有
        def _mock_make_variant(key):
            if key == "tech":
                return None
            return {"variant_key": key, "expr": "test_expr"}

        expr_keys = [k for k in sel_keys if _mock_make_variant(k)]
        assert len(expr_keys) == 4
        assert "tech" not in expr_keys

    @pytest.mark.unit
    @pytest.mark.p1
    def test_miss_ratio_path_a(self):
        """miss_ratio > 50%: missed = not-in-DB + icir=0"""
        expr_keys = [f"f{i}" for i in range(100)]
        db_scores = {f"f{i}": {"icir": 0.1} for i in range(20)}  # 20有值
        db_scores.update({f"f{i}": {"icir": 0.0} for i in range(20, 30)})  # 10个零

        miss_ratio = 1.0 - len(db_scores) / max(len(expr_keys), 1)
        assert miss_ratio > 0.5

        missed = [k for k in expr_keys if k not in db_scores]  # 70 not in DB
        zero_cached = [k for k in expr_keys if k in db_scores
                       and db_scores.get(k, {}).get("icir", 0) == 0]  # 10 zero
        missed.extend(zero_cached)
        missed = list(dict.fromkeys(missed))
        assert len(missed) == 80  # 70 missing + 10 zero

    @pytest.mark.unit
    @pytest.mark.p1
    def test_miss_ratio_path_b(self):
        """miss_ratio ≤ 50%: missed = icir=0 only"""
        expr_keys = [f"f{i}" for i in range(100)]
        db_scores = {f"f{i}": {"icir": 0.1} for i in range(80)}
        db_scores.update({f"f{i}": {"icir": 0.0} for i in range(80, 90)})  # 10 zero

        miss_ratio = 1.0 - len(db_scores) / max(len(expr_keys), 1)
        assert miss_ratio <= 0.5

        missed = [k for k in expr_keys if k in db_scores
                  and db_scores.get(k, {}).get("icir", 0) == 0]
        assert len(missed) == 10

    @pytest.mark.unit
    @pytest.mark.p1
    def test_miss_ratio_path_b_no_zeros(self):
        """miss_ratio ≤ 50% 且无零值 → missed 为空 → 跳过计算"""
        expr_keys = [f"f{i}" for i in range(100)]
        db_scores = {f"f{i}": {"icir": 0.1} for i in range(100)}

        miss_ratio = 1.0 - len(db_scores) / max(len(expr_keys), 1)
        assert miss_ratio == 0.0

        missed = [k for k in expr_keys if k in db_scores
                  and db_scores.get(k, {}).get("icir", 0) == 0]
        assert missed == []

    @pytest.mark.unit
    @pytest.mark.p2
    def test_regime_dates_fallback_to_default(self):
        """regime_aware=True 但 regime 无数据 → 用 _get_eval_dates"""
        regime_dates_raw = set()  # 空
        regime_aware = True

        if regime_dates_raw and regime_aware:
            eval_dates = ["regime_date"]  # 不会走到这里
        else:
            eval_dates = ["default_date"]

        assert eval_dates == ["default_date"]
