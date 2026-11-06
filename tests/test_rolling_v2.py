"""
滚动筛选 v2 重构验证: 预计算 + 磁盘加载 + 结果一致性
"""
import pytest
import tempfile
import os
import datetime as dt
from unittest import mock
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════
# 1. 因子面板持久化 round-trip
# ═══════════════════════════════════════════════════════

class TestPrecomputedScoresRoundTrip:
    def test_save_and_load(self):
        from core.domain.factors.validation import OverfittingChecker

        checker = OverfittingChecker()
        checker._date_expr_scores = {
            "2024-01-15": {
                "expr:momentum_20": (["000001", "000002", "000003"], [0.1, -0.2, 0.3]),
                "expr:volatility_10": (["000001", "000002"], [0.5, 0.6]),
            },
            "2024-02-15": {
                "expr:momentum_20": (["000001", "000004"], [0.15, -0.25]),
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            n = checker.save_precomputed_scores(tmpdir)
            assert n == 2

            loaded = OverfittingChecker.load_precomputed_scores(
                tmpdir, ["2024-01-15", "2024-02-15", "2024-03-15"]
            )
            assert "2024-01-15" in loaded
            assert "2024-02-15" in loaded
            assert "2024-03-15" not in loaded  # 不存在的日期静默跳过

            d = loaded["2024-01-15"]
            assert "expr:momentum_20" in d
            codes, scores = d["expr:momentum_20"]
            assert codes == ["000001", "000002", "000003"]
            assert scores == [0.1, -0.2, 0.3]

    def test_save_empty_scores(self):
        from core.domain.factors.validation import OverfittingChecker

        checker = OverfittingChecker()
        checker._date_expr_scores = {"2024-01-15": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            n = checker.save_precomputed_scores(tmpdir)
            assert n == 0  # 空 scores 不写文件

    def test_load_nonexistent_dir(self):
        from core.domain.factors.validation import OverfittingChecker

        loaded = OverfittingChecker.load_precomputed_scores(
            "/nonexistent/path/12345", ["2024-01-15"]
        )
        assert loaded == {}


# ═══════════════════════════════════════════════════════
# 2. _ensure_date_expr_scores 预注入
# ═══════════════════════════════════════════════════════

class TestEnsureDateExprScoresWithPreload:
    def test_preloaded_injection_short_circuits(self):
        """preloaded_scores 命中 → 跳过 snapshot + batch_eval_all"""
        from core.domain.factors.validation import OverfittingChecker

        checker = OverfittingChecker()
        checker.set_expr_variants([
            {"variant_key": "momentum_20", "expr": "Ref($close,20)/$close"},
        ])

        preloaded = {
            "2024-01-15": {
                "expr:momentum_20": (["000001", "000002"], [0.1, 0.2]),
            }
        }

        # 调用 _ensure_date_expr_scores 不传 precomputed 也不传 screener，
        # 如果 preloaded 生效就不会因为缺 snapshot 而崩溃
        checker._ensure_date_expr_scores(
            "2024-01-15", dt.date(2024, 1, 15),
            "all", None, None, None,
            preloaded_scores=preloaded
        )

        assert "2024-01-15" in checker._date_expr_scores
        cached = checker._date_expr_scores["2024-01-15"]
        assert "expr:momentum_20" in cached
        assert cached["expr:momentum_20"] == (["000001", "000002"], [0.1, 0.2])

    def test_preloaded_miss_falls_through_to_compute(self):
        """preloaded 不包含指定日期 → 回退到原始路径 (无 snapshot 时会安全返回空)"""
        from core.domain.factors.validation import OverfittingChecker

        checker = OverfittingChecker()
        checker.set_expr_variants([
            {"variant_key": "momentum_20", "expr": "Ref($close,20)/$close"},
        ])

        preloaded = {
            "2024-02-15": {"expr:momentum_20": (["000001"], [0.1])}
        }

        # 请求的日期不在 preloaded 中, 又没有 snapshot → 返回空
        checker._ensure_date_expr_scores(
            "2024-01-15", dt.date(2024, 1, 15),
            "all", None, None, None,
            preloaded_scores=preloaded
        )

        assert checker._date_expr_scores["2024-01-15"] == {}


# ═══════════════════════════════════════════════════════
# 3. _check_factors_batch 基本结构
# ═══════════════════════════════════════════════════════

class TestCheckFactorsBatch:
    def _make_snapshot(self, codes, n_rows=200):
        """构造最小化 snapshot: {date_key: {code: (df, name, close, volume)}}"""
        import numpy as np
        snap = {}
        for code in codes:
            df = pd.DataFrame({
                "trade_date": pd.date_range("2023-12-01", periods=n_rows, freq="B"),
                "open": np.random.randn(n_rows) + 10,
                "high": np.random.randn(n_rows) + 11,
                "low": np.random.randn(n_rows) + 9,
                "close": np.random.randn(n_rows) + 10,
                "volume": np.abs(np.random.randn(n_rows)) * 1e7,
                "amount": np.abs(np.random.randn(n_rows)) * 1e8,
                "pct_chg": np.random.randn(n_rows),
                "turnover": np.abs(np.random.randn(n_rows)) * 0.1,
            })
            snap[code] = (df, f"测试{code}", float(df["close"].iloc[-1]), float(df["volume"].iloc[-1]))
        return {"2024-06-15": snap, "2024-06-20": snap, "2024-06-25": snap}

    def test_empty_factor_list_returns_empty(self):
        from core.domain.factors.rolling import _check_factors_batch

        snapshots = self._make_snapshot(["000001", "000002"])
        calls = []

        passed, detail, directions = _check_factors_batch(
            [], {}, snapshots, "all", None,
            18, "full", None, 10, dt.date(2024, 6, 15), "2024-06-15",
            None, None,
            lambda stage, cur, total, **kw: calls.append((stage, cur, total))
        )
        assert passed == []
        assert detail == {}
        assert directions == {}

    def test_progress_reports_start(self):
        """进度回调收到首条 factor 阶段报告"""
        from core.domain.factors.rolling import _check_factors_batch

        snapshots = self._make_snapshot(["000001", "000002"])

        variants = [
            {"variant_key": "test_factor_1", "template_key": "momentum", "params": {"n": 20}},
        ]
        var_by_key = {"test_factor_1": variants[0]}

        calls = []
        # 无 preloaded_expr → 走懒缓存路径 (会因为缺 screen_cache 出错,
        # 但 _check_one 有 try/except → 返回 error detail)
        _check_factors_batch(
            variants, var_by_key, snapshots, "all", None,
            18, "fast", None, 10, dt.date(2024, 6, 15), "2024-06-15",
            None, None,
            lambda stage, cur, total, **kw: calls.append(stage)
        )
        assert "factor" in calls, f"progress stages: {calls}"

    def test_factor_transient_caches_are_released(self, monkeypatch):
        from core.domain.factors.rolling import _check_factors_batch
        from core.domain.factors.validation import OverfittingChecker

        observed = []

        def fake_check(self, *_args, **_kwargs):
            self._screen_cache["screen"] = object()
            self._neutral_cache["neutral"] = object()
            self._resid_cache["resid"] = object()
            self._injected_scores = [1.0]
            observed.append(self)
            return {
                "all_pass": True,
                "passed_count": 2,
                "direction": "long",
                "layers": [],
            }

        monkeypatch.setattr(OverfittingChecker, "check_all_layers", fake_check)
        variants = [{"variant_key": "test_factor", "expr": "$close"}]
        _check_factors_batch(
            variants, {"test_factor": variants[0]}, {}, "all", None,
            18, "full", None, 10, dt.date(2024, 6, 15), "2024-06-15",
            None, {}, lambda *_args, **_kwargs: None,
        )

        assert observed
        assert observed[0]._screen_cache == {}
        assert observed[0]._neutral_cache == {}
        assert observed[0]._resid_cache == {}
        assert observed[0]._injected_scores is None


# ═══════════════════════════════════════════════════════
# 4. _run_anchor_worker v2 路径
# ═══════════════════════════════════════════════════════

class TestRunAnchorWorkerV2:
    def test_v2_path_detected_and_used(self):
        """args_dict 包含 snap_dir + expr_dir → 走 v2 磁盘加载路径"""
        from core.domain.factors.rolling import _run_anchor_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = os.path.join(tmpdir, "snap")
            expr_dir = os.path.join(tmpdir, "expr")
            os.makedirs(snap_dir)
            os.makedirs(expr_dir)

            # 准备最小化数据: 1 个日期的 snapshot + 因子面板
            import pickle
            codes = ["000001", "000002"]
            df = pd.DataFrame({
                "trade_date": pd.date_range("2024-01-01", periods=200, freq="B"),
                "open": np.random.randn(200) + 10,
                "high": np.random.randn(200) + 11,
                "low": np.random.randn(200) + 9,
                "close": np.random.randn(200) + 10,
                "volume": np.abs(np.random.randn(200)) * 1e7,
                "amount": np.abs(np.random.randn(200)) * 1e8,
                "pct_chg": np.random.randn(200),
                "turnover": np.abs(np.random.randn(200)) * 0.1,
            })

            # 写入 3 个日期的 snapshot (train/valid/test)
            for dk in ["2024-03-15", "2024-06-15", "2024-09-15"]:
                snap = {}
                for code in codes:
                    snap[code] = (df.copy(), f"测试{code}", float(df["close"].iloc[-1]),
                                  float(df["volume"].iloc[-1]))
                with open(os.path.join(snap_dir, f"{dk}.pkl"), "wb") as f:
                    pickle.dump(snap, f, protocol=5)

            # 写入因子面板
            scores = {"expr:test": (codes, [0.1, -0.2])}
            for dk in ["2024-03-15", "2024-06-15", "2024-09-15"]:
                with open(os.path.join(expr_dir, f"{dk}.pkl"), "wb") as f:
                    pickle.dump(scores, f, protocol=5)

            args = {
                "factor_variants": [
                    {"variant_key": "test", "template_key": "test",
                     "params": {"n": 20}, "expr": "Ref($close,20)/$close"}
                ],
                "anchor_date": "2024-06-15",
                "all_dates_sorted": ["2024-03-15", "2024-06-15", "2024-09-15"],
                "pool_codes": codes,
                "pool": "a500",
                "eval_weeks": 18,
                "mode": "fast",
                "code_prefixes": None,
                "skip_layers": None,
                "n_shuffles": 10,
                "job_id": None,
                "snap_dir": snap_dir,
                "expr_dir": expr_dir,
            }

            with mock.patch("redis.Redis") as redis_client, \
                 mock.patch("core.data.fetcher.DataFetcher.get_industry_map", return_value={}):
                redis_client.return_value.setex.side_effect = ConnectionError("redis unavailable in isolated test")
                result = _run_anchor_worker(args)

            assert "akey" in result
            assert "passed" in result
            assert "detail" in result
            assert "directions" in result
            assert "label_matured_at" in result["window"]
            assert result["akey"] == "2024-06-15"
            # fast mode 只跑 L5+L1, 即使数据不完美也不应 crash

    def test_legacy_path_still_works(self, monkeypatch):
        """不传 snap_dir/expr_dir → 走旧路径 (会因为缺环境而优雅失败)"""
        import core.domain.factors.rolling as rolling
        from core.domain.factors.rolling import _run_anchor_worker

        def fail_master(*args, **kwargs):
            raise RuntimeError("test master unavailable")

        monkeypatch.setattr(rolling, "_build_master_indicators", fail_master)

        args = {
            "factor_variants": [
                {"variant_key": "test", "template_key": "test",
                 "params": {"n": 20}}
            ],
            "anchor_date": "2024-06-15",
            "all_dates_sorted": ["2024-03-15", "2024-06-15", "2024-09-15"],
            "pool_codes": ["000001"],
            "pool": "a500",
            "eval_weeks": 18,
            "mode": "fast",
            "code_prefixes": None,
            "skip_layers": None,
            "n_shuffles": 10,
            "job_id": None,
            # 不传 snap_dir, expr_dir → 旧路径
        }

        # 固定 master 构建失败，避免单测依赖 Redis/MySQL。
        with mock.patch("redis.Redis") as redis_client:
            redis_client.return_value.setex.side_effect = ConnectionError("redis unavailable in isolated test")
            result = _run_anchor_worker(args)
        assert "akey" in result
        # 旧路径可能因缺数据库而返回 error, 但不应抛未捕获异常


# ═══════════════════════════════════════════════════════
# 5. _precompute_for_anchors 结构验证
# ═══════════════════════════════════════════════════════

class TestPrecomputeForAnchors:
    def test_returns_correct_keys(self):
        """验证返回 dict 包含所有必要字段"""
        # _precompute_for_anchors 需要完整的数据库/Redis 环境，
        # 这里只验证函数签名和导入路径正确
        from core.domain.factors.rolling import _precompute_for_anchors
        import inspect

        sig = inspect.signature(_precompute_for_anchors)
        params = list(sig.parameters.keys())
        expected = ["factor_variants", "anchors", "pool_codes", "pool",
                    "lookback_months", "eval_weeks", "all_dates_sorted",
                    "code_prefixes", "job_id", "progress_cb"]
        assert params == expected, f"签名不匹配: {params}"
