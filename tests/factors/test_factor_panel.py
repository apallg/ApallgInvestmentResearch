import datetime as dt
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from core.domain.factors.panel import ResearchSnapshotStore, build_point_in_time_maps
from core.domain.factors.validation import OverfittingChecker, PackedExpressionScores
from core.data.repos.factor_panel_memmap import FactorPanelMemmapRepo


def _open_memmap_in_spawned_process(manifest_path, eval_date, queue):
    manifest, arrays = FactorPanelMemmapRepo.open(manifest_path)
    try:
        store = ResearchSnapshotStore.from_shared_arrays(
            manifest, arrays, [eval_date], adj_latest=manifest.get("adj_latest"),
            min_turnover=1,
        )
        frame = store[eval_date]["600001"][0]
        queue.put({
            "mode": store.backing_mode,
            "writeable": arrays["values"].flags.writeable,
            "shared": np.shares_memory(frame.to_numpy(copy=False), arrays["values"]),
        })
        store.clear()
    finally:
        FactorPanelMemmapRepo.close(arrays)


def _frame(start="2024-01-01", periods=140, price=10.0):
    dates = pd.bdate_range(start, periods=periods)
    close = price + np.arange(periods) * 0.01
    return pd.DataFrame({
        "trade_date": dates,
        "open": close - 0.02,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": np.full(periods, 1_000_000.0),
        "amount": np.full(periods, 20_000_000.0),
        "turnover": np.full(periods, 1.0),
        "pct_chg": np.full(periods, 0.1),
        "adj_factor": np.full(periods, 1.0),
    })


def test_expression_evaluator_supports_clamps_constant_windows_and_weekdays():
    from core.domain.factors.expression import evaluate

    frame = _frame(periods=10)
    frame.loc[frame.index[-1], "close"] = frame.loc[frame.index[-2], "close"] - 1
    dated = frame.set_index("trade_date")

    assert evaluate("Max($close-Ref($close,1),0)", dated) == pytest.approx(0.0)
    assert evaluate("Mean($close,3+1e-8)", dated) == pytest.approx(
        dated["close"].iloc[-3:].mean()
    )
    expected = dated["close"].iloc[-1] if dated.index[-1].dayofweek + 1 == 5 else 0.0
    assert evaluate("DayOfWeek()==5?$close:0", dated) == pytest.approx(expected)


def test_snapshot_store_reuses_master_memory_and_preserves_date_cutoff():
    raw = _frame()
    first = raw["trade_date"].iloc[120].date()
    second = raw["trade_date"].iloc[130].date()
    progress = []
    store = ResearchSnapshotStore(
        {"600001": raw}, [first, second], min_turnover=1,
        progress_cb=lambda current, total, date: progress.append((current, total, date)),
    )

    snap1 = store[first.isoformat()]
    snap2 = store[second.isoformat()]
    assert list(store) == [first.isoformat(), second.isoformat()]
    assert snap1["600001"][0].index[-1].date() == first
    assert snap2["600001"][0].index[-1].date() == second
    assert np.shares_memory(
        snap1["600001"][0]["close"].to_numpy(copy=False),
        store._frames[0]["close"].to_numpy(copy=False),
    )
    assert progress[-1][:2] == (2, 2)


def test_snapshot_store_is_invariant_to_appended_future_rows():
    raw = _frame(periods=130)
    eval_date = raw["trade_date"].iloc[120].date()
    base = ResearchSnapshotStore({"600001": raw.iloc[:121]}, [eval_date], min_turnover=1)
    extended = ResearchSnapshotStore({"600001": raw}, [eval_date], min_turnover=1)

    pd.testing.assert_frame_equal(
        base[eval_date.isoformat()]["600001"][0],
        extended[eval_date.isoformat()]["600001"][0],
    )


def test_snapshot_store_uses_point_in_time_name_for_st_filter():
    raw = _frame()
    raw["name"] = "测试股份"
    raw.loc[125:, "name"] = "ST测试"
    normal_date = raw["trade_date"].iloc[120].date()
    st_date = raw["trade_date"].iloc[130].date()
    store = ResearchSnapshotStore(
        {"600001": raw}, [normal_date, st_date], min_turnover=1,
    )

    assert store[normal_date.isoformat()]["600001"][1] == "测试股份"
    assert "600001" not in store[st_date.isoformat()]


def test_snapshot_store_uses_adjustment_for_low_price_filter():
    raw = _frame(price=0.5)
    eval_date = raw["trade_date"].iloc[-1].date()
    kept = ResearchSnapshotStore(
        {"600001": raw}, [eval_date], adj_latest={"600001": 3.0}, min_turnover=1,
    )
    dropped = ResearchSnapshotStore(
        {"600001": raw}, [eval_date], adj_latest={"600001": 1.0}, min_turnover=1,
    )

    assert "600001" in kept[eval_date.isoformat()]
    assert "600001" not in dropped[eval_date.isoformat()]


def test_point_in_time_market_cap_does_not_use_future_row():
    rows = {
        "600001": [
            {"trade_date": dt.date(2024, 1, 2), "total_mv": 10},
            {"trade_date": dt.date(2024, 1, 5), "total_mv": 20},
        ]
    }
    result = build_point_in_time_maps(rows, [dt.date(2024, 1, 3), dt.date(2024, 1, 5)])
    assert result["2024-01-03"]["600001"] == 10
    assert result["2024-01-05"]["600001"] == 20


def test_point_in_time_market_cap_rejects_stale_row():
    rows = {"600001": [{"trade_date": dt.date(2024, 1, 2), "total_mv": 10}]}
    result = build_point_in_time_maps(rows, [dt.date(2024, 2, 1)])
    assert result["2024-02-01"] == {}


def test_shared_store_is_materially_smaller_than_copied_date_windows():
    master = {f"600{i:03d}": _frame(periods=180) for i in range(20)}
    dates = [master["600000"]["trade_date"].iloc[index].date() for index in range(120, 180, 10)]
    store = ResearchSnapshotStore(master, dates, min_turnover=1)
    copied_bytes = 0
    for date in dates:
        for frame, _name, _close, _volume in store[date.isoformat()].values():
            copied_bytes += int(frame.copy().memory_usage(index=True, deep=True).sum())
    assert store.memory_bytes < copied_bytes * 0.4


def test_memmap_panel_roundtrip_is_read_only_and_zero_copy():
    raw = _frame()
    raw["name"] = "测试股份"
    eval_date = raw["trade_date"].iloc[125].date()
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            {"600001": raw}, directory, metadata={"adj_latest": {"600001": 1.0}},
        )
        manifest, arrays = FactorPanelMemmapRepo.open(manifest_path)
        try:
            store = ResearchSnapshotStore.from_shared_arrays(
                manifest, arrays, [eval_date], adj_latest=manifest["adj_latest"],
                min_turnover=1,
            )
            frame = store[eval_date.isoformat()]["600001"][0]
            assert store.backing_mode == "memmap"
            assert np.shares_memory(frame.to_numpy(copy=False), arrays["values"])
            assert frame.index[-1].date() == eval_date
            assert arrays["values"].flags.writeable is False
            store.clear()
        finally:
            FactorPanelMemmapRepo.close(arrays)


def test_memmap_worker_opens_panel_inside_worker_process_boundary(monkeypatch):
    import backend.service.factor_rolling as rolling_service
    monkeypatch.setattr("redis.Redis", lambda **kwargs: None)

    raw = _frame(start="2023-01-02", periods=420)
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            {"600001": raw}, directory, metadata={"adj_latest": {"600001": 1.0}},
        )
        observed = {}

        def fake_worker(args):
            observed["mode"] = args["_snapshots"].backing_mode
            observed["dates"] = len(args["_snapshots"])
            observed["writeable"] = args["_snapshots"]._shared_arrays["values"].flags.writeable
            return {"akey": args["anchor_date"], "passed": [], "detail": {}, "directions": {}}

        monkeypatch.setattr(rolling_service, "_run_anchor_worker", fake_worker)
        result = rolling_service.run_memmap_anchor_worker({
            "anchor_date": "2024-06-15",
            "panel_manifest": manifest_path,
        })

    assert result["akey"] == "2024-06-15"
    assert observed["mode"] == "memmap"
    assert observed["writeable"] is False
    assert observed["dates"] > 0


def test_memmap_worker_reuses_expression_scores_across_anchors(monkeypatch):
    import backend.service.factor_rolling as rolling_service
    monkeypatch.setattr("redis.Redis", lambda **kwargs: None)

    raw = _frame(start="2023-01-02", periods=420)
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            {"600001": raw}, directory, metadata={"adj_latest": {"600001": 1.0}},
        )
        expr_cache_dir = os.path.join(directory, "expr_scores")
        observed = []

        def fake_ensure(self, date_key, *_args, **_kwargs):
            self._date_expr_scores[date_key] = {"sig": (["600001"], [0.25])}

        def fake_worker(args):
            observed.append({
                "scores": args["_preloaded_expr"],
                "stats": args["expr_cache_stats"],
            })
            return {"akey": args["anchor_date"], "passed": [], "detail": {}, "directions": {}}

        monkeypatch.setattr(OverfittingChecker, "_ensure_date_expr_scores", fake_ensure)
        monkeypatch.setattr(rolling_service, "_run_anchor_worker", fake_worker)
        args = {
            "anchor_date": "2024-06-15",
            "panel_manifest": manifest_path,
            "expr_cache_dir": expr_cache_dir,
            "factor_variants": [{"variant_key": "x", "expr": "$close"}],
            "batch_dedup": True,
            "pool": "a500",
        }
        rolling_service.run_memmap_anchor_worker(args)
        rolling_service.run_memmap_anchor_worker(args)

    first_codes, first_scores = next(iter(observed[0]["scores"].values()))["sig"]
    second_codes, second_scores = next(iter(observed[1]["scores"].values()))["sig"]
    assert first_codes.tolist() == second_codes.tolist() == ["600001"]
    np.testing.assert_array_equal(first_scores, second_scores)
    assert observed[0]["stats"]["misses"] > 0
    assert observed[1]["stats"]["misses"] == 0
    assert observed[1]["stats"]["hits"] == observed[0]["stats"]["misses"]


def test_packed_expression_scores_preserves_order_values_and_overlay():
    raw = {
        "sig_a": (["600002", "600001"], [0.25, -0.5]),
        "sig_b": (["600001"], [1.75]),
    }
    packed = PackedExpressionScores.from_mapping(raw)

    codes_a, scores_a = packed["sig_a"]
    codes_b, scores_b = packed["sig_b"]
    assert list(packed) == ["sig_a", "sig_b"]
    assert codes_a.tolist() == ["600002", "600001"]
    assert codes_b.tolist() == ["600001"]
    np.testing.assert_array_equal(scores_a, np.array([0.25, -0.5]))
    np.testing.assert_array_equal(scores_b, np.array([1.75]))
    assert scores_a.dtype == np.float64
    assert packed.memory_bytes < 256

    packed["missing"] = (["600003"], [2.0])
    assert packed["missing"] == (["600003"], [2.0])


def test_packed_expression_scores_reach_rank_ic(monkeypatch):
    from core.data import Database
    from core.domain.factors.forward_returns import get_forward_cache

    eval_date = dt.date(2024, 1, 2)
    codes = [f"60{index:04d}" for index in range(25)]
    scores = np.arange(len(codes), dtype=float)
    variant = {"variant_key": "x", "expr": "$close"}
    checker = OverfittingChecker()
    checker.set_expr_variants([variant])
    _, signature, _, _, _ = checker._resolve_eval(variant, quiet=True)
    checker._date_expr_scores[eval_date.isoformat()] = PackedExpressionScores.from_mapping({
        signature: (codes, scores),
    })
    checker._industry_map = {}
    forward_cache = get_forward_cache()
    forward_cache.clear_memory()
    for code, value in zip(codes, scores):
        forward_cache._mem[(code, eval_date.isoformat(), 10)] = float(value)
    monkeypatch.setattr(Database, "is_available", lambda: False)

    ic, daily_ics = checker._calc_rank_ic(
        variant,
        [eval_date],
        {"pool": "a500", "snapshots": {}, "label_horizon": 10},
        return_series=True,
    )

    assert ic == pytest.approx(1.0)
    assert daily_ics == pytest.approx([1.0])


def test_expression_date_worker_writes_packed_scores():
    from backend.service.factor_rolling import run_expression_date_worker
    from backend.service.factor_panel import FactorPanelService
    from core.shared.persistence import load_pickle

    master = {
        f"60{index:04d}": _frame(periods=180, price=10.0 + index * 0.01)
        for index in range(25)
    }
    eval_date = next(iter(master.values()))["trade_date"].iloc[150].date()
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            master, directory,
            metadata={"adj_latest": {code: 1.0 for code in master}},
        )
        service = FactorPanelService()
        store, arrays, _manifest = service.open_shared_memmap(manifest_path, [eval_date])
        try:
            checker = OverfittingChecker()
            checker.set_expr_variants([{"variant_key": "x", "expr": "$close"}])
            checker._ensure_date_expr_scores(
                eval_date.isoformat(), eval_date, "a500", None, store, None,
            )
            raw = checker._date_expr_scores[eval_date.isoformat()]
        finally:
            store.clear()
            service.close_shared_memmap(arrays)
        expr_cache_dir = os.path.join(directory, "expr_scores")
        result = run_expression_date_worker({
            "eval_date": eval_date.isoformat(),
            "panel_manifest": manifest_path,
            "expr_cache_dir": expr_cache_dir,
            "factor_variants": [{"variant_key": "x", "expr": "$close"}],
            "pool": "a500",
        })
        cached = load_pickle(os.path.join(expr_cache_dir, f"{eval_date.isoformat()}.pkl"))

    assert result["cache_hit"] is False
    assert result["factor_count"] == result["expected_factor_count"] == 1
    assert result["missing_factor_keys"] == []
    assert isinstance(cached, PackedExpressionScores)
    signature = next(iter(cached))
    codes, scores = cached[signature]
    raw_codes, raw_scores = raw[signature]
    assert len(codes) == len(scores) == 25
    assert codes.tolist() == raw_codes
    np.testing.assert_array_equal(scores, np.asarray(raw_scores, dtype=np.float64))
    assert scores.dtype == np.float64


def test_expression_date_worker_fills_missing_scores_in_overlay():
    from backend.service.factor_rolling import run_expression_date_worker
    from core.shared.persistence import load_pickle

    master = {
        f"60{index:04d}": _frame(periods=180, price=10.0 + index * 0.01)
        for index in range(25)
    }
    eval_date = next(iter(master.values()))["trade_date"].iloc[150].date()
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            master, directory,
            metadata={"adj_latest": {code: 1.0 for code in master}},
        )
        expr_cache_dir = os.path.join(directory, "expr_scores")
        overlay_dir = os.path.join(directory, "overlay")
        first = run_expression_date_worker({
            "eval_date": eval_date.isoformat(),
            "panel_manifest": manifest_path,
            "expr_cache_dir": expr_cache_dir,
            "factor_variants": [{"variant_key": "close", "expr": "$close"}],
            "pool": "a500",
        })
        repaired = run_expression_date_worker({
            "eval_date": eval_date.isoformat(),
            "panel_manifest": manifest_path,
            "expr_cache_dir": expr_cache_dir,
            "expr_overlay_dir": overlay_dir,
            "factor_variants": [
                {"variant_key": "close", "expr": "$close"},
                {"variant_key": "clamp", "expr": "Max($close-Ref($close,1),0)"},
            ],
            "pool": "a500",
        })
        base = load_pickle(os.path.join(expr_cache_dir, f"{eval_date.isoformat()}.pkl"))
        overlay = load_pickle(os.path.join(overlay_dir, f"{eval_date.isoformat()}.pkl"))
        repaired_again = run_expression_date_worker({
            "eval_date": eval_date.isoformat(),
            "panel_manifest": manifest_path,
            "expr_cache_dir": expr_cache_dir,
            "expr_overlay_dir": overlay_dir,
            "factor_variants": [
                {"variant_key": "close", "expr": "$close"},
                {"variant_key": "clamp", "expr": "Max($close-Ref($close,1),0)"},
                {"variant_key": "mean", "expr": "Mean($close,5)"},
            ],
            "pool": "a500",
        })
        overlay_again = load_pickle(os.path.join(overlay_dir, f"{eval_date.isoformat()}.pkl"))

    assert first["factor_count"] == 1
    assert repaired["cache_hit"] is True
    assert repaired["factor_count"] == repaired["expected_factor_count"] == 2
    assert repaired["missing_factor_keys"] == []
    assert len(base) == 1
    assert len(overlay) == 1
    assert repaired_again["factor_count"] == repaired_again["expected_factor_count"] == 3
    assert repaired_again["missing_factor_keys"] == []
    assert len(overlay_again) == 2
    for signature in overlay:
        assert signature in overlay_again
        np.testing.assert_array_equal(overlay[signature][0], overlay_again[signature][0])
        np.testing.assert_array_equal(overlay[signature][1], overlay_again[signature][1])


def test_expression_precompute_pool_writes_cache(monkeypatch):
    from backend.service.factor_rolling import FactorRollingService
    from core.shared import config
    from core.shared.persistence import load_pickle

    master = {
        f"60{index:04d}": _frame(periods=180, price=10.0 + index * 0.01)
        for index in range(25)
    }
    eval_date = next(iter(master.values()))["trade_date"].iloc[150].date()
    monkeypatch.setattr(config, "FACTOR_EXPR_PRECOMPUTE_WORKERS", 1)
    progress = []
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            master, directory,
            metadata={"adj_latest": {code: 1.0 for code in master}},
        )
        expr_cache_dir = os.path.join(directory, "expr_scores")
        FactorRollingService._precompute_expression_scores(
            [{"variant_key": "x", "expr": "$close"}],
            [eval_date], manifest_path, expr_cache_dir,
            "a500", None, progress.append, lambda: False,
        )
        cached = load_pickle(os.path.join(expr_cache_dir, f"{eval_date.isoformat()}.pkl"))

    assert isinstance(cached, PackedExpressionScores)
    assert progress[-1]["stage"] == "expression_panel"
    assert progress[-1]["current"] == progress[-1]["total"] == 1
    assert progress[-1]["progress_fraction"] == pytest.approx(0.45)


def test_expression_precompute_rejects_missing_latest_factor(monkeypatch):
    from backend.service.factor_rolling import FactorRollingService
    from core.shared import config

    master = {
        f"60{index:04d}": _frame(periods=180, price=10.0 + index * 0.01)
        for index in range(25)
    }
    eval_date = next(iter(master.values()))["trade_date"].iloc[150].date()
    monkeypatch.setattr(config, "FACTOR_EXPR_PRECOMPUTE_WORKERS", 1)
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            master, directory,
            metadata={"adj_latest": {code: 1.0 for code in master}},
        )
        with pytest.raises(RuntimeError, match="表达式预计算不完整"):
            FactorRollingService._precompute_expression_scores(
                [{"variant_key": "bad", "expr": "$unknown_field"}],
                [eval_date], manifest_path, os.path.join(directory, "expr_scores"),
                "a500", None, None, lambda: False,
            )


def test_expression_precompute_rejects_missing_factor_on_any_date(monkeypatch):
    import multiprocessing
    from backend.service.factor_rolling import FactorRollingService

    class FakeIterator:
        def __init__(self, results):
            self._results = iter(results)

        def next(self, timeout):
            return next(self._results)

    class FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, worker, args):
            return FakeIterator([
                {"eval_date": args[0]["eval_date"], "cache_hit": False,
                 "packed_bytes": 0, "missing_factor_keys": ["bad"]},
                {"eval_date": args[1]["eval_date"], "cache_hit": False,
                 "packed_bytes": 0, "missing_factor_keys": []},
            ])

    monkeypatch.setattr(multiprocessing, "Pool", FakePool)

    with pytest.raises(RuntimeError, match="表达式预计算不完整"):
        FactorRollingService._precompute_expression_scores(
            [{"variant_key": "bad", "expr": "$unknown_field"}],
            [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "manifest", "cache", "a500", None, None, lambda: False,
        )


def test_memmap_panel_opens_in_two_spawned_processes():
    raw = _frame(start="2023-01-02", periods=320)
    eval_date = raw["trade_date"].iloc[250].date().isoformat()
    with tempfile.TemporaryDirectory() as directory:
        manifest_path, _written = FactorPanelMemmapRepo.write(
            {"600001": raw}, directory, metadata={"adj_latest": {"600001": 1.0}},
        )
        context = mp.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(
                target=_open_memmap_in_spawned_process,
                args=(manifest_path, eval_date, queue),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        results = [queue.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0

    assert results == [
        {"mode": "memmap", "writeable": False, "shared": True},
        {"mode": "memmap", "writeable": False, "shared": True},
    ]


def test_factor_panel_service_keeps_l2_data_access_out_of_domain(monkeypatch):
    import backend.service.factor_panel as module

    raw = _frame()
    dates = [raw["trade_date"].iloc[120].date(), raw["trade_date"].iloc[130].date()]

    class Repo:
        def load_universe(self, pool):
            return ["600001"]

        def load_daily_master(self, codes, start, end):
            return {"600001": raw}

        def load_latest_adj_factors(self, codes):
            return {"600001": 1.0}

        def load_market_cap_history(self, codes, start, end):
            return {"600001": [{"trade_date": dates[0], "total_mv": 100}]}

    store, market_caps, extra = module.FactorPanelService(Repo()).build_mining_context(
        "csi300", dates, risk_neutralize=False,
    )

    assert store.code_count == 1
    assert market_caps[dates[0].isoformat()]["600001"] == 100
    assert extra == {}


def test_portfolio_sort_service_injects_shared_panel_and_explicit_dates(monkeypatch):
    import core.domain.factors.portfolio_sort as sort_module
    import core.shared.persistence as persistence
    import core.shared.trading_time as trading_time
    from backend.service.factor_validation import FactorValidationService

    panel = {"2024-03-01": {}}
    captured = {}

    def fake_build(self, pool, dates, include_tech=False, progress_cb=None):
        captured["panel_dates"] = dates
        captured["include_tech"] = include_tech
        return panel, ["600001"]

    def fake_sort(factors, **kwargs):
        captured.update(kwargs)
        return ["ok"]

    monkeypatch.setattr(persistence, "today_date", lambda: dt.date(2024, 3, 1))
    monkeypatch.setattr(trading_time, "is_trading_day", lambda date: True)
    monkeypatch.setattr(
        "backend.service.factor_validation.FactorPanelService.build_snapshot_store",
        fake_build,
    )
    monkeypatch.setattr(sort_module, "run_batch_sort", fake_sort)
    monkeypatch.setattr(
        "core.domain.factors.registry.get_pool",
        lambda: type(
            "Pool", (),
            {"get_all_factors": lambda self: {"momentum": {"label": "动量"}}},
        )(),
    )

    result = FactorValidationService().portfolio_sort(
        ["momentum"], pool="csi300", eval_weeks=1,
    )

    assert result == ["ok"]
    assert captured["snapshots"] is panel
    assert captured["start"] == dt.date(2024, 2, 23)
    assert captured["end"] == dt.date(2024, 3, 1)
    assert captured["include_tech"] is True
