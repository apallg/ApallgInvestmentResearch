import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from core.domain.factors.candidate import (
    build_candidate,
    canonical_expression,
    max_abs_correlation,
    validate_expression,
)
from core.domain.factors.fitness import evaluate_factor_portfolio, summarize_ic_series
from core.domain.factors.promotion import evaluate_promotion_gate
from core.domain.factors.pool_overlay import (
    build_factor_overlay,
    normalize_overlay_scores,
    overlay_fingerprint,
)
from core.domain.factors.selection import adjust_selection_bias
from core.domain.factors.walkforward import (
    aggregate_fold_metrics,
    build_l1_validation_dates,
    build_walkforward_folds,
    purge_dates_before_boundary,
    sample_trading_dates,
)
from core.shared.infrastructure.async_task import AsyncTaskManager, TaskInfo


def test_candidate_rejects_future_window_and_normalizes_commutative_expression():
    assert validate_expression("Ref($close, -1)")["valid"] is False
    assert canonical_expression("($close + $open)") == canonical_expression("($open+$close)")
    candidate = build_candidate("$close / Ref($close, 20) - 1", seed=42)
    assert candidate["required_fields"] == ["close"]
    assert candidate["complexity"] > 0


def test_ic_summary_and_promotion_gate():
    metrics = summarize_ic_series([0.01, 0.03, -0.01, 0.02])
    assert metrics["count"] == 4
    assert metrics["positive_ratio"] == 0.75
    result = evaluate_promotion_gate({
        "oos_ic": 0.03, "oos_icir": 0.4, "positive_ratio": 0.7,
        "passed_folds": 3, "folds": 5, "decay": 0.2,
        "max_correlation": 0.2, "coverage": 1.0,
    })
    assert result["passed"] is True


def test_walkforward_dates_are_ordered_and_purged():
    train = [dt.date(2020, 1, 1) + dt.timedelta(days=30 * i) for i in range(48)]
    valid = [dt.date(2024, 1, 1) + dt.timedelta(days=30 * i) for i in range(12)]
    folds = build_walkforward_folds(train, valid, n_folds=5, purge_gap=2)
    assert len(folds) == 5
    previous = None
    for fold in folds:
        assert fold["train_end"] < fold["valid_start"]
        assert fold["purge_gap"] == 2
        assert previous is None or previous < fold["valid_start"]
        previous = fold["valid_start"]


def test_split_purge_uses_full_trading_calendar_not_sparse_eval_dates():
    calendar = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    eval_dates = calendar[::5]
    kept = purge_dates_before_boundary(
        eval_dates, calendar[30], calendar, purge_gap=6,
    )
    assert kept[-1] == calendar[20]
    assert all(calendar.index(day) + 6 < 30 for day in kept)


def test_l1_dates_are_derived_from_shared_trading_calendar():
    calendar = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(500)]
    anchor = dt.date(2025, 1, 1)
    plan = build_l1_validation_dates(anchor, calendar, purge_gap=21, step=5)
    prefetched = set(sample_trading_dates(calendar, calendar[0], calendar[-1], 10))
    prefetched.update(date for dates in plan.values() for date in dates)

    assert plan["train"]
    assert plan["valid"]
    assert plan["test"]
    assert all(date in prefetched for dates in plan.values() for date in dates)
    assert max(plan["train"]) < anchor - dt.timedelta(days=90)
    assert max(plan["valid"]) < anchor
    assert min(plan["test"]) >= anchor


def test_expression_correlation_uses_aligned_codes():
    candidate = {f"c{i}": float(i) for i in range(30)}
    references = {
        "same": {f"c{i}": float(i * 2) for i in range(30)},
        "short": {f"c{i}": float(i) for i in range(5)},
    }
    corr, key, compared = max_abs_correlation(candidate, references, min_count=20)

    assert corr == pytest.approx(1.0)
    assert key == "same"
    assert compared == 1


def test_fold_aggregate_uses_daily_ic_and_date_coverage():
    folds = [{
        "oos_ic": 0.03,
        "passed": True,
        "valid_date_count": 10,
        "oos_ic_series": [{"date": f"d{i}", "ic": -0.01} for i in range(9)]
        + [{"date": "d9", "ic": 0.39}],
    }]
    metrics = aggregate_fold_metrics(folds)
    assert metrics["positive_ratio"] == 0.1
    assert metrics["coverage"] == 1.0
    assert metrics["oos_ic"] == pytest.approx(0.03)


def test_incomplete_three_fold_run_cannot_pass_five_fold_gate():
    result = evaluate_promotion_gate({
        "oos_ic": 0.03, "selection_adjusted_oos_ic": 0.03,
        "oos_icir": 0.4, "positive_ratio": 0.7,
        "passed_folds": 3, "folds": 3, "decay": 0.2,
        "max_correlation": 0.2, "coverage": 1.0,
    })
    assert result["passed"] is False


def test_factor_sort_proxy_returns_metrics():
    scores = {f"c{i}": float(i) for i in range(20)}
    returns = {f"c{i}": 0.01 if i >= 16 else -0.01 for i in range(20)}
    result = evaluate_factor_portfolio({"2025-01-01": scores}, {"2025-01-01": returns})
    assert result["days"] == 1
    assert result["net_return"] > 0


def test_factor_overlay_is_read_only_and_deterministic():
    scores = normalize_overlay_scores({"600002": 2.0, "600001": 1.0, "600003": None, "600004": float("nan")})
    assert scores == {"600001": 0.0, "600002": 1.0}
    overlay = build_factor_overlay("$close", {"600001": 1.0, "600002": 2.0}, weight=0.8)
    assert overlay["weight"] == 0.5
    assert overlay_fingerprint(overlay) == overlay_fingerprint(dict(overlay))
    assert "pool" not in overlay


def test_factor_overlay_uses_average_rank_for_ties():
    scores = normalize_overlay_scores({"600001": 1.0, "600002": 1.0, "600003": 1.0})
    assert scores == {"600001": 0.5, "600002": 0.5, "600003": 0.5}


def test_strategy_gate_requires_non_worse_production_ab():
    from backend.service.factor_mining import FactorMiningService

    passed, reasons = FactorMiningService._strategy_gate({
        "total_return": 0.01, "sharpe": 0.02,
        "max_drawdown": 0.01, "turnover_ratio": 1.0,
    })
    assert passed is True
    assert reasons == []

    passed, reasons = FactorMiningService._strategy_gate({
        "total_return": -0.01, "sharpe": -0.1,
        "max_drawdown": -0.03, "turnover_ratio": 1.3,
    })
    assert passed is False
    assert len(reasons) == 4


def test_backtest_turnover_uses_traded_notional_not_trade_count():
    from backend.service.factor_mining import FactorMiningService

    metrics = FactorMiningService._backtest_metrics({
        "summary": {"n_periods": 10, "n_trades": 100},
        "trade_diagnostics": {"traded_notional": 1.5, "transaction_cost": 0.01},
    })
    assert metrics["turnover"] == pytest.approx(0.15)


def test_production_backtest_overlay_uses_eval_date_snapshot():
    from core.domain.backtest.screening import ScreeningBacktestRunner

    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    frame = pd.DataFrame({
        "open": [10.0, 11.0, 12.0], "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5], "close": [10.0, 11.0, 12.0],
        "volume": [100.0, 110.0, 120.0],
    }, index=dates)
    screener = SimpleNamespace(
        precompute_snapshot=lambda **kwargs: {"600001": (frame, "", frame["close"], frame["volume"])},
        fetcher=SimpleNamespace(get_industry_map=lambda: {}),
    )
    runner = object.__new__(ScreeningBacktestRunner)
    overlay = runner._prepare_factor_overlay(
        screener, dt.date(2025, 1, 3), "csi300",
        {"expr": "$close", "weight": 0.2, "risk_neutralize": False},
    )
    assert overlay["expr"] == "$close"
    assert overlay["scores"] == {"600001": 0.5}
    assert overlay["coverage"] == 1.0


def test_promotion_gate_blocks_strategy_failure():
    result = evaluate_promotion_gate({
        "oos_ic": 0.03, "oos_icir": 0.4, "positive_ratio": 0.7,
        "passed_folds": 3, "folds": 5, "decay": 0.2,
        "max_correlation": 0.2, "coverage": 1.0,
        "strategy_passed": False,
        "strategy_reasons": ["最终 OOS 净收益低于基线"],
    })
    assert result["passed"] is False
    assert "最终 OOS 净收益低于基线" in result["reasons"]


def test_selection_bias_haircut_increases_with_candidate_count():
    few = adjust_selection_bias(0.03, candidate_count=2, fold_count=5)
    many = adjust_selection_bias(0.03, candidate_count=20, fold_count=5)
    assert many["selection_haircut"] > few["selection_haircut"]
    assert many["selection_adjusted_oos_ic"] < few["selection_adjusted_oos_ic"]


def test_promotion_gate_requires_independent_final_oos():
    result = evaluate_promotion_gate({
        "oos_ic": 0.03, "selection_adjusted_oos_ic": 0.03,
        "oos_icir": 0.4, "positive_ratio": 0.7,
        "passed_folds": 3, "folds": 5, "decay": 0.2,
        "max_correlation": 0.2, "coverage": 1.0,
        "oos_independent": False,
    })
    assert result["passed"] is False
    assert "最终 OOS 与验证区间重叠，不允许晋级" in result["reasons"]


def test_new_runs_require_independent_final_oos_but_legacy_can_fallback():
    from backend.service.factor_mining import FactorMiningService

    config = {
        "train_start": "2020-01-01", "train_end": "2023-12-31",
        "valid_start": "2024-01-01", "valid_end": "2024-12-31",
    }
    with pytest.raises(ValueError, match="最终 OOS"):
        FactorMiningService._normalize_config(config)
    legacy = FactorMiningService._normalize_config({**config, "compatibility_mode": True})
    assert legacy["final_oos_start"] == config["valid_start"]
    assert legacy["compatibility_mode"] is True


def test_async_task_cancel_is_cooperative():
    manager = AsyncTaskManager()
    manager._tasks["test-cancel"] = TaskInfo(job_id="test-cancel", status="running", updated_at=1.0)
    assert manager.request_cancel("test-cancel") is True
    assert manager.is_cancel_requested("test-cancel") is True
    assert manager.get_dict("test-cancel")["status"] == "cancel_requested"


def test_persisted_running_task_is_reported_as_interrupted(monkeypatch):
    manager = AsyncTaskManager()
    persisted = TaskInfo(job_id="orphan", status="running", updated_at=1.0)
    monkeypatch.setattr(manager, "_load_from_db", lambda _job_id: persisted)

    result = manager.get_dict("orphan")

    assert result["status"] == "interrupted"
    assert result["progress"]["stage"] == "interrupted"


def test_persisted_running_mining_task_is_resubmitted_when_no_local_future(monkeypatch):
    from backend.service.factor_mining import FactorMiningService
    import backend.service.factor_mining as module

    submitted = []
    monkeypatch.setattr(module.task_manager, "get_dict", lambda _run_id: {
        "run_id": "resume-run", "status": "stalled", "progress": {},
    })
    monkeypatch.setattr(module.task_manager, "is_active", lambda _run_id: False)
    monkeypatch.setattr(
        module.task_manager, "submit_with_id",
        lambda run_id, fn, description, pool="light": submitted.append(run_id),
    )
    service = FactorMiningService()
    monkeypatch.setattr(service, "_repo", lambda method, *_args, **_kwargs: {
        "run_id": "resume-run", "status": "running", "config": {
            "train_start": "2020-01-01", "train_end": "2023-12-31",
            "valid_start": "2024-01-01", "valid_end": "2024-12-31",
        },
    } if method == "get_run" else None)

    service.get_run("resume-run")

    assert submitted == ["resume-run"]


def test_backtest_model_selection_excludes_models_after_cutoff(tmp_path, monkeypatch):
    import core.adapters.qlib.model_selection as selection

    for run_id, valid_end, ic in (("safe", "2024-12-31", 0.02), ("future", "2025-06-30", 0.08)):
        run_dir = tmp_path / "exp" / run_id
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "params").mkdir()
        (run_dir / "metrics").mkdir()
        (run_dir / "artifacts" / "params.pkl").write_bytes(b"model")
        (run_dir / "artifacts" / "factor_keys.json").write_text(
            '{"custom_factor_keys":["f1"],"train_end":"2024-06-30",'
            f'"valid_end":"{valid_end}","label_mode":"tradable_open_to_open"}}',
            encoding="utf-8")
        (run_dir / "params" / "dataset.kwargs.handler.kwargs.instruments").write_text(
            "'csi300'", encoding="utf-8")
        (run_dir / "metrics" / "IC").write_text(f"0 {ic} 0", encoding="utf-8")
    monkeypatch.setattr(selection.config, "QLIB_MLFLOW_URI", f"file:{tmp_path}")
    monkeypatch.setattr(selection, "is_disabled", lambda _exp, _run: False)

    result = selection.select_model_for_factor_set(["f1"], "2024-12-31", "csi300")

    assert result["run_id"] == "safe"
    assert result["valid_end"] == "2024-12-31"


def test_model_selection_reads_label_horizon_from_run_params(tmp_path, monkeypatch):
    import core.adapters.qlib.model_selection as selection

    run_dir = tmp_path / "exp" / "chosen"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "params").mkdir()
    (run_dir / "artifacts" / "params.pkl").write_bytes(b"model")
    (run_dir / "artifacts" / "factor_keys.json").write_text(
        '{"custom_factor_keys":["f1"],"train_end":"2023-06-30",'
        '"valid_end":"2023-12-31","label_mode":"tradable_open_to_open"}', encoding="utf-8")
    (run_dir / "params" / "label_horizon").write_text("10", encoding="utf-8")
    monkeypatch.setattr(selection.config, "QLIB_MLFLOW_URI", f"file:{tmp_path}")
    monkeypatch.setattr(selection, "is_disabled", lambda _exp, _run: False)

    selected = selection.get_model_for_factor_set("exp", "chosen", ["f1"], "2024-12-31")

    assert selected["label_horizon"] == 10


def test_backtest_context_pins_model_before_factor_validation(monkeypatch):
    from backend.service.factor_mining import FactorMiningService
    import core.adapters.qlib.model_selection as selection
    import core.domain.factors.registry as registry

    class Pool:
        def get_curated_set(self, _name):
            return {"name": "curated_1", "flat_list": ["f1"]}

        def get_curated_factors(self, _name):
            return ["f1"]

    captured = {}

    def select_model(factors, cutoff, pool=None):
        captured.update({"factors": factors, "cutoff": cutoff, "pool": pool})
        return {"model": ("exp", "run"), "factor_overlap": 1.0}

    monkeypatch.setattr(registry, "get_pool", lambda: Pool())
    monkeypatch.setattr(selection, "select_model_for_factor_set", select_model)

    FactorMiningService()._resolve_backtest_context({
        "baseline_set": "current_curated", "pool": "csi300",
        "train_end": "2023-12-31", "valid_end": "2024-12-31",
    })

    assert captured["cutoff"] == "2023-12-31"


def test_manual_backtest_model_selection_is_pinned(tmp_path, monkeypatch):
    import core.adapters.qlib.model_selection as selection

    run_dir = tmp_path / "exp" / "chosen"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "params").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "artifacts" / "params.pkl").write_bytes(b"model")
    (run_dir / "artifacts" / "factor_keys.json").write_text(
        '{"custom_factor_keys":["f1"],"train_end":"2023-06-30",'
        '"valid_end":"2023-12-31","test_end":"2024-06-30",'
        '"label_mode":"tradable_open_to_open"}', encoding="utf-8")
    (run_dir / "params" / "dataset.kwargs.handler.kwargs.instruments").write_text(
        "'csi300'", encoding="utf-8")
    monkeypatch.setattr(selection.config, "QLIB_MLFLOW_URI", f"file:{tmp_path}")
    monkeypatch.setattr(selection, "is_disabled", lambda _exp, _run: False)

    selected = selection.get_model_for_factor_set(
        "exp", "chosen", ["f1"], "2024-12-31", "csi300")

    assert selected["model"] == ("exp", "chosen")
    assert selected["observed_end"] == "2024-06-30"
    assert selected["compatible"] is True


def test_model_compatibility_requires_exact_ordered_factors_and_content_hash(tmp_path, monkeypatch):
    import core.adapters.qlib.model_selection as selection

    run_dir = tmp_path / "exp" / "chosen"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (run_dir / "params").mkdir()
    model = artifacts / "params.pkl"
    model.write_bytes(b"model-v1")
    (artifacts / "factor_keys.json").write_text(
        '{"custom_factor_keys":["f1","f2"],"train_end":"2023-06-30",'
        '"valid_end":"2023-12-31","label_mode":"tradable_open_to_open"}', encoding="utf-8")
    (run_dir / "params" / "dataset.kwargs.handler.kwargs.instruments").write_text("'csi300'", encoding="utf-8")
    monkeypatch.setattr(selection.config, "QLIB_MLFLOW_URI", f"file:{tmp_path}")
    monkeypatch.setattr(selection, "is_disabled", lambda _exp, _run: False)

    selected = selection.get_model_for_factor_set("exp", "chosen", ["f1", "f2"], "2024-12-31", "csi300")
    first = selected["artifact_fingerprint"]
    assert selected["factor_keys"] == ["f1", "f2"]
    with pytest.raises(ValueError, match="顺序"):
        selection.get_model_for_factor_set("exp", "chosen", ["f2", "f1"], "2024-12-31", "csi300")

    model.write_bytes(b"model-v2")
    second = selection.get_model_for_factor_set("exp", "chosen", ["f1", "f2"], "2024-12-31", "csi300")
    assert second["artifact_fingerprint"] != first


def test_service_uses_explicit_model_ids(monkeypatch):
    from backend.service.factor_mining import FactorMiningService
    import core.adapters.qlib.model_selection as selection
    import core.domain.factors.registry as registry

    class Pool:
        def get_curated_factors(self, _name):
            return ["f1"]

    captured = {}

    def get_model(experiment_id, run_id, factors, cutoff, pool=None):
        captured.update({
            "experiment_id": experiment_id, "run_id": run_id,
            "factors": factors, "cutoff": cutoff, "pool": pool,
        })
        return {"model": (experiment_id, run_id), "factor_overlap": 1.0}

    monkeypatch.setattr(registry, "get_pool", lambda: Pool())
    monkeypatch.setattr(selection, "get_model_for_factor_set", get_model)
    monkeypatch.setattr(
        selection, "select_model_for_factor_set",
        lambda *_args, **_kwargs: pytest.fail("手工模式不应调用自动匹配"),
    )

    context = FactorMiningService()._resolve_backtest_context({
        "baseline_set": "trend_momentum", "pool": "csi300",
        "train_end": "2024-12-31",
        "baseline_model_experiment_id": "exp", "baseline_model_run_id": "run",
    })

    assert context["model_selection_mode"] == "manual"
    assert context["model_info"]["model"] == ("exp", "run")
    assert captured["cutoff"] == "2024-12-31"


def test_manual_backtest_model_rejects_future_test_period(tmp_path, monkeypatch):
    import core.adapters.qlib.model_selection as selection

    run_dir = tmp_path / "exp" / "future-test"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "params").mkdir()
    (run_dir / "artifacts" / "params.pkl").write_bytes(b"model")
    (run_dir / "artifacts" / "factor_keys.json").write_text(
        '{"custom_factor_keys":["f1"],"valid_end":"2024-06-30",'
        '"test_end":"2025-06-30","label_mode":"tradable_open_to_open"}',
        encoding="utf-8")
    (run_dir / "params" / "dataset.kwargs.handler.kwargs.instruments").write_text(
        "'csi300'", encoding="utf-8")
    monkeypatch.setattr(selection.config, "QLIB_MLFLOW_URI", f"file:{tmp_path}")
    monkeypatch.setattr(selection, "is_disabled", lambda _exp, _run: False)

    with pytest.raises(ValueError, match="模型数据截止"):
        selection.get_model_for_factor_set(
            "exp", "future-test", ["f1"], "2024-12-31", "csi300")


def test_manual_model_ids_must_be_provided_together():
    from backend.service.factor_mining import FactorMiningService

    with pytest.raises(ValueError, match="同时提供"):
        FactorMiningService._normalize_config({
            "train_start": "2020-01-01", "train_end": "2023-12-31",
            "valid_start": "2024-01-01", "valid_end": "2024-12-31",
            "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
            "baseline_model_experiment_id": "exp",
        })


def test_oos_lock_key_is_expression_scoped_across_runs_and_sources():
    from backend.service.factor_mining import FactorMiningService

    config = {
        "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
    }
    first = FactorMiningService._oos_lock_key(
        {"candidate_key": "gp_a", "expr": "$close + $open"}, config)
    second = FactorMiningService._oos_lock_key(
        {"candidate_key": "llm_b", "expr": "$open+$close"}, config)
    assert first == second
