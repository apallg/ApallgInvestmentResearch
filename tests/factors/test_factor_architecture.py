import datetime as dt
from unittest.mock import patch

import pytest

from backend.service.factor_architecture import FactorArchitectureService
from core.adapters.qlib.factor_set_evaluator import (
    FactorSetEvaluator,
    _PreparedFactorDataset,
    _daily_spearman_matrix,
    _make_rounds,
)
from core.data.repos.factor_architecture import factor_architecture_repo
from core.shared.infrastructure.async_task import task_manager
from core.domain.factors.evidence import orient_ic, summarize_factor_evidence
from core.domain.factors.preprocessing import preprocess_candidates
from core.domain.factors.role_selection import (
    build_balanced_extension,
    build_balanced_set,
    select_core_candidates,
)
from core.domain.factors.set_evaluation import (
    analyze_core_ab,
    analyze_core_oos,
    analyze_satellite_ab,
    paired_fold_delta,
)
from core.domain.factors.validation import OverfittingChecker
from core.domain.factors.rolling import _batch_deduplicate_passed
from core.domain.factors.walkforward import evenly_sample_dates


pytestmark = pytest.mark.unit


def test_preprocessing_rejects_unresolved_and_duplicate_expressions():
    result = preprocess_candidates([
        {"factor_key": "a", "expression": "$close", "family": "价格", "data_safety": "safe"},
        {"factor_key": "b", "expression": " $close ", "family": "价格", "data_safety": "safe"},
        {"factor_key": "c", "expression": None, "skip_reason": "missing", "data_safety": "safe"},
    ])

    assert [item["factor_key"] for item in result["eligible"]] == ["a"]
    reasons = {item["factor_key"]: item["reject_reason"] for item in result["rejected"]}
    assert reasons["b"] == "duplicate_expression:a"
    assert reasons["c"] == "missing"


def test_preprocessing_accepts_isolated_factor_without_expression():
    result = preprocess_candidates([{
        "factor_key": "tech",
        "expression": None,
        "evaluation_mode": "isolation",
        "evaluation_signature": "tech",
        "data_safety": "safe",
    }])

    assert [item["factor_key"] for item in result["eligible"]] == ["tech"]
    assert result["eligible"][0]["expression_fingerprint"]


def test_preprocessing_rejects_missing_data_safety_as_unknown():
    result = preprocess_candidates([{"factor_key": "unknown", "expression": "$close"}])

    assert result["eligible"] == []
    assert result["rejected"][0]["reject_reason"] == "unsafe_data"


def test_architecture_normalize_uses_only_runtime_resolvable_variants(monkeypatch):
    from backend.service.factor_rolling import FactorRollingService

    variants = [
        {"variant_key": "size", "template_key": "size", "expr": None, "data_safety": "safe"},
        {"variant_key": "calendar", "template_key": "calendar", "expr": "DayOfWeek()==5?$close:0", "data_safety": "safe"},
        {"variant_key": "rownum", "template_key": "rownum", "expr": "RowNum()", "data_safety": "safe"},
        {"variant_key": "broken", "template_key": "broken", "expr": "[bad]", "data_safety": "safe"},
    ]
    resolved = {
        "size": ("(0-FundVal($close,1))", None),
        "calendar": ("If(DayOfWeek()==5,$close,0)", None),
        "rownum": ("$close/Ref($close,1)-1", None),
        "broken": (None, "bad"),
    }
    monkeypatch.setattr(
        FactorRollingService, "resolve_variants",
        staticmethod(lambda *_args, **_kwargs: variants),
    )
    monkeypatch.setattr(
        "backend.service.factor_architecture.resolve_factor_expression",
        lambda key: resolved[key],
    )

    class Pool:
        def get_active_factors(self):
            return {variant["variant_key"]: {} for variant in variants}

    monkeypatch.setattr("core.domain.factors.registry.get_pool", lambda: Pool())
    monkeypatch.setattr("core.domain.factors.selector.get_unified_category", lambda *_args, **_kwargs: "其他")

    config, normalized, preprocessing = FactorArchitectureService()._normalize({
        "start": "2022-01-01",
        "end": "2024-12-31",
    })

    assert config["label_mode"] == "tradable_open_to_open"
    eligible = {item["factor_key"] for item in preprocessing["eligible"]}
    assert eligible == {"size", "calendar", "rownum"}
    normalized_by_key = {item["variant_key"]: item for item in normalized}
    checker = OverfittingChecker()
    for key in eligible:
        isolation, _signature, use_expr, _ast, _outer = checker._resolve_eval(
            normalized_by_key[key], quiet=True,
        )
        assert isolation is not None or use_expr
    assert normalized_by_key["rownum"]["expr"] == "$close/Ref($close,1)-1"
    rejected = {item["factor_key"]: item["reject_reason"] for item in preprocessing["rejected"]}
    assert rejected["broken"].startswith("runtime_unresolved_expression:")


def test_architecture_rejects_non_tradable_label():
    with pytest.raises(ValueError, match="tradable_open_to_open"):
        FactorArchitectureService()._normalize({
            "start": "2022-01-01",
            "end": "2024-12-31",
            "label_mode": "close_to_close",
        })


def test_walkforward_locks_train_direction():
    checker = OverfittingChecker()
    plan = {
        "train": ["2024-01-02", "2024-01-09", "2024-01-16"],
        "valid": ["2024-10-01", "2024-10-08", "2024-10-15"],
        "test": ["2025-01-02", "2025-01-09", "2025-01-16"],
    }
    daily = [("2025-01-02", 0.04), ("2025-01-09", 0.05), ("2025-01-16", 0.06)]
    with patch.object(checker, "_calc_rank_ic", side_effect=[-0.04, -0.03, (0.05, daily)]):
        passed, _, detail = checker._check_walkforward({
            "variant": {"variant_key": "x", "expr": "$close"},
            "eval_weeks": 18,
            "anchor_date": None,
            "l1_dates": plan,
        })

    assert passed is False
    assert checker._last_walkforward["direction_sign"] == -1
    assert checker._last_walkforward["oriented_test_ic"] == pytest.approx(-0.05)
    assert checker._last_daily_ics == daily
    assert "定向Test IC=-0.0500" in detail


def test_evidence_summary_and_core_selection_use_state_floor():
    rows = []
    for fold, all_ic, high_ic, low_ic in (
        ("1", 0.03, 0.02, 0.025),
        ("2", 0.02, 0.015, 0.02),
        ("3", 0.025, 0.018, 0.022),
    ):
        for state, value in (("all", all_ic), ("high_vol", high_ic), ("low_vol", low_ic)):
            rows.append({
                "factor_key": "stable", "family": "量价", "fold_id": fold,
                "state": state, "oriented_ic": value, "coverage": 0.95,
            })
    summaries = summarize_factor_evidence(rows)
    selected, rejected = select_core_candidates(summaries, min_state_count=3)

    assert not rejected
    assert selected[0]["factor_key"] == "stable"
    assert selected[0]["state_floor_ic"] == pytest.approx(0.018)


def test_balanced_set_caps_single_family():
    candidates = [
        {"factor_key": f"a{i}", "family": "动量", "core_robustness": 1 - i / 10}
        for i in range(8)
    ] + [
        {"factor_key": f"b{i}", "family": "流动性", "core_robustness": 0.5 - i / 10}
        for i in range(4)
    ]

    result = build_balanced_set(candidates, max_count=10, family_fraction=0.2)

    assert sum(item["family"] == "动量" for item in result) == 2
    assert sum(item["family"] == "流动性" for item in result) == 2


def test_balanced_extension_preserves_base_and_adds_missing_families():
    candidates = [
        {"factor_key": "volume_1", "family": "成交量特征", "core_robustness": 1.0},
        {"factor_key": "volume_2", "family": "成交量特征", "core_robustness": 0.9},
        {"factor_key": "trend_1", "family": "趋势强度", "core_robustness": 0.8},
        {"factor_key": "vol_1", "family": "波动率特征", "core_robustness": 0.7},
        {"factor_key": "quality_1", "family": "高阶特征", "core_robustness": 0.6},
    ]

    result = build_balanced_extension(
        ["volume_1", "volume_2"], candidates, target_count=4, family_fraction=0.5,
    )

    assert [item["factor_key"] for item in result[:2]] == ["volume_1", "volume_2"]
    assert {item["family"] for item in result} == {"成交量特征", "趋势强度", "波动率特征"}


def test_unmatured_anchor_is_excluded_from_architecture_evidence():
    rolling = {
        "per_anchor": {
            "2026-01-01": {
                "window": {
                    "label_matured_at": "2026-05-01",
                    "test_start": "2026-01-01",
                    "test_end": "2026-03-31",
                },
                "detail": {
                    "f": {
                        "walkforward_metrics": {
                            "test_ic": 0.03, "oriented_test_ic": 0.03,
                            "direction_sign": 1, "test_count": 3,
                        },
                        "daily_ics": [["2026-01-02", 0.03]],
                    }
                },
            }
        }
    }
    config = {"selection_cutoff": "2026-04-30", "state_source": "test"}

    with patch.object(FactorArchitectureService, "_load_state_map", return_value={}):
        rows = FactorArchitectureService()._build_evidence(
            "run", rolling, {"f": "量价"}, config,
        )

    assert rows == []


def test_failed_rolling_factor_remains_in_core_evidence_denominator():
    rolling = {"per_anchor": {"2025-01-01": {
        "window": {"label_matured_at": "2025-05-01"},
        "detail": {"f": {
            "all_pass": False,
            "walkforward_metrics": {
                "test_ic": 0.04, "oriented_test_ic": 0.04,
                "direction_sign": 1, "test_count": 1, "passed": False,
            },
            "daily_ics": [["2025-01-02", 0.04]],
        }},
    }}}
    config = {"selection_cutoff": "2025-06-01", "state_source": "test"}
    with patch.object(FactorArchitectureService, "_load_state_map", return_value={}):
        rows = FactorArchitectureService()._build_evidence("run", rolling, {"f": "量价"}, config)
    assert len(rows) == 1
    assert rows[0]["factor_key"] == "f"
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["metrics"]["l1_pass"] is False
    assert rows[0]["metrics"]["dedup_pass"] is False


def test_evidence_positive_ratio_counts_failed_folds():
    rolling = {"per_anchor": {
        "2024-01-01": {
            "window": {"label_matured_at": "2024-05-01"},
            "detail": {"f": {
                "all_pass": False,
                "walkforward_metrics": {
                    "test_ic": -0.01, "oriented_test_ic": -0.01,
                    "direction_sign": 1, "test_count": 1, "passed": False,
                },
                "daily_ics": [["2024-01-02", -0.01]],
            }},
        },
        "2024-04-01": {
            "window": {"label_matured_at": "2024-08-01"},
            "detail": {"f": {
                "all_pass": True,
                "walkforward_metrics": {
                    "test_ic": 0.03, "oriented_test_ic": 0.03,
                    "direction_sign": 1, "test_count": 1, "passed": True,
                },
                "daily_ics": [["2024-04-02", 0.03]],
            }},
        },
    }}
    config = {"selection_cutoff": "2024-12-31", "state_source": "test"}
    with patch.object(FactorArchitectureService, "_load_state_map", return_value={}):
        rows = FactorArchitectureService()._build_evidence("run", rolling, {"f": "量价"}, config)
    summary = summarize_factor_evidence(rows)[0]
    assert summary["fold_count"] == 2
    assert summary["positive_fold_ratio"] == pytest.approx(0.5)
    assert summary["l1_pass_ratio"] == pytest.approx(0.5)
    assert summary["dedup_pass_ratio"] == pytest.approx(0.5)


def test_evidence_repair_prefilter_only_removes_impossible_core_candidates():
    rolling = {"per_anchor": {}}
    for fold in range(5):
        rolling["per_anchor"][str(fold)] = {"detail": {
            "stable": {"walkforward_metrics": {"oriented_test_ic": 0.02 if fold < 3 else -0.01}},
            "unstable": {"walkforward_metrics": {"oriented_test_ic": 0.02 if fold < 2 else -0.01}},
            "short": {"walkforward_metrics": {"oriented_test_ic": 0.02}} if fold < 2 else {},
        }}

    keys = FactorArchitectureService._repair_factor_keys(
        rolling, min_folds=3, min_positive_ratio=0.6,
    )

    assert keys == ["stable"]


def test_evidence_repair_reconstructs_sampled_test_window():
    with patch.object(OverfittingChecker, "_dates_between", side_effect=lambda start, end, step=1: [
        start + dt.timedelta(days=index) for index in range(0, (end - start).days + 1, step)
    ]):
        windows = FactorArchitectureService._repair_windows(["2025-01-01"], {
            "train_months": 12,
            "valid_months": 3,
            "test_months": 3,
            "purge_gap": 2,
            "label_horizon": 10,
            "max_eval_dates": 4,
        })

    plan = windows["2025-01-01"]
    assert len(plan["test_dates"]) == 4
    assert plan["test_dates"][0] == dt.date(2025, 1, 1)
    assert plan["window"]["label_matured_at"] == "2025-04-10"


def test_model_rounds_keep_full_train_window_and_purge_boundaries():
    rounds = _make_rounds({
        "start": "2025-01-01", "end": "2025-01-01", "step_months": 3,
        "test_months": 3, "train_months": 60, "valid_months": 6, "purge_gap": 21,
    })
    _, train, valid, test = rounds[0]
    assert train[0] == "2019-07-01"
    assert train[1] < valid[0]
    assert valid[1] < test[0]
    assert test == ("2025-01-01", "2025-03-31")


def test_explicitly_skipped_l5_is_replaced_by_batch_hard_gate():
    checker = OverfittingChecker()
    with patch.object(checker, "_check_walkforward", return_value=(True, 1.0, "ok")):
        result = checker.check_all_layers(
            {"variant_key": "f"}, mode="fast", skip_layers=[5], anchor_date=None,
        )
    assert result["all_pass"] is True
    assert next(layer for layer in result["layers"] if layer["id"] == 5)["status"] == "skipped"


def test_batch_dedup_updates_l5_gate_and_uses_train_valid_quality():
    passed = ["weak", "strong"]
    variants = {key: {"variant_key": key, "expr": "$close"} for key in passed}
    detail = {
        "weak": {
            "all_pass": True, "passed_count": 1,
            "layers": [{"id": 5, "status": "skipped", "pass": True, "detail": ""}],
            "walkforward_metrics": {
                "direction_sign": 1, "train_ic": 0.02, "oriented_valid_ic": -0.01,
            },
        },
        "strong": {
            "all_pass": True, "passed_count": 1,
            "layers": [{"id": 5, "status": "skipped", "pass": True, "detail": ""}],
            "walkforward_metrics": {
                "direction_sign": -1, "train_ic": -0.04, "oriented_valid_ic": 0.03,
            },
        },
    }
    captured = {}

    def fake_dedup(self, variants, **kwargs):
        captured.update(kwargs["selection_scores"])
        return {"kept": ["strong"], "unresolved": []}

    with patch.object(OverfittingChecker, "deduplicate_variants", fake_dedup):
        kept = _batch_deduplicate_passed(
            passed, detail, variants, {"2024-01-01": object()}, "a500", None,
            dt.date(2024, 2, 1), {"valid": [dt.date(2024, 1, 31)]}, None,
        )

    assert kept == ["strong"]
    assert captured == {"weak": -0.01, "strong": 0.03}
    assert detail["weak"]["all_pass"] is False
    assert detail["strong"]["layers"][0]["status"] == "passed"


def test_evenly_sample_dates_covers_full_window():
    values = [dt.date(2020, 1, 1) + dt.timedelta(days=index) for index in range(100)]
    sampled = evenly_sample_dates(values, 24)
    assert len(sampled) == 24
    assert sampled[0] == values[0]
    assert sampled[-1] == values[-1]


def test_orient_ic_uses_train_only():
    assert orient_ic(-0.01, -0.03) == pytest.approx(0.03)
    assert orient_ic(-0.01, 0.03) == pytest.approx(-0.03)


def test_paired_fold_delta_uses_matching_oos_folds():
    control = {"folds": [{"fold_id": "a", "rank_ic": 0.01}, {"fold_id": "b", "rank_ic": 0.02}]}
    challenger = {"folds": [{"fold_id": "a", "rank_ic": 0.03}, {"fold_id": "c", "rank_ic": 0.10}]}
    result = paired_fold_delta(control, challenger)
    assert result["fold_count"] == 1
    assert result["median_delta"] == pytest.approx(0.02)


def test_core_ab_requires_random_baseline_and_family_increment():
    folds = [
        {"fold_id": "a", "rank_ic": 0.03},
        {"fold_id": "b", "rank_ic": 0.04},
        {"fold_id": "c", "rank_ic": 0.02},
    ]
    results = {
        "core_full": {"rank_ic": 0.03, "folds": folds},
        "minus_momentum": {"folds": [
            {"fold_id": "a", "rank_ic": 0.01},
            {"fold_id": "b", "rank_ic": 0.02},
            {"fold_id": "c", "rank_ic": 0.01},
        ]},
        "random_01": {"rank_ic": 0.00, "folds": []},
        "random_02": {"rank_ic": 0.01, "folds": []},
    }
    analysis = analyze_core_ab(
        results, {"momentum": "minus_momentum"}, min_random_count=2,
    )
    assert analysis["validated"] is True
    assert analysis["random_percentile"] == 1.0


def test_core_fixed_oos_requires_reference_and_random_improvement():
    results = {
        "core_full": {"rank_ic": 0.04, "folds": [
            {"fold_id": "a", "rank_ic": 0.04},
            {"fold_id": "b", "rank_ic": 0.05},
            {"fold_id": "c", "rank_ic": 0.03},
        ]},
        "core_reference": {"rank_ic": 0.02, "folds": [
            {"fold_id": "a", "rank_ic": 0.02},
            {"fold_id": "b", "rank_ic": 0.03},
            {"fold_id": "c", "rank_ic": 0.02},
        ]},
        **{f"random_{index:02d}": {"rank_ic": 0.01} for index in range(10)},
    }

    analysis = analyze_core_oos(results, min_random_count=10)

    assert analysis["validated"] is True
    assert analysis["scope"] == "fixed_oos"
    assert analysis["winner_arm"] == "core_full"
    assert analysis["candidate_results"]["core_full"]["random_percentile"] == 1.0


def test_core_fixed_oos_selects_minimal_candidate_when_it_is_best():
    results = {
        "core_full": {"rank_ic": 0.04, "folds": [
            {"fold_id": key, "rank_ic": 0.04} for key in ("a", "b", "c")
        ]},
        "core_reference": {"rank_ic": 0.02, "folds": [
            {"fold_id": key, "rank_ic": 0.02} for key in ("a", "b", "c")
        ]},
        "core_minimal": {"rank_ic": 0.05, "folds": [
            {"fold_id": key, "rank_ic": 0.05} for key in ("a", "b", "c")
        ]},
        **{f"random_{index:02d}": {"rank_ic": 0.01} for index in range(10)},
    }

    analysis = analyze_core_oos(results, min_random_count=10)

    assert analysis["validated"] is True
    assert analysis["winner_arm"] == "core_minimal"


def test_prepared_factor_dataset_subsets_shared_frames():
    import pandas as pd

    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "SH600000")], names=["datetime", "instrument"],
    )
    columns = pd.MultiIndex.from_tuples([
        ("feature", "a"), ("feature", "b"), ("label", "LABEL0"),
    ])
    frame = pd.DataFrame([[1.0, 2.0, 0.1]], index=index, columns=columns)

    dataset = _PreparedFactorDataset({"train": frame, "valid": frame, "test": frame}, ["b"])

    assert list(dataset.prepare("train", col_set="feature").columns) == ["b"]
    assert dataset.prepare("test", col_set="label").iloc[0, 0] == pytest.approx(0.1)


def test_satellite_ab_rejects_unstable_increment():
    results = {
        "core_control": {"folds": [
            {"fold_id": "a", "rank_ic": 0.02},
            {"fold_id": "b", "rank_ic": 0.02},
            {"fold_id": "c", "rank_ic": 0.02},
        ]},
        "core_plus_trend": {"folds": [
            {"fold_id": "a", "rank_ic": 0.04},
            {"fold_id": "b", "rank_ic": 0.01},
            {"fold_id": "c", "rank_ic": 0.01},
        ]},
    }
    analysis = analyze_satellite_ab(results, "trend")
    assert analysis["validated"] is False


def test_model_ab_defaults_to_ten_random_sets_and_targets_exact_version():
    service = FactorArchitectureService()
    run = {"run_id": "run", "config": {}}
    sets = [
        {"set_id": "run_core_v1", "role": "core", "status": "rejected", "factor_keys": ["a"]},
        {"set_id": "run_core_v2", "role": "core", "status": "screened", "factor_keys": ["b"]},
    ]
    captured = {}

    def submit_now(task_id, fn, description, pool=None):
        fn()
        return task_id

    def capture_run(task_id, run, target, core, request):
        captured.update({"target": target, "request": request})

    with patch.object(factor_architecture_repo, "get_run", return_value=run), \
            patch.object(factor_architecture_repo, "list_sets", return_value=sets), \
            patch.object(factor_architecture_repo, "list_candidates", return_value=[{
                "factor_key": "a", "status": "eligible",
                "metadata": {"expression": "$close", "data_safety": "safe"},
            }]), \
            patch.object(factor_architecture_repo, "update_run") as update_run, \
            patch.object(task_manager, "submit_with_id", side_effect=submit_now), \
            patch.object(service, "_run_model_ab", side_effect=capture_run):
        result = service.create_model_ab("run", {"role": "core", "set_id": "run_core_v1"})

    assert result["set_id"] == "run_core_v1"
    assert captured["target"]["set_id"] == "run_core_v1"
    assert captured["request"]["random_count"] == 10
    assert update_run.call_args.kwargs["result"]["active_model_ab_task_id"] == result["task_id"]


def test_model_ab_rejects_persisted_unresolved_factor_before_submit():
    service = FactorArchitectureService()
    run = {"run_id": "run", "config": {}}
    sets = [{
        "set_id": "run_core_v1", "role": "core", "status": "screened",
        "factor_keys": ["flow"],
    }]
    candidates = [{
        "factor_key": "flow", "status": "eligible",
        "metadata": {"expression": None},
    }]

    with patch.object(factor_architecture_repo, "get_run", return_value=run), \
            patch.object(factor_architecture_repo, "list_sets", return_value=sets), \
            patch.object(factor_architecture_repo, "list_candidates", return_value=candidates), \
            patch.object(task_manager, "submit_with_id") as submit:
        with pytest.raises(ValueError, match="flow"):
            service.create_model_ab("run", {"role": "core", "set_id": "run_core_v1"})

    submit.assert_not_called()


def test_core_ab_random_arms_exclude_unresolved_candidates():
    arms, _ = FactorArchitectureService._build_core_ab_arms(
        ["a"],
        [
            {"factor_key": "a", "status": "eligible", "metadata": {"expression": "$close"}},
            {"factor_key": "flow", "status": "eligible", "metadata": {"expression": None}},
        ],
        {"a": "test", "flow": "test"},
        random_count=1,
        seed=1,
    )

    assert arms["random_01"] == ["a"]


def test_cancel_targets_active_model_ab_subtask():
    service = FactorArchitectureService()
    run = {"run_id": "run", "result": {"active_model_ab_task_id": "run_ab"}}

    with patch.object(factor_architecture_repo, "get_run", return_value=run), \
            patch.object(factor_architecture_repo, "update_run") as update_run, \
            patch.object(task_manager, "is_active", side_effect=lambda task_id: task_id == "run_ab"), \
            patch.object(task_manager, "request_cancel", return_value=True) as request_cancel:
        result = service.cancel("run")

    assert result["cancel_requested"] is True
    request_cancel.assert_called_once_with("run_ab")
    update_run.assert_called_once_with("run", status="cancel_requested")


def test_factor_set_evaluator_reuses_complete_fold_checkpoint():
    checkpoint = {
        "fold_id": "2025-01-01",
        "rank_ic": 0.03,
        "icir": 1.0,
        "positive_ratio": 1.0,
        "n_dates": 2,
        "daily_ics": [0.02, 0.04],
    }
    config = {
        "start": "2025-01-01", "end": "2025-01-01",
        "step_months": 3, "test_months": 3,
        "train_months": 60, "valid_months": 6,
        "purge_gap": 21, "label_horizon": 10,
        "selection_cutoff": "2025-12-31",
    }

    with patch("core.adapters.qlib.models.QlibModelManager") as manager:
        result = FactorSetEvaluator().evaluate_factor_sets(
            {"core_full": ["a"]}, config,
            completed_folds={"core_full": {"2025-01-01": checkpoint}},
        )

    manager.return_value.train.assert_not_called()
    assert result["core_full"]["rank_ic"] == pytest.approx(0.03)
    assert result["core_full"]["n_dates"] == 2


def test_daily_spearman_matrix_matches_per_factor_calculation():
    import pandas as pd

    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["a", "b", "c", "d"]],
        names=["datetime", "instrument"],
    )
    features = pd.DataFrame({
        "f1": [1, 3, 2, 4, 8, 6, 7, 5],
        "f2": [4, 1, 3, 2, 5, 8, 6, 7],
    }, index=index, dtype=float)
    target = pd.Series([2, 4, 1, 3, 7, 5, 8, 6], index=index, dtype=float)

    actual = _daily_spearman_matrix(features, target)
    for date, frame in features.groupby(level="datetime"):
        expected = frame.corrwith(target.loc[frame.index], method="spearman")
        assert actual.loc[date].to_dict() == pytest.approx(expected.to_dict())


def test_residual_evaluator_reuses_complete_fold_checkpoint():
    checkpoint = {
        "core_fold": {"fold_id": "2025-01-01", "ic_mean": 0.03, "n_dates": 1},
        "residual_rows": [{
            "factor_key": "sat", "fold_id": "2025-01-01", "date": "2025-01-02",
            "state": "low_vol", "residual_ic": 0.02,
        }],
    }
    config = {
        "start": "2025-01-01", "end": "2025-01-01",
        "step_months": 3, "test_months": 3,
        "train_months": 60, "valid_months": 6,
        "purge_gap": 21, "label_horizon": 10,
        "selection_cutoff": "2025-12-31",
    }

    with patch("core.adapters.qlib.models.QlibModelManager") as manager:
        result = FactorSetEvaluator().evaluate_residual_candidates(
            ["core"], ["sat"], config,
            completed_folds={"2025-01-01": checkpoint},
        )

    manager.return_value._build_task_config.assert_not_called()
    assert result["core_folds"] == [checkpoint["core_fold"]]
    assert result["residual_rows"] == checkpoint["residual_rows"]


def test_evaluation_checkpoint_replaces_same_arm_and_fold():
    calls = []
    with patch("core.data.repos.factor_architecture.Database.execute", side_effect=lambda sql, params: calls.append((sql, params))):
        factor_architecture_repo.save_evaluations([{
            "run_id": "run", "set_id": "set", "arm": "core_full",
            "fold_id": "2025-01-01", "metrics": {"rank_ic": 0.03},
        }])

    assert len(calls) == 2
    assert "fold_id <=> %s" in calls[0][0]
    assert "INSERT INTO factor_set_evaluations" in calls[1][0]
