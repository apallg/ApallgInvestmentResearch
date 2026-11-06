import copy
import datetime

import pytest

from core.domain.portfolio.strategy_profile import (
    DEFAULT_DIAGNOSTIC_CONFIG,
    assert_single_change,
    canonical_pool_name,
    evaluate_oos_evidence,
    evaluate_paper_evidence,
    normalize_profile_config,
    operation_config,
    profile_fingerprint,
    validate_profile_contract,
)


pytestmark = [pytest.mark.unit, pytest.mark.p0]


def valid_config():
    config = copy.deepcopy(DEFAULT_DIAGNOSTIC_CONFIG)
    config["data"]["snapshot_id"] = "a" * 64
    config["factors"]["version"] = "1.0.0"
    config["factors"]["keys"] = [f"f{i:02d}" for i in range(30)]
    config["factors"]["evidence"] = [
        {
            "key": key, "rank_ic": 0.03, "icir": 0.4, "positive_date_ratio": 0.65,
            "positive_folds": 4, "fold_count": 5, "train_rank_ic": 0.05, "coverage": 0.9,
        }
        for key in config["factors"]["keys"]
    ]
    config["model"]["refs"] = [{"experiment_id": "1", "run_id": "run-1"}]
    config["model"]["refs"][0]["artifact_fingerprint"] = "b" * 64
    config["model"]["weights"] = [1.0]
    windows = [
        ("2023-07-01", "2023-09-30"), ("2023-10-01", "2023-12-31"),
        ("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-06-30"),
        ("2024-07-01", "2024-09-30"),
    ]
    schedule = []
    for index, (start, end) in enumerate(windows):
        anchor = datetime.date.fromisoformat(start)
        month_index = anchor.year * 12 + anchor.month - 1 - 6
        valid_year, valid_month = divmod(month_index, 12)
        valid_start = datetime.date(valid_year, valid_month + 1, 1)
        train_start = valid_start.replace(year=valid_start.year - 5)
        schedule.append({
            "window_start": start, "window_end": end,
            "experiment_id": "1", "run_id": f"fold-{index}",
            "artifact_fingerprint": str(index + 1) * 64,
            "train_start": train_start.isoformat(),
            "train_end": (valid_start - datetime.timedelta(days=12)).isoformat(),
            "valid_start": valid_start.isoformat(),
            "valid_end": (anchor - datetime.timedelta(days=12)).isoformat(),
        })
    config["model"]["oos_schedule"] = schedule
    config["fingerprints"] = {"factor_definitions": "c" * 64}
    config["windows"] = {
        "train_start": "2018-01-01", "train_end": "2022-12-31",
        "valid_start": "2023-01-01", "valid_end": "2023-06-30",
        "oos_start": "2023-07-01", "oos_end": "2024-12-31",
    }
    return config


def test_profile_normalization_and_fingerprint_are_deterministic():
    left = valid_config()
    right = copy.deepcopy(left)
    right["universe"]["name"] = "a500"
    right["universe"]["code_prefixes"].reverse()
    assert canonical_pool_name("a500") == "stable_history_500d"
    assert profile_fingerprint(left) == profile_fingerprint(right)
    right["factors"]["keys"].reverse()
    assert profile_fingerprint(left) != profile_fingerprint(right)


def test_valid_profile_contract_and_locked_operation_config():
    config = valid_config()
    result = validate_profile_contract(config)
    assert result["valid"] is True
    screening = operation_config(config, "screening")
    backtest = operation_config(config, "backtest")
    assert screening["pool"] == "stable_history_500d"
    assert screening["top_n"] == 50
    assert screening["qlib_model"] == ["1", "run-1"]
    assert screening["neutralize_qlib"] is False
    assert backtest["top_n"] == 5
    assert backtest["freq"] == "week"
    assert backtest["stop_source"] == "off"
    assert backtest["fixed_model"] == ["1", "run-1"]


def test_profile_contract_rejects_unfrozen_or_unproven_inputs():
    config = valid_config()
    config["training"]["random_split"] = True
    config["model"]["refs"].append({"experiment_id": "1", "run_id": "run-2"})
    config["model"]["weights"] = [0.5, 0.5]
    config["factors"]["evidence"][0]["coverage"] = 0.5
    config["windows"]["valid_start"] = "2022-01-01"
    result = validate_profile_contract(config)
    assert result["valid"] is False
    assert any("随机切分" in reason for reason in result["errors"])
    assert any("独立run" in reason for reason in result["errors"])
    assert any("覆盖率" in reason for reason in result["errors"])
    assert any("不重叠" in reason for reason in result["errors"])


def test_profile_contract_rejects_missing_walk_forward_model_schedule():
    config = valid_config()
    config["model"]["oos_schedule"] = []
    result = validate_profile_contract(config)
    assert result["valid"] is False
    assert any("walk-forward" in reason for reason in result["errors"])


def test_research_satellite_profile_is_runnable_but_not_promotable():
    config = valid_config()
    config["factors"]["set_name"] = "core_trend_satellite"
    config["factors"]["version"] = "bdba67f459d2b928178s504cf_trend_v2"
    config["factors"]["keys"] = [f"f{i:02d}" for i in range(40)]
    config["factors"]["evidence"] = []
    config["factors"]["research_only"] = True
    config["experiment"] = {"kind": "research_candidate", "changed_field": None}
    config["model"]["shadow_refs"] = [{
        "experiment_id": "1", "run_id": "shadow-1", "role": "rebound_observer",
        "factor_set_name": "core_rebound_satellite", "factor_set_version": "rebound-v2",
        "factor_keys": ["r1"], "artifact_fingerprint": "d" * 64,
        "factor_definitions_fingerprint": "e" * 64,
        "affects_selection": False, "research_only": True,
    }]

    result = validate_profile_contract(config)
    screening = operation_config(config, "screening")

    assert result["valid"] is False
    assert any("研究候选策略" in reason for reason in result["errors"])
    assert not any("core_trend_satellite必须冻结为40" in reason for reason in result["errors"])
    assert screening["qlib_model"] == ["1", "run-1"]
    assert screening["shadow_models"] == [{
        "model": ["1", "shadow-1"], "role": "rebound_observer",
        "factor_set_name": "core_rebound_satellite", "affects_selection": False,
    }]


def test_shadow_model_cannot_change_selection():
    config = valid_config()
    config["model"]["shadow_refs"] = [{
        "experiment_id": "1", "run_id": "shadow-1", "role": "observer",
        "factor_set_name": "shadow", "factor_set_version": "v1", "factor_keys": ["r1"],
        "artifact_fingerprint": "d" * 64, "factor_definitions_fingerprint": "e" * 64,
        "affects_selection": True, "research_only": True,
    }]

    result = validate_profile_contract(config)

    assert any("不得改变排名" in reason for reason in result["errors"])


def test_profile_prediction_keeps_shadow_output_out_of_primary_ranking(monkeypatch):
    from backend.service.prediction import PredictionService

    def fake_predict(**kwargs):
        run_id = kwargs["run_id"]
        codes = ["A", "B", "C", "D", "E"] if run_id == "primary" else ["B", "C", "D", "E", "F"]
        return {
            "count": len(codes),
            "by_date": {"2026-09-03": [{"instrument": code, "score": 1.0} for code in codes]},
            "strategy_profile": kwargs.get("strategy_profile"),
        }

    monkeypatch.setattr(PredictionService, "_do_predict_dual", staticmethod(fake_predict))
    monkeypatch.setattr(PredictionService, "_do_predict", staticmethod(fake_predict))
    monkeypatch.setattr("backend.service.prediction.assert_run_path", lambda *_args: __import__("pathlib").Path("."))
    result = PredictionService._do_profile_predict(
        fake_predict,
        {
            "experiment_id": "1", "run_id": "primary", "instruments": "a500",
            "start_time": "2026-09-03", "end_time": "2026-09-03", "topk": 50,
            "skip_features": True, "code_prefixes": ["600"],
            "strategy_profile": {"promotable": False}, "profile_factor_keys": ["main"],
            "neutralize_scores": False, "industry_map": {},
            "fit_start_time": "2019-01-01", "fit_end_time": "2024-12-31",
        },
        [{
            "experiment_id": "1", "run_id": "shadow", "role": "rebound_observer",
            "factor_set_name": "core_rebound_satellite", "factor_keys": ["shadow"],
        }],
        ["历史证券状态快照尚未完整"],
    )

    assert [row["instrument"] for row in result["by_date"]["2026-09-03"]] == ["A", "B", "C", "D", "E"]
    assert result["shadow_model_observations"][0]["top5_overlap"]["2026-09-03"] == 0.8
    assert result["shadow_model_observations"][0]["changes_selection"] is False
    assert result["strategy_profile"]["promotable"] is False
    assert result["warnings"] == ["历史证券状态快照尚未完整"]


def test_model_ref_for_date_uses_frozen_oos_fold_and_never_falls_back():
    from backend.service.strategy_profile import StrategyProfileService

    profile = {"config": valid_config()}
    selected, scheduled = StrategyProfileService.model_ref_for_date(profile, "2024-02-01", True)
    assert selected["run_id"] == "fold-2"
    assert scheduled is True
    with pytest.raises(ValueError, match="未匹配"):
        StrategyProfileService.model_ref_for_date(profile, "2025-02-01", True)


@pytest.mark.parametrize("field,value", [
    ("commission_rate", -0.0003),
    ("slippage_rate", -0.0001),
    ("stamp_duty", -0.0005),
])
def test_profile_contract_rejects_negative_costs(field, value):
    config = valid_config()
    config["execution"][field] = value
    result = validate_profile_contract(config)
    assert result["valid"] is False
    assert any("成本" in reason for reason in result["errors"])


def test_profile_contract_rejects_non_pit_prices_and_unisolated_features():
    config = valid_config()
    config["data"]["price_adjustment"] = "none"
    config["features"]["timing"] = True
    config["features"]["model_routing"] = True
    config["features"]["stop_source"] = "fixed"
    result = validate_profile_contract(config)
    assert result["valid"] is False
    assert any("复权" in reason for reason in result["errors"])
    assert any("诊断基线" in reason for reason in result["errors"])


def test_ablation_allows_exactly_one_meaningful_change():
    base = valid_config()
    candidate = copy.deepcopy(base)
    candidate["score"]["neutralize"] = True
    candidate["data"]["snapshot_id"] = "b" * 64
    result = assert_single_change(base, candidate)
    assert result == {"valid": True, "changed_fields": ["score.neutralize"]}
    candidate["selection"]["top_n"] = 10
    assert assert_single_change(base, candidate)["valid"] is False


def test_oos_gate_passes_only_complete_stable_evidence():
    returns = [0.003] * 10 + [-0.001] * 2
    windows = [
        ("2023-01-01", "2023-03-31"), ("2023-04-01", "2023-06-30"),
        ("2023-07-01", "2023-09-30"), ("2023-10-01", "2023-12-31"),
        ("2024-01-01", "2024-03-31"),
    ]
    folds = [
        {
            "annual_excess": 0.04 + index * 0.001,
            "max_drawdown": -0.10,
            "one_way_turnover": 0.25,
            "excess_calmar": 0.41,
            "date_excess_returns": returns,
            "benchmark_coverage": 0.95,
            "window_start": windows[index][0],
            "window_end": windows[index][1],
            "decision_dates": [
                (datetime.date.fromisoformat(windows[index][0]) + datetime.timedelta(days=7 * week)).isoformat()
                for week in range(12)
            ],
        }
        for index in range(5)
    ]
    result = evaluate_oos_evidence(folds, champion_excess_calmar=0.3)
    assert result["passed"] is True
    assert result["summary"]["fold_count"] == 5
    folds[0]["max_drawdown"] = -0.21
    assert "最大回撤差于-20%" in evaluate_oos_evidence(folds)["failures"]


def test_oos_gate_rejects_repeated_or_overlapping_windows():
    folds = [
        {
            "annual_excess": 0.04, "max_drawdown": -0.10, "one_way_turnover": 0.20,
            "excess_calmar": 0.4, "date_excess_returns": [0.001] * 20,
            "benchmark_coverage": 0.95, "window_start": "2024-01-01", "window_end": "2024-03-31",
            "decision_dates": [(datetime.date(2024, 1, 1) + datetime.timedelta(days=7 * week)).isoformat() for week in range(12)],
        }
        for _ in range(5)
    ]
    result = evaluate_oos_evidence(folds)
    assert result["passed"] is False
    assert "OOS折时间窗口重复" in result["failures"]
    assert "OOS折时间窗口重叠" in result["failures"]


def test_oos_gate_rejects_one_day_folds_and_missing_weekly_decisions():
    folds = [
        {
            "annual_excess": 0.04, "max_drawdown": -0.10, "one_way_turnover": 0.20,
            "excess_calmar": 0.4, "date_excess_returns": [0.001],
            "decision_dates": [f"2024-0{index + 1}-01"],
            "benchmark_coverage": 0.95,
            "window_start": f"2024-0{index + 1}-01", "window_end": f"2024-0{index + 1}-01",
        }
        for index in range(5)
    ]
    result = evaluate_oos_evidence(folds)
    assert result["passed"] is False
    assert any("完整3个月" in reason for reason in result["failures"])
    assert any("周度决策日期" in reason for reason in result["failures"])


def test_paper_gate_counts_only_mature_valid_periods():
    periods = [
        {
            "matured": True, "valid": True, "excess_return": 0.002,
            "drawdown": -0.08, "one_way_turnover": 0.19, "signal_consistency": 1.0,
            "decision_date": (datetime.date(2024, 1, 1) + datetime.timedelta(days=7 * index)).isoformat(),
        }
        for index in range(20)
    ]
    assert evaluate_paper_evidence(periods, 0.2)["passed"] is True
    periods[0]["signal_consistency"] = 0.99
    assert evaluate_paper_evidence(periods, 0.2)["passed"] is False


def test_paper_gate_rejects_twenty_consecutive_days():
    periods = [
        {
            "matured": True, "valid": True, "excess_return": 0.002,
            "drawdown": -0.08, "one_way_turnover": 0.19, "signal_consistency": 1.0,
            "decision_date": (datetime.date(2024, 1, 1) + datetime.timedelta(days=index)).isoformat(),
        }
        for index in range(20)
    ]
    result = evaluate_paper_evidence(periods)
    assert result["passed"] is False
    assert any("独立周度" in reason for reason in result["failures"])


def test_live_execution_requires_a_champion(monkeypatch):
    from backend.service.strategy_profile import StrategyProfileService

    monkeypatch.setattr(StrategyProfileService, "champion", classmethod(lambda cls: None))
    with pytest.raises(ValueError, match="没有Champion"):
        StrategyProfileService.ensure_execution_allowed(None, None, None)


def test_live_execution_can_explicitly_bypass_strategy_profile_gate(monkeypatch):
    from backend.service.strategy_profile import StrategyProfileService

    monkeypatch.setattr(StrategyProfileService, "champion", classmethod(lambda cls: None))
    assert StrategyProfileService.ensure_execution_allowed(None, None, None, enforce=False) is True


def test_live_execution_requires_exact_champion_identity(monkeypatch):
    from backend.service.strategy_profile import StrategyProfileService

    monkeypatch.setattr(StrategyProfileService, "champion", classmethod(lambda cls: {
        "profile_id": "champion", "version": 3, "fingerprint": "f" * 64,
    }))
    monkeypatch.setattr(StrategyProfileService, "runtime_health_failures", classmethod(lambda cls, _profile: []))
    with pytest.raises(ValueError, match="只有当前Champion"):
        StrategyProfileService.ensure_execution_allowed("challenger", 1, "f" * 64)
    with pytest.raises(ValueError, match="指纹不一致"):
        StrategyProfileService.ensure_execution_allowed("champion", 3, "a" * 64)
    assert StrategyProfileService.ensure_execution_allowed("champion", 3, "f" * 64) is True


def test_live_execution_blocks_runtime_drift(monkeypatch):
    from backend.service.strategy_profile import StrategyProfileService

    monkeypatch.setattr(StrategyProfileService, "champion", classmethod(lambda cls: {
        "profile_id": "champion", "version": 3, "fingerprint": "f" * 64,
    }))
    monkeypatch.setattr(
        StrategyProfileService, "runtime_health_failures",
        classmethod(lambda cls, _profile: ["Champion数据内容发生漂移，禁止新增买入"]),
    )
    with pytest.raises(ValueError, match="数据内容发生漂移"):
        StrategyProfileService.ensure_execution_allowed("champion", 3, "f" * 64)


def test_runtime_health_blocks_historical_drawdown_breach(monkeypatch):
    from backend.service.strategy_profile import StrategyProfileService
    from core.data import StrategyProfileRepo

    champion = {
        "profile_id": "champion", "version": 3,
        "config": {"data": {"snapshot_id": "a" * 64}, "model": {"refs": []}},
    }
    monkeypatch.setattr(StrategyProfileRepo, "get_snapshot", classmethod(lambda cls, _snapshot_id: {"snapshot_id": "a" * 64}))
    monkeypatch.setattr(
        StrategyProfileRepo, "get_runtime_health",
        classmethod(lambda cls, _profile_id, _version: {
            "drawdown_breached": 1, "current_drawdown": -0.05, "max_drawdown": -0.21,
        }),
    )
    monkeypatch.setattr(StrategyProfileService, "_snapshot_content_matches", classmethod(lambda cls, _snapshot: True))
    failures = StrategyProfileService.runtime_health_failures(champion)
    assert any("最大回撤已越过-20%" in reason and "-21.00%" in reason for reason in failures)
