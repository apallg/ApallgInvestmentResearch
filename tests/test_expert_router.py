import numpy as np
import pandas as pd
import pytest

from core.adapters.qlib.lstm_fusion import _date_weight_array, _purged_end_date
from scripts.run_expert_router_oos import (
    ROUTER_FEATURES,
    _fit_router,
    _evaluate_prediction,
    _fit_future_vol_probability,
    _hard_vol_switch,
    _matured_history,
    _rank_fusion,
    _router_diagnostics,
    _simulate_topn_portfolio,
    _state_training_profile,
    validate_windows,
)


def _panel(periods=6):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=periods, freq="B"), ["a", "b", "c", "d", "e", "f"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.Series(np.arange(len(index), dtype=float), index=index)
    label = pd.Series(np.linspace(-0.01, 0.02, len(index)), index=index)
    return prediction, label


def test_windows_must_be_chronological_and_non_overlapping():
    validate_windows({
        "w1": {"train": ("2020-01-01", "2021-12-31"), "valid": ("2022-01-01", "2022-06-30"),
               "test": ("2022-07-01", "2022-12-31")},
        "w2": {"train": ("2020-07-01", "2022-06-30"), "valid": ("2022-07-01", "2022-12-31"),
               "test": ("2023-01-01", "2023-06-30")},
    }, ["w1", "w2"])
    with pytest.raises(ValueError, match="重叠"):
        validate_windows({
            "w1": {"train": ("2020-01-01", "2021-12-31"), "valid": ("2022-01-01", "2022-06-30"),
                   "test": ("2022-07-01", "2023-03-31")},
            "w2": {"train": ("2020-07-01", "2022-06-30"), "valid": ("2022-07-01", "2022-12-31"),
                   "test": ("2023-01-01", "2023-06-30")},
        }, ["w1", "w2"])


def test_router_only_uses_labels_matured_before_cutoff():
    history = [
        {"date": "2024-01-02", "label_matured_at": "2024-01-16", "target": 1},
        {"date": "2024-01-16", "label_matured_at": "2024-01-30", "target": 0},
    ]
    assert [row["date"] for row in _matured_history(history, "2024-01-20")] == ["2024-01-02"]


def test_non_overlapping_portfolio_periods_apply_double_sided_costs():
    prediction, label = _panel(periods=6)
    dates = sorted(set(prediction.index.get_level_values("datetime")))
    execution = {str(pd.Timestamp(d).date()): str((pd.Timestamp(d) + pd.offsets.BDay(1)).date()) for d in dates}
    maturity = {str(pd.Timestamp(d).date()): str((pd.Timestamp(d) + pd.offsets.BDay(3)).date()) for d in dates}
    result = _simulate_topn_portfolio(
        prediction, label, top_n=2, rebalance_step=2,
        execution_dates=execution, maturity_dates=maturity,
        limit_up=set(), limit_down=set(), buy_cost_rate=0.0004, sell_cost_rate=0.0009,
    )
    assert result["n_periods"] == 3
    assert result["periods"][0]["cost"] == pytest.approx(0.0004)
    assert result["periods"][0]["label_matured_at"] == maturity[result["periods"][0]["date"]]
    assert result["turnover"] >= 0
    assert result["transaction_cost"] > 0
    assert "max_drawdown" in result


@pytest.mark.p0
def test_sparse_soft_predictions_share_calendar_and_annualization_with_baseline():
    prediction, label = _panel(periods=31)
    dates = sorted(set(prediction.index.get_level_values("datetime")))
    decisions = dates[::10]
    execution = {str(d.date()): str((d + pd.offsets.BDay(1)).date()) for d in dates}
    maturity = {str(d.date()): str((d + pd.offsets.BDay(11)).date()) for d in dates}
    constraints = {"limit_up": set(), "limit_down": set()}
    sparse = prediction[prediction.index.get_level_values("datetime").isin(decisions)]
    baseline = _evaluate_prediction(
        prediction, label, execution, maturity, constraints, decision_dates=decisions)
    soft = _evaluate_prediction(
        sparse, label, execution, maturity, constraints, decision_dates=decisions)

    for key in baseline:
        assert [row["date"] for row in soft[key]["periods"]] == [str(d.date()) for d in decisions]
        assert soft[key]["n_periods"] == baseline[key]["n_periods"] == 4
        for metric in ("total_return", "annual_return", "sharpe", "max_drawdown", "transaction_cost"):
            assert soft[key][metric] == pytest.approx(baseline[key][metric])
        expected_annual = (1 + soft[key]["total_return"]) ** (252 / 10 / 4) - 1
        assert soft[key]["annual_return"] == pytest.approx(expected_annual)


def test_explicit_portfolio_calendar_rejects_missing_prediction_dates():
    prediction, label = _panel(periods=31)
    decisions = sorted(set(prediction.index.get_level_values("datetime")))[::10]
    sparse = prediction[prediction.index.get_level_values("datetime").isin(decisions[:1])]

    with pytest.raises(ValueError, match="决策日缺少"):
        _simulate_topn_portfolio(sparse, label, 1, decision_dates=decisions)


def test_final_matured_launcher_preserves_existing_lock_and_results(tmp_path, monkeypatch):
    import json
    import runpy
    import sys
    from pathlib import Path
    import scripts.run_expert_router_oos as experiment

    script = Path(__file__).parents[1] / "scripts/run_expert_router_final_matured.py"
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "data/experiments"
    output_dir.mkdir(parents=True)
    target = output_dir / "expert_router_oos_h10_matured.json"
    original = json.dumps({"final_oos_lock": {"result_hash": "locked"}, "final_oos": {"score": 1}})
    target.write_text(original, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.setattr(experiment, "FINAL_OOS", dict(experiment.FINAL_OOS))

    def main():
        assert target.read_text(encoding="utf-8") == original
        assert experiment._load_previous(target)["final_oos_lock"]["result_hash"] == "locked"
        raise RuntimeError("最终OOS已评估，禁止重复查看")

    monkeypatch.setattr(experiment, "main", main)
    with pytest.raises(RuntimeError, match="禁止重复查看"):
        runpy.run_path(str(script), run_name="__main__")
    assert target.read_text(encoding="utf-8") == original


def test_final_matured_launcher_initializes_only_new_output(tmp_path, monkeypatch):
    import json
    import runpy
    import sys
    from pathlib import Path
    from unittest.mock import Mock
    import scripts.run_expert_router_oos as experiment

    script = Path(__file__).parents[1] / "scripts/run_expert_router_final_matured.py"
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "data/experiments"
    output_dir.mkdir(parents=True)
    source = output_dir / "expert_router_oos_h10_parallel2.json"
    previous = {
        "config": {"factor_groups": {"core": ["a"], "trend": ["b"], "rebound": ["c"]},
                   "instruments": ["SH600000"]},
        "windows": {"w1": {"score": 1}},
    }
    original = json.dumps(previous)
    source.write_text(original, encoding="utf-8")
    main = Mock()
    monkeypatch.setattr(experiment, "main", main)
    monkeypatch.setattr(experiment, "FINAL_OOS", dict(experiment.FINAL_OOS))
    monkeypatch.setattr(sys, "argv", [str(script)])

    runpy.run_path(str(script), run_name="__main__")

    target = json.loads((output_dir / "expert_router_oos_h10_matured.json").read_text(encoding="utf-8"))
    assert source.read_text(encoding="utf-8") == original
    assert target["windows"] == previous["windows"]
    assert target["config"]["final_oos"]["test"] == ["2026-07-01", "2026-08-10"]
    assert target["experiment_fingerprint"] == experiment._stable_hash(target["config"])
    main.assert_called_once_with()


def test_limit_up_blocks_buy_and_limit_down_blocks_sell():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["datetime", "instrument"])
    prediction = pd.Series([2, 1, 2, 1, 1, 2, 1, 2], index=index, dtype=float)
    label = pd.Series(0.01, index=index)
    execution = {str(d.date()): str((d + pd.offsets.BDay(1)).date()) for d in dates}
    maturity = {str(d.date()): str((d + pd.offsets.BDay(2)).date()) for d in dates}
    first_exec = execution[str(dates[0].date())]
    third_exec = execution[str(dates[2].date())]
    result = _simulate_topn_portfolio(
        prediction, label, top_n=1, rebalance_step=2,
        execution_dates=execution, maturity_dates=maturity,
        limit_up={(first_exec, "a")}, limit_down={(third_exec, "b")},
    )
    assert result["periods"][0]["holdings"] == []


def test_state_training_profiles_are_role_specific_and_test_independent():
    market = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=80, freq="B"),
        "trend_strength": np.linspace(-1, 1, 80),
        "oversold_degree": np.linspace(1, -1, 80),
        "market_breadth": np.linspace(0.2, 0.8, 80),
        "realized_vol_percentile": np.linspace(1, 0, 80),
    })
    period = ("2024-01-01", "2024-04-30")
    trend_dates, trend_weights, _ = _state_training_profile(market, period, "trend", "weighted")
    rebound_dates, rebound_weights, _ = _state_training_profile(market, period, "rebound", "weighted")
    assert trend_dates is None and rebound_dates is None
    assert trend_weights != rebound_weights
    subset, _, _ = _state_training_profile(market, period, "trend", "subset")
    assert 20 <= len(subset) < len(market)


def test_router_fit_respects_maturity_and_reports_calibration_metrics():
    history = []
    for i, date in enumerate(pd.date_range("2024-01-01", periods=90, freq="B")):
        row = {key: float((i + j) % 11) / 10 for j, key in enumerate(ROUTER_FEATURES)}
        history.append({**row, "date": str(date.date()), "label_matured_at": str((date + pd.offsets.BDay(2)).date()),
                        "target": int(i % 3 == 0), "advantage": 0.01 if i % 3 == 0 else -0.005})
    router = _fit_router(history, "2024-04-30")
    assert router is not None
    assert router.training_rows < len(history)
    probability = router.predict_proba({key: 0.5 for key in ROUTER_FEATURES})
    assert 0.0 < probability < 1.0
    diagnostics = _router_diagnostics([
        {"target": 0, "rebound_probability": 0.2},
        {"target": 1, "rebound_probability": 0.8},
        {"target": 0, "rebound_probability": 0.3},
        {"target": 1, "rebound_probability": 0.7},
    ])
    assert diagnostics["auc"] == pytest.approx(1.0)
    assert diagnostics["brier"] < 0.1


def test_future_vol_probability_is_fit_inside_training_window():
    dates = pd.date_range("2020-01-01", periods=220, freq="B")
    market = pd.DataFrame({
        "date": dates,
        "market_ret_1d": np.sin(np.arange(len(dates)) / 7) * 0.01,
        "realized_vol_20": np.linspace(0.01, 0.03, len(dates)),
        "realized_vol_60": np.linspace(0.015, 0.025, len(dates)),
        "realized_vol_percentile": np.linspace(0, 1, len(dates)),
        "trend_strength": np.sin(np.arange(len(dates)) / 20),
        "oversold_degree": np.cos(np.arange(len(dates)) / 20),
        "market_breadth": np.linspace(0.3, 0.7, len(dates)),
        "dispersion": np.linspace(1, 3, len(dates)),
    })
    fitted, info = _fit_future_vol_probability(market, ("2020-01-01", "2020-09-30"))
    assert info["fallback"] is False
    assert fitted["future_vol_probability"].between(0, 1).all()


def test_rank_fusion_and_hard_switch_preserve_panel():
    prediction, _ = _panel(periods=3)
    fused = _rank_fusion(prediction, prediction, 0.5)
    assert len(fused) == len(prediction)
    market = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "future_vol_probability": [0.0, 1.0, 0.0],
    })
    switched = _hard_vol_switch(prediction, prediction, market)
    assert len(switched) == len(prediction)


def test_sequence_weights_accept_two_dimensional_object_array():
    rows = np.asarray([
        (pd.Timestamp("2024-01-01"), "a"),
        (pd.Timestamp("2024-01-02"), "b"),
    ], dtype=object)
    assert rows.ndim == 2
    assert _date_weight_array(rows, None) is None
    np.testing.assert_array_equal(
        _date_weight_array(rows, {"2024-01-01": 2.0}),
        np.asarray([2.0, 1.0], dtype=np.float32),
    )


def test_purge_uses_exact_trading_dates():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=20, freq="B"), ["a"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"value": 1.0}, index=index)
    assert _purged_end_date(frame, "2024-02-01", 5) == pd.Timestamp("2024-01-19")
