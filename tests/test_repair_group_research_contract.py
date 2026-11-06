import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
@pytest.mark.parametrize("window", [1, 2, 9, 15])
def test_research_rank_preserves_original_ties_missing_values_and_warmup(window):
    from scripts.research_feature_ops import rolling_last_rank
    values = pd.Series(np.random.default_rng(42).integers(-2, 3, size=300).astype(float))
    values.iloc[:20] = np.nan
    values.iloc[::7] = np.nan
    values.iloc[::11] = np.inf
    expected = values.rolling(window, min_periods=1).apply(lambda part: (part.iloc[-1] >= part.iloc[:-1]).mean(), raw=False)
    pd.testing.assert_series_equal(rolling_last_rank(values, window), expected)


@pytest.mark.unit
def test_research_correlation_aligns_truncated_series_without_filling_future_prices():
    from scripts.research_feature_ops import aligned_correlation
    left = pd.Series([1., 2., 4., 3., 5.], index=np.arange(3, 8))
    right = pd.Series([1., 3., 2., 4., 6., 5., 7., 8.], index=np.arange(8))
    expected = left.rolling(3, min_periods=1).corr(right)
    pd.testing.assert_series_equal(aligned_correlation(left, right, 3), expected)
    assert left.index[0] == 3
    equal = aligned_correlation(right, right * 2, 3)
    np.testing.assert_allclose(equal.iloc[1:], 1.0)


@pytest.mark.unit
def test_research_calibration_is_monotone_and_ignores_labels_outside_calibration():
    source = Path("scripts/run_repair_group_research.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "calibrate")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "calibration", "exec"), namespace)
    dates = pd.bdate_range("2024-01-01", periods=60)
    index = pd.MultiIndex.from_product([dates, ["sh600000", "sh600001"]], names=["datetime", "instrument"])
    scores = pd.Series(np.tile([0.0, 1.0], len(dates)), index=index)
    group = pd.Series(0, index=index)
    label = 0.1 * scores - 0.02
    result, _ = namespace["calibrate"](scores, label, group, scores, group)
    np.testing.assert_allclose(result, label)
    inverse, _ = namespace["calibrate"](scores, -label, group, scores, group)
    assert (inverse.to_numpy()[1::2] > inverse.to_numpy()[::2]).all()
    future_changed = label.copy()
    future_changed.loc[pd.IndexSlice[dates[40]:, :]] = 1000
    original, _ = namespace["calibrate"](scores.iloc[:80], label, group, scores, group)
    changed, _ = namespace["calibrate"](scores.iloc[:80], future_changed, group, scores, group)
    pd.testing.assert_series_equal(original, changed)


@pytest.mark.unit
@pytest.mark.parametrize("corrupt_label", [False, True])
def test_research_analysis_pairs_labels_and_keeps_holding_blocks_inside_fold(tmp_path, corrupt_label):
    out = tmp_path / "data/experiments/repair_group_research_20261025"
    out.mkdir(parents=True)
    calendar = pd.bdate_range("2023-01-02", "2023-04-28")
    calendar_path = tmp_path / "data/qlib_data/calendars/day.txt"
    calendar_path.parent.mkdir(parents=True)
    calendar_path.write_text("\n".join(calendar.strftime("%Y-%m-%d")), encoding="utf-8")
    manifest = {"folds": {"W1": {"test": ["2023-01-02", "2023-03-31"]}}, "seeds": [42, 2026, 3407], "limitations": []}
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (out / "input_audit.json").write_text("{}", encoding="utf-8")
    (out / "training_complete.json").write_text(json.dumps({"folds": ["W1"], "seeds": manifest["seeds"]}), encoding="utf-8")
    dates = calendar[calendar <= "2023-03-31"]
    codes = [f"sh{600000 + code}" for code in range(60)]
    index = pd.MultiIndex.from_product([dates, codes], names=["datetime", "instrument"])
    value = np.tile(np.linspace(-0.03, 0.03, len(codes)), len(dates))
    for seed in manifest["seeds"]:
        for arm in ("A0", "A1", "B", "C"):
            score = value if arm in ("A1", "B", "C") else -value
            frame = pd.DataFrame({"score": score, "label": value, "group_id": 0, "fallback": False}, index=index)
            if corrupt_label and seed == 42 and arm == "C":
                frame.iloc[0, frame.columns.get_loc("label")] += 0.01
            frame.to_parquet(out / f"W1_{seed}_{arm}.parquet")
    source = Path("scripts/analyze_repair_group_research.py").read_text(encoding="utf-8")
    namespace = {"__file__": str(tmp_path / "scripts/analyze_repair_group_research.py"), "__name__": "__main__"}
    if corrupt_label:
        with pytest.raises(RuntimeError, match="标签"):
            exec(compile(source, namespace["__file__"], "exec"), namespace)
        return
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    effect = decision["effects"]["A1_minus_A0"]
    assert effect["mean_return_difference"] > 0.05
    assert effect["candidate_beats_baseline_share"] == 1.0
    assert decision["effects"]["C_minus_B"]["tie_share"] == 1.0
    assert decision["deployment_eligible"] is False
    baskets = pd.read_csv(out / "paired_blocks.csv")
    positions = calendar.get_indexer(pd.to_datetime(baskets["date"]))
    assert (np.diff(positions) >= 10).all()
    assert (calendar[positions + 11] <= pd.Timestamp("2023-03-31")).all()
