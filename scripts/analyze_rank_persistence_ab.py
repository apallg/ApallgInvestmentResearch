import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "data" / "experiments" / "rank_persistence_ab_20260925_attribution_fixed"
VARIANTS = ("A_baseline", "B_stability_first", "C_strict_confirm")


def summarize_delta(values, block_size=4, draws=10000, seed=20260925):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < block_size or not np.isfinite(values).all():
        raise ValueError("配对收益差必须是足够长度的一维有限数组")
    rng = np.random.default_rng(seed)
    block_count = math.ceil(len(values) / block_size)
    starts = rng.integers(0, len(values), size=(draws, block_count))
    offsets = np.arange(block_size)
    indices = ((starts[:, :, None] + offsets) % len(values)).reshape(draws, -1)[:, :len(values)]
    means = values[indices].mean(axis=1)
    return {
        "n_periods": int(len(values)),
        "wins": int((values > 0).sum()),
        "losses": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
        "mean_difference_pp_per_week": float(values.mean() * 100),
        "median_difference_pp_per_week": float(np.median(values) * 100),
        "bootstrap_ci95_pp_per_week": [float(value * 100) for value in np.quantile(means, [0.025, 0.975])],
        "bootstrap": {
            "method": "circular_moving_block",
            "block_size_weeks": block_size,
            "replicates": draws,
            "seed": seed,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    report = json.loads((experiment_dir / "report.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(experiment_dir / "selected_stock_audit.csv", dtype={"code": str})
    period_maps = {}
    for variant in VARIANTS:
        rows = report["variants"][variant]["periods"]
        period_maps[variant] = {str(row["date"])[:10]: float(row["period_return"]) for row in rows}
        if len(period_maps[variant]) != len(rows):
            raise ValueError(f"{variant}存在重复的回测期末日期")

    common_dates = sorted(set.intersection(*(set(period_maps[name]) for name in VARIANTS)))
    if any(set(period_maps[name]) != set(common_dates) for name in VARIANTS):
        raise ValueError("A/B/C回测期末日期不完全一致，拒绝非配对比较")
    paired = {}
    for variant in VARIANTS[1:]:
        delta = np.asarray([
            period_maps[variant][day] - period_maps["A_baseline"][day]
            for day in common_dates
        ], dtype=float)
        paired[f"{variant}_minus_A_baseline"] = summarize_delta(delta)

    realized = audit[audit["period_return"].notna()].copy()
    decision_sets = {}
    coverage = {}
    for variant in VARIANTS:
        frame = realized[realized["variant"] == variant]
        decision_sets[variant] = {
            (str(row.decision_date), int(row.slot)): str(row.code)
            for row in frame.itertuples()
        }
        coverage[variant] = {
            "all_audited_positions": int((audit["variant"] == variant).sum()),
            "realized_positions": int(len(frame)),
            "distinct_realized_decisions": int(frame["decision_date"].nunique()),
            "persistent_positions": int(frame["confirmed"].astype(bool).sum()),
            "persistent_share": float(frame["confirmed"].astype(bool).mean()),
        }
    if any(coverage[name]["realized_positions"] != len(common_dates) * 5 for name in VARIANTS):
        raise ValueError("持仓审计的已实现位置数与配对回测期数×Top5不一致")

    result = {
        "status": "historical_exploratory_paired_analysis",
        "interpretation": "历史样本用于探索而非独立OOS晋级；前瞻实验仍需另行完成。",
        "period_return_alignment": report.get("period_return_alignment"),
        "period_end_dates": common_dates,
        "paired_returns": paired,
        "realized_selection_coverage": coverage,
        "B_equals_C_on_realized_positions": decision_sets["B_stability_first"] == decision_sets["C_strict_confirm"],
    }
    if not result["B_equals_C_on_realized_positions"]:
        raise ValueError("B/C在已实现持仓位上不一致，需审计策略输出")
    output_path = experiment_dir / "paired_analysis.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "paired_periods": len(common_dates),
        "paired_returns": paired,
        "realized_selection_coverage": coverage,
        "B_equals_C_on_realized_positions": result["B_equals_C_on_realized_positions"],
        "output": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
