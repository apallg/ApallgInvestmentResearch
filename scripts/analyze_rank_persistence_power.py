import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "data" / "experiments" / "rank_persistence_ab_20260925_attribution_fixed"


def paired_deltas(report):
    variants = report["variants"]
    period_maps = {}
    for name in ("A_baseline", "B_stability_first"):
        rows = variants[name]["periods"]
        period_maps[name] = {str(row["date"])[:10]: float(row["period_return"]) for row in rows}
        if len(period_maps[name]) != len(rows):
            raise ValueError(f"{name}存在重复回测期末日期")
    if set(period_maps["A_baseline"]) != set(period_maps["B_stability_first"]):
        raise ValueError("A/B成熟收益日期不完全一致，拒绝进行功效估计")
    return np.asarray([
        period_maps["B_stability_first"][day] - period_maps["A_baseline"][day]
        for day in sorted(period_maps["A_baseline"])
    ], dtype=float)


def circular_block_sample(values, sample_size, draws, block_size, rng):
    block_count = math.ceil(sample_size / block_size)
    starts = rng.integers(0, len(values), size=(draws, block_count))
    offsets = np.arange(block_size)
    indices = ((starts[:, :, None] + offsets) % len(values)).reshape(draws, -1)
    return values[indices[:, :sample_size]]


def hac_significance_power(values, sample_sizes, draws, block_size, seed):
    observed_mean = float(values.mean())
    residuals = values - observed_mean
    results = {}
    for sample_size in sample_sizes:
        rng = np.random.default_rng(seed + sample_size)
        significant = 0
        completed = 0
        while completed < draws:
            batch_size = min(1000, draws - completed)
            samples = circular_block_sample(
                residuals, sample_size, batch_size, block_size, rng,
            ) + observed_mean
            centered = samples - samples.mean(axis=1, keepdims=True)
            long_run_variance = np.sum(centered * centered, axis=1) / sample_size
            for lag in range(1, min(block_size, sample_size)):
                covariance = np.sum(centered[:, lag:] * centered[:, :-lag], axis=1) / sample_size
                weight = 1 - lag / (block_size + 1)
                long_run_variance += 2 * weight * covariance
            standard_error = np.sqrt(np.maximum(long_run_variance, 0) / sample_size)
            t_statistic = np.divide(
                samples.mean(axis=1), standard_error,
                out=np.zeros(batch_size), where=standard_error > 0,
            )
            significant += int((np.abs(t_statistic) > 1.959963984540054).sum())
            completed += batch_size
        results[str(sample_size)] = significant / draws
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.draws < 1000:
        raise ValueError("功效模拟至少需要1000次重复抽样")

    report_path = Path(args.experiment_dir) / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    deltas = paired_deltas(report)
    if len(deltas) < 8:
        raise ValueError("历史配对期数不足以估计区块依赖")
    output = {
        "status": "historical_planning_sensitivity_only",
        "source_report": str(report_path),
        "historical_paired_periods": len(deltas),
        "observed_mean_difference_pp_per_week": float(deltas.mean() * 100),
        "sample_sd_difference_pp_per_week": float(deltas.std(ddof=1) * 100),
        "lag1_autocorrelation": float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]),
        "method": {
            "alternative": "historical mean plus circularly resampled centered 4-week blocks",
            "null": "two-sided 5 percent; HAC standard error with Bartlett weights through lag 4",
            "draws": args.draws,
            "seed_base": args.seed,
            "block_size_weeks": 4,
        },
        "estimated_power": hac_significance_power(
            deltas, (52, 78, 104, 156, 208), args.draws, 4, args.seed,
        ),
        "caveat": "依赖单段历史探索效应的敏感性估算，不是正式样本量保证，也不构成前瞻策略证据。",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
