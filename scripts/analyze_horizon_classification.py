import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.horizon_multiscale_research import OUTPUT, SEEDS, calendar_index


def load_ensemble(fold, horizon, arm):
    frames = [pd.read_parquet(OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet") for seed in SEEDS]
    base = frames[0]
    for other in frames[1:]:
        if not base.index.equals(other.index) or not np.array_equal(base["label"].to_numpy(), other["label"].to_numpy()):
            raise RuntimeError(f"{fold}/{horizon}/{arm}: 种子预测样本或标签不一致")
    score = pd.concat([frame["score"].rename(str(seed)) for seed, frame in zip(SEEDS, frames)], axis=1).mean(axis=1)
    if not np.isfinite(score.to_numpy()).all() or score.isna().any():
        raise RuntimeError(f"{fold}/{horizon}/{arm}: 存在非有限预测")
    return pd.DataFrame({"score": score, "label": base["label"], "label_rank": base["label_rank"]})


def daily_top_return(frame, basket_size, dates):
    result = {}
    for date, part in frame.groupby(level="datetime", sort=True):
        part = part.dropna(subset=["score", "label"])
        if len(part) < basket_size:
            raise RuntimeError(f"{date.date()}: 共同分母不足 Top{basket_size}")
        result[date] = float(part.nlargest(basket_size, "score")["label"].mean())
    values = pd.Series(result, dtype=float).reindex(dates)
    return values


def blocked_interval(fold_values, block_length, seed=20261104, repetitions=10000):
    rng = np.random.default_rng(seed)
    samples = np.zeros(repetitions, dtype=np.float64)
    total_count = 0
    for values in fold_values:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        if not n:
            continue
        total_count += n
        blocks = int(np.ceil(n / block_length))
        starts = rng.integers(0, n, size=(repetitions, blocks))
        offsets = np.arange(block_length)
        draws = values[(starts[:, :, None] + offsets[None, None, :]) % n]
        samples += draws.reshape(repetitions, -1)[:, :n].sum(axis=1)
    if total_count == 0:
        return None
    samples /= total_count
    return {
        "ordinary_95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bonferroni_95_family20": [float(np.quantile(samples, 0.05 / 20 / 2)),
                                   float(np.quantile(samples, 1 - 0.05 / 20 / 2))],
    }


def analyze():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    folds = [f"W{i}" for i in range(1, 8)]
    calendar = calendar_index()
    comparisons = []
    for horizon in ("H10", "H20"):
        length = int(horizon[1:])
        blocks = (20, 40, 60) if horizon == "H10" else (40, 60, 80)
        for model_group in ("B3", "A2", "R4"):
            for input_name in ("D", "M"):
                model = f"{model_group}_{input_name}"
                baseline = f"U_{input_name}"
                fold_basket_diffs = []
                fold_results = []
                rank_ic_diffs = []
                downside_expert_sum = 0.0
                downside_unified_sum = 0.0
                downside_count = 0
                seed_basket_diffs = {seed: [] for seed in SEEDS}
                for fold in folds:
                    expert = load_ensemble(fold, horizon, model)
                    unified = load_ensemble(fold, horizon, baseline)
                    if not expert.index.equals(unified.index) or not np.array_equal(expert["label"].to_numpy(), unified["label"].to_numpy()):
                        raise RuntimeError(f"{fold}/{horizon}/{model}: 与统一模型共同样本不一致")
                    test_start = pd.Timestamp(manifest["folds"][fold]["test"][0])
                    test_end = pd.Timestamp(manifest["folds"][fold]["test"][1])
                    basket_dates = calendar[(calendar >= test_start) & (calendar <= test_end)]
                    full_dates = pd.DatetimeIndex(expert.index.get_level_values("datetime").unique()).sort_values()
                    formed_dates = basket_dates[np.arange(len(basket_dates)) % length == 0]
                    ex_daily = daily_top_return(expert, 5, full_dates)
                    un_daily = daily_top_return(unified, 5, full_dates)
                    daily_diff = ex_daily - un_daily
                    dates_set = set(formed_dates)
                    paired_basket = daily_diff.loc[daily_diff.index.isin(dates_set)].to_numpy()
                    fold_basket_diffs.append(paired_basket)
                    for seed in SEEDS:
                        seeded_expert = pd.read_parquet(OUTPUT / f"{fold}_{horizon}_{seed}_{model}.parquet")
                        seeded_unified = pd.read_parquet(OUTPUT / f"{fold}_{horizon}_{seed}_{baseline}.parquet")
                        if not seeded_expert.index.equals(seeded_unified.index):
                            raise RuntimeError(f"{fold}/{horizon}/{seed}: 种子模型共同样本不一致")
                        seeded_expert_return = daily_top_return(seeded_expert.rename(columns={"score": "score"}), 5, full_dates)
                        seeded_unified_return = daily_top_return(seeded_unified.rename(columns={"score": "score"}), 5, full_dates)
                        seed_basket_diffs[seed].extend((seeded_expert_return - seeded_unified_return).reindex(formed_dates).dropna().tolist())
                    ex_top50 = daily_top_return(expert, 50, full_dates)
                    un_top50 = daily_top_return(unified, 50, full_dates)
                    top50_diff = (ex_top50 - un_top50).reindex(formed_dates)
                    downside_expert = float(np.mean(np.minimum(ex_daily.reindex(formed_dates).to_numpy(), 0) ** 2))
                    downside_unified = float(np.mean(np.minimum(un_daily.reindex(formed_dates).to_numpy(), 0) ** 2))
                    rank_ic = expert.groupby(level="datetime").apply(lambda part: part["score"].corr(part["label_rank"], method="spearman"))
                    unified_rank_ic = unified.groupby(level="datetime").apply(lambda part: part["score"].corr(part["label_rank"], method="spearman"))
                    rank_ic_diff = float((rank_ic - unified_rank_ic).mean())
                    rank_ic_diffs.append(rank_ic_diff)
                    downside_expert_sum += float(np.minimum(ex_daily.reindex(formed_dates).to_numpy(), 0).dot(np.minimum(ex_daily.reindex(formed_dates).to_numpy(), 0)))
                    downside_unified_sum += float(np.minimum(un_daily.reindex(formed_dates).to_numpy(), 0).dot(np.minimum(un_daily.reindex(formed_dates).to_numpy(), 0)))
                    downside_count += len(formed_dates)
                    fold_results.append({"fold": fold, "basket_dates": len(formed_dates),
                        "expert_nonoverlap_top5": float(ex_daily.reindex(formed_dates).mean()),
                        "unified_nonoverlap_top5": float(un_daily.reindex(formed_dates).mean()),
                        "nonoverlap_top5_diff": float((ex_daily - un_daily).reindex(formed_dates).mean()),
                        "nonoverlap_top50_diff": float(top50_diff.mean()),
                        "downside_semivariance_ratio": downside_expert / downside_unified if downside_unified > 0 else None,
                        "expert_daily_rank_ic": float(rank_ic.mean()),
                        "rank_ic_diff": rank_ic_diff,
                        "positive_daily_dates": int((daily_diff > 0).sum()), "daily_dates": int(daily_diff.notna().sum())})
                sensitivity = {}
                fold = "W8"
                expert = load_ensemble(fold, horizon, model)
                unified = load_ensemble(fold, horizon, baseline)
                if not expert.index.equals(unified.index) or not np.array_equal(expert["label"].to_numpy(), unified["label"].to_numpy()):
                    raise RuntimeError(f"W8/{horizon}/{model}: 与统一模型共同样本不一致")
                test_start = pd.Timestamp(manifest["folds"][fold]["test"][0])
                test_end = pd.Timestamp(manifest["folds"][fold]["test"][1])
                basket_dates = calendar[(calendar >= test_start) & (calendar <= test_end)]
                full_dates = pd.DatetimeIndex(expert.index.get_level_values("datetime").unique()).sort_values()
                formed_dates = basket_dates[np.arange(len(basket_dates)) % length == 0]
                ex_daily = daily_top_return(expert, 5, full_dates)
                un_daily = daily_top_return(unified, 5, full_dates)
                sensitivity = {"fold": "W8", "basket_dates": len(formed_dates),
                    "expert_nonoverlap_top5": float(ex_daily.reindex(formed_dates).mean()),
                    "unified_nonoverlap_top5": float(un_daily.reindex(formed_dates).mean()),
                    "nonoverlap_top5_diff": float((ex_daily - un_daily).reindex(formed_dates).mean()),
                    "not_counted_in_primary_fold_gate": True}
                all_basket = np.concatenate(fold_basket_diffs) if fold_basket_diffs else np.array([])
                basket_block_lengths = {str(block): max(1, block // length) for block in blocks}
                seed_means = {str(seed): float(np.mean(values)) if values else None for seed, values in seed_basket_diffs.items()}
                positive_seeds = sum(value is not None and value > 0 for value in seed_means.values())
                semivariance_ratio = downside_expert_sum / downside_unified_sum if downside_unified_sum > 0 else (0.0 if downside_expert_sum == 0 else None)
                item = {"horizon": horizon, "comparison": f"{model}-{baseline}", "family_size": 20,
                        "folds": fold_results, "W8_sensitivity": sensitivity,
                        "mean_nonoverlap_top5_diff": float(np.nanmean(all_basket)) if len(all_basket) else None,
                        "nonoverlap_top5_diff_bp": float(np.nanmean(all_basket) * 10000) if len(all_basket) else None,
                        "mean_nonoverlap_top50_diff": float(np.nanmean([row["nonoverlap_top50_diff"] for row in fold_results])),
                        "seed_mean_nonoverlap_top5_diff": seed_means, "positive_seeds": positive_seeds,
                        "mean_downside_semivariance_ratio": semivariance_ratio,
                        "mean_rank_ic_diff": float(np.mean(rank_ic_diffs)) if rank_ic_diffs else None,
                        "folds_positive": int(sum(value["nonoverlap_top5_diff"] > 0 for value in fold_results)),
                        "basket_block_lengths": basket_block_lengths,
                        "nonoverlap_top5_diff_block_intervals": {str(block): blocked_interval(fold_basket_diffs, basket_block_lengths[str(block)]) for block in blocks},
                        "interval_note": "Bonferroni 20-comparison percentile intervals; paired non-overlapping basket differences are resampled in circular blocks, with trading-day block lengths converted to basket counts"}
                item["screen_pass"] = bool(item["nonoverlap_top5_diff_bp"] is not None and
                    item["nonoverlap_top5_diff_bp"] >= (10 if horizon == "H10" else 20) and
                    item["folds_positive"] >= 5 and positive_seeds >= 2 and
                    item["mean_downside_semivariance_ratio"] is not None and item["mean_downside_semivariance_ratio"] <= 1.10 and
                    item["mean_rank_ic_diff"] is not None and item["mean_rank_ic_diff"] >= -0.005)
                comparisons.append(item)
    result = {"status": "classification_evaluated", "primary_folds": folds, "sensitivity_fold": "W8",
        "research_only": True, "strict_pit_verified": False, "comparisons": comparisons,
        "caveats": ["R4 depends on unverified historical industry publication vintages and is exploratory.",
                    "B3 historical effective dates are not independently authenticated.",
                    "No transaction-cost or execution-level qualification is included."]}
    path = OUTPUT / "classification_analysis.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparisons": len(comparisons), "passed_screens": sum(x["screen_pass"] for x in comparisons), "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    analyze()
