import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.horizon_multiscale_research import OUTPUT, SEEDS, calendar_index, SEQUENCE_ARMS
from scripts.analyze_horizon_classification import blocked_interval, daily_top_return


def read_prediction(fold, horizon, seed, arm):
    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
    if not path.exists() or not path.with_suffix(".json").exists():
        raise RuntimeError(f"缺少序列预测或审计文件: {path.name}")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    from scripts.run_multiscale_group_research import sha256
    from scripts.train_horizon_sequence import signature
    expected = signature(manifest, horizon, fold, seed, arm)
    if meta.get("signature") != expected:
        raise RuntimeError(f"序列预测签名不一致: {path.name}")
    frame = pd.read_parquet(path)
    if frame.index.has_duplicates or frame["score"].isna().any() or not np.isfinite(frame["score"].to_numpy()).all():
        raise RuntimeError(f"序列预测索引重复或包含非有限分数: {path.name}")
    return frame


def ensemble(fold, horizon, arm):
    frames = [read_prediction(fold, horizon, seed, arm) for seed in SEEDS]
    base = frames[0]
    for frame in frames[1:]:
        if not base.index.equals(frame.index) or not np.array_equal(base["label"].to_numpy(), frame["label"].to_numpy()):
            raise RuntimeError(f"{fold}/{horizon}/{arm}: 跨种子样本或标签不一致")
    result = pd.DataFrame({"score": np.mean([frame["score"].to_numpy() for frame in frames], axis=0),
                           "label": base["label"], "label_rank": base["label_rank"]}, index=base.index)
    if result.isna().any().any() or not np.isfinite(result["score"]).all():
        raise RuntimeError(f"{fold}/{horizon}/{arm}: 集成预测非有限")
    return result


def analyze():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    calendar = calendar_index()
    comparisons = []
    for horizon in ("H10", "H20"):
        days = int(horizon[1:])
        block_days = (20, 40, 60) if days == 10 else (40, 60, 80)
        for reference in ("NS", "NL", "NF", "NR"):
            comparison = f"NT-{reference}"
            differences = []
            folds = []
            seed_differences = {seed: [] for seed in SEEDS}
            expert_down, base_down = 0.0, 0.0
            for fold_number in range(1, 8):
                fold = f"W{fold_number}"
                candidate = ensemble(fold, horizon, "NT")
                baseline = ensemble(fold, horizon, reference)
                if not candidate.index.equals(baseline.index) or not np.array_equal(candidate["label"].to_numpy(), baseline["label"].to_numpy()):
                    raise RuntimeError(f"{fold}/{horizon}/{comparison}: 共同样本不一致")
                test_start, test_end = map(pd.Timestamp, manifest["folds"][fold]["test"])
                test_dates = calendar[(calendar >= test_start) & (calendar <= test_end)]
                formed = test_dates[np.arange(len(test_dates)) % days == 0]
                full_dates = pd.DatetimeIndex(candidate.index.get_level_values("datetime").unique()).sort_values()
                candidate5 = daily_top_return(candidate, 5, full_dates)
                baseline5 = daily_top_return(baseline, 5, full_dates)
                delta = (candidate5 - baseline5).reindex(formed)
                if delta.isna().any():
                    raise RuntimeError(f"{fold}/{horizon}/{comparison}: 预定篮子日期缺失")
                differences.append(delta.to_numpy())
                candidate50 = daily_top_return(candidate, 50, full_dates)
                baseline50 = daily_top_return(baseline, 50, full_dates)
                cdown = np.minimum(candidate5.reindex(formed).to_numpy(), 0)
                bdown = np.minimum(baseline5.reindex(formed).to_numpy(), 0)
                expert_down += float(np.square(cdown).sum())
                base_down += float(np.square(bdown).sum())
                ic_c = candidate.groupby(level="datetime").apply(lambda frame: frame["score"].corr(frame["label_rank"], method="spearman"))
                ic_b = baseline.groupby(level="datetime").apply(lambda frame: frame["score"].corr(frame["label_rank"], method="spearman"))
                seed_fold_delta = {}
                for seed in SEEDS:
                    cs = read_prediction(fold, horizon, seed, "NT")
                    bs = read_prediction(fold, horizon, seed, reference)
                    if not cs.index.equals(bs.index):
                        raise RuntimeError(f"{fold}/{horizon}/{seed}: 单种子共同样本不一致")
                    diff = daily_top_return(cs, 5, full_dates) - daily_top_return(bs, 5, full_dates)
                    selected = diff.reindex(formed)
                    seed_differences[seed].extend(selected.tolist())
                    seed_fold_delta[str(seed)] = float(selected.mean())
                folds.append({"fold": fold, "basket_dates": len(formed),
                    "top5_candidate": float(candidate5.reindex(formed).mean()),
                    "top5_baseline": float(baseline5.reindex(formed).mean()),
                    "top5_diff": float(delta.mean()),
                    "top50_diff": float((candidate50 - baseline50).reindex(formed).mean()),
                    "rank_ic_diff": float((ic_c - ic_b).mean()),
                    "seed_top5_diff": seed_fold_delta})
            all_differences = np.concatenate(differences)
            mean_delta = float(all_differences.mean())
            seed_means = {str(seed): float(np.mean(values)) for seed, values in seed_differences.items()}
            interval_blocks = {str(block): max(1, block // days) for block in block_days}
            w8_candidate = ensemble("W8", horizon, "NT")
            w8_baseline = ensemble("W8", horizon, reference)
            start, end = map(pd.Timestamp, manifest["folds"]["W8"]["test"])
            w8_dates = calendar[(calendar >= start) & (calendar <= end)]
            w8_formed = w8_dates[np.arange(len(w8_dates)) % days == 0]
            w8_full = pd.DatetimeIndex(w8_candidate.index.get_level_values("datetime").unique()).sort_values()
            w8diff = (daily_top_return(w8_candidate, 5, w8_full) - daily_top_return(w8_baseline, 5, w8_full)).reindex(w8_formed)
            ci = {key: blocked_interval(differences, size) for key, size in interval_blocks.items()}
            top5_bp = mean_delta * 10000
            semivar_ratio = expert_down / base_down if base_down else (0.0 if expert_down == 0 else None)
            rank_diff = float(np.mean([fold["rank_ic_diff"] for fold in folds]))
            positive_folds = sum(fold["top5_diff"] > 0 for fold in folds)
            positive_seeds = sum(value > 0 for value in seed_means.values())
            lower = ci[str(block_days[0])]["bonferroni_95_family20"][0]
            passes = (top5_bp >= (10 if days == 10 else 20) and positive_folds >= 5 and
                      positive_seeds >= 2 and rank_diff >= -0.005 and semivar_ratio is not None and
                      semivar_ratio <= 1.10 and lower > 0)
            comparisons.append({"horizon": horizon, "comparison": comparison, "family_size": 20,
                "folds": folds, "W8_sensitivity_diff": float(w8diff.mean()),
                "mean_nonoverlap_top5_diff": mean_delta, "top5_diff_bp": top5_bp,
                "mean_nonoverlap_top50_diff": float(np.mean([fold["top50_diff"] for fold in folds])),
                "seed_mean_top5_diff": seed_means, "positive_folds": positive_folds,
                "positive_seeds": positive_seeds, "mean_rank_ic_diff": rank_diff,
                "downside_semivariance_ratio": semivar_ratio, "basket_block_lengths": interval_blocks,
                "nonoverlap_top5_diff_intervals": ci, "stable_improvement": bool(passes)})
    result = {"status": "sequence_evaluated", "primary_folds": [f"W{i}" for i in range(1, 8)],
        "sensitivity_fold": "W8", "research_only": True, "comparisons": comparisons,
        "caveats": ["历史窗口已被研究接触", "行业及板块历史PIT未认证", "账户级交易资格未验证"]}
    target = OUTPUT / "sequence_analysis.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparisons": len(comparisons),
        "stable_improvements": sum(item["stable_improvement"] for item in comparisons), "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    analyze()
