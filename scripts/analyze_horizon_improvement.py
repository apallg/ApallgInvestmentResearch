import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_horizon_architecture import metrics
from scripts.horizon_architecture_improvement import ARMS, signature
from scripts.horizon_architecture_research import FOLDS, HORIZONS, OUTPUT, SEEDS, write_json
from scripts.horizon_multiscale_research import calendar_index
from scripts.model_complementarity_research import paired_block_intervals


CONTRASTS = (("AX", "A0"), ("R", "AX"), ("R", "F"), ("R", "E"))


def read_frame(fold, horizon, seed, arm, budget):
    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}_improve.parquet"
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(str(path))
    record = json.loads(meta.read_text(encoding="utf-8"))
    if record.get("signature") != signature(fold, horizon, seed, arm, budget):
        raise RuntimeError(f"改进预测签名冲突: {path.name}")
    frame = pd.read_parquet(path)
    if len(frame) != record["rows"] or frame.index.has_duplicates or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError(f"改进预测无效: {path.name}")
    return frame


def evaluate():
    budget_path = OUTPUT / "improvement_budget.json"
    if not budget_path.exists():
        raise RuntimeError("等预算配置未冻结")
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    required = [(fold, horizon, seed, arm) for fold in FOLDS for horizon in HORIZONS for seed in SEEDS for arm in ARMS]
    missing = [f"{fold}_{horizon}_{seed}_{arm}" for fold, horizon, seed, arm in required
               if not (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}_improve.parquet").exists()
               or not (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}_improve.json").exists()]
    if missing:
        result = {"status": "pending", "completed": len(required) - len(missing), "required": len(required),
                  "missing_first20": missing[:20], "comparisons": []}
        write_json(OUTPUT / "improvement_analysis.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return
    calendar = calendar_index()
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    folds = {}
    for horizon in HORIZONS:
        folds[horizon] = {}
        length = int(horizon[1:])
        for fold in FOLDS:
            start, end = map(pd.Timestamp, manifest["folds"][fold]["test"])
            test_days = calendar[(calendar >= start) & (calendar <= end)]
            formed = test_days[np.arange(len(test_days)) % length == 0]
            seed_metrics = {}
            for seed in SEEDS:
                frames = {arm: read_frame(fold, horizon, seed, arm, budget) for arm in ARMS}
                first = frames["A0"]
                if any(not first.index.equals(frame.index) or not np.array_equal(first["label"].to_numpy(), frame["label"].to_numpy()) for frame in frames.values()):
                    raise RuntimeError(f"改进共同分母或标签不一致: {fold}/{horizon}/{seed}")
                seed_metrics[seed] = {arm: metrics(frame, formed) for arm, frame in frames.items()}
            ensemble_metrics = {}
            for arm in ARMS:
                frames = [read_frame(fold, horizon, seed, arm, budget) for seed in SEEDS]
                first = frames[0]
                if any(not first.index.equals(frame.index) for frame in frames[1:]):
                    raise RuntimeError(f"改进种子分母不一致: {fold}/{horizon}/{arm}")
                ensemble = first.copy()
                ensemble["score"] = np.mean([frame["score"].to_numpy(dtype=float) for frame in frames], axis=0)
                ensemble_metrics[arm] = metrics(ensemble, formed)
            folds[horizon][fold] = {"seeds": seed_metrics, "ensemble": ensemble_metrics}
    comparisons = []
    for horizon in HORIZONS:
        length = int(horizon[1:])
        blocks = (20, 40, 60) if horizon != "H20" else (40, 60, 80)
        for candidate, baseline in CONTRASTS:
            differences, fold_rows, seed_values = [], [], {seed: [] for seed in SEEDS}
            down_c, down_b, ic = [], [], []
            for fold in FOLDS[:7]:
                row = folds[horizon][fold]
                c = row["ensemble"][candidate]["basket"]
                b = row["ensemble"][baseline]["basket"]
                diff = (c["top5"] - b["top5"]).to_numpy(dtype=float)
                differences.append(diff)
                down_c.extend(np.square(np.minimum(c["top5"].to_numpy(), 0)))
                down_b.extend(np.square(np.minimum(b["top5"].to_numpy(), 0)))
                ic.append(float((row["ensemble"][candidate]["daily_ic"] - row["ensemble"][baseline]["daily_ic"]).mean()))
                for seed in SEEDS:
                    sc = row["seeds"][seed][candidate]["basket"]["top5"]
                    sb = row["seeds"][seed][baseline]["basket"]["top5"]
                    seed_values[seed].extend((sc - sb).to_numpy(dtype=float))
                fold_rows.append({"fold": fold, "top5_diff_bp": float(diff.mean() * 10000),
                                  "top20_diff_bp": float((c["top20"] - b["top20"]).mean() * 10000),
                                  "top50_diff_bp": float((c["top50"] - b["top50"]).mean() * 10000),
                                  "basket_count": len(diff)})
            intervals = paired_block_intervals(differences, tuple(day // length for day in blocks), family_size=12)
            intervals = {key: {"ordinary_95": value["ordinary_95"],
                               "bonferroni_95_family12": value["bonferroni_95_family26"]}
                         for key, value in intervals.items()}
            semivar_c, semivar_b = float(np.mean(down_c)), float(np.mean(down_b))
            ratio = semivar_c / semivar_b if semivar_b else (0.0 if semivar_c == 0 else None)
            mean = float(np.concatenate(differences).mean() * 10000)
            positive = sum(row["top5_diff_bp"] > 0 for row in fold_rows)
            seeds_positive = sum(np.mean(values) > 0 for values in seed_values.values())
            screen = mean >= {"H5": 5, "H10": 10, "H20": 20}[horizon] and positive >= 5 and seeds_positive >= 2 and np.mean(ic) >= -0.005 and ratio is not None and ratio <= 1.1
            lower = intervals[str(blocks[0] // length)]["bonferroni_95_family12"][0]
            w8 = folds[horizon]["W8"]["ensemble"]
            comparisons.append({"horizon": horizon, "comparison": f"{candidate}-{baseline}",
                "mean_top5_diff_bp": mean, "folds_positive": positive, "positive_seeds": int(seeds_positive),
                "rank_ic_diff": float(np.mean(ic)), "downside_semivariance_ratio": ratio,
                "screen_pass": bool(screen), "stable_improvement": bool(screen and lower > 0),
                "W8_top5_diff_bp": float((w8[candidate]["basket"]["top5"] - w8[baseline]["basket"]["top5"]).mean() * 10000),
                "intervals": intervals, "folds": fold_rows})
    result = {"status": "evaluated", "completed": len(required), "required": len(required),
              "comparisons": comparisons, "family_size": 12, "research_only": True,
              "strict_pit_verified": False, "execution_verified": False}
    write_json(OUTPUT / "improvement_analysis.json", result)
    print(json.dumps({"status": "evaluated", "comparisons": len(comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    evaluate()
