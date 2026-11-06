import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.horizon_architecture_research import ARMS, FOLDS, HORIZONS, OUTPUT, SEEDS, signature, write_json
from scripts.horizon_multiscale_research import calendar_index
from scripts.model_complementarity_research import paired_block_intervals


CONTRASTS = (("N", "G"), ("N", "GE"), ("GE", "G"),
             ("F50", "G"), ("F50", "N"), ("FC", "G"),
             ("FC", "N"), ("FC", "F50"), ("FR", "G"),
             ("FR", "N"), ("FR", "FC"))


def read_frame(fold, horizon, seed, arm, manifest):
    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(str(path))
    record = json.loads(meta.read_text(encoding="utf-8"))
    if record.get("signature") != signature(manifest, fold, horizon, seed, arm):
        raise RuntimeError(f"预测签名冲突: {path}")
    frame = pd.read_parquet(path)
    if len(frame) != record["rows"] or frame.index.has_duplicates or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError(f"预测样本或数值无效: {path}")
    return frame


def metrics(frame, formed):
    rows = []
    for date, part in frame.groupby(level="datetime", sort=True):
        if len(part) < 50:
            raise RuntimeError(f"Top50共同分母不足: {date}")
        ranked = part.sort_values("score", ascending=False)
        rows.append((date, ranked.iloc[:5]["label"].mean(), ranked.iloc[:20]["label"].mean(),
                     ranked.iloc[:50]["label"].mean(), part["label"].mean(),
                     part["score"].corr(part["label_rank"], method="spearman")))
    table = pd.DataFrame(rows, columns=["date", "top5", "top20", "top50", "pool", "rank_ic"]).set_index("date")
    if table.isna().any().any() or not pd.DatetimeIndex(formed).isin(table.index).all():
        raise RuntimeError("预定篮子日期或每日RankIC缺失")
    return {"basket": table.reindex(formed), "daily_ic": table["rank_ic"]}


def evaluate():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    required = [(fold, horizon, seed, arm) for fold in FOLDS for horizon in HORIZONS for seed in SEEDS for arm in ARMS]
    missing = [f"{fold}_{horizon}_{seed}_{arm}" for fold, horizon, seed, arm in required
               if not (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet").exists()
               or not (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.json").exists()]
    if missing:
        write_json(OUTPUT / "analysis.json", {"status": "pending", "completed": len(required) - len(missing),
                                               "required": len(required), "missing_first20": missing[:20],
                                               "comparisons": []})
        print(json.dumps({"status": "pending", "completed": len(required) - len(missing),
                          "required": len(required)}, ensure_ascii=False))
        return
    calendar = calendar_index()
    fold_metrics = {}
    for horizon in HORIZONS:
        length = int(horizon[1:])
        fold_metrics[horizon] = {}
        for fold in FOLDS:
            start, end = map(pd.Timestamp, manifest["folds"][fold]["test"])
            test_days = calendar[(calendar >= start) & (calendar <= end)]
            formed = test_days[np.arange(len(test_days)) % length == 0]
            seeds = {}
            for seed in SEEDS:
                frames = {arm: read_frame(fold, horizon, seed, arm, manifest) for arm in ARMS}
                first = frames["G"]
                for arm, frame in frames.items():
                    if not first.index.equals(frame.index) or not np.array_equal(first["label"].to_numpy(), frame["label"].to_numpy()) or not np.array_equal(first["label_rank"].to_numpy(), frame["label_rank"].to_numpy()):
                        raise RuntimeError(f"共同样本或标签不同: {fold}/{horizon}/{seed}/{arm}")
                seeds[seed] = {arm: metrics(frame, formed) for arm, frame in frames.items()}
            ensembles = {}
            for arm in ARMS:
                frames = [read_frame(fold, horizon, seed, arm, manifest) for seed in SEEDS]
                first = frames[0]
                if any(not first.index.equals(frame.index) for frame in frames[1:]):
                    raise RuntimeError(f"种子共同样本不同: {fold}/{horizon}/{arm}")
                ensemble = first.copy()
                ensemble["score"] = np.mean([frame["score"].to_numpy(dtype=float) for frame in frames], axis=0)
                ensembles[arm] = metrics(ensemble, formed)
            fold_metrics[horizon][fold] = {"seeds": seeds, "ensemble": ensembles}
    comparisons = []
    for horizon in HORIZONS:
        length = int(horizon[1:])
        blocks = (20, 40, 60) if horizon != "H20" else (40, 60, 80)
        for candidate, baseline in CONTRASTS:
            deltas, folds, seed_deltas = [], [], {seed: [] for seed in SEEDS}
            downside_candidate, downside_baseline = [], []
            rank_deltas = []
            for fold in FOLDS[:7]:
                current = fold_metrics[horizon][fold]
                c = current["ensemble"][candidate]["basket"]
                b = current["ensemble"][baseline]["basket"]
                delta = (c["top5"] - b["top5"]).to_numpy(dtype=float)
                deltas.append(delta)
                downside_candidate.extend(np.square(np.minimum(c["top5"].to_numpy(), 0)))
                downside_baseline.extend(np.square(np.minimum(b["top5"].to_numpy(), 0)))
                rank_deltas.append(float((current["ensemble"][candidate]["daily_ic"] - current["ensemble"][baseline]["daily_ic"]).mean()))
                for seed in SEEDS:
                    sc = current["seeds"][seed][candidate]["basket"]["top5"]
                    sb = current["seeds"][seed][baseline]["basket"]["top5"]
                    seed_deltas[seed].extend((sc - sb).to_numpy(dtype=float))
                folds.append({"fold": fold, "top5_diff_bp": float(delta.mean() * 10000),
                              "top20_diff_bp": float((c["top20"] - b["top20"]).mean() * 10000),
                              "top50_diff_bp": float((c["top50"] - b["top50"]).mean() * 10000),
                              "rank_ic_diff": rank_deltas[-1], "basket_count": len(delta)})
            intervals = paired_block_intervals(deltas, tuple(day // length for day in blocks), family_size=35)
            intervals = {key: {"ordinary_95": value["ordinary_95"],
                               "bonferroni_95_family35": value["bonferroni_95_family26"]}
                         for key, value in intervals.items()}
            semivar_c = float(np.mean(downside_candidate))
            semivar_b = float(np.mean(downside_baseline))
            ratio = semivar_c / semivar_b if semivar_b else (0.0 if semivar_c == 0 else None)
            mean = float(np.concatenate(deltas).mean() * 10000)
            positive_seeds = sum(np.mean(values) > 0 for values in seed_deltas.values())
            screen = mean >= {"H5": 5, "H10": 10, "H20": 20}[horizon] and sum(row["top5_diff_bp"] > 0 for row in folds) >= 5 and positive_seeds >= 2 and np.mean(rank_deltas) >= -0.005 and ratio is not None and ratio <= 1.1
            main_block = str(blocks[0] // length)
            lower = intervals[main_block]["bonferroni_95_family35"][0]
            w8 = fold_metrics[horizon]["W8"]["ensemble"]
            comparisons.append({"horizon": horizon, "comparison": f"{candidate}-{baseline}",
                "mean_top5_diff_bp": mean, "folds_positive": sum(row["top5_diff_bp"] > 0 for row in folds),
                "positive_seeds": int(positive_seeds), "rank_ic_diff": float(np.mean(rank_deltas)),
                "downside_semivariance_ratio": ratio, "screen_pass": bool(screen),
                "stable_improvement": bool(screen and lower > 0), "intervals": intervals,
                "W8_top5_diff_bp": float((w8[candidate]["basket"]["top5"] - w8[baseline]["basket"]["top5"]).mean() * 10000),
                "folds": folds})
    interactions = []
    for shorter in ("H5", "H10"):
        fold_values = []
        for fold in FOLDS[:7]:
            rows = []
            for horizon in ("H20", shorter):
                row = fold_metrics[horizon][fold]["ensemble"]
                rows.append(row["N"]["daily_ic"] - row["G"]["daily_ic"])
            paired = pd.concat(rows, axis=1, join="inner").dropna()
            if paired.empty:
                raise RuntimeError("期限交互没有共同日期")
            fold_values.append((paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy())
        intervals = paired_block_intervals(fold_values, (80,), family_size=35)
        intervals = {key: {"ordinary_95": value["ordinary_95"],
                           "bonferroni_95_family35": value["bonferroni_95_family26"]}
                     for key, value in intervals.items()}
        interactions.append({"comparison": f"(N-G)H20-(N-G){shorter}",
                             "mean_daily_rank_ic_diff": float(np.concatenate(fold_values).mean()),
                             "intervals": intervals})
    write_json(OUTPUT / "analysis.json", {"status": "evaluated", "completed": len(required),
        "required": len(required), "family_size": 35, "comparisons": comparisons, "interactions": interactions,
        "research_only": True, "strict_pit_verified": False, "execution_verified": False})
    print(json.dumps({"status": "evaluated", "comparisons": len(comparisons),
                      "interactions": len(interactions), "stable": sum(row["stable_improvement"] for row in comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("evaluate", "status"))
    args = parser.parse_args()
    if args.stage == "evaluate":
        evaluate()
    else:
        path = OUTPUT / "analysis.json"
        print(path.read_text(encoding="utf-8") if path.exists() else json.dumps({"status": "not_started"}))
