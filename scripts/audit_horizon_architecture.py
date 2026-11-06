import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.horizon_architecture_research import ARMS, FOLDS, HORIZONS, OUTPUT, SEEDS, signature, write_json


def audit(folds, horizons, seeds):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for fold in folds:
        for horizon in horizons:
            for seed in seeds:
                frames, metas = {}, {}
                for arm in ARMS:
                    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    meta_path = path.with_suffix(".json")
                    if not path.exists() or not meta_path.exists():
                        raise RuntimeError(f"缺少预测或审计: {path.name}")
                    metas[arm] = json.loads(meta_path.read_text(encoding="utf-8"))
                    if metas[arm]["signature"] != signature(manifest, fold, horizon, seed, arm):
                        raise RuntimeError(f"签名不符: {path.name}")
                    frames[arm] = pd.read_parquet(path)
                    frame = frames[arm]
                    if len(frame) != metas[arm]["rows"] or frame.index.has_duplicates or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
                        raise RuntimeError(f"预测不完整或非有限: {path.name}")
                    if frame["score"].min() < -1e-6 or frame["score"].max() > 1 + 1e-6:
                        raise RuntimeError(f"分数不是统一百分位: {path.name}")
                first = frames["G"]
                for arm, frame in frames.items():
                    if not frame.index.equals(first.index) or not np.array_equal(frame["label"].to_numpy(), first["label"].to_numpy()) or not np.array_equal(frame["label_rank"].to_numpy(), first["label_rank"].to_numpy()):
                        raise RuntimeError(f"共同分母或标签不一致: {fold}/{horizon}/{seed}/{arm}")
                g, n = frames["G"]["score"], frames["N"]["score"]
                if not np.allclose(frames["F50"]["score"], (g + n) / 2, atol=1e-7):
                    raise RuntimeError("F50分数不能从组件重建")
                weight = metas["FC"]["weight"]
                if not np.allclose(frames["FC"]["score"], (1 - weight) * g + weight * n, atol=1e-7):
                    raise RuntimeError("FC分数不能从组件和权重重建")
                fr = pd.Series(index=g.index, dtype=float)
                updates = metas["FR"]["weight_updates"]
                for row in updates:
                    effective = pd.Timestamp(row["effective_date"])
                    last_end = pd.Timestamp(row["last_label_end"]) if row["last_label_end"] != "NaT" else pd.NaT
                    if pd.notna(last_end) and last_end >= effective:
                        raise RuntimeError("FR权重使用了未成熟标签")
                    dates = fr.index.get_level_values("datetime")
                    mask = dates.to_period("M") == pd.Period(row["month"])
                    fr.iloc[np.flatnonzero(mask)] = row["weight"]
                if fr.isna().any() or not np.allclose(frames["FR"]["score"], (1 - fr) * g + fr * n, atol=1e-7):
                    raise RuntimeError("FR分数或逐月更新时间不一致")
                records.append({"fold": fold, "horizon": horizon, "seed": seed, "rows": len(first),
                    "fc_weight": weight, "fr_months": len(updates),
                    "ge_restore_error": metas["GE"]["ge_model"]["restore_max_abs_error"],
                    "n_restore_error": metas["N"]["n_model"].get("restore_max_abs_error"),
                    "same_scale_error_corr": float((g - first["label_rank"] / 3.46 - 0.5).corr(n - first["label_rank"] / 3.46 - 0.5)),
                    "mixed_scale_error_corr": float((g - first["label_rank"]).corr(n - first["label_rank"]))})
    report = {"status": "passed", "units": len(records), "predictions": len(records) * len(ARMS),
              "records": records, "research_only": True}
    write_json(OUTPUT / "base_audit.json", report)
    print(json.dumps({"status": "passed", "units": len(records), "predictions": len(records) * len(ARMS)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    audit(args.folds, args.horizons, args.seeds)
