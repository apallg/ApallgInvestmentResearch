import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_monthly_group_research import ARMS, OUTPUT, ROOT, SEEDS
from scripts.run_multiscale_group_research import sha256


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["source_files_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"源文件哈希不一致: {name}")
    for name, expected in manifest["code_files_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"冻结代码哈希不一致: {name}")
    for name, key in (("preheat_2018.parquet", "preheat_sha256"),
                      ("classification_snapshot.parquet", "classification_snapshot_sha256"),
                      ("stock_index_membership_snapshot.json", "membership_snapshot_sha256"),
                      ("budget_control.json", "budget_control_sha256")):
        if sha256(OUTPUT / name) != manifest[key]:
            raise RuntimeError(f"实验快照哈希不一致: {name}")
    result = {"status": "passed", "prediction_files": 0, "training_audits": 0,
              "folds": {}, "hashes_verified": True, "common_labels": True,
              "finite_scores": True, "fallback_score_matches_unified": True}
    for fold in (f"W{i}" for i in range(1, 9)):
        first_index = None
        first_label = None
        fold_rows = None
        fold_fallback = {}
        for seed in SEEDS:
            audit = OUTPUT / f"{fold}_{seed}_training.json"
            if not audit.exists():
                raise RuntimeError(f"训练审计缺失: {audit.name}")
            detail = json.loads(audit.read_text(encoding="utf-8"))
            result["training_audits"] += 1
            unified = {}
            for arm in ARMS:
                path = OUTPUT / f"{fold}_{seed}_{arm}.parquet"
                if not path.exists():
                    raise RuntimeError(f"预测缺失: {path.name}")
                frame = pd.read_parquet(path).sort_index()
                if first_index is None:
                    first_index = frame.index
                    first_label = frame["label"].to_numpy()
                    fold_rows = len(frame)
                if not frame.index.equals(first_index) or not np.allclose(frame["label"], first_label, rtol=0, atol=1e-7, equal_nan=True):
                    raise RuntimeError(f"股票日或标签不同: {path.name}")
                if not np.isfinite(frame["score"]).all() or not np.isfinite(frame["raw_score"]).all():
                    raise RuntimeError(f"预测分数非有限值: {path.name}")
                if arm in ("U0", "UM"):
                    unified[arm] = frame["score"].to_numpy()
                if arm.endswith("_0") or arm.endswith("_M"):
                    baseline = "UM" if arm.endswith("_M") else "U0"
                    fallback = frame["fallback"].to_numpy(dtype=bool)
                    if int(fallback.sum()) != detail[arm]["fallback_rows"]:
                        raise RuntimeError(f"回退计数不同: {path.name}")
                    if not np.array_equal(frame["score"].to_numpy()[fallback], unified[baseline][fallback]):
                        raise RuntimeError(f"回退分数不同: {path.name}")
                    fold_fallback.setdefault(arm, []).append(float(fallback.mean()))
                result["prediction_files"] += 1
        result["folds"][fold] = {"rows_per_prediction": fold_rows,
                                  "mean_fallback_share": {name: float(np.mean(values)) for name, values in fold_fallback.items()}}
    (OUTPUT / "output_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "prediction_files": result["prediction_files"],
                      "training_audits": result["training_audits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
