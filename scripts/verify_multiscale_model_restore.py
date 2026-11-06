import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.run_multiscale_group_research import OUTPUT
from scripts.train_multiscale_group_research import load_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    args = parser.parse_args()
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    arms, _, _, label_raw, _, eligible, _ = load_fold(manifest, args.fold)
    dates = pd.DatetimeIndex(label_raw.index.get_level_values("datetime"))
    test_start, test_end = manifest["folds"][args.fold]["test"]
    mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end)) & eligible
    result = {}
    for arm in ("D", "L", "W", "M", "U"):
        model = lgb.Booster(model_str=(OUTPUT / f"{args.fold}_{args.seed}_{arm}.txt").read_text(encoding="utf-8"))
        with (OUTPUT / f"{args.fold}_{args.seed}_{arm}.checkpoint.pkl").open("rb") as handle:
            import pickle
            checkpoint = pickle.load(handle)
        iteration = int(checkpoint["details"]["best_iteration"])
        x = arms[arm].loc[mask].iloc[:args.rows]
        saved = checkpoint["scores"]["test"].reindex(x.index).to_numpy()
        restored = model.predict(x.to_numpy(), num_threads=4, num_iteration=iteration)
        chunked = np.concatenate([model.predict(x.iloc[start:start + 512].to_numpy(), num_threads=4,
                                                num_iteration=iteration) for start in range(0, len(x), 512)])
        restore_error = float(np.max(np.abs(saved - restored)))
        chunk_error = float(np.max(np.abs(restored - chunked)))
        if restore_error > 1e-6 or chunk_error > 1e-6:
            raise RuntimeError(f"模型恢复或分片预测不一致: {arm} {restore_error} {chunk_error}")
        result[arm] = {"rows": len(x), "max_restore_error": restore_error,
                       "max_chunk_error": chunk_error}
    output = OUTPUT / f"{args.fold}_{args.seed}_restore_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output), "rows_per_arm": args.rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
