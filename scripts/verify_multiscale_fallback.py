import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = Path(os.environ.get("MULTISCALE_OUTPUT", "data/experiments/multiscale_group_research_20261103"))
    unified = pd.read_parquet(root / f"{args.fold}_{args.seed}_U.parquet")
    report = {}
    for arm in ("G6", "GI", "GS"):
        expert = pd.read_parquet(root / f"{args.fold}_{args.seed}_{arm}.parquet")
        if not expert.index.equals(unified.index) or not np.array_equal(expert["label"].to_numpy(), unified["label"].to_numpy()):
            raise RuntimeError(f"共同股票日或标签不一致: {arm}")
        fallback = expert["fallback"].to_numpy(dtype=bool)
        if not np.array_equal(expert["score"].to_numpy()[fallback], unified["score"].to_numpy()[fallback]):
            raise RuntimeError(f"回退分数不等于统一模型校准分数: {arm}")
        report[arm] = {"test_rows": len(expert), "fallback_rows": int(fallback.sum()),
                       "fallback_share": float(fallback.mean()), "same_label": True,
                       "fallback_score_equals_unified": True}
    output = root / f"{args.fold}_{args.seed}_fallback_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
