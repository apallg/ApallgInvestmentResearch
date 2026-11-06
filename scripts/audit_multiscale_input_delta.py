import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-source", default="data/experiments/fundval_repair_research_20261026")
    parser.add_argument("--new-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old_source = Path(args.old_source)
    new_source = Path(args.new_source)
    factor_keys = list(json.loads((old_source / "factors.json").read_text(encoding="utf-8")))
    columns = [*factor_keys, "label", "open", "high", "low", "close", "volume", "factor"]
    predicate = [("datetime", ">=", pd.Timestamp("2026-07-01")),
                 ("datetime", "<=", pd.Timestamp("2026-08-31"))]
    old = pd.read_parquet(old_source / "new_2025.parquet", columns=columns, filters=predicate)
    new = pd.read_parquet(new_source / "new_2025.parquet", columns=columns, filters=predicate)
    common = old.index.intersection(new.index)
    if common.empty:
        raise RuntimeError("新旧输入没有共同股票日")
    old_only = len(old.index.difference(common))
    new_only = len(new.index.difference(common))
    old = old.reindex(common)
    new = new.reindex(common)
    detail = {}
    for column in columns:
        left = old[column].to_numpy(dtype=np.float64)
        right = new[column].to_numpy(dtype=np.float64)
        both_nan = np.isnan(left) & np.isnan(right)
        close = np.isclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True)
        detail[column] = {"different_cells": int((~close).sum()),
                          "different_share": float((~close).mean()),
                          "both_missing_share": float(both_nan.mean())}
    report = {"period": ["2026-07-01", "2026-08-31"], "old_rows": len(old) + old_only, "new_rows": len(new) + new_only,
              "common_rows": len(common), "old_only_rows": old_only,
              "new_only_rows": new_only, "columns": detail,
              "research_only": True, "attribution_allowed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"common_rows": len(common), "changed_columns": sum(item["different_cells"] > 0 for item in detail.values()),
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
