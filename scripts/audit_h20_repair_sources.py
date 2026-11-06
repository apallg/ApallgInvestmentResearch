import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts.h20_confirmation_v2 import OUTPUT, digest, save_json


def main():
    root = OUTPUT / "repair/captured_sources"
    gaps = pd.read_parquet(OUTPUT / "inputs/full/2026-11-05/gap_database_audit.parquet")
    rows = []
    nontrading = []
    for code in sorted(gaps.code.unique()):
        folder = root / code
        if not all((folder / f"{name}.json").exists() for name in ("tushare", "baostock")):
            rows.append({"code": code, "status": "pending_capture"})
            continue
        for source in ("tushare", "baostock"):
            meta = json.loads((folder / f"{source}.json").read_text(encoding="utf-8"))
            if any(digest(folder / name) != sha for name, sha in meta["sha256"].items()):
                raise RuntimeError(f"快照签名冲突: {code}/{source}")
        ts = pd.read_parquet(folder / "tushare_normalized.parquet").set_index("trade_date")
        bs = pd.read_parquet(folder / "baostock_normalized.parquet").set_index("date")
        ts = ts.loc[ts.vol.gt(0)]
        traded = bs.loc[bs.tradestatus.eq(1) & bs.volume.gt(0)]
        common = ts.index.intersection(traded.index)
        mismatched = np.zeros(len(common), dtype=bool)
        for field in ("open", "high", "low", "close"):
            mismatched |= ~np.isclose(ts.loc[common, field], traded.loc[common, field], atol=0.011, rtol=0.0001)
        mismatched |= ~np.isclose(ts.loc[common, "vol"] * 100, traded.loc[common, "volume"], atol=1, rtol=0.00001)
        source_missing = ts.index.symmetric_difference(traded.index)
        expected = pd.DatetimeIndex(gaps.loc[gaps.code.eq(code), "datetime"])
        uncovered = expected.difference(ts.index)
        unverified = []
        for day in uncovered:
            if day in bs.index and bs.loc[day, "tradestatus"] == 0 and not bs.loc[day, "volume"] > 0:
                nontrading.append({"code": code, "date": str(day.date()),
                                   "baostock_sha256": digest(folder / "baostock.json"),
                                   "tushare_sha256": digest(folder / "tushare.json")})
            else:
                unverified.append(str(day.date()))
        status = "verified" if not mismatched.any() and not len(source_missing) and not unverified else "blocked_source_disagreement"
        rows.append({"code": code, "status": status, "common_rows": len(common),
                     "price_or_volume_mismatch_dates": [str(x.date()) for x in common[mismatched]],
                     "source_missing_dates": [str(x.date()) for x in source_missing],
                     "unverified_gap_dates": unverified, "expected_gaps": len(expected)})
    complete = all(row["status"] == "verified" for row in rows)
    save_json(OUTPUT / "repair/source_audit.json", {"status": "verified" if complete else "pending_or_blocked",
              "stocks": rows, "verified_nontrading": nontrading, "research_only": True})
    print(json.dumps({"verified": sum(x["status"] == "verified" for x in rows),
                      "pending": sum(x["status"] == "pending_capture" for x in rows),
                      "blocked": [x["code"] for x in rows if x["status"] == "blocked_source_disagreement"],
                      "nontrading": len(nontrading)}))


if __name__ == "__main__":
    main()
