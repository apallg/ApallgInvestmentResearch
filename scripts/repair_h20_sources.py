import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.h20_confirmation_v2 import OUTPUT, digest, save_json
from core.data.sources.historical_quotes import fetch_tushare_raw_daily, fetch_baostock_raw_daily, get_source_versions


def main():
    gaps = pd.read_parquet(OUTPUT / "inputs/full/2026-11-05/gap_database_audit.parquet")
    root = OUTPUT / "repair/captured_sources"
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    for code in sorted(gaps.code.unique()):
        folder = root / code
        folder.mkdir(exist_ok=True)
        exchange = "SH" if code.startswith("6") else "SZ"
        for source in ("tushare", "baostock"):
            meta = folder / f"{source}.json"
            if meta.exists():
                audit = json.loads(meta.read_text(encoding="utf-8"))
                if any(digest(folder / name) != value for name, value in audit["sha256"].items()):
                    raise RuntimeError(f"来源快照签名冲突: {code}/{source}")
                continue
            try:
                started = datetime.now(timezone.utc).isoformat()
                if source == "tushare":
                    result = fetch_tushare_raw_daily(f"{code}.{exchange}", "20180101", "20261105", allow_empty_daily=True)
                else:
                    result = fetch_baostock_raw_daily(code, exchange.lower(), "2018-01-01", "2026-11-05", allow_empty_daily=True)
                hashes = {}
                for name, frame in result.items():
                    target = folder / f"{source}_{name}.parquet"
                    temporary = target.with_suffix(".tmp")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(target)
                    hashes[target.name] = digest(target)
                save_json(meta, {"code": code, "source": source, "start": "2018-01-01", "end": "2026-11-05",
                                 "started_at_utc": started, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "versions": get_source_versions(), "sha256": hashes})
                print(f"captured {code} {source} rows={len(result['normalized'])}", flush=True)
            except Exception as error:
                failures.append({"code": code, "source": source, "error": str(error)})
                save_json(folder / f"{source}.failure.json", failures[-1])
                print(f"failed {code} {source}: {error}", flush=True)
    save_json(root / "status.json", {"status": "sources_captured" if not failures else "source_failures",
                                   "stocks": int(gaps.code.nunique()), "failures": failures})
    if failures:
        raise RuntimeError(f"来源采集失败 {len(failures)} 项，保留检查点")


if __name__ == "__main__":
    main()
