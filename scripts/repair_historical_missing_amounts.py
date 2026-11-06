import argparse
import sys
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.service.historical_repair import (
    _run_shadow_batch,
    get_shadow_batch,
    prepare_shadow_batch,
    prepare_shadow_release,
    publish_historical_release,
)
from core.data.connection import Database
from core.shared.persistence import load_json, load_parquet, now_iso, save_json_atomic


parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2023)
parser.add_argument("--denominator", default="468652126599")
parser.add_argument("--anchor-date", default="2026-10-23")
parser.add_argument("--batch-size", type=int, default=60)
args = parser.parse_args()

if args.batch_size < 1 or args.batch_size > 60:
    raise ValueError("batch-size 必须在 1 到 60 之间")

denominator_root = Path("data/historical_repair/denominator") / args.denominator
denominator = load_json(denominator_root / "manifest.json")
if not denominator or denominator.get("denominator_verified") is not True:
    raise RuntimeError("G0 分母不存在或未认证")

source_rows = Database.fetchall(
    "SELECT code,trade_date,volume,amount FROM daily_quotes WHERE YEAR(trade_date)=%s AND (volume IS NULL OR amount IS NULL) ORDER BY trade_date,code",
    (args.year,),
)
missing = pd.DataFrame(source_rows)
if missing.empty:
    print({"year": args.year, "candidates": 0, "published": 0})
    raise SystemExit(0)
missing["code"] = missing["code"].astype(str).str.zfill(6)
missing["trade_date"] = pd.to_datetime(missing["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
expected = load_parquet(denominator_root / f"expected_keys_{args.year}.parquet")
expected["code"] = expected["code"].astype(str).str.zfill(6)
expected["trade_date"] = pd.to_datetime(expected["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
expected = expected.loc[
    ~expected["full_day_suspended"].fillna(False).astype(bool), ["code", "trade_date"],
]
eligible = missing.merge(expected, on=["code", "trade_date"], how="inner", validate="one_to_one")
eligible["repair_fields"] = eligible.apply(
    lambda row: tuple(field for field in ("volume", "amount") if pd.isna(row[field])), axis=1,
)
eligible = eligible.loc[eligible["repair_fields"].map(bool)].copy()
run_id = uuid.uuid4().hex[:12]
run_dir = Path("data/historical_repair/amount_repair_runs")
run_dir.mkdir(parents=True, exist_ok=True)
run_path = run_dir / f"{run_id}.json"
run = {
    "run_id": run_id,
    "created_at": now_iso(),
    "year": args.year,
    "denominator_batch_id": args.denominator,
    "anchor_date": args.anchor_date,
    "candidate_rows": len(eligible),
    "published_rows": 0,
    "published_releases": [],
    "blocked": [],
    "batch_results": [],
    "status": "running",
}
save_json_atomic(run, run_path)

for (trade_date, repair_fields), date_rows in eligible.groupby(["trade_date", "repair_fields"], sort=True):
    codes = date_rows["code"].drop_duplicates().tolist()
    for offset in range(0, len(codes), args.batch_size):
        chunk = codes[offset:offset + args.batch_size]
        batch = prepare_shadow_batch(
            chunk, trade_date, trade_date, anchor_date=args.anchor_date,
            denominator_batch_id=args.denominator,
        )
        batch_id = batch["batch_id"]
        collected = _run_shadow_batch(batch_id, f"hr{batch_id}")
        shadow = get_shadow_batch(batch_id)
        batch_result = {"batch_id": batch_id, "trade_date": trade_date, "collected": collected, "published": 0}
        for item in shadow.get("work_items", []):
            if item.get("status") not in ("collected_review", "quarantined_conflict"):
                run["blocked"].append({"code": item.get("code"), "trade_date": trade_date, "reason": item.get("status")})
                continue
            try:
                release = prepare_shadow_release(
                    batch_id, item["code"], args.year, args.denominator,
                    False, repair_fields,
                )
                if release.get("status") != "ready" or release.get("repair_fields") != list(repair_fields):
                    run["blocked"].append({
                        "code": item["code"], "trade_date": trade_date,
                        "reason": release.get("blockers") or release.get("status"),
                        "release_id": release.get("release_id"),
                    })
                    continue
                published = publish_historical_release(release["release_id"])
                if published.get("status") != "published" or published.get("db_input_rows") != 1:
                    run["blocked"].append({"code": item["code"], "trade_date": trade_date, "reason": "发布后行数或状态异常"})
                    continue
                run["published_rows"] += 1
                batch_result["published"] += 1
                run["published_releases"].append(release["release_id"])
            except Exception as exc:
                run["blocked"].append({
                    "code": item.get("code"), "trade_date": trade_date,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
        run["batch_results"].append(batch_result)
        run["updated_at"] = now_iso()
        save_json_atomic(run, run_path)
        print({
            "run_id": run_id,
            "batch_id": batch_id,
            "trade_date": trade_date,
            "batch_size": len(chunk),
            "published": batch_result["published"],
            "blocked_total": len(run["blocked"]),
        }, flush=True)

run["status"] = "completed" if not run["blocked"] else "completed_with_blocked_rows"
run["updated_at"] = now_iso()
run["blocked_reason_counts"] = dict(Counter(str(item.get("reason")) for item in run["blocked"]))
save_json_atomic(run, run_path)
print({
    "run_id": run_id,
    "status": run["status"],
    "candidate_rows": run["candidate_rows"],
    "published_rows": run["published_rows"],
    "blocked_rows": len(run["blocked"]),
    "manifest": str(run_path),
}, flush=True)
