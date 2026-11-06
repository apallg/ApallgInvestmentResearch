import argparse
import datetime
import json
import sys
from pathlib import Path
from datetime import timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data.connection import Database
from core.shared.persistence import save_json_atomic


parser = argparse.ArgumentParser()
parser.add_argument("--start-date", default="2018-01-01")
args = parser.parse_args()
datetime.date.fromisoformat(args.start_date)

gap_codes = Database.fetchall("""
    SELECT DISTINCT f.code, i.ts_code
    FROM fundamental_data f
    LEFT JOIN stock_identity i ON i.symbol=f.code
    WHERE f.report_date >= %s
      AND (f.ann_date IS NULL OR f.grossprofit_margin IS NULL)
    ORDER BY f.code
""", (args.start_date,))

codes = []
seen_codes = set()
for row in gap_codes:
    ts_code = row.get("ts_code")
    code = str(row.get("code") or "").zfill(6)
    if not ts_code:
        exchange = "SH" if code.startswith("6") else "BJ" if code.startswith(("4", "8")) else "SZ"
        ts_code = f"{code}.{exchange}"
    if ts_code not in seen_codes:
        codes.append(ts_code)
        seen_codes.add(ts_code)

before = Database.fetchone("""
    SELECT COUNT(*) AS rows_count,
           SUM(ann_date IS NULL) AS null_ann_date,
           SUM(grossprofit_margin IS NULL) AS null_grossprofit_margin
    FROM fundamental_data
    WHERE report_date >= %s
""", (args.start_date,)) or {}

from backend.service.fundamental_collection import fetch_and_save_fundamentals

repair_dir = ROOT / "data/historical_repair"
repair_dir.mkdir(parents=True, exist_ok=True)
output = repair_dir / "fundamental_gap_backfill_2018_plus.json"
progress_path = repair_dir / "fundamental_gap_backfill_2018_plus.jsonl"
latest_status = {}
if progress_path.is_file():
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            latest_status[item["code"]] = item.get("status")
        except (ValueError, KeyError):
            continue

manifest = {
    "status": "running",
    "source": "TuShare fina_indicator through fetch_and_save_fundamentals",
    "start_date": args.start_date,
    "requested_security_count": len(codes),
    "completed_security_count": sum(latest_status.get(code) == "completed" for code in codes),
    "failed_security_count": sum(latest_status.get(code) == "failed" for code in codes),
    "before": {key: int(before.get(key) or 0) for key in ("rows_count", "null_ann_date", "null_grossprofit_margin")},
    "started_at": datetime.datetime.now(timezone.utc).isoformat(),
}
save_json_atomic(manifest, output)

for index, code in enumerate(codes, start=1):
    if latest_status.get(code) == "completed":
        continue
    result = fetch_and_save_fundamentals(codes=[code], max_workers=1)
    succeeded = int(result.get("failed_batches") or 0) == 0
    item = {
        "code": code,
        "status": "completed" if succeeded else "failed",
        "collector_result": result,
        "observed_at": datetime.datetime.now(timezone.utc).isoformat(),
    }
    with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        handle.flush()
    latest_status[code] = item["status"]
    manifest["completed_security_count"] = sum(latest_status.get(candidate) == "completed" for candidate in codes)
    manifest["failed_security_count"] = sum(latest_status.get(candidate) == "failed" for candidate in codes)
    if index % 100 == 0 or index == len(codes):
        manifest["updated_at"] = datetime.datetime.now(timezone.utc).isoformat()
        save_json_atomic(manifest, output)
        print({
            "current": index,
            "total": len(codes),
            "completed": manifest["completed_security_count"],
            "failed": manifest["failed_security_count"],
        }, flush=True)

after = Database.fetchone("""
    SELECT COUNT(*) AS rows_count,
           SUM(ann_date IS NULL) AS null_ann_date,
           SUM(grossprofit_margin IS NULL) AS null_grossprofit_margin
    FROM fundamental_data
    WHERE report_date >= %s
""", (args.start_date,)) or {}

manifest["status"] = "completed_with_source_gaps" if manifest["failed_security_count"] == 0 else "review_required"
manifest["after"] = {key: int(after.get(key) or 0) for key in ("rows_count", "null_ann_date", "null_grossprofit_margin")}
manifest["finished_at"] = datetime.datetime.now(timezone.utc).isoformat()
save_json_atomic(manifest, output)
print(json.dumps(manifest, ensure_ascii=False, default=str))
