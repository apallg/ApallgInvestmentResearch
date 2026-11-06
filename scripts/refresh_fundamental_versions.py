import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.service.fundamental_collection import fetch_and_save_fundamentals
from core.data.connection import Database
from core.shared.persistence import now_iso


MANIFEST_PATH = ROOT / "data/historical_repair/fundamental_refresh_2018_20261024.json"
BATCH_SIZE = 40
MAX_ATTEMPTS = 3


def save_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_name(f"{MANIFEST_PATH.stem}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for retry in range(10):
        try:
            os.replace(temporary_path, MANIFEST_PATH)
            return
        except PermissionError:
            if retry == 9:
                raise
            time.sleep(0.5)


if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") == "completed":
        print(json.dumps({"status": "already_completed", "manifest": str(MANIFEST_PATH)}, ensure_ascii=False), flush=True)
        sys.exit(0)
else:
    rows = Database.fetchall("SELECT DISTINCT code FROM fundamental_data WHERE report_date >= '2018-01-01' ORDER BY code")
    codes = [str(row["code"]) for row in rows]
    manifest = {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "scope": "distinct fundamental_data codes with report_date >= 2018-01-01",
        "target_codes": codes,
        "completed_codes": [],
        "failed_codes": [],
        "attempts_by_code": {},
        "batch_results": [],
        "current_rows_upserted": 0,
        "source_versions_saved": 0,
        "conflicted_periods": 0,
    }
    save_manifest(manifest)

completed = set(manifest.get("completed_codes", []))
target_codes = manifest.get("target_codes", [])
pending = [code for code in target_codes if code not in completed]

for attempt in range(1, MAX_ATTEMPTS + 1):
    if not pending:
        break
    failed_this_round = []
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        for code in batch:
            manifest.setdefault("attempts_by_code", {})[code] = manifest.get("attempts_by_code", {}).get(code, 0) + 1
        manifest["status"] = "running"
        manifest["updated_at"] = now_iso()
        save_manifest(manifest)
        result = fetch_and_save_fundamentals(codes=batch, max_workers=1)
        failed = set(str(code) for code in result.get("failed_codes", []))
        failed.update(str(code) for code in result.get("conflicted_codes", []))
        if not result.get("capture_id"):
            failed.update(batch)
        succeeded = [code for code in batch if code not in failed]
        completed.update(succeeded)
        failed_this_round.extend(sorted(failed))
        manifest["completed_codes"] = sorted(completed)
        manifest["failed_codes"] = sorted(set(failed_this_round))
        manifest["current_rows_upserted"] = int(manifest.get("current_rows_upserted", 0)) + int(result.get("saved", 0))
        manifest["source_versions_saved"] = int(manifest.get("source_versions_saved", 0)) + int(result.get("versions_saved", 0))
        manifest["conflicted_periods"] = int(manifest.get("conflicted_periods", 0)) + int(result.get("conflicted_periods", 0))
        manifest.setdefault("batch_results", []).append({
            "attempt": attempt,
            "input_codes": len(batch),
            "failed_codes": sorted(failed),
            "capture_id": result.get("capture_id"),
            "current_rows_upserted": int(result.get("saved", 0)),
            "source_versions_saved": int(result.get("versions_saved", 0)),
            "conflicted_periods": int(result.get("conflicted_periods", 0)),
            "conflicted_codes": sorted(set(str(code) for code in result.get("conflicted_codes", []))),
        })
        manifest["updated_at"] = now_iso()
        save_manifest(manifest)
        print(json.dumps({
            "status": "progress",
            "attempt": attempt,
            "completed_codes": len(completed),
            "target_codes": len(target_codes),
            "failed_this_round": len(failed_this_round),
            "capture_id": result.get("capture_id"),
        }, ensure_ascii=False), flush=True)
    pending = [code for code in failed_this_round if code not in completed]

manifest["status"] = "completed" if len(completed) == len(target_codes) else "partial"
manifest["completed_codes"] = sorted(completed)
manifest["failed_codes"] = sorted(set(pending))
manifest["updated_at"] = now_iso()
save_manifest(manifest)
print(json.dumps({
    "status": manifest["status"],
    "completed_codes": len(completed),
    "target_codes": len(target_codes),
    "failed_codes": manifest["failed_codes"],
    "current_rows_upserted": manifest["current_rows_upserted"],
    "source_versions_saved": manifest["source_versions_saved"],
    "conflicted_periods": manifest["conflicted_periods"],
    "manifest": str(MANIFEST_PATH),
}, ensure_ascii=False), flush=True)
