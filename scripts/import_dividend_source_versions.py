import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.service.stock_context import ensure_stock_context_tables
from core.data import Database
from core.data.repos.dividend_source_versions import ensure_dividend_source_versions, save_dividend_source_versions
from core.shared.persistence import load_json, save_json_atomic


source_dir = ROOT / "data/historical_repair/dividend_backfill"
source_path = source_dir / "source_rows_validated.jsonl"
manifest_path = source_dir / "manifest.json"
import_manifest_path = source_dir / "source_versions_import.json"
source_manifest = load_json(manifest_path) or {}
conflict_manifest = load_json(source_dir / "source_conflict_audit.json") or {}
previous_import = load_json(import_manifest_path) or {}
if source_manifest.get("status") != "completed" or not source_path.is_file():
    raise RuntimeError("分红源快照未完成或源文件不存在")

ensure_stock_context_tables()
ensure_dividend_source_versions()
digest = hashlib.sha256()
with source_path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
capture_id = digest.hexdigest()
observed_at = datetime.fromisoformat(source_manifest["finished_at"]).replace(tzinfo=None)
expected_rows = int(source_manifest.get("published_rows", 0))
written_rows = 0
chunk = []
source_row_no = 0
with source_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        source_row_no += 1
        record = json.loads(line)
        chunk.append(record)
        if len(chunk) >= 1000:
            written_rows += save_dividend_source_versions(capture_id, observed_at, chunk, source_row_no - len(chunk))
            chunk.clear()
    if chunk:
        written_rows += save_dividend_source_versions(capture_id, observed_at, chunk, source_row_no - len(chunk))

Database.execute("""UPDATE stock_dividend_events e
    JOIN (
        SELECT ts_code,end_date,ann_date,div_proc,
               COUNT(DISTINCT payload_sha256) AS payload_versions
        FROM stock_dividend_source_versions
        WHERE capture_id=%s
        GROUP BY ts_code,end_date,ann_date,div_proc
    ) s ON e.ts_code=s.ts_code AND e.end_date=s.end_date
       AND e.ann_date <=> s.ann_date AND e.div_proc <=> s.div_proc
    SET e.data_quality_status=IF(s.payload_versions > 1,'source_conflict','source_consistent_snapshot')""", (capture_id,))

conflicted_before = Database.fetchone("SELECT COUNT(*) AS n FROM stock_dividend_events WHERE data_quality_status='source_conflict'")
deleted_conflicts = Database.transaction(lambda cursor: (cursor.execute(
    "DELETE FROM stock_dividend_events WHERE data_quality_status='source_conflict'"), int(cursor.rowcount))[1])
schema_result = ensure_dividend_source_versions()
version_rows = Database.fetchone("SELECT COUNT(*) AS n FROM stock_dividend_source_versions WHERE capture_id=%s", (capture_id,))
status_rows = Database.fetchall("""SELECT data_quality_status,COUNT(*) AS n
    FROM stock_dividend_events GROUP BY data_quality_status""")
indexes = Database.fetchall("SHOW INDEX FROM stock_dividend_events")
verified_rows = int((version_rows or {}).get("n") or 0)
quarantined_total = max(
    int(previous_import.get("quarantined_canonical_rows_total", 0)),
    int(previous_import.get("conflicted_canonical_rows", 0)),
    deleted_conflicts,
)
report = {
    "status": "verified" if verified_rows == expected_rows else "review_required",
    "capture_id": capture_id,
    "observed_at": observed_at.isoformat(),
    "source_rows": source_row_no,
    "expected_rows": expected_rows,
    "insert_attempted_rows": written_rows,
    "verified_version_rows": verified_rows,
    "conflicted_canonical_rows": int((conflicted_before or {}).get("n") or 0),
    "quarantined_canonical_rows_removed_this_run": deleted_conflicts,
    "quarantined_canonical_rows_total": quarantined_total,
    "source_conflicting_business_key_groups": int(conflict_manifest.get("conflicting_business_key_groups", 0)),
    "canonical_status_counts": {row["data_quality_status"]: int(row["n"]) for row in status_rows},
    "null_safe_unique_index_installed": any(row.get("Key_name") == "uk_dividend_version_nullsafe" for row in indexes),
    "schema_changes": schema_result.get("changed", []),
}
save_json_atomic(report, import_manifest_path)
source_manifest["versioned_source_capture_id"] = capture_id
source_manifest["versioned_source_rows"] = verified_rows
source_manifest["versioned_source_status"] = report["status"]
source_manifest["canonical_conflict_rows_quarantined"] = quarantined_total
source_manifest["source_conflicting_business_key_groups"] = int(conflict_manifest.get("conflicting_business_key_groups", 0))
source_manifest["data_review_required"] = bool(conflict_manifest.get("conflicting_business_key_groups"))
save_json_atomic(source_manifest, manifest_path)
print(json.dumps(report, ensure_ascii=False))
