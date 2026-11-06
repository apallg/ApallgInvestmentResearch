import hashlib
import argparse
from collections import defaultdict
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.service.stock_context import ensure_stock_context_tables
from core.data import Database
from core.data.repos.dividend_source_versions import save_dividend_source_versions
from core.data.sources.tushare import _rate_limit, get_pro
from core.shared.persistence import load_json, save_json_atomic


root = Path(__file__).resolve().parents[1] / "data" / "historical_repair" / "dividend_backfill"
root.mkdir(parents=True, exist_ok=True)
manifest_path = root / "manifest.json"
progress_path = root / "progress.jsonl"
source_path = root / "source_rows_validated.jsonl"
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int)
args = parser.parse_args()
manifest = load_json(manifest_path) or {
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "source": "TuShare Pro dividend(ts_code=...) current full history",
    "scope": "SSE and SZSE current-listed and delisted securities",
    "source_capture_semantics": "SDK-returned DataFrame JSONL; provider wire bytes and revision IDs unavailable",
    "requested_codes": 0,
    "completed_codes": 0,
    "failed_codes": 0,
    "saturated_codes": 0,
    "published_rows": 0,
}
completed = {}
if progress_path.is_file():
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                if item.get("status") in ("published", "empty"):
                    completed[item["ts_code"]] = item
            except (ValueError, KeyError):
                continue

ensure_stock_context_tables()
source_path.touch(exist_ok=True)
pro = get_pro()
if not pro:
    raise RuntimeError("TuShare 不可用")

codes = []
for status in ("L", "D", "P"):
    _rate_limit()
    frame = pro.stock_basic(
        exchange="", list_status=status, fields="ts_code,exchange,list_status",
    )
    if frame is not None and not frame.empty:
        codes.extend(
            row["ts_code"] for row in frame.to_dict("records")
            if row.get("exchange") in ("SSE", "SZSE") and row.get("ts_code")
        )
codes = sorted(set(codes))
manifest["requested_codes"] = len(codes)
manifest["universe_observed_at"] = datetime.now(timezone.utc).isoformat()
manifest["completed_codes"] = len(completed)
manifest["status"] = "running"
manifest["started_at"] = datetime.now(timezone.utc).isoformat()
manifest["updated_at"] = manifest["started_at"]
manifest["pilot_only"] = args.limit is not None
manifest.pop("finished_at", None)
save_json_atomic(manifest, manifest_path)
if args.limit is not None:
    if args.limit < 1:
        raise ValueError("limit 必须大于 0")
    codes = codes[:args.limit]

fields = "ts_code,end_date,ann_date,div_proc,stk_div,cash_div,cash_div_tax,record_date,ex_date,pay_date,imp_ann_date"
columns = fields.split(",")
insert_sql = """
INSERT INTO stock_dividend_events
    (ts_code,end_date,ann_date,div_proc,stk_div,cash_div,cash_div_tax,record_date,ex_date,pay_date,imp_ann_date,data_quality_status)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'source_consistent_snapshot')
ON DUPLICATE KEY UPDATE
    ann_date=VALUES(ann_date),div_proc=VALUES(div_proc),stk_div=VALUES(stk_div),cash_div=VALUES(cash_div),
    cash_div_tax=VALUES(cash_div_tax),record_date=VALUES(record_date),ex_date=VALUES(ex_date),
    pay_date=VALUES(pay_date),imp_ann_date=VALUES(imp_ann_date),data_quality_status=VALUES(data_quality_status)
"""


def _date(value):
    if value is None or str(value) in ("", "nan", "None"):
        return None
    return datetime.strptime(str(value).replace("-", "")[:8], "%Y%m%d").date().isoformat()


for index, ts_code in enumerate(codes, start=1):
    if ts_code in completed:
        continue
    item = {
        "ts_code": ts_code,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _rate_limit()
        frame = pro.dividend(ts_code=ts_code, fields=fields)
        if frame is None:
            frame = pd.DataFrame(columns=columns)
        if len(frame) >= 2000:
            item.update({"status": "saturated", "rows": len(frame)})
            manifest["saturated_codes"] += 1
        else:
            source_records = []
            for raw_row in frame.to_dict("records"):
                source_row = {}
                for key, value in raw_row.items():
                    if pd.isna(value):
                        value = None
                    elif hasattr(value, "item"):
                        value = value.item()
                    source_row[key] = value
                source_records.append(source_row)
            source_text = "\n".join(
                json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":"))
                for row in source_records
            )
            source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            if source_records:
                capture_id = hashlib.sha256(
                    f"{manifest['started_at']}|{ts_code}|{source_sha256}".encode("utf-8")
                ).hexdigest()
                observed_at = datetime.fromisoformat(item["observed_at"]).replace(tzinfo=None)
                save_dividend_source_versions(capture_id, observed_at, source_records)
                with source_path.open("a", encoding="utf-8", newline="\n") as handle:
                    for row in source_records:
                        handle.write(json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
                    handle.flush()
                grouped_records = defaultdict(list)
                for row in source_records:
                    if not row.get("ts_code") or not row.get("end_date"):
                        continue
                    key = (row.get("ts_code"), row.get("end_date"), row.get("ann_date"), row.get("div_proc"))
                    grouped_records[key].append(row)
                rows = []
                conflict_keys = []
                for key, grouped_rows in grouped_records.items():
                    payloads = {
                        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                        for row in grouped_rows
                    }
                    if len(payloads) > 1:
                        conflict_keys.append(key)
                        continue
                    row = grouped_rows[0]
                    rows.append((
                        row.get("ts_code"), _date(row.get("end_date")), _date(row.get("ann_date")),
                        row.get("div_proc"), row.get("stk_div"), row.get("cash_div"),
                        row.get("cash_div_tax"), _date(row.get("record_date")), _date(row.get("ex_date")),
                        _date(row.get("pay_date")), _date(row.get("imp_ann_date")),
                    ))
                if rows:
                    Database.executemany(insert_sql, rows)
                if conflict_keys:
                    Database.executemany("""UPDATE stock_dividend_events SET data_quality_status='source_conflict'
                        WHERE ts_code=%s AND end_date=%s AND ann_date <=> %s AND div_proc <=> %s""",
                        [(code, _date(end_date), _date(ann_date), div_proc)
                         for code, end_date, ann_date, div_proc in conflict_keys])
                item.update({
                    "status": "published" if source_records else "empty",
                    "rows": len(source_records),
                    "canonical_rows": len(rows),
                    "conflicting_keys": len(conflict_keys),
                    "source_sha256": source_sha256,
                })
                manifest["published_rows"] += len(source_records)
            else:
                item.update({"status": "empty", "rows": 0, "source_sha256": source_sha256})
            manifest["completed_codes"] += 1
    except Exception as error:
        item.update({"status": "failed", "error": f"{type(error).__name__}: {error}"[:300]})
        manifest["failed_codes"] += 1

    with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    if index % 100 == 0 or index == len(codes):
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json_atomic(manifest, manifest_path)
        print({
            "current": index,
            "total": len(codes),
            "completed": manifest["completed_codes"],
            "failed": manifest["failed_codes"],
            "saturated": manifest["saturated_codes"],
            "published_rows": manifest["published_rows"],
        }, flush=True)

source_groups = defaultdict(list)
source_row_count = 0
for line_number, line in enumerate(source_path.open("r", encoding="utf-8"), start=1):
    try:
        source_row = json.loads(line)
    except ValueError:
        continue
    source_row_count += 1
    key = (source_row.get("ts_code"), source_row.get("end_date"), source_row.get("ann_date"), source_row.get("div_proc"))
    source_groups[key].append({"source_line": line_number, "row": source_row})

exact_duplicates = []
conflicts = []
conflicting_fields = {}
for key, items in source_groups.items():
    if len(items) < 2:
        continue
    payloads = {json.dumps(item["row"], sort_keys=True, ensure_ascii=False, separators=(",", ":")) for item in items}
    if len(payloads) == 1:
        exact_duplicates.append({"business_key": key, "copies": len(items), "source_lines": [item["source_line"] for item in items]})
    else:
        conflicts.append({"business_key": key, "source_rows": items})
        for field in ("stk_div", "cash_div", "cash_div_tax", "record_date", "ex_date", "pay_date", "imp_ann_date"):
            if len({str(item["row"].get(field)) for item in items}) > 1:
                conflicting_fields[field] = conflicting_fields.get(field, 0) + 1

save_json_atomic({
    "source_rows": source_row_count,
    "unique_business_keys": len(source_groups),
    "exact_duplicate_groups": len(exact_duplicates),
    "exact_duplicate_excess_rows": sum(item["copies"] - 1 for item in exact_duplicates),
    "conflicting_business_key_groups": len(conflicts),
    "conflicting_source_rows": sum(len(item["source_rows"]) for item in conflicts),
    "conflicting_fields_by_group": conflicting_fields,
    "exact_duplicates": exact_duplicates,
    "conflicts": conflicts,
}, root / "source_conflict_audit.json")
manifest["source_unique_business_keys"] = len(source_groups)
manifest["source_exact_duplicate_excess_rows"] = sum(item["copies"] - 1 for item in exact_duplicates)
manifest["source_conflicting_business_key_groups"] = len(conflicts)
manifest["source_conflict_audit"] = "source_conflict_audit.json"
manifest["data_review_required"] = bool(conflicts)
manifest["status"] = (
    "partial" if args.limit is not None
    else "completed" if not manifest["failed_codes"] and not manifest["saturated_codes"]
    else "review_required"
)
manifest["pilot_only"] = args.limit is not None
manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
save_json_atomic(manifest, manifest_path)
print({key: manifest[key] for key in ("status", "requested_codes", "completed_codes", "failed_codes", "saturated_codes", "published_rows")})
