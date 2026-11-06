import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data.connection import Database
from core.shared.persistence import save_json_atomic


parser = argparse.ArgumentParser()
parser.add_argument("--start-year", type=int, default=2012)
parser.add_argument("--end-year", type=int, default=2026)
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()
if args.start_year > args.end_year:
    raise ValueError("start-year 不能大于 end-year")

column = Database.fetchone("SHOW COLUMNS FROM daily_quotes LIKE 'turnover'") or {}
if "(12,6)" not in str(column.get("Type", "")):
    raise RuntimeError(f"daily_quotes.turnover 精度尚未升级: {column.get('Type')}")

updated_by_year = []
if not args.verify_only:
    for year in range(args.start_year, args.end_year + 1):
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"

        def _restore(cursor):
            cursor.execute("""
                UPDATE daily_quotes q
                JOIN daily_quote_turnover_repair_audit a
                  ON a.code=q.code AND a.trade_date=q.trade_date
                SET q.turnover=a.new_turnover
                WHERE q.trade_date >= %s AND q.trade_date < %s
                  AND a.source_table='daily_basic'
            """, (start, end))
            return int(cursor.rowcount)

        updated = Database.transaction(_restore)
        updated_by_year.append({"year": year, "restored_exact_values": updated})
        print({"year": year, "restored_exact_values": updated}, flush=True)

audit = Database.fetchone("""
    SELECT COUNT(*) AS mismatches
    FROM daily_quotes q
    JOIN daily_quote_turnover_repair_audit a
      ON a.code=q.code AND a.trade_date=q.trade_date
    WHERE a.source_table IN ('daily_basic','baostock','tushare_float_share_formula')
      AND NOT (q.turnover <=> a.new_turnover)
""") or {}
mismatches = int(audit.get("mismatches") or 0)
counts = Database.fetchone("""
    SELECT SUM(volume>0 AND amount>0 AND turnover=0) AS traded_zero,
           SUM(volume>0 AND amount>0 AND turnover IS NULL) AS traded_null
    FROM daily_quotes
    WHERE trade_date >= %s AND trade_date < %s
""", (f"{args.start_year}-01-01", f"{args.end_year + 1}-01-01")) or {}
audit_count = Database.fetchone("""
    SELECT COUNT(*) AS rows_count,
           SUM(old_turnover IS NULL) AS restored_from_null,
           SUM(old_turnover=0) AS restored_from_zero
    FROM daily_quote_turnover_repair_audit
    WHERE trade_date >= %s AND trade_date < %s
""", (f"{args.start_year}-01-01", f"{args.end_year + 1}-01-01")) or {}
unknown = Database.fetchall("""
    SELECT a.source_status,COUNT(*) AS rows_count
    FROM daily_quote_turnover_unknown_audit a
    JOIN daily_quotes q ON q.code=a.code AND q.trade_date=a.trade_date
    WHERE a.trade_date >= %s AND a.trade_date < %s
      AND q.turnover IS NULL AND q.volume>0 AND q.amount>0
    GROUP BY a.source_status
""", (f"{args.start_year}-01-01", f"{args.end_year + 1}-01-01"))
manifest = {
    "status": "review_required" if mismatches or int(counts.get("traded_zero") or 0) or int(counts.get("traded_null") or 0) else "verified",
    "source": "daily_basic.turnover_rate + BaoStock + validated float_share reconstruction",
    "start_year": args.start_year,
    "end_year": args.end_year,
    "audit_rows": int(audit_count.get("rows_count") or 0),
    "restored_from_null": int(audit_count.get("restored_from_null") or 0),
    "restored_from_zero": int(audit_count.get("restored_from_zero") or 0),
    "audit_rows_with_mismatch": mismatches,
    "remaining_traded_zero": int(counts.get("traded_zero") or 0),
    "remaining_traded_null": int(counts.get("traded_null") or 0),
    "unresolved_audit_by_source_status": {row["source_status"]: int(row["rows_count"]) for row in unknown},
    "updated_by_year": updated_by_year,
    "repair_rows_by_source": {
        row["source_table"]: int(row["rows_count"])
        for row in Database.fetchall("""
            SELECT source_table,COUNT(*) AS rows_count FROM daily_quote_turnover_repair_audit
            WHERE trade_date >= %s AND trade_date < %s GROUP BY source_table
        """, (f"{args.start_year}-01-01", f"{args.end_year + 1}-01-01"))
    },
    "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
output = Path(__file__).resolve().parents[1] / "data" / "historical_repair" / "daily_quote_turnover_restore.json"
output.parent.mkdir(parents=True, exist_ok=True)
save_json_atomic(manifest, output)
print(json.dumps(manifest, ensure_ascii=False), flush=True)
