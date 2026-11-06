import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data.connection import Database


parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()

eligible_sql = """
SELECT COUNT(*) AS rows_count
FROM fundamental_data f
JOIN (
    SELECT code, end_date, MIN(ann_date) AS ann_date
    FROM income_data
    WHERE ann_date IS NOT NULL
    GROUP BY code, end_date
    HAVING COUNT(DISTINCT ann_date) = 1
) i ON i.code = f.code AND i.end_date = f.report_date
WHERE f.ann_date IS NULL
"""
eligible = int((Database.fetchone(eligible_sql) or {}).get("rows_count") or 0)
if not args.apply:
    print({"eligible_unique_matches": eligible, "applied": False})
    raise SystemExit(0)

Database.execute("""
CREATE TABLE IF NOT EXISTS fundamental_ann_date_repair_audit (
    code VARCHAR(10) NOT NULL,
    report_date DATE NOT NULL,
    old_ann_date DATE DEFAULT NULL,
    new_ann_date DATE NOT NULL,
    source_ann_date DATE NOT NULL,
    repaired_at DATETIME NOT NULL,
    PRIMARY KEY (code, report_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")


def _apply(cursor):
    cursor.execute("""
        INSERT IGNORE INTO fundamental_ann_date_repair_audit
            (code, report_date, old_ann_date, new_ann_date, source_ann_date, repaired_at)
        SELECT f.code, f.report_date, f.ann_date, i.ann_date, i.ann_date, NOW()
        FROM fundamental_data f
        JOIN (
            SELECT code, end_date, MIN(ann_date) AS ann_date
            FROM income_data
            WHERE ann_date IS NOT NULL
            GROUP BY code, end_date
            HAVING COUNT(DISTINCT ann_date) = 1
        ) i ON i.code = f.code AND i.end_date = f.report_date
        WHERE f.ann_date IS NULL
    """)
    cursor.execute("""
        UPDATE fundamental_data f
        JOIN (
            SELECT code, end_date, MIN(ann_date) AS ann_date
            FROM income_data
            WHERE ann_date IS NOT NULL
            GROUP BY code, end_date
            HAVING COUNT(DISTINCT ann_date) = 1
        ) i ON i.code = f.code AND i.end_date = f.report_date
        SET f.ann_date = i.ann_date
        WHERE f.ann_date IS NULL
    """)
    updated = int(cursor.rowcount)
    if updated != eligible:
        raise RuntimeError(f"公告日回补行数变化：预期 {eligible}，实际 {updated}")
    return updated


updated = Database.transaction(_apply)
remaining = Database.fetchone("SELECT SUM(ann_date IS NULL) AS rows_count FROM fundamental_data")
audit_rows = Database.fetchone("SELECT COUNT(*) AS rows_count FROM fundamental_ann_date_repair_audit")
print({
    "eligible_unique_matches": eligible,
    "updated": updated,
    "audit_rows": int((audit_rows or {}).get("rows_count") or 0),
    "remaining_ann_date_null": int((remaining or {}).get("rows_count") or 0),
    "applied": True,
})
