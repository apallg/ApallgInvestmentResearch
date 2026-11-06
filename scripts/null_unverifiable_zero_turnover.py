import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data.connection import Database


parser = argparse.ArgumentParser()
parser.add_argument("--start-year", type=int, default=2012)
parser.add_argument("--end-year", type=int, default=2026)
args = parser.parse_args()
if args.start_year > args.end_year:
    raise ValueError("start-year 不能大于 end-year")

Database.execute("""
CREATE TABLE IF NOT EXISTS daily_quote_turnover_unknown_audit (
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    old_turnover DECIMAL(12,6),
    new_turnover DECIMAL(12,6),
    source_status VARCHAR(48) NOT NULL,
    repaired_at DATETIME NOT NULL,
    PRIMARY KEY (code, trade_date),
    KEY idx_turnover_unknown_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")
Database.execute("ALTER TABLE daily_quote_turnover_unknown_audit MODIFY COLUMN old_turnover DECIMAL(12,6) NULL")

for year in range(args.start_year, args.end_year + 1):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    def _repair(cursor):
        cursor.execute("""
            INSERT IGNORE INTO daily_quote_turnover_unknown_audit
                (code, trade_date, old_turnover, new_turnover, source_status, repaired_at)
            SELECT q.code, q.trade_date, q.turnover, NULL,
                   CASE
                     WHEN b.code IS NULL THEN 'daily_basic_missing'
                     WHEN b.float_share>0 AND (b.turnover_rate IS NULL OR b.turnover_rate<=0) THEN 'rounded_zero_float_share_available'
                     WHEN b.turnover_rate IS NULL THEN 'daily_basic_rate_null'
                     ELSE 'daily_basic_rate_zero'
                   END,
                   NOW()
            FROM daily_quotes q
            LEFT JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.volume>0 AND q.amount>0 AND q.turnover IS NULL
              AND (b.code IS NULL OR b.turnover_rate IS NULL OR b.turnover_rate<=0)
        """, (start, end))
        cursor.execute("""
            INSERT IGNORE INTO daily_quote_turnover_unknown_audit
                (code, trade_date, old_turnover, new_turnover, source_status, repaired_at)
            SELECT q.code, q.trade_date, q.turnover, NULL,
                   CASE WHEN b.float_share>0 THEN 'rounded_zero_float_share_available'
                        ELSE 'positive_volume_no_valid_daily_basic' END, NOW()
            FROM daily_quotes q
            LEFT JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.turnover=0 AND q.volume>0 AND q.amount>0
              AND (b.code IS NULL OR b.turnover_rate IS NULL OR b.turnover_rate<=0)
        """, (start, end))
        cursor.execute("""
            INSERT IGNORE INTO daily_quote_turnover_unknown_audit
                (code, trade_date, old_turnover, new_turnover, source_status, repaired_at)
            SELECT q.code, q.trade_date, q.turnover, NULL,
                   'positive_rate_rounded_to_zero', NOW()
            FROM daily_quotes q
            JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.turnover=0 AND q.volume>0 AND q.amount>0
              AND b.turnover_rate>0
        """, (start, end))
        cursor.execute("""
            UPDATE daily_quotes q
            LEFT JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            SET q.turnover=NULL
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.turnover=0 AND q.volume>0 AND q.amount>0
              AND (b.code IS NULL OR b.turnover_rate IS NULL OR b.turnover_rate<=0)
        """, (start, end))
        cursor.execute("""
            UPDATE daily_quotes q
            JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            SET q.turnover=NULL
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.turnover=0 AND q.volume>0 AND q.amount>0
              AND b.turnover_rate>0
        """, (start, end))
        return int(cursor.rowcount)

    updated = Database.transaction(_repair)
    print({"year": year, "set_null": updated}, flush=True)

audit = Database.fetchone("SELECT COUNT(*) AS rows_count FROM daily_quote_turnover_unknown_audit") or {}
print({"audit_rows": int(audit.get("rows_count") or 0), "range": [args.start_year, args.end_year]})
