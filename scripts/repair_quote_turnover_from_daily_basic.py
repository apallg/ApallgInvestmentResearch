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
CREATE TABLE IF NOT EXISTS daily_quote_turnover_repair_audit (
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    old_turnover DECIMAL(12,6),
    new_turnover DECIMAL(12,6) NOT NULL,
    source_table VARCHAR(32) NOT NULL,
    repaired_at DATETIME NOT NULL,
    PRIMARY KEY (code, trade_date),
    KEY idx_turnover_repair_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")
Database.execute("ALTER TABLE daily_quote_turnover_repair_audit MODIFY COLUMN old_turnover DECIMAL(12,6) NULL")

for year in range(args.start_year, args.end_year + 1):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    def _repair(cursor):
        cursor.execute("""
            INSERT IGNORE INTO daily_quote_turnover_repair_audit
                (code, trade_date, old_turnover, new_turnover, source_table, repaired_at)
            SELECT q.code, q.trade_date, q.turnover, b.turnover_rate, 'daily_basic', NOW()
            FROM daily_quotes q
            JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.volume>0 AND q.amount>0
              AND (q.turnover IS NULL OR q.turnover=0) AND b.turnover_rate>0
        """, (start, end))
        cursor.execute("""
            UPDATE daily_quotes q
            JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            SET q.turnover=b.turnover_rate
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.volume>0 AND q.amount>0
              AND (q.turnover IS NULL OR q.turnover=0) AND b.turnover_rate>0
        """, (start, end))
        restored_from_daily_basic = int(cursor.rowcount)
        cursor.execute("""
            INSERT IGNORE INTO daily_quote_turnover_repair_audit
                (code,trade_date,old_turnover,new_turnover,source_table,repaired_at)
            SELECT q.code,q.trade_date,q.turnover,
                   ROUND(q.volume/(b.float_share*100),6),'tushare_float_share_formula',NOW()
            FROM daily_quotes q
            JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.volume>0 AND q.amount>0 AND b.float_share>0
              AND (q.turnover IS NULL OR q.turnover=0)
              AND (b.turnover_rate IS NULL OR b.turnover_rate<=0)
              AND ROUND(q.volume/(b.float_share*100),6)>0
        """, (start, end))
        cursor.execute("""
            UPDATE daily_quotes q
            JOIN daily_basic b ON b.code=q.code AND b.trade_date=q.trade_date
            SET q.turnover=ROUND(q.volume/(b.float_share*100),6)
            WHERE q.trade_date >= %s AND q.trade_date < %s
              AND q.volume>0 AND q.amount>0 AND b.float_share>0
              AND (q.turnover IS NULL OR q.turnover=0)
              AND (b.turnover_rate IS NULL OR b.turnover_rate<=0)
              AND ROUND(q.volume/(b.float_share*100),6)>0
        """, (start, end))
        restored_from_float_share = int(cursor.rowcount)
        cursor.execute("""
            UPDATE daily_quote_turnover_unknown_audit a
            JOIN daily_quote_turnover_repair_audit r ON r.code=a.code AND r.trade_date=a.trade_date
            SET a.new_turnover=r.new_turnover,a.source_status='reconstructed_from_float_share',a.repaired_at=NOW()
            WHERE a.trade_date >= %s AND a.trade_date < %s
              AND r.source_table='tushare_float_share_formula'
        """, (start, end))
        return {
            "from_daily_basic": restored_from_daily_basic,
            "from_float_share": restored_from_float_share,
        }

    updated = Database.transaction(_repair)
    print({"year": year, "updated": updated}, flush=True)

audit = Database.fetchone("SELECT COUNT(*) AS rows_count FROM daily_quote_turnover_repair_audit") or {}
print({"audit_rows": int(audit.get("rows_count") or 0), "range": [args.start_year, args.end_year]})
