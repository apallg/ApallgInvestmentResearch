import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import Database
from core.data.sources.historical_quotes import fetch_baostock_raw_daily, get_source_versions
from core.shared.persistence import save_json_atomic


parser = argparse.ArgumentParser()
parser.add_argument("--publish", action="store_true")
args = parser.parse_args()

targets = Database.fetchall("""
    SELECT q.code, CAST(q.trade_date AS CHAR) AS trade_date, q.volume, q.amount,
           q.turnover, a.source_status
    FROM daily_quotes q
    JOIN daily_quote_turnover_unknown_audit a ON a.code=q.code AND a.trade_date=q.trade_date
    WHERE q.turnover IS NULL AND q.volume>0 AND q.amount>0
    ORDER BY q.code,q.trade_date
""")
if not targets:
    print(json.dumps({"status": "no_candidates", "repaired": 0}, ensure_ascii=False), flush=True)
    raise SystemExit(0)

by_code = {}
for row in targets:
    by_code.setdefault(row["code"], []).append(row)

observed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
fetched = {}
for code, rows in by_code.items():
    start = min(row["trade_date"] for row in rows)
    end = max(row["trade_date"] for row in rows)
    exchange = "sh" if code.startswith(("6", "9")) else "sz"
    source = fetch_baostock_raw_daily(code, exchange, start, end)["normalized"]
    fetched[code] = {str(row["date"])[:10]: row for row in source.to_dict("records")}

sample = fetch_baostock_raw_daily("600000", "sh", "2026-10-08", "2026-10-23")["normalized"]
sample_basic = Database.fetchall("""
    SELECT CAST(trade_date AS CHAR) AS trade_date,turnover_rate
    FROM daily_basic WHERE code='600000' AND trade_date BETWEEN '2026-10-08' AND '2026-10-23'
      AND turnover_rate IS NOT NULL
""")
sample_lookup = {row["trade_date"]: float(row["turnover_rate"]) for row in sample_basic}
sample_matches = []
for row in sample.to_dict("records"):
    day = str(row["date"])[:10]
    if day in sample_lookup and row.get("turn") is not None:
        difference = abs(float(row["turn"]) - sample_lookup[day])
        sample_matches.append({"trade_date": day, "difference": difference})
if len(sample_matches) < 5 or any(row["difference"] > 0.000001 for row in sample_matches):
    raise RuntimeError("BaoStock 与 TuShare 换手率口径交叉核验未通过")

eligible = []
skipped = []
for target in targets:
    source = fetched[target["code"]].get(target["trade_date"])
    if not source or source.get("turn") is None or float(source["turn"]) <= 0:
        skipped.append({"code": target["code"], "trade_date": target["trade_date"], "reason": "BaoStock 未返回正换手率"})
        continue
    if int(source["tradestatus"]) != 1:
        skipped.append({"code": target["code"], "trade_date": target["trade_date"], "reason": "BaoStock 标记非交易状态"})
        continue
    if float(source["volume"]) != float(target["volume"]) or abs(float(source["amount"]) - float(target["amount"])) > 0.01:
        skipped.append({"code": target["code"], "trade_date": target["trade_date"], "reason": "独立源成交量或成交额不一致"})
        continue
    eligible.append({
        "code": target["code"], "trade_date": target["trade_date"],
        "old_turnover": target["turnover"], "new_turnover": float(source["turn"]),
        "volume": float(source["volume"]), "amount": float(source["amount"]),
        "source_status": target["source_status"],
    })

manifest = {
    "status": "shadow_verified" if not args.publish else "validated_pending_publish",
    "source": "BaoStock query_history_k_data_plus.turn",
    "source_versions": get_source_versions(),
    "observed_at": observed_at,
    "candidate_rows": len(targets),
    "eligible_rows": eligible,
    "skipped_rows": skipped,
    "semantic_crosscheck": {"reference": "600000.SH vs daily_basic.turnover_rate", "matches": sample_matches},
    "writes_data": False,
}
artifact = Path(__file__).resolve().parents[1] / "data/historical_repair/turnover_baostock" / f"{observed_at[:10]}_{datetime.datetime.now().strftime('%H%M%S')}.json"
save_json_atomic(manifest, artifact)

if args.publish and eligible:
    def _publish(cursor):
        for row in eligible:
            cursor.execute("""
                UPDATE daily_quotes SET turnover=%s
                WHERE code=%s AND trade_date=%s AND turnover IS NULL AND volume=%s AND amount=%s
            """, (row["new_turnover"], row["code"], row["trade_date"], row["volume"], row["amount"]))
            if cursor.rowcount != 1:
                raise RuntimeError(f"行情行已变化或不匹配，拒绝发布: {row['code']} {row['trade_date']}")
            cursor.execute("""
                INSERT INTO daily_quote_turnover_repair_audit
                    (code,trade_date,old_turnover,new_turnover,source_table,repaired_at)
                VALUES (%s,%s,%s,%s,'baostock',NOW())
            """, (row["code"], row["trade_date"], row["old_turnover"], row["new_turnover"]))
            cursor.execute("""
                UPDATE daily_quote_turnover_unknown_audit
                SET new_turnover=%s,source_status='restored_baostock',repaired_at=NOW()
                WHERE code=%s AND trade_date=%s AND new_turnover IS NULL
            """, (row["new_turnover"], row["code"], row["trade_date"]))
            if cursor.rowcount != 1:
                raise RuntimeError(f"缺失审计记录已变化，拒绝发布: {row['code']} {row['trade_date']}")
        return len(eligible)

    manifest["published_rows"] = Database.transaction(_publish)
    manifest["status"] = "published"
    manifest["writes_data"] = True
    save_json_atomic(manifest, artifact)

print(json.dumps({"manifest": str(artifact), **manifest}, ensure_ascii=False, default=str), flush=True)
