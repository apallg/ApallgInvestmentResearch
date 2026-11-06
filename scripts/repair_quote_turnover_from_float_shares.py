import argparse
import datetime
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import Database
from core.data.sources.historical_quotes import fetch_baostock_raw_daily, get_source_versions
from core.data.sources.tushare import _rate_limit, get_pro
from core.shared.persistence import save_json_atomic


parser = argparse.ArgumentParser()
parser.add_argument("--publish", action="store_true")
args = parser.parse_args()

targets = Database.fetchall("""
    SELECT q.code,CAST(q.trade_date AS CHAR) AS trade_date,q.volume,q.amount,q.turnover,a.source_status
    FROM daily_quotes q
    JOIN daily_quote_turnover_unknown_audit a ON a.code=q.code AND a.trade_date=q.trade_date
    WHERE q.turnover IS NULL AND q.volume>0 AND q.amount>0
    ORDER BY q.code,q.trade_date
""")
if not targets:
    print(json.dumps({"status": "no_candidates", "repaired": 0}, ensure_ascii=False), flush=True)
    raise SystemExit(0)

pro = get_pro()
if not pro:
    raise RuntimeError("TuShare 不可用")

by_code = {}
for row in targets:
    by_code.setdefault(row["code"], []).append(row)

observed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
eligible = []
source_checks = []
for code, rows in by_code.items():
    start = (datetime.date.fromisoformat(min(row["trade_date"] for row in rows)) - datetime.timedelta(days=15)).isoformat()
    end = (datetime.date.fromisoformat(max(row["trade_date"] for row in rows)) + datetime.timedelta(days=15)).isoformat()
    ts_code = f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"
    _rate_limit()
    frame = pro.daily_basic(
        ts_code=ts_code, start_date=start.replace("-", ""), end_date=end.replace("-", ""),
        fields="ts_code,trade_date,turnover_rate,float_share",
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"TuShare daily_basic 无记录: {ts_code} {start}~{end}")
    basic = {str(row["trade_date"]): row for row in frame.to_dict("records")}
    source_checks.append({"code": code, "rows": len(frame), "positive_rate_checks": []})

    for record in frame.to_dict("records"):
        day = str(record.get("trade_date") or "")
        if not day or not record.get("float_share"):
            continue
        iso_day = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        quote = Database.fetchone(
            "SELECT volume FROM daily_quotes WHERE code=%s AND trade_date=%s", (code, iso_day),
        )
        if not quote or record.get("turnover_rate") is None or float(record["turnover_rate"]) <= 0:
            continue
        derived = Decimal(str(quote["volume"])) / (Decimal(str(record["float_share"])) * Decimal("100"))
        rounded4 = float(derived.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        matched = rounded4 == float(record["turnover_rate"])
        source_checks[-1]["positive_rate_checks"].append({
            "trade_date": iso_day, "source_turnover_rate": float(record["turnover_rate"]),
            "derived_4dp": rounded4, "matched": matched,
        })

    good_checks = source_checks[-1]["positive_rate_checks"]
    if not good_checks or not all(item["matched"] for item in good_checks):
        raise RuntimeError(f"{ts_code} 的流通股本复算校验不足或不一致")

    baostock = fetch_baostock_raw_daily(
        code, "sh" if code.startswith(("6", "9")) else "sz", start, end,
    )["normalized"]
    independent = {str(row["date"])[:10]: row for row in baostock.to_dict("records")}

    for target in rows:
        source = basic.get(target["trade_date"].replace("-", ""))
        other = independent.get(target["trade_date"])
        if not source or source.get("float_share") is None or float(source.get("float_share") or 0) <= 0:
            raise RuntimeError(f"TuShare 当日流通股本缺失: {ts_code} {target['trade_date']}")
        if source.get("turnover_rate") is None or float(source["turnover_rate"]) != 0:
            raise RuntimeError(f"目标行不是 TuShare 精度舍入零值: {ts_code} {target['trade_date']}")
        if not other or int(other.get("tradestatus") or 0) != 1:
            raise RuntimeError(f"BaoStock 未确认目标日为交易日: {ts_code} {target['trade_date']}")
        if float(other.get("volume") or 0) != float(target["volume"]):
            raise RuntimeError(f"BaoStock 与项目成交量不一致: {ts_code} {target['trade_date']}")
        if abs(float(other.get("amount") or 0) - float(target["amount"])) > 0.01:
            raise RuntimeError(f"BaoStock 与项目成交额不一致: {ts_code} {target['trade_date']}")

        full_precision = Decimal(str(target["volume"])) / (Decimal(str(source["float_share"])) * Decimal("100"))
        turnover = full_precision.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        if turnover <= 0:
            raise RuntimeError(f"复算结果低于数据库精度: {ts_code} {target['trade_date']}")
        eligible.append({
            "code": code, "trade_date": target["trade_date"], "old_turnover": target["turnover"],
            "new_turnover": str(turnover), "volume": float(target["volume"]),
            "amount": float(target["amount"]), "float_share_10k": float(source["float_share"]),
            "source_turnover_rate_rounded": float(source["turnover_rate"]),
            "source_status": target["source_status"],
            "baostock_volume": float(other["volume"]), "baostock_amount": float(other["amount"]),
            "baostock_turn_rounded": float(other["turn"]),
            "derived_full_precision": str(full_precision),
        })

manifest = {
    "status": "shadow_verified" if not args.publish else "validated_pending_publish",
    "source": "TuShare daily_basic.float_share plus daily_quotes.volume",
    "formula": "turnover_rate_percent = volume_shares / (float_share_10k * 100)",
    "source_documentation": "https://tushare.pro/document/2?doc_id=32",
    "source_versions": get_source_versions(),
    "observed_at": observed_at,
    "candidate_rows": len(targets),
    "eligible_rows": eligible,
    "source_checks": source_checks,
    "writes_data": False,
}
artifact = Path(__file__).resolve().parents[1] / "data/historical_repair/turnover_float_share" / f"{observed_at[:10]}_{datetime.datetime.now().strftime('%H%M%S')}.json"
save_json_atomic(manifest, artifact)

if args.publish:
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
                VALUES (%s,%s,%s,%s,'tushare_float_share_formula',NOW())
            """, (row["code"], row["trade_date"], row["old_turnover"], row["new_turnover"]))
            cursor.execute("""
                UPDATE daily_quote_turnover_unknown_audit
                SET new_turnover=%s,source_status='reconstructed_from_float_share',repaired_at=NOW()
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
