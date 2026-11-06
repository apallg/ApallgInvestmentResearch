import argparse
import datetime
import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import Database
from core.data.sources.historical_quotes import get_source_versions
from core.data.sources.tushare import _rate_limit, get_pro
from core.shared.persistence import save_json_atomic


parser = argparse.ArgumentParser()
parser.add_argument("--publish", action="store_true")
args = parser.parse_args()


def _safe_float(value):
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

targets = Database.fetchall("""
    SELECT q.code,CAST(q.trade_date AS CHAR) AS trade_date,q.volume,q.amount,q.turnover,a.new_turnover
    FROM daily_quotes q
    JOIN daily_quote_turnover_repair_audit a ON a.code=q.code AND a.trade_date=q.trade_date
    WHERE a.source_table='tushare_float_share_formula'
      AND q.turnover=a.new_turnover
    ORDER BY q.code,q.trade_date
""")
if not targets:
    print(json.dumps({"status": "no_candidates", "restored": 0}, ensure_ascii=False), flush=True)
    raise SystemExit(0)

pro = get_pro()
if not pro:
    raise RuntimeError("TuShare 不可用")

by_code = {}
for row in targets:
    by_code.setdefault(row["code"], []).append(row)

source_rows = []
for code, rows in by_code.items():
    start = (datetime.date.fromisoformat(min(row["trade_date"] for row in rows)) - datetime.timedelta(days=15)).isoformat()
    end = (datetime.date.fromisoformat(max(row["trade_date"] for row in rows)) + datetime.timedelta(days=15)).isoformat()
    ts_code = f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"
    _rate_limit()
    frame = pro.daily_basic(
        ts_code=ts_code, start_date=start.replace("-", ""), end_date=end.replace("-", ""),
        fields="ts_code,trade_date,total_mv,circ_mv,float_share,pe,pb,turnover_rate",
    )
    lookup = {} if frame is None or frame.empty else {
        str(row["trade_date"]): row for row in frame.to_dict("records")
    }
    for target in rows:
        raw = lookup.get(target["trade_date"].replace("-", ""))
        if not raw or raw.get("float_share") is None or float(raw["float_share"]) <= 0:
            raise RuntimeError(f"TuShare daily_basic 缺少有效股本: {ts_code} {target['trade_date']}")
        if raw.get("turnover_rate") is None or float(raw["turnover_rate"]) != 0:
            raise RuntimeError(f"TuShare 原始换手率与本次精度恢复契约不符: {ts_code} {target['trade_date']}")
        computed = Decimal(str(target["volume"])) / (Decimal(str(raw["float_share"])) * Decimal("100"))
        if computed.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) != Decimal(str(target["new_turnover"])):
            raise RuntimeError(f"TuShare 股本与已发布换手率不一致: {ts_code} {target['trade_date']}")
        source_rows.append({
            "code": code, "trade_date": target["trade_date"],
            "total_mv": _safe_float(raw.get("total_mv")), "circ_mv": _safe_float(raw.get("circ_mv")),
            "float_share": _safe_float(raw.get("float_share")), "pe": _safe_float(raw.get("pe")),
            "pb": _safe_float(raw.get("pb")), "turnover_rate": _safe_float(raw.get("turnover_rate")),
            "quote_volume": _safe_float(target["volume"]), "quote_amount": _safe_float(target["amount"]),
            "published_turnover": str(target["new_turnover"]),
        })

observed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest = {
    "status": "shadow_verified" if not args.publish else "validated_pending_publish",
    "source": "TuShare daily_basic historical rows",
    "source_versions": get_source_versions(),
    "observed_at": observed_at,
    "candidate_rows": len(targets),
    "source_rows": source_rows,
    "writes_data": False,
}
artifact = Path(__file__).resolve().parents[1] / "data/historical_repair/daily_basic_turnover_source" / f"{observed_at[:10]}_{datetime.datetime.now().strftime('%H%M%S')}.json"
save_json_atomic(manifest, artifact)

if args.publish:
    def _publish(cursor):
        for row in source_rows:
            cursor.execute("""
                INSERT INTO daily_basic (code,trade_date,total_mv,circ_mv,float_share,pe,pb,turnover_rate)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    total_mv=COALESCE(VALUES(total_mv),total_mv),
                    circ_mv=COALESCE(VALUES(circ_mv),circ_mv),
                    float_share=COALESCE(VALUES(float_share),float_share),
                    pe=COALESCE(VALUES(pe),pe),
                    pb=COALESCE(VALUES(pb),pb),
                    turnover_rate=VALUES(turnover_rate)
            """, (
                row["code"], row["trade_date"], row["total_mv"], row["circ_mv"], row["float_share"],
                row["pe"], row["pb"], row["turnover_rate"],
            ))
        return len(source_rows)

    manifest["published_rows"] = Database.transaction(_publish)
    manifest["status"] = "published"
    manifest["writes_data"] = True
    save_json_atomic(manifest, artifact)

print(json.dumps({"manifest": str(artifact), **manifest}, ensure_ascii=False, default=str), flush=True)
