import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data import Database
from core.data.north_flow_migration import ensure_north_flow_quarantine
from core.shared.persistence import save_json_atomic


cutoff = datetime.date(2024, 8, 16)
ensure_north_flow_quarantine()
source = Database.fetchone(
    "SELECT COUNT(*) AS n FROM north_flow WHERE trade_date > %s AND code <> 'MARKET'",
    (cutoff,),
) or {}
source_rows = int(source.get("n") or 0)
Database.execute("""
    INSERT INTO north_flow_quarantine
        (code, trade_date, net_flow, quarantine_reason)
    SELECT code, trade_date, net_flow,
           'Invalid future-dated carry-forward: source daily individual holding data ends 2024-08-16'
    FROM north_flow
    WHERE trade_date > %s AND code <> 'MARKET'
    ON DUPLICATE KEY UPDATE net_flow=VALUES(net_flow)
""", (cutoff,))
verification = Database.fetchone("""
    SELECT COUNT(*) AS n
    FROM north_flow s
    LEFT JOIN north_flow_quarantine q ON q.code=s.code AND q.trade_date=s.trade_date
    WHERE s.trade_date > %s AND s.code <> 'MARKET'
      AND (q.code IS NULL OR NOT (q.net_flow <=> s.net_flow))
""", (cutoff,)) or {}
if int(verification.get("n") or 0) != 0:
    raise RuntimeError(f"North flow quarantine verification failed: {verification}")
archived = Database.fetchone(
    "SELECT COUNT(*) AS n FROM north_flow_quarantine WHERE trade_date > %s AND code <> 'MARKET'",
    (cutoff,),
) or {}
archived_rows = int(archived.get("n") or 0)
if archived_rows < source_rows:
    raise RuntimeError(f"North flow archive incomplete: source={source_rows}, archived={archived_rows}")
Database.execute("DELETE FROM north_flow WHERE trade_date > %s AND code <> 'MARKET'", (cutoff,))
remaining = Database.fetchone(
    "SELECT COUNT(*) AS n FROM north_flow WHERE trade_date > %s AND code <> 'MARKET'",
    (cutoff,),
) or {}
if int(remaining.get("n") or 0) != 0:
    raise RuntimeError(f"Invalid north flow rows remain: {remaining}")

artifact = {
    "status": "verified",
    "cutoff_last_valid_data_date": cutoff.isoformat(),
    "source_rows_found": source_rows,
    "source_rows_archived_total": archived_rows,
    "invalid_rows_remaining": int(remaining.get("n") or 0),
    "market_summary_rows_preserved": True,
    "reason": "AkShare individual holding series latest date is 2024-08-16; rows stamped after exchange disclosure change were not observations for those dates",
    "official_notice_url": "https://investor.szse.cn/szhk/hkbussiness/news/t20240726_608353.html",
    "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
output = ROOT / "data/historical_repair/north_flow_quarantine_20261024.json"
save_json_atomic(artifact, output)
print(json.dumps(artifact, ensure_ascii=False))
