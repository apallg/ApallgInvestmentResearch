import hashlib
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data.connection import Database
from core.shared.persistence import save_json_atomic


table = "analyst_forecast"
archive = "analyst_forecast_semantic_quarantine_20261024"
manifest_path = Path("data/historical_repair/analyst_forecast_quarantine_20261024.json")
manifest_path.parent.mkdir(parents=True, exist_ok=True)


def _digest(rows):
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


source_rows = Database.fetchall(f"SELECT * FROM {table} ORDER BY id")
source_count = len(source_rows)
source_sha256 = _digest(source_rows)
Database.execute(f"CREATE TABLE IF NOT EXISTS {archive} LIKE {table}")


def _quarantine(cursor):
    cursor.execute(f"INSERT IGNORE INTO {archive} SELECT * FROM {table}")
    cursor.execute(f"SELECT COUNT(*) AS rows_count FROM {archive}")
    archived = int(cursor.fetchone()["rows_count"])
    if archived != source_count:
        raise RuntimeError(f"归档行数不匹配：源 {source_count}，归档 {archived}")
    cursor.execute(f"DELETE FROM {table}")
    if int(cursor.rowcount) != source_count:
        raise RuntimeError(f"活动表清空行数不匹配：预期 {source_count}，实际 {cursor.rowcount}")
    return archived


if source_count:
    archived_count = Database.transaction(_quarantine)
else:
    archived_count = int((Database.fetchone(f"SELECT COUNT(*) AS rows_count FROM {archive}") or {}).get("rows_count") or 0)

archive_rows = Database.fetchall(f"SELECT * FROM {archive} ORDER BY id")
archive_sha256 = _digest(archive_rows)
active_count = int((Database.fetchone(f"SELECT COUNT(*) AS rows_count FROM {table}") or {}).get("rows_count") or 0)
if archived_count != len(archive_rows) or archive_sha256 != source_sha256 or active_count != 0:
    raise RuntimeError("隔离后归档指纹或活动表状态核验失败")

manifest = {
    "status": "quarantined",
    "reason": "历史 analyst_forecast 记录语义来自业绩快报，不是分析师一致预期；不再作为有效预测数据使用",
    "source_table": table,
    "archive_table": archive,
    "rows": archived_count,
    "source_sha256": source_sha256,
    "archive_sha256": archive_sha256,
    "active_rows_after": active_count,
}
save_json_atomic(manifest, manifest_path)
print(manifest)
