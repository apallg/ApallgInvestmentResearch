import hashlib
import uuid
from pathlib import Path

import pandas as pd

from core.data.repos.historical_repair import HistoricalRepairRepo
from core.data.repos.quotes import _quote_payload_hash
from core.domain.data_quality.historical_alias_cleanup import assess_legacy_alias_row_cleanup
from core.shared.config import QLIB_DATA_DIR
from core.shared.persistence import load_json, load_parquet, now_iso, save_json_atomic, save_parquet
from backend.service.qlib_access import qlib_read_locked


def _root():
    return Path(__file__).resolve().parents[2] / "data/historical_repair/alias_cleanup"


def _manifest_path(release_id):
    if not release_id.isalnum() or len(release_id) != 12:
        raise ValueError("旧代码清理发布 ID 格式无效")
    return _root() / release_id / "manifest.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@qlib_read_locked
def prepare_legacy_alias_cleanup(old_code, trade_date):
    old_code = str(old_code).zfill(6)
    trade_date = str(trade_date)[:10]
    alias_catalog = load_json(Path(__file__).resolve().parents[2] / "data/historical_repair/identity/stock_code_aliases.json")
    aliases = [item for item in (alias_catalog or {}).get("aliases", []) if item.get("old_symbol") == old_code]
    if len(aliases) != 1:
        raise ValueError("没有唯一的已核验旧代码别名记录")
    alias = aliases[0]
    old_rows = HistoricalRepairRepo.get_shard_backup(old_code, trade_date, trade_date) or []
    canonical_rows = HistoricalRepairRepo.get_shard_backup(alias["canonical_symbol"], trade_date, trade_date) or []
    old_row = old_rows[0] if len(old_rows) == 1 else None
    canonical_row = canonical_rows[0] if len(canonical_rows) == 1 else None
    assessment = assess_legacy_alias_row_cleanup(alias, old_row, canonical_row)
    release_id = uuid.uuid4().hex[:12]
    release_dir = _manifest_path(release_id).parent
    release_dir.mkdir(parents=True, exist_ok=False)
    old_path = release_dir / "preimage_old_quote.parquet"
    canonical_path = release_dir / "preimage_canonical_quote.parquet"
    save_parquet(pd.DataFrame(old_rows), old_path)
    save_parquet(pd.DataFrame(canonical_rows), canonical_path)
    old_hash = _quote_payload_hash(old_row) if old_row else None
    canonical_hash = _quote_payload_hash(canonical_row) if canonical_row else None
    qlib_path = Path(QLIB_DATA_DIR) / "instruments" / "all.txt"
    if not qlib_path.is_file():
        assessment["blockers"].append("Qlib instruments/all.txt 不存在，无法核对旧代码是否仍属于模型池")
    else:
        qlib_codes = {line.split("\t", 1)[0].strip() for line in qlib_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        qlib_old_code = f"sz{old_code}"
        if qlib_old_code in qlib_codes:
            assessment["blockers"].append("旧代码仍在 Qlib all.txt 中，清理前必须处理 Qlib 特征")
    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "status": "ready" if assessment["status"] == "ready" and not assessment["blockers"] else "blocked",
        "created_at": now_iso(),
        "old_code": old_code,
        "canonical_code": alias["canonical_symbol"],
        "trade_date": trade_date,
        "alias": alias,
        "assessment": assessment,
        "preimage_old_path": str(old_path.resolve()),
        "preimage_old_sha256": _sha256(old_path),
        "preimage_old_payload_sha256": old_hash,
        "preimage_canonical_path": str(canonical_path.resolve()),
        "preimage_canonical_sha256": _sha256(canonical_path),
        "preimage_canonical_payload_sha256": canonical_hash,
        "official_database_writes": False,
        "qlib_writes": False,
        "publishable": assessment["status"] == "ready" and not assessment["blockers"],
    }
    save_json_atomic(manifest, _manifest_path(release_id))
    return manifest


def apply_legacy_alias_cleanup(release_id):
    path = _manifest_path(release_id)
    manifest = load_json(path)
    if not manifest or manifest.get("release_id") != release_id or manifest.get("status") != "ready" or not manifest.get("publishable"):
        raise RuntimeError("旧代码清理未通过发布预检")
    old_path = Path(manifest["preimage_old_path"])
    canonical_path = Path(manifest["preimage_canonical_path"])
    if _sha256(old_path) != manifest["preimage_old_sha256"] or _sha256(canonical_path) != manifest["preimage_canonical_sha256"]:
        raise RuntimeError("旧代码清理 pre-image 哈希不符")
    old_rows = HistoricalRepairRepo.get_shard_backup(manifest["old_code"], manifest["trade_date"], manifest["trade_date"]) or []
    canonical_rows = HistoricalRepairRepo.get_shard_backup(manifest["canonical_code"], manifest["trade_date"], manifest["trade_date"]) or []
    if len(old_rows) != 1 or len(canonical_rows) != 1:
        raise RuntimeError("正式库旧代码行或规范代码对照行数量已变化")
    if _quote_payload_hash(old_rows[0]) != manifest["preimage_old_payload_sha256"]:
        raise RuntimeError("正式库旧代码行指纹已变化，拒绝清理")
    if _quote_payload_hash(canonical_rows[0]) != manifest["preimage_canonical_payload_sha256"]:
        raise RuntimeError("正式库规范代码行指纹已变化，拒绝清理")
    provenance = {
        "batch_id": release_id,
        "task_id": f"alias-cleanup-{release_id}",
        "source": "tushare+baostock",
        "provider_version": "tushare+baostock",
        "adjustment_mode": "qfq",
        "anchor_date": manifest["trade_date"],
        "input_payload_sha256": manifest["preimage_old_sha256"],
    }
    manifest["status"] = "publishing"
    manifest["publish_started_at"] = now_iso()
    save_json_atomic(manifest, path)
    result = HistoricalRepairRepo.delete_legacy_alias_row(
        manifest["old_code"], manifest["trade_date"], manifest["canonical_code"],
        manifest["preimage_old_payload_sha256"], manifest["preimage_canonical_payload_sha256"], provenance,
    )
    manifest["deleted_payload_sha256"] = result["deleted_hash"]
    manifest["status"] = "published"
    manifest["official_database_writes"] = True
    manifest["published_at"] = now_iso()
    save_json_atomic(manifest, path)
    return manifest


def rollback_legacy_alias_cleanup(release_id):
    path = _manifest_path(release_id)
    manifest = load_json(path)
    if not manifest or manifest.get("release_id") != release_id or manifest.get("status") != "published":
        raise ValueError("只有已发布的旧代码清理可以回滚")
    if _sha256(manifest["preimage_old_path"]) != manifest["preimage_old_sha256"]:
        raise RuntimeError("旧代码清理 pre-image 哈希不符，拒绝回滚")
    old_frame = load_parquet(manifest["preimage_old_path"])
    canonical_rows = HistoricalRepairRepo.get_shard_backup(manifest["canonical_code"], manifest["trade_date"], manifest["trade_date"]) or []
    if len(old_frame) != 1 or len(canonical_rows) != 1:
        raise RuntimeError("回滚所需 pre-image 或规范代码对照行数量异常")
    provenance = {
        "batch_id": release_id,
        "task_id": f"alias-rollback-{release_id}",
        "source": "tushare+baostock",
        "provider_version": "tushare+baostock",
        "adjustment_mode": "qfq",
        "anchor_date": manifest["trade_date"],
        "input_payload_sha256": manifest["preimage_old_sha256"],
        "canonical_code": manifest["canonical_code"],
    }
    manifest["status"] = "rolling_back"
    manifest["rollback_started_at"] = now_iso()
    save_json_atomic(manifest, path)
    result = HistoricalRepairRepo.restore_legacy_alias_row(
        old_frame.iloc[0].to_dict(), manifest["preimage_canonical_payload_sha256"], provenance,
    )
    manifest["status"] = "rolled_back"
    manifest["official_database_writes"] = False
    manifest["rollback_payload_sha256"] = result["restored_hash"]
    manifest["rolled_back_at"] = now_iso()
    save_json_atomic(manifest, path)
    return manifest
