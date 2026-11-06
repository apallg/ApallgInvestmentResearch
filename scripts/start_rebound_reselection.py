"""创建一个只重跑反弹卫星残差筛选的研究任务。"""
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data.repos.factor_architecture import factor_architecture_repo


PARENT_RUN = "bdba67f459d2b928178s504cf"


def main():
    factor_architecture_repo.ensure_tables()
    parent = factor_architecture_repo.get_run(PARENT_RUN)
    if not parent or parent.get("status") != "completed":
        raise RuntimeError("旧核心/卫星架构运行不存在或未完成")
    for row in factor_architecture_repo.list_runs(100):
        config = row.get("config") or {}
        if config.get("profile") == "rebound_reselection_tradable_v1" and row.get("status") in {
            "pending", "running",
        }:
            print(json.dumps({"run_id": row["run_id"], "status": row["status"], "reused": True}))
            return

    old_sets = factor_architecture_repo.list_sets(PARENT_RUN)
    old_core = next(item for item in old_sets if item.get("role") == "core" and item.get("status") == "validated")
    candidates = factor_architecture_repo.list_candidates(PARENT_RUN)
    eligible = [item for item in candidates if item.get("status") == "eligible"]
    rejected = [item for item in candidates if item.get("status") == "rejected"]
    run_id = f"rb10trad_{uuid.uuid4().hex[:12]}"
    config = dict(parent.get("config") or {})
    config.update({
        "profile": "rebound_reselection_tradable_v1",
        "label_mode": "tradable_open_to_open",
        "label_horizon": 10,
        "selection_cutoff": "2025-06-30",
        "write_pool": False,
        "reselect_role": "rebound",
        "reselect_parent_run": PARENT_RUN,
    })
    factor_architecture_repo.create_run(
        run_id, config, parent.get("pool_snapshot_hash") or "", config["selection_cutoff"],
    )
    factor_architecture_repo.save_candidates(run_id, eligible, rejected)
    core_set_id = f"{run_id}_core_control"
    core_config = dict(old_core.get("selection_config") or {})
    core_config.update({"label_mode": "tradable_open_to_open", "copied_from": old_core.get("set_id")})
    core_metrics = dict(old_core.get("metrics") or {})
    core_metrics.update({"copied_from": old_core.get("set_id"), "control_only": True})
    fingerprint = hashlib.sha256(json.dumps({"run_id": run_id, "role": "core", "keys": old_core["factor_keys"]}, sort_keys=True).encode()).hexdigest()
    factor_architecture_repo.save_set(
        core_set_id, run_id, "core", old_core["factor_keys"], core_config,
        core_metrics, fingerprint, version_no=1, status="validated",
    )
    print(json.dumps({
        "run_id": run_id,
        "parent_run": PARENT_RUN,
        "core_set_id": core_set_id,
        "eligible_candidates": len(eligible),
        "label_mode": config["label_mode"],
        "selection_cutoff": config["selection_cutoff"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
