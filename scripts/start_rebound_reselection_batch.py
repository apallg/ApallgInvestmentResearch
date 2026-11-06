"""创建可恢复的反弹卫星重筛任务，先用旧证据做候选预筛，不改生产池。"""
import hashlib
import json
import statistics
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data.connection import Database
from core.data.repos.factor_architecture import factor_architecture_repo


PARENT_RUN = "bdba67f459d2b928178s504cf"
OLD_REBOUND = {
    "a158_low20", "a158_resi20", "a158_roc20", "fund_earn_mom",
    "sales_growth_n8_ma_window20", "sales_growth_n4_ma_window20",
    "a191_rev_07", "north_buy_prop", "open_net_buy_intensity",
    "heavyweight_overweight_n10", "a158_ma5_20",
}


def _old_high_vol_scores():
    rows = Database.fetchall(
        "SELECT arm,fold_id,metrics_json FROM factor_set_evaluations "
        "WHERE run_id=%s AND arm LIKE %s",
        (PARENT_RUN, "residual:%:high_vol"),
    )
    grouped = {}
    for row in rows:
        metrics = json.loads(row.get("metrics_json") or "{}")
        key = metrics.get("factor_key") or str(row["arm"])[9:-9]
        value = metrics.get("oriented_ic")
        if value is not None:
            grouped.setdefault(key, []).append(float(value))
    return {
        key: (statistics.median(values), sum(value > 0 for value in values) / len(values), len(values))
        for key, values in grouped.items()
    }


def main():
    factor_architecture_repo.ensure_tables()
    parent = factor_architecture_repo.get_run(PARENT_RUN)
    if not parent or parent.get("status") != "completed":
        raise RuntimeError("旧架构运行不存在或未完成")
    candidates = factor_architecture_repo.list_candidates(PARENT_RUN)
    eligible = [item for item in candidates if item.get("status") == "eligible"]
    core_keys = set(next(
        item["factor_keys"] for item in factor_architecture_repo.list_sets(PARENT_RUN)
        if item.get("role") == "core" and item.get("status") == "validated"
    ))
    scores = _old_high_vol_scores()
    ranked = sorted(
        (item for item in eligible if item["factor_key"] not in core_keys and item["factor_key"] not in OLD_REBOUND),
        key=lambda item: scores.get(item["factor_key"], (-999.0, 0.0, 0)),
        reverse=True,
    )
    selected_keys = OLD_REBOUND | {item["factor_key"] for item in ranked[:85]}
    selected = [item for item in eligible if item["factor_key"] in selected_keys]
    rejected = [item for item in candidates if item.get("status") == "rejected"]
    rejected += [
        {**item, "reject_reason": "old_high_vol_prefilter"}
        for item in eligible if item["factor_key"] not in selected_keys
    ]

    run_id = f"rb10batch_{uuid.uuid4().hex[:10]}"
    config = dict(parent.get("config") or {})
    config.update({
        "profile": "rebound_reselection_tradable_batch_v1",
        "label_mode": "tradable_open_to_open",
        "label_horizon": 10,
        "selection_cutoff": "2025-06-30",
        "write_pool": False,
        "reselect_role": "rebound",
        "reselect_parent_run": PARENT_RUN,
        "candidate_prefilter": "old_high_vol_residual_top85_plus_old_rebound",
    })
    factor_architecture_repo.create_run(
        run_id, config, parent.get("pool_snapshot_hash") or "", config["selection_cutoff"],
    )
    factor_architecture_repo.save_candidates(run_id, selected, rejected)
    old_core = next(item for item in factor_architecture_repo.list_sets(PARENT_RUN)
                    if item.get("role") == "core" and item.get("status") == "validated")
    core_set_id = f"{run_id}_core_control"
    fingerprint = hashlib.sha256(json.dumps({"run_id": run_id, "keys": old_core["factor_keys"]}, sort_keys=True).encode()).hexdigest()
    factor_architecture_repo.save_set(
        core_set_id, run_id, "core", old_core["factor_keys"],
        {"label_mode": "tradable_open_to_open", "copied_from": old_core.get("set_id")},
        {"copied_from": old_core.get("set_id"), "control_only": True},
        fingerprint, version_no=1, status="validated",
    )
    print(json.dumps({
        "run_id": run_id,
        "candidate_count": len(selected),
        "old_rebound_count": len(OLD_REBOUND),
        "prefilter_count": len(selected) - len(OLD_REBOUND),
        "label_mode": config["label_mode"],
        "selection_cutoff": config["selection_cutoff"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
