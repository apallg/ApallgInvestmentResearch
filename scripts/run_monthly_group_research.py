import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_multiscale_group_research import sha256

SOURCE = Path(os.environ.get("MONTHLY_GROUP_SOURCE", ROOT / "data/experiments/multiscale_group_research_20261103/current_isolated_input"))
REFERENCE = ROOT / "data/experiments/multiscale_group_research_current_v2_20261103"
OUTPUT = Path(os.environ.get("MONTHLY_GROUP_OUTPUT", ROOT / "data/experiments/monthly_group_research_v4_20261103"))
SEEDS = (42, 2026, 3407)
ARMS = ("D", "L", "DM", "LM", "U0", "UM", "SL4_0", "PB4_0", "IC6_0", "GS8_0",
        "SL4_M", "PB4_M", "IC6_M", "GS8_M")
COMPARISONS = (("DM", "D"), ("LM", "L"), ("U0", "D"), ("UM", "U0"),
               *((f"{name}_0", "U0") for name in ("SL4", "PB4", "IC6", "GS8")),
               *((f"{name}_M", "UM") for name in ("SL4", "PB4", "IC6", "GS8")))


def preflight():
    reference = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    if reference["status"] != "preflight_passed":
        raise RuntimeError("上一轮冻结清单状态异常")
    pointer_path = "data\\qlib_data\\current.json"
    for path, expected in reference["source_files_sha256"].items():
        if path == pointer_path:
            continue
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"冻结输入发生变化: {path}")
    original_pointer = json.loads((SOURCE / "input_identity.json").read_text(encoding="utf-8"))["qlib_pointer_sha256"]
    if original_pointer != reference["source_files_sha256"][pointer_path]:
        raise RuntimeError("隔离面板对应的Qlib代际签名不符")
    current_pointer = sha256(ROOT / pointer_path)
    preheat = OUTPUT / "preheat_2018.parquet"
    preheat_audit = OUTPUT / "preheat_audit.json"
    if not preheat.exists() or not preheat_audit.exists():
        raise RuntimeError("缺少已验证的2018年月线预热快照")
    audited = json.loads(preheat_audit.read_text(encoding="utf-8"))
    if audited["status"] != "passed" or audited["preheat_sha256"] != sha256(preheat):
        raise RuntimeError("2018年预热快照哈希或审计状态异常")
    snapshot = OUTPUT / "classification_snapshot.parquet"
    snapshot_audit = OUTPUT / "classification_audit.json"
    if not snapshot.exists() or not snapshot_audit.exists():
        raise RuntimeError("缺少全历史统一分类快照")
    classified = json.loads(snapshot_audit.read_text(encoding="utf-8"))
    if classified["snapshot_sha256"] != sha256(snapshot):
        raise RuntimeError("分类快照哈希与审计不符")
    budget = OUTPUT / "budget_control.json"
    if not budget.exists() or json.loads(budget.read_text(encoding="utf-8"))["status"] != "frozen_before_formal_outer_analysis":
        raise RuntimeError("缺少预跑后冻结的训练时间预算对照")
    membership = REFERENCE / "stock_index_membership_snapshot.json"
    if sha256(membership) != reference["membership_snapshot_sha256"]:
        raise RuntimeError("历史行业归属快照发生变化")
    if SOURCE.resolve() != (ROOT / "data/experiments/multiscale_group_research_20261103/current_isolated_input").resolve():
        raise RuntimeError("当前命令必须使用已冻结的隔离输入")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / membership.name
    if target.exists() and sha256(target) != sha256(membership):
        raise RuntimeError("实验目录已有不匹配的行业快照")
    shutil.copy2(membership, target)
    paths = [ROOT / f"scripts/{name}" for name in ("monthly_group_features.py", "train_monthly_group_research.py",
                                                     "analyze_monthly_group_research.py", "run_monthly_group_research.py",
                                                     "assess_monthly_group_research.py", "extract_monthly_group_preheat.py",
                                                     "prepare_monthly_group_snapshots.py", "freeze_monthly_group_budget.py")]
    manifest = {
        "status": "preflight_passed", "research_only": True, "strict_pit_verified": False,
        "source_panel_end": reference["source_panel_end"],
        "source_files_sha256": {name: value for name, value in reference["source_files_sha256"].items() if name != pointer_path},
        "original_qlib_pointer_sha256": original_pointer, "current_qlib_pointer_sha256_at_preflight": current_pointer,
        "qlib_pointer_drift": current_pointer != original_pointer,
        "membership_snapshot_sha256": sha256(target), "factor_keys": reference["factor_keys"],
        "preheat_sha256": sha256(preheat), "preheat_generation": audited["generation"],
        "classification_snapshot_sha256": sha256(snapshot),
        "budget_control_sha256": sha256(budget),
        "folds": reference["folds"], "seeds": list(SEEDS), "arms": list(ARMS),
        "comparisons": [f"{left}-{right}" for left, right in COMPARISONS],
        "interaction_names": [f"{name}_interaction" for name in ("SL4", "PB4", "IC6", "GS8")],
        "candidate_gate": {"minimum_bp": 10, "positive_full_folds": 5, "positive_seeds": 2},
        "primary_folds": [f"W{i}" for i in range(1, 8)], "sensitivity_fold": "W8",
        "membership_pit_verified": False, "execution_eligible": False,
        "code_files_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
    }
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("实验清单变化；不可覆盖或续跑")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(path), "arms": len(ARMS)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "train", "analyze", "assess", "confirm"))
    parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if set(args.folds) - {f"W{i}" for i in range(1, 9)} or set(args.seeds) - set(SEEDS):
        raise ValueError("折或种子不在冻结契约内")
    if args.stage == "preflight":
        preflight()
        return
    if args.stage == "confirm":
        raise RuntimeError("融合确认必须等待完整 LightGBM 候选筛选和预算对照配置冻结")
    command = [sys.executable, "-B", str(ROOT / f"scripts/{'train' if args.stage == 'train' else 'analyze'}_monthly_group_research.py"),
               "--folds", *args.folds, "--seeds", *map(str, args.seeds)]
    if args.stage == "assess":
        command = [sys.executable, "-B", str(ROOT / "scripts/assess_monthly_group_research.py")]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
