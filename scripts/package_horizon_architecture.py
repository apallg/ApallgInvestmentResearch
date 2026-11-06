import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import torch

from scripts.horizon_architecture_research import HORIZONS, OLD, OUTPUT, SEEDS, write_json


def choose_structure(horizon, base, improvement):
    basic = {(row["horizon"], row["comparison"]): row for row in base["comparisons"]}
    improved = {(row["horizon"], row["comparison"]): row for row in improvement["comparisons"]}
    stable = lambda name: basic[(horizon, name)]["stable_improvement"]
    choice = "N" if stable("N-G") else "G"
    if stable("F50-G") and stable("F50-N"):
        choice = "F50"
    if stable("FC-G") and stable("FC-N") and (choice != "F50" or stable("FC-F50")):
        choice = "FC"
    if stable("FR-G") and stable("FR-N") and stable("FR-FC"):
        choice = "FR"
    anchor = "G" if horizon in ("H5", "H10") else "N"
    if choice == anchor and improved[(horizon, "AX-A0")]["stable_improvement"]:
        choice = "AX"
        if all(improved[(horizon, name)]["stable_improvement"] for name in ("R-AX", "R-F", "R-E")):
            choice = "R"
    return choice


def model_path(horizon, seed, component):
    prefix = f"W8_{horizon}_{seed}"
    if component == "G":
        return (OUTPUT if horizon == "H5" else OLD) / (f"{prefix}_G.txt" if horizon == "H5" else f"{prefix}_sequence_lgb.txt")
    if component == "N":
        return (OUTPUT if horizon == "H5" else OLD) / (f"{prefix}_N.final.pt" if horizon == "H5" else f"{prefix}_N0.final.pt")
    if component == "AX":
        return OUTPUT / f"{prefix}_AX.{'txt' if horizon in ('H5', 'H10') else 'final.pt'}"
    if component == "R":
        return OUTPUT / f"{prefix}_R_aux.txt"
    raise ValueError(component)


def artifact(path, features):
    if not path.exists():
        raise RuntimeError(f"候选工件不存在: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".txt":
        model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        if model.num_feature() != len(features):
            raise RuntimeError(f"LGB特征数不符: {path}")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("daily_features") != len(features) + 1 or state.get("restore_max_abs_error", 1) > 1e-6:
            raise RuntimeError(f"LSTM输入或恢复审计不符: {path}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def package():
    base_path, improvement_path = OUTPUT / "analysis.json", OUTPUT / "improvement_analysis.json"
    if not base_path.exists() or not improvement_path.exists():
        raise RuntimeError("基础或改进分析缺失")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    improvement = json.loads(improvement_path.read_text(encoding="utf-8"))
    if base.get("status") != "evaluated" or improvement.get("status") != "evaluated":
        raise RuntimeError("全矩阵未评价，不能封装候选")
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    decisions = []
    for horizon in HORIZONS:
        choice = choose_structure(horizon, base, improvement)
        base_features = manifest["factor_keys"]
        x_features = None
        if choice in ("AX", "R"):
            from scripts.horizon_architecture_improvement import X_COLUMNS
            x_features = list(X_COLUMNS)
        features = base_features + (x_features or [])
        members = []
        for seed in SEEDS:
            paths = {}
            if choice in ("G", "N"):
                paths[choice] = artifact(model_path(horizon, seed, choice), features)
            elif choice in ("F50", "FC", "FR"):
                paths = {arm: artifact(model_path(horizon, seed, arm), features) for arm in ("G", "N")}
            else:
                paths["AX"] = artifact(model_path(horizon, seed, "AX"), features)
                if choice == "R":
                    paths["R"] = artifact(model_path(horizon, seed, "R"), features)
            weight = None
            if choice in ("FC", "FR"):
                meta = json.loads((OUTPUT / f"W8_{horizon}_{seed}_{choice}.json").read_text(encoding="utf-8"))
                weight = meta["weight"] if choice == "FC" else meta["weight_updates"]
            if choice == "R":
                meta = json.loads((OUTPUT / f"W8_{horizon}_{seed}_R_improve.json").read_text(encoding="utf-8"))
                weight = meta["residual_weight"]
            members.append({"seed": seed, "artifacts": paths, "weight": weight})
        decisions.append({"horizon": horizon, "structure": choice, "factor_keys": features,
            "members": members, "score_scale": "within-day full-universe percentile rank",
            "fallback": None, "historical_folds": "W1-W7", "sensitivity_fold": "W8",
            "candidate_package_loadable": True, "live_prediction_qualified": False,
            "shadow_eligible": choice != "G", "strict_pit_verified": False,
            "trade_execution_verified": False, "production_switch_allowed": False})
    result = {"schema_version": 1, "status": "research_candidate_packaged",
              "base_analysis_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
              "improvement_analysis_sha256": hashlib.sha256(improvement_path.read_bytes()).hexdigest(),
              "decisions": decisions, "production_model_unchanged": True}
    write_json(OUTPUT / "candidate_package.json", result)
    print(json.dumps({"status": result["status"], "structures": {row["horizon"]: row["structure"] for row in decisions}}, ensure_ascii=False))


if __name__ == "__main__":
    package()
