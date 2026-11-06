import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from backend.service.horizon_candidate import predict_horizon_candidate
from scripts.horizon_architecture_research import OUTPUT, write_json


def command(script, stage=None, args=()):
    values = [sys.executable, str(ROOT / "scripts" / script)]
    if stage:
        values.append(stage)
    values.extend(args)
    subprocess.run(values, cwd=ROOT, check=True)


def shadow(args):
    package_path = OUTPUT / "candidate_package.json"
    if not package_path.exists():
        raise RuntimeError("候选包尚未形成")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    profiles = [row for row in package["decisions"] if row["horizon"] == args.horizon]
    if len(profiles) != 1 or not profiles[0]["shadow_eligible"]:
        raise RuntimeError("该期限尚无达到研究门槛的影子候选")
    date = pd.Timestamp(args.date)
    if date <= pd.Timestamp("2026-09-15"):
        raise RuntimeError("影子日期必须晚于已接触的历史研究测试期")
    feature_path, denominator_path = Path(args.feature_panel), Path(args.denominator)
    calendar_path = Path(args.calendar)
    if not feature_path.exists() or not denominator_path.exists() or not calendar_path.exists():
        raise RuntimeError("影子输入或统一交易日历缺失")
    features = pd.read_parquet(feature_path).sort_index()
    denominator = pd.read_parquet(denominator_path).sort_index()
    if features.index.get_level_values("datetime").max() > date:
        raise RuntimeError("影子特征面板包含决策日之后的样本")
    expected_index = denominator.index[denominator.index.get_level_values("datetime") == date]
    if len(expected_index) < 50 or expected_index.has_duplicates:
        raise RuntimeError("影子共同股票池不足或重复")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text(encoding="utf-8").splitlines()))
    score, metadata = predict_horizon_candidate(package_path, args.horizon, features, expected_index, calendar)
    output_dir = OUTPUT / "shadow" / args.date / args.horizon
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions.parquet"
    signatures = {name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in
                  (("package", package_path), ("features", feature_path), ("denominator", denominator_path), ("calendar", calendar_path))}
    meta_path = output_dir / "audit.json"
    if path.exists() or meta_path.exists():
        if not path.exists() or not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")).get("signatures") != signatures:
            raise RuntimeError("影子记录已存在且签名冲突")
        return
    temp = output_dir / "predictions.tmp"
    score.to_frame("score").to_parquet(temp, compression="zstd")
    temp.replace(path)
    write_json(meta_path, {"status": "prediction_recorded", "date": args.date, "horizon": args.horizon,
        "rows": len(score), "signatures": signatures, "metadata": metadata,
        "execution_recorded": False, "label_matured": False, "production_model_unchanged": True})
    print(json.dumps({"status": "prediction_recorded", "date": args.date, "horizon": args.horizon,
                      "rows": len(score)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "freeze", "repair", "prerun", "train", "evaluate",
                                          "improve-prerun", "improve-budget", "improve-train", "improve-evaluate",
                                          "package", "shadow", "status", "run-all"))
    parser.add_argument("--horizons", nargs="+", choices=("H5", "H10", "H20"))
    parser.add_argument("--folds", nargs="+", choices=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 2026, 3407))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--horizon", choices=("H5", "H10", "H20"))
    parser.add_argument("--date")
    parser.add_argument("--feature-panel")
    parser.add_argument("--denominator")
    parser.add_argument("--calendar", default=str(ROOT / "data/qlib_data/calendars/day.txt"))
    args = parser.parse_args()
    flags = []
    if args.horizons:
        flags += ["--horizons", *args.horizons]
    if args.folds:
        flags += ["--folds", *args.folds]
    if args.seeds:
        flags += ["--seeds", *map(str, args.seeds)]
    if args.resume:
        flags.append("--resume")
    if args.stage in ("preflight", "freeze", "prerun", "train"):
        command("horizon_architecture_research.py", args.stage, flags)
    elif args.stage == "repair":
        command("audit_horizon_architecture.py", args=flags)
    elif args.stage == "evaluate":
        command("analyze_horizon_architecture.py", "evaluate")
    elif args.stage in ("improve-prerun", "improve-budget", "improve-train"):
        mapped = {"improve-prerun": "prerun", "improve-budget": "budget", "improve-train": "train"}[args.stage]
        command("horizon_architecture_improvement.py", mapped, flags)
    elif args.stage == "improve-evaluate":
        command("analyze_horizon_improvement.py")
    elif args.stage == "package":
        command("package_horizon_architecture.py")
    elif args.stage == "shadow":
        if not all((args.horizon, args.date, args.feature_panel, args.denominator)):
            raise RuntimeError("影子记录需要期限、日期、特征面板和共同分母")
        shadow(args)
    elif args.stage == "status":
        command("horizon_architecture_research.py", "status")
        command("horizon_architecture_workflow.py", "status")
    else:
        for script, stage in (("horizon_architecture_research.py", "preflight"),
                              ("horizon_architecture_research.py", "freeze"),
                              ("horizon_architecture_research.py", "prerun"),
                              ("horizon_architecture_research.py", "train"),
                              ("audit_horizon_architecture.py", None),
                              ("analyze_horizon_architecture.py", "evaluate"),
                              ("horizon_architecture_improvement.py", "prerun"),
                              ("horizon_architecture_improvement.py", "budget"),
                              ("horizon_architecture_improvement.py", "train"),
                              ("analyze_horizon_improvement.py", None),
                              ("package_horizon_architecture.py", None)):
            command(script, stage, ["--resume"] if script.endswith("improvement.py") and stage in ("prerun", "train") else ())


if __name__ == "__main__":
    main()
