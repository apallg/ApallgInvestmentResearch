import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_horizon_architecture import read_frame
from scripts.horizon_architecture_research import OUTPUT as PREVIOUS, SEEDS
from scripts.model_complementarity_research import paired_block_intervals


OUTPUT = Path(os.environ.get("H20_CONFIRMATION_OUTPUT", "G:/AShareResearch/h20_confirmation_v1"))
ARMS = ("G", "N", "F75", "F50")
COMPARISONS = (("N", "G"), ("F75", "N"), ("F75", "G"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def sorted_stocks(frame):
    stock = frame.index.get_level_values("instrument").str.lower().to_numpy()
    order = np.lexsort((stock, -frame["score"].to_numpy(dtype=float)))
    return frame.iloc[order]


def historical():
    manifest_path = PREVIOUS / "manifest.json"
    analysis_path = PREVIOUS / "analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "evaluated" or analysis.get("completed") != 432:
        raise RuntimeError("旧基础矩阵未通过完整评价")
    if shutil.disk_usage(OUTPUT.anchor).free < 10 * 1024 ** 3:
        raise RuntimeError("G盘空间低于10GiB")
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    fold_values = {f"{candidate}-{baseline}": [] for candidate, baseline in COMPARISONS}
    folds = []
    for number in range(1, 9):
        fold = f"W{number}"
        first = None
        by_arm = {arm: [] for arm in ARMS}
        for seed in SEEDS:
            g = read_frame(fold, "H20", seed, "G", manifest)
            n = read_frame(fold, "H20", seed, "N", manifest)
            if not g.index.equals(n.index) or not np.array_equal(g["label"].to_numpy(), n["label"].to_numpy()):
                raise RuntimeError(f"G/N样本或标签不一致: {fold}/{seed}")
            if first is not None and (not first.index.equals(g.index) or not np.array_equal(first["label"].to_numpy(), g["label"].to_numpy())):
                raise RuntimeError(f"种子样本或标签不一致: {fold}/{seed}")
            first = g
            for arm, values in (("G", g["score"]), ("N", n["score"]),
                                ("F75", 0.25 * g["score"] + 0.75 * n["score"]),
                                ("F50", 0.5 * g["score"] + 0.5 * n["score"])):
                by_arm[arm].append(values.to_numpy(dtype=np.float64))
        start, end = map(pd.Timestamp, manifest["folds"][fold]["test"])
        days = calendar[(calendar >= start) & (calendar <= end)][::20]
        metrics = {}
        picks = {}
        rank_ic = {}
        for arm in ARMS:
            frame = first.copy()
            frame["score"] = np.mean(by_arm[arm], axis=0)
            rank_ic[arm] = float(frame.groupby(level="datetime", sort=True).apply(
                lambda part: part["score"].corr(part["label_rank"], method="spearman")).mean())
            rows = []
            picks[arm] = {}
            for day in days:
                part = frame.loc[frame.index.get_level_values("datetime") == day]
                if len(part) < 50 or part["score"].isna().any() or not np.isfinite(part["score"].to_numpy()).all():
                    raise RuntimeError(f"{fold}/{arm}/{day.date()} 股票池或分数无效")
                chosen = sorted_stocks(part)
                codes = chosen.index.get_level_values("instrument")[:5].tolist()
                picks[arm][str(day.date())] = codes
                rows.append({"date": str(day.date()), "top5": float(chosen["label"].iloc[:5].mean()),
                             "top20": float(chosen["label"].iloc[:20].mean()),
                             "top50": float(chosen["label"].iloc[:50].mean()),
                             "pool": float(part["label"].mean())})
            metrics[arm] = rows
        overlap = [len(set(picks["F75"][day]) & set(picks["N"][day])) / 5 for day in picks["N"]]
        switches = []
        for day in picks["N"]:
            added = sorted(set(picks["F75"][day]) - set(picks["N"][day]))
            removed = sorted(set(picks["N"][day]) - set(picks["F75"][day]))
            contribution = (sum(float(first.loc[(pd.Timestamp(day), code), "label"]) for code in added)
                            - sum(float(first.loc[(pd.Timestamp(day), code), "label"]) for code in removed)) / 5
            switches.append({"date": day, "added": added, "removed": removed,
                             "top5_diff_from_switch": contribution})
        fold_result = {"fold": fold, "arms": metrics, "picks": picks, "mean_rank_ic": rank_ic,
                       "F75_N_top5_overlap": float(np.mean(overlap)), "F75_N_switches": switches}
        folds.append(fold_result)
        if number <= 7:
            for candidate, baseline in COMPARISONS:
                key = f"{candidate}-{baseline}"
                fold_values[key].append(np.array([a["top5"] - b["top5"] for a, b in
                    zip(metrics[candidate], metrics[baseline])], dtype=float))
    comparisons = []
    for name, values in fold_values.items():
        intervals = paired_block_intervals(values, (2, 3, 4), family_size=3)
        intervals = {key: {"ordinary_95": value["ordinary_95"],
                            "bonferroni_95_family3_exploratory": value["bonferroni_95_family26"]}
                     for key, value in intervals.items()}
        old = next((row for row in analysis["comparisons"] if row["horizon"] == "H20"
                    and row["comparison"] == name), None)
        comparisons.append({"comparison": name, "mean_top5_diff_bp": float(np.concatenate(values).mean() * 10000),
                            "folds_positive": int(sum(v.mean() > 0 for v in values)),
                            "exploratory_intervals": intervals,
                            "original_35_comparison": old})
    result = {"status": "historical_recalculated", "folds": folds, "comparisons": comparisons,
              "previous_manifest_sha256": digest(manifest_path), "previous_analysis_sha256": digest(analysis_path),
              "historical_neutralization": "unqualified: historical industry PIT and market-cap vintage not certified",
              "forward_confirmation": False, "production_qualified": False}
    save_json(OUTPUT / "historical.json", result)
    print(json.dumps({"status": result["status"], "comparisons": comparisons}, ensure_ascii=False, default=str))


def preflight():
    source = ROOT / "data/experiments/multiscale_group_research_20261103/current_isolated_input/context/coverage.json"
    coverage = json.loads(source.read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    result = {"status": "historical_ready" if (PREVIOUS / "analysis.json").exists() else "historical_missing",
              "isolated_input_last_date": coverage["last_date"], "calendar_last_date": str(calendar.max().date()),
              "historical_industry_pit_verified": bool(coverage["industry_pit_verified"]),
              "historical_denominator_verified": bool(coverage["history_denominator_verified"]),
              "forward_input_ready": False,
              "forward_input_requirement": "fresh decision-date feature panel, full denominator, PIT market cap/industry and complete calendar",
              "C_free_gib": round(shutil.disk_usage("C:/").free / 1024 ** 3, 2),
              "G_free_gib": round(shutil.disk_usage("G:/").free / 1024 ** 3, 2)}
    print(json.dumps(result, ensure_ascii=False))


def freeze():
    historical_path = OUTPUT / "historical.json"
    if not historical_path.exists():
        raise RuntimeError("先完成历史复核")
    previous = json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))
    result = {"status": "protocol_frozen_pending_forward_inputs", "horizon": 20,
              "arms": {"G": 0.0, "N": 1.0, "F75": 0.75, "F50": 0.5},
              "primary_comparisons": [f"{a}-{b}" for a, b in COMPARISONS],
              "seeds": list(SEEDS), "factor_keys": previous["factor_keys"],
              "lightgbm_parameters": previous["lightgbm_parameters"],
              "train_months": 36, "valid_months": 3, "calibrate_months": 4,
              "retrain_trading_days": 63, "observation_weeks": 52, "quality_check_weeks": 26,
              "formation_trading_days": 20, "label": "Open(t+21)/Open(t+1)-1",
              "tie_break": "score descending, normalized stock code ascending",
              "historical_sha256": digest(historical_path), "previous_manifest_sha256": digest(PREVIOUS / "manifest.json"),
              "code_sha256": digest(Path(__file__)), "research_observation_eligible": False,
              "data_qualified": False, "trade_qualified": False, "production_qualified": False,
              "observation_start": None}
    target = OUTPUT / "protocol.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != result:
        raise RuntimeError("已冻结方案签名冲突")
    save_json(target, result)
    print(json.dumps({"status": result["status"], "path": str(target)}, ensure_ascii=False))


def status():
    paths = {name: OUTPUT / name for name in ("historical.json", "protocol.json", "observation.json")}
    from scripts.h20_confirmation_forward import quality_status
    print(json.dumps({"files": {name: path.exists() for name, path in paths.items()},
                      "forward_records": len(list((OUTPUT / "records").glob("*/predictions.json"))) if (OUTPUT / "records").exists() else 0,
                      "quality": quality_status()}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "historical", "freeze", "retrain", "record", "evaluate", "status"))
    args, unknown = parser.parse_known_args()
    if unknown and args.stage not in ("retrain", "record", "evaluate"):
        parser.error(f"不支持的参数: {unknown}")
    if args.stage == "preflight":
        preflight()
    elif args.stage == "historical":
        historical()
    elif args.stage == "freeze":
        freeze()
    elif args.stage == "status":
        status()
    else:
        from scripts.h20_confirmation_forward import main as forward_main
        forward_main()


if __name__ == "__main__":
    main()
