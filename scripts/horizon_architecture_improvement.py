import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.adapters.qlib.factor_set_evaluator import _PreparedFactorDataset
from core.adapters.qlib.horizon_candidate import fit_residual_weight, percentile_rank
from core.adapters.qlib.rank_models import RankLGBModel
from scripts import train_horizon_classification as classification
from scripts.horizon_architecture_research import FOLDS, HORIZONS, OUTPUT, SEEDS, fit_lgb, write_json
from scripts.horizon_multiscale_research import calendar_index, SOURCE
from scripts.model_complementarity_research import infer_n0
from scripts.run_multiscale_group_research import sha256
from scripts.train_horizon_sequence import fit_sequence


X_COLUMNS = (
    "context_log_total_mv", "context_log_circ_mv", "context_turnover", "context_log_amount",
    "context_market_ret20", "context_market_ret60", "context_market_vol20", "context_market_vol60",
    "context_market_breadth", "context_beta60", "context_beta120", "context_corr60",
    "context_corr120", "context_residvol60", "context_residvol120",
    "context_turnover_mean20", "context_turnover_mean60", "context_log_amount_mean20",
    "context_log_amount_mean60", "context_missing_size", "context_missing_liquidity",
    "context_missing_market_relation",
)
ARMS = ("A0", "AX", "F", "R", "E")


def signature(fold, horizon, seed, arm, budget):
    payload = {"base_manifest": sha256(OUTPUT / "manifest.json"),
               "base_labels": sha256(OUTPUT / "label_audit.json"),
               "context": sha256(SOURCE / "context/context_features.parquet"),
               "code": sha256(Path(__file__)), "fold": fold, "horizon": horizon,
               "seed": seed, "arm": arm, "budget": budget, "columns": X_COLUMNS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def make_inner_masks(index, labels, outer_train, cohort_start, cohort_end):
    dates = index.get_level_values("datetime")
    valid_start = cohort_start - pd.DateOffset(months=3)
    label_end = labels["label_end"].to_numpy()
    usable = labels["label"].notna().to_numpy()
    train = outer_train & (dates < valid_start) & usable & (label_end < valid_start.to_datetime64())
    valid = outer_train & (dates >= valid_start) & (dates < cohort_start) & usable & (label_end < cohort_start.to_datetime64())
    cohort = outer_train & (dates >= cohort_start) & (dates < cohort_end) & usable
    if not train.any() or not valid.any() or not cohort.any():
        raise RuntimeError(f"内层外推样本不足: {cohort_start.date()}")
    if labels.loc[train, "label_end"].max() >= valid_start or labels.loc[valid, "label_end"].max() >= cohort_start:
        raise RuntimeError("内层标签前视")
    return {"train": train, "valid": valid, "calibrate": cohort, "test": cohort}


def fit_auxiliary(frame, train_index, train_target, valid_index, valid_target, prediction_indices, params, path):
    features = list(frame.columns)
    train = pd.concat({"feature": frame.loc[train_index], "label": train_target.to_frame("LABEL0")}, axis=1)
    valid = pd.concat({"feature": frame.loc[valid_index], "label": valid_target.to_frame("LABEL0")}, axis=1)
    model = RankLGBModel(**params)
    model.fit(_PreparedFactorDataset({"train": train, "valid": valid}, features), verbose_eval=0, log_metrics=False)
    path.write_text(model.model.model_to_string(), encoding="utf-8")
    import lightgbm as lgb
    restored = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
    predictions = {}
    for name, index in prediction_indices.items():
        values = frame.loc[index].to_numpy(dtype=np.float32)
        raw = model.model.predict(values, num_threads=4)
        check = restored.predict(values[:min(10000, len(values))], num_threads=4)
        if np.max(np.abs(raw[:len(check)] - check)) > 1e-6:
            raise RuntimeError("辅助模型保存恢复误差超过1e-6")
        predictions[name] = pd.Series(raw, index=index)
    return predictions


def train(args):
    base_manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("status") != "frozen":
        raise RuntimeError("基础矩阵未冻结")
    for relative, expected in base_manifest["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"改进输入签名变化: {relative}")
    budget_path = OUTPUT / "improvement_budget.json"
    if not budget_path.exists() and args.stage != "prerun":
        raise RuntimeError("必须先冻结等训练时间预算")
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if args.stage != "prerun" else {name: {"members": 1} for name in HORIZONS}
    artifact_root = OUTPUT / "improvement_prerun" if args.stage == "prerun" else OUTPUT
    artifact_root.mkdir(parents=True, exist_ok=True)
    classification.OUTPUT = OUTPUT
    calendar = calendar_index()
    for horizon in args.horizons:
        anchor = "G" if horizon in ("H5", "H10") else "N"
        for fold in args.folds:
            inputs, _, _, labels, outer_masks, _ = classification.load_fold(fold, int(horizon[1:]))
            factors = base_manifest["factor_keys"]
            if not set(X_COLUMNS).issubset(inputs["D"].columns):
                raise RuntimeError("冻结上下文字段缺失")
            frame = inputs["D"][[*factors, *X_COLUMNS]].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            index = frame.index
            train_dates = index.get_level_values("datetime")[outer_masks["train"]]
            last_train = train_dates.max()
            cohort_starts = [last_train + pd.Timedelta(days=1) - pd.DateOffset(months=18 - 6 * number) for number in range(3)]
            cohort_ends = [start + pd.DateOffset(months=6) for start in cohort_starts]
            for seed in args.seeds:
                paths = {arm: artifact_root / f"{fold}_{horizon}_{seed}_{arm}_improve.parquet" for arm in ARMS}
                if all(path.exists() and path.with_suffix(".json").exists() for path in paths.values()):
                    if any(json.loads(path.with_suffix(".json").read_text(encoding="utf-8")).get("signature") != signature(fold, horizon, seed, arm, budget) for arm, path in paths.items()):
                        raise RuntimeError("改进矩阵续跑签名冲突")
                    continue
                params = {**base_manifest["lightgbm_parameters"], "num_threads": 4, "seed": seed,
                          "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed}
                params.pop("seed_policy", None)
                params.pop("verbose_eval", None)
                started = time.perf_counter()
                a0 = pd.read_parquet(OUTPUT / f"{fold}_{horizon}_{seed}_{anchor}.parquet")
                if not a0.index.equals(index[outer_masks["test"]]):
                    raise RuntimeError("基础锚点共同分母变化")
                if anchor == "G":
                    ax, meta = fit_lgb(frame, labels, outer_masks, list(frame.columns), params,
                                       artifact_root / f"{fold}_{horizon}_{seed}_AX.txt", False, calendar)
                    model = __import__("lightgbm").Booster(model_str=(artifact_root / f"{fold}_{horizon}_{seed}_AX.txt").read_text(encoding="utf-8"))
                    valid_index = index[outer_masks["valid"]]
                    ax["valid"] = pd.Series(model.predict(frame.loc[valid_index].to_numpy(), num_threads=4), index=valid_index)
                else:
                    values = frame.to_numpy(dtype=np.float32)
                    ax, _, meta = fit_sequence(values, index, labels["label_rank"].to_numpy(dtype=np.float32),
                        outer_masks, calendar, None, "N0", seed, artifact_root / f"{fold}_{horizon}_{seed}_AX.pt",
                        args.resume, signature(fold, horizon, seed, "AX_model", budget), len(frame.columns), None)
                    valid_index = index[outer_masks["valid"]]
                    ax.update(infer_n0(artifact_root / f"{fold}_{horizon}_{seed}_AX.final.pt", values, index,
                         {"valid": np.flatnonzero(outer_masks["valid"])}, calendar))
                anchor_done = time.perf_counter()
                oof = []
                oof_audits = []
                for number, (start, end) in enumerate(zip(cohort_starts, cohort_ends)):
                    inner = make_inner_masks(index, labels, outer_masks["train"], start, end)
                    if anchor == "G":
                        inner_scores, _ = fit_lgb(frame, labels, inner, list(frame.columns), params,
                            artifact_root / f"{fold}_{horizon}_{seed}_OOF{number}.txt", False, calendar)
                        score = inner_scores["test"]
                    else:
                        inner_scores, _, _ = fit_sequence(values, index, labels["label_rank"].to_numpy(dtype=np.float32),
                            inner, calendar, None, "N0", seed + 10000 * (number + 1),
                            artifact_root / f"{fold}_{horizon}_{seed}_OOF{number}.pt", args.resume,
                            signature(fold, horizon, seed, f"OOF{number}", budget), len(frame.columns), None)
                        score = inner_scores["test"]
                    oof.append(percentile_rank(score))
                    oof_audits.append({"cohort_start": str(start.date()), "cohort_end": str(end.date()),
                                       "training_rows": int(inner["train"].sum()),
                                       "valid_rows": int(inner["valid"].sum()), "oof_rows": len(score)})
                oof_rank = pd.concat(oof).sort_index()
                oof_done = time.perf_counter()
                if oof_rank.index.has_duplicates or not oof_rank.index.isin(index[outer_masks["train"]]).all():
                    raise RuntimeError("OOF残差样本重复或越过外层训练段")
                pct = labels["label_rank"].div(3.46).add(0.5)
                residual_target = pct.reindex(oof_rank.index) - oof_rank
                valid_rank = percentile_rank(ax["valid"])
                cal_rank = percentile_rank(ax["calibrate"])
                test_rank = percentile_rank(ax["test"])
                valid_residual = pct.reindex(valid_rank.index) - valid_rank
                prediction_indices = {"calibrate": ax["calibrate"].index, "test": ax["test"].index}
                auxiliary = {}
                auxiliary_times = {}
                for arm, train_target, valid_target in (("F", pct.reindex(oof_rank.index), pct.reindex(valid_rank.index)),
                                                        ("R", residual_target, valid_residual)):
                    auxiliary_start = time.perf_counter()
                    auxiliary[arm] = fit_auxiliary(frame, oof_rank.index, train_target, valid_rank.index,
                        valid_target, prediction_indices, params,
                        artifact_root / f"{fold}_{horizon}_{seed}_{arm}_aux.txt")
                    auxiliary_times[arm] = time.perf_counter() - auxiliary_start
                f_cal, f_test = (percentile_rank(auxiliary["F"][name]) for name in ("calibrate", "test"))
                f_lambda = fit_residual_weight(cal_rank, f_cal - cal_rank, pct.reindex(cal_rank.index))
                r_lambda = fit_residual_weight(cal_rank, auxiliary["R"]["calibrate"], pct.reindex(cal_rank.index))
                scores = {"A0": a0["score"], "AX": test_rank,
                          "F": test_rank + f_lambda * (f_test - test_rank),
                          "R": test_rank + r_lambda * auxiliary["R"]["test"]}
                members = [test_rank]
                for member in range(1, budget[horizon]["members"]):
                    member_params = {**params, "seed": seed + 10000 * member,
                                     "bagging_seed": seed + 10000 * member,
                                     "feature_fraction_seed": seed + 10000 * member,
                                     "data_random_seed": seed + 10000 * member}
                    if anchor == "G":
                        member_score, _ = fit_lgb(frame, labels, outer_masks, list(frame.columns), member_params,
                            artifact_root / f"{fold}_{horizon}_{seed}_E{member}.txt", False, calendar)
                        members.append(percentile_rank(member_score["test"]))
                    else:
                        member_score, _, _ = fit_sequence(values, index, labels["label_rank"].to_numpy(dtype=np.float32),
                            outer_masks, calendar, None, "N0", seed + 10000 * member,
                            artifact_root / f"{fold}_{horizon}_{seed}_E{member}.pt", args.resume,
                            signature(fold, horizon, seed, f"E{member}", budget), len(frame.columns), None)
                        members.append(percentile_rank(member_score["test"]))
                scores["E"] = sum(members) / len(members)
                for arm, score in scores.items():
                    path = paths[arm]
                    meta_path = path.with_suffix(".json")
                    sig = signature(fold, horizon, seed, arm, budget)
                    if path.exists() or meta_path.exists():
                        if not path.exists() or not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")).get("signature") != sig:
                            raise RuntimeError("改进方案文件冲突")
                        continue
                    output = pd.DataFrame({"score": score, "label": labels["label"].reindex(score.index),
                        "label_rank": labels["label_rank"].reindex(score.index)})
                    if not output.index.equals(a0.index) or output.isna().any().any() or not np.isfinite(output.to_numpy()).all():
                        raise RuntimeError("改进方案共同样本或数值无效")
                    temp = path.with_suffix(".tmp")
                    output.to_parquet(temp, compression="zstd")
                    temp.replace(path)
                    write_json(meta_path, {"signature": sig, "arm": arm, "rows": len(output),
                        "anchor": anchor, "X_columns": X_COLUMNS, "oof": oof_audits,
                        "residual_weight": r_lambda, "direct_weight": f_lambda,
                        "budget_members": len(members), "anchor_model": meta,
                        "timing": {"anchor_seconds": anchor_done - started,
                                   "oof_seconds": oof_done - anchor_done,
                                   "auxiliary_F_seconds": auxiliary_times["F"],
                                   "auxiliary_R_seconds": auxiliary_times["R"]},
                        "elapsed_seconds": time.perf_counter() - started,
                        "strict_pit_verified": False})
                    print(json.dumps({"fold": fold, "horizon": horizon, "seed": seed,
                                      "arm": arm, "rows": len(output)}, ensure_ascii=False), flush=True)


def budget():
    target = OUTPUT / "improvement_budget.json"
    values = {}
    for horizon in HORIZONS:
        measurements = []
        for fold in ("W1", "W8"):
            path = OUTPUT / "improvement_prerun" / f"{fold}_{horizon}_42_R_improve.json"
            if not path.exists():
                raise RuntimeError(f"预算缺少工程预跑: {path.name}")
            timing = json.loads(path.read_text(encoding="utf-8"))["timing"]
            if timing["anchor_seconds"] <= 0:
                raise RuntimeError("主模型计时无效")
            measurements.append({"fold": fold, **timing,
                "cost_ratio": (timing["anchor_seconds"] + timing["oof_seconds"] + timing["auxiliary_R_seconds"]) / timing["anchor_seconds"]})
        members = max(1, int(np.ceil(np.mean([row["cost_ratio"] for row in measurements]))))
        values[horizon] = {"members": members, "measurements": measurements}
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != values:
        raise RuntimeError("等预算冻结配置冲突")
    write_json(target, values)
    print(json.dumps({name: row["members"] for name, row in values.items()}, ensure_ascii=False))


def status():
    count = sum((OUTPUT / f"{fold}_{horizon}_{seed}_{arm}_improve.parquet").exists()
                for fold in FOLDS for horizon in HORIZONS for seed in SEEDS for arm in ARMS)
    print(json.dumps({"status": "complete" if count == 360 else "pending", "completed": count,
                      "planned": 360}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prerun", "train", "budget", "status"))
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "status":
        status()
    elif args.stage == "budget":
        budget()
    else:
        if args.stage == "prerun":
            args.folds, args.seeds = ["W1", "W8"], [42]
        train(args)
