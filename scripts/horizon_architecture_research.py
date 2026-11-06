import argparse
import hashlib
import json
import os
import pickle
import shutil
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
from core.adapters.qlib.horizon_candidate import blend_rank, fit_rank_weight, mature_update_dates, percentile_rank
from core.adapters.qlib.rank_models import RankLGBModel
from scripts import train_horizon_classification as classification
from scripts.horizon_multiscale_research import calendar_index, common_universe_mask, load_panel, make_calendar_labels, rank_labels, segments
from scripts.model_complementarity_research import infer_n0
from scripts.run_multiscale_group_research import sha256
from scripts.train_horizon_sequence import fit_sequence, verify_restore


OLD = ROOT / "data/experiments/horizon_multiscale_research_v2_4m"
OUTPUT = Path(os.environ.get("HORIZON_ARCHITECTURE_OUTPUT", "G:/AShareResearch/horizon_architecture_v1")).resolve()
FOLDS = tuple(f"W{number}" for number in range(1, 9))
HORIZONS = ("H5", "H10", "H20")
SEEDS = (42, 2026, 3407)
ARMS = ("G", "GE", "N", "F50", "FC", "FR")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def signature(manifest, fold, horizon, seed, arm):
    payload = {"manifest": sha256(OUTPUT / "manifest.json"), "label_audit": sha256(OUTPUT / "label_audit.json"),
               "fold": fold, "horizon": horizon, "seed": seed, "arm": arm,
               "weight_code": sha256(ROOT / "core/adapters/qlib/horizon_candidate.py"),
               "training_code": sha256(Path(__file__))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preflight():
    if not OLD.joinpath("manifest.json").exists() or not OLD.joinpath("label_audit.json").exists():
        raise RuntimeError("缺少旧窗口冻结清单及标签审计")
    old_manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    old_audit = json.loads((OLD / "label_audit.json").read_text(encoding="utf-8"))
    if old_audit.get("status") != "labels_frozen" or len(old_manifest["factor_keys"]) != 40:
        raise RuntimeError("旧标签或原40因子契约无效")
    for relative, expected in old_manifest["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"冻结源数据改变: {relative}")
    for name, expected in old_manifest["code_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"旧研究代码改变: {name}")
    free = shutil.disk_usage(OUTPUT.anchor).free
    if free < 10 * 1024 ** 3:
        raise RuntimeError("G盘未达到10GiB保留空间")
    report = {"status": "passed", "output": str(OUTPUT), "free_gib": round(free / 1024 ** 3, 2),
              "old_manifest_sha256": sha256(OLD / "manifest.json"),
              "old_label_audit_sha256": sha256(OLD / "label_audit.json"),
              "historical_research_only": True, "strict_pit_verified": False, "execution_verified": False}
    print(json.dumps(report, ensure_ascii=False))
    return report


def freeze():
    check = preflight()
    manifest_path = OUTPUT / "manifest.json"
    previous = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    manifest = {"status": "frozen", "horizons": {"H5": 5, "H10": 10, "H20": 20},
                "folds": previous["folds"], "factor_keys": previous["factor_keys"],
                "lightgbm_parameters": previous["lightgbm_parameters"],
                "source_sha256": previous["source_sha256"], "code_sha256": previous["code_sha256"],
                "old_manifest_sha256": check["old_manifest_sha256"], "old_label_audit_sha256": check["old_label_audit_sha256"],
                "seeds": list(SEEDS), "arms": list(ARMS), "planned_predictions": 432,
                "primary_folds": list(FOLDS[:7]), "sensitivity_fold": "W8",
                "score_scale": "within-day full-universe percentile rank", "label_percentile": "label_rank/3.46+0.5",
                "calibration": {"FC_penalty": 0.01, "FR_lookback_decision_days": 126,
                                "FR_min_mature_dates": 60, "FR_monthly_max_change": 0.1},
                "research_only": True, "strict_pit_verified": False, "execution_verified": False}
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("冻结清单冲突，拒绝覆盖")
    write_json(manifest_path, manifest)
    for name in ("stock_identity_snapshot.json", "stock_index_membership_snapshot.json"):
        source, target = OLD / name, OUTPUT / name
        if target.exists() and sha256(target) != sha256(source):
            raise RuntimeError(f"身份快照冲突: {name}")
        if not target.exists():
            shutil.copy2(source, target)
    labels_audit = {"status": "labels_frozen", "labels": {}, "old_label_audit_sha256": check["old_label_audit_sha256"]}
    for horizon in ("H10", "H20"):
        source, target = OLD / f"labels_{horizon}.parquet", OUTPUT / f"labels_{horizon}.parquet"
        if target.exists() and sha256(target) != sha256(source):
            raise RuntimeError(f"冻结标签冲突: {horizon}")
        if not target.exists():
            shutil.copy2(source, target)
        labels_audit["labels"][horizon] = {"sha256": sha256(target), "source_sha256": sha256(source)}
    target = OUTPUT / "labels_H5.parquet"
    if not target.exists():
        calendar = calendar_index()
        panel = load_panel("2018-01-01", "2026-10-27", ["open", "close", "volume"])
        label, label_end = make_calendar_labels(panel, calendar, 5)
        label = label.where(common_universe_mask(panel))
        frame = pd.DataFrame({"label": label, "label_end": label_end, "label_rank": rank_labels(label, calendar)})
        frame.to_parquet(target.with_suffix(".tmp"), compression="zstd")
        target.with_suffix(".tmp").replace(target)
        del panel, frame
    labels_audit["labels"]["H5"] = {"sha256": sha256(target)}
    for horizon in HORIZONS:
        frame = pd.read_parquet(OUTPUT / f"labels_{horizon}.parquet")
        horizon_audit = {}
        for fold in FOLDS:
            masks = segments(manifest["folds"][fold], frame["label"], frame["label_end"], int(horizon[1:]))
            horizon_audit[fold] = {name: {"rows": int(mask.sum()),
                "last_label_end": str(frame.loc[mask, "label_end"].max())} for name, mask in masks.items()}
            for current, next_segment in (("train", "valid"), ("valid", "calibrate"), ("calibrate", "test")):
                if frame.loc[masks[current], "label_end"].max() >= pd.Timestamp(manifest["folds"][fold][next_segment][0]):
                    raise RuntimeError(f"{fold}/{horizon}/{current} 标签未净化")
        labels_audit["labels"][horizon]["folds"] = horizon_audit
    audit_path = OUTPUT / "label_audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8")) != labels_audit:
        raise RuntimeError("标签审计清单冲突")
    write_json(audit_path, labels_audit)
    print(json.dumps({"status": "frozen", "output": str(OUTPUT), "horizons": list(HORIZONS)}, ensure_ascii=False))


def saved_components(fold, horizon, seed, base, index, masks, calendar):
    old_manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    model_data = pickle.loads((OLD / f"{fold}_{horizon}_{seed}_sequence_lgb.pkl").read_bytes())
    import lightgbm as lgb
    booster = lgb.Booster(model_str=(OLD / f"{fold}_{horizon}_{seed}_sequence_lgb.txt").read_text(encoding="utf-8"))
    outputs = {}
    for name in ("calibrate", "test"):
        target = index[masks[name]]
        raw = model_data["scores"][name]
        if not raw.index.equals(target):
            raise RuntimeError(f"旧LGB样本冲突: {fold}/{horizon}/{seed}/{name}")
        sample = base.loc[target].to_numpy()[:10000]
        check = booster.predict(sample, num_threads=4)
        if np.max(np.abs(check - raw.to_numpy()[:len(sample)])) > 1e-6:
            raise RuntimeError("旧LGB恢复误差超过1e-6")
        outputs[name] = raw
    sequence_path = OLD / f"{fold}_{horizon}_{seed}_N0.final.pt"
    raw_n = infer_n0(sequence_path, base.to_numpy(dtype=np.float32), index,
                      {name: np.flatnonzero(masks[name]) for name in ("calibrate", "test")}, calendar)
    for name in outputs:
        if not raw_n[name].index.equals(outputs[name].index):
            raise RuntimeError("LGB与LSTM共同样本不一致")
    return outputs, raw_n, {"lgb_model": str(OLD / f"{fold}_{horizon}_{seed}_sequence_lgb.txt"),
                            "lstm_model": str(sequence_path), "factor_count": len(old_manifest["factor_keys"])}


def fit_lgb(frame, labels, masks, features, params, model_path, train_every_five, calendar):
    session = calendar.get_indexer(frame.index.get_level_values("datetime"))
    train_mask = masks["train"] & ((session % 5 == 0) if train_every_five else True)
    groups = {}
    for name, mask in (("train", train_mask), ("valid", masks["valid"])):
        groups[name] = pd.concat({"feature": frame.loc[mask],
            "label": labels.loc[mask, "label_rank"].to_frame("LABEL0")}, axis=1)
    model = RankLGBModel(**params)
    model.fit(_PreparedFactorDataset(groups, features), verbose_eval=0, log_metrics=False)
    predictions = {name: pd.Series(model.model.predict(frame.loc[mask].to_numpy(), num_threads=4), index=frame.index[mask])
                   for name, mask in masks.items() if name in ("calibrate", "test")}
    model_path.write_text(model.model.model_to_string(), encoding="utf-8")
    restore_error = verify_restore(model, model_path, frame.loc[masks["test"]].to_numpy())
    return predictions, {"train_rows": int(train_mask.sum()), "restore_max_abs_error": restore_error, "model": str(model_path)}


def monthly_weights(g, n, labels, fc_weight, calendar):
    combined = pd.DataFrame({"g": pd.concat([g["calibrate"], g["test"]]),
                             "n": pd.concat([n["calibrate"], n["test"]])}).sort_index()
    test_dates = pd.DatetimeIndex(g["test"].index.get_level_values("datetime").unique()).sort_values()
    all_dates = pd.DatetimeIndex(combined.index.get_level_values("datetime").unique()).sort_values()
    end_by_date = labels["label_end"].groupby(level="datetime").max().reindex(all_dates)
    weights = pd.Series(index=g["test"].index, dtype=float)
    audits = []
    previous = fc_weight
    for month, dates in pd.Series(test_dates, index=test_dates).groupby(test_dates.to_period("M")):
        first = dates.iloc[0]
        eligible = mature_update_dates(all_dates, end_by_date, first, 126)
        eligible = eligible[eligible < first]
        reason = "updated"
        current = previous
        if len(eligible) < 60:
            current, reason = fc_weight, "fewer_than_60_mature_decision_dates"
        else:
            selected = combined.index.get_level_values("datetime").isin(eligible)
            indices = combined.index[selected]
            target = labels["label_rank"].reindex(indices).div(3.46).add(0.5)
            fitted = fit_rank_weight(combined.loc[indices, "g"], combined.loc[indices, "n"], target,
                                     center=fc_weight)
            current = float(np.clip(fitted, max(0.0, previous - 0.1), min(1.0, previous + 0.1)))
        mask = weights.index.get_level_values("datetime").isin(dates.to_numpy())
        weights.iloc[np.flatnonzero(mask)] = current
        audits.append({"month": str(month), "effective_date": str(first.date()), "weight": current,
                       "mature_dates": len(eligible), "last_label_end": str(end_by_date.reindex(eligible).max()),
                       "reason": reason})
        previous = current
    if weights.isna().any():
        raise RuntimeError("动态权重未覆盖所有预测日")
    return weights, audits


def train(args):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "label_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "labels_frozen":
        raise RuntimeError("必须先冻结标签")
    calendar = calendar_index()
    classification.OUTPUT = OUTPUT
    for horizon in args.horizons:
        for fold in args.folds:
            inputs, _, _, labels, masks, load_audit = classification.load_fold(fold, int(horizon[1:]))
            features = manifest["factor_keys"]
            base = inputs["D"][features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            index = base.index
            write_json(OUTPUT / f"{fold}_{horizon}_input_audit.json", load_audit)
            for seed in args.seeds:
                wanted = [arm for arm in ARMS if not (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet").exists()]
                if not wanted:
                    for arm in ARMS:
                        meta = json.loads((OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.json").read_text(encoding="utf-8"))
                        if meta.get("signature") != signature(manifest, fold, horizon, seed, arm):
                            raise RuntimeError("续跑签名不匹配")
                    continue
                params = {**manifest["lightgbm_parameters"], "num_threads": 4, "seed": seed,
                          "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed}
                params.pop("seed_policy", None)
                params.pop("verbose_eval", None)
                started = time.perf_counter()
                if horizon == "H5":
                    g, g_meta = fit_lgb(base, labels, masks, features, params, OUTPUT / f"{fold}_{horizon}_{seed}_G.txt", False, calendar)
                    rows = base.to_numpy(dtype=np.float32)
                    n, _, n_meta = fit_sequence(rows, index, labels["label_rank"].to_numpy(dtype=np.float32), masks,
                        calendar, None, "N0", seed, OUTPUT / f"{fold}_{horizon}_{seed}_N.pt", args.resume,
                        signature(manifest, fold, horizon, seed, "N"), len(features), None)
                else:
                    g, n, g_meta = saved_components(fold, horizon, seed, base, index, masks, calendar)
                    n_meta = {"model": g_meta["lstm_model"]}
                ge, ge_meta = fit_lgb(base, labels, masks, features, params, OUTPUT / f"{fold}_{horizon}_{seed}_GE.txt", True, calendar)
                ranks = {name: {component: percentile_rank(predictions[name]) for component, predictions in
                                (("G", g), ("GE", ge), ("N", n))} for name in ("calibrate", "test")}
                target = labels["label_rank"].reindex(ranks["calibrate"]["G"].index).div(3.46).add(0.5)
                fc = fit_rank_weight(ranks["calibrate"]["G"], ranks["calibrate"]["N"], target)
                fr, updates = monthly_weights({name: ranks[name]["G"] for name in ranks},
                                              {name: ranks[name]["N"] for name in ranks}, labels, fc, calendar)
                scores = {"G": ranks["test"]["G"], "GE": ranks["test"]["GE"], "N": ranks["test"]["N"],
                          "F50": blend_rank(ranks["test"]["G"], ranks["test"]["N"], 0.5),
                          "FC": blend_rank(ranks["test"]["G"], ranks["test"]["N"], fc),
                          "FR": (1 - fr) * ranks["test"]["G"] + fr * ranks["test"]["N"]}
                for arm, score in scores.items():
                    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    meta_path = path.with_suffix(".json")
                    sig = signature(manifest, fold, horizon, seed, arm)
                    if path.exists() or meta_path.exists():
                        if not path.exists() or not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")).get("signature") != sig:
                            raise RuntimeError(f"续跑文件冲突: {path.name}")
                        continue
                    frame = pd.DataFrame({"score": score,
                        "label": labels["label"].reindex(score.index),
                        "label_rank": labels["label_rank"].reindex(score.index)})
                    if not frame.index.equals(index[masks["test"]]) or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
                        raise RuntimeError(f"共同样本或预测值无效: {path.name}")
                    temporary = path.with_suffix(".tmp")
                    frame.to_parquet(temporary, compression="zstd")
                    temporary.replace(path)
                    write_json(meta_path, {"signature": sig, "arm": arm, "rows": len(frame), "weight": fc if arm == "FC" else None,
                        "weight_updates": updates if arm == "FR" else None,
                        "g_model": g_meta, "ge_model": ge_meta, "n_model": n_meta,
                        "elapsed_seconds": time.perf_counter() - started,
                        "score_scale": manifest["score_scale"], "calibration": "none"})
                    print(json.dumps({"fold": fold, "horizon": horizon, "seed": seed, "arm": arm,
                                      "rows": len(frame)}, ensure_ascii=False), flush=True)


def status():
    count = sum((OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet").exists()
                and (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.json").exists()
                for fold in FOLDS for horizon in HORIZONS for seed in SEEDS for arm in ARMS)
    print(json.dumps({"status": "complete" if count == 432 else "pending", "completed": count,
                      "planned": 432, "output": str(OUTPUT)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "freeze", "prerun", "train", "status"))
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    elif args.stage == "freeze":
        freeze()
    elif args.stage == "status":
        status()
    else:
        if args.stage == "prerun":
            args.folds, args.seeds = ["W1", "W8"], [42]
        train(args)


if __name__ == "__main__":
    main()
