import argparse
import gc
import hashlib
import json
import os
import pickle
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
from joblib import parallel_backend
from qlib.data.dataset.processor import Fillna, ProcessInf

from core.adapters.qlib.custom_processors import CSWinsorize
from core.adapters.qlib.factor_set_evaluator import _PreparedFactorDataset
from core.adapters.qlib.models import DEFAULT_PARAMS, PARAM_PROFILES
from core.adapters.qlib.rank_models import RankLGBModel
from scripts.horizon_multiscale_features import build_groups
from scripts.monthly_group_features import build_monthly_features
from scripts.run_multiscale_group_research import sha256
from scripts.horizon_multiscale_research import OUTPUT, SOURCE, PRIOR, ROOT, calendar_index
from scripts.train_monthly_group_research import industry_groups


def calibrate(cal_score, raw_label, group, pred, pred_group):
    frame = pd.concat({"score": cal_score, "label": raw_label.reindex(cal_score.index), "group": group.reindex(cal_score.index)}, axis=1).dropna()
    frame["weight"] = 1.0 / frame.groupby(level="datetime")["score"].transform("size")
    overall = frame
    maps = {}
    for group_id in [-1, *sorted(frame.loc[frame["group"] >= 0, "group"].unique().astype(int))]:
        part = overall if group_id == -1 else frame.loc[frame["group"] == group_id]
        if part.empty:
            continue
        x, y, w = (part[name].to_numpy(dtype=float) for name in ("score", "label", "weight"))
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        variance = np.average((x - mx) ** 2, weights=w)
        slope = np.average((x - mx) * (y - my), weights=w) / variance if variance > 1e-12 else 0.0
        intercept = my - slope * mx
        if group_id < 0:
            if slope <= 1e-12:
                slope, intercept = 1.0, my - mx
            maps[-1] = (intercept, slope)
        else:
            days = part.index.get_level_values("datetime").nunique()
            if days < 40 or slope <= 1e-12:
                maps[group_id] = maps[-1]
            else:
                shrink = days / (days + 40.0)
                maps[group_id] = (shrink * intercept + (1 - shrink) * maps[-1][0],
                                  shrink * slope + (1 - shrink) * maps[-1][1])
    result = maps[-1][0] + maps[-1][1] * pred
    for group_id, (intercept, slope) in maps.items():
        if group_id >= 0:
            result.loc[pred_group.reindex(pred.index) == group_id] = intercept + slope * pred.loc[pred_group.reindex(pred.index) == group_id]
    return result.astype(np.float32), maps


def model_signature(manifest, horizon, fold, seed, arm, group_id=None):
    labels_hash = sha256(OUTPUT / "label_audit.json")
    payload = json.dumps({"manifest": sha256(OUTPUT / "manifest.json"), "labels": labels_hash,
        "horizon": horizon, "fold": fold, "seed": seed, "arm": arm, "group": group_id,
        "parameters": manifest["lightgbm_parameters"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_restore(model, model_path, values):
    import lightgbm as lgb
    sample = values[:min(10000, len(values))]
    if not len(sample):
        return 0.0
    expected = model.model.predict(sample, num_threads=4)
    restored = lgb.Booster(model_str=model_path.read_text(encoding="utf-8")).predict(sample, num_threads=4)
    error = float(np.max(np.abs(expected - restored)))
    if error > 1e-6:
        raise RuntimeError(f"LightGBM保存恢复误差超过1e-6: {error}")
    return error


def load_fold(fold, horizon):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    window = manifest["folds"][fold]
    start = pd.Timestamp(window["train"][0]) - pd.DateOffset(months=15)
    end = pd.Timestamp(window["test"][1])
    filters = [("datetime", ">=", start), ("datetime", "<=", end)]
    parts = [pd.read_parquet(path, filters=filters) for path in sorted(SOURCE.glob("new_*.parquet"))]
    base = pd.concat(parts).sort_index()
    factor_keys = manifest["factor_keys"]
    context_path = SOURCE / "context/context_features.parquet"
    context = pd.read_parquet(context_path, filters=filters).reindex(base.index)
    context_keys = json.loads(context_path.with_name("coverage.json").read_text(encoding="utf-8"))["feature_columns"]
    preheat = pd.read_parquet(ROOT / "data/experiments/monthly_group_research_v4_20261103/preheat_2018.parquet")
    quotes = pd.concat([preheat, base[["open", "high", "low", "close", "volume"]]]).sort_index()
    if quotes.index.has_duplicates:
        quotes = quotes[~quotes.index.duplicated(keep="last")]
    calendar = calendar_index()
    listing_rows = json.loads((OUTPUT / "stock_identity_snapshot.json").read_text(encoding="utf-8"))
    listing_dates = {}
    for row in listing_rows:
        if row.get("list_date"):
            symbol, exchange = str(row["ts_code"]).split(".", 1)
            prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(exchange.upper())
            if prefix:
                listing_dates[prefix + symbol] = pd.Timestamp(row["list_date"])
    monthly = build_monthly_features(quotes, calendar).reindex(base.index)
    monthly = monthly.assign(**{f"missing_{name}": monthly[name].isna().astype(np.float32) for name in monthly}).fillna(0)
    group_snapshot, r4_continuous, group_audit = build_groups(quotes, context.reindex(quotes.index), calendar, listing_dates)
    groups = group_snapshot.reindex(base.index)
    history_industry, _ = industry_groups(quotes.index)
    history_industry = history_industry.reindex(base.index)
    prior_style = pd.read_parquet(PRIOR / "classification_snapshot.parquet",
                                  filters=filters, columns=["GS8"]).reindex(base.index)["GS8"]
    calibration_group = ((history_industry.clip(lower=0) * 9 + prior_style.clip(lower=0))
                         .where((history_industry >= 0) & (prior_style >= 0), -1).astype(np.int16))
    del history_industry, prior_style
    r4 = r4_continuous.reindex(base.index)
    extra = pd.DataFrame(index=base.index)
    group_counts = {"B3": 3, "A2": 2, "R4": 4}
    for name, count in group_counts.items():
        for group_id in range(count):
            extra[f"{name}_{group_id}"] = (groups[name] == group_id).astype(np.float32)
        extra[f"{name}_unknown"] = (groups[name] < 0).astype(np.float32)
    for column in r4:
        extra[column] = r4[column].astype(np.float32)
        extra[f"missing_{column}"] = r4[column].isna().astype(np.float32)
    common = pd.concat([context[context_keys].astype(np.float32), extra], axis=1).replace([np.inf, -np.inf], np.nan)
    common = common.fillna(0)
    original = base[factor_keys]
    wrapped = CSWinsorize()(pd.concat({"feature": original}, axis=1))
    if np.isinf(wrapped.to_numpy()).any():
        with parallel_backend("threading"):
            wrapped = ProcessInf()(wrapped)
    original = Fillna()(wrapped)["feature"].astype(np.float32)
    inputs = {"D": pd.concat([original, common], axis=1),
              "M": pd.concat([original, monthly, common], axis=1)}
    label_frame = pd.read_parquet(OUTPUT / f"labels_H{horizon}.parquet",
                                  filters=[("datetime", ">=", start), ("datetime", "<=", end)])
    label_frame = label_frame.reindex(base.index)
    masks = {}
    for i, segment in enumerate(("train", "valid", "calibrate", "test")):
        low, high = map(pd.Timestamp, window[segment])
        date_values = label_frame.index.get_level_values("datetime")
        mask = (date_values >= low) & (date_values <= high) & label_frame["label"].notna().to_numpy()
        if segment != "test":
            next_segment = ("valid", "calibrate", "test")[i]
            cutoff = pd.Timestamp(window[next_segment][0])
            mask &= label_frame["label_end"].to_numpy() < cutoff.to_datetime64()
        masks[segment] = mask
    audit = {"rows": len(base), "label_rows": int(label_frame["label"].notna().sum()),
             "groups_unknown_share": groups.lt(0).mean().to_dict(), "features": {key: len(value.columns) for key, value in inputs.items()},
             "identity_coverage": len(listing_dates), "group_audit": group_audit,
             "segments": {name: int(mask.sum()) for name, mask in masks.items()}}
    return inputs, groups, calibration_group, label_frame, masks, audit


def train(args):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    labels_audit = json.loads((OUTPUT / "label_audit.json").read_text(encoding="utf-8"))
    if labels_audit.get("status") != "labels_frozen":
        raise RuntimeError("标签未冻结")
    for relative, expected in manifest["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"输入签名不符: {relative}")
    for relative, expected in manifest["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"代码签名不符: {relative}")
    from core.adapters.qlib.models import DEFAULT_PARAMS, PARAM_PROFILES
    for horizon in args.horizons:
        for fold in args.folds:
            inputs, groups, calibration_group, labels, masks, audit = load_fold(fold, int(horizon[1:]))
            (OUTPUT / f"{fold}_{horizon}_input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            raw, details = {}, {}
            for seed in args.seeds:
                params = {**DEFAULT_PARAMS["lightgbm"], **PARAM_PROFILES["conservative"]["lightgbm"],
                          "num_threads": 4, "seed": seed, "bagging_seed": seed,
                          "feature_fraction_seed": seed, "data_random_seed": seed}
                params.pop("verbose_eval", None)
                for input_name in ("D", "M"):
                    x = inputs[input_name]
                    arm = f"U_{input_name}"
                    checkpoint = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.pkl"
                    signature = model_signature(manifest, horizon, fold, seed, arm)
                    if checkpoint.exists() and args.resume:
                        data = pickle.loads(checkpoint.read_bytes())
                        if data.get("signature") != signature:
                            raise RuntimeError(f"断点签名不匹配: {checkpoint.name}")
                        raw[arm], details[arm] = data["scores"], data["details"]
                    else:
                        frames = {segment: pd.concat({"feature": x.loc[masks[segment]],
                            "label": labels.loc[masks[segment], "label_rank"].to_frame("LABEL0")}, axis=1)
                                  for segment in ("train", "valid")}
                        model = RankLGBModel(**params)
                        started = time.perf_counter()
                        model.fit(_PreparedFactorDataset(frames, list(x.columns)), verbose_eval=0, log_metrics=False)
                        pred = {segment: pd.Series(model.model.predict(x.loc[masks[segment]].to_numpy(), num_threads=4),
                                                   index=x.index[masks[segment]]) for segment in ("calibrate", "test")}
                        raw[arm] = pred
                        details[arm] = {"elapsed_seconds": time.perf_counter() - started,
                                        "best_iteration": model.model.best_iteration, "feature_count": len(x.columns)}
                        model_path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.txt"
                        model_path.write_text(model.model.model_to_string(), encoding="utf-8")
                        details[arm]["restore_max_abs_error"] = verify_restore(model, model_path, x.loc[masks["test"]].to_numpy())
                        temporary = checkpoint.with_suffix(".tmp")
                        temporary.write_bytes(pickle.dumps({"signature": signature, "scores": pred, "details": details[arm]}, protocol=5))
                        temporary.replace(checkpoint)
                        print(json.dumps({"fold": fold, "horizon": horizon, "seed": seed, "arm": arm, **details[arm]}), flush=True)
                        del model, frames
                        gc.collect()
                for input_name in ("D", "M"):
                    base_arm = f"U_{input_name}"
                    x = inputs[input_name]
                    for classifier, count in (("B3", 3), ("A2", 2), ("R4", 4)):
                        arm = f"{classifier}_{input_name}"
                        pred = {segment: raw[base_arm][segment].copy() for segment in ("calibrate", "test")}
                        arm_detail = {"experts": {}, "fallback": {}}
                        for group_id in range(count):
                            group_mask = groups[classifier] == group_id
                            train_mask = masks["train"] & group_mask.to_numpy()
                            valid_mask = masks["valid"] & group_mask.to_numpy()
                            cal_mask = masks["calibrate"] & group_mask.to_numpy()
                            test_mask = masks["test"] & group_mask.to_numpy()
                            idx = x.index[train_mask]
                            info = {"rows": int(train_mask.sum()), "stocks": int(idx.get_level_values("instrument").nunique()),
                                    "days": int(idx.get_level_values("datetime").nunique()),
                                    "calibration_days": int(x.index[cal_mask].get_level_values("datetime").nunique())}
                            if info["rows"] < 50000 or info["stocks"] < 100 or info["days"] < 500 or info["calibration_days"] < 40:
                                info["status"] = "sample_gate_fallback"
                                arm_detail["experts"][str(group_id)] = info
                                continue
                            expert_path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}_{group_id}.pkl"
                            expert_signature = model_signature(manifest, horizon, fold, seed, arm, group_id)
                            if expert_path.exists() and args.resume:
                                expert = pickle.loads(expert_path.read_bytes())
                                if expert.get("signature") != expert_signature:
                                    raise RuntimeError(f"断点签名不匹配: {expert_path.name}")
                                expert_pred, info = expert["scores"], expert["details"]
                            else:
                                frames = {segment: pd.concat({"feature": x.loc[mask],
                                    "label": labels.loc[mask, "label_rank"].to_frame("LABEL0")}, axis=1)
                                          for segment, mask in (("train", train_mask), ("valid", valid_mask))}
                                model = RankLGBModel(**params)
                                started = time.perf_counter()
                                try:
                                    model.fit(_PreparedFactorDataset(frames, list(x.columns)), verbose_eval=0, log_metrics=False)
                                    expert_pred = {segment: pd.Series(model.model.predict(x.loc[mask].to_numpy(), num_threads=4), index=x.index[mask])
                                                   for segment, mask in (("calibrate", cal_mask), ("test", test_mask))}
                                    info.update({"status": "trained", "elapsed_seconds": time.perf_counter() - started,
                                                 "best_iteration": model.model.best_iteration})
                                    model_path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}_{group_id}.txt"
                                    model_path.write_text(model.model.model_to_string(), encoding="utf-8")
                                    info["restore_max_abs_error"] = verify_restore(model, model_path, x.loc[test_mask].to_numpy())
                                    temporary = expert_path.with_suffix(".tmp")
                                    temporary.write_bytes(pickle.dumps({"signature": expert_signature, "scores": expert_pred, "details": info}, protocol=5))
                                    temporary.replace(expert_path)
                                except Exception as error:
                                    info.update({"status": "fit_failure_fallback", "reason": f"{type(error).__name__}: {str(error)[:300]}"})
                                    arm_detail["experts"][str(group_id)] = info
                                    continue
                                print(json.dumps({"fold": fold, "horizon": horizon, "seed": seed, "arm": arm, "group": group_id, **info}), flush=True)
                                del model, frames
                                gc.collect()
                            arm_detail["experts"][str(group_id)] = info
                            for segment in ("calibrate", "test"):
                                pred[segment].loc[expert_pred[segment].index] = expert_pred[segment]
                        raw[arm] = pred
                        details[arm] = arm_detail
                calibrated = {}
                for arm, score in raw.items():
                    classifier = arm.split("_")[0]
                    if classifier in ("B3", "A2", "R4"):
                        route_group = groups[classifier]
                        baseline = f"U_{arm.split('_')[1]}"
                    else:
                        route_group = pd.Series(-1, index=score["calibrate"].index)
                        baseline = None
                    if baseline is None:
                        cal_group = calibration_group
                        test_group = calibration_group
                    else:
                        cal_group = calibration_group.reindex(score["calibrate"].index)
                        test_group = calibration_group.reindex(score["test"].index)
                    value, maps = calibrate(score["calibrate"], labels["label"], cal_group,
                                             score["test"], test_group)
                    if baseline is not None:
                        missing_route = route_group.reindex(value.index) < 0
                        value.loc[missing_route] = calibrated[baseline].reindex(value.index).loc[missing_route]
                        for gid, status in details[arm]["experts"].items():
                            if status.get("status") != "trained":
                                fallback = groups[classifier].reindex(value.index) == int(gid)
                                value.loc[fallback] = calibrated[baseline].reindex(value.index).loc[fallback]
                    calibrated[arm] = value
                    expected_index = labels.index[masks["test"]]
                    if not value.index.equals(expected_index) or value.isna().any() or not np.isfinite(value.to_numpy()).all():
                        raise RuntimeError(f"{fold}/{horizon}/{seed}/{arm}: 预测未覆盖共同股票日或含非有限分数")
                    output = pd.DataFrame({"raw_score": score["test"], "score": value,
                        "label": labels["label"].reindex(value.index), "label_rank": labels["label_rank"].reindex(value.index),
                        "group": groups[classifier].reindex(value.index) if classifier in ("B3", "A2", "R4") else -1})
                    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    temp = path.with_suffix(".tmp")
                    output.to_parquet(temp, compression="zstd")
                    temp.replace(path)
                    details.setdefault(arm, {})["calibration_maps"] = {str(key): list(val) for key, val in maps.items()}
                (OUTPUT / f"{fold}_{horizon}_{seed}_training.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            del inputs, groups, calibration_group, labels, masks
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", choices=("H10", "H20"), default=("H10", "H20"))
    parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())
