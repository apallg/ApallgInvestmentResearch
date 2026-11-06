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
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
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
from scripts.monthly_group_features import build_daily_features, build_monthly_features
from scripts.run_monthly_group_research import OUTPUT, ROOT, SOURCE, SEEDS, ARMS
from scripts.run_multiscale_group_research import sha256


def industry_groups(index):
    rows = json.loads((OUTPUT / "stock_index_membership_snapshot.json").read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError("历史一级行业表为空")
    codes = sorted({str(row["index_code"]) for row in rows if row["index_code"]})
    code_ids = {code: number for number, code in enumerate(codes)}
    membership = {}
    for row in rows:
        ts_code = str(row["ts_code"] or "")
        if "." not in ts_code or not row["index_code"]:
            continue
        symbol, exchange = ts_code.split(".", 1)
        prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(exchange.upper())
        if prefix:
            membership.setdefault(prefix + symbol, []).append((pd.Timestamp(row["in_date"]), pd.Timestamp(row["out_date"]) if row["out_date"] else pd.Timestamp.max, code_ids[str(row["index_code"])]))
    result = np.full(len(index), -1, dtype=np.int16)
    positions = pd.Series(np.arange(len(index)), index=index)
    for instrument, segment in positions.groupby(level="instrument", sort=False):
        offsets = segment.to_numpy()
        dates = pd.DatetimeIndex(segment.index.get_level_values("datetime"))
        for start, end, group_id in membership.get(instrument, []):
            mask = (dates >= start) & (dates <= end)
            if (result[offsets[mask]] >= 0).any():
                raise RuntimeError(f"历史行业重叠冲突: {instrument}")
            result[offsets[mask]] = group_id
    return pd.Series(result, index=index, name="industry_group"), codes


def calibrate(calibration_score, calibration_label, calibration_groups, prediction, prediction_groups):
    frame = pd.concat({"score": calibration_score, "label": calibration_label.reindex(calibration_score.index),
                       "group": calibration_groups.reindex(calibration_score.index)}, axis=1).dropna()
    frame["weight"] = 1.0 / frame.groupby(level="datetime")["score"].transform("size")
    mappings = {}
    for group_id in [-1] + sorted(frame["group"].unique().astype(int).tolist()):
        part = frame if group_id == -1 else frame.loc[frame["group"] == group_id]
        if len(part) == 0:
            continue
        x, y, weight = (part[column].to_numpy(dtype=float) for column in ("score", "label", "weight"))
        mx, my = np.average(x, weights=weight), np.average(y, weights=weight)
        variance = np.average((x - mx) ** 2, weights=weight)
        slope = np.average((x - mx) * (y - my), weights=weight) / variance if variance > 1e-12 else 0.0
        intercept = my - slope * mx
        if group_id == -1:
            if slope <= 1e-12:
                slope, intercept = 1.0, my - mx
            mappings[-1] = (intercept, slope)
        else:
            days = part.index.get_level_values("datetime").nunique()
            if days < 40 or slope <= 1e-12:
                mappings[group_id] = mappings[-1]
            else:
                shrink = days / (days + 40.0)
                mappings[group_id] = (shrink * intercept + (1 - shrink) * mappings[-1][0],
                                      shrink * slope + (1 - shrink) * mappings[-1][1])
    result = mappings[-1][0] + mappings[-1][1] * prediction
    target_groups = prediction_groups.reindex(prediction.index)
    for group_id, (intercept, slope) in mappings.items():
        if group_id >= 0:
            mask = target_groups == group_id
            result.loc[mask] = intercept + slope * prediction.loc[mask]
    return result.astype(np.float32), mappings


def load_fold(manifest, fold_name):
    window = manifest["folds"][fold_name]
    warm_start = pd.Timestamp(window["train"][0]) - pd.DateOffset(months=15)
    end = pd.Timestamp(window["test"][1])
    predicate = [("datetime", ">=", warm_start), ("datetime", "<=", end)]
    source_paths = sorted(SOURCE.glob("new_2*.parquet"))
    base = pd.concat([pd.read_parquet(path, filters=predicate) for path in source_paths]).sort_index()
    if base.index.has_duplicates or base.empty:
        raise RuntimeError("行情因子面板为空或股票日重复")
    context_path = SOURCE / "context/context_features.parquet"
    context = pd.read_parquet(context_path, filters=predicate).reindex(base.index)
    context_keys = json.loads(context_path.with_name("coverage.json").read_text(encoding="utf-8"))["feature_columns"]
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    preheat = pd.read_parquet(OUTPUT / "preheat_2018.parquet", filters=[("datetime", ">=", warm_start)])
    quotes = pd.concat([preheat, base[["open", "high", "low", "close", "volume"]]]).sort_index()
    daily = build_daily_features(quotes).reindex(base.index)
    monthly = build_monthly_features(quotes, calendar).reindex(base.index)
    history_industry, industry_codes = industry_groups(quotes.index)
    industry = history_industry.reindex(base.index)
    group = pd.read_parquet(OUTPUT / "classification_snapshot.parquet",
                            filters=[("datetime", ">=", pd.Timestamp(base.index.get_level_values("datetime").min())),
                                     ("datetime", "<=", end)]).reindex(base.index)
    if group.isna().any().any():
        raise RuntimeError("冻结分类快照未覆盖共同股票日")
    group_audit = {name: [] for name in group}
    calibration_group = ((industry.clip(lower=0) * 9 + group["GS8"].clip(lower=0))
                         .where((industry >= 0) & (group["GS8"] >= 0), -1)).astype(np.int16)
    dates = pd.DatetimeIndex(base.index.get_level_values("datetime"))
    eligible = (np.isfinite(base["label"]) & (base["label"] > -1) &
                np.isfinite(base["close"]) & (base["close"] > 0) &
                np.isfinite(base["volume"]) & (base["volume"] > 0))
    denominator_root = ROOT / "data/historical_repair/denominator/d8f7e2602f5b"
    expected = pd.concat([pd.read_parquet(denominator_root / f"expected_keys_{year}.parquet",
                                          columns=["code", "trade_date", "full_day_suspended"])
                          for year in range(max(2019, warm_start.year), end.year + 1)])
    expected = expected.loc[~expected["full_day_suspended"]]
    expected_index = pd.MultiIndex.from_arrays([pd.to_datetime(expected["trade_date"]), expected["code"].astype(str).str.zfill(6)])
    source_index = pd.MultiIndex.from_arrays([dates, base.index.get_level_values("instrument").str[2:]])
    eligible &= source_index.isin(expected_index)
    if context[context_keys].isna().any(axis=1).loc[eligible].any():
        raise RuntimeError("合格共同样本的历史上下文股票日不齐")
    label_raw = base["label"].where(eligible)
    label_rank = label_raw.groupby(level="datetime").rank(pct=True).sub(0.5).mul(3.46)
    original = base[manifest["factor_keys"]]
    wrapped = CSWinsorize()(pd.concat({"feature": original}, axis=1))
    if np.isinf(wrapped.to_numpy()).any():
        with parallel_backend("threading"):
            wrapped = ProcessInf()(wrapped)
    original = Fillna()(wrapped)["feature"].astype(np.float32)
    daily = daily.assign(**{f"missing_{name}": daily[name].isna().astype(np.float32) for name in daily}).fillna(0)
    monthly = monthly.assign(**{f"missing_{name}": monthly[name].isna().astype(np.float32) for name in monthly}).fillna(0)
    extra_context = pd.DataFrame(index=base.index)
    for group_id in range(len(industry_codes)):
        extra_context[f"industry_{group_id}"] = (industry == group_id).astype(np.float32)
    for name, count in (("SL4", 4), ("PB4", 4), ("IC6", 6), ("GS8", 8)):
        for group_id in range(count):
            extra_context[f"{name}_{group_id}"] = (group[name] == group_id).astype(np.float32)
        extra_context[f"{name}_unknown"] = (group[name] < 0).astype(np.float32)
    common = pd.concat([context[context_keys].astype(np.float32), extra_context], axis=1)
    arms = {
        "D": original,
        "L": pd.concat([original, daily], axis=1),
        "DM": pd.concat([original, monthly], axis=1),
        "LM": pd.concat([original, daily, monthly], axis=1),
        "U0": pd.concat([original, common], axis=1),
        "UM": pd.concat([original, monthly, common], axis=1),
    }
    audit = {"rows": len(base), "eligible_rows": int(eligible.sum()), "industry_groups": len(industry_codes),
             "unknown_industry_share": float((industry < 0).mean()),
             "unknown_groups": {name: float((group[name] < 0).mean()) for name in group},
             "group_months": group_audit,
             "feature_counts": {arm: len(frame.columns) for arm, frame in arms.items()},
             "context_missing_rates": {key: float(context[key].mean()) for key in context_keys if key.startswith("context_missing_")}}
    del base, context, original, common, expected, expected_index, source_index
    gc.collect()
    return arms, group, calibration_group, label_raw, label_rank, eligible, audit


def train(args):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if sha256(OUTPUT / "preheat_2018.parquet") != manifest["preheat_sha256"]:
        raise RuntimeError("2018年预热快照已变化")
    if sha256(OUTPUT / "classification_snapshot.parquet") != manifest["classification_snapshot_sha256"]:
        raise RuntimeError("全历史分类快照已变化")
    if sha256(OUTPUT / "stock_index_membership_snapshot.json") != manifest["membership_snapshot_sha256"]:
        raise RuntimeError("行业归属冻结快照已变化")
    for name, expected in manifest["source_files_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"冻结输入已变动: {name}")
    for name, expected in manifest.get("code_files_sha256", {}).items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"冻结研究代码已变动: {name}")
    for fold_name in args.folds:
        window = manifest["folds"][fold_name]
        arms, groups, calibration_groups, label_raw, label_rank, eligible, audit = load_fold(manifest, fold_name)
        (OUTPUT / f"{fold_name}_input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        dates = pd.DatetimeIndex(label_raw.index.get_level_values("datetime"))
        masks = {name: (dates >= pd.Timestamp(window[name][0])) & (dates <= pd.Timestamp(window[name][1])) & eligible
                 for name in ("train", "valid", "calibrate", "test")}
        for seed in args.seeds:
            params = {**DEFAULT_PARAMS["lightgbm"], **PARAM_PROFILES["conservative"]["lightgbm"],
                      "num_threads": 4, "seed": seed, "bagging_seed": seed,
                      "feature_fraction_seed": seed, "data_random_seed": seed}
            params.pop("verbose_eval", None)
            raw_scores = {}
            details = {}
            for arm in ("D", "L", "DM", "LM", "U0", "UM"):
                output = OUTPUT / f"{fold_name}_{seed}_{arm}.checkpoint.pkl"
                if output.exists():
                    with output.open("rb") as handle:
                        data = pickle.load(handle)
                    raw_scores[arm], details[arm] = data["scores"], data["details"]
                    continue
                x = arms[arm]
                frames = {segment: pd.concat({"feature": x.loc[masks[segment]],
                                              "label": label_rank.loc[masks[segment]].to_frame("LABEL0")}, axis=1)
                          for segment in ("train", "valid")}
                model = RankLGBModel(**params)
                start = time.perf_counter()
                model.fit(_PreparedFactorDataset(frames, list(x.columns)), verbose_eval=0, log_metrics=False)
                prediction = {segment: pd.Series(model.model.predict(x.loc[masks[segment]].to_numpy(), num_threads=4),
                                                 index=x.index[masks[segment]]) for segment in ("calibrate", "test")}
                raw_scores[arm] = prediction
                details[arm] = {"elapsed_seconds": time.perf_counter() - start, "best_iteration": model.model.best_iteration,
                                "feature_count": len(x.columns)}
                temporary = output.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    pickle.dump({"scores": prediction, "details": details[arm]}, handle, protocol=5)
                temporary.replace(output)
                (OUTPUT / f"{fold_name}_{seed}_{arm}.txt").write_text(model.model.model_to_string(), encoding="utf-8")
                print(json.dumps({"fold": fold_name, "seed": seed, "arm": arm, **details[arm]}), flush=True)
                del model, frames
                gc.collect()
            for arm, group_name, baseline in ((f"{name}_{suffix}", name, unified)
                                               for suffix, unified in (("0", "U0"), ("M", "UM"))
                                               for name in ("SL4", "PB4", "IC6", "GS8")):
                x = arms[baseline]
                group_ids = groups[group_name]
                raw_scores[arm] = {segment: raw_scores[baseline][segment].copy() for segment in ("calibrate", "test")}
                fallback = pd.Series(True, index=raw_scores[arm]["test"].index)
                details[arm] = {"experts": {}, "fallback_rows": None}
                for group_id in sorted(group_ids[group_ids >= 0].unique()):
                    group_masks = {segment: masks[segment] & (group_ids == group_id) for segment in ("train", "valid", "calibrate", "test")}
                    index = x.index[group_masks["train"]]
                    info = {"rows": len(index), "stocks": index.get_level_values("instrument").nunique(),
                            "days": index.get_level_values("datetime").nunique(),
                            "calibration_days": x.index[group_masks["calibrate"]].get_level_values("datetime").nunique()}
                    if info["rows"] < 50000 or info["stocks"] < 100 or info["days"] < 500 or info["calibration_days"] < 40:
                        info["status"] = "sample_gate_fallback"
                        details[arm]["experts"][str(group_id)] = info
                        continue
                    output = OUTPUT / f"{fold_name}_{seed}_{arm}_{group_id}.checkpoint.pkl"
                    if output.exists():
                        with output.open("rb") as handle:
                            data = pickle.load(handle)
                        prediction, info = data["scores"], data["details"]
                    else:
                        frames = {segment: pd.concat({"feature": x.loc[group_masks[segment]],
                                                      "label": label_rank.loc[group_masks[segment]].to_frame("LABEL0")}, axis=1)
                                  for segment in ("train", "valid")}
                        model = RankLGBModel(**params)
                        start = time.perf_counter()
                        try:
                            model.fit(_PreparedFactorDataset(frames, list(x.columns)), verbose_eval=0, log_metrics=False)
                            prediction = {segment: pd.Series(model.model.predict(x.loc[group_masks[segment]].to_numpy(), num_threads=4),
                                                             index=x.index[group_masks[segment]]) for segment in ("calibrate", "test")}
                        except Exception as error:
                            info["status"] = "fit_failure_fallback"
                            info["reason"] = f"{type(error).__name__}: {str(error)[:300]}"
                            details[arm]["experts"][str(group_id)] = info
                            continue
                        info.update({"status": "trained", "elapsed_seconds": time.perf_counter() - start,
                                     "best_iteration": model.model.best_iteration})
                        temporary = output.with_suffix(".tmp")
                        with temporary.open("wb") as handle:
                            pickle.dump({"scores": prediction, "details": info}, handle, protocol=5)
                        temporary.replace(output)
                        (OUTPUT / f"{fold_name}_{seed}_{arm}_{group_id}.txt").write_text(model.model.model_to_string(), encoding="utf-8")
                        print(json.dumps({"fold": fold_name, "seed": seed, "arm": arm, "group": int(group_id), **info}), flush=True)
                    details[arm]["experts"][str(group_id)] = info
                    for segment in ("calibrate", "test"):
                        raw_scores[arm][segment].loc[prediction[segment].index] = prediction[segment]
                    fallback.loc[prediction["test"].index] = False
                    del prediction
                    gc.collect()
                details[arm]["fallback_rows"] = int(fallback.sum())
                details[arm]["fallback_share"] = float(fallback.mean())
                details[arm]["fallback_mask"] = fallback
            calibrated_scores = {}
            for arm, values in raw_scores.items():
                score, maps = calibrate(values["calibrate"], label_raw, calibration_groups,
                                        values["test"], calibration_groups)
                if arm not in ("D", "L", "DM", "LM", "U0", "UM"):
                    fallback_mask = details[arm]["fallback_mask"]
                    baseline = "UM" if arm.endswith("_M") else "U0"
                    score.loc[fallback_mask] = calibrated_scores[baseline].loc[fallback_mask]
                if not score.index.equals(label_raw.loc[masks["test"]].index) or score.isna().any() or not np.isfinite(score).all():
                    raise RuntimeError(f"{fold_name} {seed} {arm} 预测未覆盖共同样本")
                calibrated_scores[arm] = score
                result = pd.DataFrame({"raw_score": values["test"], "score": score,
                                       "label": label_raw.loc[masks["test"]],
                                       **{f"{name}_group": groups.loc[masks["test"], name] for name in groups},
                                       "fallback": details[arm].get("fallback_mask", pd.Series(False, index=score.index))})
                path = OUTPUT / f"{fold_name}_{seed}_{arm}.parquet"
                temporary = path.with_suffix(".tmp")
                result.to_parquet(temporary, compression="zstd")
                temporary.replace(path)
                details[arm].pop("fallback_mask", None)
                details[arm]["calibration_maps"] = {str(key): list(value) for key, value in maps.items()}
            (OUTPUT / f"{fold_name}_{seed}_training.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            del raw_scores
            gc.collect()
        del arms, groups, calibration_groups, label_raw, label_rank, eligible
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if set(args.folds) - {f"W{i}" for i in range(1, 9)} or set(args.seeds) - set(SEEDS):
        raise ValueError("折或种子不在冻结契约内")
    train(args)
