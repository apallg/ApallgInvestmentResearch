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
os.environ.setdefault("HORIZON_RESEARCH_OUTPUT", "data/experiments/horizon_multiscale_research_v2_4m")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.data.dataset.handler import DataHandlerLP

from core.adapters.qlib.factor_set_evaluator import _PreparedFactorDataset
from core.adapters.qlib.models import DEFAULT_PARAMS, PARAM_PROFILES
from core.adapters.qlib.rank_models import RankLGBModel, _rank_ic_feval
from scripts.horizon_multiscale_research import OUTPUT as LEGACY_OUTPUT, SEEDS, calendar_index
from scripts.run_multiscale_group_research import sha256
from scripts.train_horizon_classification import calibrate, load_fold


LEGACY_OUTPUT = LEGACY_OUTPUT.resolve()
OUTPUT = ROOT / "data/experiments/model_complementarity_research_v1_20261105"
FOLDS = tuple(f"W{i}" for i in range(1, 9))
HORIZONS = ("H10", "H20")
DE_ARMS = ("BFS", "BSR", "BDE")
BASE_ARMS = ("B3", "BFS", "BSR", "BDE", "BUD_FS", "BUD_SR", "BUD_DE")
BASE_INPUT = tuple(__import__("json").loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))["factor_keys"])


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def task_signature(fold, horizon, seed, arm):
    payload = {"legacy_manifest": sha256(LEGACY_OUTPUT / "manifest.json"),
               "legacy_label_audit": sha256(LEGACY_OUTPUT / "label_audit.json"),
               "fold": fold, "horizon": horizon, "seed": seed, "arm": arm,
               "features": list(BASE_INPUT), "code": sha256(Path(__file__))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def params_for_seed(seed):
    params = {**DEFAULT_PARAMS["lightgbm"], **PARAM_PROFILES["conservative"]["lightgbm"],
              "num_threads": 4, "seed": seed, "bagging_seed": seed,
              "feature_fraction_seed": seed, "data_random_seed": seed}
    params.pop("verbose_eval", None)
    return params


def frames_for(inputs, labels, masks):
    x = inputs["D"][list(BASE_INPUT)].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    frames = {name: pd.concat({"feature": x.loc[masks[name]],
                              "label": labels.loc[masks[name], "label_rank"].to_frame("LABEL0")}, axis=1)
              for name in ("train", "valid")}
    return x, frames


def legacy_raw_scores(fold, horizon, seed):
    path = LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_sequence_lgb.pkl"
    data = pickle.loads(path.read_bytes())
    expected = hashlib.sha256(json.dumps({"manifest": sha256(LEGACY_OUTPUT / "manifest.json"),
        "labels": sha256(LEGACY_OUTPUT / "label_audit.json"), "horizon": horizon, "fold": fold,
        "seed": seed, "arm": "LGB40", "parameters": json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))["lightgbm_parameters"],
        "epochs": 80, "batch": 4096}, sort_keys=True).encode()).hexdigest()
    if data.get("signature") != expected:
        raise RuntimeError(f"旧LGB原始分数签名不一致: {path.name}")
    return data["scores"]


def fit_plain_member(fold, horizon, outer_seed, member, x, frames, labels, masks, index):
    seed = outer_seed + 10000 * member
    cache_path = OUTPUT / "member_cache" / f"{fold}_{horizon}_{outer_seed}_{member}.pkl"
    model_path = OUTPUT / "member_cache" / f"{fold}_{horizon}_{outer_seed}_{member}.txt"
    signature = task_signature(fold, horizon, outer_seed, f"plain_member_{member}")
    if cache_path.exists() and model_path.exists():
        data = pickle.loads(cache_path.read_bytes())
        if data.get("signature") != signature:
            raise RuntimeError(f"普通集成成员续跑签名不一致: {cache_path.name}")
        return data["scores"], float(data["elapsed_seconds"]), model_path
    params = params_for_seed(seed)
    model = RankLGBModel(**params)
    started = time.perf_counter()
    model.fit(_PreparedFactorDataset(frames, list(BASE_INPUT)), verbose_eval=0, log_metrics=False)
    scores = {segment: pd.Series(model.model.predict(x.loc[masks[segment], list(BASE_INPUT)].to_numpy(), num_threads=4),
                                 index=index[masks[segment]]) for segment in ("calibrate", "test")}
    elapsed = time.perf_counter() - started
    temp_model = model_path.with_suffix(".tmp")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model.write_text(model.model.model_to_string(), encoding="utf-8")
    temp_model.replace(model_path)
    restored = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    sample = x.loc[masks["test"], list(BASE_INPUT)].to_numpy()[:10000]
    restore_error = float(np.max(np.abs(restored.predict(sample, num_threads=4) -
                                      model.model.predict(sample, num_threads=4)))) if len(sample) else 0.0
    if restore_error > 1e-6:
        raise RuntimeError(f"普通集成成员模型恢复误差超过1e-6: {restore_error}")
    temp_cache = cache_path.with_suffix(".tmp")
    temp_cache.write_bytes(pickle.dumps({"signature": signature, "scores": scores,
        "elapsed_seconds": elapsed, "best_iteration": model.model.best_iteration,
        "restore_max_abs_error": restore_error}, protocol=5))
    temp_cache.replace(cache_path)
    del model, restored
    gc.collect()
    return scores, elapsed, model_path


class MemoryBoundDoubleEnsemble(DEnsembleModel):
    def __init__(self, base_seed, enable_sr, enable_fs, params):
        self.base_seed = base_seed
        clean = dict(params)
        loss = clean.pop("loss", "mse")
        early_stopping_rounds = clean.pop("early_stopping_rounds", 50)
        clean.pop("verbose_eval", None)
        super().__init__(base_model="gbm", loss=loss, num_models=3, enable_sr=enable_sr,
                         enable_fs=enable_fs, alpha1=1.0, alpha2=1.0, bins_sr=10, bins_fs=5,
                         decay=1.0, sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
                         sub_weights=[1.0, 1.0, 1.0], epochs=1000,
                         early_stopping_rounds=early_stopping_rounds, **clean)

    def train_submodel(self, df_train, df_valid, weights, features):
        member = len(self.ensemble)
        params = dict(self.params)
        seed = self.base_seed + 10000 * member
        params.update({"metric": "None", "num_threads": 4, "seed": seed,
                       "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed})
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"].values.squeeze()
        x_valid, y_valid = df_valid["feature"].loc[:, features], df_valid["label"].values.squeeze()
        dtrain = lgb.Dataset(x_train, label=y_train, weight=weights.to_numpy(dtype=np.float64))
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        day_ids = df_valid.index.get_level_values("datetime").factorize(sort=False)[0]
        dvalid._day_ids = day_ids
        dvalid._label_rank = pd.Series(y_valid).groupby(day_ids).rank().to_numpy()
        callbacks = [lgb.log_evaluation(0)]
        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False))
        return lgb.train(params, dtrain, num_boost_round=self.epochs,
                         valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
                         feval=_rank_ic_feval, callbacks=callbacks)

    def retrieve_loss_curve(self, model, df_train, features):
        x_train = df_train["feature"].loc[:, features]
        y_train = df_train["label"].values.squeeze()
        rows, trees = x_train.shape[0], model.num_trees()
        part = max(int(trees * 0.1), 1)
        start_sum, end_sum = np.zeros(rows), np.zeros(rows)
        start_count = end_count = 0
        prediction = np.zeros(rows)
        matrix = x_train.to_numpy()
        for tree in range(trees):
            prediction += model.predict(matrix, start_iteration=tree, num_iteration=1)
            if tree < part or tree >= trees - part:
                ranks = pd.Series(np.square(y_train - prediction)).rank(pct=True).to_numpy()
                if tree < part:
                    start_sum += ranks
                    start_count += 1
                if tree >= trees - part:
                    end_sum += ranks
                    end_count += 1
        return pd.DataFrame({"start": start_sum / max(start_count, 1),
                             "end": end_sum / max(end_count, 1)})

    def sample_reweight(self, loss_curve, loss_values, k_th):
        h1 = (-loss_values).rank(pct=True).to_numpy()
        h2 = pd.Series(loss_curve["end"].to_numpy() / np.maximum(loss_curve["start"].to_numpy(), 1e-12)).rank(pct=True).to_numpy()
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins", observed=False)["h_value"].mean()
        weights = np.zeros(len(h), dtype=np.float64)
        for bucket in h_avg.index:
            weights[(h["bins"] == bucket).to_numpy()] = 1.0 / (self.decay ** k_th * h_avg[bucket] + 0.1)
        return pd.Series(weights)

    def feature_selection(self, df_train, loss_values):
        x_train, y_train = df_train["feature"], df_train["label"].values.squeeze()
        features, model_count = x_train.columns, len(self.ensemble)
        x_shuffled = x_train.copy()
        g_values = np.zeros(len(features), dtype=np.float64)
        for i, feature in enumerate(features):
            x_shuffled.loc[:, feature] = np.random.permutation(x_shuffled[feature].to_numpy())
            prediction = np.zeros(len(x_train), dtype=np.float64)
            for model, selected in zip(self.ensemble, self.sub_features):
                prediction += model.predict(x_shuffled.loc[:, selected].to_numpy()) / model_count
            loss = np.square(y_train - prediction)
            difference = loss - loss_values.to_numpy()
            g_values[i] = np.mean(difference) / (np.std(difference) + 1e-7)
            x_shuffled.loc[:, feature] = x_train[feature].to_numpy()
        grouped = pd.DataFrame({"g_value": g_values}, index=features)
        grouped["bins"] = pd.cut(grouped["g_value"].replace(np.nan, 0), self.bins_fs)
        selected = []
        for i, bucket in enumerate(sorted(grouped["bins"].dropna().unique(), reverse=True)):
            bucket_features = grouped.index[grouped["bins"] == bucket].to_numpy()
            count = int(np.ceil(self.sample_ratios[min(i, len(self.sample_ratios) - 1)] * len(bucket_features)))
            if count:
                selected.extend(np.random.choice(bucket_features, size=count, replace=False).tolist())
        selected = sorted(set(selected))
        if not selected:
            raise RuntimeError("DoubleEnsemble特征选择结果为空")
        return pd.Index(selected)


def fit_double_arm(arm, fold, horizon, seed, x, frames, labels, masks, calibration_group, index):
    signature = task_signature(fold, horizon, seed, arm)
    raw_path = OUTPUT / "diversity_cache" / f"{fold}_{horizon}_{seed}_{arm}.pkl"
    model_meta_path = OUTPUT / "diversity_models" / f"{fold}_{horizon}_{seed}_{arm}.json"
    if raw_path.exists() and model_meta_path.exists():
        data = pickle.loads(raw_path.read_bytes())
        meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
        if data.get("signature") != signature or meta.get("signature") != signature:
            raise RuntimeError(f"DoubleEnsemble续跑签名不一致: {arm}")
        return data["scores"], float(meta["elapsed_seconds"])
    started = time.perf_counter()
    params = params_for_seed(seed)
    de = MemoryBoundDoubleEnsemble(seed, enable_sr=arm in ("BSR", "BDE"),
                                   enable_fs=arm in ("BFS", "BDE"), params=params)
    np.random.seed(seed)
    dataset = _PreparedFactorDataset(frames, list(BASE_INPUT))
    de.fit(dataset)
    scores = {}
    for segment in ("calibrate", "test"):
        selected = x.loc[masks[segment], list(BASE_INPUT)]
        prediction = np.zeros(len(selected), dtype=np.float64)
        total_weight = sum(de.sub_weights)
        for model, features, weight in zip(de.ensemble, de.sub_features, de.sub_weights):
            prediction += model.predict(selected.loc[:, features].to_numpy(), num_threads=4) * weight
        scores[segment] = pd.Series(prediction / total_weight, index=index[masks[segment]])
    elapsed = time.perf_counter() - started
    members = []
    model_dir = OUTPUT / "diversity_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for member, (model, features) in enumerate(zip(de.ensemble, de.sub_features)):
        path = model_dir / f"{fold}_{horizon}_{seed}_{arm}_{member}.txt"
        temp = path.with_suffix(".tmp")
        temp.write_text(model.model_to_string(), encoding="utf-8")
        temp.replace(path)
        restored = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        sample = x.loc[masks["test"], list(features)].to_numpy()[:10000]
        expected = model.predict(sample, num_threads=4) if len(sample) else np.empty(0)
        actual = restored.predict(sample, num_threads=4) if len(sample) else np.empty(0)
        error = float(np.max(np.abs(expected - actual))) if len(sample) else 0.0
        if error > 1e-6:
            raise RuntimeError(f"DoubleEnsemble模型恢复误差超过1e-6: {fold}/{horizon}/{seed}/{arm}/{member}={error}")
        members.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "features": list(features),
                        "restore_max_abs_error": error})
    data = {"signature": signature, "scores": scores}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temp_raw = raw_path.with_suffix(".tmp")
    temp_raw.write_bytes(pickle.dumps(data, protocol=5))
    temp_raw.replace(raw_path)
    meta = {"signature": signature, "arm": arm, "elapsed_seconds": elapsed,
            "num_models": len(members), "members": members,
            "enable_sr": de.enable_sr, "enable_fs": de.enable_fs,
            "loss_curve_storage": "streamed rank means, O(N) auxiliary memory",
            "decay": 1.0, "early_stopping_rounds": de.early_stopping_rounds,
            "epochs": de.epochs, "base_seed": seed,
            "recovery_max_abs_error": max(item["restore_max_abs_error"] for item in members)}
    save_json(model_meta_path, meta)
    del de, dataset
    gc.collect()
    return scores, elapsed


def fit_diversity_task(fold, horizon, seed, prereun=False):
    inputs, groups, calibration_group, labels, masks, load_audit = load_fold(fold, int(horizon[1:]))
    x, frames = frames_for(inputs, labels, masks)
    index = inputs["D"].index
    started = time.perf_counter()
    double_times, metrics = {}, {}
    for arm in DE_ARMS:
        _, elapsed = fit_double_arm(arm, fold, horizon, seed, x, frames, labels, masks, calibration_group, index)
        double_times[arm] = elapsed
        metrics[arm] = {"elapsed_seconds": elapsed}
    original_raw = legacy_raw_scores(fold, horizon, seed)
    base_elapsed = None
    budget_file = OUTPUT / "budget_config.json"
    task_arms = BASE_ARMS if budget_file.exists() and not prereun else ("B3", *DE_ARMS)
    for arm in task_arms:
        if arm in DE_ARMS:
            raw_path = OUTPUT / "diversity_cache" / f"{fold}_{horizon}_{seed}_{arm}.pkl"
            raw = pickle.loads(raw_path.read_bytes())["scores"]
        else:
            member_count = 3 if arm == "B3" else int(json.loads(budget_file.read_text(encoding="utf-8"))["members"][arm])
            member_scores = {0: original_raw}
            for member in range(1, member_count):
                score, elapsed, model_path = fit_plain_member(fold, horizon, seed, member, x, frames, labels, masks, index)
                member_scores[member] = score
                if member == 1:
                    base_elapsed = elapsed
            raw = {}
            for segment in ("calibrate", "test"):
                ranked = [value[segment].groupby(level="datetime").rank(pct=True) for value in member_scores.values()]
                raw[segment] = sum(ranked) / len(ranked)
            metrics[arm] = {"member_count": member_count, "plain_member_seconds": base_elapsed}
        if arm in DE_ARMS:
            raw = {segment: raw[segment].groupby(level="datetime").rank(pct=True) for segment in ("calibrate", "test")}
        calibrated, maps = calibrate(raw["calibrate"], labels["label"], calibration_group,
                                     raw["test"], calibration_group)
        frame = pd.DataFrame({"score": calibrated, "label": labels["label"].reindex(calibrated.index),
                              "label_rank": labels["label_rank"].reindex(calibrated.index)})
        if not frame.index.equals(index[masks["test"]]) or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
            raise RuntimeError(f"{fold}/{horizon}/{seed}/{arm}: 输出覆盖或分数异常")
        out = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
        signature = task_signature(fold, horizon, seed, arm)
        temp = out.with_suffix(".tmp")
        frame.to_parquet(temp, compression="zstd")
        temp.replace(out)
        save_json(out.with_suffix(".json"), {"signature": signature, "arm": arm,
            "rows": len(frame), "calibration_rows": len(raw["calibrate"]), "metrics": metrics.get(arm),
            "recovery_max_abs_error": 0.0, "calibration_group": "historical_industry_x_GS8"})
        del raw, frame
        gc.collect()
    if prereun:
        save_json(OUTPUT / "prerun" / f"{fold}_{horizon}_{seed}.json", {
            "fold": fold, "horizon": horizon, "seed": seed,
            "base_model_seconds": base_elapsed, "double_ensemble_seconds": double_times,
            "elapsed_seconds": time.perf_counter() - started, "sample_counts": load_audit["segments"]})
    del inputs, groups, calibration_group, labels, masks, x, frames
    gc.collect()
    return base_elapsed, double_times


def freeze_budget():
    paths = sorted((OUTPUT / "prerun").glob("W*_H*_42.json"))
    if len(paths) != 4:
        raise RuntimeError(f"等预算冻结要求W1/W8×H10/H20四个预跑，当前{len(paths)}")
    data = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    base = [row["base_model_seconds"] for row in data if row["base_model_seconds"]]
    if len(base) != 4:
        raise RuntimeError("缺少单模型耗时，不能冻结预算成员数")
    mean_base = float(np.mean(base))
    ratios, members = {}, {}
    for arm in DE_ARMS:
        times = [row["double_ensemble_seconds"][arm] for row in data]
        ratio = float(np.mean(times) / mean_base)
        ratios[arm] = {"mean_seconds": float(np.mean(times)), "mean_model_seconds": mean_base,
                       "time_ratio": ratio, "measurements_seconds": times}
        suffix = {"BFS": "BUD_FS", "BSR": "BUD_SR", "BDE": "BUD_DE"}[arm]
        members[suffix] = max(1, int(np.ceil(ratio)))
    result = {"status": "frozen", "basis": "mean W1/W8 and H10/H20 DoubleEnsemble elapsed divided by mean single LightGBM elapsed",
              "ratios": ratios, "members": members, "extra_member_seed": "outer_seed+10000*member_index",
              "frozen_before_primary_evaluation": True}
    save_json(OUTPUT / "budget_config.json", result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prerun", "freeze-budget", "train", "status"))
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if args.stage == "status":
        done = sum(1 for horizon in HORIZONS for fold in FOLDS for seed in SEEDS for arm in BASE_ARMS
                   if (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet").exists())
        print(json.dumps({"completed_predictions": done, "planned_predictions": 336,
                          "budget_config_exists": (OUTPUT / "budget_config.json").exists()}, ensure_ascii=False))
        return
    if args.stage == "freeze-budget":
        freeze_budget()
        return
    if args.stage == "prerun":
        tasks = [(fold, horizon, 42) for fold in ("W1", "W8") for horizon in HORIZONS]
        for fold, horizon, seed in tasks:
            fit_diversity_task(fold, horizon, seed, prereun=True)
        freeze_budget()
    else:
        if not (OUTPUT / "budget_config.json").exists():
            raise RuntimeError("完整训练前必须冻结等预算成员数")
        for fold in args.folds:
            for horizon in args.horizons:
                for seed in args.seeds:
                    fit_diversity_task(fold, horizon, seed, prereun=False)


if __name__ == "__main__":
    main()
