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
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import parallel_backend
from core.adapters.qlib.custom_processors import CSWinsorize
from core.adapters.qlib.factor_set_evaluator import _PreparedFactorDataset
from core.adapters.qlib.models import DEFAULT_PARAMS, PARAM_PROFILES
from core.adapters.qlib.rank_models import RankLGBModel
from qlib.data.dataset.processor import Fillna, ProcessInf

parser = argparse.ArgumentParser()
parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
args = parser.parse_args()
OUT = ROOT / "data/experiments/repair_group_research_20261025"
factors = json.loads((OUT / "factors.json").read_text(encoding="utf-8"))
audit = json.loads((OUT / "input_audit.json").read_text(encoding="utf-8"))
preflight = json.loads((OUT / "preflight.json").read_text(encoding="utf-8"))
denominator_root = ROOT / "data/historical_repair/denominator" / preflight["quality"]["denominator"]["batch_id"]
keys = list(factors)
calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
windows = {}
for fold in range(1, 9):
    test_start = pd.Timestamp("2023-01-01") + pd.DateOffset(months=6 * (fold - 1))
    valid_start = test_start - pd.DateOffset(months=6)
    calibration_start = test_start - pd.DateOffset(months=3)
    train_start = valid_start - pd.DateOffset(years=3)
    train_end = calendar[calendar < valid_start][-12]
    valid_end = calendar[calendar < calibration_start][-12]
    calibration_end = calendar[calendar < test_start][-12]
    test_end = min(test_start + pd.DateOffset(months=6) - pd.Timedelta(days=1), pd.Timestamp("2026-09-15"))
    windows[f"W{fold}"] = {"train": [str(train_start.date()), str(train_end.date())], "valid": [str(valid_start.date()), str(valid_end.date())], "calibrate": [str(calibration_start.date()), str(calibration_end.date())], "test": [str(test_start.date()), str(test_end.date())]}
if set(args.folds) - set(windows):
    raise ValueError("未知测试折")
context_path = OUT / "context/context_features.parquet"
coverage = json.loads(context_path.with_name("coverage.json").read_text(encoding="utf-8"))
if coverage["feature_contract_version"] != 2 or coverage["membership_interval_conflicts"]:
    raise RuntimeError("上下文结构或区间冲突检查未通过")
context_keys = coverage["feature_columns"]
for generation in ("old", "new"):
    if not (OUT / f"{generation}_extraction_complete.json").exists():
        raise RuntimeError(f"{generation} 数据抽取尚未完成")
native = pd.read_parquet(OUT / "native_reference_2019.parquet")
equivalent = pd.read_parquet(OUT / "new_2019.parquet")
if not native.equals(equivalent):
    raise RuntimeError("原生和等价时序排名的完整分片不一致，禁止训练")
(OUT / "ts_rank_equivalence.json").write_text(json.dumps({"passed": True, "rows": len(native), "columns": len(native.columns), "comparison": "exact DataFrame.equals, all 40 factors plus labels and prices over 2019-2020"}, indent=2), encoding="utf-8")
del native, equivalent
gc.collect()
input_files = {}
for path in [OUT / f"{generation}_{year}.parquet" for generation in ("old", "new") for year in (2019, 2021, 2023, 2025)] + [OUT / "native_reference_2019.parquet", context_path] + sorted(denominator_root.glob("expected_keys_*.parquet")):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    input_files[str(path.relative_to(ROOT))] = digest.hexdigest()
manifest = {
    "research_only": True, "strict_pit_verified": False, "deployment_eligible": False,
    "authorization": "用户2026-10-25明确选择先做隔离研究实验，结果不得作为严格样本外或上线证据",
    "model_family": "LightGBM with validation RankIC early stopping; no LSTM",
    "data_effect_scope": audit["scope"], "folds": {name: windows[name] for name in args.folds}, "seeds": args.seeds,
    "arms": {"A0": "2026-10-25 prewrite inputs, pooled 40 features", "A1": "current inputs, pooled same 40 features", "B": "current inputs, pooled 40+39 context features", "C": "same B features, six industry experts with calibrated B fallback"},
    "comparisons": {"A1-A0": "today's input generation effect only", "B-A1": "context information increment", "C-B": "splitting increment", "C-A1": "combined increment"},
    "label": "Open(t+11)/Open(t+1)-1; none; every 10 calendar trading sessions",
    "evaluation": "paired same-row RankIC and Top5/Top50 label-basket gross return; not an execution/account backtest",
    "sample_rule": "current provider intervals intersect frozen G0 expected tradable stock-days; common finite label greater than -1 (exclude zero entry/exit artifacts); positive close and volume at signal; identical arms; features may be missing and filled under common rule",
    "denominator_batch": preflight["quality"]["denominator"]["batch_id"],
    "preprocessing": "original 40: existing daily CSWinsorize, ProcessInf, Fillna; context already includes missing flags; fit no future statistics",
    "calibration": "distinct purged quarter, daily equal weights; positive pooled slope else identity slope; group shrinks D/(D+40), <40 dates or nonpositive slope uses pooled map",
    "expert_minimum": {"stocks": 100, "days": 500, "rows": 50000, "calibration_dates": 40},
    "seeds_are_independent_market_samples": False,
    "limitations": ["historical publication vintages and independent certification remain unavailable", "full pre-repair October dataset absent", "fixed existing factor list selected using past research", "no actual execution, limit/ST status or fee-adjusted NAV", "equal per-model parameters without tuning; expert system consumes more total compute", "cannot infer effects for LGB+LSTM or all possible stock groupings", "expanded 2019 context panel misses 98% overall coverage for market/industry relations", "shared research Corr computation aligns constant masks by date for unequal feature intervals; production adapter unchanged"],
    "input_audit_sha256": hashlib.sha256((OUT / "input_audit.json").read_bytes()).hexdigest(),
    "factor_sha256": hashlib.sha256((OUT / "factors.json").read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_files_sha256": input_files,
    "source_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["core/adapters/qlib/rank_models.py", "core/adapters/qlib/custom_processors.py", "core/adapters/qlib/custom_handler.py", "scripts/extract_repair_group_features.py", "scripts/research_feature_ops.py"]},
    "context_coverage": coverage,
}
manifest_path = OUT / "manifest.json"
if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
    raise RuntimeError("实验契约已改变，禁止混用已有结果")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def calibrate(calibration_score, calibration_label, calibration_group, score, group):
    frame = pd.concat({"score": calibration_score, "label": calibration_label.reindex(calibration_score.index), "group": calibration_group.reindex(calibration_score.index)}, axis=1).dropna()
    frame["weight"] = 1.0 / frame.groupby(level="datetime")["score"].transform("size")
    maps = {}
    for group_id in [-1] + list(range(7)):
        part = frame if group_id == -1 else frame.loc[frame["group"] == group_id]
        days = part.index.get_level_values("datetime").nunique()
        if days == 0:
            maps[group_id] = maps[-1]
            continue
        x, y, weight = (part[column].to_numpy(dtype=float) for column in ("score", "label", "weight"))
        mx, my = np.average(x, weights=weight), np.average(y, weights=weight)
        variance = np.average((x - mx) ** 2, weights=weight)
        slope = np.average((x - mx) * (y - my), weights=weight) / variance if variance > 1e-12 else 0.0
        if group_id == -1:
            slope = slope if slope > 1e-12 else 1.0
            maps[-1] = (my - slope * mx, slope)
        elif days < 40 or slope <= 1e-12:
            maps[group_id] = maps[-1]
        else:
            shrink = days / (days + 40.0)
            maps[group_id] = (shrink * (my - slope * mx) + (1 - shrink) * maps[-1][0], shrink * slope + (1 - shrink) * maps[-1][1])
    result = pd.Series(np.nan, index=score.index)
    for group_id, (intercept, slope) in maps.items():
        if group_id >= 0:
            mask = group.reindex(result.index) == group_id
            result.loc[mask] = intercept + slope * score.loc[mask]
    return result, maps


for fold_name in args.folds:
    window = windows[fold_name]
    first, last = window["train"][0], window["test"][1]
    predicates = [("datetime", ">=", pd.Timestamp(first)), ("datetime", "<=", pd.Timestamp(last))]
    new = pd.concat([pd.read_parquet(path, filters=predicates) for path in sorted(OUT.glob("new_2*.parquet"))]).sort_index()
    old = pd.concat([pd.read_parquet(path, filters=predicates) for path in sorted(OUT.glob("old_2*.parquet"))]).sort_index()
    if not new.index.equals(old.index) or not np.array_equal(new["label"].to_numpy(), old["label"].to_numpy(), equal_nan=True):
        raise RuntimeError("旧新数据股票日或原始标签不一致，禁止归因")
    label = new["label"].rename("LABEL0")
    expected = pd.concat([pd.read_parquet(denominator_root / f"expected_keys_{year}.parquet", columns=["code", "trade_date", "full_day_suspended"]) for year in range(pd.Timestamp(first).year, pd.Timestamp(last).year + 1)])
    expected = expected.loc[~expected["full_day_suspended"]]
    expected_index = pd.MultiIndex.from_arrays([pd.to_datetime(expected["trade_date"]), expected["code"].astype(str).str.zfill(6)], names=["datetime", "code"])
    quote_index = pd.MultiIndex.from_arrays([new.index.get_level_values("datetime"), new.index.get_level_values("instrument").str[2:]], names=["datetime", "code"])
    in_expected = quote_index.isin(expected_index)
    eligible = in_expected & np.isfinite(label) & (label > -1) & np.isfinite(new["close"]) & np.isfinite(new["volume"]) & (new["close"] > 0) & (new["volume"] > 0)
    del expected, expected_index, quote_index
    if (new["open"] < 0).any():
        raise RuntimeError("开盘价出现负数，不能仅按收益率识别非法标签")
    (OUT / f"{fold_name}_sample_audit.json").write_text(json.dumps({"all_rows": len(new), "outside_verified_expected_days": int((~in_expected).sum()), "eligible_rows": int(eligible.sum()), "infinite_labels": int(np.isinf(label).sum()), "labels_at_most_minus_one": int((label <= -1).sum()), "nonpositive_signal_close": int((new["close"] <= 0).sum()), "nonpositive_signal_volume": int((new["volume"] <= 0).sum())}, indent=2), encoding="utf-8")
    context = pd.read_parquet(context_path, filters=predicates).reindex(new.index)
    aligned = context[context_keys].notna().all(axis=1)
    if not aligned.loc[eligible].all():
        raise RuntimeError(f"合格共同样本的上下文缺失{int((~aligned & eligible).sum())}行，禁止静默缩小样本")
    group = context["group_id"].fillna(6).astype(int)
    original_new = new[keys]
    original_old = original_new.copy()
    old_keys = [key for key in keys if key in old.columns]
    original_old.loc[:, old_keys] = old[old_keys]
    old_feature_changes = ((original_old != original_new) & ~(original_old.isna() & original_new.isna())).sum()
    (OUT / f"{fold_name}_feature_changes.json").write_text(json.dumps({key: int(value) for key, value in old_feature_changes.items()}, indent=2), encoding="utf-8")
    (OUT / f"{fold_name}_constant_features.json").write_text(json.dumps({arm: [key for key in keys if frame[key].dropna().nunique() <= 1] for arm, frame in (("A0", original_old), ("A1", original_new))}, indent=2), encoding="utf-8")
    identical_inputs = int(old_feature_changes.sum()) == 0
    processed = {}
    for arm, features in (([("A0", original_old)] if not identical_inputs else []) + [("A1", original_new)]):
        frame = pd.concat({"feature": features}, axis=1)
        frame = CSWinsorize()(frame)
        if np.isinf(frame.to_numpy()).any():
            with parallel_backend("threading"):
                frame = ProcessInf()(frame)
        frame = Fillna()(frame)
        processed[arm] = frame["feature"].astype(np.float32)
    if identical_inputs:
        processed["A0"] = processed["A1"]
    processed["B"] = pd.concat([processed["A1"], context[context_keys].fillna(0).astype(np.float32)], axis=1)
    del new, old, original_new, original_old, context, frame
    gc.collect()
    for seed in args.seeds:
        paths = {arm: OUT / f"{fold_name}_{seed}_{arm}.parquet" for arm in ("A0", "A1", "B", "C")}
        if all(path.exists() for path in paths.values()):
            continue
        scores = {}
        calibration_predictions = {}
        calibrations = {}
        expert_info = {}
        params = {**DEFAULT_PARAMS["lightgbm"], **PARAM_PROFILES["conservative"]["lightgbm"], "num_threads": 4, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed}
        params.pop("verbose_eval", None)
        models = ([] if identical_inputs else [("A0", None)]) + [("A1", None), ("B", None)] + [("C", group_id) for group_id in range(6)]
        for arm, group_id in models:
            destination = arm if group_id is None else f"C{group_id}"
            x = processed["B" if arm == "C" else arm]
            masks = {}
            for segment, (start, end) in window.items():
                dates = x.index.get_level_values("datetime")
                masks[segment] = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)) & eligible
                if group_id is not None:
                    masks[segment] &= group == group_id
            if group_id is not None:
                index = x.index[masks["train"]]
                info = {"rows": len(index), "stocks": index.get_level_values("instrument").nunique(), "days": index.get_level_values("datetime").nunique(), "calibration_days": x.index[masks["calibrate"]].get_level_values("datetime").nunique()}
                expert_info[str(group_id)] = info
                if info["rows"] < 50000 or info["stocks"] < 100 or info["days"] < 500 or info["calibration_days"] < 40:
                    info["status"] = "sample_gate_fallback_to_B"
                    continue
            checkpoint_path = OUT / f"{fold_name}_{seed}_{destination}.checkpoint.pkl"
            if checkpoint_path.exists():
                with checkpoint_path.open("rb") as handle:
                    checkpoint = pickle.load(handle)
                scores[destination] = checkpoint["scores"]
                calibration_predictions[destination] = checkpoint["calibration_prediction"]
                calibrations[destination] = checkpoint["calibration"]
                if group_id is not None:
                    expert_info[str(group_id)]["status"] = "trained_checkpoint"
                continue
            frames = {segment: pd.concat({"feature": x.loc[masks[segment]], "label": label.loc[masks[segment]].to_frame()}, axis=1) for segment in ("train", "valid")}
            started = time.perf_counter()
            (OUT / "status.json").write_text(json.dumps({"state": "running", "phase": "training", "fold": fold_name, "seed": seed, "arm": destination, "completed_model_fits": len(list(OUT.glob("*.checkpoint.pkl"))), "research_only": True}, indent=2), encoding="utf-8")
            print(json.dumps({"stage": "fit", "fold": fold_name, "seed": seed, "arm": arm, "group": group_id, "train_rows": len(frames["train"])}), flush=True)
            model = RankLGBModel(**params)
            try:
                model.fit(_PreparedFactorDataset(frames, list(x.columns)), verbose_eval=0, log_metrics=False)
            except Exception as exc:
                if group_id is None:
                    raise
                expert_info[str(group_id)].update({"status": "training_failure_fallback_to_B", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"})
                del frames, model
                gc.collect()
                continue
            predicted = {segment: pd.Series(model.model.predict(x.loc[masks[segment]].to_numpy(), num_threads=4), index=x.index[masks[segment]]) for segment in ("calibrate", "test")}
            (OUT / f"{fold_name}_{seed}_{destination}.txt").write_text(model.model.model_to_string(), encoding="utf-8")
            score, mapping = calibrate(predicted["calibrate"], label, group, predicted["test"], group)
            scores[destination] = pd.DataFrame({"raw_score": predicted["test"], "score": score, "label": label.reindex(score.index), "group_id": group.reindex(score.index), "fallback": False})
            calibration_predictions[destination] = predicted["calibrate"]
            calibrations[destination] = {"maps": mapping, "best_iteration": model.model.best_iteration, "elapsed_seconds": time.perf_counter() - started}
            checkpoint_temporary = checkpoint_path.with_suffix(".pkl.tmp")
            with checkpoint_temporary.open("wb") as handle:
                pickle.dump({"scores": scores[destination], "calibration_prediction": calibration_predictions[destination], "calibration": calibrations[destination]}, handle, protocol=5)
            checkpoint_temporary.replace(checkpoint_path)
            if group_id is not None:
                expert_info[str(group_id)]["status"] = "trained"
            del frames, model, predicted
            gc.collect()
            print(json.dumps({"stage": "fit_done", "fold": fold_name, "seed": seed, "arm": destination, **calibrations[destination]}, default=str), flush=True)
        scores["C"] = scores["B"].copy()
        scores["C"]["fallback"] = True
        expert_keys = [f"C{group_id}" for group_id in range(6) if f"C{group_id}" in scores]
        expert_calibration = pd.concat([calibration_predictions[key] for key in expert_keys]) if expert_keys else pd.Series(dtype=float)
        expert_test = pd.concat([scores[key]["raw_score"] for key in expert_keys]) if expert_keys else pd.Series(dtype=float)
        if len(expert_test):
            expert_score, expert_maps = calibrate(expert_calibration, label, group, expert_test, group)
            calibrations["C_pooled"] = {"maps": expert_maps}
        for group_id in range(6):
            if f"C{group_id}" in scores:
                part = scores[f"C{group_id}"]
                part["score"] = expert_score.reindex(part.index)
                scores["C"].loc[part.index, :] = part
        if identical_inputs:
            scores["A0"] = scores["A1"].copy()
            calibrations["A0"] = {"reused": "A1", "reason": "all raw feature values and labels identical under fixed seed; no separate fit needed"}
        for arm, path in paths.items():
            scores[arm].assign(arm=arm, fold=fold_name, seed=seed).to_parquet(path, compression="zstd")
        (OUT / f"{fold_name}_{seed}_training.json").write_text(json.dumps({"models": calibrations, "experts": expert_info}, indent=2, default=str), encoding="utf-8")
        del scores
        gc.collect()
    del processed, group, label
    gc.collect()
(OUT / "training_complete.json").write_text(json.dumps({"folds": args.folds, "seeds": args.seeds, "research_only": True}), encoding="utf-8")
