import argparse
import copy
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from core.adapters.qlib.factor_set_evaluator import _PreparedFactorDataset
from core.adapters.qlib.models import DEFAULT_PARAMS, PARAM_PROFILES
from core.adapters.qlib.rank_models import RankLGBModel
from scripts.horizon_multiscale_research import OUTPUT, SOURCE, PRIOR, SEEDS, SEQUENCE_ARMS, calendar_index
from scripts.horizon_sequence_features import build_daily_long_features, build_monthly_sequence_table
from scripts.horizon_sequence_models import HorizonFusionNet
from scripts.run_multiscale_group_research import sha256
from scripts.train_horizon_classification import load_fold, calibrate, verify_restore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def signature(manifest, horizon, fold, seed, arm):
    value = json.dumps({"manifest": sha256(OUTPUT / "manifest.json"), "labels": sha256(OUTPUT / "label_audit.json"),
                        "horizon": horizon, "fold": fold, "seed": seed, "arm": arm,
                        "parameters": manifest["lightgbm_parameters"], "epochs": 80, "batch": 4096}, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def build_daily_positions(index, calendar, endpoints, length=20):
    sessions = pd.DatetimeIndex(calendar)
    date_positions = sessions.get_indexer(index.get_level_values("datetime"))
    if (date_positions < 0).any():
        raise RuntimeError("日线序列包含交易日历外日期")
    positions_by_stock = {}
    stock_codes = index.get_level_values("instrument")
    row_series = pd.Series(np.arange(len(index), dtype=np.int32), index=index)
    for stock, rows in row_series.groupby(level="instrument", sort=False):
        row_positions = rows.to_numpy()
        lookup = np.full(len(sessions), -1, dtype=np.int32)
        lookup[date_positions[row_positions]] = row_positions
        positions_by_stock[stock] = lookup
    endpoint_dates = endpoints.get_level_values("datetime")
    endpoint_series = pd.Series(np.arange(len(endpoints), dtype=np.int32), index=endpoints)
    result = np.full((len(endpoints), length), -1, dtype=np.int32)
    for stock, endpoint_rows in endpoint_series.groupby(level="instrument", sort=False):
        rows = endpoint_rows.to_numpy()
        ends = sessions.get_indexer(endpoint_dates[rows])
        offsets = ends[:, None] - np.arange(length - 1, -1, -1, dtype=np.int32)[None, :]
        valid = (offsets >= 0) & (offsets < len(sessions))
        values = positions_by_stock[stock][np.clip(offsets, 0, len(sessions) - 1)]
        result[rows] = np.where(valid, values, -1)
    return result


def build_monthly_lookup(monthly_table, calendar):
    periods = pd.period_range(pd.DatetimeIndex(calendar).min().to_period("M"),
                              pd.DatetimeIndex(calendar).max().to_period("M"), freq="M")
    tables = {}
    for stock in monthly_table.index.get_level_values("instrument").unique():
        data = monthly_table.xs(stock, level="instrument").reindex(periods)
        values = data.to_numpy(dtype=np.float32)
        values[:, :6] = np.where(np.isfinite(values[:, :6]), values[:, :6], 0.0)
        values[:, 6:] = np.where(np.isfinite(values[:, 6:]), values[:, 6:], 1.0)
        tables[stock] = values
    session_dates = pd.DatetimeIndex(calendar)
    last_day = pd.Series(session_dates, index=session_dates.to_period("M")).groupby(level=0).last()
    month_end_dates = pd.DatetimeIndex(last_day.to_numpy())
    month_end_pos = np.searchsorted(session_dates.to_numpy(), month_end_dates.to_numpy())
    return periods, tables, month_end_dates, month_end_pos


def monthly_batch(index, rows, lookup, calendar, repeat_latest=False, scale=None):
    periods, tables, month_ends, _ = lookup
    dates = index.get_level_values("datetime")[rows]
    stocks = index.get_level_values("instrument")[rows]
    values = np.zeros((len(rows), 12, 12), dtype=np.float32)
    values[:, :, 6:] = 1.0
    end_positions = np.searchsorted(month_ends.to_numpy(), dates.to_numpy(), side="right") - 1
    offsets = end_positions[:, None] - np.arange(11, -1, -1, dtype=np.int64)[None, :]
    valid = offsets >= 0
    safe = np.maximum(offsets, 0)
    stock_series = pd.Series(np.arange(len(stocks), dtype=np.int32), index=pd.Index(stocks))
    for stock, row_group in stock_series.groupby(level=0, sort=False):
        selected = row_group.to_numpy()
        if stock not in tables:
            continue
        table = tables[stock]
        values[selected] = table[safe[selected]]
        values[selected] *= valid[selected, :, None]
        values[selected, :, 6:] = np.where(valid[selected, :, None], values[selected, :, 6:], 1.0)
    if repeat_latest:
        latest = values[:, -1, :].copy()
        values[:, :, :6] = latest[:, None, :6]
        for column in range(6):
            missing_latest = latest[:, 6 + column] >= 1
            values[missing_latest, :, column] = 0
            values[missing_latest, :, 6 + column] = 1
    if scale is not None:
        mean, std = scale
        for column in range(6):
            valid = values[:, :, 6 + column] == 0
            values[:, :, column] = np.where(valid, (values[:, :, column] - mean[column]) / std[column], 0)
    return values


def monthly_scaler(index, rows, lookup, calendar):
    total = np.zeros(6, dtype=np.float64)
    squares = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)
    for start in range(0, len(rows), 4096):
        values = monthly_batch(index, rows[start:start + 4096], lookup, calendar)
        for column in range(6):
            valid = values[:, :, 6 + column] == 0
            observed = values[:, :, column][valid].astype(np.float64)
            total[column] += observed.sum()
            squares[column] += np.square(observed).sum()
            counts[column] += len(observed)
    means = np.divide(total, np.maximum(counts, 1))
    variance = np.divide(squares, np.maximum(counts, 1)) - means ** 2
    scales = np.maximum(np.sqrt(np.maximum(variance, 0)), 1e-6)
    return means.astype(np.float32), scales.astype(np.float32)


def weighted_scaler(values, train_positions, continuous_count):
    counts = np.bincount(train_positions[train_positions >= 0].ravel(), minlength=len(values)).astype(np.float64)
    means = np.zeros(values.shape[1], dtype=np.float64)
    scales = np.ones(values.shape[1], dtype=np.float64)
    total = counts.sum()
    for column in range(continuous_count):
        source = values[:, column].astype(np.float64)
        means[column] = np.dot(source, counts) / max(total, 1)
        variance = np.dot((source - means[column]) ** 2, counts) / max(total, 1)
        scales[column] = max(np.sqrt(variance), 1e-6)
    return means.astype(np.float32), scales.astype(np.float32)


def transform_batch(values, positions, mean, scale, continuous_count):
    batch = np.zeros((*positions.shape, values.shape[1] + 1), dtype=np.float32)
    valid = positions >= 0
    safe = np.maximum(positions, 0)
    batch[:, :, :-1] = values[safe]
    for column in range(continuous_count):
        batch[:, :, column] = (batch[:, :, column] - mean[column]) / scale[column]
    batch[:, :, :-1] *= valid[:, :, None]
    batch[:, :, -1] = valid.astype(np.float32)
    return batch


def fit_sequence(values, index, labels, masks, calendar, monthly_lookup, arm, seed, model_path, resume, task_signature, continuous_count, month_scale):
    dates = index.get_level_values("datetime")
    session_pos = pd.DatetimeIndex(calendar).get_indexer(dates)
    endpoint = np.arange(len(index), dtype=np.int32)
    valid_sequence = np.ones(len(index), dtype=bool)
    for offset in range(20):
        positions = session_pos - (19 - offset)
        valid_sequence &= positions >= 0
    chosen = {}
    for name in ("train", "valid", "calibrate", "test"):
        mask = masks[name] & valid_sequence
        if name == "train":
            mask &= session_pos % 5 == 0
        chosen[name] = endpoint[mask]
    if len(chosen["train"]) < 1000 or len(chosen["valid"]) < 100:
        raise RuntimeError(f"{arm} 有效训练/早停端点不足")
    train_index = index.take(chosen["train"])
    train_positions = build_daily_positions(index, calendar, train_index)
    mean, scale = weighted_scaler(values, train_positions, continuous_count)
    daily_positions = {name: build_daily_positions(index, calendar, index.take(rows)) for name, rows in chosen.items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HorizonFusionNet(values.shape[1] + 1, arm).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best_loss, best_state, wait, start_epoch = float("inf"), None, 0, 0
    if resume and model_path.exists():
        stored = torch.load(model_path, map_location="cpu", weights_only=False)
        if stored.get("signature") != task_signature:
            raise RuntimeError(f"序列断点签名不匹配: {model_path.name}")
        model.load_state_dict(stored["model"])
        optimizer.load_state_dict(stored["optimizer"])
        best_loss, best_state, wait, start_epoch = stored["best_loss"], stored["best_state"], stored["wait"], stored["epoch"] + 1
        np.random.set_state(stored["numpy_state"])
        torch.set_rng_state(stored["torch_state"])
        if torch.cuda.is_available() and stored.get("cuda_state") is not None:
            torch.cuda.set_rng_state_all(stored["cuda_state"])

    def batch_for(name, ids):
        pos = daily_positions[name][ids]
        daily = transform_batch(values, pos, mean, scale, continuous_count)
        y = labels[chosen[name][ids]]
        monthly = monthly_batch(index, chosen[name][ids], monthly_lookup, calendar,
                                repeat_latest=arm == "NR", scale=month_scale) if arm in ("NF", "NT", "NR") else None
        return daily, monthly, y

    started = time.perf_counter()
    for epoch in range(start_epoch, 80):
        model.train()
        order = np.random.permutation(len(chosen["train"]))
        for start in range(0, len(order), 4096):
            ids = order[start:start + 4096]
            daily, month, truth = batch_for("train", ids)
            x = torch.from_numpy(daily).to(DEVICE)
            m = torch.from_numpy(month).to(DEVICE) if month is not None else None
            y = torch.from_numpy(truth).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, m), y)
            loss.backward()
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(chosen["valid"]), 4096):
                ids = np.arange(start, min(start + 4096, len(chosen["valid"])))
                daily, month, truth = batch_for("valid", ids)
                x = torch.from_numpy(daily).to(DEVICE)
                m = torch.from_numpy(month).to(DEVICE) if month is not None else None
                y = torch.from_numpy(truth).to(DEVICE)
                total += float(loss_fn(model(x, m), y).item()) * len(ids)
                count += len(ids)
        validation_loss = total / count
        if validation_loss < best_loss - 1e-6:
            best_loss, best_state, wait = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        payload = {"signature": task_signature, "model": state, "optimizer": optimizer.state_dict(),
                   "best_state": best_state, "best_loss": best_loss, "wait": wait, "epoch": epoch,
                   "numpy_state": np.random.get_state(), "torch_state": torch.get_rng_state(),
                   "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temporary = model_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(model_path)
        if wait >= 10:
            break
    if best_state is None:
        raise RuntimeError(f"{arm} 早停未生成有限模型")
    model.load_state_dict(best_state)
    model.eval()
    predictions = {}
    for name in ("calibrate", "test"):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(chosen[name]), 4096):
                ids = np.arange(start, min(start + 4096, len(chosen[name])))
                daily, month, _ = batch_for(name, ids)
                x = torch.from_numpy(daily).to(DEVICE)
                m = torch.from_numpy(month).to(DEVICE) if month is not None else None
                chunks.append(model(x, m).cpu().numpy())
        predictions[name] = pd.Series(np.concatenate(chunks) if chunks else np.empty(0, np.float32),
                                      index=index.take(chosen[name]))
    sample_count = min(4096, len(chosen["test"]))
    restored = HorizonFusionNet(values.shape[1] + 1, arm).to(DEVICE)
    restored.load_state_dict(best_state)
    restored.eval()
    if sample_count:
        sample_rows = np.arange(sample_count)
        sample_daily, sample_month, _ = batch_for("test", sample_rows)
        with torch.no_grad():
            expected = model(torch.from_numpy(sample_daily).to(DEVICE),
                              torch.from_numpy(sample_month).to(DEVICE) if sample_month is not None else None)
            actual = restored(torch.from_numpy(sample_daily).to(DEVICE),
                              torch.from_numpy(sample_month).to(DEVICE) if sample_month is not None else None)
        restore_error = float(torch.max(torch.abs(expected - actual)).item())
        if restore_error > 1e-6:
            raise RuntimeError(f"{arm} 神经网络保存恢复误差超过1e-6: {restore_error}")
    else:
        restore_error = 0.0
    final_path = model_path.with_suffix(".final.pt")
    final_temp = final_path.with_suffix(".tmp")
    torch.save({"signature": task_signature, "arm": arm, "daily_features": values.shape[1] + 1,
                "model": {key: value.detach().cpu() for key, value in best_state.items()},
                "mean": mean, "scale": scale, "restore_max_abs_error": restore_error}, final_temp)
    final_temp.replace(final_path)
    detail = {"arm": arm, "epochs": epoch + 1, "best_validation_loss": best_loss,
              "elapsed_seconds": time.perf_counter() - started, "train_endpoints": len(chosen["train"]),
              "valid_endpoints": len(chosen["valid"]), "calibration_endpoints": len(chosen["calibrate"]),
              "test_endpoints": len(chosen["test"]), "device": str(DEVICE),
              "restore_max_abs_error": restore_error}
    return predictions, {"model": best_state, "mean": mean, "scale": scale}, detail


def train(args):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "label_audit.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"序列轨道输入签名变化: {relative}")
    for relative, expected in manifest["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"序列轨道代码签名变化: {relative}")
    calendar = calendar_index()
    for horizon in args.horizons:
        for fold in args.folds:
            inputs, groups, calibration_group, label_frame, masks, load_audit = load_fold(fold, int(horizon[1:]))
            index = inputs["D"].index
            features = manifest["factor_keys"]
            daily_base = inputs["D"][features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            start = pd.Timestamp(index.get_level_values("datetime").min())
            end = pd.Timestamp(index.get_level_values("datetime").max())
            filters = [("datetime", ">=", start - pd.DateOffset(months=18)), ("datetime", "<=", end)]
            panel = pd.concat([pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"], filters=filters)
                               for path in sorted(SOURCE.glob("new_*.parquet"))]).sort_index()
            preheat = pd.read_parquet(PRIOR / "preheat_2018.parquet")
            quotes = pd.concat([preheat, panel]).sort_index()
            quotes = quotes.loc[~quotes.index.duplicated(keep="last")]
            daily_long = build_daily_long_features(quotes, calendar).reindex(index)
            snapshot_month = inputs["M"].iloc[:, len(features):len(features) + 16]
            monthly_table = build_monthly_sequence_table(quotes, calendar)
            monthly_lookup = build_monthly_lookup(monthly_table, calendar)
            train_session = calendar.get_indexer(index.get_level_values("datetime"))
            monthly_train_rows = np.flatnonzero(masks["train"] & (train_session % 5 == 0))
            month_scale = monthly_scaler(index, monthly_train_rows, monthly_lookup, calendar)
            daily_frames = {
                "N0": daily_base,
                "NL": pd.concat([daily_base, daily_long], axis=1),
                "NS": pd.concat([daily_base, snapshot_month], axis=1),
            }
            raw_label = label_frame["label"].to_numpy(dtype=np.float32)
            label_rank = label_frame["label_rank"].to_numpy(dtype=np.float32)
            row_masks = {name: mask for name, mask in masks.items()}
            raw_predictions = {}
            for seed in args.seeds:
                restore_error = None
                lgb_path = OUTPUT / f"{fold}_{horizon}_{seed}_sequence_lgb.txt"
                lgb_values = {}
                if args.resume and lgb_path.exists():
                    model_data = pickle.loads(lgb_path.with_suffix(".pkl").read_bytes())
                    if model_data.get("signature") != signature(manifest, horizon, fold, seed, "LGB40"):
                        raise RuntimeError("序列共用LGB断点签名不匹配")
                    lgb_values = model_data["scores"]
                    import lightgbm as lgb
                    booster = lgb.Booster(model_str=lgb_path.read_text(encoding="utf-8"))
                    sample_index = lgb_values["test"].index[:10000]
                    restored_scores = booster.predict(daily_base.reindex(sample_index).to_numpy(), num_threads=4)
                    restore_error = float(np.max(np.abs(restored_scores - lgb_values["test"].reindex(sample_index).to_numpy())))
                    if restore_error > 1e-6:
                        raise RuntimeError(f"序列共用LGB恢复误差超过1e-6: {restore_error}")
                else:
                    params = {**DEFAULT_PARAMS["lightgbm"], **PARAM_PROFILES["conservative"]["lightgbm"],
                              "num_threads": 4, "seed": seed, "bagging_seed": seed,
                              "feature_fraction_seed": seed, "data_random_seed": seed}
                    frames = {name: pd.concat({"feature": daily_base.loc[mask],
                                 "label": label_frame.loc[mask, "label_rank"].to_frame("LABEL0")}, axis=1)
                              for name, mask in row_masks.items() if name in ("train", "valid")}
                    model = RankLGBModel(**params)
                    model.fit(_PreparedFactorDataset(frames, features), verbose_eval=0, log_metrics=False)
                    for name in ("calibrate", "test"):
                        mask = row_masks[name]
                        lgb_values[name] = pd.Series(model.model.predict(daily_base.loc[mask].to_numpy(), num_threads=4), index=index[mask])
                    lgb_path.write_text(model.model.model_to_string(), encoding="utf-8")
                    restore_error = verify_restore(model, lgb_path, daily_base.loc[row_masks["test"]].to_numpy())
                    lgb_path.with_suffix(".pkl").write_bytes(pickle.dumps({"signature": signature(manifest, horizon, fold, seed, "LGB40"), "scores": lgb_values}, protocol=5))
                    del model, frames
                raw_predictions[seed] = {}
                for arm in SEQUENCE_ARMS:
                    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    state_path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.pt"
                    sig = signature(manifest, horizon, fold, seed, arm)
                    if args.resume and path.exists():
                        stored = pd.read_parquet(path)
                        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                        if meta.get("signature") != sig:
                            raise RuntimeError(f"序列预测断点签名不匹配: {path.name}")
                        raw_predictions[seed][arm] = stored
                        continue
                    if arm in ("NF", "NT", "NR"):
                        arm_values = daily_base.to_numpy(dtype=np.float32)
                        continuous_count = arm_values.shape[1]
                    else:
                        frame = daily_frames[arm]
                        arm_values = frame.to_numpy(dtype=np.float32)
                        continuous_count = len(features) + (12 if arm == "NL" else 8 if arm == "NS" else 0)
                    labels = label_rank
                    if arm in ("NF", "NT", "NR"):
                        raw_nn, bundle, detail = fit_sequence(arm_values, index, labels, row_masks, calendar,
                            monthly_lookup, arm, seed, state_path, args.resume, sig, continuous_count, month_scale)
                    else:
                        raw_nn, bundle, detail = fit_sequence(arm_values, index, labels, row_masks, calendar,
                            monthly_lookup, arm, seed, state_path, args.resume, sig, continuous_count, month_scale)
                    fused = {}
                    for name in ("calibrate", "test"):
                        target = lgb_values[name].index
                        nn_score = raw_nn[name]
                        fused[name] = 0.5 * nn_score.groupby(level="datetime").rank(pct=True).reindex(target) + \
                                      0.5 * lgb_values[name].groupby(level="datetime").rank(pct=True).reindex(target)
                    score, maps = calibrate(fused["calibrate"], label_frame["label"], calibration_group,
                                            fused["test"], calibration_group)
                    out = pd.DataFrame({"score": score, "label": label_frame["label"].reindex(score.index),
                                        "label_rank": label_frame["label_rank"].reindex(score.index)})
                    if not out.index.equals(index[row_masks["test"]]) or not np.isfinite(out["score"]).all():
                        raise RuntimeError(f"{fold}/{horizon}/{seed}/{arm}: 融合预测未完整覆盖或非有限")
                    temp = path.with_suffix(".tmp")
                    out.to_parquet(temp, compression="zstd")
                    temp.replace(path)
                    path.with_suffix(".json").write_text(json.dumps({"signature": sig, **detail,
                        "calibration_maps": {str(k): list(v) for k, v in maps.items()},
                        "lgb_restore_max_abs_error": restore_error if 'restore_error' in locals() else None,
                        "monthly_scaler_mean": month_scale[0].tolist(), "monthly_scaler_scale": month_scale[1].tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
                    raw_predictions[seed][arm] = out
                    print(json.dumps({"fold": fold, "horizon": horizon, "seed": seed, **detail}, ensure_ascii=False), flush=True)
                gc.collect()
            del inputs, groups, calibration_group, label_frame, panel, quotes, daily_base, daily_long, snapshot_month, monthly_table, monthly_lookup
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", choices=("H10", "H20"), default=("H10", "H20"))
    parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())
