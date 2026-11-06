import argparse
import copy
import gc
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
from sklearn.preprocessing import StandardScaler

from core.adapters.qlib.lstm_fusion import _LSTMReg, _DEVICE
from scripts.run_monthly_group_research import OUTPUT, SEEDS
from scripts.run_multiscale_group_research import sha256
from scripts.train_monthly_group_research import calibrate, load_fold


def predict(model, values):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            batch = torch.from_numpy(values[start:start + 4096]).float().to(_DEVICE)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def fit_component(x, label, masks, seed, group=None):
    x = x.sort_index(level=[1, 0]).fillna(0)
    index = x.index
    values = x.to_numpy(dtype=np.float32)
    targets = label.reindex(index).to_numpy(dtype=np.float32)
    selected = {name: pd.Series(masks[name], index=label.index).reindex(index).to_numpy(dtype=bool)
                for name in ("train", "valid", "calibrate", "test")}
    active = group.reindex(index).to_numpy(dtype=bool) if group is not None else np.ones(len(index), dtype=bool)
    endpoints = []
    instruments = index.get_level_values("instrument").to_numpy()
    boundaries = np.r_[0, np.flatnonzero(instruments[1:] != instruments[:-1]) + 1, len(index)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        endpoints.append(np.arange(left + 19, right, dtype=np.int32))
    endpoints = np.concatenate(endpoints)
    positions = {}
    for name in selected:
        keep = selected[name][endpoints] & active[endpoints]
        if name in ("train", "valid"):
            keep &= np.isfinite(targets[endpoints])
        if name == "train":
            keep &= ((endpoints - boundaries[np.searchsorted(boundaries, endpoints, side="right") - 1] - 19) % 5 == 0)
        positions[name] = endpoints[keep]
    if len(positions["train"]) < 1000 or len(positions["valid"]) < 100:
        raise RuntimeError("LSTM 有效训练或早停序列不足")
    counts = np.zeros(len(values) + 1, dtype=np.int32)
    np.add.at(counts, positions["train"] - 19, 1)
    np.add.at(counts, positions["train"] + 1, -1)
    weights = np.cumsum(counts[:-1])
    scaler = StandardScaler()
    scaler.fit(values, sample_weight=weights)
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    offsets = np.arange(19, -1, -1, dtype=np.int32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _LSTMReg(x.shape[1]).to(_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0
    started = time.perf_counter()
    for epoch in range(80):
        model.train()
        shuffled = np.random.permutation(positions["train"])
        for start in range(0, len(shuffled), 4096):
            batch_positions = shuffled[start:start + 4096]
            batch = (values[batch_positions[:, None] - offsets] - mean) / scale
            features = torch.from_numpy(batch).to(_DEVICE)
            truth = torch.from_numpy(targets[batch_positions]).to(_DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(features), truth)
            loss.backward()
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(positions["valid"]), 4096):
                batch_positions = positions["valid"][start:start + 4096]
                batch = (values[batch_positions[:, None] - offsets] - mean) / scale
                features = torch.from_numpy(batch).to(_DEVICE)
                truth = torch.from_numpy(targets[batch_positions]).to(_DEVICE)
                total += float(loss_fn(model(features), truth).item()) * len(batch_positions)
                count += len(batch_positions)
        validation_loss = total / count
        if validation_loss < best_loss - 1e-6:
            best_loss, best_state, wait = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= 10:
                break
    if best_state is None:
        raise RuntimeError("LSTM 早停损失非有限值")
    model.load_state_dict(best_state)
    predictions = {}
    for name in ("calibrate", "test"):
        parts = []
        for start in range(0, len(positions[name]), 4096):
            batch_positions = positions[name][start:start + 4096]
            batch = (values[batch_positions[:, None] - offsets] - mean) / scale
            parts.append(predict(model, batch))
        predictions[name] = pd.Series(np.concatenate(parts) if parts else np.empty(0), index=index[positions[name]])
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    detail = {"train_sequences": len(positions["train"]), "valid_sequences": len(positions["valid"]),
              "elapsed_seconds": time.perf_counter() - started, "feature_count": x.shape[1],
              "missing_sequence_calibration": int(selected["calibrate"].sum() - len(predictions["calibrate"])),
              "missing_sequence_test": int(selected["test"].sum() - len(predictions["test"])),
              "epochs": epoch + 1}
    del model, values, targets, endpoints
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, {"state": state, "mean": scaler.mean_, "scale": scaler.scale_, "features": list(x.columns)}, detail


def rank(series, target):
    return series.groupby(level="datetime").rank(pct=True).reindex(target).fillna(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assessment = json.loads((OUTPUT / "assessment.json").read_text(encoding="utf-8"))
    if assessment["candidate_comparisons"] != ["DM-D", "SL4_M-UM"]:
        raise RuntimeError("候选列表与冻结后的筛选结果不同")
    if set(args.folds) - set(manifest["folds"]) or set(args.seeds) - set(SEEDS):
        raise ValueError("折或种子超出冻结矩阵")
    for name, expected in manifest["source_files_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"冻结输入变化: {name}")
    for name, expected in manifest["code_files_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"冻结代码变化: {name}")
    fusion_root = OUTPUT / "fusion_confirmation"
    fusion_root.mkdir(exist_ok=True)
    for fold in args.folds:
        window = manifest["folds"][fold]
        arms, groups, calibration_group, raw_label, rank_label, eligible, audit = load_fold(manifest, fold)
        dates = pd.DatetimeIndex(raw_label.index.get_level_values("datetime"))
        masks = {name: (dates >= pd.Timestamp(window[name][0])) & (dates <= pd.Timestamp(window[name][1])) & eligible
                 for name in ("train", "valid", "calibrate", "test")}
        for seed in args.seeds:
            path = fusion_root / f"{fold}_{seed}_audit.json"
            if path.exists() and all((fusion_root / f"{fold}_{seed}_{arm}.parquet").exists()
                                     for arm in ("D", "DM", "UM", "SL4_M")):
                continue
            component = {}
            details = {}
            for arm in ("D", "DM", "UM"):
                checkpoint = fusion_root / f"{fold}_{seed}_{arm}.pkl"
                if checkpoint.exists():
                    with checkpoint.open("rb") as handle:
                        stored = pickle.load(handle)
                else:
                    scores, bundle, detail = fit_component(arms[arm], rank_label, masks, seed)
                    stored = {"scores": scores, "bundle": bundle, "detail": detail}
                    temporary = checkpoint.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        pickle.dump(stored, handle, protocol=5)
                    temporary.replace(checkpoint)
                component[arm] = stored["scores"]
                details[arm] = stored["detail"]
                print(json.dumps({"fold": fold, "seed": seed, "arm": arm, **stored["detail"]}), flush=True)
            baseline = "UM"
            expert = {name: component[baseline][name].copy() for name in ("calibrate", "test")}
            expert_detail = {}
            lgb_detail = json.loads((OUTPUT / f"{fold}_{seed}_training.json").read_text(encoding="utf-8"))["SL4_M"]
            for group_id in range(4):
                if lgb_detail["experts"].get(str(group_id), {}).get("status") != "trained":
                    expert_detail[str(group_id)] = {"status": "lgb_fallback"}
                    continue
                group_mask = groups["SL4"] == group_id
                checkpoint = fusion_root / f"{fold}_{seed}_SL4_M_{group_id}.pkl"
                if checkpoint.exists():
                    with checkpoint.open("rb") as handle:
                        stored = pickle.load(handle)
                else:
                    try:
                        scores, bundle, detail = fit_component(arms[baseline], rank_label, masks, seed, group_mask)
                        stored = {"scores": scores, "bundle": bundle, "detail": detail}
                    except Exception as error:
                        expert_detail[str(group_id)] = {"status": "lstm_fit_fallback", "reason": f"{type(error).__name__}: {error}"}
                        continue
                    temporary = checkpoint.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        pickle.dump(stored, handle, protocol=5)
                    temporary.replace(checkpoint)
                for name in ("calibrate", "test"):
                    expert[name].loc[stored["scores"][name].index] = stored["scores"][name]
                expert_detail[str(group_id)] = {"status": "trained", **stored["detail"]}
                print(json.dumps({"fold": fold, "seed": seed, "arm": "SL4_M", "group": group_id, **stored["detail"]}), flush=True)
            component["SL4_M"] = expert
            details["SL4_M"] = expert_detail
            calibrated = {}
            for arm in ("D", "DM", "UM", "SL4_M"):
                lgb = {}
                if arm != "SL4_M":
                    with (OUTPUT / f"{fold}_{seed}_{arm}.checkpoint.pkl").open("rb") as handle:
                        lgb = pickle.load(handle)["scores"]
                else:
                    with (OUTPUT / f"{fold}_{seed}_UM.checkpoint.pkl").open("rb") as handle:
                        lgb = {name: value.copy() for name, value in pickle.load(handle)["scores"].items()}
                    for group_id in range(4):
                        if expert_detail[str(group_id)]["status"] != "trained":
                            continue
                        with (OUTPUT / f"{fold}_{seed}_SL4_M_{group_id}.checkpoint.pkl").open("rb") as handle:
                            group_scores = pickle.load(handle)["scores"]
                        for name in ("calibrate", "test"):
                            lgb[name].loc[group_scores[name].index] = group_scores[name]
                fused = {}
                for name in ("calibrate", "test"):
                    target = raw_label.loc[masks[name]].index
                    fused[name] = 0.5 * rank(lgb[name], target) + 0.5 * rank(component[arm][name], target)
                score, maps = calibrate(fused["calibrate"], raw_label, calibration_group,
                                        fused["test"], calibration_group)
                if arm == "SL4_M":
                    fallback = ~groups.loc[masks["test"], "SL4"].isin([int(key) for key, value in expert_detail.items()
                                                                         if value["status"] == "trained"])
                    score.loc[fallback] = calibrated["UM"].loc[fallback]
                calibrated[arm] = score
                result = pd.DataFrame({"score": score, "label": raw_label.loc[masks["test"]]})
                if not result.index.equals(raw_label.loc[masks["test"]].index) or not np.isfinite(score).all():
                    raise RuntimeError(f"融合输出缺失: {fold} {seed} {arm}")
                temporary = fusion_root / f"{fold}_{seed}_{arm}.tmp"
                result.to_parquet(temporary, compression="zstd")
                temporary.replace(fusion_root / f"{fold}_{seed}_{arm}.parquet")
                details[f"{arm}_calibration"] = {str(key): list(value) for key, value in maps.items()}
            path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        del arms, groups, calibration_group, raw_label, rank_label, eligible
        gc.collect()


if __name__ == "__main__":
    main()
