"""LGB + LSTM 分数层融合 (0.5/0.5 秩融合)

在合并因子集上同时训练:
  - LGB (qlib lightgbm_rank, 截面非线性)
  - LSTM (PyTorch, 20 日因子序列, 时序)
对两者预测逐日截面秩做 0.5/0.5 融合, 得到 fused 预测 + 各自/融合 IC。

设计要点 (与 scripts/experiment_lgb_lstm_fusion.py 已验证的口径一致):
  - LGB 走 qlib DataHandlerLP (截面)。
  - LSTM 需要 20 日历史: 用连续特征面板构建序列, 再按 purge 边界切分 train/valid/test,
    避免标签前视。
"""
import copy
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager, _require_tradable_label_mode
from core.adapters.qlib.factor_set_evaluator import _PreparedFactorDataset, _select_feature_keys
from qlib.data.dataset.weight import Reweighter

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SEQ_LEN = 20
_BUNDLE_SCHEMA_VERSION = 1
_PREDICTION_CACHE_VERSION = 2
_PREDICTION_CACHE_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def _rank_by_date(series):
    return series.groupby(level="datetime").rank(pct=True)


def _as_series(value):
    if isinstance(value, pd.Series):
        return value.astype(float)
    if isinstance(value, pd.DataFrame):
        if "score" in value.columns:
            return value["score"].astype(float)
        return value.iloc[:, 0].astype(float)
    return pd.Series(value, dtype=float)


def _slice_dates(frame, start, end):
    dates = pd.to_datetime(frame.index.get_level_values("datetime"))
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame.loc[mask]


def _purged_end_date(frame, boundary, purge_gap):
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("datetime")).unique()).sort_values()
    dates = dates[dates < pd.Timestamp(boundary)]
    if len(dates) <= int(purge_gap):
        raise RuntimeError(f"purge 后训练日期不足: dates={len(dates)}, gap={purge_gap}")
    return dates[-int(purge_gap) - 1]


def _date_weight_array(index, weights):
    if not weights:
        return None
    if isinstance(index, pd.MultiIndex):
        values = index.get_level_values("datetime")
    else:
        values = [row[0] for row in index]
    dates = pd.to_datetime(values).strftime("%Y-%m-%d")
    return np.asarray([float(weights.get(d, 1.0)) for d in dates], dtype=np.float32)


class _DateReweighter(Reweighter):
    def __init__(self, weights):
        self.weights = dict(weights or {})

    def reweight(self, data):
        return _date_weight_array(data.index, self.weights)


def _validate_lstm_bundle(bundle, factor_keys, seq_len, fusion_weight):
    if not isinstance(bundle, dict):
        raise RuntimeError("LSTM bundle 格式无效")
    if list(bundle.get("factor_keys") or []) != list(factor_keys):
        raise RuntimeError("LSTM bundle 因子顺序与模型不一致")
    if int(bundle.get("seq_len", -1)) != int(seq_len):
        raise RuntimeError("LSTM bundle 序列长度与模型不一致")
    if abs(float(bundle.get("fusion_weight", -1)) - float(fusion_weight)) > 1e-12:
        raise RuntimeError("LSTM bundle 融合权重与模型不一致")
    scaler = bundle.get("scaler") or {}
    mean = np.asarray(scaler.get("mean"), dtype=np.float32)
    scale = np.asarray(scaler.get("scale"), dtype=np.float32)
    if mean.shape != (len(factor_keys),) or scale.shape != (len(factor_keys),):
        raise RuntimeError("LSTM bundle scaler 维度与因子数不一致")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("LSTM bundle scaler 包含无效值")
    if not isinstance(bundle.get("state_dict"), dict):
        raise RuntimeError("LSTM bundle 缺少模型权重")
    return mean, scale, bundle["state_dict"]


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _instrument_codes(provider, instruments):
    if isinstance(instruments, (list, tuple, set)):
        return sorted({str(code).strip().lower() for code in instruments if str(code).strip()})
    path = provider / "instruments" / f"{instruments}.txt"
    if not path.exists():
        return [str(instruments)]
    return sorted({line.split("\t", 1)[0].strip().lower()
                   for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})


def _qlib_data_signature(provider, instruments):
    digest = hashlib.sha256()
    calendar_path = provider / "calendars" / "day.txt"
    if calendar_path.exists():
        digest.update(calendar_path.read_bytes())
    codes = _instrument_codes(provider, instruments)
    digest.update("\n".join(codes).encode("utf-8"))
    file_count = 0
    total_size = 0
    latest_mtime = 0
    features_dir = provider / "features"
    for code in codes:
        code_dir = features_dir / code
        if not code_dir.exists():
            continue
        for path in code_dir.iterdir():
            if not path.is_file():
                continue
            stat = path.stat()
            file_count += 1
            total_size += stat.st_size
            latest_mtime = max(latest_mtime, stat.st_mtime_ns)
    digest.update(f"{file_count}:{total_size}:{latest_mtime}".encode("ascii"))
    return digest.hexdigest()


def _prediction_cache_key(experiment_id, run_id, predict_start, predict_end,
                          instruments, factor_keys, seq_len, fusion_weight,
                          artifact_signature, data_signature, feature_signature=None):
    payload = {
        "version": _PREDICTION_CACHE_VERSION,
        "experiment_id": str(experiment_id),
        "run_id": str(run_id),
        "predict_start": str(predict_start),
        "predict_end": str(predict_end),
        "instruments": list(instruments) if isinstance(instruments, (list, tuple)) else str(instruments),
        "factor_keys": list(factor_keys),
        "seq_len": int(seq_len),
        "fusion_weight": float(fusion_weight),
        "artifact_signature": artifact_signature,
        "data_signature": data_signature,
    }
    payload['feature_contract'] = feature_signature
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_prediction_cache(path, cache_key):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        if cached.get("key") == cache_key and isinstance(cached.get("prediction"), pd.Series):
            return cached["prediction"]
    except Exception as exc:
        logger.warning("[LGB+LSTM] 预测缓存读取失败: %s", exc)
    return None


def _save_pickle_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(temp_path, "wb") as handle:
        pickle.dump(value, handle, protocol=5)
    os.replace(temp_path, path)


def _build_sequences(features, labels, seg_map, seq_len=_SEQ_LEN, train_step=5):
    features = features.sort_index(level=[1, 0])
    X_list, y_list, idx_list, seg_list = [], [], [], []
    for inst, grp in features.groupby(level="instrument"):
        g = grp.sort_index(level="datetime")
        arr = g.values.astype(np.float32)
        lab = labels.reindex(g.index).values.astype(np.float32)
        seg = seg_map.reindex(g.index).values
        n = len(arr)
        if n < seq_len:
            continue
        for i in range(seq_len - 1, n):
            if seg[i] == "train" and (i - (seq_len - 1)) % train_step != 0:
                continue
            X_list.append(arr[i - seq_len + 1:i + 1])
            y_list.append(lab[i])
            idx_list.append(g.index[i])
            seg_list.append(seg[i])
    return (np.stack(X_list), np.asarray(y_list, dtype=np.float32),
            idx_list, np.asarray(seg_list))


def _build_inference_sequences(features, start, seq_len=_SEQ_LEN):
    features = features.sort_index(level=[1, 0])
    X_list, idx_list = [], []
    start_ts = pd.Timestamp(start)
    for _, grp in features.groupby(level="instrument"):
        g = grp.sort_index(level="datetime")
        arr = g.values.astype(np.float32)
        if len(arr) < seq_len:
            continue
        dates = pd.to_datetime(g.index.get_level_values("datetime"))
        for i in range(seq_len - 1, len(arr)):
            if dates[i] < start_ts:
                continue
            X_list.append(arr[i - seq_len + 1:i + 1])
            idx_list.append(g.index[i])
    if not X_list:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), []
    return np.stack(X_list), idx_list


def _rebuild_sequence_scaler(features, train_end, seq_len=_SEQ_LEN, train_step=5):
    total_weight = 0.0
    total_sum = np.zeros(features.shape[1], dtype=np.float64)
    total_sq_sum = np.zeros(features.shape[1], dtype=np.float64)
    end_ts = pd.Timestamp(train_end)
    for _, grp in features.groupby(level="instrument"):
        g = grp.sort_index(level="datetime")
        values = g.values.astype(np.float64)
        dates = pd.to_datetime(g.index.get_level_values("datetime"))
        weights = np.zeros(len(values), dtype=np.float64)
        for i in range(seq_len - 1, len(values), train_step):
            if dates[i] > end_ts:
                break
            weights[i - seq_len + 1:i + 1] += 1.0
        active = weights > 0
        if not active.any():
            continue
        values = values[active]
        sample_weights = weights[active, None]
        total_weight += float(sample_weights.sum())
        total_sum += np.sum(values * sample_weights, axis=0)
        total_sq_sum += np.sum(values * values * sample_weights, axis=0)
    if total_weight <= 0:
        raise RuntimeError("无法从训练期特征重建 LSTM scaler")
    mean = total_sum / total_weight
    variance = np.maximum(total_sq_sum / total_weight - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class _LSTMReg(nn.Module):
    def __init__(self, d_feat, hidden=32, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_feat, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _train_lstm(X_tr, y_tr, X_va, y_va, d_feat, epochs=80, batch=4096, patience=10, seed=42,
                sample_weights=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _LSTMReg(d_feat).to(_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    X_tr = np.ascontiguousarray(X_tr)
    y_tr = np.ascontiguousarray(y_tr)

    def _eval(X, y):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i:i + batch]).float().to(_DEVICE)
                yb = torch.from_numpy(y[i:i + batch]).float().to(_DEVICE)
                total += float(lossf(model(xb), yb).item()) * len(xb)
                n += len(xb)
        return total / max(n, 1)

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_tr))
        for i in range(0, len(X_tr), batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(X_tr[idx]).float().to(_DEVICE)
            yb = torch.from_numpy(y_tr[idx]).float().to(_DEVICE)
            opt.zero_grad()
            pred = model(xb)
            if sample_weights is None:
                loss = lossf(pred, yb)
            else:
                wb = torch.from_numpy(sample_weights[idx]).float().to(_DEVICE)
                loss = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1e-6)
            loss.backward()
            opt.step()
        va_loss = _eval(X_va, y_va)
        if va_loss < best_val - 1e-6:
            best_val, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _daily_ic(pred, label):
    frame = pd.concat({"prediction": pred, "label": label}, axis=1).dropna()
    ic = frame.groupby(level="datetime").apply(
        lambda g: g["prediction"].corr(g["label"], method="spearman")).dropna()
    return ic


def _metrics(ic):
    ic = ic.astype(float)
    if len(ic) == 0:
        return {"ic_mean": None, "icir": None, "n": 0}
    return {
        "ic_mean": float(ic.mean()),
        "icir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else None,
        "n": int(len(ic)),
    }


def train_lgb_lstm_fusion(
    factor_keys, instruments, train_period, valid_period, test_period,
    label_horizon=10, label_norm="csrank", fusion_weight=0.5,
    param_profile="conservative", lstm_epochs=80, pool="a500",
    label_mode="tradable_open_to_open", dataset=None, train_dates=None, train_weights=None,
    extra_features=None, train_sample_filter=None, seed=42, calibration_period=None,
):
    """训练 LGB + LSTM, 返回融合预测与各模型指标。

    Returns: {fused_pred, lgb_pred, lstm_pred, label, fused, lgb, lstm}
    """
    _require_tradable_label_mode(label_mode)
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    manager = QlibModelManager()
    params = dict(PARAM_PROFILES[param_profile]["lightgbm"])
    params.update({"seed": int(seed), "bagging_seed": int(seed), "feature_fraction_seed": int(seed), "data_random_seed": int(seed)})
    d_feat = len(factor_keys)

    # ---- 连续分段 (供 LSTM 序列) ----
    anchor = pd.Timestamp(test_period[0])
    valid_start = pd.Timestamp(valid_period[0])
    cont_train = (train_period[0], (valid_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    cont_valid = (valid_period[0], (anchor - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    cont_test = test_period

    task_config = manager._build_task_config(
        "lightgbm_rank", "WhiteBoxHandler", instruments,
        cont_train, cont_valid, cont_test,
        model_params=params, label_horizon=label_horizon,
        custom_factor_keys=factor_keys, label_norm=label_norm,
        enable_cross_sectional=False, enable_winsorize=True,
        label_mode=label_mode,
    )
    source = dataset if dataset is not None else init_instance_by_config(task_config["dataset"])
    input_contract = getattr(getattr(source, 'handler', None), '_feature_contract', None)
    from core.adapters.qlib.fundval import require_dataset_contract
    require_dataset_contract(source, factor_keys)
    train_frame, valid_frame, test_frame = source.prepare(
        ["train", "valid", "test"], col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    frames = {"train": train_frame, "valid": valid_frame, "test": test_frame}
    if extra_features is not None:
        for frame in frames.values():
            aligned = extra_features.reindex(frame.index)
            if aligned.isna().any().any():
                raise RuntimeError("实验上下文特征未覆盖全部Qlib样本")
            for key in aligned.columns:
                frame[("feature", key)] = aligned[key].to_numpy()
    label_test = _as_series(_PreparedFactorDataset(frames, factor_keys).prepare("test", col_set="label"))

    def _filter_samples(frame):
        if train_sample_filter is None or frame.empty:
            return frame
        mask = np.asarray(train_sample_filter(frame.index), dtype=bool)
        if len(mask) != len(frame):
            raise RuntimeError("训练样本过滤器返回长度不匹配")
        return frame.loc[mask]

    # purge 边界
    valid_start_ts = pd.Timestamp(cont_valid[0])
    purge_gap = int(label_horizon) + 1
    train_end_dt = _purged_end_date(train_frame, valid_start_ts, purge_gap)
    if calibration_period:
        calibration_start = pd.Timestamp(calibration_period[0])
        valid_end_dt = _purged_end_date(valid_frame, calibration_start, purge_gap)
        calibration_end_dt = _purged_end_date(valid_frame, anchor, purge_gap)
    else:
        valid_end_dt = _purged_end_date(valid_frame, anchor, purge_gap)

    # ---- LGB (purge 切片) ----
    lgb_train = _slice_dates(train_frame, cont_train[0], train_end_dt)
    lgb_train = _filter_samples(lgb_train)
    if train_dates is not None:
        selected = set(pd.to_datetime(list(train_dates)).strftime("%Y-%m-%d"))
        dates = pd.to_datetime(lgb_train.index.get_level_values("datetime")).strftime("%Y-%m-%d")
        lgb_train = lgb_train.loc[dates.isin(selected)]
    lgb_valid = _slice_dates(valid_frame, cont_valid[0], valid_end_dt)
    lgb_valid = _filter_samples(lgb_valid)
    lgb_frames = {"train": lgb_train, "valid": lgb_valid, "test": test_frame}
    if calibration_period:
        lgb_calibration = _filter_samples(_slice_dates(valid_frame, calibration_start, calibration_end_dt))
        lgb_frames["calibrate"] = lgb_calibration
    lgb_ds = _PreparedFactorDataset(lgb_frames, factor_keys)
    lgb_model = init_instance_by_config(task_config["model"])
    reweighter = _DateReweighter(train_weights) if train_weights else None
    lgb_model.fit(lgb_ds, verbose_eval=0, log_metrics=False, reweighter=reweighter)
    lgb_valid_pred = _as_series(lgb_model.predict(lgb_ds, "valid"))
    lgb_pred = _as_series(lgb_model.predict(lgb_ds, "test"))
    valid_label = _as_series(lgb_ds.prepare("valid", col_set="label"))
    if calibration_period:
        lgb_calibration_pred = _as_series(lgb_model.predict(lgb_ds, "calibrate"))
        calibration_label = _as_series(lgb_ds.prepare("calibrate", col_set="label"))
    else:
        lgb_calibration_pred = lgb_valid_pred
        calibration_label = valid_label

    # ---- LSTM (连续序列 + purge 切分) ----
    lab_panel = pd.concat([_as_series(lgb_ds.prepare(s, col_set="label")) for s in (("train", "valid", "calibrate", "test") if calibration_period else ("train", "valid", "test"))])
    del lgb_train, lgb_valid, lgb_frames, lgb_ds
    gc.collect()
    feat_panel = pd.concat([_select_feature_keys(frames[s], factor_keys) for s in ("train", "valid", "test")])
    seg_map = pd.Series(
        ["train"] * len(train_frame) + ["valid"] * len(valid_frame) + ["test"] * len(test_frame),
        index=feat_panel.index,
    )
    del frames, train_frame, valid_frame, test_frame
    gc.collect()
    X_seq, y_seq, idx_seq, seg_seq = _build_sequences(feat_panel, lab_panel, seg_map)
    index_names = list(feat_panel.index.names)
    del feat_panel, lab_panel, seg_map
    gc.collect()
    last_dates = pd.to_datetime([t[0] for t in idx_seq])
    seg_seq = np.full(len(X_seq), "drop", dtype=object)
    train_mask = last_dates <= train_end_dt
    if train_dates is not None:
        selected = set(pd.to_datetime(list(train_dates)).strftime("%Y-%m-%d"))
        train_mask &= np.isin(last_dates.strftime("%Y-%m-%d"), list(selected))
    seg_seq[train_mask] = "train"
    vmask = (last_dates >= pd.Timestamp(cont_valid[0])) & (last_dates <= valid_end_dt)
    seg_seq[vmask] = "valid"
    if calibration_period:
        cmask = (last_dates >= calibration_start) & (last_dates <= calibration_end_dt)
        seg_seq[cmask] = "calibrate"
    seg_seq[last_dates >= anchor] = "test"
    if train_sample_filter is not None:
        sample_keep = np.asarray(train_sample_filter(pd.MultiIndex.from_tuples(
            [tuple(row) for row in idx_seq], names=index_names)), dtype=bool)
        if len(sample_keep) != len(seg_seq):
            raise RuntimeError("序列训练样本过滤器返回长度不匹配")
        seg_seq[(seg_seq == "train") & ~sample_keep] = "drop"
        seg_seq[(seg_seq == "valid") & ~sample_keep] = "drop"
        seg_seq[(seg_seq == "calibrate") & ~sample_keep] = "drop"
    keep = seg_seq != "drop"
    X_seq, y_seq, idx_seq, seg_seq = X_seq[keep], y_seq[keep], np.asarray(idx_seq, dtype=object)[keep], seg_seq[keep]
    del last_dates, keep

    from sklearn.preprocessing import StandardScaler
    m_tr = seg_seq == "train"
    m_va = seg_seq == "valid"
    m_cal = seg_seq == "calibrate"
    m_te = seg_seq == "test"
    if calibration_period and not m_cal.any():
        raise RuntimeError(
            f"校准序列为空: calibration={calibration_period}, "
            f"calibration_end={calibration_end_dt}, calibration_labels={len(calibration_label)}, "
            f"train={int(m_tr.sum())}, valid={int(m_va.sum())}, test={int(m_te.sum())}"
        )
    seq_weights = _date_weight_array(idx_seq, train_weights)
    scaler = StandardScaler()
    scaler_weights = None
    if seq_weights is not None:
        scaler_weights = np.repeat(seq_weights[m_tr], X_seq.shape[1])
    scaler.fit(X_seq[m_tr].reshape(-1, d_feat), sample_weight=scaler_weights)
    seq_shape = X_seq.shape
    X_seq_s = scaler.transform(X_seq.reshape(-1, d_feat)).reshape(seq_shape)
    del X_seq
    gc.collect()

    lstm_model = _train_lstm(
        X_seq_s[m_tr].astype(np.float32), y_seq[m_tr].astype(np.float32),
        X_seq_s[m_va].astype(np.float32), y_seq[m_va].astype(np.float32), d_feat,
        epochs=lstm_epochs,
        sample_weights=seq_weights[m_tr] if seq_weights is not None else None,
        seed=int(seed),
    )
    lstm_model.eval()
    _pred_batch = 4096
    _valid_preds = []
    X_va = X_seq_s[m_va].astype(np.float32)
    with torch.no_grad():
        for _i in range(0, len(X_va), _pred_batch):
            _xb = torch.from_numpy(X_va[_i:_i + _pred_batch]).to(_DEVICE)
            _valid_preds.append(lstm_model(_xb).cpu().numpy())
    lstm_valid = np.concatenate(_valid_preds)
    va_idx = np.asarray(idx_seq, dtype=object)[m_va]
    valid_index = pd.MultiIndex.from_tuples([tuple(t) for t in va_idx], names=index_names)
    lstm_valid_pred = pd.Series(lstm_valid, index=valid_index)
    del X_va, _valid_preds, lstm_valid, va_idx, valid_index
    if calibration_period:
        _calibration_preds = []
        X_cal = X_seq_s[m_cal].astype(np.float32)
        with torch.no_grad():
            for _i in range(0, len(X_cal), _pred_batch):
                _xb = torch.from_numpy(X_cal[_i:_i + _pred_batch]).to(_DEVICE)
                _calibration_preds.append(lstm_model(_xb).cpu().numpy())
        lstm_calibration = np.concatenate(_calibration_preds)
        cal_idx = np.asarray(idx_seq, dtype=object)[m_cal]
        calibration_index = pd.MultiIndex.from_tuples([tuple(t) for t in cal_idx], names=index_names)
        lstm_calibration_pred = pd.Series(lstm_calibration, index=calibration_index)
        del X_cal, _calibration_preds, lstm_calibration, cal_idx, calibration_index
    else:
        lstm_calibration_pred = lstm_valid_pred
    _preds = []
    X_te = X_seq_s[m_te].astype(np.float32)
    with torch.no_grad():
        for _i in range(0, len(X_te), _pred_batch):
            _xb = torch.from_numpy(X_te[_i:_i + _pred_batch]).to(_DEVICE)
            _preds.append(lstm_model(_xb).cpu().numpy())
    lstm_test = np.concatenate(_preds)
    del X_te
    te_idx = np.asarray(idx_seq, dtype=object)[m_te]
    lstm_pred = pd.Series(
        lstm_test,
        index=pd.MultiIndex.from_tuples([tuple(t) for t in te_idx], names=index_names),
    )
    del X_seq_s, y_seq, idx_seq, seg_seq, seq_weights, m_tr, m_va, m_cal, m_te, te_idx
    gc.collect()

    # ---- 0.5/0.5 秩融合 ----
    lgb_valid_rank = _rank_by_date(lgb_valid_pred)
    lstm_valid_rank = _rank_by_date(lstm_valid_pred).reindex(lgb_valid_rank.index)
    valid_pred = (1 - fusion_weight) * lgb_valid_rank + fusion_weight * lstm_valid_rank
    valid_pred = valid_pred.reindex(valid_label.index)
    lgb_calibration_rank = _rank_by_date(lgb_calibration_pred)
    lstm_calibration_rank = _rank_by_date(lstm_calibration_pred).reindex(lgb_calibration_rank.index)
    calibration_pred = (1 - fusion_weight) * lgb_calibration_rank + fusion_weight * lstm_calibration_rank
    calibration_pred = calibration_pred.reindex(calibration_label.index)
    lgb_rank = _rank_by_date(lgb_pred)
    lstm_rank = _rank_by_date(lstm_pred).reindex(lgb_rank.index)
    fused_pred = (1 - fusion_weight) * lgb_rank + fusion_weight * lstm_rank
    fused_pred = fused_pred.reindex(label_test.index)

    lgb_model._feature_contract = input_contract
    result = {
        "fused_pred": fused_pred,
        "valid_pred": valid_pred,
        "valid_label": valid_label,
        "calibration_pred": calibration_pred,
        "calibration_label": calibration_label,
        "lgb_pred": lgb_pred,
        "lstm_pred": lstm_pred,
        "label": label_test,
        "fused": _metrics(_daily_ic(fused_pred, label_test)),
        "lgb": _metrics(_daily_ic(lgb_pred, label_test)),
        "lstm": _metrics(_daily_ic(lstm_pred, label_test)),
        # 供持久化: qlib LGB 模型 + LSTM 权重(CPU 化, 便于 pickle)
        "lgb_model": lgb_model,
        'feature_contract': input_contract,
        "lstm_state": {k: v.detach().cpu() for k, v in lstm_model.state_dict().items()},
        "lstm_scaler": {
            "mean": scaler.mean_.astype(float),
            "scale": scaler.scale_.astype(float),
        },
    }

    del source, lstm_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def persist_fusion_run(
    fr, factor_keys, instruments, train_period, valid_period, test_period,
    label_horizon=10, label_norm="csrank", fusion_weight=0.5,
    param_profile="conservative", experiment_name="a_share_quant",
    label_mode="tradable_open_to_open", training_metadata=None,
):
    """把 LGB+LSTM 融合训练结果落成 MLflow run（供已训练模型页展示/复用）。

    落盘内容与 qlib 单模型 run 兼容:
      - params.pkl           = qlib LGB 模型 (list_trained_models 的识别信号)
      - params_lstm.pkl      = LSTM 权重 (CPU 化 state_dict, 标识融合模型)
      - pred.pkl / label.pkl = 融合预测与标签 (可回测/预测)
      - factor_keys.json     = 因子 key 清单 (factor_label 解析)
      - params/ 下记 model.class=LGB+LSTM 与训练分段等
      - metrics/ 下记 IC/ICIR/LGB_IC/LSTM_IC
    Returns: MLflow run_id (str | None)
    """
    _require_tradable_label_mode(label_mode)
    import json
    from qlib.workflow import R

    with R.start(experiment_name=experiment_name):
        logged_params = {
            "model.class": "LGB+LSTM",
            "model.kwargs.fusion_weight": str(float(fusion_weight)),
            "model.kwargs.lstm_seq_len": str(_SEQ_LEN),
            "dataset.kwargs.handler.class": "WhiteBoxHandler",
            "dataset.kwargs.handler.kwargs.factor_keys": str(list(factor_keys or [])),
            "dataset.kwargs.handler.kwargs.instruments": (
                instruments if isinstance(instruments, str) else f"list_{len(instruments)}_stocks"),
            "dataset.kwargs.segments.train": f"('{train_period[0]}', '{train_period[1]}')",
            "dataset.kwargs.segments.valid": f"('{valid_period[0]}', '{valid_period[1]}')",
            "dataset.kwargs.segments.test": f"('{test_period[0]}', '{test_period[1]}')",
            "dataset.kwargs.handler.kwargs.label": (
                f"Ref($open, -{label_horizon + 1}) / Ref($open, -1) - 1"
                if label_mode == "tradable_open_to_open"
                else f"Ref($close, -{label_horizon}) / $close - 1"
            ),
            "label_mode": str(label_mode),
            "label_norm": str(label_norm),
            "param_profile": str(param_profile),
            "label_horizon": str(label_horizon),
            "fusion_weight": str(float(fusion_weight)),
        }
        metadata = dict(training_metadata or {})
        for key in (
            "training_mode", "expert_role", "state_subset_version",
            "state_subset_threshold", "state_subset_total_dates",
            "state_subset_n_dates", "state_subset_ratio",
        ):
            if metadata.get(key) is not None:
                logged_params[key] = str(metadata[key])
        R.log_params(**logged_params)

        save_kwargs = {}
        if fr.get("lgb_model") is not None:
            save_kwargs["params.pkl"] = fr["lgb_model"]
        if fr.get("lstm_state") is not None:
            save_kwargs["params_lstm.pkl"] = fr["lstm_state"]
            save_kwargs["lstm_bundle.pkl"] = {
                "schema_version": _BUNDLE_SCHEMA_VERSION,
                "state_dict": fr["lstm_state"],
                "scaler": fr.get("lstm_scaler"),
                "factor_keys": list(factor_keys or []),
                "seq_len": _SEQ_LEN,
                "fusion_weight": float(fusion_weight),
                "instruments": list(instruments) if isinstance(instruments, (list, tuple)) else instruments,
                "training_metadata": metadata,
            }
        if fr.get("fused_pred") is not None:
            save_kwargs["pred.pkl"] = fr["fused_pred"]
        if fr.get("label") is not None:
            save_kwargs["label.pkl"] = fr["label"]
        save_kwargs["factor_keys.json"] = json.dumps(
            {"custom_factor_keys": list(factor_keys or [])}, ensure_ascii=False)
        if fr.get('feature_contract') is not None:
            save_kwargs['feature_contract.json'] = json.dumps(fr['feature_contract'], ensure_ascii=False)
        if metadata:
            save_kwargs["expert_training_profile.json"] = json.dumps(
                metadata, ensure_ascii=False)
        R.save_objects(**save_kwargs)

        metrics = {}
        fused_m = fr.get("fused") or {}
        if fused_m.get("ic_mean") is not None:
            metrics["IC"] = float(fused_m["ic_mean"])
            metrics["Rank IC"] = float(fused_m["ic_mean"])
        if fused_m.get("icir") is not None:
            metrics["ICIR"] = float(fused_m["icir"])
        for prefix, key in (("LGB", "lgb"), ("LSTM", "lstm")):
            sub = fr.get(key) or {}
            if sub.get("ic_mean") is not None:
                metrics[f"{prefix}_IC"] = float(sub["ic_mean"])
        if metrics:
            R.log_metrics(**metrics)

        rec = R.get_recorder()
        run_id = getattr(rec, "id", None) if rec is not None else None
    return run_id


def predict_lgb_lstm_fusion(experiment_id, run_id, predict_start, predict_end,
                            instruments="all", dataset=None, use_cache=True):
    """加载已落盘的 LGB+LSTM 并执行真实秩融合预测。"""
    import ast
    from qlib.data.dataset.handler import DataHandlerLP
    from core.shared import config as app_config

    started_at = time.perf_counter()

    mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", "")).resolve()
    run_dir = (mlruns_dir / str(experiment_id) / str(run_id)).resolve()
    if not run_dir.is_relative_to(mlruns_dir):
        raise ValueError("模型路径超出 MLflow 目录")
    artifacts = run_dir / "artifacts"
    params_dir = run_dir / "params"
    if not (artifacts / "params.pkl").exists() or not (artifacts / "params_lstm.pkl").exists():
        raise RuntimeError(f"LGB+LSTM 模型文件不完整: {experiment_id}/{run_id}")

    def _param(name, default=None):
        path = params_dir / name
        return path.read_text(encoding="utf-8").strip() if path.exists() else default

    def _period(name, default):
        raw = _param(name)
        if not raw:
            return default
        try:
            value = ast.literal_eval(raw)
            return tuple(value)
        except Exception:
            return default

    factor_keys = []
    factor_path = artifacts / "factor_keys.json"
    if factor_path.exists():
        try:
            raw = pickle.loads(factor_path.read_bytes())
            if isinstance(raw, str):
                import json
                raw = json.loads(raw)
            factor_keys = list((raw or {}).get("custom_factor_keys") or [])
        except Exception:
            try:
                import json
                factor_keys = list(json.loads(factor_path.read_text(encoding="utf-8")).get("custom_factor_keys") or [])
            except Exception:
                pass
    if not factor_keys:
        try:
            factor_keys = list(ast.literal_eval(_param("dataset.kwargs.handler.kwargs.factor_keys", "[]")))
        except Exception:
            pass
    if not factor_keys:
        raise RuntimeError("融合模型缺少 factor_keys，无法恢复训练特征顺序")

    from core.adapters.qlib.fundval import require_run_contract
    require_run_contract(artifacts, factor_keys)
    from core.adapters.qlib.fundval import current_feature_contract
    feature_signature = current_feature_contract(factor_keys)

    train_period = _period("dataset.kwargs.segments.train", ("2022-01-01", "2024-12-31"))
    valid_period = _period("dataset.kwargs.segments.valid", ("2025-01-01", "2025-06-30"))
    label_horizon = int(float(_param("label_horizon", "10")))
    label_mode = _param("label_mode", "close_to_close")
    fusion_weight = float(_param("fusion_weight", "0.5"))
    seq_len = int(float(_param("model.kwargs.lstm_seq_len", str(_SEQ_LEN))))
    profile = _param("param_profile", "conservative")
    params = dict(PARAM_PROFILES.get(profile, PARAM_PROFILES["conservative"])["lightgbm"])

    bundle_path = artifacts / "lstm_bundle.pkl"
    bundle = None
    if bundle_path.exists():
        with open(bundle_path, "rb") as handle:
            bundle = pickle.load(handle)
        scaler_mean, scaler_scale, state = _validate_lstm_bundle(
            bundle, factor_keys, seq_len, fusion_weight)

    manager = QlibModelManager()
    history_start = train_period[0]

    provider = Path(str(app_config.QLIB_PROVIDER_URI).replace("file:", "")).resolve()
    cache_path = None
    cache_key = None
    data_signature_before = None
    if dataset is None and use_cache:
        fingerprint_started = time.perf_counter()
        data_signature_before = _qlib_data_signature(provider, instruments)
        logger.info("[LGB+LSTM] 数据指纹耗时 %.2fs", time.perf_counter() - fingerprint_started)
    if bundle is not None and dataset is None and use_cache:
        artifact_signature = hashlib.sha256(
            (_file_sha256(artifacts / "params.pkl") +
             _file_sha256(artifacts / "params_lstm.pkl") +
             _file_sha256(bundle_path)).encode("ascii")).hexdigest()
        cache_key = _prediction_cache_key(
            experiment_id, run_id, predict_start, predict_end, instruments,
            factor_keys, seq_len, fusion_weight, artifact_signature, data_signature_before, feature_signature)
        cache_path = provider / "prediction_cache" / "lgb_lstm" / f"{cache_key}.pkl"
        cached = _load_prediction_cache(cache_path, cache_key)
        if cached is not None:
            logger.info("[LGB+LSTM] 持久化预测缓存命中: %s/%s %s~%s (%d条)",
                        experiment_id, run_id, predict_start, predict_end, len(cached))
            return cached, None

    dataset_started = time.perf_counter()
    task_config = manager._build_task_config(
        "lightgbm_rank", "WhiteBoxHandler", instruments,
        train_period, valid_period, (predict_start, predict_end),
        model_params=params, label_horizon=label_horizon,
        custom_factor_keys=factor_keys, label_norm=_param("label_norm", "csrank"),
        enable_cross_sectional=False, enable_winsorize=True,
        label_mode=label_mode,
    )
    task_config["dataset"]["kwargs"]["segments"]["inference_history"] = (history_start, predict_end)
    if dataset is not None:
        source = dataset
    else:
        from qlib.utils import init_instance_by_config
        source = init_instance_by_config(task_config["dataset"])
    from core.adapters.qlib.fundval import require_dataset_contract
    require_dataset_contract(source, factor_keys)
    logger.info("[LGB+LSTM] Dataset 准备完成: history=%s output=%s~%s, 耗时 %.2fs",
                history_start, predict_start, predict_end, time.perf_counter() - dataset_started)

    lgb_started = time.perf_counter()
    with open(artifacts / "params.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    lgb_pred = _as_series(lgb_model.predict(source, "test"))
    lgb_pred = _slice_dates(lgb_pred, predict_start, predict_end)
    logger.info("[LGB+LSTM] LGB 推理耗时 %.2fs (%d条)",
                time.perf_counter() - lgb_started, len(lgb_pred))

    sequence_started = time.perf_counter()
    try:
        test_features = source.prepare(
            "inference_history", col_set="feature", data_key=DataHandlerLP.DK_L)
    except (KeyError, ValueError):
        test_features = source.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    test_features = _select_feature_keys(test_features, factor_keys)
    X_seq, idx_seq = _build_inference_sequences(test_features, predict_start, seq_len)
    if not len(X_seq):
        raise RuntimeError("预测区间没有足够的 LSTM 序列样本")
    logger.info("[LGB+LSTM] 序列构建耗时 %.2fs (%d条)",
                time.perf_counter() - sequence_started, len(X_seq))

    if bundle is None:
        scaler_started = time.perf_counter()
        train_features = source.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        train_features = _select_feature_keys(train_features, factor_keys)
        train_end = _purged_end_date(train_features, pd.Timestamp(valid_period[0]), label_horizon + 1)
        scaler_mean, scaler_scale = _rebuild_sequence_scaler(
            train_features, train_end, seq_len=seq_len)
        with open(artifacts / "params_lstm.pkl", "rb") as f:
            state = pickle.load(f)
        bundle = {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "state_dict": state,
            "scaler": {"mean": scaler_mean.astype(float), "scale": scaler_scale.astype(float)},
            "factor_keys": list(factor_keys),
            "seq_len": int(seq_len),
            "fusion_weight": float(fusion_weight),
            "migrated_from_training_period": list(train_period),
        }
        with _PREDICTION_CACHE_LOCK:
            _save_pickle_atomic(bundle_path, bundle)
        logger.warning("[LGB+LSTM] 旧 run scaler 已重建并固化 bundle: %s/%s, 耗时 %.2fs",
                       experiment_id, run_id, time.perf_counter() - scaler_started)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    X_seq = ((X_seq - scaler_mean) / scaler_scale).astype(np.float32)

    lstm_started = time.perf_counter()
    lstm_model = _LSTMReg(len(factor_keys)).to(_DEVICE)
    lstm_model.load_state_dict(state)
    lstm_model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_seq), 4096):
            xb = torch.from_numpy(X_seq[i:i + 4096]).to(_DEVICE)
            preds.append(lstm_model(xb).cpu().numpy())
    lstm_pred = pd.Series(
        np.concatenate(preds),
        index=pd.MultiIndex.from_tuples([tuple(v) for v in idx_seq], names=test_features.index.names),
    )
    lstm_pred = _slice_dates(lstm_pred, predict_start, predict_end)
    common = lgb_pred.index.intersection(lstm_pred.index)
    if common.empty:
        raise RuntimeError("LGB 与 LSTM 预测无可融合的共同截面")
    fused = (1 - fusion_weight) * _rank_by_date(lgb_pred.reindex(common)) + \
        fusion_weight * _rank_by_date(lstm_pred.reindex(common))
    fused = fused.sort_index()
    raw_components = {
        "lgb_raw_model_score": lgb_pred.reindex(common).sort_index(),
        "lstm_raw_model_score": lstm_pred.reindex(common).sort_index(),
    }
    logger.info("[LGB+LSTM] LSTM+融合耗时 %.2fs, 总耗时 %.2fs (%d条)",
                time.perf_counter() - lstm_started, time.perf_counter() - started_at, len(fused))

    data_signature_after = None
    if dataset is None and use_cache:
        data_signature_after = _qlib_data_signature(provider, instruments)
    if (data_signature_before is not None and data_signature_after is not None and
            data_signature_before != data_signature_after):
        logger.warning("[LGB+LSTM] 预测期间 Qlib 数据发生变化，拒绝写入缓存")
        return fused, source
    if cache_path is None and dataset is None and use_cache:
        artifact_signature = hashlib.sha256(
            (_file_sha256(artifacts / "params.pkl") +
             _file_sha256(artifacts / "params_lstm.pkl") +
             _file_sha256(bundle_path)).encode("ascii")).hexdigest()
        cache_key = _prediction_cache_key(
            experiment_id, run_id, predict_start, predict_end, instruments,
            factor_keys, seq_len, fusion_weight, artifact_signature, data_signature_after, feature_signature)
        cache_path = provider / "prediction_cache" / "lgb_lstm" / f"{cache_key}.pkl"
    if cache_path is not None and cache_key is not None:
        from core.shared.fundamental_contract import FundamentalDataError
        if current_feature_contract(factor_keys) != feature_signature:
            raise FundamentalDataError('预测期间基本面缓存或交易日历发生变化')
        metadata = {
            "schema_version": _PREDICTION_CACHE_VERSION,
            "model_id": f"{experiment_id}/{run_id}",
            "experiment_id": str(experiment_id),
            "run_id": str(run_id),
            "artifact_signature": artifact_signature,
            "data_signature": data_signature_after or data_signature_before,
            "score_semantics": "daily_cross_sectional_rank_fusion",
            "raw_model_score_available": False,
            "raw_component_scores_available": True,
            "component_semantics": {
                "lgb_raw_model_score": "raw_model_score",
                "lstm_raw_model_score": "raw_model_score",
            },
            "factor_keys": list(factor_keys),
            "seq_len": int(seq_len),
            "fusion_weight": float(fusion_weight),
            "train_period": list(train_period),
            "valid_period": list(valid_period),
            "predict_period": [str(predict_start), str(predict_end)],
            "instruments": list(instruments) if isinstance(instruments, (list, tuple)) else str(instruments),
        }
        with _PREDICTION_CACHE_LOCK:
            _save_pickle_atomic(cache_path, {
                "key": cache_key,
                "prediction": fused,
                "components": raw_components,
                "metadata": metadata,
            })
        logger.info("[LGB+LSTM] 预测缓存已保存: %s", cache_path.name)
    return fused, source
