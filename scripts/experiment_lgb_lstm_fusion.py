"""LGB + LSTM 融合 vs LGB + Ridge 融合 实验

在核心+卫星合并因子集上, 对比两种双模型融合:
  - fusion_lgb_ridge: 0.5*rank(LGB) + 0.5*rank(Ridge)
  - fusion_lgb_lstm : 0.5*rank(LGB) + 0.5*rank(LSTM)

说明: 为让 LSTM 有完整的时间序列历史, 本实验采用连续分段(无 purge gap),
所有 arm 共用同一数据口径, 相对比较公平; 测试集 RankIC 与之前 purge 口径略有不同。

用法:
    python scripts/experiment_lgb_lstm_fusion.py --role trend --folds 12
"""
import argparse
import copy
import datetime as dt
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from core.adapters.qlib.config import init_qlib
from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager
from core.adapters.qlib.factor_set_evaluator import (
    _rank_by_date, _as_series, _select_feature_keys, _PreparedFactorDataset,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20


def load_sets():
    with open(os.path.join(ROOT, "core", "domain", "factors", "factor_pool_state.json"),
              encoding="utf-8") as f:
        pool = json.load(f)
    cur = pool["curated"]
    return {
        "core": list(cur["core_base"]["flat_list"]),
        "trend": list(cur["trend_satellite"]["flat_list"]),
        "rebound": list(cur["rebound_satellite"]["flat_list"]),
    }


def _daily_corr(frame):
    return frame.groupby(level="datetime").apply(
        lambda g: g["prediction"].corr(g["label"], method="spearman")
    ).dropna()


def _slice_dates(frame, start, end):
    dates = pd.to_datetime(frame.index.get_level_values("datetime"))
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame.loc[mask]


def _metrics(daily):
    daily = daily.astype(float)
    if len(daily) == 0:
        return {"rank_ic": None, "icir": None, "positive_ratio": None, "n": 0}
    return {
        "rank_ic": float(daily.mean()),
        "icir": float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else None,
        "positive_ratio": float((daily > 0).mean()),
        "n": int(len(daily)),
    }


def _contiguous_periods(anchor_str):
    """连续分段: train=[-60m, -6m), valid=[-6m, anchor), test=[anchor, +3m)"""
    anchor = pd.Timestamp(anchor_str)
    train_start = anchor - pd.DateOffset(months=66)
    valid_start = anchor - pd.DateOffset(months=6)
    test_end = anchor + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    return (
        (train_start.strftime("%Y-%m-%d"), (valid_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
        (valid_start.strftime("%Y-%m-%d"), (anchor - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
        (anchor.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
    )


def _build_sequences(features, labels, seg_map, seq_len=SEQ_LEN, train_step=5):
    """从面板(索引=datetime,instrument)构建每个 instrument 的滑动窗口序列。

    train 段按 train_step 下采样(重叠窗口信息冗余大), valid/test 全量。
    返回 X [N,seq_len,F](float32), y [N](float32), last_index [N], seg [N]。
    """
    features = features.sort_index(level=[1, 0])  # sort by instrument then datetime
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


class _LSTMReg(nn.Module):
    def __init__(self, d_feat, hidden=32, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_feat, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _train_lstm(X_tr, y_tr, X_va, y_va, d_feat, epochs=80, batch=4096, patience=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _LSTMReg(d_feat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    X_tr = np.ascontiguousarray(X_tr)
    y_tr = np.ascontiguousarray(y_tr)
    best_val = float("inf")
    best_state = None
    wait = 0

    def _eval(X, y):
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i:i + batch]).float().to(DEVICE)
                yb = torch.from_numpy(y[i:i + batch]).float().to(DEVICE)
                total += float(lossf(model(xb), yb).item()) * len(xb)
                n += len(xb)
        return total / max(n, 1)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_tr))
        for i in range(0, len(X_tr), batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(X_tr[idx]).float().to(DEVICE)
            yb = torch.from_numpy(y_tr[idx]).float().to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        va_loss = _eval(X_va, y_va)
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_experiment(role, max_folds=None, start_fold=1):
    init_qlib()
    sets = load_sets()
    merged_keys = list(dict.fromkeys(sets["core"] + sets[role]))
    d_feat = len(merged_keys)
    print(f"role={role} merged={d_feat} device={DEVICE} start_fold={start_fold}")

    anchors = ["2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01",
               "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01",
               "2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
    if max_folds:
        anchors = anchors[:max_folds]

    manager = QlibModelManager()
    params = dict(PARAM_PROFILES["conservative"]["lightgbm"])
    arms = ["merged_lgb", "merged_ridge", "merged_lstm", "fusion_lgb_ridge", "fusion_lgb_lstm"]
    checkpoint_path = os.path.join(ROOT, "data", "experiments", f"exp_{role}_lstm.json")
    accum = {a: [] for a in arms}
    if start_fold > 1 and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            saved = json.load(f).get("accum", {})
        for a in arms:
            accum[a] = list(saved.get(a, []))
        print(f"resume from checkpoint: { {a: len(v) for a, v in accum.items()} }")

    def _save():
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"role": role, "accum": accum}, f)
        except Exception:
            pass

    for idx, anchor_str in enumerate(anchors[start_fold - 1:], start_fold):
        t0 = time.perf_counter()
        print(f"[fold {idx}/{len(anchors)}] anchor={anchor_str}", flush=True)
        train_period, valid_period, test_period = _contiguous_periods(anchor_str)

        task_config = manager._build_task_config(
            "lightgbm_rank", "WhiteBoxHandler", "a500",
            train_period, valid_period, test_period,
            model_params=params, label_horizon=10,
            custom_factor_keys=merged_keys, label_norm="csrank",
            enable_cross_sectional=False, enable_winsorize=True,
        )
        source = init_instance_by_config(task_config["dataset"])
        train_frame, valid_frame, test_frame = source.prepare(
            ["train", "valid", "test"], col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        frames = {"train": train_frame, "valid": valid_frame, "test": test_frame}
        label_test = _as_series(_PreparedFactorDataset(frames, merged_keys).prepare("test", col_set="label"))

        # purge 边界: 与 factor_set_evaluator._make_rounds 一致 (21 交易日 ≈ 30 自然日)
        purge_cal = 30
        valid_start_ts = pd.Timestamp(valid_period[0])
        train_end_dt = valid_start_ts - pd.Timedelta(days=purge_cal)
        valid_end_dt = pd.Timestamp(anchor_str) - pd.Timedelta(days=purge_cal)

        # ---- LGB / Ridge: 用 purge 切片训练/验证, 避免标签前视 ----
        lgb_train_frame = _slice_dates(train_frame, train_period[0], train_end_dt)
        lgb_valid_frame = _slice_dates(valid_frame, valid_period[0], valid_end_dt)
        lgb_frames = {"train": lgb_train_frame, "valid": lgb_valid_frame, "test": test_frame}
        ds = _PreparedFactorDataset(lgb_frames, merged_keys)
        lgb_model = init_instance_by_config(task_config["model"])
        lgb_model.fit(ds, verbose_eval=0, log_metrics=False)
        lgb_pred = _as_series(lgb_model.predict(ds, "test"))

        # ---- Ridge ----
        X_tr_df = _select_feature_keys(lgb_train_frame, merged_keys)
        X_te_df = _select_feature_keys(test_frame, merged_keys)
        y_tr = _as_series(ds.prepare("train", col_set="label")).values
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_df.values, y_tr)
        ridge_pred = pd.Series(ridge.predict(X_te_df.values), index=test_frame.index)

        # ---- LSTM (连续序列) ----
        feat_panel = pd.concat([
            _select_feature_keys(frames[s], merged_keys) for s in ("train", "valid", "test")
        ])
        lab_panel = pd.concat([
            _as_series(ds.prepare(s, col_set="label")) for s in ("train", "valid", "test")
        ])
        seg_map = pd.Series(
            ["train"] * len(train_frame) + ["valid"] * len(valid_frame) + ["test"] * len(test_frame),
            index=feat_panel.index,
        )
        X_seq, y_seq, idx_seq, seg_seq = _build_sequences(feat_panel, lab_panel, seg_map)
        # 按 purge 边界重分配 segment, 并丢弃 purge 区序列(防标签前视)
        last_dates = pd.to_datetime([t[0] for t in idx_seq])
        seg_seq = np.full(len(X_seq), "drop", dtype=object)
        seg_seq[last_dates <= train_end_dt] = "train"
        vmask = (last_dates >= pd.Timestamp(valid_period[0])) & (last_dates <= valid_end_dt)
        seg_seq[vmask] = "valid"
        seg_seq[last_dates >= pd.Timestamp(anchor_str)] = "test"
        keep = seg_seq != "drop"
        X_seq, y_seq, idx_seq, seg_seq = X_seq[keep], y_seq[keep], np.asarray(idx_seq, dtype=object)[keep], seg_seq[keep]
        n_tr = int((seg_seq == "train").sum())
        n_va = int((seg_seq == "valid").sum())
        n_te = int((seg_seq == "test").sum())
        print(f"    seq built: {len(X_seq)} (train={n_tr} valid={n_va} test={n_te}) shape={X_seq.shape}", flush=True)

        m_tr = seg_seq == "train"
        m_va = seg_seq == "valid"
        m_te = seg_seq == "test"
        scaler = StandardScaler()
        scaler.fit(X_seq[m_tr].reshape(-1, d_feat))
        X_seq_s = scaler.transform(X_seq.reshape(-1, d_feat)).reshape(X_seq.shape)

        lstm_model = _train_lstm(
            X_seq_s[m_tr].astype(np.float32), y_seq[m_tr].astype(np.float32),
            X_seq_s[m_va].astype(np.float32), y_seq[m_va].astype(np.float32),
            d_feat,
        )
        print("    lstm trained", flush=True)
        lstm_model.eval()
        with torch.no_grad():
            lstm_test = lstm_model(torch.from_numpy(X_seq_s[m_te].astype(np.float32)).to(DEVICE)).cpu().numpy()
        te_idx = np.asarray(idx_seq, dtype=object)[m_te]
        lstm_pred = pd.Series(
            lstm_test,
            index=pd.MultiIndex.from_tuples([tuple(t) for t in te_idx], names=feat_panel.index.names),
        )

        # ---- 融合 ----
        lgb_rank = _rank_by_date(lgb_pred)
        ridge_rank = _rank_by_date(ridge_pred)
        lstm_rank = _rank_by_date(lstm_pred)

        f_lr = pd.concat({"prediction": 0.5 * lgb_rank + 0.5 * ridge_rank, "label": label_test}, axis=1).dropna()
        accum["fusion_lgb_ridge"].extend(_daily_corr(f_lr).tolist())
        # lstm_rank 索引与 label_test 对齐
        f_ll = pd.concat({
            "prediction": (0.5 * lgb_rank + 0.5 * lstm_rank).reindex(label_test.index),
            "label": label_test,
        }, axis=1).dropna()
        accum["fusion_lgb_lstm"].extend(_daily_corr(f_ll).tolist())

        accum["merged_lgb"].extend(_daily_corr(pd.concat({"prediction": lgb_pred, "label": label_test}, axis=1).dropna()).tolist())
        accum["merged_ridge"].extend(_daily_corr(pd.concat({"prediction": ridge_pred, "label": label_test}, axis=1).dropna()).tolist())
        accum["merged_lstm"].extend(_daily_corr(pd.concat({"prediction": lstm_pred.reindex(label_test.index), "label": label_test}, axis=1).dropna()).tolist())

        print(f"    done in {time.perf_counter()-t0:.0f}s", flush=True)
        _save()
        del source, train_frame, valid_frame, test_frame, frames, ds, lstm_model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n================== 汇总 ==================")
    print(f"{'arm':20s} {'RankIC':>8s} {'ICIR':>8s} {'正率':>7s} {'n':>5s}")
    result = {}
    for arm in arms:
        m = _metrics(pd.Series(accum[arm], dtype=float))
        result[arm] = m
        print(f"{arm:20s} {m['rank_ic'] if m['rank_ic'] is not None else float('nan'):>8.4f} "
              f"{m['icir'] if m['icir'] is not None else float('nan'):>8.4f} "
              f"{m['positive_ratio']:>7.4f} {m['n']:>5d}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="trend", choices=["trend", "rebound"])
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--start-fold", type=int, default=1)
    args = ap.parse_args()
    run_experiment(args.role, args.folds, args.start_fold)
