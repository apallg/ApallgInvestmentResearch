
# -*- coding: utf-8 -*-
"""regime6 态单次 OOS 评测: 正确率 + 可靠性 + 生产可用性判断 (point-in-time, 无前视)"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from core.data.fetcher import DataFetcher
from core.domain.regime.predictor import (
    get_regime_predictor, _make_regime_labels, _soft_position_ratio,
    _soft_signal_weights, STATE_NAMES, RATIO_MAP,
)

def main():
    rp = get_regime_predictor()
    train_end = pd.Timestamp(rp.last_train_date) if rp.last_train_date else None

    fetcher = DataFetcher()
    df = fetcher.get_index_daily("沪深300", days=365 * 2)
    df = df.sort_values("date").reset_index(drop=True)
    n_all = len(df)

    labels = _make_regime_labels(df, rp.label_horizon)
    close = df["close"].astype(float)
    fwd_ret = (close.shift(-rp.label_horizon) / close - 1).values
    rets = close.pct_change(fill_method=None)
    vol_annual = (rets.rolling(20).std() * np.sqrt(250)).values

    # 标签的波动维度 (leak 检查用): 当前 20d 波动率 (标签的"波动"维) vs 未来波动率
    fwd_vol = (rets.shift(-1).rolling(20).std() * np.sqrt(250)).values

    # 特征里的 vol_20d (训练特征之一)
    from core.domain.timing.model import _make_features as _mf
    feat_df = _mf(df, None)
    vol20_feat = None
    if "vol_20d" in feat_df.columns:
        vol20_feat = feat_df["vol_20d"].values

    # ---- OOS 逐日预测 (point-in-time) ----
    preds, truths, dates, probs_all = [], [], [], []
    skip = 0
    for i in range(len(df)):
        d = df["date"].iloc[i]
        if train_end is not None and d <= train_end:
            continue
        if i < 60 or i >= n_all - rp.label_horizon:
            continue
        p = rp.predict_from_df(df.iloc[:i + 1])
        if p is None:
            skip += 1
            continue
        preds.append(p)
        probs_all.append(p)
        truths.append(int(labels[i]))
        dates.append(d)

    if not truths:
        print(json.dumps({"error": "no OOS samples"}, ensure_ascii=False))
        return

    P = np.array(preds)
    T = np.array(truths)
    n = len(T)
    top1 = (P.argmax(1) == T).mean()
    top2 = np.any(np.argsort(P, 1)[:, -2:] == T[:, None], 1).mean()

    # 多分类指标
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, precision_recall_fscore_support
    macro_auc = None
    try:
        macro_auc = float(roc_auc_score(T, P, multi_class="ovr", average="macro"))
    except Exception:
        pass
    brier = float(np.mean(np.sum((P - np.eye(6)[T]) ** 2, axis=1)))
    try:
        ll = float(log_loss(T, P))
    except Exception:
        ll = None

    # 基线
    from collections import Counter
    mc = Counter(T).most_common(1)[0][1] / len(T)
    rand = 1 / 6

    # 混淆矩阵 + 每类 precision/recall/f1
    cm = np.zeros((6, 6), dtype=int)
    for p_, t_ in zip(P.argmax(1), T):
        cm[t_, p_] += 1
    prf = precision_recall_fscore_support(T, P.argmax(1), labels=list(range(6)), zero_division=0)

    # 方向归并 (up/down/flat) 与 波动归并 (volatile/steady)
    dir_map = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
    vol_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    Td = np.array([dir_map[t] for t in T]); Pd = np.array([dir_map[p] for p in P.argmax(1)])
    Tv = np.array([vol_map[t] for t in T]); Pv = np.array([vol_map[p] for p in P.argmax(1)])
    dir_acc = (Pd == Td).mean()
    vol_acc = (Pv == Tv).mean()
    # 波动归并的概率版: P_volatile = P[:,0]+P[:,1]+P[:,2]
    P_vol = P[:, :3].sum(1)
    try:
        vol_auc = float(roc_auc_score(Tv, P_vol))
    except Exception:
        vol_auc = None
    # 方向 3 分类 AUC
    Pd_prob = np.vstack([P[:, [0, 3]].sum(1), P[:, [1, 4]].sum(1), P[:, [2, 5]].sum(1)]).T
    try:
        dir_auc = float(roc_auc_score(Td, Pd_prob, multi_class="ovr", average="macro"))
    except Exception:
        dir_auc = None

    # ---- 可靠性 ----
    conf = P.max(1)
    # ECE (5 bins)
    edges = np.linspace(0, 1, 6)
    ece, cal_rows = 0.0, []
    for b in range(5):
        m = (conf >= edges[b]) & (conf < edges[b + 1])
        if b == 4:
            m = conf >= edges[b]
        if m.sum() == 0:
            continue
        p_avg = conf[m].mean(); acc = (P[m].argmax(1) == T[m]).mean()
        cal_rows.append((round(edges[b], 2), round(edges[b + 1], 2), int(m.sum()), round(p_avg, 3), round(acc, 3)))
        ece += (m.sum() / n) * abs(p_avg - acc)

    # 相邻日 flip rate (top state 变化) 与 position_ratio 换手
    top_seq = P.argmax(1)
    ratios = np.array([_soft_position_ratio(P[i]) for i in range(n)])
    flips = int((np.diff(top_seq) != 0).sum())
    flip_rate = flips / max(1, n - 1)
    ratio_churn = float(np.abs(np.diff(ratios)).mean())
    ratio_std = float(ratios.std())

    # 模型预测类别覆盖
    pred_classes = Counter(P.argmax(1))
    true_classes = Counter(T)
    never_predicted = [STATE_NAMES[i] for i in range(6) if pred_classes.get(i, 0) == 0]
    never_true = [STATE_NAMES[i] for i in range(6) if true_classes.get(i, 0) == 0]

    # 下游数量: soft position_ratio 与未来收益/未来波动的相关性 (生产上真正用的量)
    valid = ~np.isnan(fwd_ret[60:n + 60]) if False else True
    # 对齐: dates index i → P index. 记录对应 i
    idxs = []
    for i in range(len(df)):
        d = df["date"].iloc[i]
        if train_end is not None and d <= train_end:
            continue
        if i < 60 or i >= n_all - rp.label_horizon:
            continue
        idxs.append(i)
    idxs = np.array(idxs)
    fwd_r = np.array([fwd_ret[i] for i in idxs])
    fwd_v = np.array([fwd_vol[i] for i in idxs])
    cur_v = np.array([vol_annual[i] for i in idxs])
    m1 = ~np.isnan(fwd_r) & ~np.isnan(fwd_v)
    corr_ratio_fwdret = float(np.corrcoef(ratios[m1], fwd_r[m1])[0, 1]) if m1.sum() > 5 else None
    corr_ratio_fwdvol = float(np.corrcoef(ratios[m1], fwd_v[m1])[0, 1]) if m1.sum() > 5 else None
    corr_conf_vol = float(np.corrcoef(P_vol[m1], fwd_v[m1])[0, 1]) if m1.sum() > 5 else None
    # 波动状态: 预测 volatile 的概率 vs 未来波动排序能力 (spearman)
    from scipy.stats import spearmanr
    sp_vol = float(spearmanr(P_vol[m1], fwd_v[m1]).statistic) if m1.sum() > 5 else None
    # 波动真值下 ratio 的区分度
    r_vol_true = ratios[Tv == 0].mean() if (Tv == 0).any() else None
    r_steady_true = ratios[Tv == 1].mean() if (Tv == 1).any() else None

    # leak 检查: 标签"波动"维 (当前 vol) vs 特征 vol_20d 相关性; 以及当前 vol vs 未来 vol
    corr_labelvol_feat = None
    if vol20_feat is not None:
        lv = vol_annual[idxs]; fv = vol20_feat[idxs]
        m2 = ~np.isnan(lv) & ~np.isnan(fv)
        corr_labelvol_feat = float(np.corrcoef(lv[m2], fv[m2])[0, 1]) if m2.sum() > 5 else None
    m3 = ~np.isnan(cur_v) & ~np.isnan(fwd_v)
    corr_cur_future_vol = float(np.corrcoef(cur_v[m3], fwd_v[m3])[0, 1]) if m3.sum() > 5 else None

    # 最新一期预测 (今天)
    latest = None
    try:
        latest = rp.predict_regime()
    except Exception as e:
        latest = {"error": str(e)}

    # 特征里的 hmm
    has_hmm = any(str(c).startswith("hmm_") for c in (rp.feature_names or []))

    report = {
        "model": {
            "model_type": rp.model_type,
            "label_horizon": rp.label_horizon,
            "last_train_date": str(train_end.date()) if train_end is not None else None,
            "n_features": len(rp.feature_names or []),
            "has_hmm_features": has_hmm,
            "reported_valid_acc": rp.valid_acc,
            "reported_valid_top2": rp.valid_top2,
            "reported_valid_macro_auc": rp.valid_macro_auc,
            "ratio_map": {STATE_NAMES[i]: RATIO_MAP.get(i) for i in range(6)},
        },
        "oos": {
            "n_days": n,
            "date_from": str(dates[0].date()), "date_to": str(dates[-1].date()),
            "skipped": skip,
        },
        "accuracy": {
            "top1": round(float(top1), 4),
            "top2": round(float(top2), 4),
            "macro_auc": round(macro_auc, 4) if macro_auc else None,
            "brier": round(brier, 4),
            "log_loss": round(ll, 4) if ll is not None else None,
            "baseline_majority": round(mc, 4),
            "baseline_random": round(rand, 4),
        },
        "marginal": {
            "direction_acc_3cls": round(float(dir_acc), 4),
            "direction_macro_auc": round(dir_auc, 4) if dir_auc else None,
            "volatility_acc_2cls": round(float(vol_acc), 4),
            "volatility_auc": round(vol_auc, 4) if vol_auc is not None else None,
        },
        "per_class": {
            "states": STATE_NAMES,
            "true_dist": {STATE_NAMES[i]: int(true_classes.get(i, 0)) for i in range(6)},
            "pred_dist": {STATE_NAMES[i]: int(pred_classes.get(i, 0)) for i in range(6)},
            "never_predicted": never_predicted,
            "never_true": never_true,
            "precision": {STATE_NAMES[i]: round(float(prf[0][i]), 3) for i in range(6)},
            "recall": {STATE_NAMES[i]: round(float(prf[1][i]), 3) for i in range(6)},
            "f1": {STATE_NAMES[i]: round(float(prf[2][i]), 3) for i in range(6)},
            "confusion": cm.tolist(),
        },
        "reliability": {
            "ece": round(ece, 4),
            "calibration": cal_rows,
            "confidence_mean": round(float(conf.mean()), 4),
            "confidence_min": round(float(conf.min()), 4),
            "confidence_max": round(float(conf.max()), 4),
            "flip_rate_adjacent": round(float(flip_rate), 4),
            "n_flips": int(flips),
            "ratio_churn_mean_abs": round(float(ratio_churn), 4),
            "ratio_std": round(ratio_std, 4),
            "ratio_mean": round(float(ratios.mean()), 4),
            "ratio_min": round(float(ratios.min()), 4),
            "ratio_max": round(float(ratios.max()), 4),
        },
        "downstream_value": {
            "corr_ratio_fwd_ret": round(corr_ratio_fwdret, 4) if corr_ratio_fwdret is not None else None,
            "corr_ratio_fwd_vol": round(corr_ratio_fwdvol, 4) if corr_ratio_fwdvol is not None else None,
            "corr_Pvol_fwd_vol": round(corr_conf_vol, 4) if corr_conf_vol is not None else None,
            "spearman_Pvol_fwd_vol": round(sp_vol, 4) if sp_vol is not None else None,
            "ratio_mean_when_true_volatile": round(r_vol_true, 4) if r_vol_true is not None else None,
            "ratio_mean_when_true_steady": round(r_steady_true, 4) if r_steady_true is not None else None,
        },
        "leak_check": {
            "corr_label_vol_dim_vs_feature_vol20d": round(corr_labelvol_feat, 4) if corr_labelvol_feat is not None else None,
            "corr_current_vol_vs_future_vol": round(corr_cur_future_vol, 4) if corr_cur_future_vol is not None else None,
        },
        "latest_prediction": latest,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))

if __name__ == "__main__":
    main()
