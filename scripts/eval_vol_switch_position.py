
# -*- coding: utf-8 -*-
"""评测当前部署的波动率切换 + 仓位控制 (point-in-time, 无前视)"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from core.data.fetcher import DataFetcher
from core.domain.regime.vol_predictor import get_vol_classifier, _future_vol_labels
from core.domain.regime.predictor import get_regime_predictor, _soft_position_ratio, STATE_NAMES, RATIO_MAP

ANCHOR_VOL = {"volatile_up": 0.225, "volatile_down": 0.209, "volatile_flat": 0.228,
              "steady_up": 0.112, "steady_down": 0.121, "steady_flat": 0.117}
MID_LO, MID_HI = 0.36, 0.59

def main():
    vol = get_vol_classifier(10)
    rp = get_regime_predictor()
    train_end = pd.Timestamp(vol.last_train_date) if vol.last_train_date else None
    vol_median = float(vol.vol_median)

    fetcher = DataFetcher()
    df = fetcher.get_index_daily("沪深300", days=365 * 2)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    close = df["close"].astype(float)
    daily_ret = close.pct_change(fill_method=None).values
    fut_vol = _future_vol_labels(df, 10).values
    fwd_ret10 = (close.shift(-10) / close - 1).values

    # ---- 逐日 point-in-time: P_vol + 6态概率 ----
    rows = []
    for i in range(len(df)):
        d = df["date"].iloc[i]
        if train_end is not None and d <= train_end:
            if d < pd.Timestamp("2025-01-01"):
                continue
            if (i % 2) != 0:
                continue
        if i < 60 or i >= n - 10:
            continue
        sl = df.iloc[:i + 1]
        p_vol = vol.predict_proba(sl)
        probs = rp.predict_from_df(sl)
        rows.append({"i": i, "date": d, "p_vol": p_vol,
                     "probs": probs, "fut_vol": fut_vol[i],
                     "fwd_ret10": fwd_ret10[i], "dret": daily_ret[i]})
    R = [r for r in rows if r["p_vol"] is not None and r["probs"] is not None]
    oos = [r for r in R if train_end is None or r["date"] > train_end]
    train_tail = [r for r in R if r["date"] <= train_end]

    def ratio_of(probs):
        pos_base = _soft_position_ratio(probs)
        impl = sum(float(probs[i]) * ANCHOR_VOL[s] for i, s in enumerate(STATE_NAMES))
        scalar = 1.0 / max(1.0, impl / 0.17)
        scalar = max(0.70, min(1.0, scalar))
        return max(0.15, min(1.00, pos_base * scalar))

    Pv = np.array([r["p_vol"] for r in oos]); FV = np.array([r["fut_vol"] for r in oos])
    FR = np.array([r["fwd_ret10"] for r in oos]); DR = np.array([r["dret"] for r in oos])
    y = (FV > vol_median).astype(int)  # 部署模型自己的标签口径
    m = ~np.isnan(FV)

    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
    acc = accuracy_score(y[m], (Pv[m] > 0.5).astype(int))
    auc = roc_auc_score(y[m], Pv[m])
    bacc = balanced_accuracy_score(y[m], (Pv[m] > 0.5).astype(int))
    # 阈值敏感性: 0.3/0.5/0.7
    sens = {str(t): round(float(accuracy_score(y[m], (Pv[m] > t).astype(int))), 4) for t in (0.3, 0.5, 0.7)}

    # 三档分布 + 各档未来波动/收益
    tier = np.where(Pv > MID_HI, "high", np.where(Pv < MID_LO, "low", "mid"))
    tiers = {"high": 0, "mid": 0, "low": 0}
    tier_futvol = {"high": [], "mid": [], "low": []}
    tier_fwdret = {"high": [], "mid": [], "low": []}
    tier_futvolfrac = {"high": 0, "mid": 0, "low": 0}
    for t, fv_, fr_, yy in zip(tier, FV, FR, y):
        tiers[t] += 1
        if not np.isnan(fv_):
            tier_futvol[t].append(fv_)
            tier_futvolfrac[t] += int(yy)
        if not np.isnan(fr_):
            tier_fwdret[t].append(fr_)
    tier_report = {}
    for t in ("low", "mid", "high"):
        tier_report[t] = {
            "n": tiers[t],
            "fut_vol_mean": round(float(np.mean(tier_futvol[t])), 4) if tier_futvol[t] else None,
            "fwd_ret10_mean_pct": round(float(np.mean(tier_fwdret[t]) * 100), 2) if tier_fwdret[t] else None,
            "volatile_frac": round(tier_futvolfrac[t] / max(1, tiers[t]), 3),
        }

    # 校准 (P_vol vs 实际 volatile 频率, 5 bins)
    edges = np.linspace(0, 1, 6)
    ece, cal = 0.0, []
    for b in range(5):
        mm = (Pv >= edges[b]) & (Pv < edges[b + 1])
        if b == 4:
            mm = Pv >= edges[b]
        mm = mm & m
        if mm.sum() == 0:
            continue
        p_avg = Pv[mm].mean(); frac = y[mm].mean()
        cal.append((round(edges[b], 2), round(edges[b + 1], 2), int(mm.sum()), round(p_avg, 3), round(frac, 3)))
        ece += (mm.sum() / m.sum()) * abs(p_avg - frac)

    # 日间 flip: 档位切换率 + P_vol 一阶差分
    tier_seq = tier
    tier_flip = float(np.mean(tier_seq[1:] != tier_seq[:-1]))
    pv_diff = float(np.abs(np.diff(Pv)).mean())

    # 漂移: OOS vs train_tail 的 P_vol 分布
    Pv_train = np.array([r["p_vol"] for r in train_tail])
    drift = {
        "train_tail_n": len(Pv_train),
        "train_tail_pvol_mean": round(float(Pv_train.mean()), 3),
        "train_tail_pvol_std": round(float(Pv_train.std()), 3),
        "oos_pvol_mean": round(float(Pv.mean()), 3),
        "oos_pvol_std": round(float(Pv.std()), 3),
        "oos_high_frac": round(float((Pv > MID_HI).mean()), 3),
        "oos_low_frac": round(float((Pv < MID_LO).mean()), 3),
        "train_tail_high_frac": round(float((Pv_train > MID_HI).mean()), 3),
        "train_tail_low_frac": round(float((Pv_train < MID_LO).mean()), 3),
    }

    # ---- 仓位控制 (生产公式) ----
    ratios = np.array([ratio_of(r["probs"]) for r in oos])
    ratio_binary = ratios.copy()
    ratio_cond = np.where(tier == "mid", 0.0, ratios)  # 三档: 中波 → 空仓
    out = {
        "vol_model": {
            "label_horizon": vol.label_horizon,
            "last_train_date": str(vol.last_train_date),
            "vol_median": round(float(vol_median), 5),
            "n_features": len(vol.feature_names or []),
            "reported_valid_acc": round(float(vol.valid_acc), 4),
            "reported_valid_auc": round(float(vol.valid_auc), 4),
            "reported_valid_balanced": round(float(vol.valid_balanced_accuracy), 4),
        },
        "oos": {"n": len(oos), "date_from": str(oos[0]["date"].date()), "date_to": str(oos[-1]["date"].date()),
                "volatile_frac": round(float(y[m].mean()), 3)},
        "vol_accuracy": {
            "acc_p05": round(float(acc), 4),
            "balanced_acc": round(float(bacc), 4),
            "auc": round(float(auc), 4),
            "threshold_sensitivity": sens,
            "documented_claim": "93.6% acc / 0.972 AUC (docs/regime_model_iterations.md, 临时训练口径)",
        },
        "tiers": tier_report,
        "calibration": {"ece": round(ece, 4), "bins": cal, "pvol_mean": round(float(Pv.mean()), 3),
                        "pvol_min": round(float(Pv.min()), 3), "pvol_max": round(float(Pv.max()), 3)},
        "reliability": {
            "tier_flip_rate": round(tier_flip, 4),
            "pvol_abs_diff_mean": round(pv_diff, 4),
            "drift": drift,
        },
        "position_control": {
            "formula": "clip(soft6ratio * vol_scalar, 0.15, 1.0); 三档中波→0 (use_conditional_switch 默认 True)",
            "ratio_binary": {"mean": round(float(ratios.mean()), 3), "std": round(float(ratios.std()), 3),
                             "min": round(float(ratios.min()), 3), "max": round(float(ratios.max()), 3)},
            "ratio_cond": {"mean": round(float(ratio_cond[ratio_cond > 0].mean()), 3) if (ratio_cond > 0).any() else None,
                           "zero_frac": round(float((ratio_cond == 0).mean()), 3),
                           "std": round(float(ratio_cond.std()), 3)},
            "churn_cond": round(float(np.abs(np.diff(ratio_cond)).mean()), 4),
            "churn_binary": round(float(np.abs(np.diff(ratios)).mean()), 4),
        },
    }

    # 排序力 (fwd 10d ret)
    mf = m & ~np.isnan(FR)
    corr_b = float(np.corrcoef(ratios[mf], FR[mf])[0, 1])
    corr_c = float(np.corrcoef(ratio_cond[mf], FR[mf])[0, 1])
    q = np.quantile(ratio_cond[ratio_cond > 0], [0, 1/3, 2/3, 1.0]) if (ratio_cond > 0).any() else None
    terc = None
    if q is not None:
        terc = []
        for k in range(3):
            mm2 = (ratio_cond > 0) & (FR >= q[k]) & (FR <= q[k+1]) & ~np.isnan(FR)
            terc.append(round(float(FR[mm2].mean() * 100), 2))
    out["position_value"] = {
        "corr_ratio_binary_fwd_ret": round(corr_b, 4),
        "corr_ratio_cond_fwd_ret": round(corr_c, 4),
        "ratio_cond_tercile_fwd_ret_pct": terc,
        "corr_ratio_cond_fwd_vol": round(float(np.corrcoef(ratio_cond[m], FV[m])[0, 1]), 4),
    }

    # 朴素日频策略示意 (无成本, 日频再平衡): ratio × 当日指数收益
    def strat_cum(ratio_series):
        r = ratio_series * DR
        return np.cumprod(1 + r) - 1
    cum_bh = np.cumprod(1 + DR) - 1
    cum_cond = strat_cum(ratio_cond)
    cum_bin = strat_cum(ratio_binary)
    def mdd(c):
        peak = np.maximum.accumulate(1 + c)
        return float(((1 + c) / peak - 1).min())
    out["naive_illustrative"] = {
        "note": "日频再平衡、零成本、仅指数, 示意排序力, 不代表生产净值",
        "bh_total": round(float(cum_bh[-1] * 100), 1),
        "binary_total": round(float(cum_bin[-1] * 100), 1),
        "cond_total": round(float(cum_cond[-1] * 100), 1),
        "bh_mdd": round(mdd(cum_bh) * 100, 1),
        "binary_mdd": round(mdd(cum_bin) * 100, 1),
        "cond_mdd": round(mdd(cum_cond) * 100, 1),
        "cond_zero_days": int((ratio_cond == 0).sum()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

if __name__ == "__main__":
    main()
