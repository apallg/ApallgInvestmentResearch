"""
验证 LSTM 能否补充树模型 (LGB) 的短时序能力。

三个判据 (融合前先跑, 决定要不要动 train_dual_model 的融合逻辑):
  1. LSTM 样本外(test) IC 是否 > 0          — 入场券, 不过线融合免谈
  2. spearman(LSTM, LGB) 预测相关性 < 0.7    — >0.7 融合无增量, 稀释
  3. LSTM 对 LGB 残差的 IC > 0.005           — 真正的"补" (树解释不了的部分)

LSTM 和 LGB 共用同一批白盒因子 + 同一 DatasetH, 只有模型结构不同 (同源对照)。
若判据 3 不过, 说明增量不在"换模型结构", 而在"换信息源" (原始价量序列)。

用法:
  python scripts/verify_lstm_fusion.py --n_epochs 40 --gpu 0
  python scripts/verify_lstm_fusion.py --direction trend_momentum --label_horizon 10
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

LOG_PATH = "logs/verify_lstm_fusion.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def daily_ic(pred, label):
    """逐日截面 IC (pearson). 输入 Series, MultiIndex=(datetime, instrument)."""
    df = pd.DataFrame({"p": pred, "l": label}).dropna()
    ic = df.groupby("datetime").apply(lambda g: g["p"].corr(g["l"]))
    return ic.dropna()


def residual_ic(pred_target, pred_base, label):
    """逐日截面: 用 base 预测回归 target 预测, 残差对 label 的 IC。

    残差 = target 中与 base 正交的部分, 残差 IC 才是 base 解释不了的新信息量。
    """
    df = pd.DataFrame({"t": pred_target, "b": pred_base, "l": label}).dropna()
    rows = {}
    for dt, g in df.groupby("datetime"):
        if len(g) < 10:
            continue
        b = g["b"].values.astype(float)
        t = g["t"].values.astype(float)
        l = g["l"].values.astype(float)
        A = np.vstack([b, np.ones(len(b))]).T
        coef, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        resid = t - (coef[0] * b + coef[1])
        if resid.std() > 0 and l.std() > 0:
            rows[dt] = float(np.corrcoef(resid, l)[0, 1])
    return pd.Series(rows).dropna()


def yearly_report(ic, name):
    idx = pd.DatetimeIndex(ic.index)
    parts = [f"{name} IC mean={ic.mean():+.4f}"]
    for yr, g in ic.groupby(idx.year):
        std = g.std()
        icir = g.mean() / std if std and std > 0 else float("nan")
        parts.append(f"{yr}: IC={g.mean():+.4f} ICIR={icir:+.2f} (n={len(g)})")
    return "  ".join(parts)


def to_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="trend_momentum")
    ap.add_argument("--instruments", default="a500")
    ap.add_argument("--label_horizon", type=int, default=10)
    ap.add_argument("--n_epochs", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--hidden_size", type=int, default=32)
    args = ap.parse_args()

    log("=" * 70)
    log(f"LSTM 融合验证 | direction={args.direction} instruments={args.instruments} "
        f"label_horizon={args.label_horizon} n_epochs={args.n_epochs}")

    # ── 初始化 qlib + 读因子 ──
    from core.adapters.qlib.config import init_qlib
    init_qlib()
    from core.domain.factors.registry import get_pool
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    from core.adapters.qlib.models import QlibModelManager

    factor_keys = get_pool().get_curated_factors(args.direction)
    valid_keys = [k for k in factor_keys if resolve_factor_expression(k)[0] is not None]
    d_feat = len(valid_keys)
    log(f"因子: {len(factor_keys)} 精选 / {d_feat} 有效 -> d_feat={d_feat}")
    if d_feat == 0:
        log("无有效因子, 退出")
        return

    train = ("2022-01-01", "2024-12-31")
    valid = ("2025-01-01", "2025-06-30")
    test = ("2025-07-01", "2026-06-30")

    manager = QlibModelManager()

    # ── LGB 基线 (树模型) ──
    try:
        opt = json.load(open("data/optuna_params/trend_momentum.json", encoding="utf-8"))
        lgb_params = {**opt["lgb_best_params"], "early_stopping_rounds": 50, "verbose_eval": -1}
    except Exception:
        lgb_params = None
        log("未找到 optuna 参数, 用默认 LGB 参数")

    log(">>> 训练 LGB 基线...")
    t0 = time.time()
    res_lgb = manager.train(
        model_name="lightgbm", factor_set="WhiteBoxHandler", instruments=args.instruments,
        train_period=train, valid_period=valid, test_period=test,
        model_params=lgb_params, label_horizon=args.label_horizon,
        custom_factor_keys=valid_keys, label_norm="none",
        enable_cross_sectional=False, enable_winsorize=True,
    )
    pred_lgb = to_series(res_lgb["predictions"])
    label = to_series(res_lgb["label"])
    log(f"LGB 完成 ({time.time()-t0:.0f}s), 官方 test IC={res_lgb.get('ic_mean')}")

    # ── LSTM (短时序) ──
    import torch
    has_gpu = torch.cuda.is_available()
    gpu = args.gpu if has_gpu else -1  # -1 强制 CPU
    lstm_params = {
        "d_feat": d_feat, "n_epochs": args.n_epochs, "batch_size": 2000,
        "hidden_size": args.hidden_size, "num_layers": 2, "dropout": 0.1,
        "lr": 0.001, "GPU": gpu, "early_stop": 10, "seed": 42,
    }
    log(f">>> 训练 LSTM (d_feat={d_feat}, epochs={args.n_epochs}, "
        f"{'GPU' if has_gpu else 'CPU(慢, 建议装 CUDA 或减 epochs)'})...")
    t0 = time.time()
    res_lstm = manager.train(
        model_name="lstm", factor_set="WhiteBoxHandler", instruments=args.instruments,
        train_period=train, valid_period=valid, test_period=test,
        model_params=lstm_params, label_horizon=args.label_horizon,
        custom_factor_keys=valid_keys, label_norm="none",
        enable_cross_sectional=False, enable_winsorize=True,
    )
    pred_lstm = to_series(res_lstm["predictions"])
    log(f"LSTM 完成 ({time.time()-t0:.0f}s), 官方 test IC={res_lstm.get('ic_mean')}")

    # ── 对齐 index ──
    common = pred_lgb.index.intersection(pred_lstm.index).intersection(label.index)
    pl = pred_lgb.reindex(common).astype(float)
    pn = pred_lstm.reindex(common).astype(float)
    lb = label.reindex(common).astype(float)
    log(f"对齐后样本 {len(common)} (lgb={len(pred_lgb)} lstm={len(pred_lstm)} label={len(label)})")
    if len(common) < 100:
        log("样本不足, index 未对齐, 退出")
        return

    # ── 判据 1: 独立 IC ──
    ic_lgb = daily_ic(pl, lb)
    ic_lstm = daily_ic(pn, lb)
    log("[判据1] 独立样本外 IC:")
    log("  " + yearly_report(ic_lgb, "LGB"))
    log("  " + yearly_report(ic_lstm, "LSTM"))

    # ── 判据 2: 预测相关性 ──
    from scipy.stats import spearmanr
    corr, _ = spearmanr(pl.values, pn.values)
    log(f"[判据2] spearman(LSTM, LGB) = {corr:.4f}  (阈值 0.7, >0.7 融合无增量)")

    # ── 判据 3: 残差 IC (LSTM 中与 LGB 正交部分的预测力) ──
    ric = residual_ic(pn, pl, lb)
    log("[判据3] LSTM 对 LGB 残差 IC:")
    log("  " + yearly_report(ric, "残差"))
    log(f"  残差 IC mean={ric.mean():+.4f}  (阈值 +0.005, <=0 说明无新信息)")

    # ── 等权融合 IC (无选择偏差, 粗略参考) ──
    fused_eq = daily_ic((pl + pn) / 2.0, lb)
    log(f"[参考] 等权融合 IC mean={fused_eq.mean():+.4f}  (LGB={ic_lgb.mean():+.4f} LSTM={ic_lstm.mean():+.4f})")

    # ── 结论 ──
    log("-" * 70)
    reasons = []
    if ic_lstm.mean() <= 0:
        reasons.append(f"LSTM 无样本外 alpha (IC={ic_lstm.mean():+.4f} <= 0)")
    if corr > 0.7:
        reasons.append(f"预测高度相关 (corr={corr:.4f} > 0.7)")
    if ric.mean() < 0.005:
        reasons.append(f"残差 IC 过小 ({ric.mean():+.4f} < 0.005), 无新信息")
    if reasons:
        log("结论: 不建议融合 LSTM —— " + "; ".join(reasons))
    else:
        log("结论: 三项判据通过, 值得尝试分数层融合 (建议 LSTM 份额 20-30%)")
    log("=" * 70)


if __name__ == "__main__":
    main()
