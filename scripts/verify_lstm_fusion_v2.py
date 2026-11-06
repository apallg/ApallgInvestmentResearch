"""
第 2 步: 更长验证期 + valid 段独立算融合权重, 验证 LSTM 融合是否真有效。

与 v1 的差异:
  1. 训练期往前扩一年 (2020-2023, 4年), test 段拉长到 1.5 年 (2025-01 ~ 2026-06)
  2. valid 段 (2024 全年) 独立算 LGB/LSTM 的 IC 作为融合权重, 避免 test 选择偏差
  3. 对比: 单 LGB / 单 LSTM / 等权融合 / valid-IC 加权融合 的 test 段 IC + 分年 ICIR

用法: python scripts/verify_lstm_fusion_v2.py --n_epochs 40 --gpu 0
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

LOG_PATH = "logs/verify_lstm_fusion_v2.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def daily_ic(pred, label):
    df = pd.DataFrame({"p": pred, "l": label}).dropna()
    return df.groupby("datetime").apply(lambda g: g["p"].corr(g["l"])).dropna()


def yearly(ic):
    idx = pd.DatetimeIndex(ic.index)
    parts = []
    for yr, g in ic.groupby(idx.year):
        std = g.std()
        icir = g.mean() / std if std and std > 0 else float("nan")
        parts.append(f"{yr}: {g.mean():+.4f} ({icir:+.2f}, n={len(g)})")
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
    log(f"LSTM 融合验证 v2 | direction={args.direction} label_horizon={args.label_horizon} "
        f"n_epochs={args.n_epochs}")

    from core.adapters.qlib.config import init_qlib
    init_qlib()
    from core.domain.factors.registry import get_pool
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    from core.adapters.qlib.models import QlibModelManager, _ic_fusion_weights
    from core.adapters.qlib.config import build_dataset_cached
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP

    factor_keys = get_pool().get_curated_factors(args.direction)
    valid_keys = [k for k in factor_keys if resolve_factor_expression(k)[0] is not None]
    d_feat = len(valid_keys)
    log(f"因子: {len(factor_keys)} 精选 / {d_feat} 有效 -> d_feat={d_feat}")

    train = ("2020-01-01", "2023-12-31")   # 4年
    valid = ("2024-01-01", "2024-12-31")   # 1年, 独立算权重
    test = ("2025-01-01", "2026-06-30")    # 1.5年

    manager = QlibModelManager()

    # ── 构建共享 dataset (LGB 与 LSTM 的 dataset 配置完全一致) ──
    lgb_cfg = manager._build_task_config(
        "lightgbm", "WhiteBoxHandler", args.instruments,
        train, valid, test,
        model_params=None, label_horizon=args.label_horizon,
        custom_factor_keys=valid_keys, label_norm="none",
        enable_cross_sectional=False, enable_winsorize=True,
    )
    log("构建 DatasetH (首次 2-5 分钟)...")
    t0 = time.time()
    dataset = build_dataset_cached(lgb_cfg["dataset"])
    log(f"DatasetH 就绪 ({time.time()-t0:.0f}s)")

    # ── 训练 LGB ──
    try:
        opt = json.load(open("data/optuna_params/trend_momentum.json", encoding="utf-8"))
        lgb_params = {**opt["lgb_best_params"], "early_stopping_rounds": 50, "verbose_eval": -1}
    except Exception:
        lgb_params = None
    lgb_model_cfg = dict(lgb_cfg["model"])
    if lgb_params:
        lgb_model_cfg["kwargs"] = {**lgb_model_cfg["kwargs"], **lgb_params}
    log("训练 LGB...")
    t0 = time.time()
    lgb_model = init_instance_by_config(lgb_model_cfg)
    lgb_model.fit(dataset)
    pred_lgb_valid = to_series(lgb_model.predict(dataset, segment="valid"))
    pred_lgb_test = to_series(lgb_model.predict(dataset, segment="test"))
    log(f"LGB 完成 ({time.time()-t0:.0f}s)")

    # ── 训练 LSTM ──
    import torch
    has_gpu = torch.cuda.is_available()
    gpu = args.gpu if has_gpu else -1
    lstm_cfg = manager._build_task_config(
        "lstm", "WhiteBoxHandler", args.instruments,
        train, valid, test,
        model_params={"d_feat": d_feat, "n_epochs": args.n_epochs, "batch_size": 2000,
                      "hidden_size": args.hidden_size, "num_layers": 2, "dropout": 0.1,
                      "lr": 0.001, "GPU": gpu, "early_stop": 10, "seed": 42},
        label_horizon=args.label_horizon, custom_factor_keys=valid_keys,
        label_norm="none", enable_cross_sectional=False, enable_winsorize=True,
    )
    log(f"训练 LSTM (d_feat={d_feat}, {'GPU' if has_gpu else 'CPU'})...")
    t0 = time.time()
    lstm_model = init_instance_by_config(lstm_cfg["model"])
    lstm_model.fit(dataset)
    pred_lstm_valid = to_series(lstm_model.predict(dataset, segment="valid"))
    pred_lstm_test = to_series(lstm_model.predict(dataset, segment="test"))
    log(f"LSTM 完成 ({time.time()-t0:.0f}s)")

    # ── label (valid + test) ──
    label_valid = to_series(dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R))
    label_test = to_series(dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R))

    # ── valid 段 IC → 融合权重 ──
    ic_lgb_valid = daily_ic(pred_lgb_valid, label_valid)
    ic_lstm_valid = daily_ic(pred_lstm_valid, label_valid)
    lgb_valid_mean = float(ic_lgb_valid.mean())
    lstm_valid_mean = float(ic_lstm_valid.mean())
    log(f"valid IC: LGB={lgb_valid_mean:+.4f}  LSTM={lstm_valid_mean:+.4f}")
    w_lgb, w_lstm = _ic_fusion_weights(lgb_valid_mean, lstm_valid_mean)
    log(f"valid-IC 加权: w_lgb={w_lgb:.3f} w_lstm={w_lstm:.3f}")

    # ── test 段评估 ──
    common = pred_lgb_test.index.intersection(pred_lstm_test.index).intersection(label_test.index)
    pl = pred_lgb_test.reindex(common).astype(float)
    pn = pred_lstm_test.reindex(common).astype(float)
    lb = label_test.reindex(common).astype(float)
    log(f"test 段对齐样本: {len(common)}")

    ic_lgb = daily_ic(pl, lb)
    ic_lstm = daily_ic(pn, lb)
    ic_eq = daily_ic((pl + pn) / 2.0, lb)
    ic_w = daily_ic(w_lgb * pl + w_lstm * pn, lb)

    log("-" * 70)
    log(f"[单 LGB ]  test IC={ic_lgb.mean():+.4f}  " + yearly(ic_lgb))
    log(f"[单 LSTM]  test IC={ic_lstm.mean():+.4f}  " + yearly(ic_lstm))
    log(f"[等权融合] test IC={ic_eq.mean():+.4f}  " + yearly(ic_eq))
    log(f"[valid加权] test IC={ic_w.mean():+.4f}  " + yearly(ic_w))
    log("-" * 70)
    gain = ic_eq.mean() - ic_lgb.mean()
    log(f"等权融合 vs 单LGB: {gain:+.4f} ({gain/abs(ic_lgb.mean())*100:+.1f}%)")
    gain_w = ic_w.mean() - ic_lgb.mean()
    log(f"valid加权 vs 单LGB: {gain_w:+.4f} ({gain_w/abs(ic_lgb.mean())*100:+.1f}%)")
    log("=" * 70)


if __name__ == "__main__":
    main()
