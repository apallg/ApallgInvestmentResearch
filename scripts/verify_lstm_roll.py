"""
滚动窗口 (walk-forward) 验证: 公平比较 LGB / LSTM / 融合的窗口敏感性。

背景: 单点对比 (v1/v2) 不足以定论, 因为树模型也有窗口敏感 (LGB 2026H1 IC 0.0017)。
滚动窗口下, 每个窗口独立训练 LGB + LSTM, valid 算权重, test 评估, 汇总 IC 序列。

回答的问题:
  1. LGB 和 LSTM 的 IC 均值/方差各是多少 (谁更稳, 谁更强)
  2. 融合 (等权 + valid加权) 相对单 LGB 的增益, 在各窗口是否稳定为正
  3. 在 LGB 的弱窗口, LSTM 能否兜底 (削峰填谷 / 减回撤)

用法: python scripts/verify_lstm_roll.py --n_epochs 40 --gpu 0
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

LOG_PATH = "logs/verify_lstm_roll.log"

# 3 个滚动窗口: train 3年 / valid 1年 / test 1年, 各向后滑一年
WINDOWS = [
    (("2019-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")),
    (("2020-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")),
    (("2021-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")),
]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def daily_ic(pred, label):
    df = pd.DataFrame({"p": pred, "l": label}).dropna()
    return df.groupby("datetime").apply(lambda g: g["p"].corr(g["l"])).dropna()


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
    log(f"LSTM 滚动窗口验证 | direction={args.direction} label_horizon={args.label_horizon}")

    from core.adapters.qlib.config import init_qlib
    init_qlib()
    from core.domain.factors.registry import get_pool
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    from core.adapters.qlib.models import QlibModelManager, _ic_fusion_weights
    from core.adapters.qlib.config import build_dataset_cached
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP
    import torch
    has_gpu = torch.cuda.is_available()
    gpu = args.gpu if has_gpu else -1

    factor_keys = get_pool().get_curated_factors(args.direction)
    valid_keys = [k for k in factor_keys if resolve_factor_expression(k)[0] is not None]
    d_feat = len(valid_keys)
    log(f"因子 {d_feat} 有效, d_feat={d_feat}, GPU={'yes' if has_gpu else 'no'}")

    try:
        opt = json.load(open("data/optuna_params/trend_momentum.json", encoding="utf-8"))
        lgb_params = {**opt["lgb_best_params"], "early_stopping_rounds": 50, "verbose_eval": -1}
    except Exception:
        lgb_params = None

    manager = QlibModelManager()
    rows = []

    for wi, (train, valid, test) in enumerate(WINDOWS):
        log("-" * 70)
        log(f"窗口 {wi+1}: train {train[0]}~{train[1]}  valid {valid[0]}~{valid[1]}  test {test[0]}~{test[1]}")

        # 共享 dataset
        base_cfg = manager._build_task_config(
            "lightgbm", "WhiteBoxHandler", args.instruments,
            train, valid, test,
            model_params=None, label_horizon=args.label_horizon,
            custom_factor_keys=valid_keys, label_norm="none",
            enable_cross_sectional=False, enable_winsorize=True,
        )
        dataset = build_dataset_cached(base_cfg["dataset"])

        # LGB
        lgb_cfg = dict(base_cfg["model"])
        if lgb_params:
            lgb_cfg["kwargs"] = {**lgb_cfg["kwargs"], **lgb_params}
        lgb_model = init_instance_by_config(lgb_cfg)
        lgb_model.fit(dataset)
        pl_valid = to_series(lgb_model.predict(dataset, segment="valid"))
        pl_test = to_series(lgb_model.predict(dataset, segment="test"))

        # LSTM
        lstm_cfg = manager._build_task_config(
            "lstm", "WhiteBoxHandler", args.instruments,
            train, valid, test,
            model_params={"d_feat": d_feat, "n_epochs": args.n_epochs, "batch_size": 2000,
                          "hidden_size": args.hidden_size, "num_layers": 2, "dropout": 0.1,
                          "lr": 0.001, "GPU": gpu, "early_stop": 10, "seed": 42},
            label_horizon=args.label_horizon, custom_factor_keys=valid_keys,
            label_norm="none", enable_cross_sectional=False, enable_winsorize=True,
        )
        lstm_model = init_instance_by_config(lstm_cfg["model"])
        lstm_model.fit(dataset)
        pn_valid = to_series(lstm_model.predict(dataset, segment="valid"))
        pn_test = to_series(lstm_model.predict(dataset, segment="test"))

        label_valid = to_series(dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R))
        label_test = to_series(dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R))

        # valid IC → 权重
        ic_l_valid = daily_ic(pl_valid, label_valid)
        ic_n_valid = daily_ic(pn_valid, label_valid)
        l_valid, n_valid = float(ic_l_valid.mean()), float(ic_n_valid.mean())
        w_l, w_n = _ic_fusion_weights(l_valid, n_valid)

        # test IC
        common = pl_test.index.intersection(pn_test.index).intersection(label_test.index)
        pl = pl_test.reindex(common).astype(float)
        pn = pn_test.reindex(common).astype(float)
        lb = label_test.reindex(common).astype(float)
        ic_l = daily_ic(pl, lb)
        ic_n = daily_ic(pn, lb)
        ic_eq = daily_ic((pl + pn) / 2.0, lb)
        ic_w = daily_ic(w_l * pl + w_n * pn, lb)

        row = {
            "window": wi + 1, "test": test[1][:4],
            "lgb_valid": l_valid, "lstm_valid": n_valid, "w_lgb": w_l, "w_lstm": w_n,
            "lgb_test": float(ic_l.mean()), "lstm_test": float(ic_n.mean()),
            "eq_test": float(ic_eq.mean()), "w_test": float(ic_w.mean()),
        }
        rows.append(row)
        log(f"  valid IC: LGB={l_valid:+.4f} LSTM={n_valid:+.4f} -> w={w_l:.2f}/{w_n:.2f}")
        log(f"  test  IC: LGB={row['lgb_test']:+.4f} LSTM={row['lstm_test']:+.4f} "
            f"等权={row['eq_test']:+.4f} valid加权={row['w_test']:+.4f}")

    # ── 汇总 ──
    log("=" * 70)
    log("汇总 (test 段 IC):")
    df = pd.DataFrame(rows)
    for col, name in [("lgb_test", "LGB"), ("lstm_test", "LSTM"),
                      ("eq_test", "等权融合"), ("w_test", "valid加权")]:
        s = df[col]
        log(f"  {name:10s}: mean={s.mean():+.4f} std={s.std():.4f} "
            f"min={s.min():+.4f} max={s.max():+.4f}")
    gain_eq = (df["eq_test"] - df["lgb_test"])
    gain_w = (df["w_test"] - df["lgb_test"])
    log(f"  等权融合增益: mean={gain_eq.mean():+.4f} std={gain_eq.std():.4f} "
        f"(负窗口 {sum(gain_eq < 0)}/{len(gain_eq)})")
    log(f"  valid加权增益: mean={gain_w.mean():+.4f} std={gain_w.std():.4f} "
        f"(负窗口 {sum(gain_w < 0)}/{len(gain_w)})")
    log(f"  LGB 弱窗口 (<0.01) 兜底: 见逐窗口 LSTM test IC")
    log("=" * 70)


if __name__ == "__main__":
    main()
