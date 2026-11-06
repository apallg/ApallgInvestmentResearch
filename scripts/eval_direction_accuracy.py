"""评估方向正确率: 6 状态归并成 3 方向 (up/down/flat), 对比 6 状态 Top1。

方向归并:
  up   = volatile_up(0) + steady_up(3)
  down = volatile_down(1) + steady_down(4)
  flat = volatile_flat(2) + steady_flat(5)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DIR_UP, DIR_DOWN, DIR_FLAT = 0, 1, 2
STATE_TO_DIR = [DIR_UP, DIR_DOWN, DIR_FLAT, DIR_UP, DIR_DOWN, DIR_FLAT]
DIR_NAMES = ["up", "down", "flat"]


def main():
    from core.data.fetcher import DataFetcher
    from core.domain.regime.predictor import get_regime_predictor, _make_regime_labels

    df = DataFetcher().get_index_daily("沪深300", days=365 * 2)
    df = df.sort_values("date").reset_index(drop=True)

    rp = get_regime_predictor()
    train_end = pd.Timestamp(rp.last_train_date)
    labels = _make_regime_labels(df, rp.label_horizon)

    P6, T6 = [], []
    for i in range(len(df)):
        if df["date"].iloc[i] <= train_end or i < 60 or i >= len(df) - rp.label_horizon:
            continue
        probs = rp.predict_from_df(df.iloc[:i + 1])
        if probs is None:
            continue
        P6.append(probs)
        T6.append(int(labels[i]))

    P6 = np.array(P6)
    T6 = np.array(T6)

    # 6 状态 top1
    top1_6 = (P6.argmax(1) == T6).mean()

    # 方向归并: 概率求和
    P_dir = np.zeros((len(P6), 3))
    for d in range(3):
        cols = [s for s in range(6) if STATE_TO_DIR[s] == d]
        P_dir[:, d] = P6[:, cols].sum(axis=1)
    T_dir = np.array([STATE_TO_DIR[t] for t in T6])

    top1_dir = (P_dir.argmax(1) == T_dir).mean()

    print(f"样本外 n={len(T6)}")
    print(f"  6 状态 Top1: {top1_6:.3f}  (随机 1/6=0.167)")
    print(f"  方向 Top1:   {top1_dir:.3f}  (随机 1/3=0.333)")
    print()

    # 方向混淆矩阵
    cm = np.zeros((3, 3), dtype=int)
    for p, t in zip(P_dir.argmax(1), T_dir):
        cm[t, p] += 1
    print("方向混淆矩阵 (行=真实, 列=预测):")
    print("      " + "".join(f"{DIR_NAMES[j]:>7}" for j in range(3)))
    for i in range(3):
        print(f"{DIR_NAMES[i]:5s}" + "".join(f"{cm[i, j]:>7}" for j in range(3)))
    # 各方向 recall
    print()
    for i in range(3):
        tot = cm[i].sum()
        if tot:
            print(f"  {DIR_NAMES[i]:5s} recall={cm[i,i]/tot:.3f}  (n={tot})")


if __name__ == "__main__":
    main()
