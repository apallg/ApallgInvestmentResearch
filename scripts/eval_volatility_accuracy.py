"""评估波动归并正确率: 6 状态归并成 2 波动 (volatile/steady)。

波动归并:
  volatile = volatile_up(0) + volatile_down(1) + volatile_flat(2)
  steady   = steady_up(3) + steady_down(4) + steady_flat(5)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

STATE_TO_VOL = [0, 0, 0, 1, 1, 1]  # 0=volatile, 1=steady
VOL_NAMES = ["volatile", "steady"]


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

    P_vol = np.zeros((len(P6), 2))
    P_vol[:, 0] = P6[:, :3].sum(axis=1)  # volatile 概率
    P_vol[:, 1] = P6[:, 3:].sum(axis=1)  # steady 概率
    T_vol = np.array([STATE_TO_VOL[t] for t in T6])

    top1_vol = (P_vol.argmax(1) == T_vol).mean()

    cm = np.zeros((2, 2), dtype=int)
    for p, t in zip(P_vol.argmax(1), T_vol):
        cm[t, p] += 1

    print(f"样本外 n={len(T6)}")
    print(f"  6 状态 Top1:  {(P6.argmax(1) == T6).mean():.3f}  (随机 0.167)")
    print(f"  波动 Top1:    {top1_vol:.3f}  (随机 0.500)")
    print()
    print("波动混淆矩阵 (行=真实, 列=预测):")
    print("          " + "".join(f"{VOL_NAMES[j]:>10}" for j in range(2)))
    for i in range(2):
        print(f"{VOL_NAMES[i]:10s}" + "".join(f"{cm[i, j]:>10}" for j in range(2)))
    print()
    for i in range(2):
        tot = cm[i].sum()
        if tot:
            print(f"  {VOL_NAMES[i]:9s} recall={cm[i,i]/tot:.3f}  (n={tot})")
    # 波动概率的判别度 (AUC 用概率)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(T_vol, P_vol[:, 1])
        print(f"\n  波动分类 AUC = {auc:.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
