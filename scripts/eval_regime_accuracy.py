"""评估 RegimePredictor 市场状态的样本外正确率 (point-in-time, 无前视)。

用法: python scripts/eval_regime_accuracy.py [采样步长=5]

对模型 train_end 之后的每个采样日, 用截至当日的切片 predict_from_df (point-in-time),
与 post-hoc 标签 (未来 horizon 日收益×波动率) 对比, 输出 Top1/Top2 准确率 + 混淆矩阵。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def main():
    from core.data.fetcher import DataFetcher
    from core.domain.regime.predictor import get_regime_predictor, _make_regime_labels, STATE_NAMES

    step = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    fetcher = DataFetcher()
    df = fetcher.get_index_daily("沪深300", days=365 * 2)
    if df is None or df.empty:
        print("无法获取指数数据")
        return
    df = df.sort_values("date").reset_index(drop=True)

    rp = get_regime_predictor()
    if rp is None or not rp._is_loaded():
        print("RegimePredictor 未加载")
        return

    train_end = pd.Timestamp(rp.last_train_date) if rp.last_train_date else None
    print(f"模型 model_type={rp.model_type} label_horizon={rp.label_horizon} train_end={rp.last_train_date}")
    print(f"样本外区间: > {rp.last_train_date}")

    labels = _make_regime_labels(df, rp.label_horizon)

    preds, truths = [], []
    n_skip = 0
    for i in range(len(df)):
        d = df["date"].iloc[i]
        if train_end is not None and d <= train_end:
            continue
        if i < 60:
            continue
        if i >= len(df) - rp.label_horizon:
            continue  # 标签需要未来 horizon 日
        if (i - 60) % step != 0:
            continue
        probs = rp.predict_from_df(df.iloc[:i + 1])
        if probs is None:
            n_skip += 1
            continue
        preds.append(probs)
        truths.append(int(labels[i]))

    if not truths:
        print("无样本外样本 (模型 train_end 已到最新?)")
        return

    P = np.array(preds)
    T = np.array(truths)
    top1 = (P.argmax(1) == T).mean()
    top2 = (np.any(np.argsort(P, 1)[:, -2:] == T[:, None], 1)).mean()

    print(f"\n样本外 {len(T)} 个采样日 (步长{step}, 跳过{n_skip}):")
    print(f"  Top1 准确率: {top1:.3f}   (随机猜 1/6 = {1/6:.3f})")
    print(f"  Top2 准确率: {top2:.3f}")

    cm = np.zeros((6, 6), dtype=int)
    for p, t in zip(P.argmax(1), T):
        cm[t, p] += 1
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print("          " + "".join(f"{s[:6]:>8}" for s in STATE_NAMES))
    for i, s in enumerate(STATE_NAMES):
        print(f"{s:10s}" + "".join(f"{cm[i, j]:>8}" for j in range(6)))


if __name__ == "__main__":
    main()
