"""排查 RegimePredictor 样本外退化 + 系统性不预测上涨的根因。

输出:
  1. HMM 特征模型信息 (是否全历史训练 → 前视/漂移)
  2. 训练/验证/样本外 6 类标签分布 (标签是否 drift)
  3. 样本外模型概率输出均值 (是否整体偏 down/flat)
  4. tree / linear 分离的样本外 Top1 (fusion 是否拖累 / 谁有偏差)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

STATE_NAMES = ["volatile_up", "volatile_down", "volatile_flat",
               "steady_up", "steady_down", "steady_flat"]


def _dist(labels):
    if len(labels) == 0:
        return {}
    u, c = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def main():
    from core.data.fetcher import DataFetcher
    from core.domain.regime.predictor import get_regime_predictor, _make_regime_labels

    df = DataFetcher().get_index_daily("沪深300", days=365 * 10)
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= pd.Timestamp("2015-01-01")]

    rp = get_regime_predictor()
    print(f"模型 model_type={rp.model_type} label_horizon={rp.label_horizon} train_end={rp.last_train_date}\n")

    # ── 1. HMM 特征模型 ──
    try:
        from core.domain.regime.classifier import HMMRegimeClassifier
        hmm = HMMRegimeClassifier.load()
        if hmm is not None:
            print(f"[1] HMM 特征模型: n_states={hmm.n_states} train_date={hmm._train_date}")
            print(f"    (train_date 若晚于 {rp.last_train_date}, 则 hmm_* 特征在训练时已泄漏未来 → train/serve skew)")
        else:
            print("[1] HMM 特征模型: 未持久化 (每次现场训练, 特征不稳定)")
    except Exception as e:
        print(f"[1] HMM 检查失败: {e}")
    print()

    # ── 2. 标签分布 ──
    labels = _make_regime_labels(df, rp.label_horizon)
    d = df["date"]
    train_end = pd.Timestamp(rp.last_train_date)
    train_mask = (d >= pd.Timestamp("2015-01-01")) & (d <= train_end)
    oos_mask = d > train_end
    # 训练集 = train_end 之前 6 年 (与动态训练区间对齐)
    train_mask6 = (d >= (train_end - pd.Timedelta(days=365 * 6))) & (d <= train_end)
    print("[2] 标签分布 (真实状态占比):")
    for name, m in [("训练集(6年)", train_mask6), ("样本外", oos_mask)]:
        dist = _dist(labels[m.values])
        total = sum(dist.values())
        parts = "  ".join(f"{STATE_NAMES[i][:8]}={dist.get(i,0)/total:.2f}" if total else f"{STATE_NAMES[i][:8]}=0" for i in range(6))
        print(f"    {name:10s} n={total:4d}  {parts}")
    print()

    # ── 3 & 4. 样本外概率输出 + tree/linear 分离 ──
    oos_idx = [i for i in range(len(df)) if d.iloc[i] > train_end and 60 <= i < len(df) - rp.label_horizon]
    P_all, T_all = [], []
    for i in oos_idx:
        probs = rp.predict_from_df(df.iloc[:i + 1])
        if probs is None:
            continue
        P_all.append(probs)
        T_all.append(int(labels[i]))
    P_all = np.array(P_all)
    T_all = np.array(T_all)
    if len(P_all) == 0:
        print("[3/4] 无样本外样本")
        return

    print(f"[3] 样本外模型概率输出均值 (n={len(P_all)}):")
    for i, s in enumerate(STATE_NAMES):
        print(f"    {s:14s} 均值={P_all[:, i].mean():.3f}  最大={P_all[:, i].max():.3f}")
    print()

    # 真实标签 argmax vs 预测 argmax 的分布
    pred_argmax = P_all.argmax(1)
    true_argmax = T_all
    print(f"    预测 argmax 分布: " + "  ".join(f"{STATE_NAMES[i][:8]}={(pred_argmax==i).sum()}" for i in range(6)))
    print(f"    真实 argmax 分布: " + "  ".join(f"{STATE_NAMES[i][:8]}={(true_argmax==i).sum()}" for i in range(6)))
    print()

    # tree vs linear 分离
    print("[4] tree / linear 分离的样本外 Top1:")
    for label, model in [("tree", rp), ("linear", rp._fusion_partner)]:
        if model is None or not model._is_loaded():
            print(f"    {label}: 未加载")
            continue
        P = []
        for i in oos_idx:
            probs = model.predict_from_df(df.iloc[:i + 1])
            if probs is not None:
                P.append(probs)
        P = np.array(P)
        if len(P) == 0:
            print(f"    {label}: 无样本")
            continue
        top1 = (P.argmax(1) == T_all[:len(P)]).mean()
        # 各状态预测计数
        cnt = {STATE_NAMES[i][:8]: int((P.argmax(1) == i).sum()) for i in range(6)}
        print(f"    {label:6s} Top1={top1:.3f}  预测分布={cnt}")


if __name__ == "__main__":
    main()
