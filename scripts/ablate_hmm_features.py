"""消融实验: 去掉 hmm_* 特征重训 tree, 对比样本外 Top1 与预测分布。

训练后评估, 最后恢复原 regime_h10.pkl (不破坏现有模型)。
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core.domain.regime.predictor import RegimePredictor, _make_regime_labels, STATE_NAMES, MODEL_DIR
from core.shared.persistence import today_date


def _resolve_dates():
    from datetime import timedelta
    end = today_date()
    valid_end = end.isoformat()
    valid_start = (end - timedelta(days=180)).isoformat()
    train_end = (end - timedelta(days=181)).isoformat()
    train_start = (end - timedelta(days=181 + 365 * 6)).isoformat()
    return train_start, train_end, valid_start, valid_end


def main():
    pkl = MODEL_DIR / "regime_h10.pkl"
    bak = MODEL_DIR / "regime_h10.pkl.bak"
    if pkl.exists():
        shutil.copy(pkl, bak)

    try:
        train_start, train_end, valid_start, valid_end = _resolve_dates()
        model = RegimePredictor(label_horizon=10, model_type="tree", use_hmm_features=False)
        result = model.fit(train_start, train_end, valid_start, valid_end)
        n_hmm = sum(1 for c in model.feature_names if c.startswith("hmm_"))
        print(f"无HMM tree 训练完成: valid_acc={result.get('valid_acc')}  特征数={len(model.feature_names)}  hmm特征数={n_hmm}\n")

        from core.data.fetcher import DataFetcher
        df = DataFetcher().get_index_daily("沪深300", days=365 * 2)
        df = df.sort_values("date").reset_index(drop=True)
        labels = _make_regime_labels(df, 10)
        train_end_ts = pd.Timestamp(train_end)

        P, T = [], []
        for i in range(len(df)):
            d = df["date"].iloc[i]
            if d <= train_end_ts or i < 60 or i >= len(df) - 10:
                continue
            probs = model.predict_from_df(df.iloc[:i + 1])
            if probs is None:
                continue
            P.append(probs)
            T.append(int(labels[i]))

        P = np.array(P)
        T = np.array(T)
        top1 = (P.argmax(1) == T).mean()
        top2 = (np.any(np.argsort(P, 1)[:, -2:] == T[:, None], 1)).mean()
        print(f"无HMM tree 样本外 (n={len(T)}): Top1={top1:.3f}  Top2={top2:.3f}")
        print("预测 argmax 分布:")
        for i, s in enumerate(STATE_NAMES):
            print(f"    {s:14s} {(P.argmax(1) == i).sum()}")
        print("\n对比: 有HMM tree 样本外 Top1=0.349 (诊断脚本 [4] 结果)")
    finally:
        if bak.exists():
            shutil.copy(bak, pkl)
            bak.unlink()
            print("\n已恢复原 regime_h10.pkl")


if __name__ == "__main__":
    main()
