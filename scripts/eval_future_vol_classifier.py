"""验证「未来波动率二分类」模型的真实正确率 (point-in-time, 无 leak)。

标签: 未来 horizon 日波动率 > 训练集中位数 = volatile, 否则 steady (真实未来波动率, 非 leak)
特征: _make_features (无 HMM), 去掉非平稳价量列
模型: LGBMClassifier 二分类
评估: 样本外 (train_end 之后) 正确率 + AUC
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def _future_vol_labels(index_df, horizon=10):
    """未来 horizon 日波动率 (真实标签, 用未来收益)"""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon 必须为正整数")
    rets = index_df["close"].pct_change(fill_method=None)
    fwd_rets = rets.shift(-1)
    fut_vol = fwd_rets.iloc[::-1].rolling(horizon, min_periods=horizon).std().iloc[::-1] * np.sqrt(250)
    return fut_vol


def main():
    from core.data.fetcher import DataFetcher
    from core.domain.timing.model import _make_features

    df = DataFetcher().get_index_daily("沪深300", days=365 * 10)
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= pd.Timestamp("2015-01-01")]

    horizon = 10
    feat = _make_features(df, None)
    feat["date"] = pd.to_datetime(df["date"])
    feat["fut_vol"] = _future_vol_labels(df, horizon)

    # 特征列: 数值型, 去掉非平稳价量 + date + fut_vol
    _nonstationary = {"open", "high", "low", "close", "volume", "amount"}
    feature_cols = [c for c in feat.columns if c not in {"date", "fut_vol"}
                    and c not in _nonstationary
                    and feat[c].dtype in (np.float64, np.int64, float, int)]

    feat = feat.dropna(subset=feature_cols + ["fut_vol"])

    train_end = pd.Timestamp("2026-02-14")
    train_mask = feat["date"] <= train_end
    oos_mask = feat["date"] > train_end

    # 标签: 训练集中位数二分 (避免未来阈值)
    med = feat.loc[train_mask, "fut_vol"].median()
    y = (feat["fut_vol"] > med).astype(int)

    X_train = feat.loc[train_mask, feature_cols].values
    y_train = y[train_mask].values
    X_oos = feat.loc[oos_mask, feature_cols].values
    y_oos = y[oos_mask].values

    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=200, max_depth=5, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
        min_child_samples=30, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import roc_auc_score, accuracy_score
    p_oos = model.predict_proba(X_oos)[:, 1]
    acc = accuracy_score(y_oos, p_oos > 0.5)
    auc = roc_auc_score(y_oos, p_oos)

    print(f"未来波动率二分类 (标签=未来10日波动率>中位数, 无 leak):")
    print(f"  训练集 n={len(y_train)}, 样本外 n={len(y_oos)}")
    print(f"  样本外正确率 = {acc:.3f}  AUC = {auc:.3f}")
    print()
    print(f"  特征数 = {len(feature_cols)} (无 HMM, 无非平稳价量)")
    # 特征重要性 top5
    imp = np.argsort(model.feature_importances_)[::-1][:5]
    print("  Top5 特征:")
    for i in imp:
        print(f"    {feature_cols[i]:20s} {model.feature_importances_[i]:.0f}")


if __name__ == "__main__":
    main()
