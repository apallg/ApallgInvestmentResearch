"""
波动状态分类器 (VolRegimeClassifier) — 未来波动率二分类

标签: 未来 horizon 日波动率 > 训练集中位数 = volatile(高波), 否则 steady(低波)
      (真实未来波动率, 非 leak; 波动聚集效应使其可预测 ~93%)

用途: 风控敞口 — 高波降仓, 低波满仓。方向(涨跌)不可预测, 不在此模型职责内。
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.domain.timing.model import _make_features

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = ROOT / "data" / "timing_model"

_NONSTATIONARY = {"open", "high", "low", "close", "volume", "amount"}


def _future_vol_labels(index_df, horizon=10):
    """未来 horizon 日年化波动率 (真实标签, 用未来收益 t+1..t+horizon)"""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon 必须为正整数")
    rets = index_df["close"].pct_change(fill_method=None)
    fwd_rets = rets.shift(-1)
    return fwd_rets.iloc[::-1].rolling(horizon, min_periods=horizon).std().iloc[::-1] * np.sqrt(250)


class VolRegimeClassifier:
    """未来波动率二分类器 (volatile/steady)"""

    def __init__(self, label_horizon=10):
        self.label_horizon = label_horizon
        self.lgb = None
        self.feature_names = None
        self.vol_median = None  # 训练集未来波动率中位数 (标签二分阈值)
        self.last_train_date = None
        self.valid_acc = None
        self.valid_balanced_accuracy = None
        self.valid_auc = None

    def fit(self, train_start, train_end, valid_start, valid_end, persist=True):
        from core.data.fetcher import DataFetcher

        fetcher = DataFetcher()
        index_df = fetcher.get_index_daily("沪深300", days=365 * 10)
        if index_df is None or index_df.empty:
            raise RuntimeError("无法获取沪深300日线数据")

        index_df = index_df.sort_values("date").reset_index(drop=True)
        index_df = index_df[index_df["date"] >= pd.Timestamp("2015-01-01")]

        df = _make_features(index_df, None)
        df["date"] = pd.to_datetime(df["date"])
        df["fut_vol"] = _future_vol_labels(index_df, self.label_horizon)

        feature_cols = [c for c in df.columns if c not in {"date", "fut_vol"}
                        and c not in _NONSTATIONARY
                        and df[c].dtype in (np.float64, np.int64, float, int)]
        df = df.dropna(subset=feature_cols + ["fut_vol"])

        train_mask = (df["date"] >= pd.Timestamp(train_start)) & (df["date"] <= pd.Timestamp(train_end))
        valid_mask = (df["date"] >= pd.Timestamp(valid_start)) & (df["date"] <= pd.Timestamp(valid_end))

        # 标签阈值用训练集 (point-in-time, 避免未来)
        self.vol_median = float(df.loc[train_mask, "fut_vol"].median())
        y = (df["fut_vol"] > self.vol_median).astype(int)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = y[train_mask].values
        X_valid = df.loc[valid_mask, feature_cols].values
        y_valid = y[valid_mask].values

        if len(X_train) < 100 or len(X_valid) < 20:
            raise RuntimeError(f"训练样本不足: train={len(X_train)}, valid={len(X_valid)}")

        from lightgbm import LGBMClassifier
        self.lgb = LGBMClassifier(
            n_estimators=200, max_depth=5, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
            min_child_samples=30, random_state=42, verbose=-1,
        )
        self.lgb.fit(X_train, y_train)
        self.feature_names = feature_cols

        from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
        p_valid = self.lgb.predict_proba(X_valid)[:, 1]
        self.valid_acc = float(accuracy_score(y_valid, p_valid > 0.5))
        self.valid_auc = float(roc_auc_score(y_valid, p_valid))
        self.valid_balanced_accuracy = float(balanced_accuracy_score(y_valid, p_valid > 0.5))

        self.last_train_date = train_end
        if persist:
            self._save()
        logger.info(f"[VolRegime] 训练完成 valid_acc={self.valid_acc:.3f} "
                    f"balanced_acc={self.valid_balanced_accuracy:.3f} valid_auc={self.valid_auc:.3f}")

        return {
            "model_type": "vol_regime",
            "task_type": "vol_regime",
            "n_features": len(self.feature_names),
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "valid_acc": round(self.valid_acc, 4),
            "valid_balanced_accuracy": round(self.valid_balanced_accuracy, 4),
            "valid_auc": round(self.valid_auc, 4),
            "vol_median": round(self.vol_median, 6),
        }

    def predict_proba(self, index_df):
        """预测 volatile 概率 [0,1] (point-in-time, 用 index_df 截至当日)"""
        if self.lgb is None or not self.feature_names:
            return None
        df = _make_features(index_df, None)
        df = df.dropna(subset=self.feature_names)
        if df.empty:
            return None
        for c in self.feature_names:
            if c not in df.columns:
                df[c] = 0
        X = df[self.feature_names].iloc[-1:].values
        return float(self.lgb.predict_proba(X)[0, 1])

    def _save(self):
        path = MODEL_DIR / f"vol_regime_h{self.label_horizon}.pkl"
        from core.shared.persistence import save_pickle
        save_pickle({
            "label_horizon": self.label_horizon,
            "feature_names": self.feature_names,
            "vol_median": self.vol_median,
            "last_train_date": self.last_train_date,
            "valid_acc": self.valid_acc,
            "valid_balanced_accuracy": self.valid_balanced_accuracy,
            "valid_auc": self.valid_auc,
            "lgb": self.lgb,
        }, str(path))
        logger.info(f"[VolRegime] 模型已保存: {path}")

    def _load(self):
        path = MODEL_DIR / f"vol_regime_h{self.label_horizon}.pkl"
        if not path.exists():
            return False
        try:
            from core.shared.persistence import load_pickle
            data = load_pickle(str(path))
            self.label_horizon = data.get("label_horizon", self.label_horizon)
            self.feature_names = data.get("feature_names")
            self.vol_median = data.get("vol_median")
            self.last_train_date = data.get("last_train_date")
            self.valid_acc = data.get("valid_acc")
            self.valid_balanced_accuracy = data.get("valid_balanced_accuracy")
            self.valid_auc = data.get("valid_auc")
            self.lgb = data.get("lgb")
            return self.lgb is not None
        except Exception as e:
            logger.error(f"[VolRegime] 加载失败: {e}")
            return False

    def _is_loaded(self):
        return self.lgb is not None and self.feature_names is not None


# 全局单例
_vol_classifier = None
_vol_lock = None
_vol_classifiers = {}


def get_vol_classifier(label_horizon=10):
    global _vol_classifier, _vol_lock, _vol_classifiers
    label_horizon = int(label_horizon)
    cached = _vol_classifiers.get(label_horizon)
    if cached is not None and cached._is_loaded():
        return cached
    if _vol_lock is None:
        import threading
        _vol_lock = threading.Lock()
    with _vol_lock:
        cached = _vol_classifiers.get(label_horizon)
        if cached is not None and cached._is_loaded():
            return cached
        m = VolRegimeClassifier(label_horizon=label_horizon)
        if m._load():
            _vol_classifiers[label_horizon] = m
            if label_horizon == 10:
                _vol_classifier = m
            return m
        _vol_classifiers[label_horizon] = m
        if label_horizon == 10:
            _vol_classifier = m
        return m
