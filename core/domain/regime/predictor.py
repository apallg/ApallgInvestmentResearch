"""
统一市场状态预测器 (Regime Predictor)

一路输出 → 两路消费:
  1. 仓位系数: Σ P(state) × RATIO_MAP[state]
  2. 信号源权重: Σ P(state) × REGIME_WEIGHTS[state]

标签由未来 N 日实际收益和波动率定义 (post-hoc), 不依赖规则引擎或 HMM。
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.domain.timing.model import _make_features, _get_regime_features
from core.domain.regime.types import Regime

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # core/domain/regime/predictor.py → project root
MODEL_DIR = ROOT / "data" / "timing_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 标签定义 ──
# 2×3 网格: 波动率 (高/低 按中位数二分) × 方向 (涨/跌/平 按33/67百分位三分)
# 标签 = 未来 horizon 日实际收益和波动率决定 (post-hoc)
STATE_NAMES = [
    "volatile_up",    # 0: 高波快涨
    "volatile_down",  # 1: 高波快跌
    "volatile_flat",  # 2: 高波震荡
    "steady_up",      # 3: 稳步上涨
    "steady_down",    # 4: 稳步下跌
    "steady_flat",    # 5: 低波盘整
]

# ── 仓位映射 (默认值, 可由 config/regime_ratio.json 覆盖) ──
_DEFAULT_RATIO_MAP = {
    0: 1.50, 1: 0.15, 2: 0.50,
    3: 1.50, 4: 0.15, 5: 0.50,
}
RATIO_MAP = dict(_DEFAULT_RATIO_MAP)

_RATIO_CONFIG_PATH = ROOT / "config" / "regime_ratio.json"


def _load_ratio_config():
    """从配置文件加载仓位映射, 不存在则用默认值"""
    global RATIO_MAP
    try:
        from core.shared.persistence import load_json
        data = load_json(str(_RATIO_CONFIG_PATH))
        if data:
            for k in range(6):
                if str(k) in data:
                    RATIO_MAP[k] = float(data[str(k)])
    except Exception:
        RATIO_MAP = dict(_DEFAULT_RATIO_MAP)


def save_ratio_config(ratios: dict, stops: dict = None):
    """保存仓位映射到配置文件"""
    from core.shared.persistence import save_json
    data = {str(k): v for k, v in sorted(ratios.items())}
    if stops:
        data['stops'] = {str(k): v for k, v in sorted(stops.items())}
    save_json(data, str(_RATIO_CONFIG_PATH))
    for k, v in ratios.items():
        RATIO_MAP[int(k)] = float(v)


_load_ratio_config()


def _make_regime_labels(index_df, horizon=10, train_end=None):
    """多分类标签: 2×3 网格 (波动率×方向)

    波动率: 按中位数二分 (高/低)
    方向: 按 33%/67% 百分位三分 (涨/平/跌)
    共 6 类, 各类 ~350-450 样本 (1.2x 平衡度)

    train_end 给定时阈值只在训练期 (date <= train_end) 统计, 避免混入验证期样本。
    """
    close = index_df["close"].astype(float)
    fwd_ret = close.shift(-horizon) / close - 1

    rets = close.pct_change(fill_method=None)
    vol_annual = rets.rolling(20).std() * np.sqrt(250)

    n = len(close)
    labels = np.full(n, 5, dtype=int)  # 默认 STEADY_FLAT

    # 阈值只在训练期统计 (避免前向偏差)
    valid_mask = fwd_ret.notna() & vol_annual.notna()
    if train_end is not None and "date" in index_df.columns:
        valid_mask = valid_mask & (pd.to_datetime(index_df["date"]) <= pd.Timestamp(train_end))
    if valid_mask.sum() < 30:
        return labels

    vol_vals = vol_annual[valid_mask].values
    ret_vals = fwd_ret[valid_mask].values
    vol_median = np.median(vol_vals)
    ret_top = np.percentile(ret_vals, 67)
    ret_bot = np.percentile(ret_vals, 33)

    for i in range(n - horizon):
        ret = fwd_ret.iloc[i]
        vol = vol_annual.iloc[i]
        if pd.isna(ret) or pd.isna(vol):
            labels[i] = 5  # steady_flat
        elif vol > vol_median:
            if ret > ret_top:
                labels[i] = 0  # volatile_up
            elif ret < ret_bot:
                labels[i] = 1  # volatile_down
            else:
                labels[i] = 2  # volatile_flat
        else:
            if ret > ret_top:
                labels[i] = 3  # steady_up
            elif ret < ret_bot:
                labels[i] = 4  # steady_down
            else:
                labels[i] = 5  # steady_flat

    return labels


def _auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def _multi_auc(y_true, y_prob_2d):
    """多分类 macro AUC (One-vs-Rest)"""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob_2d, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _estimate_transition_matrix(labels, n_classes=6, laplace=1.0, self_clip=0.95):
    """从标签序列统计状态转移矩阵 (CRF 平滑用)。

    重叠窗口自相关会虚高自转移, 对角 clip 到 self_clip 防止过度粘滞。
    """
    trans = np.zeros((n_classes, n_classes))
    for t in range(1, len(labels)):
        trans[int(labels[t - 1]), int(labels[t])] += 1
    trans += laplace  # 拉普拉斯平滑, 避免零转移
    trans /= trans.sum(axis=1, keepdims=True)
    for i in range(n_classes):
        if trans[i, i] > self_clip:
            excess = trans[i, i] - self_clip
            trans[i, i] = self_clip
            off = [j for j in range(n_classes) if j != i]
            off_sum = trans[i, off].sum()
            if off_sum > 0:
                trans[i, off] += excess * trans[i, off] / off_sum
    return trans / trans.sum(axis=1, keepdims=True)


class RegimePredictor:
    """统一市场状态预测器

    训练: 监督多分类 (6状态), 标签由未来收益定义
    预测: 状态概率分布 → 仓位系数 + 信号源权重
    """

    def __init__(self, label_horizon=10, model_type="linear", use_hmm_features=True):
        self.label_horizon = label_horizon
        self.model_type = model_type
        self.use_hmm_features = use_hmm_features
        self.lgb = None
        self.xgb = None
        self.elasticnet = None
        self.mlp = None
        self.scaler = None
        self.feature_names = None
        self.valid_acc = None
        self.valid_top2 = None
        self.valid_macro_auc = None
        self.lgb_acc = None
        self.xgb_acc = None
        self.en_acc = None
        self.mlp_acc = None
        self.class_weights = None
        self.last_train_date = None
        self.transition_matrix = None  # 学得的 6×6 状态转移矩阵 (CRF 平滑用)
        self._fusion_partner = None  # 融合模式: 另一个模型实例

    # ── 训练 ──

    def fit(self, train_start, train_end, valid_start, valid_end):
        from core.data.fetcher import DataFetcher
        from core.domain.timing.breadth import get_breadth_history

        fetcher = DataFetcher()
        index_df = fetcher.get_index_daily("沪深300", days=365 * 10)
        if index_df is None or index_df.empty:
            raise RuntimeError("无法获取沪深300日线数据")

        index_df = index_df.sort_values("date").reset_index(drop=True)
        index_df = index_df[index_df["date"] >= pd.Timestamp("2015-01-01")]
        if len(index_df) < 200:
            raise RuntimeError(f"数据不足: {len(index_df)} 行 < 200")

        breadth_df = get_breadth_history(
            str(index_df["date"].min().date()),
            str(index_df["date"].max().date()),
        )
        if breadth_df is not None and not breadth_df.empty:
            import polars as pl
            if isinstance(breadth_df, pl.DataFrame):
                breadth_df = breadth_df.to_pandas()
            else:
                breadth_df = breadth_df.reset_index()

        df = _make_features(index_df, breadth_df)
        df["label"] = _make_regime_labels(index_df, self.label_horizon, train_end=train_end)

        # HMM 特征 (消融开关: use_hmm_features=False 时跳过)
        if self.use_hmm_features:
            regime_df = _get_regime_features(index_df)
            if regime_df is not None and "date" in regime_df.columns and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                df = df.merge(regime_df, on="date", how="left")
                for c in regime_df.columns:
                    if c != "date" and c in df.columns:
                        df[c] = df[c].fillna(0)

        drop_cols = ["date", "label"]
        feature_cols = [c for c in df.columns if c not in drop_cols
                        and df[c].dtype in (np.float64, np.int64, np.int32, float, int)]

        # 线性模型去掉非平稳的原始价量列 (open/high/low/close/volume/amount)
        # 这些列的绝对值随市场价格漂移, 会导致概率极端偏向单一类别
        if self.model_type == "linear":
            _nonstationary = {"open", "high", "low", "close", "volume", "amount"}
            feature_cols = [c for c in feature_cols if c not in _nonstationary]
        df = df.dropna(subset=feature_cols + ["label"])

        train_mask = (df["date"] >= pd.Timestamp(train_start)) & (df["date"] <= pd.Timestamp(train_end))
        valid_mask = (df["date"] >= pd.Timestamp(valid_start)) & (df["date"] <= pd.Timestamp(valid_end))

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "label"].values.astype(int)
        X_valid = df.loc[valid_mask, feature_cols].values
        y_valid = df.loc[valid_mask, "label"].values.astype(int)

        if len(X_train) < 100:
            raise RuntimeError(f"训练样本不足: {len(X_train)} < 100")
        if len(X_valid) < 20:
            raise RuntimeError(f"验证样本不足: {len(X_valid)} < 20")

        # 检查类别覆盖
        train_classes = set(np.unique(y_train))
        valid_classes = set(np.unique(y_valid))
        missing = set(range(6)) - train_classes
        if missing:
            logger.info(f"[Regime] 训练集缺少类别: {[STATE_NAMES[c] for c in missing]}, "
                           f"将使用 class_weight='balanced'")
        if set(range(6)) - valid_classes:
            logger.info(f"[Regime] 验证集缺少类别: "
                           f"{[STATE_NAMES[c] for c in set(range(6)) - valid_classes]} (该时间段正常的)")

        # 类别权重 (处理不平衡)
        from sklearn.utils.class_weight import compute_class_weight
        try:
            cw = compute_class_weight("balanced", classes=np.array(sorted(train_classes)), y=y_train)
            self.class_weights = {c: float(w) for c, w in zip(sorted(train_classes), cw)}
        except Exception:
            self.class_weights = None

        self.feature_names = feature_cols

        # 统计训练集标签转移矩阵 (排除末尾 horizon 个伪标签 5), 供 CRF 平滑用
        n_eff = max(1, len(y_train) - self.label_horizon)
        self.transition_matrix = _estimate_transition_matrix(y_train[:n_eff], n_classes=6)

        if self.model_type == "tree":
            self._fit_tree(X_train, y_train, X_valid, y_valid, len(train_classes))
        else:
            self._fit_linear(X_train, y_train, X_valid, y_valid)

        self.last_train_date = train_end
        self._save()

        return {
            "model_type": self.model_type,
            "task_type": "regime",
            "n_classes": 6,
            "n_features": len(self.feature_names),
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "train_period": f"{train_start} ~ {train_end}",
            "valid_period": f"{valid_start} ~ {valid_end}",
            "valid_acc": round(self.valid_acc, 4) if self.valid_acc is not None else None,
            "valid_top2": round(self.valid_top2, 4) if self.valid_top2 is not None else None,
            "valid_macro_auc": round(self.valid_macro_auc, 4) if self.valid_macro_auc is not None else None,
            "lgb_acc": round(self.lgb_acc, 4) if self.lgb_acc is not None else None,
            "xgb_acc": round(self.xgb_acc, 4) if self.xgb_acc is not None else None,
            "en_acc": round(self.en_acc, 4) if self.en_acc is not None else None,
            "mlp_acc": round(self.mlp_acc, 4) if self.mlp_acc is not None else None,
            "class_distribution": {STATE_NAMES[k]: int(v) for k, v in
                                   zip(*np.unique(y_train, return_counts=True))},
            "feature_names": self.feature_names,
            "transition_matrix": self.transition_matrix.tolist(),
        }

    def _fit_tree(self, X_train, y_train, X_valid, y_valid, n_train_classes):
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        n_classes = 6
        lgb_kwargs = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 63,
            "n_jobs": 4,
            "min_child_samples": 50,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.3,
            "reg_lambda": 0.5,
            "verbose": -1,
            "random_state": 42,
            "class_weight": self.class_weights or "balanced",
        }

        self.lgb = LGBMClassifier(n_estimators=200, early_stopping_rounds=50, **lgb_kwargs)
        self.lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

        xgb_kwargs = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
            "max_depth": 5,
            "learning_rate": 0.03,
            "n_estimators": 300,
            "n_jobs": 4,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.3,
            "reg_lambda": 0.5,
            "random_state": 42,
        }
        self.xgb = XGBClassifier(early_stopping_rounds=50, **xgb_kwargs)
        self.xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        lgb_prob = self.lgb.predict_proba(X_valid)
        xgb_prob = self.xgb.predict_proba(X_valid)
        fusion_prob = (lgb_prob + xgb_prob) / 2

        self.lgb_acc = float((lgb_prob.argmax(1) == y_valid).mean())
        self.xgb_acc = float((xgb_prob.argmax(1) == y_valid).mean())
        self.valid_acc = float((fusion_prob.argmax(1) == y_valid).mean())
        self.valid_top2 = self._top_k_acc(fusion_prob, y_valid, k=2)
        self.valid_macro_auc = _multi_auc(y_valid, fusion_prob)

        if self.valid_macro_auc is not None:
            logger.info(f"[Regime] LGB+XGB 训练完成 "
                        f"Acc={self.valid_acc:.3f} Top2={self.valid_top2:.3f} AUC={self.valid_macro_auc:.3f}")
        else:
            logger.info(f"[Regime] LGB+XGB 训练完成 "
                        f"Acc={self.valid_acc:.3f} Top2={self.valid_top2:.3f} AUC=N/A")

    def _fit_linear(self, X_train, y_train, X_valid, y_valid):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_valid_s = self.scaler.transform(X_valid)

        en_kwargs = {
            "solver": "saga",
            "l1_ratio": 0.5,
            "C": 0.1,
            "max_iter": 5000,
            "class_weight": self.class_weights or "balanced",
            "random_state": 42,
        }
        self.elasticnet = LogisticRegression(**en_kwargs)
        self.elasticnet.fit(X_train_s, y_train)

        mlp_kwargs = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 800,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
        }
        self.mlp = MLPClassifier(**mlp_kwargs)
        self.mlp.fit(X_train_s, y_train)

        en_prob = self.elasticnet.predict_proba(X_valid_s)
        mlp_prob = self.mlp.predict_proba(X_valid_s)
        fusion_prob = (en_prob + mlp_prob) / 2

        self.en_acc = float((en_prob.argmax(1) == y_valid).mean())
        self.mlp_acc = float((mlp_prob.argmax(1) == y_valid).mean())
        self.valid_acc = float((fusion_prob.argmax(1) == y_valid).mean())
        self.valid_top2 = self._top_k_acc(fusion_prob, y_valid, k=2)
        self.valid_macro_auc = _multi_auc(y_valid, fusion_prob)

        if self.valid_macro_auc is not None:
            logger.info(f"[Regime] EN+MLP 训练完成 "
                        f"Acc={self.valid_acc:.3f} Top2={self.valid_top2:.3f} AUC={self.valid_macro_auc:.3f}")
        else:
            logger.info(f"[Regime] EN+MLP 训练完成 "
                        f"Acc={self.valid_acc:.3f} Top2={self.valid_top2:.3f} AUC=N/A")

    @staticmethod
    def _top_k_acc(probs, y_true, k=2):
        top_k = np.argsort(probs, axis=1)[:, -k:]
        hits = np.any(top_k == y_true.reshape(-1, 1), axis=1)
        return float(hits.mean())

    # ── 预测 ──

    def predict_regime(self, as_of=None):
        """预测当前市场状态概率分布

        Returns:
            dict with:
              - probs: {state_name: probability}
              - position_ratio: 建议仓位 (0.2~0.9)
              - signal_weights: 信号源权重 (dict)
              - confidence: 最高概率
              - top_state: 最可能的状态
              - as_of: 预测日期
        """
        probs = self._predict_proba(as_of=as_of)
        if probs is None:
            return {
                "probs": None,
                "position_ratio": None,
                "signal_weights": None,
                "confidence": None,
                "top_state": None,
                "model_type": self.model_type,
                "as_of": as_of or "latest",
                "error": "模型未加载或预测失败",
            }

        top_idx = int(np.argmax(probs))
        position = _soft_position_ratio(probs)
        weights = _soft_signal_weights(probs)

        return {
            "probs": {STATE_NAMES[i]: round(float(probs[i]), 4) for i in range(6)},
            "position_ratio": round(position, 4),
            "signal_weights": weights,
            "confidence": round(float(probs[top_idx]), 4),
            "top_state": STATE_NAMES[top_idx],
            "model_type": self.model_type,
            "as_of": as_of or "latest",
        }

    def predict_from_df(self, index_df, breadth_df=None):
        """从历史数据切片预测 — 供回测使用

        Args:
            index_df: 指数日线 DataFrame (date, open, high, low, close, volume)
            breadth_df: 可选, 市场宽度 DataFrame (trade_date, up_ratio, ...)

        Returns:
            numpy array shape (n_classes,) — 概率分布, 或 None
        """
        if not self._is_loaded():
            self._load()
        if not self._is_loaded():
            return None

        # 统一列名: 回测传入 trade_date, 实时拉取是 date
        if "trade_date" in index_df.columns and "date" not in index_df.columns:
            index_df = index_df.rename(columns={"trade_date": "date"})
        index_df = index_df.sort_values("date").reset_index(drop=True)

        if breadth_df is not None and not breadth_df.empty:
            import polars as pl
            if isinstance(breadth_df, pl.DataFrame):
                breadth_df = breadth_df.to_pandas()
            if "trade_date" not in breadth_df.columns:
                breadth_df = breadth_df.reset_index()

        df = _make_features(index_df, breadth_df)

        if any(c.startswith("hmm_") for c in self.feature_names):
            regime_df = _get_regime_features(index_df)
            if regime_df is not None and "date" in regime_df.columns:
                df["date"] = pd.to_datetime(df["date"])
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                df = df.merge(regime_df, on="date", how="left")
                for c in regime_df.columns:
                    if c != "date" and c in df.columns:
                        df[c] = df[c].fillna(0)

        for c in self.feature_names:
            if c not in df.columns:
                df[c] = 0

        df[self.feature_names] = df[self.feature_names].ffill().bfill().fillna(0)
        latest = df.iloc[-1:]
        X = latest[self.feature_names].values
        probs = self._raw_predict_proba(X)
        if probs is not None and self._fusion_partner is not None:
            partner_probs = self._fusion_partner.predict_from_df(index_df, breadth_df)
            if partner_probs is not None:
                probs = (probs + partner_probs) / 2
        return probs

    def _predict_proba(self, as_of=None):
        if not self._is_loaded():
            self._load()
        if not self._is_loaded():
            logger.warning("[Regime] _predict_proba: 模型未加载")
            return None

        from core.data.fetcher import DataFetcher
        from core.domain.timing.breadth import get_breadth

        fetcher = DataFetcher()
        index_df = fetcher.get_index_daily("沪深300", days=365 * 2)
        if index_df is None or index_df.empty:
            logger.warning("[Regime] _predict_proba: 无法获取指数数据")
            return None

        index_df = index_df.sort_values("date").reset_index(drop=True)
        if as_of:
            index_df = index_df[index_df["date"] <= pd.Timestamp(as_of)]

        eval_date = as_of or str(index_df["date"].iloc[-1].date())
        breadth = get_breadth(date=eval_date)
        breadth_df = None
        if breadth:
            breadth["trade_date"] = eval_date
            breadth_df = pd.DataFrame([breadth])

        df = _make_features(index_df, breadth_df)

        if any(c.startswith("hmm_") for c in self.feature_names):
            regime_df = _get_regime_features(index_df)
            if regime_df is not None and "date" in regime_df.columns:
                df["date"] = pd.to_datetime(df["date"])
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                df = df.merge(regime_df, on="date", how="left")
                for c in regime_df.columns:
                    if c != "date" and c in df.columns:
                        df[c] = df[c].fillna(0)

        for c in self.feature_names:
            if c not in df.columns:
                df[c] = 0

        # 取最新完整行; 用 ffill 填宽度/HMM等单日特征
        df[self.feature_names] = df[self.feature_names].ffill().bfill().fillna(0)
        latest = df.iloc[-1:]
        X = latest[self.feature_names].values
        probs = self._raw_predict_proba(X)

        # 融合模式: 与 partner 模型平均概率
        if probs is not None and self._fusion_partner is not None:
            partner_probs = self._fusion_partner._predict_proba(as_of=as_of)
            if partner_probs is not None:
                probs = (probs + partner_probs) / 2

        return probs

    def _raw_predict_proba(self, X):
        """单模型概率预测 (不涉及融合)"""
        if self.model_type == "tree" or (self.model_type == "fusion" and self.lgb is not None):
            if self.lgb is None or self.xgb is None:
                return None
            lgb_p = self.lgb.predict_proba(X)[0]
            xgb_p = self.xgb.predict_proba(X)[0]
            return (lgb_p + xgb_p) / 2
        else:
            if self.elasticnet is None or self.mlp is None:
                return None
            if self.scaler is not None:
                X = self.scaler.transform(X)
            en_p = self.elasticnet.predict_proba(X)[0]
            mlp_p = self.mlp.predict_proba(X)[0]
            return (en_p + mlp_p) / 2

    def _is_loaded(self):
        if self._fusion_partner is not None:
            return self._fusion_partner._is_loaded()
        if self.model_type == "tree":
            return self.lgb is not None and self.xgb is not None
        return self.elasticnet is not None and self.mlp is not None

    # ── 持久化 ──

    def _save(self):
        suffix = "_linear" if self.model_type == "linear" else ""
        path = MODEL_DIR / f"regime_h{self.label_horizon}{suffix}.pkl"
        data = {
            "model_type": self.model_type,
            "label_horizon": self.label_horizon,
            "feature_names": self.feature_names,
            "valid_acc": self.valid_acc,
            "valid_top2": self.valid_top2,
            "valid_macro_auc": self.valid_macro_auc,
            "last_train_date": self.last_train_date,
            "class_weights": self.class_weights,
            "transition_matrix": self.transition_matrix,
        }
        if self.model_type == "tree":
            data.update({"lgb": self.lgb, "xgb": self.xgb,
                         "lgb_acc": self.lgb_acc, "xgb_acc": self.xgb_acc})
        else:
            data.update({"elasticnet": self.elasticnet, "mlp": self.mlp,
                         "scaler": self.scaler, "en_acc": self.en_acc, "mlp_acc": self.mlp_acc})
        from core.shared.persistence import save_pickle
        save_pickle(data, str(path))
        logger.info(f"[Regime] 模型已保存: {path}")

    def _load(self):
        suffix = "_linear" if self.model_type == "linear" else ""
        path = MODEL_DIR / f"regime_h{self.label_horizon}{suffix}.pkl"
        if not path.exists():
            # 尝试不带后缀的旧格式
            path = MODEL_DIR / f"regime_h{self.label_horizon}.pkl"
            if not path.exists():
                logger.warning(f"[Regime] 模型文件不存在: regime_h{self.label_horizon}")
                return
        self._load_path(path)

    def _load_path(self, path_or_name):
        if isinstance(path_or_name, (str, Path)):
            path = Path(path_or_name)
            if not path.is_absolute():
                path = MODEL_DIR / path
            if not path.suffix:
                path = path.with_suffix(".pkl")
        else:
            path = path_or_name

        if not path.exists():
            logger.warning(f"[Regime] 模型文件不存在: {path}")
            return

        try:
            from core.shared.persistence import load_pickle
            data = load_pickle(str(path))
            self.model_type = data.get("model_type", "linear")
            self.label_horizon = data.get("label_horizon", self.label_horizon)
            self.feature_names = data.get("feature_names")
            self.valid_acc = data.get("valid_acc")
            self.valid_top2 = data.get("valid_top2")
            self.valid_macro_auc = data.get("valid_macro_auc")
            self.last_train_date = data.get("last_train_date")
            self.class_weights = data.get("class_weights")
            self.transition_matrix = data.get("transition_matrix")
            if self.model_type == "tree":
                self.lgb = data.get("lgb")
                self.xgb = data.get("xgb")
                self.lgb_acc = data.get("lgb_acc")
                self.xgb_acc = data.get("xgb_acc")
            else:
                self.elasticnet = data.get("elasticnet")
                self.mlp = data.get("mlp")
                self.scaler = data.get("scaler")
                self.en_acc = data.get("en_acc")
                self.mlp_acc = data.get("mlp_acc")
            logger.info(f"[Regime] 已加载模型: {path}")
        except Exception as e:
            logger.error(f"[Regime] 模型加载失败 {path}: {e}")


# ── 软融合 ──

def _soft_position_ratio(probs):
    """状态概率 → 加权仓位"""
    ratio = 0.0
    for i in range(6):
        ratio += float(probs[i]) * RATIO_MAP.get(i, 0.5)
    return round(ratio, 4)


def _soft_signal_weights(probs):
    """状态概率分布 → 软加权信号源权重

    复用 REGIME_WEIGHTS, 对每种状态对应的权重按概率加权平均。
    """
    from core.domain.regime.weights import REGIME_WEIGHTS
    state_to_regime = {
        0: Regime.VOLATILE_UP,
        1: Regime.VOLATILE_DOWN,
        2: Regime.VOLATILE_FLAT,
        3: Regime.STEADY_UP,
        4: Regime.STEADY_DOWN,
        5: Regime.STEADY_FLAT,
    }

    result = {}
    for i in range(6):
        regime = state_to_regime[i]
        w = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.STEADY_FLAT])
        for source, weight in w.items():
            result[source] = result.get(source, 0.0) + float(probs[i]) * weight

    total = sum(result.values())
    if total > 0:
        result = {k: round(v / total, 4) for k, v in result.items()}
    return result


# ── 全局单例 ──

_regime_predictor = None
_regime_lock = None


def _get_regime_lock():
    global _regime_lock
    if _regime_lock is None:
        import threading
        _regime_lock = threading.Lock()
    return _regime_lock


def get_regime_predictor():
    """获取/初始化 RegimePredictor (懒加载, 自动融合 tree+linear)

    如果 tree 和 linear 模型都存在 → 自动融合两个模型的概率
    只有一个 → 使用单个模型
    """
    global _regime_predictor
    if _regime_predictor is not None and _regime_predictor._is_loaded():
        return _regime_predictor

    with _get_regime_lock():
        if _regime_predictor is not None and _regime_predictor._is_loaded():
            return _regime_predictor

        if _regime_predictor is not None:
            _regime_predictor._load()
            if _regime_predictor._is_loaded():
                return _regime_predictor

        # 尝试加载 tree 和 linear
        tree_model = None
        linear_model = None
        for h in [10, 5, 20]:
            if tree_model is None:
                t = RegimePredictor(label_horizon=h, model_type="tree")
                t._load()
                if t._is_loaded():
                    tree_model = t
            if linear_model is None:
                l = RegimePredictor(label_horizon=h, model_type="linear")
                l._load()
                if l._is_loaded():
                    linear_model = l

        # 两个都有 → 融合
        if tree_model is not None and linear_model is not None:
            tree_model._fusion_partner = linear_model
            tree_model.model_type = "fusion"
            _regime_predictor = tree_model
            logger.info("[Regime] Tree+Linear 融合模式已激活")
            return tree_model

        # 只有一个
        model = tree_model or linear_model
        if model is not None:
            _regime_predictor = model
            return model

        if _regime_predictor is None:
            _regime_predictor = RegimePredictor()
        return _regime_predictor
