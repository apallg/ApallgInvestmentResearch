"""
波动率预测器 (Regime Predictor V2)

精简版 — 只做一件事: 预测未来 N 日年化波动率 → 仓位缩放系数。
V1 负责方向, V2 负责风控, 各司其职。

position_final = V1.position_base × V2.vol_scalar
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.domain.timing.model import _make_features, _get_regime_features

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # core/domain/regime/predictor_v2.py → project root
MODEL_DIR = ROOT / "data" / "timing_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 仓位缩放: pred_vol 相对于历史中位数的比例 → 系数 [0.7, 1.0]
# 高波 (>中位数) → scalar < 1 (降仓)
# 低波 (<中位数) → scalar = 1.0 (满仓)
VOL_MEDIAN = 0.17       # 历史年化波动率中位数 (~17%)
SCALAR_MIN = 0.70       # 最大降仓幅度 (pred_vol 极高时仓位最低打 7 折)
SCALAR_MAX = 1.00       # 低波时不加杠杆


def _vol_to_scalar(pred_vol):
    """pred_vol (0~1) → 仓位缩放系数 [0.7, 1.0]"""
    if pred_vol is None or np.isnan(pred_vol):
        return 1.0
    ratio = float(pred_vol) / VOL_MEDIAN
    scalar = 1.0 / max(ratio, 1.0)  # 只在 vol > median 时降仓
    return round(max(SCALAR_MIN, min(SCALAR_MAX, scalar)), 4)


def _make_regression_targets(index_df, horizon=10):
    """计算回归目标: 未来 horizon 日年化波动率"""
    close = index_df["close"].astype(float)
    rets = close.pct_change(fill_method=None)
    fwd_rets = rets.shift(-1)
    y_vol = np.full(len(close), np.nan, dtype=float)
    for i in range(len(close) - horizon):
        window = fwd_rets.iloc[i + 1:i + horizon + 1].dropna()
        if len(window) >= max(3, horizon // 2):
            y_vol[i] = float(window.std() * np.sqrt(250))
    valid = ~np.isnan(y_vol)
    return y_vol, valid


class RegimePredictorV2:
    """波动率预测器 — 只预测 vol, 输出仓位缩放系数"""

    def __init__(self, label_horizon=10):
        self.label_horizon = label_horizon
        self.vol_model = None
        self.scaler = None
        self.feature_names = None
        self.vol_r2 = None
        self.vol_mae = None
        self.last_train_date = None

    # ── 训练 ──

    def fit(self, train_start, train_end, valid_start, valid_end):
        from core.data.fetcher import DataFetcher
        from core.domain.timing.breadth import get_breadth_history
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

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
        y_vol, target_valid = _make_regression_targets(index_df, self.label_horizon)

        regime_df = _get_regime_features(index_df)
        if regime_df is not None and "date" in regime_df.columns and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            regime_df["date"] = pd.to_datetime(regime_df["date"])
            df = df.merge(regime_df, on="date", how="left")
            for c in regime_df.columns:
                if c != "date" and c in df.columns:
                    df[c] = df[c].fillna(0)

        drop_cols = ["date"]
        feature_cols = [c for c in df.columns if c not in drop_cols
                        and df[c].dtype in (np.float64, np.int64, np.int32, float, int)]
        _nonstationary = {"open", "high", "low", "close", "volume", "amount"}
        feature_cols = [c for c in feature_cols if c not in _nonstationary]

        df["y_vol"] = y_vol
        df["target_valid"] = target_valid
        df = df.dropna(subset=feature_cols + ["y_vol"])
        df = df[df["target_valid"]]

        if len(df) < 200:
            raise RuntimeError(f"有效样本不足: {len(df)} < 200")

        df["date"] = pd.to_datetime(df["date"])
        train_mask = (df["date"] >= pd.Timestamp(train_start)) & (df["date"] <= pd.Timestamp(train_end))
        valid_mask = (df["date"] >= pd.Timestamp(valid_start)) & (df["date"] <= pd.Timestamp(valid_end))

        X_train = df.loc[train_mask, feature_cols].values.astype(float)
        y_vol_train = df.loc[train_mask, "y_vol"].values.astype(float)
        X_valid = df.loc[valid_mask, feature_cols].values.astype(float)
        y_vol_valid = df.loc[valid_mask, "y_vol"].values.astype(float)

        if len(X_train) < 100:
            raise RuntimeError(f"训练样本不足: {len(X_train)} < 100")
        if len(X_valid) < 20:
            raise RuntimeError(f"验证样本不足: {len(X_valid)} < 20")

        self.feature_names = feature_cols
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_valid_s = self.scaler.transform(X_valid)

        self.vol_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=30,
            validation_fraction=0.15, n_iter_no_change=20, random_state=42,
        )
        self.vol_model.fit(X_train_s, y_vol_train)

        from sklearn.metrics import r2_score, mean_absolute_error
        vol_pred = self.vol_model.predict(X_valid_s)
        self.vol_r2 = float(r2_score(y_vol_valid, vol_pred))
        self.vol_mae = float(mean_absolute_error(y_vol_valid, vol_pred))

        logger.info(f"[RegimeV2] 训练完成 vol_R²={self.vol_r2:.4f} vol_MAE={self.vol_mae:.4f} "
                    f"n_train={len(X_train)} n_valid={len(X_valid)}")

        self.last_train_date = train_end
        self._save()
        return {
            "model_type": "volatility",
            "task_type": "regime_v2",
            "n_features": len(self.feature_names),
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "train_period": f"{train_start} ~ {train_end}",
            "valid_period": f"{valid_start} ~ {valid_end}",
            "vol_r2": round(self.vol_r2, 4),
            "vol_mae": round(self.vol_mae, 4),
            "feature_names": self.feature_names,
        }

    # ── 预测 ──

    def predict_regime(self, as_of=None):
        """预测波动率 → 仓位缩放系数

        Returns:
            pred_vol: 预测未来 N 日年化波动率
            vol_scalar: 仓位缩放系数 [0.7, 1.0]
        """
        pred_vol = self._predict_vol(as_of=as_of)
        if pred_vol is None:
            return {
                "pred_vol": None,
                "vol_scalar": 1.0,
                "model_type": "volatility",
                "as_of": as_of or "latest",
                "error": "模型未加载或预测失败",
            }
        return {
            "pred_vol": round(float(pred_vol), 6),
            "vol_scalar": _vol_to_scalar(pred_vol),
            "model_type": "volatility",
            "as_of": as_of or "latest",
        }

    def predict_from_df(self, index_df, breadth_df=None):
        """从历史数据切片预测 — 供回测使用

        Returns:
            vol_scalar: 仓位缩放系数 [0.7, 1.0]
        """
        if not self._is_loaded():
            self._load()
        if not self._is_loaded():
            return 1.0

        index_df = index_df.sort_values("date").reset_index(drop=True)

        if breadth_df is not None and not breadth_df.empty:
            import polars as pl
            if isinstance(breadth_df, pl.DataFrame):
                breadth_df = breadth_df.to_pandas()
            if "trade_date" not in breadth_df.columns:
                breadth_df = breadth_df.reset_index()

        df = _make_features(index_df, breadth_df)
        if df is None or df.empty or len(df) == 0:
            return 1.0

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

        if self.scaler is not None:
            X = self.scaler.transform(X)

        pred_vol = float(self.vol_model.predict(X)[0])
        pred_vol = max(pred_vol, 0.02)
        return _vol_to_scalar(pred_vol)

    def _predict_vol(self, as_of=None):
        if not self._is_loaded():
            self._load()
        if not self._is_loaded():
            return None

        from core.data.fetcher import DataFetcher
        from core.domain.timing.breadth import get_breadth

        fetcher = DataFetcher()
        index_df = fetcher.get_index_daily("沪深300", days=365 * 2)
        if index_df is None or index_df.empty:
            return None

        index_df = index_df.sort_values("date").reset_index(drop=True)
        if as_of:
            index_df = index_df[index_df["date"] <= pd.Timestamp(as_of)]

        last_date = str(index_df["date"].iloc[-1].date())
        breadth = get_breadth(date=last_date)
        breadth_df = None
        if breadth:
            breadth["trade_date"] = last_date
            breadth_df = pd.DataFrame([breadth])

        df = _make_features(index_df, breadth_df)
        if df is None or df.empty or len(df) == 0:
            return None

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
        X = df[self.feature_names].iloc[-1:].values

        if self.scaler is not None:
            X = self.scaler.transform(X)

        try:
            pred_vol = float(self.vol_model.predict(X)[0])
        except Exception as e:
            logger.error(f"[RegimeV2] 预测异常: {e}")
            return None

        return float(np.clip(pred_vol, 0.02, 0.50))

    def _is_loaded(self):
        return self.vol_model is not None and self.scaler is not None

    # ── 持久化 ──

    def _save(self):
        path = MODEL_DIR / f"regime_v2_h{self.label_horizon}.pkl"
        data = {
            "label_horizon": self.label_horizon,
            "feature_names": self.feature_names,
            "vol_r2": self.vol_r2,
            "vol_mae": self.vol_mae,
            "last_train_date": self.last_train_date,
            "vol_model": self.vol_model,
            "scaler": self.scaler,
        }
        from core.shared.persistence import save_pickle
        save_pickle(data, str(path))
        logger.info(f"[RegimeV2] 模型已保存: {path}")

    def _load(self):
        path = MODEL_DIR / f"regime_v2_h{self.label_horizon}.pkl"
        if not path.exists():
            logger.warning(f"[RegimeV2] 模型文件不存在: {path}")
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
            logger.warning(f"[RegimeV2] 模型文件不存在: {path}")
            return

        try:
            from core.shared.persistence import load_pickle
            data = load_pickle(str(path))
            self.label_horizon = data.get("label_horizon", self.label_horizon)
            self.feature_names = data.get("feature_names")
            self.vol_r2 = data.get("vol_r2")
            self.vol_mae = data.get("vol_mae")
            self.last_train_date = data.get("last_train_date")
            self.vol_model = data.get("vol_model")
            self.scaler = data.get("scaler")
            logger.info(f"[RegimeV2] 已加载模型: {path}")
        except Exception as e:
            logger.error(f"[RegimeV2] 模型加载失败 {path}: {e}")


# ── 全局单例 ──

_regime_predictor_v2 = None
_regime_v2_lock = None


def _get_regime_v2_lock():
    global _regime_v2_lock
    if _regime_v2_lock is None:
        import threading
        _regime_v2_lock = threading.Lock()
    return _regime_v2_lock


def get_regime_predictor_v2():
    global _regime_predictor_v2
    if _regime_predictor_v2 is not None and _regime_predictor_v2._is_loaded():
        return _regime_predictor_v2

    with _get_regime_v2_lock():
        if _regime_predictor_v2 is not None and _regime_predictor_v2._is_loaded():
            return _regime_predictor_v2

        if _regime_predictor_v2 is not None:
            _regime_predictor_v2._load()
            if _regime_predictor_v2._is_loaded():
                return _regime_predictor_v2

        for h in [10, 5, 20]:
            p = RegimePredictorV2(label_horizon=h)
            p._load()
            if p._is_loaded():
                _regime_predictor_v2 = p
                return p

        if _regime_predictor_v2 is None:
            _regime_predictor_v2 = RegimePredictorV2()
        return None
