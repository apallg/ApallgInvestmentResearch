import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

__all__ = ["IndexTimingModel", "get_timing_ratio", "get_model",
           "prob_to_ratio", "annual_return", "max_drawdown", "win_rate"]

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # core/domain/timing/model.py → project root
MODEL_DIR = ROOT / "data" / "timing_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
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
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
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

ELASTICNET_PARAMS = {
    "solver": "saga",
    "l1_ratio": 0.5,
    "C": 0.1,
    "max_iter": 5000,
    "class_weight": "balanced",
    "random_state": 42,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "alpha": 0.001,
    "learning_rate_init": 0.001,
    "max_iter": 800,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 42,
}

def prob_to_ratio(prob):
    """sigmoid 平滑仓位映射: prob 0→0.2, 0.3→0.33, 0.5→0.62, 0.7→0.86, 1→0.98.
    比硬编码阶梯过渡更平滑, 减少频繁调仓。"""
    import numpy as np
    return round(float(0.2 + 0.8 / (1 + np.exp(-8 * (prob - 0.45)))), 2)


def get_timing_ratio(as_of=None):
    try:
        from core.domain.portfolio.consistency import check_panic
        if check_panic(as_of):
            return 0.0
    except Exception:
        pass
    model = IndexTimingModel()
    result = model.predict(as_of=as_of)
    return result.get("position_ratio", 1.0)


def _make_features(index_df, breadth_df=None):
    """从 sh000300 日线 + 市场宽度 构造特征矩阵"""
    df = index_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    amount = df["amount"].astype(float)

    # 收益率
    for n in [1, 5, 10, 20]:
        df[f"ret_{n}d"] = close.pct_change(n, fill_method=None)

    # MA 偏离
    for n in [5, 10, 20, 60]:
        ma = close.rolling(n).mean()
        df[f"ma{n}_dev"] = close / ma - 1

    # 波动率
    for n in [5, 10, 20]:
        df[f"vol_{n}d"] = close.pct_change(fill_method=None).rolling(n).std()

    # 量比
    vol_ma20 = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_ma20

    # 振幅
    df["amplitude"] = (high - low) / close

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带位置
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_position"] = (close - bb_ma) / (bb_std * 2 + 1e-9)

    # 连涨/连跌
    df["consecutive"] = 0
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            df.loc[df.index[i], "consecutive"] = (
                df.loc[df.index[i - 1], "consecutive"] + 1
                if df.loc[df.index[i - 1], "consecutive"] > 0
                else 1
            )
        else:
            df.loc[df.index[i], "consecutive"] = (
                df.loc[df.index[i - 1], "consecutive"] - 1
                if df.loc[df.index[i - 1], "consecutive"] < 0
                else -1
            )

    # 额比
    amount_ma20 = amount.rolling(20).mean()
    df["amount_ratio"] = amount / amount_ma20

    # 波动率/换手率牛熊指标 (华泰证券, 与指数走势负相关 r≈-0.55)
    ret_1d = close.pct_change(fill_method=None)
    vol_std_200 = ret_1d.rolling(200).std()
    turnover_source = _turnover_feature_source(df)
    turnover_col = (pd.to_numeric(df["turnover"], errors="coerce")
                    if turnover_source == "turnover" else volume)
    turnover_ma_200 = turnover_col.rolling(200).mean()
    df["vol_turnover"] = vol_std_200 / turnover_ma_200.replace(0, np.nan)

    # 合并市场宽度
    if breadth_df is not None:
        br_cols = [
            "up_ratio", "up_volume_ratio", "limit_up", "limit_down",
            "new_high_approx", "new_low_approx",
            "pct_median", "pct_std", "amplitude_median", "turnover_median",
        ]
        for c in br_cols:
            if c in breadth_df.columns and "trade_date" in breadth_df.columns:
                s = breadth_df[["trade_date", c]].copy()
                s["trade_date"] = s["trade_date"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
                s["trade_date"] = pd.to_datetime(s["trade_date"]).dt.strftime("%Y-%m-%d")
                s = s.set_index("trade_date")[c]
                df["_date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df[c] = df["_date_str"].map(s)
        if "_date_str" in df.columns:
            df = df.drop(columns=["_date_str"])

    return df


def _turnover_feature_source(index_df):
    if "turnover" not in index_df.columns:
        return "volume_proxy"
    turnover = pd.to_numeric(index_df["turnover"], errors="coerce")
    valid_turnover = turnover[np.isfinite(turnover)]
    if valid_turnover.empty or valid_turnover.eq(0).all():
        return "volume_proxy"
    return "turnover"


def _make_labels(index_df, horizon=5, threshold=0.015):
    """标签: 未来 horizon 日收益率 > threshold"""
    close = index_df["close"].astype(float)
    fwd_ret = close.shift(-horizon) / close - 1
    return (fwd_ret > threshold).astype(int)


class IndexTimingModel:

    def __init__(self, label_horizon=5, threshold=0.015, model_type="tree"):
        self.label_horizon = label_horizon
        self.threshold = threshold
        self.model_type = model_type
        self.lgb = None
        self.xgb = None
        self.elasticnet = None
        self.mlp = None
        self.scaler = None
        self.feature_names = None
        self.train_auc = None
        self.valid_auc = None
        self.lgb_auc = None
        self.xgb_auc = None
        self.en_auc = None
        self.mlp_auc = None
        self.last_train_date = None
        self.turnover_feature_source = None

    def fit(self, train_start, train_end, valid_start, valid_end):
        from core.data.fetcher import DataFetcher
        from core.domain.timing.breadth import get_breadth_history

        fetcher = DataFetcher()
        index_df = fetcher.get_index_daily("沪深300", days=365 * 10)
        if index_df.empty:
            raise RuntimeError("无法获取沪深300日线数据")

        index_df = index_df.sort_values("date").reset_index(drop=True)
        index_df = index_df[index_df["date"] >= pd.Timestamp("2015-01-01")]
        self.turnover_feature_source = _turnover_feature_source(index_df)

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
        df["label"] = _make_labels(index_df, self.label_horizon, self.threshold)

        # 拼接 HMM 状态概率特征
        regime_df = _get_regime_features(index_df)
        if regime_df is not None and "date" in regime_df.columns and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            regime_df["date"] = pd.to_datetime(regime_df["date"])
            df = df.merge(regime_df, on="date", how="left")
            for c in regime_df.columns:
                if c != "date" and c in df.columns:
                    df[c] = df[c].fillna(0)

        drop_cols = ["date", "label"]
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in (np.float64, np.int64, np.int32, float, int)]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols + ["label"])

        train_mask = (df["date"] >= pd.Timestamp(train_start)) & (df["date"] <= pd.Timestamp(train_end))
        valid_mask = (df["date"] >= pd.Timestamp(valid_start)) & (df["date"] <= pd.Timestamp(valid_end))

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "label"].values
        X_valid = df.loc[valid_mask, feature_cols].values
        y_valid = df.loc[valid_mask, "label"].values

        if len(X_train) < 100 or len(X_valid) < 20:
            raise RuntimeError(f"训练数据不足: train={len(X_train)}, valid={len(X_valid)}")

        self.feature_names = feature_cols

        if self.model_type == "linear":
            self._fit_linear(X_train, y_train, X_valid, y_valid)
        else:
            self._fit_tree(X_train, y_train, X_valid, y_valid)

        self.last_train_date = str(pd.Timestamp(df.loc[train_mask, "date"].max()).date())
        self._save()

        return {
            "model_type": self.model_type,
            "lgb_auc": round(self.lgb_auc, 4) if self.lgb_auc else None,
            "xgb_auc": round(self.xgb_auc, 4) if self.xgb_auc else None,
            "en_auc": round(self.en_auc, 4) if self.en_auc else None,
            "mlp_auc": round(self.mlp_auc, 4) if self.mlp_auc else None,
            "fusion_auc": round(self.valid_auc, 4),
            "n_features": len(self.feature_names),
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "train_period": f"{train_start} ~ {train_end}",
            "valid_period": f"{valid_start} ~ {valid_end}",
            "feature_names": self.feature_names,
            "turnover_feature_source": self.turnover_feature_source,
        }

    def _fit_tree(self, X_train, y_train, X_valid, y_valid):
        self.lgb = LGBMClassifier(n_estimators=200, early_stopping_rounds=50, **{k: v for k, v in LGB_PARAMS.items() if k not in ("n_estimators",)})
        self.lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

        self.xgb = XGBClassifier(early_stopping_rounds=50, **{k: v for k, v in XGB_PARAMS.items() if k != "n_estimators"})
        self.xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        lgb_prob = self.lgb.predict_proba(X_valid)[:, 1]
        xgb_prob = self.xgb.predict_proba(X_valid)[:, 1]
        self.lgb_auc = _auc(y_valid, lgb_prob)
        self.xgb_auc = _auc(y_valid, xgb_prob)
        self.valid_auc = _auc(y_valid, (lgb_prob + xgb_prob) / 2)

        logger.info(f"[Timing] LGB+XGB 训练完成 LGB_AUC={self.lgb_auc:.4f} XGB_AUC={self.xgb_auc:.4f} Fusion_AUC={self.valid_auc:.4f}")

    def _fit_linear(self, X_train, y_train, X_valid, y_valid):
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_valid_s = self.scaler.transform(X_valid)

        self.elasticnet = LogisticRegression(**ELASTICNET_PARAMS)
        self.elasticnet.fit(X_train_s, y_train)

        self.mlp = MLPClassifier(**MLP_PARAMS)
        self.mlp.fit(X_train_s, y_train)

        en_prob = self.elasticnet.predict_proba(X_valid_s)[:, 1]
        mlp_prob = self.mlp.predict_proba(X_valid_s)[:, 1]
        self.en_auc = _auc(y_valid, en_prob)
        self.mlp_auc = _auc(y_valid, mlp_prob)
        self.valid_auc = _auc(y_valid, (en_prob + mlp_prob) / 2)

        logger.info(f"[Timing] ElasticNet+MLP 训练完成 EN_AUC={self.en_auc:.4f} MLP_AUC={self.mlp_auc:.4f} Fusion_AUC={self.valid_auc:.4f}")

    def _prediction_result(self, as_of=None):
        if not self._is_loaded():
            self._load()
        if not self._is_loaded():
            return {"prob": None, "error": "模型未训练或未加载"}
        if not self.feature_names:
            return {"prob": None, "error": "模型缺少特征清单，请重新训练"}
        if self.turnover_feature_source not in ("turnover", "volume_proxy"):
            return {"prob": None, "error": "模型未记录换手率特征来源，请重新训练"}

        from core.data.fetcher import DataFetcher
        from core.domain.timing.breadth import get_breadth

        fetcher = DataFetcher()
        index_df = fetcher.get_index_daily("沪深300", days=365 * 2)
        if index_df is None or index_df.empty:
            return {"prob": None, "error": "无法获取指数日线数据"}
        index_df = index_df.sort_values("date").reset_index(drop=True)

        if as_of:
            index_df = index_df[index_df["date"] <= pd.Timestamp(as_of)]
        if index_df.empty:
            return {"prob": None, "error": "请求日期之前没有可用指数日线"}

        last_date = str(index_df["date"].iloc[-1].date())
        current_turnover_source = _turnover_feature_source(index_df)
        metadata = {
            "sample_date": last_date,
            "model_last_train_date": self.last_train_date,
            "label_horizon": self.label_horizon,
            "model_type": self.model_type,
            "turnover_feature_source": current_turnover_source,
        }
        if current_turnover_source != self.turnover_feature_source:
            return {"prob": None, **metadata,
                    "error": "换手率特征来源与模型训练契约不一致，请重新训练"}
        breadth = get_breadth(date=last_date)
        breadth_df = None
        if breadth:
            breadth["trade_date"] = last_date
            breadth_df = pd.DataFrame([breadth])

        df = _make_features(index_df, breadth_df)

        # 拼接 HMM 状态概率 (预测时用全量历史跑一次 HMM)
        if any(c.startswith("hmm_") for c in self.feature_names):
            regime_df = _get_regime_features(index_df)
            if regime_df is not None and "date" in regime_df.columns:
                df["date"] = pd.to_datetime(df["date"])
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                df = df.merge(regime_df, on="date", how="left")
                for c in regime_df.columns:
                    if c != "date" and c in df.columns:
                        df[c] = df[c].fillna(0)

        missing_features = [c for c in self.feature_names if c not in df.columns]
        if missing_features:
            return {"prob": None, **metadata,
                    "error": "预测必需特征缺失: " + ", ".join(missing_features)}
        latest = df.iloc[-1:]
        if latest.empty:
            return {"prob": None, **metadata, "error": "没有可用于预测的完整日期"}

        latest_features = latest[self.feature_names].replace([np.inf, -np.inf], np.nan)
        missing_values = latest_features.columns[latest_features.isna().any()].tolist()
        if missing_values:
            return {"prob": None, **metadata,
                    "error": "最新样本缺少必需特征值: " + ", ".join(missing_values)}

        X = latest_features.values
        if self.model_type == "linear":
            if self.scaler is not None:
                X = self.scaler.transform(X)
            en_prob = float(self.elasticnet.predict_proba(X)[:, 1][0])
            mlp_prob = float(self.mlp.predict_proba(X)[:, 1][0])
            prob = (en_prob + mlp_prob) / 2
        else:
            lgb_prob = float(self.lgb.predict_proba(X)[:, 1][0])
            xgb_prob = float(self.xgb.predict_proba(X)[:, 1][0])
            prob = (lgb_prob + xgb_prob) / 2
        if not np.isfinite(prob):
            return {"prob": None, **metadata, "error": "模型输出概率无效"}
        return {"prob": prob, **metadata}

    def predict_proba(self, as_of=None):
        return self._prediction_result(as_of).get("prob")

    def predict(self, as_of=None):
        result = self._prediction_result(as_of)
        prob = result.get("prob")
        if prob is None:
            return {"position_ratio": None, **result}
        signal = "bullish" if prob > 0.5 else "bearish"
        return {
            "prob": round(prob, 4),
            "position_ratio": prob_to_ratio(prob),
            "signal": signal,
            **{key: value for key, value in result.items() if key != "prob"},
            "as_of": result["sample_date"],
        }

    def _is_loaded(self):
        if self.model_type == "linear":
            return self.elasticnet is not None and self.mlp is not None
        return self.lgb is not None and self.xgb is not None

    def backtest(self, start, end):
        if not self._is_loaded():
            self._load()

        from core.data.fetcher import DataFetcher
        from core.domain.timing.breadth import get_breadth_history

        fetcher = DataFetcher()
        index_df = fetcher.get_index_daily("沪深300", days=365 * 10)
        index_df = index_df.sort_values("date").reset_index(drop=True)

        breadth_df = get_breadth_history("2015-01-01", str(index_df["date"].max().date()))
        if breadth_df is not None and not breadth_df.empty:
            import polars as pl
            if isinstance(breadth_df, pl.DataFrame):
                breadth_df = breadth_df.to_pandas()
            else:
                breadth_df = breadth_df.reset_index()

        df = _make_features(index_df, breadth_df)
        current_turnover_source = _turnover_feature_source(index_df)
        if current_turnover_source != self.turnover_feature_source:
            return {"error": "换手率特征来源与模型训练契约不一致，请重新训练"}

        # HMM 特征
        if any(c.startswith("hmm_") for c in self.feature_names):
            regime_df = _get_regime_features(index_df)
            if regime_df is not None and "date" in regime_df.columns:
                df["date"] = pd.to_datetime(df["date"])
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                df = df.merge(regime_df, on="date", how="left")
                for c in regime_df.columns:
                    if c != "date" and c in df.columns:
                        df[c] = df[c].fillna(0)

        missing_features = [c for c in self.feature_names if c not in df.columns]
        if missing_features:
            return {"error": "回测必需特征缺失: " + ", ".join(missing_features)}

        df[self.feature_names] = df[self.feature_names].replace([np.inf, -np.inf], np.nan)
        valid = df.dropna(subset=self.feature_names)
        valid = valid[(valid["date"] >= pd.Timestamp(start)) & (valid["date"] <= pd.Timestamp(end))]

        if valid.empty:
            return {"error": "回测区间无有效数据"}

        X = valid[self.feature_names].values
        if self.model_type == "linear":
            if self.scaler is not None:
                X = self.scaler.transform(X)
            en_prob = self.elasticnet.predict_proba(X)[:, 1]
            mlp_prob = self.mlp.predict_proba(X)[:, 1]
            prob = (en_prob + mlp_prob) / 2
        else:
            lgb_prob = self.lgb.predict_proba(X)[:, 1]
            xgb_prob = self.xgb.predict_proba(X)[:, 1]
            prob = (lgb_prob + xgb_prob) / 2

        ratios = [prob_to_ratio(p) for p in prob]
        index_ret = valid["ret_1d"].values if "ret_1d" in valid.columns else valid["close"].pct_change(fill_method=None).values
        strategy_ret = [r * (ret if not np.isnan(ret) else 0) for r, ret in zip(ratios[:-1], index_ret[1:])]

        cumulative = 1.0
        equity = [1.0]
        for r in strategy_ret:
            cumulative *= (1 + r)
            equity.append(cumulative)

        benchmark = 1.0
        bench_equity = [1.0]
        for r in index_ret:
            if not np.isnan(r):
                benchmark *= (1 + r)
            bench_equity.append(benchmark)

        hold_equity = [1.0]
        hold = 1.0
        for r in index_ret:
            if not np.isnan(r):
                hold *= (1 + r)
            hold_equity.append(hold)

        dates = [str(d.date()) for d in valid["date"][:len(equity)]]

        return {
            "dates": dates,
            "equity": [round(e, 4) for e in equity],
            "benchmark_equity": [round(e, 4) for e in bench_equity[:len(equity)]],
            "hold_equity": [round(e, 4) for e in hold_equity[:len(equity)]],
            "annual_return": round(annual_return(equity[-1], len(dates)), 4),
            "max_drawdown": round(max_drawdown(equity), 4),
            "win_rate": round(win_rate(strategy_ret), 4),
            "benchmark_return": round(annual_return(bench_equity[len(equity) - 1], len(dates)), 4),
        }

    def _save(self):
        suffix = "_linear" if self.model_type == "linear" else ""
        path = MODEL_DIR / f"timing_h{self.label_horizon}{suffix}.pkl"
        data = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "turnover_feature_source": self.turnover_feature_source,
            "label_horizon": self.label_horizon, "threshold": self.threshold,
            "valid_auc": self.valid_auc,
            "last_train_date": self.last_train_date,
        }
        if self.model_type == "linear":
            data.update({
                "elasticnet": self.elasticnet, "mlp": self.mlp,
                "scaler": self.scaler,
                "en_auc": self.en_auc, "mlp_auc": self.mlp_auc,
            })
        else:
            data.update({
                "lgb": self.lgb, "xgb": self.xgb,
                "lgb_auc": self.lgb_auc, "xgb_auc": self.xgb_auc,
            })
        from core.shared.persistence import save_pickle
        save_pickle(data, str(path))
        logger.info(f"[Timing] 模型已保存: {path}")

    def _load(self):
        requested = self.label_horizon
        for h in [requested, 10, 20, 5]:
            path = MODEL_DIR / f"timing_h{h}.pkl"
            if path.exists():
                self.label_horizon = h
                if h != requested:
                    logger.info(f"[Timing] h{requested} 模型不存在, 回退使用 h{h}")
                self._load_path(path)
                return
        logger.warning(f"[Timing] 模型文件不存在: {MODEL_DIR / f'timing_h{self.label_horizon}.pkl'}")

    def _load_path(self, path_or_name):
        from pathlib import Path
        if isinstance(path_or_name, Path):
            path = path_or_name
        else:
            path = MODEL_DIR / path_or_name
            if not path.suffix:
                path = path.with_suffix(".pkl")
        if not path.exists():
            logger.warning(f"[Timing] 模型文件不存在: {path}")
            return
        try:
            from core.shared.persistence import load_pickle
            data = load_pickle(str(path))
            self.model_type = data.get("model_type", "tree")
            self.feature_names = data.get("feature_names")
            self.turnover_feature_source = data.get("turnover_feature_source")
            self.label_horizon = data.get("label_horizon", self.label_horizon)
            self.threshold = data.get("threshold", self.threshold)
            self.valid_auc = data.get("valid_auc")
            self.last_train_date = data.get("last_train_date")
            if self.model_type == "linear":
                self.elasticnet = data.get("elasticnet")
                self.mlp = data.get("mlp")
                self.scaler = data.get("scaler")
                self.en_auc = data.get("en_auc")
                self.mlp_auc = data.get("mlp_auc")
            else:
                self.lgb = data.get("lgb")
                self.xgb = data.get("xgb")
                self.lgb_auc = data.get("lgb_auc")
                self.xgb_auc = data.get("xgb_auc")
            logger.info(f"[Timing] 已加载模型 ({self.model_type}): {path}")
        except Exception as e:
            logger.error(f"[Timing] 模型加载失败 {path}: {e}")


_cached_model = None


def get_model():
    global _cached_model
    if _cached_model is not None and _cached_model._is_loaded():
        return _cached_model
    # 扫描可用模型, 优先 tree → linear, h5 → h10 → h20
    for mt in ["tree", "linear"]:
        for h in [5, 10, 20]:
            m = IndexTimingModel(label_horizon=h, model_type=mt)
            m._load()
            if m._is_loaded():
                _cached_model = m
                return m
    _cached_model = IndexTimingModel()
    return _cached_model


def _get_regime_features(index_df):
    """获取 HMM 6 状态概率序列, 作为择时/Regime模型的附加特征

    优先加载已持久化的 HMM 模型, 失败时才现场训练。
    index_df 需包含 date, close, high, low, volume 列.
    """
    import numpy as np
    try:
        from core.domain.regime.classifier import HMMRegimeClassifier

        # 优先用已持久化的模型 (每天重训一次, 稳定)
        hmm = HMMRegimeClassifier.load()
        if hmm is None:
            hmm = HMMRegimeClassifier()
            if not hmm.fit(index_df.copy()):
                return None
            hmm.save()
        if hmm._model is None or np.any(np.isnan(hmm._model.startprob_)):
            return None

        features = hmm._extract_features(index_df)
        if features.empty:
            return None

        X = features.values.astype(np.float64)
        if not hasattr(hmm, '_feat_mean') or hmm._feat_mean is None:
            return None
        X = (X - hmm._feat_mean) / (hmm._feat_std + 1e-8)

        # 维度对齐: 模型特征数 vs 当前特征数
        expect_dim = len(hmm._feat_mean)
        if X.shape[1] != expect_dim:
            X = X[:, :expect_dim]

        probs_all = hmm._model.predict_proba(X)

        result = pd.DataFrame(index=features.index)
        for s_id, regime in hmm._state_map.items():
            col = f"hmm_{regime.value}"
            result[col] = probs_all[:, s_id] if s_id < probs_all.shape[1] else 0.0

        if "date" in index_df.columns:
            result["date"] = index_df.loc[features.index, "date"].values

        logger.info(f"[Timing] HMM 特征: {list(result.columns)} ({len(result)} 行)")
        return result
    except Exception as e:
        logger.warning(f"[Timing] HMM 特征提取失败: {e}")
        return None


def _auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.0


def annual_return(final_equity, n_days):
    years = n_days / 252
    if years <= 0:
        return 0
    return final_equity ** (1 / years) - 1


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (np.array(equity) - peak) / peak
    return float(np.min(dd))


def win_rate(rets):
    if not rets:
        return 0
    return sum(1 for r in rets if r > 0) / len(rets)
