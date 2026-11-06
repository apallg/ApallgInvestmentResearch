"""
多指标信号融合投票 — 对已附指标 DataFrame 做加权投票，产出综合交易信号
"""
from dataclasses import dataclass, field
import pandas as pd
import logging
import warnings

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "MACD": 1.0,
    "RSI": 0.8,
    "MA": 0.7,
    "KDJ": 0.5,
    "BB": 0.6,
    "VOLUME": 0.3,
}


@dataclass
class FusionResult:
    score: float = 0.0           # -max_weight ~ +max_weight
    action: str = "hold"         # buy / sell / hold
    confidence: float = 0.0      # 0.0 ~ 1.0
    indicator_votes: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


class SignalFusion:
    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._max_weight = sum(abs(w) for w in self.weights.values())

    def evaluate(self, df: pd.DataFrame) -> FusionResult:
        if len(df) < 2:
            return FusionResult()

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        votes = {}
        reasons = []

        v = self._vote_macd(prev, latest)
        votes["MACD"] = v
        if v != 0:
            reasons.append(("MACD", "金叉" if v > 0 else "死叉"))

        v = self._vote_rsi(latest)
        votes["RSI"] = v
        if v != 0:
            reasons.append(("RSI", "超卖" if v > 0 else "超买"))

        v = self._vote_ma(latest)
        votes["MA"] = v
        if v != 0:
            reasons.append(("MA", "多头排列" if v > 0 else "空头排列"))

        v = self._vote_kdj(prev, latest)
        votes["KDJ"] = v
        if v != 0:
            reasons.append(("KDJ", "金叉" if v > 0 else "死叉"))

        v = self._vote_bb(latest)
        votes["BB"] = v
        if v != 0:
            reasons.append(("BB", "触及下轨" if v > 0 else "触及上轨"))

        v = self._vote_volume(latest)
        votes["VOLUME"] = v
        if v != 0:
            reasons.append(("VOLUME", "放量上涨" if v > 0 else "放量下跌"))

        weighted_sum = sum(self.weights.get(k, 0) * v for k, v in votes.items())
        confidence = abs(weighted_sum) / self._max_weight if self._max_weight > 0 else 0

        if weighted_sum > 0 and confidence >= 0.3:
            action = "buy"
        elif weighted_sum < 0 and confidence >= 0.3:
            action = "sell"
        else:
            action = "hold"

        return FusionResult(
            score=round(weighted_sum, 3),
            action=action,
            confidence=round(confidence, 3),
            indicator_votes=votes,
            reasons=[f"{name}: {reason}" for name, reason in reasons],
        )

    def _safe(self, row, *keys):
        for k in keys:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return v
        return 0

    def _vote_macd(self, prev, latest) -> int:
        macd = self._safe(latest, "MACD")
        sig = self._safe(latest, "MACD_signal")
        prev_macd = self._safe(prev, "MACD")
        prev_sig = self._safe(prev, "MACD_signal")
        if prev_macd < prev_sig and macd >= sig:
            return 1
        if prev_macd > prev_sig and macd <= sig:
            return -1
        return 0

    def _vote_rsi(self, latest) -> int:
        rsi = self._safe(latest, "RSI14")
        if rsi < 30:
            return 1
        if rsi > 70:
            return -1
        return 0

    def _vote_ma(self, latest) -> int:
        ma5 = self._safe(latest, "MA5")
        ma10 = self._safe(latest, "MA10")
        ma20 = self._safe(latest, "MA20")
        if ma5 and ma10 and ma20:
            if ma5 > ma20 and ma5 > ma10:
                return 1
            if ma5 < ma20 and ma5 < ma10:
                return -1
        return 0

    def _vote_kdj(self, prev, latest) -> int:
        k = self._safe(latest, "KDJ_K")
        d = self._safe(latest, "KDJ_D")
        prev_k = self._safe(prev, "KDJ_K")
        prev_d = self._safe(prev, "KDJ_D")
        if prev_k < prev_d and k >= d and k < 30:
            return 1
        if prev_k > prev_d and k <= d and k > 70:
            return -1
        return 0

    def _vote_bb(self, latest) -> int:
        close = self._safe(latest, "close")
        lower = self._safe(latest, "BB_lower")
        upper = self._safe(latest, "BB_upper")
        if close and lower and close <= lower:
            return 1
        if close and upper and close >= upper:
            return -1
        return 0

    def _vote_volume(self, latest) -> int:
        vol = self._safe(latest, "volume")
        vol_ma20 = self._safe(latest, "VOL_MA20")
        pct = self._safe(latest, "pct_chg")
        if vol and vol_ma20 and vol > 1.5 * vol_ma20:
            if pct > 1:
                return 1
            if pct < -1:
                return -1
        return 0


# ══════════════════════════════════════════════
# ML 技术指标融合 (LightGBM 替代固定权重)
# ══════════════════════════════════════════════

def _extract_ml_features(df):
    """从已附指标的 DataFrame 提取 ML 模型输入特征 (13维)"""
    import numpy as np
    latest = df.iloc[-1]
    features = {}

    def _v(key, default=0.0):
        val = latest.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    features["MACD_DIF"] = _v("MACD")
    features["MACD_signal"] = _v("MACD_signal")
    features["MACD_hist"] = features["MACD_DIF"] - features["MACD_signal"]
    features["RSI14"] = _v("RSI14", 50)
    features["MA5"] = _v("MA5")
    features["MA20"] = _v("MA20")
    features["ma_ratio"] = features["MA5"] / max(features["MA20"], 0.01) - 1.0
    features["KDJ_K"] = _v("KDJ_K", 50)
    features["KDJ_D"] = _v("KDJ_D", 50)
    features["KDJ_diff"] = features["KDJ_K"] - features["KDJ_D"]

    bb_upper = _v("BB_upper")
    bb_lower = _v("BB_lower")
    close = _v("close")
    bb_range = bb_upper - bb_lower
    features["BB_position"] = (close - bb_lower) / max(bb_range, 0.01) if bb_range > 0 else 0.5

    vol = _v("volume")
    vol_ma20 = _v("VOL_MA20")
    features["vol_ratio"] = vol / max(vol_ma20, 1) if vol_ma20 > 0 else 1.0
    features["pct_chg"] = _v("pct_chg")

    return list(features.values()), list(features.keys())


def train_ml_fusion(n_stocks=300, random_state=42):
    """批量训练 LightGBM 技术指标融合模型

    随机选 n_stocks 只股票 → 批量拉全量日线 → 批量算指标 → 批量取标签
    保存到 data/ml_tech_fusion.pkl
    """
    import numpy as np
    import lightgbm as lgb
    import pickle
    import os
    import datetime as _dt

    from core.data import Database
    if not Database.is_available():
        logger.warning("[ML融合] 数据库不可用")
        return None

    # 1. 随机选股 (有足够数据的)
    stock_rows = Database.fetchall(
        "SELECT code FROM daily_quotes GROUP BY code "
        "HAVING COUNT(*) >= 120 ORDER BY RAND() LIMIT %s",
        [n_stocks],
    )
    if not stock_rows:
        logger.warning("[ML融合] 无候选股票")
        return None

    codes = [r["code"] for r in stock_rows]
    logger.info(f"[ML融合] 候选 {len(codes)} 只, 批量拉数据...")

    # 2. 批量拉取全部日线
    placeholders = ",".join(["%s"] * len(codes))
    rows = Database.fetchall(
        f"SELECT code, trade_date, open, high, low, close, volume, amount "
        f"FROM daily_quotes WHERE code IN ({placeholders}) "
        f"ORDER BY code, trade_date",
        codes,
    )
    if not rows:
        return None

    import pandas as pd
    df_all = pd.DataFrame(rows)
    df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
    df_all["close"] = df_all["close"].astype(float)
    df_all["high"] = df_all["high"].astype(float)
    df_all["low"] = df_all["low"].astype(float)
    df_all["volume"] = df_all["volume"].astype(float)

    # 3. 批量算标签: LEAD(close, 5) 按 code 分组
    df_all["fwd_close"] = df_all.groupby("code")["close"].transform(lambda x: x.shift(-5))
    df_all["label"] = (df_all["fwd_close"] > df_all["close"]).astype(int)
    df_all = df_all.dropna(subset=["fwd_close"])

    # 4. 逐 code 批量计算指标 → 构建训练集
    from core.domain.screening.indicators import Analyzer
    X_list, y_list = [], []
    processed = 0

    for code, grp in df_all.groupby("code"):
        if len(grp) < 100:
            continue
        try:
            grp = grp.sort_values("trade_date")
            # 计算技术指标
            df_ind = Analyzer.add_indicators(grp.reset_index(drop=True))

            # 对每一行 (排除最后5行因为没有标签) 提取特征
            valid_idx = df_ind["fwd_close"].notna()
            for i in df_ind[valid_idx].index:
                if i < 60:  # 需要足够历史算指标
                    continue
                row_df = df_ind.iloc[:i+1]
                feats, _ = _extract_ml_features(row_df)
                if any(abs(f) > 1e8 for f in feats):
                    continue
                X_list.append(feats)
                y_list.append(int(df_ind.at[i, "label"]))

            processed += 1
            if processed % 50 == 0:
                logger.info(f"[ML融合] 已处理 {processed}/{len(df_all['code'].unique())} 只, "
                            f"样本 {len(X_list)}")
        except Exception:
            continue

    if len(X_list) < 500:
        logger.warning(f"[ML融合] 样本不足: {len(X_list)}")
        return None

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list)
    logger.info(f"[ML融合] 训练样本: {len(X_list)}, 正例率: {y.mean():.2%}")

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, num_leaves=16,
        learning_rate=0.05, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, verbose=-1,
    )
    model.fit(X, y)

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "ml_tech_fusion.pkl")
    from core.shared.persistence import save_pickle, today_date
    save_pickle({
        "model": model,
        "trained_at": str(today_date()),
        "n_samples": len(X_list),
    }, path)
    logger.info(f"[ML融合] 模型已保存: {path} (n={len(X_list)})")
    return model


def _load_ml_model():
    """加载已训练的 ML 融合模型"""
    import os
    from core.shared.persistence import load_pickle
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "ml_tech_fusion.pkl")
    data = load_pickle(path)
    if data is None:
        return None
    return data.get("model")


class MLSignalFusion:
    """ML 驱动的技术指标融合 — 接口兼容 SignalFusion"""

    def __init__(self, weights=None):
        self._model = _load_ml_model()
        self._fallback = SignalFusion(weights) if not self._model else None

    @property
    def is_ready(self):
        return self._model is not None

    def evaluate(self, df):
        """ML 预测 [-1, 1] 技术分, 失败时回退规则模型"""
        import numpy as np

        if self._model is not None:
            try:
                feats, names = _extract_ml_features(df)
                X = np.array([feats], dtype=np.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob = self._model.predict_proba(X)[0, 1]
                score = round(float(np.tanh((prob - 0.5) * 3)), 3)

                # 提取原因 (特征重要性 top-3)
                reasons = []
                try:
                    importances = self._model.feature_importances_
                    idx = np.argsort(-importances)[:3]
                    for i in idx:
                        reasons.append(f"{names[i]}={feats[i]:.3f}")
                except Exception:
                    pass

                action = "buy" if score > 0.15 else ("sell" if score < -0.15 else "hold")
                confidence = min(abs(score) * 2, 1.0)

                return FusionResult(
                    score=score, action=action, confidence=round(confidence, 3),
                    indicator_votes={"ml_score": score}, reasons=reasons,
                )
            except Exception as e:
                logger.debug(f"[ML融合] 预测失败, 回退规则: {e}")

        if self._fallback:
            return self._fallback.evaluate(df)
        return FusionResult()

    def ensure_trained(self):
        """确保模型可用, 不可用时触发训练 (阻塞)"""
        if self._model is not None:
            return True
        model = train_ml_fusion()
        if model is not None:
            self._model = model
            self._fallback = None
            return True
        return False


def get_fusion(weights=None):
    """工厂函数: 根据配置返回 ML 或规则融合器"""
    from core.shared import config
    if getattr(config, "SIGNAL_FUSION_ML_ENABLED", False):
        ml = MLSignalFusion(weights)
        if ml.is_ready:
            return ml
    return SignalFusion(weights)
