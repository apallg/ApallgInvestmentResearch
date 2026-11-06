"""
市场状态识别器 — 基于沪深300指数技术指标判断当前市场处于哪种状态
四种状态: trending_up(趋势上涨) / trending_down(趋势下跌) / ranging(震荡) / volatile(高波动)
"""
import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd
import numpy as np

from core.shared.config import REGIME_DEFAULT_INDEX, REGIME_LOOKBACK_DAYS, REGIME_VOLATILE_PERCENTILE
from core.shared.persistence import today_date
from core.domain.regime.types import Regime, HMM_TO_LEGACY

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    regime: Regime
    confidence: float
    sub_scores: dict
    details: dict
    as_of_date: str

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "sub_scores": self.sub_scores,
            "details": self.details,
            "as_of_date": self.as_of_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeResult":
        return cls(
            regime=Regime(d["regime"]),
            confidence=d["confidence"],
            sub_scores=d["sub_scores"],
            details=d.get("details", ""),
            as_of_date=d.get("as_of_date", ""),
        )


class MarketRegimeClassifier:
    def __init__(self, index_name=None, lookback_days=None):
        self.index_name = index_name or REGIME_DEFAULT_INDEX
        self.lookback_days = lookback_days or REGIME_LOOKBACK_DAYS
        self._last_regime = None
        self._last_confidence = 0.0

    def classify(self, as_of=None):
        """判断当前市场状态。as_of 为 None 时取最新，传 date 时用于回测"""
        from core.data.fetcher import DataFetcher

        fetcher = DataFetcher()
        df = fetcher.get_index_daily(self.index_name, days=self.lookback_days)
        if df is None or df.empty:
            return RegimeResult(
                regime=Regime.STEADY_FLAT, confidence=0.0,
                sub_scores={}, details={"error": "无法获取指数数据"},
                as_of_date=str(as_of or today_date()),
            )

        if as_of is not None:
            date_col = "trade_date" if "trade_date" in df.columns else "date"
            df = df[df[date_col] <= pd.Timestamp(as_of)]
            if df.empty:
                return RegimeResult(
                    regime=Regime.STEADY_FLAT, confidence=0.0,
                    sub_scores={}, details={"error": f"日期{as_of}前无数据"},
                    as_of_date=str(as_of),
                )

        return self._classify_df(df)

    def classify_from_df(self, df):
        """从预加载的DataFrame判断状态，回测用"""
        if df is None or df.empty:
            return RegimeResult(
                regime=Regime.STEADY_FLAT, confidence=0.0,
                sub_scores={}, details={"error": "空数据"},
                as_of_date="",
            )
        return self._classify_df(df)

    def _classify_df(self, df):
        df = df.copy()
        if "trade_date" not in df.columns and "date" in df.columns:
            df["trade_date"] = df["date"]
        if len(df) < 60:
            return self._classify_short(df)

        df["trade_date"] = pd.to_datetime(df["trade_date"])

        from core.domain.screening.indicators import Analyzer
        try:
            df = Analyzer.add_regime_indicators(df)
        except Exception:
            pass

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        adx_val = self._compute_adx(high, low, close)

        details = {
            "index": self.index_name,
            "latest_close": float(close.iloc[-1]),
            "adx": round(adx_val, 2),
            "n_days": len(df),
        }

        s_up, d_up = self._score_trending_up(df, adx_val)
        s_down, d_down = self._score_trending_down(df)
        s_range, d_range = self._score_ranging(df, adx_val)
        s_vol, d_vol = self._score_volatile(df)

        sub_scores = {
            "steady_up": round(s_up, 3),
            "steady_down": round(s_down, 3),
            "steady_flat": round(s_range, 3),
            "volatile_flat": round(s_vol, 3),
        }
        details.update({f"details_{k}": v for k, v in
                        [("up", d_up), ("down", d_down), ("range", d_range), ("vol", d_vol)]})

        scores = [
            (Regime.STEADY_UP, s_up),
            (Regime.STEADY_DOWN, s_down),
            (Regime.STEADY_FLAT, s_range),
            (Regime.VOLATILE_FLAT, s_vol),
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        winner, winner_score = scores[0]
        total = max(sum(s[1] for s in scores), 0.001)
        confidence = round(winner_score / total, 3)

        winner, confidence = self._apply_hysteresis(winner, confidence)

        as_of = str(df["trade_date"].iloc[-1].date())

        # numpy类型 → Python原生类型,确保JSON可序列化
        def _native(v):
            if isinstance(v, (np.bool_,)): return bool(v)
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, dict): return {k: _native(vv) for k, vv in v.items()}
            return v

        return RegimeResult(
            regime=winner, confidence=float(confidence),
            sub_scores={k: float(v) for k, v in sub_scores.items()},
            details={k: _native(v) for k, v in details.items()},
            as_of_date=as_of,
        )

    def _classify_short(self, df):
        """数据不足60天时的简化判断"""
        close = df["close"].astype(float)
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        if len(close) < 20:
            return RegimeResult(Regime.STEADY_FLAT, 0.0,
                              {"steady_up": 0, "steady_down": 0, "steady_flat": 0.5, "volatile_flat": 0.5},
                              {"error": "数据不足20天", "n_days": len(df)},
                              str(df["trade_date"].iloc[-1].date()) if "trade_date" in df.columns else "")
        if ma5.iloc[-1] > ma20.iloc[-1]:
            return RegimeResult(Regime.STEADY_UP, 0.5,
                              {"steady_up": 0.5, "steady_down": 0, "steady_flat": 0.3, "volatile_flat": 0.2},
                              {"n_days": len(df), "note": "简化判断"},
                              str(df["trade_date"].iloc[-1].date()))
        else:
            return RegimeResult(Regime.STEADY_DOWN, 0.5,
                              {"steady_up": 0, "steady_down": 0.5, "steady_flat": 0.3, "volatile_flat": 0.2},
                              {"n_days": len(df), "note": "简化判断"},
                              str(df["trade_date"].iloc[-1].date()))

    def _score_trending_up(self, df, adx_val):
        close = df["close"].astype(float)
        ma5 = df.get("MA5", close.rolling(5).mean())
        ma10 = df.get("MA10", close.rolling(10).mean())
        ma20 = df.get("MA20", close.rolling(20).mean())
        ma60 = df.get("MA60", close.rolling(60).mean())

        bull_alignment = (ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1])
        n_bull = sum(1 for i in range(max(0, len(df)-10), len(df))
                     if ma5.iloc[i] > ma10.iloc[i] > ma20.iloc[i] > ma60.iloc[i])

        above_ma20 = close.iloc[-1] > ma20.iloc[-1]
        above_ma60 = close.iloc[-1] > ma60.iloc[-1]
        adx_strong = adx_val > 25
        adx_moderate = adx_val > 20

        recent_return = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0

        score = 0.0
        if bull_alignment:
            score += 0.35
        if above_ma20:
            score += 0.15
        if above_ma60:
            score += 0.10
        if adx_strong:
            score += 0.20
        elif adx_moderate:
            score += 0.10
        if n_bull >= 6:
            score += 0.10
        if recent_return > 0.03:
            score += 0.10

        details = {"bull_alignment": bull_alignment, "above_ma20": above_ma20,
                   "above_ma60": above_ma60, "adx_strong": adx_strong,
                   "n_bull_days": n_bull, "recent_20d_return": round(recent_return * 100, 2)}
        return round(score, 3), details

    def _score_trending_down(self, df):
        close = df["close"].astype(float)
        ma5 = df.get("MA5", close.rolling(5).mean())
        ma10 = df.get("MA10", close.rolling(10).mean())
        ma20 = df.get("MA20", close.rolling(20).mean())
        ma60 = df.get("MA60", close.rolling(60).mean())

        bear_alignment = (ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1])
        n_bear = sum(1 for i in range(max(0, len(df)-10), len(df))
                     if ma5.iloc[i] < ma10.iloc[i] < ma20.iloc[i] < ma60.iloc[i])
        below_ma60 = close.iloc[-1] < ma60.iloc[-1]

        recent_return = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0

        score = 0.0
        if bear_alignment:
            score += 0.40
        if below_ma60:
            score += 0.25
        if n_bear >= 6:
            score += 0.15
        if recent_return < -0.05:
            score += 0.20
        elif recent_return < -0.02:
            score += 0.10

        details = {"bear_alignment": bear_alignment, "below_ma60": below_ma60,
                   "n_bear_days": n_bear, "recent_20d_return": round(recent_return * 100, 2)}
        return round(score, 3), details

    def _score_ranging(self, df, adx_val):
        close = df["close"].astype(float)
        if "BB_upper" not in df.columns or "BB_lower" not in df.columns:
            return 0.0, {"error": "无布林带数据"}

        bb_upper = df["BB_upper"].astype(float)
        bb_lower = df["BB_lower"].astype(float)
        bb_width = (bb_upper - bb_lower) / close

        hist_width = bb_width.iloc[:-20] if len(bb_width) > 20 else bb_width
        current_width = bb_width.iloc[-1]
        hist_median = hist_width.median() if len(hist_width) > 0 else current_width

        touches = 0
        for i in range(max(0, len(df)-40), len(df)):
            if close.iloc[i] >= bb_upper.iloc[i] * 0.985:
                touches += 1
            if close.iloc[i] <= bb_lower.iloc[i] * 1.015:
                touches += 1

        adx_low = adx_val < 20
        width_narrow = current_width < hist_median * 0.6 if hist_median > 0 else False

        score = 0.0
        if width_narrow:
            score += 0.35
        if adx_low:
            score += 0.30
        touch_factor = min(touches / 4, 1.0)
        score += 0.20 * touch_factor
        ma20_val = df.get("MA20", close.rolling(20).mean()).iloc[-1]
        if abs(close.iloc[-1] - ma20_val) / ma20_val < 0.02:
            score += 0.10

        details = {"bb_width_pct": round(current_width * 100, 2),
                   "hist_median_width_pct": round(hist_median * 100, 2),
                   "width_narrow": width_narrow, "adx_low": adx_low,
                   "touches_40d": touches}
        return round(min(score, 1.0), 3), details

    def _score_volatile(self, df):
        close = df["close"].astype(float)
        returns = close.pct_change(fill_method=None).dropna()
        if len(returns) < 20:
            return 0.0, {"error": "数据不足"}

        vol_20d = returns.iloc[-20:].std() * (250 ** 0.5)

        # 历史滚动波动率: shift(1) 使窗口右端点为 i-1 (不含当前), 与原 for 循环语义一致
        rolling_vol = (returns.rolling(20).std().shift(1) * (250 ** 0.5)).dropna()
        if rolling_vol.empty:
            return 0.0, {"vol_20d_annual": round(vol_20d * 100, 1)}

        pct = REGIME_VOLATILE_PERCENTILE
        percentile_75 = np.percentile(rolling_vol, pct * 100)

        raw = vol_20d / max(percentile_75, 0.001) - 1.0
        score = max(0.0, min(1.0, raw))

        atr_ratio = 0.0
        if "ATR14" in df.columns:
            atr = df["ATR14"].astype(float)
            atr_ratio = atr.iloc[-1] / max(close.iloc[-1], 0.01)
        elif "MA5" in df.columns:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            tr = pd.concat([high - low,
                           (high - close.shift()).abs(),
                           (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_ratio = atr.iloc[-1] / max(close.iloc[-1], 0.01)

        if atr_ratio > 0.03:
            score = min(1.0, score + 0.15)

        details = {"vol_20d_annual": round(vol_20d * 100, 1),
                   "vol_pct": round(vol_20d / max(percentile_75, 0.001) * 100, 1),
                   "above_75pct": vol_20d > percentile_75,
                   "atr_ratio_pct": round(atr_ratio * 100, 2)}
        return round(score, 3), details

    def _compute_adx(self, high, low, close, period=14):
        try:
            from ta.trend import ADXIndicator
            adx_ind = ADXIndicator(high=high, low=low, close=close, window=period)
            adx = adx_ind.adx()
            return float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0
        except Exception:
            return self._compute_adx_fallback(high, low, close, period)

    def _compute_adx_fallback(self, high, low, close, period=14):
        """手动计算ADX,不依赖ta库的ADXIndicator"""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
        minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(period).mean()
        return float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0

    def _apply_hysteresis(self, winner, confidence):
        if self._last_regime is None:
            self._last_regime = winner
            self._last_confidence = confidence
            return winner, confidence

        if self._last_regime == winner:
            self._last_confidence = confidence
            return winner, confidence

        if self._last_confidence > 0.7 and confidence < 0.5:
            return self._last_regime, self._last_confidence

        self._last_regime = winner
        self._last_confidence = confidence
        return winner, confidence

    def get_rolling_regime_series(self, df=None, step=5):
        """预计算全时间段状态序列，回测用。df 为 None 时自动获取指数数据"""
        if df is None:
            from core.data.fetcher import DataFetcher
            df = DataFetcher().get_index_daily(self.index_name, days=self.lookback_days)
            if df is None or df.empty:
                return pd.DataFrame()

        df = df.copy()
        if "trade_date" not in df.columns and "date" in df.columns:
            df["trade_date"] = df["date"]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)

        from core.domain.screening.indicators import Analyzer
        try:
            df = Analyzer.add_regime_indicators(df)
        except Exception:
            pass

        rows = []
        min_len = 60
        for i in range(min_len, len(df), step):
            window = df.iloc[:i+1]
            result = self.classify_from_df(window)
            rows.append({
                "date": window["trade_date"].iloc[-1].date(),
                "regime": result.regime.value,
                "confidence": result.confidence,
                "steady_up": result.sub_scores.get("steady_up", 0),
                "steady_down": result.sub_scores.get("steady_down", 0),
                "steady_flat": result.sub_scores.get("steady_flat", 0),
                "volatile_flat": result.sub_scores.get("volatile_flat", 0),
            })

        return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# HMM 市场状态分类器 (6 隐状态)
# ══════════════════════════════════════════════

class HMMRegimeClassifier:
    """隐马尔可夫模型市场状态识别 (已弃用, 保留用于 _get_regime_features 生成 HMM 特征)

    6 个隐状态由数据自动推断，不依赖人工规则。
    特征: 5d/20d 收益、20d 波动率、量比、振幅、宽度
    """
    def __init__(self, index_name=None, n_states=6):
        self.index_name = index_name or REGIME_DEFAULT_INDEX
        self.n_states = n_states
        self._model = None
        self._state_map = None
        self._last_state = None
        self._trained = False
        self._train_date = None
        self._retraining = False

    def _enrich_features(self, df):
        """用市场宽度和北向资金数据丰富指数日线 DataFrame。
        失败时静默回退，不阻塞主流程。
        """
        from core.shared.config import REGIME_HMM_EXTRA_FEATURES
        if not REGIME_HMM_EXTRA_FEATURES:
            return df

        df = df.copy()
        if "trade_date" not in df.columns:
            return df

        dates = pd.to_datetime(df["trade_date"])
        start = dates.min().strftime("%Y-%m-%d")
        end = dates.max().strftime("%Y-%m-%d")

        # 市场宽度
        try:
            from core.domain.timing.breadth import get_breadth_history
            breadth_df = get_breadth_history(start, end)
            if not breadth_df.empty:
                breadth_df = breadth_df.reset_index()
                breadth_df["trade_date"] = pd.to_datetime(breadth_df["trade_date"])
                df["trade_date_dt"] = dates
                df = df.merge(breadth_df[["trade_date", "up_ratio", "pct_median"]],
                              on="trade_date", how="left")
                df.drop(columns=["trade_date_dt"], inplace=True, errors="ignore")
        except Exception as e:
            logger.warning(f"[HMM] 市场宽度获取失败: {e}")

        # 北向资金
        try:
            from core.data import Database
            rows = Database.fetchall(
                "SELECT trade_date, north_net FROM north_flow_summary "
                "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date",
                (start, end)
            )
            if rows:
                nf = pd.DataFrame(rows, columns=["trade_date", "north_net"])
                nf["trade_date"] = pd.to_datetime(nf["trade_date"])
                df["trade_date_dt"] = dates
                df = df.merge(nf, on="trade_date", how="left")
                df.drop(columns=["trade_date_dt"], inplace=True, errors="ignore")
        except Exception as e:
            logger.warning(f"[HMM] 北向资金获取失败: {e}")

        return df

    def _extract_features(self, df):
        """从指数日线提取 HMM 输入特征"""
        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(1e8, index=df.index)

        features = pd.DataFrame(index=df.index)

        rets = close.pct_change(fill_method=None)
        features["ret_5d"] = close.pct_change(5, fill_method=None).fillna(0)
        features["ret_20d"] = close.pct_change(20, fill_method=None).fillna(0)
        features["vol_20d"] = rets.rolling(20).std().fillna(0)
        features["vol_ratio"] = (features["vol_20d"] /
                                 features["vol_20d"].rolling(60).mean().fillna(features["vol_20d"]))
        features["vol_ratio"] = features["vol_ratio"].fillna(1.0)

        vol_ma20 = volume.rolling(20).mean().fillna(volume)
        features["volume_ratio"] = (volume / vol_ma20).fillna(1.0).clip(0.1, 5.0)

        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close
        features["amplitude"] = ((high - low) / close).fillna(0).clip(0, 0.2)

        features["ma_dev_20d"] = ((close - close.rolling(20).mean()) /
                                  close.rolling(20).mean()).fillna(0)

        # 扩展特征: 市场宽度 + 北向资金 (REGIME_HMM_EXTRA_FEATURES=false 时跳过)
        from core.shared.config import REGIME_HMM_EXTRA_FEATURES
        if REGIME_HMM_EXTRA_FEATURES:
            features["up_ratio"] = pd.to_numeric(df.get("up_ratio", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)
            features["pct_median"] = pd.to_numeric(df.get("pct_median", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
            features["north_flow_net"] = pd.to_numeric(df.get("north_net", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

        return features.dropna()

    def fit(self, df=None, random_state=42):
        """训练 HMM 模型。df 为 None 时自动拉取沪深300历史数据"""
        from hmmlearn import hmm
        import numpy as np

        if df is None:
            from core.data.fetcher import DataFetcher
            df = DataFetcher().get_index_daily(self.index_name, days=2000)

        if df is None or df.empty:
            logger.warning("[HMM] 无训练数据")
            return False

        df = self._enrich_features(df)
        features = self._extract_features(df)
        if len(features) < 252:
            logger.warning(f"[HMM] 样本不足: {len(features)} < 252")
            return False

        X = features.values.astype(np.float64)
        # 标准化
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-8
        X = (X - self._feat_mean) / self._feat_std

        def _try_fit(n_comp):
            for cov_type in ("diag", "full"):
                try:
                    m = hmm.GaussianHMM(
                        n_components=n_comp,
                        covariance_type=cov_type,
                        n_iter=200,
                        random_state=random_state,
                        tol=1e-3,
                    )
                    m.fit(X)
                    if np.any(np.isnan(m.startprob_)) or np.any(np.isnan(m.transmat_)):
                        continue
                    return m
                except Exception as e:
                    logger.warning(f"[HMM] {cov_type}/{n_comp}state 训练失败: {e}")
                    continue
            return None

        for n in [self.n_states, 4, 3]:
            self._model = _try_fit(n)
            if self._model is not None:
                if n != self.n_states:
                    logger.info(f"[HMM] {self.n_states}状态不稳定, 降为{n}状态")
                    self.n_states = n
                break

        if self._model is None:
            logger.error(f"[HMM] 所有尝试均失败")
            return False

        # 为每个隐状态标注经济含义
        states = self._model.predict(X)
        self._label_states(features, states, X)

        self._trained = True
        self._train_date = today_date()
        logger.info(f"[HMM] 训练完成, {self.n_states} 个状态, "
                    f"样本 {len(X)}, 对数似然 {self._model.score(X):.1f}")
        self._log_state_summary(features, states)
        return True

    def _label_states(self, features, states, X):
        """根据各状态的平均特征 (收益 × 波动), 映射到经济含义"""
        state_stats = {}
        for s in range(self.n_states):
            mask = states == s
            count = int(mask.sum())
            if count == 0:
                state_stats[s] = {"count": 0, "ret_20d": 0.0, "vol_20d": 0.0}
            else:
                state_stats[s] = {
                    "count": count,
                    "ret_20d": float(features["ret_20d"].values[mask].mean()),
                    "vol_20d": float(features["vol_20d"].values[mask].mean()),
                }

        if len(state_stats) < 2:
            self._state_map = {s: Regime.STEADY_FLAT for s in state_stats}
            return

        rets = np.array([v["ret_20d"] for v in state_stats.values()])
        vols = np.array([v["vol_20d"] for v in state_stats.values()])
        ret_med = np.median(rets)
        vol_med = np.median(vols)
        vol_extreme = vol_med * 2.0

        mapping = {}
        for s_id, stats in state_stats.items():
            ret = stats["ret_20d"]
            vol = stats["vol_20d"]
            count = stats["count"]

            if count < 5 or (vol > vol_extreme and abs(ret) > 0.05):
                mapping[s_id] = Regime.VOLATILE_FLAT
            elif ret > ret_med + 0.005 and vol <= vol_med:
                mapping[s_id] = Regime.STEADY_UP
            elif ret > ret_med:
                mapping[s_id] = Regime.VOLATILE_UP
            elif ret < ret_med - 0.005 and vol <= vol_med:
                mapping[s_id] = Regime.STEADY_DOWN
            elif ret < ret_med:
                mapping[s_id] = Regime.VOLATILE_DOWN
            else:
                mapping[s_id] = Regime.STEADY_FLAT

        # 去重: 确保每个标签只用一次, 重复的分配次优标签
        used = set()
        final = {}
        # 按 |ret| 排序, 小波动优先分配标签
        for s_id, _ in sorted(mapping.items(), key=lambda x: abs(state_stats[x[0]]["ret_20d"])):
            label = mapping[s_id]
            if label not in used:
                final[s_id] = label
                used.add(label)
            else:
                # 分配下一个可用的标签
                fallback = [Regime.STEADY_FLAT, Regime.VOLATILE_UP, Regime.VOLATILE_DOWN,
                           Regime.STEADY_UP, Regime.STEADY_DOWN, Regime.VOLATILE_FLAT]
                for fb in fallback:
                    if fb not in used:
                        final[s_id] = fb
                        used.add(fb)
                        break
                else:
                    final[s_id] = Regime.STEADY_FLAT

        # 确保所有 n_states 都被映射 (极小状态也可能被 predict 输出)
        for s in range(self.n_states):
            if s not in final:
                final[s] = Regime.STEADY_FLAT

        self._state_map = final

    def _log_state_summary(self, features, states):
        if self._state_map is None:
            return
        for state_id, regime in sorted(self._state_map.items(), key=lambda x: x[1].value):
            mask = states == state_id
            n = mask.sum()
            if n == 0:
                logger.info(f"  [{regime.value}] n=   0  (0%)")
                continue
            ret = features["ret_20d"].values[mask].mean()
            vol = features["vol_20d"].values[mask].mean()
            logger.info(f"  [{regime.value}] n={n:>4}  avg_ret_20d={ret:>+.4f}  "
                        f"avg_vol={vol:.4f}  ({n/len(states)*100:.0f}%)")

    def predict(self, df):
        """预测当前市场状态 (自动检测模型过期并后台重训)"""
        import numpy as np
        from core.shared.infrastructure.thread_pool import thread_pool

        if not self._trained or self._model is None:
            return RegimeResult(
                regime=Regime.STEADY_FLAT, confidence=0.0,
                sub_scores={}, details={"error": "HMM未训练"},
                as_of_date=str(today_date()),
            )

        # 检查模型是否过期, 过期则后台异步重训 (不阻塞本次预测)
        from core.shared.config import REGIME_HMM_RETRAIN_DAYS as _max_days
        if self._train_date and not self._retraining:
            age = (today_date() - self._train_date).days
            if age > _max_days:
                logger.info(f"[HMM] 模型已过期 ({age}天 > {_max_days}天), 触发后台重训...")
                self._retraining = True

                def _retrain():
                    try:
                        success = self.fit()
                        if success:
                            self.save()
                            logger.info("[HMM] 后台重训完成")
                    except Exception as e:
                        logger.warning(f"[HMM] 后台重训失败: {e}")
                    finally:
                        self._retraining = False

                thread_pool.light.submit(_retrain)

        df = self._enrich_features(df)
        features = self._extract_features(df)
        if features.empty:
            return RegimeResult(
                regime=Regime.STEADY_FLAT, confidence=0.0,
                sub_scores={}, details={"error": "特征提取失败"},
                as_of_date=str(df["trade_date"].iloc[-1].date())
                if "trade_date" in df.columns else str(today_date()),
            )

        X = features.values.astype(np.float64)

        # 特征维度变化检测 (旧模型7维 vs 新模型10维)
        if X.shape[1] != len(self._feat_mean):
            logger.warning(f"[HMM] 特征维度不匹配 (模型{len(self._feat_mean)}维 vs 当前{X.shape[1]}维), 触发重训")
            if not self._retraining:
                self._retraining = True
                def _retrain_dim():
                    try:
                        self.fit()
                        self.save()
                        logger.info("[HMM] 维度变更重训完成")
                    except Exception as e:
                        logger.warning(f"[HMM] 维度变更重训失败: {e}")
                    finally:
                        self._retraining = False
                thread_pool.light.submit(_retrain_dim)
            # 本次继续用旧维度特征 (裁剪到模型期望维度)
            min_dim = min(X.shape[1], len(self._feat_mean))
            X = X[:, :min_dim]
            self._feat_mean = self._feat_mean[:min_dim]
            self._feat_std = self._feat_std[:min_dim]

        X = (X - self._feat_mean) / self._feat_std

        # 获取最近 60 个观测的最可能状态
        n_recent = min(60, len(X))
        X_recent = X[-n_recent:]
        states = self._model.predict(X_recent)
        latest_state = states[-1]

        # 状态概率
        try:
            state_probs = self._model.predict_proba(X_recent)[-1]
        except Exception:
            state_probs = None

        regime = self._state_map.get(latest_state, Regime.STEADY_FLAT)

        # 置信度 = HMM 对当前观测的真实后验概率
        hmm_confidence = float(state_probs[latest_state]) if state_probs is not None else 0.0
        # 稳定性 = 最新状态在过去 60 天中的占比 (辅助指标)
        stability = (states == latest_state).sum() / n_recent

        # 构建 sub_scores (所有6个状态的概率)
        sub_scores = {}
        for s_id, reg in self._state_map.items():
            prob = float(state_probs[s_id]) if state_probs is not None else 0.0
            sub_scores[reg.value] = round(prob, 3)

        as_of = str(df["trade_date"].iloc[-1].date()) if "trade_date" in df.columns else str(today_date())

        # 转移概率矩阵 → 下一期状态预测
        next_state_probs = {}
        if state_probs is not None:
            transmat = self._model.transmat_
            next_probs = state_probs @ transmat
            for s_id, reg in self._state_map.items():
                next_state_probs[reg.value] = round(float(next_probs[s_id]), 3)

        details = {
            "method": "hmm",
            "n_states": self.n_states,
            "hmm_confidence": round(hmm_confidence, 3),
            "stability": round(stability, 3),
            "hmm_state_id": int(latest_state),
            "next_state_probs": next_state_probs,
        }

        return RegimeResult(
            regime=regime,
            confidence=round(hmm_confidence, 3),
            sub_scores=sub_scores,
            details=details,
            as_of_date=as_of,
        )

    def get_state_transition_matrix(self):
        """返回状态转移概率矩阵 (用于仓位调整)"""
        if self._model is None:
            return None
        transmat = self._model.transmat_
        labels = [self._state_map.get(i, Regime.STEADY_FLAT).value
                  for i in range(self.n_states)]
        return {"labels": labels, "matrix": transmat.tolist()}

    def save(self, path=None):
        """保存模型"""
        import os
        from core.shared.persistence import save_pickle
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "hmm_regime.pkl")
        save_pickle({
            "model": self._model,
            "state_map": self._state_map,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
            "n_states": self.n_states,
            "index_name": self.index_name,
            "train_date": self._train_date,
        }, path)
        logger.info(f"[HMM] 模型已保存: {path}")

    @classmethod
    def load(cls, path=None):
        """加载模型"""
        import os
        from core.shared.persistence import load_pickle
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "hmm_regime.pkl")
        data = load_pickle(path)
        if data is None:
            return None
        inst = cls(index_name=data["index_name"], n_states=data["n_states"])
        inst._model = data["model"]
        inst._state_map = data["state_map"]
        inst._feat_mean = data["feat_mean"]
        inst._feat_std = data["feat_std"]
        inst._trained = True
        inst._train_date = data.get("train_date")
        logger.info(f"[HMM] 模型已加载: {path}"
                    + (f" (训练于 {inst._train_date})" if inst._train_date else ""))
        return inst


# 全局 HMM 单例
_hmm_classifier = None
_hmm_lock = None


def _get_hmm_lock():
    global _hmm_lock
    if _hmm_lock is None:
        import threading
        _hmm_lock = threading.Lock()
    return _hmm_lock


def get_hmm_classifier():
    """获取/初始化 HMM 分类器 (懒加载 + 自动训练)"""
    global _hmm_classifier
    if _hmm_classifier is not None and _hmm_classifier._trained:
        return _hmm_classifier

    with _get_hmm_lock():
        if _hmm_classifier is not None and _hmm_classifier._trained:
            return _hmm_classifier

        _hmm_classifier = HMMRegimeClassifier.load()
        if _hmm_classifier is not None:
            return _hmm_classifier

        _hmm_classifier = HMMRegimeClassifier()
        success = _hmm_classifier.fit()
        if success:
            _hmm_classifier.save()
        return _hmm_classifier


_cache = {"result": None, "timestamp": 0}
REGIME_KEY = "regime:current"


def get_current_regime(ttl=300, apply_macro=True, smooth=True):
    """市场状态查询 — 统一入口 get_unified_regime

    缓存: 进程内 + Redis 双写, ttl 秒过期。
    失败时返回默认安全状态 (ranging), 不阻塞下游。
    apply_macro: 状态层面宏观修正开关 (前端可调)。
    smooth: 实时路径 CRF 序列平滑开关 (前端可调)。
    """
    import time
    from core.shared.infrastructure import redis_client as rds

    now = time.time()
    cache_key = f"regime:current:{apply_macro}:{smooth}"

    if _cache.get("key") == cache_key and _cache["result"] is not None and (now - _cache["timestamp"]) < ttl:
        return _cache["result"]

    raw = rds._get(cache_key)
    if raw is not None:
        try:
            result = RegimeResult.from_dict(raw)
            _cache["result"] = result
            _cache["key"] = cache_key
            _cache["timestamp"] = now
            return result
        except Exception:
            pass

    # ── 统一入口: 状态判定 (6 状态) + 仓位缩放 ──
    try:
        from core.domain.regime.unified import get_unified_regime
        rr = get_unified_regime(as_of=None, apply_macro=apply_macro, smooth=smooth)
    except Exception as e:
        logger.warning(f"[Regime] 统一入口失败: {e}")
        rr = None

    if rr is not None and rr.sub_scores:
        from core.domain.regime.predictor import _soft_position_ratio, _soft_signal_weights, STATE_NAMES
        top = rr.regime.value
        conf = rr.confidence
        probs = rr.sub_scores
        probs_arr = [probs.get(s, 0.0) for s in STATE_NAMES]
        position_base = _soft_position_ratio(probs_arr)
        weights = _soft_signal_weights(probs_arr)

        # 隐含波动率 → 仓位缩放 (6类概率 × 锚点波动率)
        ANCHOR_VOL = {
            "volatile_up": 0.225, "volatile_down": 0.209, "volatile_flat": 0.228,
            "steady_up": 0.112, "steady_down": 0.121, "steady_flat": 0.117,
        }
        implied_vol = sum(probs.get(s, 0) * ANCHOR_VOL.get(s, 0.17) for s in probs)
        vol_scalar = round(1.0 / max(1.0, implied_vol / 0.17), 4)
        vol_scalar = max(0.70, min(1.0, vol_scalar))

        # 预测波动率 (RegimePredictorV2) → 暴露 pred_vol 字段; V2 不可用时回退 implied_vol
        pred_vol = None
        try:
            from core.domain.regime.predictor_v2 import get_regime_predictor_v2
            _v2 = get_regime_predictor_v2()
            if _v2 is not None and _v2._is_loaded():
                _pr = _v2.predict_regime()
                if isinstance(_pr, dict) and _pr.get("pred_vol"):
                    pred_vol = float(_pr["pred_vol"])
        except Exception as e:
            logger.debug(f"[Regime] pred_vol 获取失败: {e}")
        if pred_vol is None:
            pred_vol = round(implied_vol, 4)

        # clamp 上限 1.00, 与回测引擎 min(1.00) 对齐 — 防止 config 缺失时回退默认 1.50 导致超仓
        position = round(max(0.15, min(1.00, position_base * vol_scalar)), 4)

        result = RegimeResult(
            regime=Regime(top),
            confidence=round(conf, 4),
            sub_scores=probs,
            details={
                "method": "unified",
                "model_type": rr.details.get("method", ""),
                "position_ratio": position,
                "signal_weights": weights,
                "position_base": position_base,
                "vol_scalar": vol_scalar,
                "implied_vol": round(implied_vol, 4),
                "pred_vol": round(pred_vol, 4),
                "macro": rr.details.get("macro"),
            },
            as_of_date=rr.as_of_date,
        )
        logger.info(f"[Regime] unified → {top} "
                   f"base={position_base:.2f} x vol={vol_scalar:.2f}(implied={implied_vol:.3f}) = {position:.2f}")
        _cache["result"] = result
        _cache["key"] = cache_key
        _cache["timestamp"] = now
        rds._set(cache_key, result.to_dict(), expire=ttl)
        return result

    # 安全默认值: 震荡 + 低置信度
    result = RegimeResult(
        regime=Regime.STEADY_FLAT, confidence=0.3,
        sub_scores={"steady_flat": 0.5, "steady_up": 0.2, "steady_down": 0.2, "volatile_flat": 0.1},
        details={"method": "fallback", "position_ratio": 0.5, "note": "RegimePredictor 不可用, 返回安全默认值"},
        as_of_date=str(today_date()),
    )
    logger.warning(f"[Regime] RegimePredictor 不可用, 返回安全默认值 ranging")
    return result


def get_regime_probs(apply_macro=True, smooth=True):
    """获取完整的市场状态预测

    Returns:
        dict | None: probs/position_ratio/signal_weights/pred_vol/vol_scalar
    """
    result = get_current_regime(apply_macro=apply_macro, smooth=smooth)
    if result is None:
        return None
    d = result.details
    return {
        "probs": result.sub_scores,
        "position_ratio": d.get("position_ratio"),
        "signal_weights": d.get("signal_weights"),
        "confidence": result.confidence,
        "top_state": result.regime.value,
        "model_type": d.get("model_type", ""),
        "as_of": result.as_of_date,
        "pred_vol": d.get("pred_vol"),
        "vol_scalar": d.get("vol_scalar"),
        "position_base": d.get("position_base"),
        "method": d.get("method"),
    }


class RuleBasedRegimeClassifier:
    """确定性规则分类器 — 不需要训练, 100% 可复现。

    规则: 20日收益方向 (up/down/flat) × 波动率分位数 (volatile/steady)
    - ret_20d > 0.02 → up;  ret_20d < -0.02 → down;  else flat
    - vol_20d 的 252 天分位数 > 0.7 → volatile;  else steady
    """

    DIR_THRESHOLD = 0.02
    VOL_PCT_THRESHOLD = 0.7

    SIX_STATES = ["volatile_up", "volatile_down", "volatile_flat",
                  "steady_up", "steady_down", "steady_flat"]

    def __init__(self, index_df=None):
        self._index_df = index_df

    def set_index_df(self, df):
        self._index_df = df

    def classify(self, as_of=None):
        df = self._index_df
        if df is None or df.empty:
            return RegimeResult(
                regime=Regime.STEADY_FLAT, confidence=0.0,
                sub_scores={}, details={"error": "无指数数据"},
                as_of_date=str(as_of or ""),
            )

        date_col = "trade_date" if "trade_date" in df.columns else "date"
        close_col = "close" if "close" in df.columns else None
        if close_col is None:
            return RegimeResult(
                regime=Regime.STEADY_FLAT, confidence=0.0,
                sub_scores={}, details={"error": "无收盘价列"},
                as_of_date=str(as_of or ""),
            )

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            df = df[df[date_col] <= as_of_ts]
        if len(df) < 252:
            logger.warning(f"[RuleBased] 数据不足252天 (仅{len(df)}行), 无法分类, as_of={as_of}")
            return RegimeResult(
                regime=Regime("steady_flat"), confidence=0.0,
                sub_scores={s: 0.0 for s in self.SIX_STATES},
                details={"method": "rule_based", "error": f"数据不足252天, 仅{len(df)}行"},
                as_of_date=str(as_of or ""),
            )

        df["ret"] = df[close_col].pct_change(fill_method=None)
        df["ret_20d"] = df[close_col].pct_change(20, fill_method=None)
        df["vol_20d"] = df["ret"].rolling(20).std() * np.sqrt(252)
        df["vol_pct"] = df["vol_20d"].rolling(252).rank(pct=True)

        latest = df.iloc[-1]
        raw_ret = latest.get("ret_20d", 0)
        raw_pct = latest.get("vol_pct", 0.5)
        ret = 0.0 if raw_ret is None or (isinstance(raw_ret, float) and np.isnan(raw_ret)) else (raw_ret or 0)
        pct = 0.5 if raw_pct is None or (isinstance(raw_pct, float) and np.isnan(raw_pct)) else (raw_pct or 0.5)

        vol_state = "volatile" if pct > self.VOL_PCT_THRESHOLD else "steady"
        if ret > self.DIR_THRESHOLD:
            dir_state = "up"
        elif ret < -self.DIR_THRESHOLD:
            dir_state = "down"
        else:
            dir_state = "flat"

        regime_str = f"{vol_state}_{dir_state}"
        regime = Regime(regime_str)
        confidence = max(abs(ret) * 5, abs(pct - 0.5) * 2)
        confidence = round(min(confidence, 0.99), 4)

        return RegimeResult(
            regime=regime, confidence=confidence,
            sub_scores={s: 1.0 if s == regime_str else 0.0 for s in self.SIX_STATES},
            details={
                "method": "rule_based",
                "ret_20d": round(float(ret), 6),
                "vol_20d": round(float(latest.get("vol_20d", 0) or 0), 6),
                "vol_pct": round(float(pct), 4),
                "direction": dir_state,
                "vol_state": vol_state,
            },
            as_of_date=str(as_of or ""),
        )

    def fit(self, index_df):
        self._index_df = index_df


def hot_inject_regime_cache(ttl=300):
    """预热 RegimePredictor 缓存 — 调用 get_current_regime() 写入 Redis"""
    try:
        result = get_current_regime(ttl=ttl)
        logger.info(f"[RegimeCache] 热注入完成 regime={result.regime.value} "
                    f"method={result.details.get('method','?')} ttl={ttl}s")
    except Exception as e:
        logger.error(f"[RegimeCache] 热注入失败: {e}")
