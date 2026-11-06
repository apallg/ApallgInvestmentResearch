"""
IntradayPullbackDetector — 日内回调低点检测 (L3 纯计算)

职责: 接收 5分钟 OHLC DataFrame + 参考价位, 计算支撑位/回调信号/入场评分。
禁调: L1/L2/L4, DB, HTTP, datetime.now() / date.today_date() / time.time()。

数据流: L2 获取实时分时 kline → L3 纯计算 → IntradayResult
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


NEAR_SUPPORT_THRESHOLD = 0.015
RSI_OVERSOLD = 30
VOLUME_SHRINK_RATIO = 0.6
HAMMER_LOWER_SHADOW_RATIO = 2.0
HAMMER_UPPER_SHADOW_MAX = 0.1
VOLUME_PANIC_RATIO = 3.0
DEEP_DECLINE_THRESHOLD = 0.03


@dataclass
class SupportLevel:
    name: str
    price: float
    distance_pct: float


@dataclass
class PullbackSignal:
    type: str
    description: str
    strength: int


@dataclass
class IntradayResult:
    code: str
    current_price: float = 0.0
    open_price: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    supports: List[SupportLevel] = field(default_factory=list)
    nearest_support: Optional[SupportLevel] = None
    signals: List[PullbackSignal] = field(default_factory=list)
    entry_quality_score: float = 0.0
    entry_quality_label: str = ""
    risk_warning: Optional[str] = None
    recent_bars: List[dict] = field(default_factory=list)
    vwap: float = 0.0
    volume_ma: float = 0.0


class IntradayPullbackDetector:

    def analyze(self,
                df_5min,
                prev_day_low: float,
                prev_day_high: float,
                prev_day_close: float,
                reference_date: str,
                ma5: Optional[float] = None,
                ma10: Optional[float] = None,
                ma20: Optional[float] = None,
                limit_down_price: Optional[float] = None,
                ) -> IntradayResult:
        if not self._validate_data(df_5min):
            return self._fallback_analyze(
                current_price=prev_day_close,
                prev_day_low=prev_day_low,
                prev_day_high=prev_day_high,
                ma5=ma5,
                ma10=ma10,
                ma20=ma20,
                reference_date=reference_date)

        o = df_5min['open'].iloc[0]
        h = df_5min['high'].max()
        l = df_5min['low'].min()
        c = df_5min['close'].iloc[-1]

        vwap = self._calc_vwap(df_5min)
        supports = self._calc_supports(c, o, prev_day_low, vwap, l, ma5, ma10, ma20)
        nearest = supports[0] if supports else None

        rsi14 = self._calc_rsi(df_5min['close'], 14)

        vol_shrink, vol_ratio = self._detect_volume_shrink(df_5min)

        signals = []
        if nearest and nearest.distance_pct >= 0:
            if nearest.distance_pct <= NEAR_SUPPORT_THRESHOLD:
                signals.append(PullbackSignal(
                    type="near_support",
                    description=f"接近{nearest.name} ({nearest.distance_pct:.1%})",
                    strength=2))
        elif nearest and nearest.distance_pct < 0:
            signals.append(PullbackSignal(
                type="near_support",
                description=f"已跌破{nearest.name}",
                strength=1))

        if vol_shrink:
            signals.append(PullbackSignal(
                type="volume_shrink",
                description=f"量能萎缩至开盘均量{vol_ratio:.0%}",
                strength=2 if vol_ratio < 0.4 else 1))

        if rsi14 is not None and rsi14 < RSI_OVERSOLD:
            signals.append(PullbackSignal(
                type="rsi_oversold",
                description=f"RSI(14)={rsi14:.0f} 超卖",
                strength=2 if rsi14 < 20 else 1))

        hammer_bar = self._detect_hammer(df_5min)
        if hammer_bar is not None:
            signals.append(PullbackSignal(
                type="hammer",
                description=f"锤子线 ({hammer_bar})",
                strength=2 if hammer_bar == "最新K线" else 1))

        score, label = self._score_entry(
            supports, signals, c, vwap, vol_ratio if vol_shrink else 1.0, rsi14)

        risk = self._check_risk(
            df_5min, c, o, prev_day_high, prev_day_close, limit_down_price)

        bars = self._extract_recent_bars(df_5min, 48)
        vol_ma = df_5min['volume'].iloc[:6].mean() if len(df_5min) >= 6 else df_5min['volume'].mean()

        return IntradayResult(
            code="",
            current_price=round(c, 3),
            open_price=round(o, 3),
            day_high=round(h, 3),
            day_low=round(l, 3),
            supports=supports,
            nearest_support=nearest,
            signals=signals,
            entry_quality_score=round(score, 1),
            entry_quality_label=label,
            risk_warning=risk,
            recent_bars=bars,
            vwap=round(vwap, 3),
            volume_ma=round(vol_ma, 0))

    def _validate_data(self, df) -> bool:
        if df is None:
            return False
        try:
            if hasattr(df, 'empty') and df.empty:
                return False
            for col in ('open', 'high', 'low', 'close', 'volume'):
                if col not in df.columns:
                    return False
            return len(df) >= 1
        except Exception:
            return False

    def _fallback_analyze(self, current_price, prev_day_low, prev_day_high,
                          ma5, ma10, ma20, reference_date) -> IntradayResult:
        supports = self._calc_supports(
            current_price, None, prev_day_low, None, None, ma5, ma10, ma20)
        nearest = supports[0] if supports else None
        signals = []
        if nearest and 0 <= nearest.distance_pct <= NEAR_SUPPORT_THRESHOLD:
            signals.append(PullbackSignal(
                type="near_support",
                description=f"接近{nearest.name} ({nearest.distance_pct:.1%})",
                strength=1))
        score = min(50.0, 25.0 + (25.0 * (1 - min(nearest.distance_pct, 0.03) / 0.03)) if nearest else 25.0)
        label = "日K参考" if score < 30 else "一般"
        return IntradayResult(
            code="",
            current_price=round(current_price, 3),
            supports=supports,
            nearest_support=nearest,
            signals=signals,
            entry_quality_score=round(score, 1),
            entry_quality_label=label,
            # 缺日内数据不是日内风险: 置 None 避免 decide_execution 把它当风险否决买入
            risk_warning=None)

    def _calc_vwap(self, df) -> float:
        vol = df['volume'].values
        typ = ((df['high'].values + df['low'].values + df['close'].values) / 3.0)
        total_vol = vol.sum()
        if total_vol == 0:
            return df['close'].iloc[-1]
        return float((typ * vol).sum() / total_vol)

    def _calc_supports(self, current_price, open_price, prev_day_low, vwap,
                       day_low, ma5, ma10, ma20) -> List[SupportLevel]:
        levels = []
        if prev_day_low and prev_day_low > 0:
            levels.append(SupportLevel(
                name="前日低点",
                price=round(prev_day_low, 3),
                distance_pct=round((current_price - prev_day_low) / prev_day_low, 4)))
        if open_price and open_price > 0:
            levels.append(SupportLevel(
                name="开盘价",
                price=round(open_price, 3),
                distance_pct=round((current_price - open_price) / open_price, 4)))
        if vwap and vwap > 0:
            levels.append(SupportLevel(
                name="VWAP",
                price=round(vwap, 3),
                distance_pct=round((current_price - vwap) / vwap, 4)))
        if day_low and day_low > 0 and day_low != (levels[0].price if levels else None):
            levels.append(SupportLevel(
                name="今日低点",
                price=round(day_low, 3),
                distance_pct=round((current_price - day_low) / day_low, 4)))
        for name, val in [("MA5", ma5), ("MA10", ma10), ("MA20", ma20)]:
            if val and val > 0 and val < current_price:
                levels.append(SupportLevel(
                    name=name,
                    price=round(val, 3),
                    distance_pct=round((current_price - val) / val, 4)))
        levels.sort(key=lambda x: x.distance_pct if x.distance_pct >= 0 else float('inf'))
        return levels

    def _calc_rsi(self, closes, period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        diffs = closes.diff().dropna().tail(period)
        gains = diffs.clip(lower=0)
        losses = (-diffs).clip(lower=0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _detect_hammer(self, df) -> Optional[str]:
        if len(df) < 1:
            return None
        for idx, label in [(-1, "最新K线"), (-2, "前一根")]:
            if len(df) < abs(idx):
                continue
            bar = df.iloc[idx]
            body = abs(bar['close'] - bar['open'])
            lower_shadow = min(bar['open'], bar['close']) - bar['low']
            upper_shadow = bar['high'] - max(bar['open'], bar['close'])
            if body == 0:
                continue
            if lower_shadow / body >= HAMMER_LOWER_SHADOW_RATIO and upper_shadow / body <= HAMMER_UPPER_SHADOW_MAX:
                return label
        return None

    def _detect_volume_shrink(self, df) -> Tuple[bool, float]:
        if len(df) < 6:
            return False, 1.0
        open_vol_ma = df['volume'].iloc[:6].mean()
        if open_vol_ma == 0:
            return False, 1.0
        recent_vol = df['volume'].iloc[-3:].mean()
        ratio = recent_vol / open_vol_ma
        return ratio < VOLUME_SHRINK_RATIO, round(ratio, 2)

    def _score_entry(self, supports, signals, current_price, vwap,
                     volume_shrink_ratio, rsi) -> Tuple[float, str]:
        score = 0.0

        proximity = 100
        for s in supports:
            if s.name in ("前日低点", "MA20", "MA10"):
                if s.distance_pct >= 0:
                    proximity = min(proximity, s.distance_pct)
        if proximity <= 0.005:
            score += 40.0
        elif proximity <= 0.01:
            score += 32.0
        elif proximity <= 0.015:
            score += 24.0
        elif proximity <= 0.02:
            score += 16.0
        else:
            score += 4.0

        if volume_shrink_ratio < 0.4:
            score += 20.0
        elif volume_shrink_ratio < VOLUME_SHRINK_RATIO:
            score += 14.0
        elif volume_shrink_ratio < 0.8:
            score += 8.0

        if rsi is not None:
            if rsi < 20:
                score += 15.0
            elif rsi < 30:
                score += 10.5
            elif rsi < 40:
                score += 6.0
            elif rsi < 50:
                score += 3.0

        if any(s.type == "hammer" for s in signals):
            hammer_s = next(s for s in signals if s.type == "hammer")
            score += 15.0 if hammer_s.strength == 2 else 9.0

        if score >= 70:
            return score, "优秀"
        elif score >= 50:
            return score, "良好"
        elif score >= 30:
            return score, "一般"
        return score, "观望"

    def _check_risk(self, df, current_price, open_price, prev_day_high,
                    prev_day_close, limit_down_price) -> Optional[str]:
        risk_parts = []
        if open_price and open_price > 0 and prev_day_high and prev_day_high > 0:
            if open_price > prev_day_high and current_price < open_price:
                if (open_price - current_price) / open_price > DEEP_DECLINE_THRESHOLD:
                    risk_parts.append("高开低走")
        if open_price and prev_day_close and prev_day_close > 0:
            if open_price < prev_day_close and current_price < open_price:
                risk_parts.append("低开续跌")
        if limit_down_price and current_price <= limit_down_price:
            risk_parts.append("触及跌停价")
        if len(df) >= 6:
            close_early = df['close'].iloc[:6].mean()
            if close_early > 0 and current_price < close_early * 0.95:
                risk_parts.append("持续走低")
        vol_ma = df['volume'].iloc[:6].mean() if len(df) >= 6 else df['volume'].mean()
        recent_vol = df['volume'].iloc[-3:].mean()
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) >= 2 else last_close
        if vol_ma > 0 and recent_vol > vol_ma * VOLUME_PANIC_RATIO and last_close < prev_close:
            risk_parts.append("放量下跌")
        return "; ".join(risk_parts) if risk_parts else None

    def _extract_recent_bars(self, df, n: int = 48) -> List[dict]:
        tail = df.tail(n)
        bars = []
        for idx, row in tail.iterrows():
            t = str(idx)
            if hasattr(idx, 'strftime'):
                t = idx.strftime('%H:%M')
            elif ' ' in t:
                t = t.split(' ')[-1][:5]
            bars.append({
                "time": t,
                "open": round(float(row['open']), 3),
                "high": round(float(row['high']), 3),
                "low": round(float(row['low']), 3),
                "close": round(float(row['close']), 3),
                "volume": int(row['volume']),
            })
        return bars
