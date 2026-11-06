"""
股票因子库 — 技术/动量/波动率/价值/质量/规模 (Fama-French 风格分类)
所有函数签名: fn(df, **params) -> float  (-1..1)
"""

import math as _math
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# 技术指标因子 (technical)
# ═══════════════════════════════════════════

def factor_macd(df) -> float:
    try:
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            return 0.0
        macd = float(df["macd"].iloc[-1])
        signal = float(df["macd_signal"].iloc[-1])
        prev_macd = float(df["macd"].iloc[-2]) if len(df) >= 2 else macd
        prev_signal = float(df["macd_signal"].iloc[-2]) if len(df) >= 2 else signal
        if pd.isna(macd) or pd.isna(signal):
            return 0.0
        if prev_macd <= prev_signal and macd > signal:
            return 1.0
        elif prev_macd >= prev_signal and macd < signal:
            return -1.0
        return 0.0
    except Exception:
        return 0.0


def factor_rsi(df, oversold=30, overbought=70) -> float:
    try:
        if "rsi14" not in df.columns:
            return 0.0
        rsi = float(df["rsi14"].iloc[-1])
        if pd.isna(rsi):
            return 0.0
        if rsi < oversold:
            return 1.0
        elif rsi > overbought:
            return -1.0
        return np.tanh((rsi - 50) / 25) * 0.5
    except Exception:
        return 0.0


def factor_ma_alignment(df) -> float:
    try:
        cols = ["ma5", "ma10", "ma20", "ma60"]
        vals = {}
        for c in cols:
            if c in df.columns:
                v = float(df[c].iloc[-1])
                if not pd.isna(v):
                    vals[c] = v
        if len(vals) >= 3:
            keys = sorted(vals.keys(), key=lambda x: int(x[2:]))
            aligned = all(vals[keys[i]] >= vals[keys[i+1]] for i in range(len(keys)-1))
            if aligned:
                return 1.0
            reversed_aligned = all(vals[keys[i]] <= vals[keys[i+1]] for i in range(len(keys)-1))
            if reversed_aligned:
                return -1.0
        return 0.0
    except Exception:
        return 0.0


def factor_kdj(df) -> float:
    try:
        for c in ["k", "d", "j"]:
            if c not in df.columns:
                return 0.0
        k = float(df["k"].iloc[-1])
        d = float(df["d"].iloc[-1])
        prev_k = float(df["k"].iloc[-2]) if len(df) >= 2 else k
        prev_d = float(df["d"].iloc[-2]) if len(df) >= 2 else d
        j = float(df["j"].iloc[-1])
        if pd.isna(k) or pd.isna(d):
            return 0.0
        if j < 20:
            return 0.8
        if j > 80:
            return -0.5
        if prev_k <= prev_d and k > d and k < 40:
            return 0.6
        if prev_k >= prev_d and k < d and k > 60:
            return -0.4
        return 0.0
    except Exception:
        return 0.0


def factor_bollinger(df) -> float:
    try:
        for c in ["bb_upper", "bb_middle", "bb_lower", "close"]:
            if c not in df.columns:
                return 0.0
        close = float(df["close"].iloc[-1])
        upper = float(df["bb_upper"].iloc[-1])
        lower = float(df["bb_lower"].iloc[-1])
        middle = float(df["bb_middle"].iloc[-1])
        if pd.isna(upper) or pd.isna(lower):
            return 0.0
        if close <= lower:
            return 1.0
        elif close >= upper:
            return -0.8
        elif close > middle:
            return -0.2
        return 0.2
    except Exception:
        return 0.0


def factor_volume(df) -> float:
    try:
        if "vol_ma5" not in df.columns or "vol_ma20" not in df.columns:
            return 0.0
        vol = float(df["volume"].iloc[-1])
        vol_ma5 = float(df["vol_ma5"].iloc[-1])
        vol_ma20 = float(df["vol_ma20"].iloc[-1])
        if pd.isna(vol_ma5) or pd.isna(vol_ma20) or vol_ma5 <= 0:
            return 0.0
        close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else close
        price_up = close > prev_close
        vol_ratio = vol / vol_ma5
        if price_up and vol_ratio > 1.5:
            return 0.7
        elif not price_up and vol_ratio > 1.5:
            return -0.5
        return np.tanh((vol_ratio - 1.0) * 0.5) * 0.3
    except Exception:
        return 0.0


# ═══════════════════════════════════════════
# 动量因子 (momentum)
# ═══════════════════════════════════════════

def factor_momentum(df, n=20) -> float:
    try:
        closes = df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        ret_n = closes.iloc[-1] / closes.iloc[-n] - 1 if closes.iloc[-n] > 0 else 0
        half = max(n // 2, 5)
        half_closes = closes.iloc[-half:]
        slope_recent = (half_closes.iloc[-1] - half_closes.iloc[0]) / max(abs(half_closes.iloc[0]), 0.01)
        raw = ret_n * 0.6 + slope_recent * 0.4
        return round(_math.tanh(raw * 2), 4)
    except Exception:
        return 0.0


def factor_reversal(df, n=5) -> float:
    try:
        closes = df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        ret = closes.iloc[-1] / closes.iloc[-n] - 1
        return round(np.tanh(-ret * 10), 4)
    except Exception:
        return 0.0


def factor_trend_momentum(df, n=60) -> float:
    try:
        closes = df["close"].astype(float)
        if len(closes) < n:
            return factor_momentum(df, n=20)
        ret_n = closes.iloc[-1] / closes.iloc[-n] - 1
        returns = closes.pct_change(fill_method=None).dropna().iloc[-n:]
        up_ratio = (returns > 0).mean()
        raw = ret_n * 3.0 + up_ratio - 0.5
        return round(np.tanh(raw), 4)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════
# 波动率因子 (volatility)
# ═══════════════════════════════════════════

def factor_volatility(df, n=20) -> float:
    try:
        closes = df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        returns = closes.pct_change(fill_method=None).dropna().iloc[-n:]
        if len(returns) < 5:
            return 0.0
        vol = returns.std()
        mean_ret = returns.mean()
        sharpe_like = mean_ret / max(vol, 0.001)
        ann_vol = vol * (250 ** 0.5)
        vol_score = 1.0 - ann_vol / 0.6
        raw = sharpe_like * 0.3 + vol_score * 0.7
        return round(_math.tanh(raw * 1.5), 4)
    except Exception:
        return 0.0


def factor_vol_regime(df, n=20) -> float:
    try:
        closes = df["close"].astype(float)
        if len(closes) < n * 2:
            return 0.0
        returns = closes.pct_change(fill_method=None).dropna()
        recent_vol = returns.iloc[-n:].std()
        hist_vol = returns.iloc[-(2*n):-n].std()
        if hist_vol < 1e-9:
            return 0.0
        vol_ratio = recent_vol / hist_vol
        if vol_ratio > 1.5:
            return -1.0
        elif vol_ratio < 0.5:
            return 1.0
        return round(-(vol_ratio - 1.0), 4)
    except Exception:
        return 0.0


def factor_max_drawdown(df, n=20) -> float:
    try:
        closes = df["close"].astype(float).iloc[-n:]
        if len(closes) < n:
            return 0.0
        peak = closes.expanding().max()
        dd = (closes - peak) / peak
        max_dd = abs(dd.min())
        return round(-np.tanh(max_dd * 10), 4)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════
# 价值因子 (value — 代理实现)
# ═══════════════════════════════════════════

def _score_ep(df, n=5):
    try:
        if len(df) < n:
            return 0.0
        avg_amount = float(np.mean(df["amount"].astype(float).tail(n)))
        if avg_amount <= 0:
            return 0.0
        roe_proxy = float(df["close"].iloc[-1] / df["close"].iloc[-max(n, len(df)-1)] - 1) if len(df) >= n else 0
        ep = max(-0.5, min(0.5, roe_proxy))
        return round(float(_math.tanh(ep * 3)), 4)
    except Exception:
        return 0.0


def _score_bp(df, n=20):
    try:
        if len(df) < n:
            return 0.0
        avg_amount = float(np.mean(df["amount"].astype(float).tail(n)))
        if avg_amount <= 0:
            return 0.0
        close = float(df["close"].iloc[-1])
        bp_proxy = avg_amount / (close * 1e8) if close > 0 else 0
        return round(float(_math.tanh(bp_proxy - 1)), 4)
    except Exception:
        return 0.0


def _score_dp(df, n=60):
    try:
        if len(df) < n:
            return 0.0
        returns = df["close"].astype(float).pct_change(fill_method=None).tail(n)
        vol = float(np.std(returns))
        ret = float(np.mean(returns))
        if vol <= 0:
            return 0.0
        sharpe_like = ret / vol
        return round(float(_math.tanh(sharpe_like * 2)), 4)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════
# 质量因子 (quality — 代理实现)
# ═══════════════════════════════════════════

def _score_roe(df, n=20):
    try:
        if len(df) < n:
            return 0.0
        closes = df["close"].astype(float)
        ret_60d = float(closes.iloc[-1] / closes.iloc[-min(60, len(closes))] - 1) if len(closes) >= 60 else 0
        dd = float(min(closes.tail(n) / closes.tail(n).cummax() - 1))
        quality = ret_60d + dd
        return round(float(_math.tanh(quality * 2)), 4)
    except Exception:
        return 0.0


def _score_gross_margin(df, n=20):
    try:
        if len(df) < n:
            return 0.0
        high = df["high"].astype(float).tail(n)
        low = df["low"].astype(float).tail(n)
        close = df["close"].astype(float).tail(n)
        avg_price_level = float(np.mean(close))
        avg_spread = float(np.mean((high - low) / close))
        score = (avg_price_level / 100) * (1 - avg_spread * 2)
        return round(float(_math.tanh(score - 0.5)), 4)
    except Exception:
        return 0.0


def _score_debt_ratio(df, n=20):
    try:
        if len(df) < n:
            return 0.0
        volume = df["volume"].astype(float).tail(n)
        amount = df["amount"].astype(float).tail(n)
        turnover = float(np.mean(volume)) / max(float(np.mean(amount)), 1)
        stability = 1.0 - float(np.std(amount) / max(np.mean(amount), 1))
        score = stability - turnover * 10
        return round(float(_math.tanh(score)), 4)
    except Exception:
        return 0.0


def _score_accruals(df, n=60):
    try:
        if len(df) < n:
            return 0.0
        close = df["close"].astype(float).tail(n)
        volume = df["volume"].astype(float).tail(n)
        price_trend = float(np.polyfit(range(n), close, 1)[0]) / float(np.mean(close))
        vol_trend = float(np.polyfit(range(n), volume, 1)[0]) / max(float(np.mean(volume)), 1)
        accrual_quality = price_trend - vol_trend
        return round(float(_math.tanh(accrual_quality * 20)), 4)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════
# 规模因子 (size — 代理实现)
# ═══════════════════════════════════════════

def _score_size(df, n=20):
    try:
        if len(df) < n:
            return 0.0
        amount = df["amount"].astype(float).tail(n)
        avg_amount = float(np.mean(amount)) if len(amount) > 0 else 0
        if avg_amount <= 0:
            return 0.0
        log_size = _math.log(avg_amount)
        z = (17.0 - log_size) / 2.5
        return round(float(_math.tanh(z)), 4)
    except Exception:
        return 0.0


def _score_cap_median(df, n=60):
    try:
        if len(df) < n:
            return 0.0
        amounts = df["amount"].astype(float).tail(n)
        median = float(np.median(amounts)) if len(amounts) > 0 else 0
        if median <= 0:
            return 0.0
        z = (17.0 - _math.log(median)) / 2.5
        return round(float(_math.tanh(z)), 4)
    except Exception:
        return 0.0
