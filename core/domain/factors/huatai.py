"""
华泰多因子模型 — 从 daily_quotes K线数据直接计算的因子

连续信号 (score: -1..1):
  ht_reversal_5    — 短期反转 (5日反转效应)
  ht_reversal_20   — 中期反转 (20日反转效应)
  ht_momentum_60   — 长期动量 (60日趋势跟踪)
  ht_max_dd        — 最大回撤 (20日风险指标)
  ht_skewness      — 收益偏度 (正偏=稳健上涨)
  ht_kurtosis      — 收益峰度 (高峰=极端波动)
  ht_abn_turnover  — 异常换手率 (换手率偏离均值)
  ht_turnover_chg  — 换手率变化 (换手率趋势)
  ht_illiquidity   — 非流动性 (Amihud指标,低流动性溢价)
  ht_vol_chg       — 成交量变化 (放量/缩量)
  ht_price_pos     — 价格位置 (当前价在60日高低区间的相对位置)

离散信号 (buy/hold/sell):
  ht_reversal_buy  — 超跌反弹: 5日跌超5%→买入
  ht_ma_cross      — 均线金叉: MA5上穿MA20→买入
  ht_new_high      — 新高突破: 创60日新高→买入
  ht_vol_price_div — 量价背离: 价涨量缩→卖出
"""
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════
# 连续信号
# ══════════════════════════════════════════════

def factor_reversal_5(df):
    """短期反转: 5日跌得多→买入信号"""
    try:
        close = df["close"].astype(float)
        if len(close) < 5:
            return 0.0
        ret5 = close.iloc[-1] / close.iloc[-5] - 1
        return round(np.tanh(-ret5 * 10), 4)
    except Exception:
        return 0.0


def factor_reversal_20(df):
    """中期反转: 20日跌得多→买入信号"""
    try:
        close = df["close"].astype(float)
        if len(close) < 20:
            return 0.0
        ret20 = close.iloc[-1] / close.iloc[-20] - 1
        return round(np.tanh(-ret20 * 5), 4)
    except Exception:
        return 0.0


def factor_momentum_60(df):
    """长期动量: 60日趋势跟踪"""
    try:
        close = df["close"].astype(float)
        if len(close) < 60:
            return factor_momentum_20(df)
        ret60 = close.iloc[-1] / close.iloc[-60] - 1
        # 计算趋势一致性: 近60日中上涨天数占比
        returns = close.pct_change(fill_method=None).dropna().iloc[-60:]
        up_ratio = (returns > 0).mean()
        raw = ret60 * 3.0 + up_ratio - 0.5
        return round(np.tanh(raw), 4)
    except Exception:
        return 0.0


def factor_momentum_20(df):
    """20日动量 (fallback)"""
    try:
        close = df["close"].astype(float)
        if len(close) < 20:
            return 0.0
        ret20 = close.iloc[-1] / close.iloc[-20] - 1
        return round(np.tanh(ret20 * 4), 4)
    except Exception:
        return 0.0


def factor_max_dd(df):
    """20日最大回撤 (回撤越小越好)"""
    try:
        close = df["close"].astype(float).iloc[-20:]
        if len(close) < 10:
            return 0.0
        peak = close.expanding().max()
        dd = (close / peak - 1).min()
        # dd在-0.2 ~ 0之间, -0.05→0.76, -0.2→0
        return round(np.tanh((dd + 0.1) * 8), 4)
    except Exception:
        return 0.0


def factor_skewness(df):
    """20日收益偏度 (正偏=稳健上涨, 负偏=暴跌风险)"""
    try:
        returns = df["close"].astype(float).pct_change(fill_method=None).dropna().iloc[-20:]
        if len(returns) < 10:
            return 0.0
        skew = returns.skew()
        return round(np.tanh(skew * 0.5), 4)
    except Exception:
        return 0.0


def factor_kurtosis(df):
    """20日收益峰度 (高值=极端波动, 低值=温和走势)"""
    try:
        returns = df["close"].astype(float).pct_change(fill_method=None).dropna().iloc[-20:]
        if len(returns) < 10:
            return 0.0
        kurt = returns.kurtosis()
        # 峰度>3为厚尾, 反向下分
        return round(np.tanh((3.0 - kurt) * 0.3), 4)
    except Exception:
        return 0.0


def factor_abnormal_turnover(df):
    """异常换手率: 当前换手率相对20日均值的偏离"""
    try:
        if "turnover" not in df.columns:
            return 0.0
        turnover = df["turnover"].astype(float)
        if len(turnover) < 20:
            return 0.0
        avg20 = turnover.iloc[-21:-1].mean()
        if avg20 <= 0:
            return 0.0
        ratio = turnover.iloc[-1] / avg20 - 1.0
        # 异常高换手→卖出(过度关注), 异常低换手+横盘→可能蓄势
        return round(np.tanh(-ratio * 2), 4)
    except Exception:
        return 0.0


def factor_turnover_change(df):
    """换手率变化: 近5日换手率趋势"""
    try:
        if "turnover" not in df.columns:
            return 0.0
        turnover = df["turnover"].astype(float)
        if len(turnover) < 20:
            return 0.0
        recent5 = turnover.iloc[-5:].mean()
        prev15 = turnover.iloc[-20:-5].mean()
        if prev15 <= 0:
            return 0.0
        chg = recent5 / prev15 - 1.0
        return round(np.tanh(chg * 3), 4)
    except Exception:
        return 0.0


def factor_illiquidity(df):
    """Amihud 非流动性指标: |日收益|/成交额, 20日均值"""
    try:
        close = df["close"].astype(float)
        amount = df["amount"].astype(float)
        if len(close) < 20:
            return 0.0
        returns = close.pct_change(fill_method=None).abs().iloc[-20:]
        amt = amount.iloc[-20:]
        illiq = (returns / (amt + 1)).mean() * 1e8
        # 非流动性越高→流动性越差→可能有流动性折价溢价
        return round(np.tanh(illiq * 0.5), 4)
    except Exception:
        return 0.0


def factor_volume_change(df):
    """成交量变化: 近5日均量相对20日均量的变化"""
    try:
        volume = df["volume"].astype(float)
        if len(volume) < 20:
            return 0.0
        avg5 = volume.iloc[-5:].mean()
        avg20 = volume.iloc[-20:].mean()
        if avg20 <= 0:
            return 0.0
        ratio = avg5 / avg20 - 1.0
        return round(np.tanh(ratio * 2), 4)
    except Exception:
        return 0.0


def factor_price_position(df):
    """价格位置: 当前价在60日高低区间的百分比位置"""
    try:
        close = df["close"].astype(float)
        if len(close) < 60:
            return 0.0
        high60 = close.iloc[-60:].max()
        low60 = close.iloc[-60:].min()
        if high60 <= low60:
            return 0.0
        pos = (close.iloc[-1] - low60) / (high60 - low60)
        # 低位→买入机会, 高位→卖出风险
        return round(np.tanh((0.5 - pos) * 3), 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════
# 离散信号 (-1/0/1)
# ══════════════════════════════════════════════

def signal_reversal_buy(df):
    """超跌反弹信号: 5日跌超5%且放量→买入, 5日涨超5%→卖出"""
    try:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        if len(close) < 10:
            return 0
        ret5 = close.iloc[-1] / close.iloc[-5] - 1
        avg_vol = volume.iloc[-10:].mean()
        cur_vol = volume.iloc[-1]
        if ret5 < -0.05 and cur_vol > avg_vol * 1.2:
            return 1
        if ret5 > 0.05:
            return -1
        return 0
    except Exception:
        return 0


def signal_ma_cross(df):
    """均线金叉信号: MA5上穿MA20→买入, 下穿→卖出"""
    try:
        close = df["close"].astype(float)
        if len(close) < 21:
            return 0
        ma5 = close.iloc[-5:].mean()
        ma5_prev = close.iloc[-6:-1].mean()
        ma20 = close.iloc[-20:].mean()
        ma20_prev = close.iloc[-21:-1].mean()
        if ma5 > ma20 and ma5_prev <= ma20_prev:
            return 1
        if ma5 < ma20 and ma5_prev >= ma20_prev:
            return -1
        return 0
    except Exception:
        return 0


def signal_new_high(df):
    """新高突破信号: 创60日新高且放量→买入"""
    try:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        if len(close) < 60:
            return 0
        high60 = close.iloc[-61:-1].max()
        cur = close.iloc[-1]
        avg_vol = volume.iloc[-20:].mean()
        cur_vol = volume.iloc[-1]
        if cur > high60 * 1.01:
            return 1
        if cur_vol > avg_vol * 1.5:
            return 1 if close.pct_change(fill_method=None).iloc[-1] > 0 else 0
        return 0
    except Exception:
        return 0


def signal_vol_price_divergence(df):
    """量价背离信号: 价涨量缩→卖出, 价跌量增→买入"""
    try:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        if len(close) < 10:
            return 0
        ret5 = close.iloc[-1] / close.iloc[-5] - 1
        vol5 = volume.iloc[-5:].mean()
        vol20 = volume.iloc[-20:-5].mean() if len(volume) >= 20 else vol5
        if vol20 <= 0:
            return 0
        vol_chg = vol5 / vol20 - 1.0
        if ret5 > 0.02 and vol_chg < -0.15:
            return -1
        if ret5 < -0.02 and vol_chg > 0.3:
            return 1
        return 0
    except Exception:
        return 0


# ══════════════════════════════════════════════
# 白盒因子: 行业残差动量 / 量价背离(连续) / 资金异动
# ══════════════════════════════════════════════

def factor_industry_residual(df, code=None, industry_returns=None):
    """行业残差动量: 个股20日收益 - 同行业平均20日收益。超额>0→加分"""
    try:
        close = df["close"].astype(float)
        if len(close) < 20:
            return 0.0
        stock_ret = close.iloc[-1] / close.iloc[-20] - 1
        if industry_returns is not None and code is not None:
            ind_ret = industry_returns.get(code, None)
            if ind_ret is not None:
                residual = stock_ret - ind_ret
                return round(np.tanh(residual * 8), 4)
        return round(np.tanh(stock_ret * 4), 4)
    except Exception:
        return 0.0


def factor_volume_price_divergence(df):
    """量价背离(连续版): 近10日价涨量缩程度→-1..1"""
    try:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        if len(close) < 10:
            return 0.0
        ret10 = close.iloc[-1] / close.iloc[-10] - 1
        vol5 = volume.iloc[-5:].mean()
        vol20 = volume.iloc[-20:-5].mean() if len(volume) >= 20 else vol5
        if vol20 <= 0:
            return 0.0
        vol_ratio = vol5 / vol20
        # 价涨量缩→负分, 价跌量增→正分
        divergence = -ret10 * (vol_ratio - 1.0) * 5
        return round(np.tanh(divergence), 4)
    except Exception:
        return 0.0


def factor_capital_anomaly(df, code=None, capital_flow_data=None):
    """资金异动: 近5日主力净流入/成交额均值 → -1..1"""
    try:
        if capital_flow_data is not None and code is not None:
            flows = capital_flow_data.get(code, [])
            if flows:
                amount = df["amount"].astype(float).iloc[-5:].mean()
                if amount > 0:
                    avg_flow = sum(f for f in flows) / len(flows)
                    ratio = avg_flow / amount
                    return round(np.tanh(ratio * 100), 4)
        # 无资金流数据时用成交量变化近似
        volume = df["volume"].astype(float)
        amount = df["amount"].astype(float)
        if len(volume) < 20:
            return 0.0
        vol_ratio = volume.iloc[-5:].mean() / max(volume.iloc[-20:].mean(), 1)
        amt_ratio = amount.iloc[-5:].mean() / max(amount.iloc[-20:].mean(), 1)
        return round(np.tanh((vol_ratio - amt_ratio) * 3), 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════
# 因子注册表
# ══════════════════════════════════════════════

CONTINUOUS_FACTORS = {
    "ht_reversal_5":     {"func": factor_reversal_5,     "label": "短期反转(5日)", "weight": 0.08},
    "ht_reversal_20":    {"func": factor_reversal_20,    "label": "中期反转(20日)", "weight": 0.10},
    "ht_momentum_60":    {"func": factor_momentum_60,    "label": "长期动量(60日)", "weight": 0.12},
    "ht_max_dd":         {"func": factor_max_dd,         "label": "最大回撤(20日)", "weight": 0.08},
    "ht_skewness":       {"func": factor_skewness,       "label": "收益偏度", "weight": 0.06},
    "ht_kurtosis":       {"func": factor_kurtosis,       "label": "收益峰度", "weight": 0.06},
    "ht_abn_turnover":   {"func": factor_abnormal_turnover, "label": "异常换手率", "weight": 0.10},
    "ht_turnover_chg":   {"func": factor_turnover_change,   "label": "换手率变化", "weight": 0.08},
    "ht_illiquidity":    {"func": factor_illiquidity,    "label": "非流动性(Amihud)", "weight": 0.06},
    "ht_vol_chg":        {"func": factor_volume_change,  "label": "成交量变化", "weight": 0.10},
    "ht_price_pos":      {"func": factor_price_position, "label": "价格位置", "weight": 0.10},
    "wb_ind_residual":   {"func": factor_industry_residual, "label": "行业残差动量", "weight": 0.10},
    "wb_vp_divergence":  {"func": factor_volume_price_divergence, "label": "量价背离(连续)", "weight": 0.08},
    "wb_capital_anomaly":{"func": factor_capital_anomaly, "label": "资金异动", "weight": 0.08},
}

DISCRETE_SIGNALS = {
    "ht_reversal_buy":      {"func": signal_reversal_buy,      "label": "超跌反弹", "weight": 0.25},
    "ht_ma_cross":          {"func": signal_ma_cross,          "label": "均线金叉", "weight": 0.30},
    "ht_new_high":          {"func": signal_new_high,          "label": "新高突破", "weight": 0.25},
    "ht_vol_price_div":     {"func": signal_vol_price_divergence, "label": "量价背离", "weight": 0.20},
}


def compute_huatai_score(df, factor_weights=None, signal_weights=None):
    """计算华泰多因子综合评分。
    连续因子: 加权求和 → -1..1
    离散信号: 加权投票 → -1..1
    总评分: 连续*0.7 + 离散*0.3
    """
    if factor_weights is None:
        factor_weights = {k: v["weight"] for k, v in CONTINUOUS_FACTORS.items()}
    if signal_weights is None:
        signal_weights = {k: v["weight"] for k, v in DISCRETE_SIGNALS.items()}

    # 连续因子加权
    cont_score = 0.0
    cont_w = 0.0
    for key, fw in factor_weights.items():
        if fw <= 0:
            continue
        entry = CONTINUOUS_FACTORS.get(key)
        if entry is None:
            continue
        s = entry["func"](df)
        cont_score += fw * s
        cont_w += fw

    # 离散信号投票
    disc_score = 0.0
    disc_w = 0.0
    for key, sw in signal_weights.items():
        if sw <= 0:
            continue
        entry = DISCRETE_SIGNALS.get(key)
        if entry is None:
            continue
        vote = entry["func"](df)
        disc_score += sw * vote
        disc_w += sw

    if cont_w > 0:
        cont_score /= cont_w
    if disc_w > 0:
        disc_score /= disc_w

    total = cont_score * 0.7 + disc_score * 0.3
    return round(max(-1.0, min(1.0, total)), 4)


def compute_single_factor(df, key):
    """计算单个因子的值,供 Phase 1 信号测试使用"""
    if key in CONTINUOUS_FACTORS:
        return CONTINUOUS_FACTORS[key]["func"](df)
    if key in DISCRETE_SIGNALS:
        return float(DISCRETE_SIGNALS[key]["func"](df))
    return 0.0
