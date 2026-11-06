"""
白盒因子模板库 — 假设先行的结构化因子定义

每套模板包含:
- 经济学逻辑描述（赚什么钱、什么市态有效）
- 固定算子结构（只变参数窗口，不拼接随机公式）
- 参数定义（遍历范围）
- Python评分函数模板
- Qlib表达式（用于模型训练）
"""
import math as _math

# ══════════════════════════════════════════════
# 模板定义
# ══════════════════════════════════════════════

FACTOR_TEMPLATES = {}

def _reg(templates):
    FACTOR_TEMPLATES.update(templates)

# ── A类: 基本面逻辑 ──

_reg({
    "wb_peg_adj": {
        "category": "fundamental",
        "label": "PEG调整估值",
        "dataflow": "price_with_db",
        "data_safety": "caveat",
        "logic_desc": (
            "经济学逻辑: PEG<1说明成长性被低估, ROE>15%确认盈利能力。"
            "二者同时满足的股票兼具'好公司+好价格'特征, 存在估值修复驱动的Alpha。"
            "赚估值回归的钱, 熊市末期和震荡市有效, 纯概念炒作市可能失效。"
        ),
        "operator": "tanh((1-PEG)*peg_w + (ROE-0.15)*roe_w)",
        "default_n": 20,
        "param_definitions": [
            {"name": "peg_w", "type": "float", "default": 2.0, "range": [0.5, 4.0], "step": 0.5},
            {"name": "roe_w", "type": "float", "default": 3.0, "range": [0.5, 5.0], "step": 0.5},
        ],
        "available_cols": ["close", "open", "high", "low", "volume", "amount"],
        "scoring_template": '''def _score_wb_peg_adj(code, df, n={default_n}, preloaded_data=None):
    """PEG调整估值"""
    try:
        if preloaded_data is None or code not in preloaded_data:
            return 0.0
        db = preloaded_data[code]
        peg = db.get("peg", 999)
        roe = db.get("roe", -999)
        if peg > 900 or roe < -900:
            return 0.0
        import math as _math
        peg_w = {peg_w}
        roe_w = {roe_w}
        score = _math.tanh((1.0 - peg) * peg_w + (roe - 0.15) * roe_w)
        return round(float(score), 4)
    except Exception:
        return 0.0''',
        "qlib_expression": "# PEG/ROE依赖外部数据库，需通过custom_handler注入\n# 近似: 低PE+正ROE → -Rank($pe_ttm) + Rank(Delta($roe, 4))",
    },

    "wb_earnings_mo": {
        "category": "fundamental",
        "label": "盈利动量",
        "dataflow": "price_with_db",
        "data_safety": "caveat",
        "logic_desc": (
            "经济学逻辑: 净利润增速连续2季加速→盈利趋势向好。"
            "市场对盈利拐点反应滞后, 存在趋势跟随Alpha。"
            "赚盈利超预期的钱, 业绩披露季尤为有效。"
        ),
        "operator": "tanh(Δ_profit_growth * 5)",
        "default_n": 20,
        "param_definitions": [
            {"name": "accel_w", "type": "float", "default": 5.0, "range": [2.0, 10.0], "step": 1.0},
        ],
        "available_cols": ["close", "open", "high", "low", "volume", "amount"],
        "scoring_template": '''def _score_wb_earnings_mo(code, df, n={default_n}, preloaded_data=None):
    """盈利动量: 净利润增速连续加速"""
    try:
        if preloaded_data is None or code not in preloaded_data:
            return 0.0
        db = preloaded_data[code]
        profit_growth_q = db.get("profit_growth_q", [])
        if len(profit_growth_q) < 2:
            return 0.0
        import math as _math
        accel = profit_growth_q[-1] - profit_growth_q[-2]
        accel_w = {accel_w}
        return round(_math.tanh(accel * accel_w), 4)
    except Exception:
        return 0.0''',
        "qlib_expression": "# 依赖财报数据，需通过custom_handler注入",
    },
})

# ── B类: 资金行为逻辑 ──

_reg({
    "wb_capital_anomaly": {
        "category": "flow",
        "label": "资金异动",
        "dataflow": "price_only",
        "data_safety": "safe",
        "logic_desc": (
            "经济学逻辑: 成交量异常放大但价格未同步上涨→主力在悄悄建仓。"
            "量比放大说明有大资金介入, 价格滞后反应说明筹码尚未派发, 存在跟随窗口。"
            "缩量阴跌后量能突然放大时最有效, 单边暴跌中可能失效。"
        ),
        "operator": "tanh(volume_ratio * price_divergence_sign)",
        "default_n": 5,
        "param_definitions": [
            {"name": "n", "type": "int", "default": 5, "range": [3, 5, 10, 20]},
            {"name": "vol_threshold", "type": "float", "default": 1.5, "range": [1.2, 1.5, 2.0, 2.5]},
        ],
        "available_cols": ["close", "volume", "amount"],
        "scoring_template": '''def _score_wb_capital_anomaly(df, n={default_n}):
    """资金异动: 量比异常放大 + 价格未涨 = 建仓信号"""
    try:
        closes = df["close"].astype(float)
        volumes = df["volume"].astype(float)
        if len(closes) < n + 20:
            return 0.0
        recent_vol = volumes.iloc[-n:].mean()
        hist_vol = volumes.iloc[-(n+20):-n].mean()
        if hist_vol <= 0:
            return 0.0
        vol_ratio = recent_vol / hist_vol
        ret_n = closes.iloc[-1] / closes.iloc[-n] - 1.0
        import math as _math
        if vol_ratio > {vol_threshold} and ret_n < 0.03:
            return round(_math.tanh((vol_ratio - 1.0) * 3.0), 4)
        elif vol_ratio > {vol_threshold} and ret_n > 0.05:
            return round(-_math.tanh((vol_ratio - 1.0) * 2.0), 4)
        return 0.0
    except Exception:
        return 0.0''',
        "qlib_expression": (
            "({vol_threshold} - Mean($volume, {n}) / Mean($volume, {n}+20)) > 0 ? "
            "($volume / Mean($volume, 20) - 1) * ({close_sign}) : 0"
        ),
    },

    "wb_smart_money": {
        "category": "flow",
        "label": "聪明钱跟随",
        "dataflow": "price_only",
        "data_safety": "safe",
        "logic_desc": (
            "经济学逻辑: 量升价稳→有资金在主动承接抛压。"
            "成交额放大说明市场关注度上升, 价格站稳短期均线说明买方力量强于卖方。"
            "趋势启动初期最有效, 高位放量滞涨反而是派发信号。"
        ),
        "operator": "tanh(amount_ratio * price_stability)",
        "default_n": 10,
        "param_definitions": [
            {"name": "n", "type": "int", "default": 10, "range": [5, 10, 20]},
            {"name": "stability_window", "type": "int", "default": 5, "range": [3, 5, 10]},
        ],
        "available_cols": ["close", "volume", "amount"],
        "scoring_template": '''def _score_wb_smart_money(df, n={default_n}):
    """聪明钱跟随: 成交额放大 + 价格站稳均线"""
    try:
        closes = df["close"].astype(float)
        amounts = df["amount"].astype(float)
        if len(closes) < n + 20:
            return 0.0
        recent_amount = amounts.iloc[-n:].mean()
        hist_amount = amounts.iloc[-(n+20):-n].mean()
        if hist_amount <= 0:
            return 0.0
        amount_ratio = recent_amount / hist_amount
        ma = closes.iloc[-{stability_window}:].mean()
        price_stability = closes.iloc[-1] / ma
        import math as _math
        score = _math.tanh((amount_ratio - 1.0) * 2.0 + (price_stability - 1.0) * 5.0)
        return round(float(score), 4)
    except Exception:
        return 0.0''',
        "qlib_expression": (
            "Tanh((Mean($amount, {n}) / Mean($amount, {n}+20) - 1) * 2.0 + "
            "($close / Mean($close, {stability_window}) - 1) * 5.0)"
        ),
    },
})

# ── C类: 量价行为逻辑 ──

_reg({
    "wb_ind_residual": {
        "category": "price_volume",
        "label": "行业残差动量",
        "dataflow": "price_only",
        "data_safety": "safe",
        "logic_desc": (
            "经济学逻辑: 剔除行业Beta后的个股超额收益具有动量效应。"
            "行业景气上行时, 龙头股先涨, 滞后股补涨; 行业下行时, 优质股抗跌。"
            "残差动量>0说明个股强于同行业平均水平, 存在惯性延续。"
            "行业轮动明显的市场中效果最好, 普涨普跌的单边市场效果减弱。"
        ),
        "operator": "tanh((stock_ret - industry_ret) * 8), fallback: tanh(stock_ret * 4)",
        "default_n": 20,
        "param_definitions": [
            {"name": "n", "type": "int", "default": 20, "range": [5, 10, 20, 30]},
        ],
        "available_cols": ["close"],
        "scoring_template": '''def _score_wb_ind_residual(code, df, n={default_n}, industry_returns=None):
    """行业残差动量: 个股收益 - 行业平均收益"""
    try:
        closes = df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        stock_ret = closes.iloc[-1] / closes.iloc[-n] - 1.0
        import math as _math
        if industry_returns is not None:
            ind_ret = industry_returns.get(code)
            if ind_ret is not None:
                residual = stock_ret - ind_ret
                return round(_math.tanh(residual * 8.0), 4)
        return round(_math.tanh(stock_ret * 4.0), 4)
    except Exception:
        return 0.0''',
        "qlib_expression": (
            "Ref($close, 1) / Ref($close, {n}) - 1"
        ),
    },

    "wb_vp_divergence": {
        "category": "price_volume",
        "label": "量价背离",
        "dataflow": "price_only",
        "data_safety": "safe",
        "logic_desc": (
            "经济学逻辑: 价涨量缩→上涨动力衰减(派发), 价跌量缩→抛压衰竭(吸筹)。"
            "Corr(price, volume)为负说明量价背离, 是趋势反转的早期信号。"
            "震荡市中反转信号最可靠, 趋势单边市中容易过早逆势。"
        ),
        "operator": "tanh(-Corr(close, volume, n))",
        "default_n": 10,
        "param_definitions": [
            {"name": "n", "type": "int", "default": 10, "range": [5, 10, 20]},
        ],
        "available_cols": ["close", "volume"],
        "scoring_template": '''def _score_wb_vp_divergence(df, n={default_n}):
    """量价背离: 负相关→反转信号, 正相关→趋势确认"""
    try:
        closes = df["close"].astype(float)
        volumes = df["volume"].astype(float)
        if len(closes) < n:
            return 0.0
        close_chg = closes.iloc[-n:].pct_change(fill_method=None).dropna()
        vol_chg = volumes.iloc[-n:].pct_change(fill_method=None).dropna()
        if len(close_chg) < 5:
            return 0.0
        min_len = min(len(close_chg), len(vol_chg))
        corr = close_chg.iloc[-min_len:].corr(vol_chg.iloc[-min_len:])
        import math as _math
        return round(_math.tanh(-corr * 4.0), 4)
    except Exception:
        return 0.0''',
        "qlib_expression": (
            "-Corr($close / Ref($close, 1) - 1, $volume / Ref($volume, 1) - 1, {n})"
        ),
    },

    "wb_vol_regime": {
        "category": "price_volume",
        "label": "波动率区间突破",
        "dataflow": "price_only",
        "data_safety": "safe",
        "logic_desc": (
            "经济学逻辑: 低波动蓄势→箱体被压缩→突破时动量最强。"
            "布林带收窄(带宽缩小)说明多空力量趋于均衡, 突破方向即新趋势方向。"
            "赚趋势启动的钱, 窄幅震荡后的放量突破最有效。"
        ),
        "operator": "tanh(breakout_strength / ATR - regime_penalty)",
        "default_n": 20,
        "param_definitions": [
            {"name": "n", "type": "int", "default": 20, "range": [10, 14, 20, 30]},
            {"name": "breakout_window", "type": "int", "default": 10, "range": [5, 10, 15]},
        ],
        "available_cols": ["close", "high", "low", "volume"],
        "scoring_template": '''def _score_wb_vol_regime(df, n={default_n}):
    """波动率区间突破"""
    try:
        closes = df["close"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        volumes = df["volume"].astype(float)
        if len(closes) < n + 5:
            return 0.0
        tr = (highs.diff().abs().combine(lows.diff().abs(), max)
              .combine((highs - lows), max))
        atr = tr.iloc[-n:].mean()
        if atr <= 0:
            return 0.0
        breakout_window = {breakout_window}
        range_high = highs.iloc[-breakout_window:].max()
        range_low = lows.iloc[-breakout_window:].min()
        mid = closes.iloc[-1]
        up_break = (mid - range_high) / atr if mid > range_high else 0.0
        dn_break = (range_low - mid) / atr if mid < range_low else 0.0
        vol_ratio = volumes.iloc[-5:].mean() / max(volumes.iloc[-20:].mean(), 1)
        import math as _math
        score = _math.tanh((up_break - dn_break) * 1.5 + (vol_ratio - 1.0) * 2.0)
        return round(float(score), 4)
    except Exception:
        return 0.0''',
        "qlib_expression": (
            "($close - Max($close, {breakout_window})) / "
            "Mean(Max($high - $low, Abs($high - Ref($close, 1)), Abs($low - Ref($close, 1))), {n})"
        ),
    },

    "wb_turnover_price": {
        "category": "price_volume",
        "label": "换手率-价格协同",
        "dataflow": "price_only",
        "data_safety": "safe",
        "logic_desc": (
            "经济学逻辑: 换手率上升+价格站稳均线→量价配合的强势确认。"
            "高换手说明市场参与度高, 价格站稳MA20说明多头占优。"
            "两者同向时趋势延续概率高, 背离时(高换手+价格破位)反而是顶部信号。"
        ),
        "operator": "tanh(turnover_ratio * price_position)",
        "default_n": 20,
        "param_definitions": [
            {"name": "n", "type": "int", "default": 20, "range": [10, 20, 30]},
            {"name": "ma_period", "type": "int", "default": 20, "range": [10, 20, 30]},
        ],
        "available_cols": ["close", "volume", "turnover"],
        "scoring_template": '''def _score_wb_turnover_price(df, n={default_n}):
    """换手率-价格协同"""
    try:
        closes = df["close"].astype(float)
        if "turnover" in df.columns:
            turnovers = df["turnover"].astype(float)
        elif "volume" in df.columns:
            turnovers = df["volume"].astype(float)
        else:
            return 0.0
        if len(closes) < n + 5:
            return 0.0
        recent_turnover = turnovers.iloc[-n:].mean()
        hist_turnover = turnovers.iloc[-(n*2):-n].mean()
        if hist_turnover <= 0:
            return 0.0
        turnover_ratio = recent_turnover / hist_turnover
        ma_period = {ma_period}
        price_position = closes.iloc[-1] / closes.iloc[-ma_period:].mean() - 1.0
        import math as _math
        score = _math.tanh((turnover_ratio - 1.0) * 2.0 + price_position * 4.0)
        return round(float(score), 4)
    except Exception:
        return 0.0''',
        "qlib_expression": (
            "(Mean($turnover, {n}) / Mean($turnover, {n}*2) - 1) * "
            "($close / Mean($close, {ma_period}) - 1)"
        ),
    },
})

# ── 传统因子集 → Qlib 单因子表达式映射 ──
# 这些 traditional key 原只有 label/active（因子源占位），无 Qlib 表达式，
# 训练时被 resolve_factor_expression 跳过。补单因子表达式使其可参与 Qlib 训练。
_reg({
    "momentum": {
        "category": "价格动量",
        "label": "动量因子",
        "dataflow": "price_only",
        "logic_desc": "20日价格动量，追涨。",
        "qlib_expression": "$close / Ref($close, -20) - 1",
    },
    "volatility": {
        "category": "波动率特征",
        "label": "波动率",
        "dataflow": "price_only",
        "logic_desc": "20日收益波动率，风险度量。",
        "qlib_expression": "Std($close / Ref($close, -1) - 1, 20)",
    },
    "size": {
        "category": "规模",
        "label": "规模因子(SMB)",
        "dataflow": "price_with_db",
        "logic_desc": "小市值溢价（总市值对数取负）。",
        "qlib_expression": "(0 - FundVal($close, 1))",
    },
    "value": {
        "category": "估值",
        "label": "价值因子(HML)",
        "dataflow": "price_with_db",
        "logic_desc": "低市净率估值（PB取负）。",
        "qlib_expression": "(0 - FundVal($close, 4))",
    },
    "quality": {
        "category": "盈利",
        "label": "质量因子(QMJ)",
        "dataflow": "price_with_db",
        "logic_desc": "高ROE盈利质量。",
        "qlib_expression": "FundVal($close, 5)",
    },
    "size_turnover": {
        "category": "流动性",
        "label": "小市值换手",
        "dataflow": "price_with_db",
        "logic_desc": "小市值叠加高换手，散户/游资活跃标的。",
        "qlib_expression": "(0 - FundVal($close, 1)) * FundVal($close, 9)",
    },
    "tech": {
        "category": "技术形态",
        "label": "技术指标",
        "dataflow": "price_only",
        "logic_desc": "20日RSI，超买超卖技术指标。",
        "qlib_expression": "Sum(Max($close / Ref($close, -1) - 1, 0), 20) / (Sum(Abs($close / Ref($close, -1) - 1), 20) + 1e-8)",
    },
})


# ══════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════

def get_template(key):
    return FACTOR_TEMPLATES.get(key)

def get_templates_by_category(category=None):
    if category:
        return {k: v for k, v in FACTOR_TEMPLATES.items() if v.get("category") == category}
    return dict(FACTOR_TEMPLATES)

def get_all_template_keys():
    return list(FACTOR_TEMPLATES.keys())

def get_categories():
    cats = set(t.get("category") for t in FACTOR_TEMPLATES.values())
    return sorted(cats)
