"""
动态权重映射表 — 不同市场状态下推荐使用的信号权重和技术预设
"""
from core.domain.regime.types import Regime

REGIME_WEIGHTS = {
    # 2×3 网格: 波动率 (高/低) × 方向 (涨/跌/平)
    Regime.VOLATILE_UP: {
        "qlib": 0.35,
        "tech": 0.08,
        "sentiment": 0.10,
        "fundamental": 0.08,
        "flow": 0.08,
        "north_flow": 0.12,
        "size_turnover": 0.06,
        "volatility": 0.08,
        "illiq": 0.00, "overnight_gap": 0.00,
        "ht_factors": 0.00, "wb_industry": 0.02, "wb_divergence": 0.01, "wb_moneyflow": 0.02,
        "momentum": 0.00,
    },
    Regime.VOLATILE_DOWN: {
        "qlib": 0.30,
        "tech": 0.08,
        "sentiment": 0.08,
        "fundamental": 0.20,
        "flow": 0.06,
        "north_flow": 0.08,
        "size_turnover": 0.08,
        "volatility": 0.10,
        "illiq": 0.00, "overnight_gap": 0.00,
        "ht_factors": 0.00, "wb_industry": 0.01, "wb_divergence": 0.00, "wb_moneyflow": 0.01,
        "momentum": 0.00,
    },
    Regime.VOLATILE_FLAT: {
        "qlib": 0.32,
        "tech": 0.06,
        "sentiment": 0.08,
        "fundamental": 0.20,
        "flow": 0.08,
        "north_flow": 0.08,
        "size_turnover": 0.08,
        "volatility": 0.08,
        "illiq": 0.00, "overnight_gap": 0.00,
        "ht_factors": 0.00, "wb_industry": 0.01, "wb_divergence": 0.00, "wb_moneyflow": 0.01,
        "momentum": 0.00,
    },
    Regime.STEADY_UP: {
        "qlib": 0.38,
        "tech": 0.10,
        "sentiment": 0.10,
        "fundamental": 0.08,
        "flow": 0.05,
        "north_flow": 0.10,
        "size_turnover": 0.05,
        "volatility": 0.04,
        "illiq": 0.00, "overnight_gap": 0.00,
        "ht_factors": 0.00, "wb_industry": 0.05, "wb_divergence": 0.00, "wb_moneyflow": 0.01,
        "momentum": 0.00,
    },
    Regime.STEADY_DOWN: {
        "qlib": 0.30,
        "tech": 0.08,
        "sentiment": 0.08,
        "fundamental": 0.22,
        "flow": 0.06,
        "north_flow": 0.08,
        "size_turnover": 0.08,
        "volatility": 0.08,
        "illiq": 0.00, "overnight_gap": 0.00,
        "ht_factors": 0.00, "wb_industry": 0.01, "wb_divergence": 0.00, "wb_moneyflow": 0.01,
        "momentum": 0.00,
    },
    Regime.STEADY_FLAT: {
        "qlib": 0.30,
        "tech": 0.18,
        "sentiment": 0.08,
        "fundamental": 0.08,
        "flow": 0.10,
        "north_flow": 0.10,
        "size_turnover": 0.08,
        "volatility": 0.06,
        "illiq": 0.00, "overnight_gap": 0.00,
        "ht_factors": 0.00, "wb_industry": 0.01, "wb_divergence": 0.00, "wb_moneyflow": 0.01,
        "momentum": 0.00,
    },
}

REGIME_TECH_PRESETS = {
    Regime.VOLATILE_UP: "trend",
    Regime.VOLATILE_DOWN: "reversal",
    Regime.VOLATILE_FLAT: "reversal",
    Regime.STEADY_UP: "momentum",
    Regime.STEADY_DOWN: "all",
    Regime.STEADY_FLAT: "all",
}

REGIME_SKIPS = {
    Regime.VOLATILE_UP: {},
    Regime.VOLATILE_DOWN: {},
    Regime.VOLATILE_FLAT: {},
    Regime.STEADY_UP: {},
    Regime.STEADY_DOWN: {},
    Regime.STEADY_FLAT: {},
}

REGIME_LABELS = {
    Regime.VOLATILE_UP: "高波快涨",
    Regime.VOLATILE_DOWN: "高波快跌",
    Regime.VOLATILE_FLAT: "高波震荡",
    Regime.STEADY_UP: "稳步上涨",
    Regime.STEADY_DOWN: "稳步下跌",
    Regime.STEADY_FLAT: "低波盘整",
}

REGIME_RATIONALE = {
    Regime.VOLATILE_UP: "高波快涨：方向向上但波动大，Qlib+北向主导跟随趋势，控制仓位防回撤",
    Regime.VOLATILE_DOWN: "高波快跌：波动大且方向向下，基本面防御+低波筛选抗跌，观望为主",
    Regime.VOLATILE_FLAT: "高波震荡：方向不明但波动大，降低技术依赖，基本面+低波提供安全边际",
    Regime.STEADY_UP: "稳步上涨：A股最佳做多窗口，Qlib+动量双驱动，可适度提高仓位",
    Regime.STEADY_DOWN: "稳步下跌：阴跌不止，基本面防御优先，降低动量和技术依赖，等待右侧信号",
    Regime.STEADY_FLAT: "低波盘整：窄幅震荡，技术指标捕捉超买超卖，RSI/布林带做反转交易",
}


# 市场状态 → 策略方向精选集
REGIME_DIRECTION = {
    Regime.VOLATILE_UP: "trend_momentum",     # 高波向上 → 趋势动量
    Regime.VOLATILE_DOWN: "oversold_rebound",  # 高波快跌 → 超卖反弹
    Regime.VOLATILE_FLAT: "volume_breakout",   # 高波震荡 → 量价突破 (回测最优: 9/12胜)
    Regime.STEADY_UP: "trend_momentum",        # 稳步上涨 → 趋势动量
    Regime.STEADY_DOWN: "volume_breakout",     # 稳步下跌 → 量价突破
    Regime.STEADY_FLAT: "oversold_rebound",    # 低波盘整 → 超卖反弹 (回测最优: 均值最高)
}


def get_regime_direction(regime):
    """返回某市场状态对应的策略方向精选集名称"""
    if isinstance(regime, str):
        regime = Regime(regime)
    return REGIME_DIRECTION.get(regime, "trend_momentum")


def get_vol_direction(p_vol, vol_threshold=0.5, market_breadth=None, oversold_degree=None,
                      mid_lo=0.36, mid_hi=0.59, breadth_th=0.40, oversold_th=-0.01,
                      use_conditional=False):
    """波动率 → 方向模型: 高波 oversold_rebound(防御), 低波 trend_momentum(进攻)。

    替代 REGIME_DIRECTION 的 6 状态方向映射 (方向不可预测)。

    use_conditional=True 时按 OOS 实验结论 (docs/verify_fixed_vs_switch_env_subset.md §5) 升级为三档:
      - p_vol > mid_hi (高波): 仅当 宽度低 或 超跌深 → oversold_rebound; 否则 trend(少亏)
      - p_vol < mid_lo (低波): 无超跌 → trend; 深超跌 → oversold_rebound(少亏)
      - 中波死亡区 (mid_lo~mid_hi): 两个模型都亏 → 返回 "neutral" (建议空仓, 不使用模型)
    阈值默认来自实验数据: 波动率三档 [0.36,0.59], 宽度 0.40, 超跌 -0.01。
    缺宽度/超跌时回退原二值逻辑。
    """
    if p_vol is None:
        return "trend_momentum"
    if use_conditional and market_breadth is not None and oversold_degree is not None:
        if p_vol > mid_hi:
            if market_breadth <= breadth_th or oversold_degree >= oversold_th:
                return "oversold_rebound"
            return "trend_momentum"
        if p_vol < mid_lo:
            if oversold_degree <= oversold_th:
                return "oversold_rebound"
            return "trend_momentum"
        return "neutral"  # 中波死亡区: 空仓, 不使用模型
    return "oversold_rebound" if (p_vol is not None and p_vol > vol_threshold) else "trend_momentum"


# 规则分类器 4 状态 → HMM 6 状态的近似映射，兼容 market_regime_history 历史数据
_RULE_REGIME_FALLBACK = {
    "trend_momentum": ["trending_up"],
    "oversold_rebound": ["trending_down", "ranging"],
    "volume_breakout": ["volatile"],
    "value_defense": ["trending_down"],
}


def get_direction_regimes(direction_name):
    """返回某策略方向对应的所有市场状态值列表，用于 regime-aware ICIR 计算。
    包含 HMM 6 状态 + 规则分类器 4 状态回退，兼容历史数据。
    """
    regimes = []
    for regime, direction in REGIME_DIRECTION.items():
        if direction == direction_name:
            regimes.append(regime.value)
    regimes.extend(_RULE_REGIME_FALLBACK.get(direction_name, []))
    return regimes


def get_regime_weights(regime, sector_bias=None):
    """返回某市场状态对应的 source_weights dict"""
    if isinstance(regime, str):
        regime = Regime(regime)
    base = dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.STEADY_FLAT]))

    if sector_bias == "偏多" and regime != Regime.VOLATILE_UP:
        base["north_flow"] = base.get("north_flow", 0.10) + 0.03
        base["momentum"] = base.get("momentum", 0.10) + 0.02
    elif sector_bias == "偏空" and regime != Regime.VOLATILE_DOWN:
        base["volatility"] = base.get("volatility", 0.10) + 0.03
        base["fundamental"] = base.get("fundamental", 0.10) + 0.02
    elif sector_bias == "均衡":
        pass

    total = sum(v for v in base.values() if v > 0)
    if total > 0:
        base = {k: round(v / total, 3) for k, v in base.items()}

    return base


def get_regime_tech_preset(regime):
    if isinstance(regime, str):
        regime = Regime(regime)
    return REGIME_TECH_PRESETS.get(regime, "all")


def get_regime_skips(regime):
    if isinstance(regime, str):
        regime = Regime(regime)
    return dict(REGIME_SKIPS.get(regime, {}))


# 策略方向 → 活跃信号源（未列出的信号源在该方向下权重归零）
# 注: momentum/illiq/overnight_gap/ht_factors 已从权重表归零, 不再出现在任何方向
DIRECTION_ACTIVE_SOURCES = {
    "trend_momentum": {
        "qlib", "tech", "north_flow", "sentiment", "flow",
        "fundamental", "wb_industry", "wb_moneyflow",
    },
    "oversold_rebound": {
        "qlib", "volatility", "flow", "size_turnover",
        "sentiment", "wb_divergence", "north_flow",
    },
    "volume_breakout": {
        "qlib", "tech", "flow", "volatility", "size_turnover",
        "wb_industry", "wb_divergence", "wb_moneyflow",
    },
    "value_defense": {
        "qlib", "fundamental", "quality", "sentiment",
        "north_flow", "wb_industry",
    },
}

DIRECTION_RATIONALE = {
    "trend_momentum": (
        "趋势市中动量因子主导，跟随趋势不抄底。"
        "关闭低波(趋势中低波股滞后)、基本面(价值陷阱)、反转信号(逆势)。"
    ),
    "oversold_rebound": (
        "高波动中识别超卖反弹：波动率+流动性捕捉恐慌放量，"
        "北向资金跟踪聪明钱抄底。关闭动量(急跌中无效)、基本面(短期不兑现)。"
    ),
    "volume_breakout": (
        "震荡市中量价信号最有效：成交量+资金流捕捉突破，"
        "技术指标全开。关闭动量(震荡中反复止损)、基本面(短期无效)。"
    ),
    "value_defense": (
        "下跌市中基本面防御主导：低估值+高质量+稳定现金流。"
        "关闭动量(下跌中无效)、技术指标(趋势向下失效)，聚焦价值锚定。"
    ),
}


def get_direction_active_sources(direction_name):
    """返回某策略方向下应保持活跃的信号源集合"""
    return DIRECTION_ACTIVE_SOURCES.get(direction_name, set())


def filter_weights_by_direction(weights, direction_name):
    """按策略方向过滤信号源权重：未入选源归零，入选源按原比例重新归一化"""
    active = DIRECTION_ACTIVE_SOURCES.get(direction_name)
    if not active:
        return dict(weights)
    filtered = {k: (v if k in active else 0.0) for k, v in weights.items()}
    total = sum(filtered.values())
    if total > 0:
        filtered = {k: round(v / total, 3) for k, v in filtered.items()}
    return filtered


def get_macro_risk_multiplier():
    """宏观风险乘数, 叠加到各信号源权重上。高风险时整体降权。"""
    try:
        from core.domain.macro.analyzer import MacroAnalyzer
        risk = MacroAnalyzer().calculate_macro_risk_score()
        score = risk.get("score", 0)
        if score < 0.4:
            return 1.0
        elif score < 0.7:
            return 0.85
        else:
            return 0.65
    except Exception:
        return 1.0


# ── 软融合: 状态概率分布 → 加权权重 ──

# 2×3 网格: 0=volatile_up, 1=volatile_down, 2=volatile_flat, 3=steady_up, 4=steady_down, 5=steady_flat
_STATE_TO_REGIME = {
    0: Regime.VOLATILE_UP,
    1: Regime.VOLATILE_DOWN,
    2: Regime.VOLATILE_FLAT,
    3: Regime.STEADY_UP,
    4: Regime.STEADY_DOWN,
    5: Regime.STEADY_FLAT,
}

STATE_RATIO_MAP = {
    0: 1.50,  # volatile_up
    1: 0.15,  # volatile_down
    2: 0.50,  # volatile_flat
    3: 1.50,  # steady_up
    4: 0.15,  # steady_down
    5: 0.50,  # steady_flat
}


def get_soft_regime_weights(probs):
    """状态概率分布 → 软加权信号源权重

    Args:
        probs: dict or list/array of 6 probabilities
    """
    if isinstance(probs, dict):
        result = {}
        for state_name, prob in probs.items():
            try:
                regime = Regime(state_name)
            except ValueError:
                continue
            w = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.STEADY_FLAT])
            for source, weight in w.items():
                result[source] = result.get(source, 0.0) + float(prob) * weight
        total = sum(result.values())
        return {k: round(v / total, 4) for k, v in result.items()} if total > 0 else result

    result = {}
    for i, prob in enumerate(probs):
        if i >= 6:
            break
        regime = _STATE_TO_REGIME.get(i, Regime.STEADY_FLAT)
        w = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.STEADY_FLAT])
        for source, weight in w.items():
            result[source] = result.get(source, 0.0) + float(prob) * weight
    total = sum(result.values())
    if total > 0:
        result = {k: round(v / total, 4) for k, v in result.items()}
    return result


def get_soft_position_ratio(probs):
    """状态概率分布 → 加权仓位系数"""
    if isinstance(probs, dict):
        state_map = {"volatile_up": 1.50, "volatile_down": 0.15, "volatile_flat": 0.50,
                     "steady_up": 1.50, "steady_down": 0.15, "steady_flat": 0.50}
        ratio = 0.0
        for state_name, prob in probs.items():
            ratio += float(prob) * state_map.get(state_name, 0.5)
        return round(ratio, 4)

    ratio = 0.0
    for i, prob in enumerate(probs):
        if i >= 6:
            break
        ratio += float(prob) * STATE_RATIO_MAP.get(i, 0.5)
    return round(ratio, 4)
