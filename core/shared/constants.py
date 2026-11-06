"""
全局常量 — 所有散落在 V1 各处的常量集中于此，作为唯一来源。
"""
from dataclasses import dataclass

# ── 股票排除规则 ──
# 排除 ETF (上交所 51/52/56/58, 深交所 15/16), B股 (920/900/200)
EXCLUDED_PREFIXES = frozenset([
    "51", "52", "56", "58",  # 上交所 ETF
    "15", "16",               # 深交所 ETF
    "920", "900", "200",      # B股
])
B_SHARE_PREFIXES = frozenset(["920", "900", "200"])
ST_PATTERNS = ("ST", "*ST")

# ── 数据过滤 ──
MIN_LIQUIDITY_AMOUNT = 10_000_000  # 近20日均成交额阈值 (元)
MIN_TRADING_DAYS = 60              # 最少上市交易日

# ── 因子校验 ──
L5_CORR_THRESHOLD = 0.75           # 去冗余相关性阈值
L1_MIN_ABS_IC = 0.02               # Walk-Forward 最低 |Test IC|

# ── 反过拟合校验顺序 ──
OVERFITTING_LAYERS = ("L5", "L1", "L4", "L7", "L8")  # 一票否决早剪枝

# ── Qlib 训练 ──
LABEL_HORIZON = 20
LGB_PARAMS = {"lr": 0.03, "max_depth": 7, "num_leaves": 127}
# XGBoost: 不设 missing=-999.0 (Qlib 用 NaN 表示缺失)

# ── 调仓规则 ──
REBALANCE_TOP_HOLD = 5           # Top-5 → 持有
REBALANCE_TOP_WATCH = 10         # Top-6~10 → 观察降权
STOP_LOSS_THRESHOLD = -0.08      # 浮亏 ≥ 8% → 强制止损
MAX_TURNOVER = 0.50              # 最大换手率 50%
MAX_SINGLE_WEIGHT = 0.25         # 单只最大权重 25%
MAX_SECTOR_WEIGHT = 0.30         # 单行业最大权重 30%

# ── 通知 ──
WECOM_MSG_MAX_BYTES = 4096
RISK_DETAIL_BUDGET_RATIO = 0.75

# ── 情绪源熔断 ──
SENTIMENT_CIRCUIT_MAX_FAILS = 50

# ── 市场状态 ──
# 6 状态: volatile_up/down/flat, steady_up/down/flat
REGIME_STATES = [
    "volatile_up", "volatile_down", "volatile_flat",
    "steady_up", "steady_down", "steady_flat",
]
REGIME_LABELS = {
    "volatile_up": "高波上涨",
    "volatile_down": "高波下跌",
    "volatile_flat": "高波震荡",
    "steady_up": "低波上涨",
    "steady_down": "低波下跌",
    "steady_flat": "低波震荡",
}

# ── 择时 ──
TIMING_POSITION_MIN = 0.20       # 最低仓位
TIMING_POSITION_MAX = 1.00       # 最高仓位

# ── 调仓频率 (唯一来源) ──
@dataclass(frozen=True)
class FreqMeta:
    days: int              # 每期日历天数
    pd_freq: str           # pandas 日期锚
    stride: bool           # 是否走交易日 stride (day/2d/3d/4d)
    cooldown_stop: int     # 止损冷冻期数 (止损 2 周换算为期数)
    cooldown_profit: int   # 止盈冷冻期数 (止盈 1 周换算为期数)
    periods_per_year: int  # 年化期数
    hold_mult: int         # 持有区 Top-N 倍数


REBALANCE_FREQ = {
    "day":    FreqMeta(1,  "B",      True,  10, 5, 252, 8),
    "2d":     FreqMeta(2,  "2B",     True,  5,  3, 126, 8),
    "3d":     FreqMeta(3,  "3B",     True,  4,  2, 84,  8),
    "4d":     FreqMeta(4,  "4B",     True,  3,  2, 63,  8),
    "5d":     FreqMeta(5,  "5B",     True,  2,  1, 50,  8),
    "10d":    FreqMeta(10, "10B",    True,  1,  1, 25,  8),
    "20d":    FreqMeta(20, "20B",    True,  1,  1, 13,  8),
    "week":   FreqMeta(7,  "W-THU",  False, 2,  1, 52,  2),
    "biweek": FreqMeta(14, "2W-THU", False, 1,  1, 26,  2),
    "3w":     FreqMeta(21, "3W-THU", False, 1,  1, 17,  2),
    "month":  FreqMeta(21, "MS",     False, 1,  1, 12,  2),
}


def get_freq_meta(freq):
    """返回频率元数据；未知频率回退 week。"""
    return REBALANCE_FREQ.get(freq, REBALANCE_FREQ["week"])


def cooldown_periods(freq, stop_weeks=None, profit_weeks=None):
    """止损/止盈冷冻期数: 周数→期数换算, None = freq 默认派生。

    止损默认 2 周 (cooldown_stop), 止盈默认 1 周 (cooldown_profit)。
    """
    m = get_freq_meta(freq)
    stop = max(1, round(stop_weeks * m.cooldown_stop / 2)) if stop_weeks else m.cooldown_stop
    profit = max(1, round(profit_weeks * m.cooldown_profit)) if profit_weeks else m.cooldown_profit
    return stop, profit
