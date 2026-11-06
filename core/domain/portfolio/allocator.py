"""
统一持仓分配引擎.

职责: 权重分配 + 止盈止损信号 + 换手率约束.
不负责: 选股、订单生成、涨跌停判断、持有区判断、冷冻期/现金管理.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def compute_one_way_turnover(target_weights, current_weights):
    target_weights = target_weights or {}
    current_weights = current_weights or {}
    all_codes = set(target_weights) | set(current_weights)
    stock_turnover = sum(
        abs(target_weights.get(code, 0) - current_weights.get(code, 0))
        for code in all_codes
    )
    cash_turnover = abs(sum(target_weights.values()) - sum(current_weights.values()))
    return (stock_turnover + cash_turnover) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AllocationConfig:
    sizing: str = "score_weighted"
    score_key: str = "score"
    score_power: float = 2.0

    max_single_pct: float = 0.20
    max_sector_pct: float = 0.30
    single_floor_pct: float = 0.0
    target_sum: float = 1.0
    min_weight_threshold: float = 0.005
    apply_double_cap: bool = True

    impact_cost_lambda: float = 0.0

    max_turnover: float = 0.30
    turnover_cap_single_at_end: bool = True

    stop_loss_warn: float = -0.05
    stop_loss_full: float = -0.10
    take_profit_half: float = 0.15
    take_profit_full: float = 0.30

    cooldown_stop_periods: int = 2
    cooldown_profit_periods: int = 1

    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    stamp_duty: float = 0.0005

    cash_yield_annual: float = 0.03
    frequency_days: int = 7
    tolerance_band: float = 0.12

    max_redistribution_rounds: int = 20

    @property
    def buy_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate

    @property
    def sell_cost_rate(self) -> float:
        return self.commission_rate + self.stamp_duty + self.slippage_rate


# ═══════════════════════════════════════════════════════════════════════════
# 结果类型
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RiskAction:
    code: str
    action: str = "none"
    reason: str = ""
    new_cooldown: int = 0
    trigger_price: Optional[float] = None


def classify_risk_action(pnl_pct, stop_loss_warn, stop_loss_full,
                         take_profit_half, take_profit_full, already_half=False):
    """阶梯止盈止损判断。纯 L3 函数——不依赖 DB/文件/时间函数。

    优先级: 止损清仓 > 止盈清仓 > 止损减半 > 止盈减半。
    already_half=True 时不再触发减半 (已减半，只判断清仓档)。

    返回 (action, reason)，action ∈ {"full", "half", None}。
    """
    if already_half:
        if pnl_pct <= stop_loss_full:
            return "full", f"止损清仓 (浮亏 {pnl_pct:.1%} ≤ {stop_loss_full:.0%})"
        if pnl_pct >= take_profit_full:
            return "full", f"止盈清仓 (浮盈 +{pnl_pct:.1%} ≥ +{take_profit_full:.0%})"
    else:
        if pnl_pct <= stop_loss_full:
            return "full", f"止损清仓 (浮亏 {pnl_pct:.1%} ≤ {stop_loss_full:.0%})"
        if pnl_pct >= take_profit_full:
            return "full", f"止盈清仓 (浮盈 +{pnl_pct:.1%} ≥ +{take_profit_full:.0%})"
        if pnl_pct <= stop_loss_warn:
            return "half", f"止损减半 (浮亏 {pnl_pct:.1%} ≤ {stop_loss_warn:.0%})"
        if pnl_pct >= take_profit_half:
            return "half", f"止盈减半 (浮盈 +{pnl_pct:.1%} ≥ +{take_profit_half:.0%})"
    return None, ""


def classify_dynamic_trailing(high_series, current_low, lookback=20, trail_full=0.05,
                              trail_half=None, already_half=False):
    """动态 trailing stop 判断。纯 L3 函数——不依赖 DB/文件/时间函数。

    止损线 = 近 lookback 日最高价 * (1 - trail_full)
    减半线 = 近 lookback 日最高价 * (1 - trail_half)  (默认 trail_full/2)

    Args:
        high_series: 截至前一交易日的历史最高价序列 (不含当前日, 防前视)
        current_low: 当前交易日最低价 (触发判定用)
        already_half: 已减半时不再触发减半, 只判清仓档

    返回 (action, reason)，action ∈ {"full", "half", None}。
    """
    if trail_half is None:
        trail_half = trail_full / 2
    if high_series is None:
        return None, ""
    window = [float(v) for v in list(high_series)[-lookback:] if v is not None and float(v) > 0]
    if not window:
        return None, ""
    highest = max(window)
    if current_low is None or current_low <= 0:
        return None, ""
    if current_low <= highest * (1 - trail_full):
        return "full", f"动态回撤清仓 (跌破近{lookback}日最高回撤{trail_full:.1%}线)"
    if not already_half and current_low <= highest * (1 - trail_half):
        return "half", f"动态回撤减半 (跌破近{lookback}日最高回撤{trail_half:.1%}线)"
    return None, ""


def compute_atr(high, low, close, period=14):
    if period < 1:
        return None
    prices = []
    for h, l, c in zip(high, low, close):
        try:
            hv, lv, cv = float(h), float(l), float(c)
        except (TypeError, ValueError):
            continue
        if math.isnan(hv) or math.isnan(lv) or math.isnan(cv):
            continue
        prices.append((hv, lv, cv))
    if len(prices) < period + 1:
        return None
    trs = []
    for i, (h, l, c) in enumerate(prices[1:], start=1):
        if h <= 0 or l <= 0:
            continue
        prev_close = prices[i - 1][2]
        if prev_close <= 0:
            prev_close = h
        trs.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def classify_atr_stop(entry_price, atr, atr_mult, current_low):
    if not entry_price or entry_price <= 0:
        return None, ""
    if not atr or atr <= 0:
        return None, ""
    if not atr_mult or atr_mult <= 0:
        return None, ""
    if current_low is None or current_low <= 0:
        return None, ""
    stop_price = entry_price - atr_mult * atr
    if current_low <= stop_price:
        return "full", f"ATR止损清仓 (最低价 {current_low:.2f} 跌破 {stop_price:.2f})"
    return None, ""


@dataclass
class AllocationResult:
    target_weights: dict
    risk_actions: dict
    turnover: float = 0.0
    diagnostics: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# PositionAllocator
# ═══════════════════════════════════════════════════════════════════════════

class PositionAllocator:

    def __init__(self, config: Optional[AllocationConfig] = None):
        self.config = config or AllocationConfig()

    # ── 公共 API ──

    def allocate(self, candidates, current_weights, sector_map=None,
                 cooldown=None, timing_ratio=1.0):
        if not candidates:
            return AllocationResult({}, {})

        if cooldown is None:
            cooldown = {}

        raw = self._compute_raw_weights(candidates)
        raw = self._apply_impact_penalty(raw, candidates)

        if self.config.single_floor_pct > 0:
            raw = self._apply_clamp_and_redistribute(raw)

        raw = {k: v for k, v in raw.items()
               if k not in cooldown or cooldown[k] <= 0}

        raw = self._apply_position_caps(raw, sector_map)
        raw = self._normalize_with_timing(raw, timing_ratio)

        if self.config.apply_double_cap:
            raw = self._apply_position_caps(raw, sector_map)

        raw = self._apply_turnover_limit(raw, current_weights)

        return AllocationResult(
            target_weights=raw,
            risk_actions={},
            turnover=self.compute_turnover(raw, current_weights),
        )

    def allocate_weights_only(self, candidates, current_weights=None,
                               sector_map=None, cooldown=None):
        if not candidates:
            return {}

        if cooldown is None:
            cooldown = {}
        if current_weights is None:
            current_weights = {}

        raw = self._compute_raw_weights(candidates)
        raw = self._apply_impact_penalty(raw, candidates)

        if self.config.single_floor_pct > 0:
            raw = self._apply_clamp_and_redistribute(raw)

        raw = {k: v for k, v in raw.items()
               if k not in cooldown or cooldown[k] <= 0}

        raw = self._apply_position_caps(raw, sector_map)
        raw = self._normalize_with_timing(raw, 1.0)

        if self.config.apply_double_cap:
            raw = self._apply_position_caps(raw, sector_map)

        return raw

    def classify_risk(self, holdings_info, half_sold=None,
                       use_intraday_low_high=True):
        if half_sold is None:
            half_sold = set()

        actions = {}
        for h in holdings_info:
            code = h["code"]
            already_half = code in half_sold

            if use_intraday_low_high:
                action = self._classify_by_intraday(h, already_half)
            else:
                action = self._classify_by_pnl(h, already_half)

            if action:
                actions[code] = action

        return actions

    # ── 步骤1: 原始权重计算 ──

    def _compute_raw_weights(self, candidates):
        n = len(candidates)
        cfg = self.config

        if cfg.sizing == "equal":
            w = cfg.target_sum / n
            return {c.get("code"): w for c in candidates}

        if cfg.sizing == "risk_parity" and n >= 3:
            rp = self._risk_parity_weights(candidates)
            if rp:
                return rp
            logger.info("[Allocator] 风险平价失败, 回退到 score_weighted")

        if cfg.sizing == "proportional":
            scores = [max(0, c.get(cfg.score_key, 0)) for c in candidates]
            total = sum(scores)
            if total <= 0:
                w = cfg.target_sum / n
                return {c.get("code"): w for c in candidates}
            return {c.get("code"): s / total * cfg.target_sum
                    for c, s in zip(candidates, scores)}

        total = sum(max(c.get(cfg.score_key, 0), 0.001) ** cfg.score_power
                    for c in candidates)
        if total <= 0:
            w = cfg.target_sum / n
            return {c.get("code"): w for c in candidates}
        return {c.get("code"): max(c.get(cfg.score_key, 0), 0.001) ** cfg.score_power / total
                * cfg.target_sum
                for c in candidates}

    # ── 步骤1b: 冲击成本惩罚 ──

    def _apply_impact_penalty(self, weights, candidates):
        lam = self.config.impact_cost_lambda
        if lam <= 0:
            return weights
        result = dict(weights)
        for c in candidates:
            code = c.get("code")
            impact = c.get("impact_cost", 0)
            if impact > 0 and code in result:
                penalty = lam * math.tanh(impact * 20)
                result[code] = max(0, result[code] - penalty * 0.05)
        des_sum = sum(result.values())
        if des_sum > 0:
            result = {k: v / des_sum * self.config.target_sum for k, v in result.items()}
        return result

    # ── 步骤2: 单只上下限夹逼 + 迭代重分配 ──

    def _apply_clamp_and_redistribute(self, weights):
        if not weights:
            return weights
        lo = self.config.single_floor_pct
        hi = self.config.max_single_pct
        target = self.config.target_sum

        result = {}
        for code, w in weights.items():
            result[code] = max(lo, min(hi, w))

        cur_sum = sum(result.values())
        if abs(cur_sum - target) < 1e-6:
            return result

        for _ in range(self.config.max_redistribution_rounds):
            free_codes = [c for c in result if lo < result[c] < hi]
            if not free_codes:
                break
            gap = target - sum(result.values())
            if abs(gap) < 1e-6:
                break
            adj = gap / len(free_codes)
            for c in free_codes:
                result[c] = max(lo, min(hi, result[c] + adj))
            if abs(target - sum(result.values())) < 1e-6:
                break

        return result

    # ── 步骤3: 仓位上限裁剪 (单只 + 行业) ──

    def _apply_position_caps(self, weights, sector_map):
        if not weights:
            return weights

        result = dict(weights)
        orig_sum = sum(weights.values())

        for _ in range(10):
            for c in result:
                if result[c] > self.config.max_single_pct:
                    result[c] = self.config.max_single_pct

            sector_totals = {}
            for c, w in result.items():
                ind = self._get_industry(c, sector_map)
                sector_totals[ind] = sector_totals.get(ind, 0) + w

            for ind, sw in sector_totals.items():
                if sw > self.config.max_sector_pct and sw > 0:
                    scale = self.config.max_sector_pct / sw
                    for c in result:
                        if self._get_industry(c, sector_map) == ind:
                            result[c] *= scale

            s = sum(result.values())
            if s > 0:
                result = {k: v / s * orig_sum for k, v in result.items()}

            worst_sector = max(
                (sum(w for cc, w in result.items()
                     if self._get_industry(cc, sector_map) == ind)
                 for ind in set(self._get_industry(c, sector_map) for c in result)),
                default=0)
            worst_single = max(result.values(), default=0)
            if (worst_sector <= self.config.max_sector_pct + 0.005 and
                    worst_single <= self.config.max_single_pct + 0.005):
                break

        return result

    @staticmethod
    def _get_industry(code, sector_map):
        if not sector_map:
            return ""
        v = sector_map.get(code) or sector_map.get(str(code))
        if isinstance(v, dict):
            return v.get("industry", "")
        return ""

    # ── 步骤4: 归一化到 timing_ratio ──

    def _normalize_with_timing(self, weights, timing_ratio):
        if not weights:
            return {}
        target = self.config.target_sum * timing_ratio
        if target <= 0:
            return {}
        w_sum = sum(weights.values())
        if w_sum <= 0:
            return {}
        return {k: v / w_sum * target for k, v in weights.items()}

    # ── 步骤5: 换手率约束 ──

    def _apply_turnover_limit(self, target_weights, current_weights):
        if not target_weights:
            return target_weights
        if not current_weights:
            return target_weights

        all_codes = set(list(target_weights.keys()) + list(current_weights.keys()))
        raw_turnover = compute_one_way_turnover(target_weights, current_weights)

        if raw_turnover <= self.config.max_turnover:
            return target_weights

        scale = self.config.max_turnover / raw_turnover
        result = {}
        for c in all_codes:
            cur = current_weights.get(c, 0)
            tgt = target_weights.get(c, 0)
            result[c] = cur + (tgt - cur) * scale

        return {k: v for k, v in result.items() if v > 1e-12}

    # ── 止盈止损分类 ──

    def _classify_by_intraday(self, h, already_half):
        entry = h.get("entry_price", 0)
        if not entry or entry <= 0:
            return None

        p_low = h.get("p_low")
        p_high = h.get("p_high")
        ret_low = (p_low / entry - 1) if p_low is not None else None
        ret_high = (p_high / entry - 1) if p_high is not None else None

        code = h["code"]
        cfg = self.config

        if already_half:
            if ret_low is not None and ret_low <= cfg.stop_loss_full:
                return RiskAction(code, "full",
                                  f"止损清仓 (最低价触及 {cfg.stop_loss_full:.0%})",
                                  cfg.cooldown_stop_periods)
            if ret_high is not None and ret_high >= cfg.take_profit_full:
                return RiskAction(code, "full",
                                  f"止盈清仓 (最高价触及 +{cfg.take_profit_full:.0%})",
                                  cfg.cooldown_profit_periods)
        else:
            if ret_low is not None and ret_low <= cfg.stop_loss_full:
                return RiskAction(code, "full",
                                  f"止损清仓 (最低价触及 {cfg.stop_loss_full:.0%})",
                                  cfg.cooldown_stop_periods)
            if ret_high is not None and ret_high >= cfg.take_profit_full:
                return RiskAction(code, "full",
                                  f"止盈清仓 (最高价触及 +{cfg.take_profit_full:.0%})",
                                  cfg.cooldown_profit_periods)
            if ret_low is not None and ret_low <= cfg.stop_loss_warn:
                return RiskAction(code, "half",
                                  f"止损减半 (最低价触及 {cfg.stop_loss_warn:.0%})")
            if ret_high is not None and ret_high >= cfg.take_profit_half:
                return RiskAction(code, "half",
                                  f"止盈减半 (最高价触及 +{cfg.take_profit_half:.0%})")
        return None

    def _classify_by_pnl(self, h, already_half):
        pnl_pct = h.get("pnl_pct", 0)
        code = h["code"]
        cfg = self.config

        action, reason = classify_risk_action(
            pnl_pct, cfg.stop_loss_warn, cfg.stop_loss_full,
            cfg.take_profit_half, cfg.take_profit_full,
            already_half=already_half,
        )
        if action is None:
            return None
        new_cooldown = (cfg.cooldown_stop_periods if action == "full" and pnl_pct <= cfg.stop_loss_full
                        else cfg.cooldown_profit_periods if action == "full" else 0)
        return RiskAction(code, action, reason, new_cooldown)

    # ── 风险平价 ──

    def _risk_parity_weights(self, candidates):
        try:
            import numpy as np
            from scipy.optimize import minimize
        except ImportError:
            return None

        codes = [c.get("code") for c in candidates]
        returns_data = [c.get("returns") for c in candidates]
        if not all(r is not None and len(r) >= 20 for r in returns_data):
            logger.info("[Allocator] 风险平价数据不足, 回退")
            return None

        try:
            returns_matrix = np.column_stack([np.array(r[-60:]) for r in returns_data])
            cov = self._ledoit_wolf_shrinkage(returns_matrix)
            n = len(codes)
            w0 = np.ones(n) / n

            def risk_budget_objective(w):
                sigma = np.sqrt(w @ cov @ w)
                risk_contrib = w * (cov @ w) / sigma
                risk_budget = sigma / n
                return np.sum((risk_contrib - risk_budget) ** 2)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0.0, self.config.max_single_pct) for _ in range(n)]
            result = minimize(risk_budget_objective, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 200, "ftol": 1e-8})
            if result.success:
                target_sum = self.config.target_sum
                return {codes[i]: max(0.0, float(result.x[i])) * target_sum
                        for i in range(n)}
        except Exception:
            pass
        return None

    @staticmethod
    def _ledoit_wolf_shrinkage(returns_matrix):
        try:
            import numpy as np
        except ImportError:
            return np.cov(returns_matrix.T)

        try:
            from pypfopt import risk_models
            return risk_models.CovarianceShrinkage(
                returns_matrix, frequency=252).ledoit_wolf()
        except ImportError:
            pass

        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns_matrix)
            return lw.covariance_ * 252
        except ImportError:
            return np.cov(returns_matrix.T)

    # ── 辅助 ──

    @staticmethod
    def compute_turnover(target, current):
        return compute_one_way_turnover(target, current)


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_REGIME_STOPS = {
    0: {"warn": 4, "full": 8, "profitHalf": 12, "profitFull": 25},
    1: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 25},
    2: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
    3: {"warn": 7, "full": 14, "profitHalf": 20, "profitFull": 40},
    4: {"warn": 6, "full": 12, "profitHalf": 18, "profitFull": 35},
    5: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
}


def load_regime_stops():
    """从 shared/config 加载每状态的止盈止损参数 (委托给 _portfolio.load_regime_stops)."""
    from core.shared.config._portfolio import load_regime_stops as _loader
    return _loader()


def get_regime_stop_params(regime_state=None, user_overrides=None):
    if user_overrides is None:
        user_overrides = {}

    stops_cfg = load_regime_stops()
    state = regime_state if isinstance(regime_state, int) else -1
    entry = stops_cfg.get(state, {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30})

    stop_warn = user_overrides.get("stop_loss_warn")
    stop_full = user_overrides.get("stop_loss_full")
    take_half = user_overrides.get("take_profit_half")
    take_full = user_overrides.get("take_profit_full")

    if stop_warn is not None or stop_full is not None or take_half is not None or take_full is not None:
        source = "user"
    elif state in stops_cfg:
        source = f"regime:{state}"
    else:
        source = "default"

    result = {
        "stop_loss_warn": -(entry["warn"] / 100.0) if stop_warn is None else stop_warn,
        "stop_loss_full": -(entry["full"] / 100.0) if stop_full is None else stop_full,
        "take_profit_half": entry["profitHalf"] / 100.0 if take_half is None else take_half,
        "take_profit_full": entry["profitFull"] / 100.0 if take_full is None else take_full,
        "source": source,
    }
    return result


def build_allocation_config(freq="week", cooldown_stop_weeks=None, cooldown_profit_weeks=None, **overrides):
    from core.shared import config as app_config
    from core.shared import get_freq_meta, cooldown_periods

    _freq_meta = get_freq_meta(freq)
    freq_days = _freq_meta.days

    # 显式参数 > env 覆盖 (0 = 不覆盖) > freq 默认派生
    if cooldown_stop_weeks is None:
        cooldown_stop_weeks = getattr(app_config, "PORTFOLIO_COOLDOWN_STOP_WEEKS", 0) or None
    if cooldown_profit_weeks is None:
        cooldown_profit_weeks = getattr(app_config, "PORTFOLIO_COOLDOWN_PROFIT_WEEKS", 0) or None
    cooldown_stop_periods, cooldown_profit_periods = cooldown_periods(
        freq, cooldown_stop_weeks, cooldown_profit_weeks)

    defaults = dict(
        sizing=getattr(app_config, "PORTFOLIO_DEFAULT_SIZING", "score_weighted"),
        max_single_pct=getattr(app_config, "PORTFOLIO_DEFAULT_MAX_SINGLE", 0.20),
        max_sector_pct=getattr(app_config, "PORTFOLIO_DEFAULT_MAX_SECTOR", 0.30),
        max_turnover=getattr(app_config, "PORTFOLIO_DEFAULT_MAX_TURNOVER", 0.30),
        target_sum=getattr(app_config, "PORTFOLIO_DEFAULT_TARGET_SUM", 1.0),
        single_floor_pct=getattr(app_config, "PORTFOLIO_DEFAULT_SINGLE_FLOOR", 0.0),
        stop_loss_warn=getattr(app_config, "PORTFOLIO_STOP_LOSS_WARN", -0.05),
        stop_loss_full=getattr(app_config, "PORTFOLIO_STOP_LOSS_FULL", -0.10),
        take_profit_half=getattr(app_config, "PORTFOLIO_TAKE_PROFIT_HALF", 0.15),
        take_profit_full=getattr(app_config, "PORTFOLIO_TAKE_PROFIT_FULL", 0.30),
        min_weight_threshold=getattr(app_config, "PORTFOLIO_MIN_WEIGHT_THRESHOLD", 0.005),
        commission_rate=getattr(app_config, "PORTFOLIO_COMMISSION_RATE", 0.0003),
        slippage_rate=getattr(app_config, "PORTFOLIO_SLIPPAGE_RATE", 0.0001),
        stamp_duty=getattr(app_config, "PORTFOLIO_STAMP_DUTY", 0.0005),
        impact_cost_lambda=(
            getattr(app_config, "PORTFOLIO_IMPACT_LAMBDA", 0.3)
            if getattr(app_config, "PORTFOLIO_IMPACT_ENABLED", True) else 0.0
        ),
        cash_yield_annual=getattr(app_config, "PORTFOLIO_CASH_YIELD_ANNUAL", 0.03),
        frequency_days=freq_days,
        cooldown_stop_periods=cooldown_stop_periods,
        cooldown_profit_periods=cooldown_profit_periods,
        apply_double_cap=True,
        turnover_cap_single_at_end=True,
    )
    defaults.update(overrides)
    return AllocationConfig(**defaults)
