"""
市场状态枚举 — 独立模块, 无任何项目内依赖。
供 market_regime、regime_weights、regime_predictor 等共同引用。
"""
from enum import Enum


class Regime(Enum):
    VOLATILE_UP = "volatile_up"
    VOLATILE_DOWN = "volatile_down"
    VOLATILE_FLAT = "volatile_flat"
    STEADY_UP = "steady_up"
    STEADY_DOWN = "steady_down"
    STEADY_FLAT = "steady_flat"


HMM_TO_LEGACY = {
    Regime.STEADY_UP: Regime.STEADY_UP,
    Regime.VOLATILE_UP: Regime.VOLATILE_UP,
    Regime.STEADY_DOWN: Regime.STEADY_DOWN,
    Regime.VOLATILE_DOWN: Regime.VOLATILE_DOWN,
    Regime.VOLATILE_FLAT: Regime.VOLATILE_FLAT,
    Regime.STEADY_FLAT: Regime.STEADY_FLAT,
}
