import os

PORTFOLIO_DEFAULT_SIZING = os.getenv("PORTFOLIO_DEFAULT_SIZING", "score_weighted")

PORTFOLIO_DEFAULT_MAX_SINGLE = float(os.getenv("PORTFOLIO_DEFAULT_MAX_SINGLE", "0.20"))
PORTFOLIO_DEFAULT_MAX_SECTOR = float(os.getenv("PORTFOLIO_DEFAULT_MAX_SECTOR", "0.30"))
PORTFOLIO_DEFAULT_MAX_TURNOVER = float(os.getenv("PORTFOLIO_DEFAULT_MAX_TURNOVER", "0.30"))

PORTFOLIO_DEFAULT_TARGET_SUM = float(os.getenv("PORTFOLIO_DEFAULT_TARGET_SUM", "1.00"))
PORTFOLIO_DEFAULT_MIN_CASH = float(os.getenv("PORTFOLIO_DEFAULT_MIN_CASH", "0.00"))
PORTFOLIO_DEFAULT_SINGLE_FLOOR = float(os.getenv("PORTFOLIO_DEFAULT_SINGLE_FLOOR", "0.0"))

PORTFOLIO_STOP_LOSS_WARN = float(os.getenv("PORTFOLIO_STOP_LOSS_WARN", "-0.05"))
PORTFOLIO_STOP_LOSS_FULL = float(os.getenv("PORTFOLIO_STOP_LOSS_FULL", "-0.10"))
PORTFOLIO_TAKE_PROFIT_HALF = float(os.getenv("PORTFOLIO_TAKE_PROFIT_HALF", "0.15"))
PORTFOLIO_TAKE_PROFIT_FULL = float(os.getenv("PORTFOLIO_TAKE_PROFIT_FULL", "0.30"))

PORTFOLIO_MIN_WEIGHT_THRESHOLD = float(os.getenv("PORTFOLIO_MIN_WEIGHT_THRESHOLD", "0.005"))

PORTFOLIO_IMPACT_ENABLED = os.getenv("PORTFOLIO_IMPACT_ENABLED", "true").lower() == "true"
PORTFOLIO_IMPACT_LAMBDA = float(os.getenv("PORTFOLIO_IMPACT_LAMBDA", "0.3"))

PORTFOLIO_COOLDOWN_STOP_WEEKS = int(os.getenv("PORTFOLIO_COOLDOWN_STOP_WEEKS", "0"))
PORTFOLIO_COOLDOWN_PROFIT_WEEKS = int(os.getenv("PORTFOLIO_COOLDOWN_PROFIT_WEEKS", "0"))

PORTFOLIO_COMMISSION_RATE = float(os.getenv("PORTFOLIO_COMMISSION_RATE", "0.0003"))
PORTFOLIO_SLIPPAGE_RATE = float(os.getenv("PORTFOLIO_SLIPPAGE_RATE", "0.0001"))
PORTFOLIO_STAMP_DUTY = float(os.getenv("PORTFOLIO_STAMP_DUTY", "0.0005"))

PORTFOLIO_CASH_YIELD_ANNUAL = float(os.getenv("PORTFOLIO_CASH_YIELD_ANNUAL", "0.03"))

PORTFOLIO_TURNOVER_DYNAMIC_CORRELATION = os.getenv(
    "PORTFOLIO_TURNOVER_DYNAMIC_CORRELATION", "false").lower() == "true"


def load_regime_stops():
    """从 config/regime_ratio.json 加载每状态的止盈止损参数.

    返回 {regime_id: {warn, full, profitHalf, profitFull}} 字典.
    文件不存在或解析失败时返回内置默认值.
    """
    import json
    from pathlib import Path

    _DEFAULT = {
        0: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
        1: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
        2: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
        3: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
        4: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
        5: {"warn": 5, "full": 10, "profitHalf": 15, "profitFull": 30},
    }
    try:
        path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "config" / "regime_ratio.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stops = data.get("stops", {})
        result = {}
        for k, v in stops.items():
            result[int(k)] = v
        return result if result else dict(_DEFAULT)
    except Exception:
        return dict(_DEFAULT)


__all__ = [
    "PORTFOLIO_DEFAULT_SIZING",
    "PORTFOLIO_DEFAULT_MAX_SINGLE",
    "PORTFOLIO_DEFAULT_MAX_SECTOR",
    "PORTFOLIO_DEFAULT_MAX_TURNOVER",
    "PORTFOLIO_DEFAULT_TARGET_SUM",
    "PORTFOLIO_DEFAULT_MIN_CASH",
    "PORTFOLIO_DEFAULT_SINGLE_FLOOR",
    "PORTFOLIO_STOP_LOSS_WARN",
    "PORTFOLIO_STOP_LOSS_FULL",
    "PORTFOLIO_TAKE_PROFIT_HALF",
    "PORTFOLIO_TAKE_PROFIT_FULL",
    "PORTFOLIO_MIN_WEIGHT_THRESHOLD",
    "PORTFOLIO_IMPACT_ENABLED",
    "PORTFOLIO_IMPACT_LAMBDA",
    "PORTFOLIO_COOLDOWN_STOP_WEEKS",
    "PORTFOLIO_COOLDOWN_PROFIT_WEEKS",
    "PORTFOLIO_COMMISSION_RATE",
    "PORTFOLIO_SLIPPAGE_RATE",
    "PORTFOLIO_STAMP_DUTY",
    "PORTFOLIO_CASH_YIELD_ANNUAL",
    "PORTFOLIO_TURNOVER_DYNAMIC_CORRELATION",
    "load_regime_stops",
]
