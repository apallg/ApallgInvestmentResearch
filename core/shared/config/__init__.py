import os

from ._data import *
from ._factors import *
from ._trading import *
from ._regime import *
from ._notify import *
from ._portfolio import *
from ._logging import *

_CONFIG_ISSUES = []


def _issue(level, message):
    _CONFIG_ISSUES.append((level, message))


def _check_int_range(name, value, min_v=None, max_v=None):
    if not isinstance(value, int) or isinstance(value, bool):
        _issue("error", f"{name}={value!r} 类型错误, 期望 int (当前: {type(value).__name__})")
        return
    if min_v is not None and value < min_v:
        _issue("error", f"{name}={value} 超出最小值 {min_v}")
    if max_v is not None and value > max_v:
        _issue("warning", f"{name}={value} 超出推荐最大值 {max_v}")


def _check_float_range(name, value, min_v=None, max_v=None):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _issue("error", f"{name}={value!r} 类型错误, 期望 float (当前: {type(value).__name__})")
        return
    if min_v is not None and value < min_v:
        _issue("warning", f"{name}={value} 低于推荐最小值 {min_v}")
    if max_v is not None and value > max_v:
        _issue("warning", f"{name}={value} 超出推荐最大值 {max_v}")


def _check_bool(name, raw_value):
    if raw_value not in ("true", "false", "True", "False", "TRUE", "FALSE"):
        _issue("error", f"{name}=({raw_value!r}) 不是有效布尔值, 应填 true 或 false")


def validate_config():
    _CONFIG_ISSUES.clear()

    if BROKER_TYPE not in (BROKER_FAKE, BROKER_QMT, BROKER_EASYT):
        _issue("error", f"BROKER_TYPE={BROKER_TYPE!r} 不受支持")
    if BROKER_TYPE == BROKER_EASYT and not str(EASYTRAIDER_ACCOUNT_ID or "").strip():
        _issue("error", "EASYTRAIDER_ACCOUNT_ID 未设置，Easytrader账户无法安全归属")

    _check_int_range("MYSQL_PORT", MYSQL_PORT, 1, 65535)
    _check_int_range("MYSQL_POOL_SIZE", MYSQL_POOL_SIZE, 2, 100)
    if MYSQL_PASSWORD == "your_mysql_password_here":
        _issue("warning", "MYSQL_PASSWORD 仍为占位值 'your_mysql_password_here', MySQL 已禁用")

    _check_int_range("REDIS_PORT", REDIS_PORT, 1, 65535)
    _check_int_range("REDIS_DB", REDIS_DB, 0, 15)
    _check_int_range("REDIS_MAX_CONNECTIONS", REDIS_MAX_CONNECTIONS, 2, 200)
    _check_int_range("REDIS_SOCKET_TIMEOUT", REDIS_SOCKET_TIMEOUT, 1, 60)
    _check_int_range("LIVE_QUOTE_MAX_AGE", LIVE_QUOTE_MAX_AGE, 1, 30)
    _check_int_range("LIVE_DISPLAY_MAX_AGE", LIVE_DISPLAY_MAX_AGE, 1, 120)
    if "miniqmt" in REALTIME_QUOTE_CHAIN or "tushare" in REALTIME_QUOTE_CHAIN:
        _issue("error", "实时行情链路禁止配置 miniQMT 或 Tushare")
    if LOW_FREQUENCY_SOURCE != "tushare":
        _issue("warning", f"LOW_FREQUENCY_SOURCE={LOW_FREQUENCY_SOURCE!r}, 当前项目建议使用 tushare")

    _check_float_range("QLIB_MODEL_MIN_ICIR", QLIB_MODEL_MIN_ICIR, -5.0, 5.0)
    _check_float_range("QLIB_MODEL_MIN_IC_WIN_RATE", QLIB_MODEL_MIN_IC_WIN_RATE, 0.0, 1.0)
    if FACTOR_PANEL_DTYPE not in ("float32", "float64"):
        _issue("error", f"FACTOR_PANEL_DTYPE={FACTOR_PANEL_DTYPE!r} 仅支持 float32/float64")
    _check_int_range("FACTOR_ROLLING_WORKERS", FACTOR_ROLLING_WORKERS, 1, 6)
    _check_int_range("FACTOR_EXPR_PRECOMPUTE_WORKERS", FACTOR_EXPR_PRECOMPUTE_WORKERS, 1, 4)
    _check_int_range("QLIB_NUM_THREADS", QLIB_NUM_THREADS, 1, 32)
    _check_int_range("QLIB_TRAIN_TIMEOUT", QLIB_TRAIN_TIMEOUT, 60, 86400)

    for name in ["FUSION_BUY_WEIGHT_TECH", "FUSION_BUY_WEIGHT_QLIB", "FUSION_BUY_WEIGHT_STRATEGY",
                 "FUSION_BUY_WEIGHT_SENTIMENT", "FUSION_BUY_WEIGHT_NORTH_FLOW",
                 "FUSION_BUY_WEIGHT_SIZE_TURNOVER"]:
        val = globals().get(name, 0)
        _check_float_range(name, val, 0.0, 1.0)
    buy_total = (FUSION_BUY_WEIGHT_TECH + FUSION_BUY_WEIGHT_QLIB +
                 FUSION_BUY_WEIGHT_STRATEGY + FUSION_BUY_WEIGHT_SENTIMENT +
                 FUSION_BUY_WEIGHT_NORTH_FLOW + FUSION_BUY_WEIGHT_SIZE_TURNOVER)
    if not (0.9 <= buy_total <= 1.1):
        _issue("warning", f"选股侧融合权重之和={buy_total:.2f}, 偏离 1.0 较远")

    _check_float_range("BUY_SIDE_L0_MIN_SCORE", BUY_SIDE_L0_MIN_SCORE, 0.0, 1.0)
    _check_float_range("BUY_SIDE_L0_MIN_TURNOVER", BUY_SIDE_L0_MIN_TURNOVER, 1e5, 1e10)
    _check_int_range("BUY_SIDE_L1_TOP_N", BUY_SIDE_L1_TOP_N, 5, 500)
    _check_int_range("BUY_SIDE_DEBATE_AGENTS", BUY_SIDE_DEBATE_AGENTS, 1, 15)

    _check_float_range("BUY_SIDE_IMPACT_LAMBDA", BUY_SIDE_IMPACT_LAMBDA, 0.0, 1.0)
    _check_float_range("BUY_SIDE_IMPACT_ETA", BUY_SIDE_IMPACT_ETA, 0.0, 1.0)

    _check_int_range("COLLECT_THREAD_COUNT", COLLECT_THREAD_COUNT, 1, 64)
    _check_int_range("COLLECT_MIN_DAILY_ROWS", COLLECT_MIN_DAILY_ROWS, 1, 10000)
    _check_int_range("COLLECT_CONCEPT_SECTORS_CONCURRENCY", COLLECT_CONCEPT_SECTORS_CONCURRENCY, 1, 32)

    _check_int_range("REGIME_LOOKBACK_DAYS", REGIME_LOOKBACK_DAYS, 60, 1000)
    _check_float_range("REGIME_VOLATILE_PERCENTILE", REGIME_VOLATILE_PERCENTILE, 0.5, 0.99)
    _check_int_range("REGIME_CACHE_TTL", REGIME_CACHE_TTL, 60, 86400)

    _check_float_range("BARRA_RISK_WARNING_THRESHOLD", BARRA_RISK_WARNING_THRESHOLD, 0.1, 1.0)
    _check_float_range("BARRA_RISK_DANGER_THRESHOLD", BARRA_RISK_DANGER_THRESHOLD, 0.1, 1.0)
    if BARRA_RISK_DANGER_THRESHOLD < BARRA_RISK_WARNING_THRESHOLD:
        _issue("error", f"BARRA danger({BARRA_RISK_DANGER_THRESHOLD}) < warning({BARRA_RISK_WARNING_THRESHOLD})")

    _check_float_range("MACRO_RISK_WARNING_THRESHOLD", MACRO_RISK_WARNING_THRESHOLD, 0.1, 1.0)
    _check_float_range("MACRO_RISK_DANGER_THRESHOLD", MACRO_RISK_DANGER_THRESHOLD, 0.1, 1.0)
    if MACRO_RISK_DANGER_THRESHOLD < MACRO_RISK_WARNING_THRESHOLD:
        _issue("error", f"MACRO danger({MACRO_RISK_DANGER_THRESHOLD}) < warning({MACRO_RISK_WARNING_THRESHOLD})")

    _check_int_range("ERROR_LOG_RETENTION_DAYS", ERROR_LOG_RETENTION_DAYS, 1, 365)

    if LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        _issue("warning", f"LOG_LEVEL={LOG_LEVEL} 非标准级别, 建议 DEBUG/INFO/WARNING/ERROR/CRITICAL")
    _check_int_range("LOG_MAX_BYTES", LOG_MAX_BYTES, 1048576, 1073741824)
    _check_int_range("LOG_BACKUP_COUNT", LOG_BACKUP_COUNT, 1, 50)

    _check_float_range("PORTFOLIO_DEFAULT_MAX_SINGLE", PORTFOLIO_DEFAULT_MAX_SINGLE, 0.05, 0.50)
    _check_float_range("PORTFOLIO_DEFAULT_MAX_SECTOR", PORTFOLIO_DEFAULT_MAX_SECTOR, 0.05, 0.60)
    _check_float_range("PORTFOLIO_DEFAULT_MAX_TURNOVER", PORTFOLIO_DEFAULT_MAX_TURNOVER, 0.05, 0.80)
    _check_float_range("PORTFOLIO_STOP_LOSS_WARN", PORTFOLIO_STOP_LOSS_WARN, -0.20, 0.0)
    _check_float_range("PORTFOLIO_STOP_LOSS_FULL", PORTFOLIO_STOP_LOSS_FULL, -0.30, 0.0)
    if PORTFOLIO_STOP_LOSS_WARN < PORTFOLIO_STOP_LOSS_FULL:
        _issue("error", f"止损 warn({PORTFOLIO_STOP_LOSS_WARN}) < full({PORTFOLIO_STOP_LOSS_FULL}) — warn 应比 full 更浅")
    _check_float_range("PORTFOLIO_TAKE_PROFIT_HALF", PORTFOLIO_TAKE_PROFIT_HALF, 0.0, 0.50)
    _check_float_range("PORTFOLIO_TAKE_PROFIT_FULL", PORTFOLIO_TAKE_PROFIT_FULL, 0.0, 1.0)
    if PORTFOLIO_TAKE_PROFIT_HALF > PORTFOLIO_TAKE_PROFIT_FULL:
        _issue("error", f"止盈 half({PORTFOLIO_TAKE_PROFIT_HALF}) > full({PORTFOLIO_TAKE_PROFIT_FULL})")

    bool_keys = {
        "FORCE_CLOSE_ENABLED": os.getenv("FORCE_CLOSE_ENABLED", "true"),
        "QLIB_ENABLED": os.getenv("QLIB_ENABLED", "true"),
        "QLIB_PRED_NEUTRALIZE": os.getenv("QLIB_PRED_NEUTRALIZE", "false"),
        "FACTOR_PANEL_ENABLED": os.getenv("FACTOR_PANEL_ENABLED", "true"),
        "SIGNAL_FUSION_ENABLED": os.getenv("SIGNAL_FUSION_ENABLED", "false"),
        "SIGNAL_FUSION_ML_ENABLED": os.getenv("SIGNAL_FUSION_ML_ENABLED", "true"),
        "COLLECT_ENABLED": os.getenv("COLLECT_ENABLED", "true"),
        "SMOKE_TEST_ENABLED": os.getenv("SMOKE_TEST_ENABLED", "true"),
        "BARRA_RISK_ENABLED": os.getenv("BARRA_RISK_ENABLED", "true"),
        "MACRO_COLLECT_ENABLED": os.getenv("MACRO_COLLECT_ENABLED", "true"),
        "REDIS_ENABLED": os.getenv("REDIS_ENABLED", "true"),
        "ERROR_LOG_ENABLED": os.getenv("ERROR_LOG_ENABLED", "true"),
        "PORTFOLIO_IMPACT_ENABLED": os.getenv("PORTFOLIO_IMPACT_ENABLED", "true"),
        "PORTFOLIO_TURNOVER_DYNAMIC_CORRELATION": os.getenv(
            "PORTFOLIO_TURNOVER_DYNAMIC_CORRELATION", "false"),
    }
    for key, raw in bool_keys.items():
        _check_bool(key, raw)

    errors = sum(1 for lv, _ in _CONFIG_ISSUES if lv == "error")
    warnings_count = sum(1 for lv, _ in _CONFIG_ISSUES if lv == "warning")
    return {"errors": errors, "warnings": warnings_count, "details": list(_CONFIG_ISSUES)}


def get_config_issues():
    return list(_CONFIG_ISSUES)
