import os
import json

REGIME_DEFAULT_INDEX = os.getenv("REGIME_DEFAULT_INDEX", "沪深300")
REGIME_LOOKBACK_DAYS = int(os.getenv("REGIME_LOOKBACK_DAYS", "252"))
REGIME_VOLATILE_PERCENTILE = float(os.getenv("REGIME_VOLATILE_PERCENTILE", "0.75"))
REGIME_CACHE_TTL = int(os.getenv("REGIME_CACHE_TTL", "300"))
REGIME_USE_HMM = os.getenv("REGIME_USE_HMM", "true").lower() == "true"
REGIME_HMM_RETRAIN_DAYS = int(os.getenv("REGIME_HMM_RETRAIN_DAYS", "30"))
REGIME_HMM_EXTRA_FEATURES = os.getenv("REGIME_HMM_EXTRA_FEATURES", "true").lower() == "true"

_REGIME_MODEL_TYPE_BONUS_RAW = os.getenv("REGIME_MODEL_TYPE_BONUS", "")

_REGIME_MODEL_TYPE_BONUS_DEFAULT = {
    "trending_up":   {"lightgbm": 1.15, "xgboost": 1.15, "catboost": 1.05, "linear": 0.85},
    "bull_steady":   {"lightgbm": 1.15, "xgboost": 1.15, "catboost": 1.05, "linear": 0.85},
    "trending_down": {"catboost": 1.15, "lightgbm": 1.05, "xgboost": 1.05, "linear": 0.90},
    "bear_steady":   {"catboost": 1.15, "lightgbm": 1.05, "xgboost": 1.05, "linear": 0.90},
    "ranging":       {"linear": 1.20, "lightgbm": 1.05, "xgboost": 1.05, "catboost": 0.90},
    "volatile":      {"catboost": 1.15, "xgboost": 1.10, "lightgbm": 1.05, "linear": 0.80},
}


def _load_regime_model_type_bonus():
    try:
        parsed = json.loads(_REGIME_MODEL_TYPE_BONUS_RAW)
        if not isinstance(parsed, dict) or not all(isinstance(v, dict) for v in parsed.values()):
            return _REGIME_MODEL_TYPE_BONUS_DEFAULT
        return parsed
    except (json.JSONDecodeError, TypeError):
        return _REGIME_MODEL_TYPE_BONUS_DEFAULT


REGIME_MODEL_TYPE_BONUS = _load_regime_model_type_bonus()

SECTOR_FUSION_ENABLED = os.getenv("SECTOR_FUSION_ENABLED", "false").lower() == "true"
SECTOR_FUSION_LAMBDA = float(os.getenv("SECTOR_FUSION_LAMBDA", "0.2"))
SECTOR_DIVERGENCE_MIN_CONF = float(os.getenv("SECTOR_DIVERGENCE_MIN_CONF", "0.5"))

MACRO_COLLECT_ENABLED = os.getenv("MACRO_COLLECT_ENABLED", "true").lower() == "true"
MACRO_RISK_WARNING_THRESHOLD = float(os.getenv("MACRO_RISK_WARNING_THRESHOLD", "0.4"))
MACRO_RISK_DANGER_THRESHOLD = float(os.getenv("MACRO_RISK_DANGER_THRESHOLD", "0.7"))
MACRO_CACHE_TTL = int(os.getenv("MACRO_CACHE_TTL", "3600"))
MACRO_RISK_CACHE_TTL = int(os.getenv("MACRO_RISK_CACHE_TTL", "300"))

__all__ = [
    "REGIME_DEFAULT_INDEX", "REGIME_LOOKBACK_DAYS", "REGIME_VOLATILE_PERCENTILE",
    "REGIME_CACHE_TTL", "REGIME_USE_HMM", "REGIME_HMM_RETRAIN_DAYS",
    "REGIME_HMM_EXTRA_FEATURES", "REGIME_MODEL_TYPE_BONUS",
    "SECTOR_FUSION_ENABLED", "SECTOR_FUSION_LAMBDA", "SECTOR_DIVERGENCE_MIN_CONF",
    "MACRO_COLLECT_ENABLED", "MACRO_RISK_WARNING_THRESHOLD",
    "MACRO_RISK_DANGER_THRESHOLD", "MACRO_CACHE_TTL", "MACRO_RISK_CACHE_TTL",
]
