import os
import json
import hashlib
from pathlib import Path

from ._data import ROOT_DIR

QLIB_ENABLED = os.getenv("QLIB_ENABLED", "true").lower() == "true"
QLIB_DATA_ROOT = ROOT_DIR / "data" / "qlib_data"
_qlib_pointer_path = QLIB_DATA_ROOT / "current.json"
QLIB_DATA_GENERATION_ID = None
if _qlib_pointer_path.is_file():
    with _qlib_pointer_path.open("r", encoding="utf-8") as _pointer_file:
        _qlib_pointer = json.load(_pointer_file)
    QLIB_DATA_GENERATION_ID = _qlib_pointer.get("generation_id")
    if not QLIB_DATA_GENERATION_ID or Path(QLIB_DATA_GENERATION_ID).name != QLIB_DATA_GENERATION_ID:
        raise RuntimeError(f"Qlib 当前代指针无效: {_qlib_pointer_path}")
    QLIB_DATA_DIR = QLIB_DATA_ROOT / "generations" / QLIB_DATA_GENERATION_ID
    _qlib_generation_manifest = QLIB_DATA_DIR / "generation.json"
    if not _qlib_generation_manifest.is_file():
        raise RuntimeError(f"Qlib 当前代目录或清单缺失: {QLIB_DATA_DIR}")
    _qlib_generation_bytes = _qlib_generation_manifest.read_bytes()
    with _qlib_generation_manifest.open("r", encoding="utf-8") as _manifest_file:
        _qlib_generation = json.load(_manifest_file)
    if (
        _qlib_generation.get("generation_id") != QLIB_DATA_GENERATION_ID
        or _qlib_generation.get("status") != "ready"
        or _qlib_pointer.get("manifest_sha256") != hashlib.sha256(_qlib_generation_bytes).hexdigest()
        or not (QLIB_DATA_DIR / "calendars" / "day.txt").is_file()
        or not (QLIB_DATA_DIR / "instruments" / "all.txt").is_file()
        or not (QLIB_DATA_DIR / "features").is_dir()
    ):
        raise RuntimeError(f"Qlib 当前代清单校验失败: {_qlib_generation_manifest}")
else:
    QLIB_DATA_DIR = QLIB_DATA_ROOT
_configured_qlib_provider_uri = os.getenv("QLIB_PROVIDER_URI")
if QLIB_DATA_GENERATION_ID and _configured_qlib_provider_uri:
    _configured_qlib_path = _configured_qlib_provider_uri.removeprefix("file:")
    if Path(_configured_qlib_path).resolve() != QLIB_DATA_DIR.resolve():
        raise RuntimeError("QLIB_PROVIDER_URI 与当前Qlib数据代不一致")
QLIB_PROVIDER_URI = _configured_qlib_provider_uri or str(QLIB_DATA_DIR.resolve())
QLIB_MLFLOW_URI = os.getenv("QLIB_MLFLOW_URI", "file:" + str((ROOT_DIR / "mlruns").resolve()))
QLIB_EXPERIMENT_NAME = os.getenv("QLIB_EXPERIMENT_NAME", "a_share_quant")
QLIB_DEFAULT_MODEL = os.getenv("QLIB_DEFAULT_MODEL", "lightgbm")
QLIB_DEFAULT_FACTOR_SET = os.getenv("QLIB_DEFAULT_FACTOR_SET", "Alpha158")
QLIB_NUM_THREADS = int(os.getenv("QLIB_NUM_THREADS", "4"))
QLIB_TRAIN_TIMEOUT = int(os.getenv("QLIB_TRAIN_TIMEOUT", "3600"))
QLIB_PRED_NEUTRALIZE = os.getenv("QLIB_PRED_NEUTRALIZE", "false").lower() == "true"
QLIB_MODEL_MIN_ICIR = float(os.getenv("QLIB_MODEL_MIN_ICIR", "0.3"))
QLIB_MODEL_MIN_IC_WIN_RATE = float(os.getenv("QLIB_MODEL_MIN_IC_WIN_RATE", "0.55"))
FACTOR_PANEL_ENABLED = os.getenv("FACTOR_PANEL_ENABLED", "true").lower() == "true"
FACTOR_PANEL_DTYPE = os.getenv("FACTOR_PANEL_DTYPE", "float64").lower()
FACTOR_ROLLING_WORKERS = int(os.getenv("FACTOR_ROLLING_WORKERS", "6"))
FACTOR_EXPR_PRECOMPUTE_WORKERS = int(os.getenv("FACTOR_EXPR_PRECOMPUTE_WORKERS", "4"))
FACTOR_POOL_STATE_PATH = os.getenv(
    "APALLG_FACTOR_POOL_STATE_PATH",
    str(ROOT_DIR / "data" / "runtime" / "factor_pool_state.json"),
)

SIGNAL_FUSION_ENABLED = os.getenv("SIGNAL_FUSION_ENABLED", "false").lower() == "true"
SIGNAL_FUSION_MODE = os.getenv("SIGNAL_FUSION_MODE", "filter")
SIGNAL_FUSION_MIN_CONFIDENCE = float(os.getenv("SIGNAL_FUSION_MIN_CONFIDENCE", "0.6"))
SIGNAL_FUSION_WEIGHT_MACD = float(os.getenv("SIGNAL_FUSION_WEIGHT_MACD", "1.0"))
SIGNAL_FUSION_WEIGHT_RSI = float(os.getenv("SIGNAL_FUSION_WEIGHT_RSI", "0.8"))
SIGNAL_FUSION_WEIGHT_MA = float(os.getenv("SIGNAL_FUSION_WEIGHT_MA", "0.7"))
SIGNAL_FUSION_WEIGHT_KDJ = float(os.getenv("SIGNAL_FUSION_WEIGHT_KDJ", "0.5"))
SIGNAL_FUSION_WEIGHT_BB = float(os.getenv("SIGNAL_FUSION_WEIGHT_BB", "0.6"))
SIGNAL_FUSION_WEIGHT_VOLUME = float(os.getenv("SIGNAL_FUSION_WEIGHT_VOLUME", "0.3"))
SIGNAL_FUSION_ML_ENABLED = os.getenv("SIGNAL_FUSION_ML_ENABLED", "true").lower() == "true"

DENSE_ZONE_PRICE_THRESHOLD_PCT = 2.0
DENSE_ZONE_WINDOW_DAY = 40
DENSE_ZONE_WINDOW_WEEK = 30
DENSE_ZONE_WINDOW_MONTH = 12

BUY_SIDE_L0_MIN_SCORE = float(os.getenv("BUY_SIDE_L0_MIN_SCORE", "0.25"))
BUY_SIDE_L0_MIN_TURNOVER = float(os.getenv("BUY_SIDE_L0_MIN_TURNOVER", "1e7"))
BUY_SIDE_L1_TOP_N = int(os.getenv("BUY_SIDE_L1_TOP_N", "50"))
BUY_SIDE_PREFILTER_N = int(os.getenv("BUY_SIDE_PREFILTER_N", "0"))
BUY_SIDE_PREFILTER_POOL = os.getenv("BUY_SIDE_PREFILTER_POOL", "")
BUY_SIDE_DEBATE_TOP = int(os.getenv("BUY_SIDE_DEBATE_TOP", "5"))
BUY_SIDE_DEBATE_AGENTS = int(os.getenv("BUY_SIDE_DEBATE_AGENTS", "7"))

BUY_SIDE_L0_DISABLED_SOURCES = os.getenv("BUY_SIDE_L0_DISABLED_SOURCES", "")
BUY_SIDE_IMPACT_ENABLED = os.getenv("BUY_SIDE_IMPACT_ENABLED", "true").lower() == "true"
BUY_SIDE_IMPACT_LAMBDA = float(os.getenv("BUY_SIDE_IMPACT_LAMBDA", "0.3"))
BUY_SIDE_IMPACT_ETA = float(os.getenv("BUY_SIDE_IMPACT_ETA", "0.15"))

FUSION_BUY_WEIGHT_TECH = float(os.getenv("FUSION_BUY_WEIGHT_TECH", "0.18"))
FUSION_BUY_WEIGHT_QLIB = float(os.getenv("FUSION_BUY_WEIGHT_QLIB", "0.35"))
FUSION_BUY_WEIGHT_STRATEGY = float(os.getenv("FUSION_BUY_WEIGHT_STRATEGY", "0.20"))
FUSION_BUY_WEIGHT_SENTIMENT = float(os.getenv("FUSION_BUY_WEIGHT_SENTIMENT", "0.10"))
FUSION_BUY_WEIGHT_NORTH_FLOW = float(os.getenv("FUSION_BUY_WEIGHT_NORTH_FLOW", "0.07"))
FUSION_BUY_WEIGHT_SIZE_TURNOVER = float(os.getenv("FUSION_BUY_WEIGHT_SIZE_TURNOVER", "0.10"))

FUSION_HOLD_WEIGHT_TECH = float(os.getenv("FUSION_HOLD_WEIGHT_TECH", "0.22"))
FUSION_HOLD_WEIGHT_QLIB = float(os.getenv("FUSION_HOLD_WEIGHT_QLIB", "0.12"))
FUSION_HOLD_WEIGHT_STRATEGY = float(os.getenv("FUSION_HOLD_WEIGHT_STRATEGY", "0.18"))
FUSION_HOLD_WEIGHT_SENTIMENT = float(os.getenv("FUSION_HOLD_WEIGHT_SENTIMENT", "0.08"))
FUSION_HOLD_WEIGHT_NORTH_FLOW = float(os.getenv("FUSION_HOLD_WEIGHT_NORTH_FLOW", "0.05"))
FUSION_HOLD_WEIGHT_SIZE_TURNOVER = float(os.getenv("FUSION_HOLD_WEIGHT_SIZE_TURNOVER", "0.08"))
FUSION_HOLD_WEIGHT_PNL = float(os.getenv("FUSION_HOLD_WEIGHT_PNL", "0.25"))
FUSION_HOLD_WEIGHT_AI_DEBATE = float(os.getenv("FUSION_HOLD_WEIGHT_AI_DEBATE", "0.10"))

BARRA_RISK_ENABLED = os.getenv("BARRA_RISK_ENABLED", "true").lower() == "true"
BARRA_RISK_NOTIFY = os.getenv("BARRA_RISK_NOTIFY", "false").lower() == "true"
BARRA_RISK_WARNING_THRESHOLD = float(os.getenv("BARRA_RISK_WARNING_THRESHOLD", "0.3"))
BARRA_RISK_DANGER_THRESHOLD = float(os.getenv("BARRA_RISK_DANGER_THRESHOLD", "0.5"))

WEEKLY_REBALANCE_SIZING = os.getenv("WEEKLY_REBALANCE_SIZING", "score")

__all__ = [
    "QLIB_ENABLED", "QLIB_DATA_ROOT", "QLIB_DATA_DIR", "QLIB_DATA_GENERATION_ID",
    "QLIB_PROVIDER_URI", "QLIB_MLFLOW_URI",
    "QLIB_EXPERIMENT_NAME", "QLIB_DEFAULT_MODEL", "QLIB_DEFAULT_FACTOR_SET",
    "QLIB_NUM_THREADS", "QLIB_TRAIN_TIMEOUT", "QLIB_PRED_NEUTRALIZE",
    "QLIB_MODEL_MIN_ICIR", "QLIB_MODEL_MIN_IC_WIN_RATE",
    "FACTOR_PANEL_ENABLED", "FACTOR_PANEL_DTYPE", "FACTOR_ROLLING_WORKERS",
    "FACTOR_EXPR_PRECOMPUTE_WORKERS", "FACTOR_POOL_STATE_PATH",
    "SIGNAL_FUSION_ENABLED", "SIGNAL_FUSION_MODE", "SIGNAL_FUSION_MIN_CONFIDENCE",
    "SIGNAL_FUSION_WEIGHT_MACD", "SIGNAL_FUSION_WEIGHT_RSI", "SIGNAL_FUSION_WEIGHT_MA",
    "SIGNAL_FUSION_WEIGHT_KDJ", "SIGNAL_FUSION_WEIGHT_BB", "SIGNAL_FUSION_WEIGHT_VOLUME",
    "SIGNAL_FUSION_ML_ENABLED",
    "DENSE_ZONE_PRICE_THRESHOLD_PCT", "DENSE_ZONE_WINDOW_DAY",
    "DENSE_ZONE_WINDOW_WEEK", "DENSE_ZONE_WINDOW_MONTH",
    "BUY_SIDE_L0_MIN_SCORE", "BUY_SIDE_L0_MIN_TURNOVER", "BUY_SIDE_L1_TOP_N",
    "BUY_SIDE_PREFILTER_N", "BUY_SIDE_PREFILTER_POOL",
    "BUY_SIDE_DEBATE_TOP", "BUY_SIDE_DEBATE_AGENTS",
    "BUY_SIDE_L0_DISABLED_SOURCES", "BUY_SIDE_IMPACT_ENABLED",
    "BUY_SIDE_IMPACT_LAMBDA", "BUY_SIDE_IMPACT_ETA",
    "FUSION_BUY_WEIGHT_TECH", "FUSION_BUY_WEIGHT_QLIB", "FUSION_BUY_WEIGHT_STRATEGY",
    "FUSION_BUY_WEIGHT_SENTIMENT", "FUSION_BUY_WEIGHT_NORTH_FLOW", "FUSION_BUY_WEIGHT_SIZE_TURNOVER",
    "FUSION_HOLD_WEIGHT_TECH", "FUSION_HOLD_WEIGHT_QLIB", "FUSION_HOLD_WEIGHT_STRATEGY",
    "FUSION_HOLD_WEIGHT_SENTIMENT", "FUSION_HOLD_WEIGHT_NORTH_FLOW", "FUSION_HOLD_WEIGHT_SIZE_TURNOVER",
    "FUSION_HOLD_WEIGHT_PNL", "FUSION_HOLD_WEIGHT_AI_DEBATE",
    "BARRA_RISK_ENABLED", "BARRA_RISK_NOTIFY",
    "BARRA_RISK_WARNING_THRESHOLD", "BARRA_RISK_DANGER_THRESHOLD",
    "WEEKLY_REBALANCE_SIZING",
]
