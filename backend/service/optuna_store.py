"""Optuna 搜索结果持久化 — 从 common.py 迁出。"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OPTUNA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "optuna_params"


def save_optuna_result(factor_set_name, result):
    """存储 Optuna 搜索结果到磁盘 (按因子集名索引, 新结果覆盖旧结果)"""
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = factor_set_name.replace("/", "_").replace("\\", "_")
    fpath = OPTUNA_DIR / f"{safe_name}.json"
    data = {
        "factor_set": factor_set_name,
        "saved_at": __import__('datetime').datetime.now().isoformat(),
        "lgb_best_params": result.get("lgb_best_params"),
        "xgb_best_params": result.get("xgb_best_params"),
        "lgb_best_ic": result.get("lgb_best_ic"),
        "xgb_best_ic": result.get("xgb_best_ic"),
        "ic_mean": result.get("ic_mean"),
        "lgb_weight": result.get("lgb_weight"),
        "xgb_weight": result.get("xgb_weight"),
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[Optuna] 结果已保存: {fpath}")


def load_optuna_result(factor_set_name):
    """从磁盘加载 Optuna 结果, 无则返回 None"""
    safe_name = factor_set_name.replace("/", "_").replace("\\", "_")
    fpath = OPTUNA_DIR / f"{safe_name}.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
