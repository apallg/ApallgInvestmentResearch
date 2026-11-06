"""DEPRECATED: 活跃路由已不再依赖本模块，仅保留给旧 collect.py/quality.py 等废弃文件。"""
import os
import threading
import logging
import uuid
from pathlib import Path
from typing import Optional

from core.domain.screening.exclusion import EXCLUDED_PREFIXES, B_SHARE_PREFIXES, is_excluded
from core.data.sources.tencent_kline import fetch_tencent_kline_raw
from core.shared.infrastructure.async_task import task_manager
from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)

# 并发保护
_sync_lock = threading.Lock()
_train_jobs = {}
_train_lock = threading.Lock()
_collect_job_id = None

# Optuna 结果持久化
_OPTUNA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "optuna_params"

def _save_optuna_result(factor_set_name, result):
    """存储 Optuna 搜索结果到磁盘 (按因子集名索引, 新结果覆盖旧结果)"""
    import json as _json
    _OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = factor_set_name.replace("/", "_").replace("\\", "_")
    fpath = _OPTUNA_DIR / f"{safe_name}.json"
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
        _json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[Optuna] 结果已保存: {fpath}")
def _load_optuna_result(factor_set_name):
    """从磁盘加载 Optuna 结果, 无则返回 None"""
    import json as _json
    safe_name = factor_set_name.replace("/", "_").replace("\\", "_")
    fpath = _OPTUNA_DIR / f"{safe_name}.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None

# ── Pydantic Models ──
def _get_qlib_data_dir():
    from core.shared import config
    from core.adapters.qlib.generation import assert_no_active_historical_release

    assert_no_active_historical_release(config.QLIB_DATA_ROOT)
    return config.QLIB_DATA_DIR
def _resolve_instruments(instruments: str):
    """解析组合股票池，如 csi300+csi500 → 合并去重后的股票列表。
    普通值 (csi300/all/a500) 原样返回，Qlib 原生支持。
    """
    from backend.service.instrument_resolver import resolve_instruments
    return resolve_instruments(instruments)
def _build_filtered_instruments(pool: str, code_prefixes: list):
    """根据股票池和代码前缀构建自定义 instruments 列表"""
    from core.data.fetcher import DataFetcher
    fetcher = DataFetcher()

    # 获取池内全部股票代码
    if pool == "csi300":
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex("000300")
        codes = _filter_st_stocks(df, "成分券代码", "成分券名称")
    elif pool == "csi500":
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex("000905")
        codes = _filter_st_stocks(df, "成分券代码", "成分券名称")
    elif pool == "csi300+csi500":
        import akshare as ak
        df300 = ak.index_stock_cons_weight_csindex("000300")
        codes300 = set(_filter_st_stocks(df300, "成分券代码", "成分券名称"))
        df500 = ak.index_stock_cons_weight_csindex("000905")
        codes500 = set(_filter_st_stocks(df500, "成分券代码", "成分券名称"))
        codes = list(codes300 | codes500)
    elif pool == "small_cap":
        from core.data.sources.tushare import get_small_cap_codes
        codes = get_small_cap_codes()
    elif pool in ("a500", "stable_history_500d", "all"):
        all_codes = _get_all_stock_codes(fetcher)
        if pool in ("a500", "stable_history_500d"):
            # 过滤有 ≥500 条K线记录的股票
            from core.data import Database
            codes = []
            batch_size = 500
            for bs in range(0, len(all_codes), batch_size):
                batch = all_codes[bs:bs + batch_size]
                ph = ",".join(["%s"] * len(batch))
                rows = Database.fetchall(
                    f"SELECT code, COUNT(*) as cnt FROM daily_quotes "
                    f"WHERE code IN ({ph}) GROUP BY code HAVING cnt >= 500",
                    tuple(batch),
                )
                codes.extend(r["code"] for r in rows)
        else:
            codes = all_codes
    else:
        codes = _get_all_stock_codes(fetcher)

    # 前缀过滤
    if code_prefixes:
        prefixes = tuple(str(p) for p in code_prefixes)
        codes = [c for c in codes if c.startswith(prefixes)]

    # 转为 Qlib 格式 (600519 → sh600519, 000001 → sz000001)
    from core.adapters.qlib.utils import mysql_to_qlib_code
    qlib_codes = [mysql_to_qlib_code(c) for c in codes]

    logger.info(f"[instruments] pool={pool} prefixes={code_prefixes} → {len(qlib_codes)} stocks")
    return qlib_codes if qlib_codes else pool  # 回退到原始 pool（防止空列表）
def _filter_st_stocks(df, code_col, name_col):
    """从指数成分股 DataFrame 中剔除 ST/*ST 股票"""
    codes = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, ""))
        if "ST" in name:
            continue
        codes.append(str(row[code_col]))
    return codes
def _get_all_stock_codes(fetcher):
    """获取全A股代码列表（已过滤 ST 和退市股）"""
    from core.data.sources.tushare import get_all_stock_codes
    try:
        info = get_all_stock_codes()
        return list(info)
    except Exception:
        from core.data import Database
        rows = Database.fetchall("SELECT DISTINCT code FROM daily_quotes ORDER BY code")
        return [r["code"] for r in rows]


# ── Endpoints ──
def _resolve_factor_label(m, run_dir):
    """解析 WhiteBoxHandler 模型的实际因子集名称"""
    import json
    fs = m.get("factor_set", "")
    if "WhiteBox" not in fs:
        if "Alpha158" in fs:
            return "Alpha158"
        if "Alpha360" in fs:
            return "Alpha360"
        return None

    factor_keys = None

    # 1. 双模型 factor_keys.json
    fk_json = run_dir / "artifacts" / "factor_keys.json"
    if fk_json.exists():
        try:
            factor_keys = json.loads(fk_json.read_text(encoding="utf-8")).get("custom_factor_keys", [])
        except Exception:
            pass

    # 2. 单模型 params 文件
    if not factor_keys:
        fk_param = run_dir / "params" / "dataset.kwargs.handler.kwargs.factor_keys"
        if fk_param.exists():
            try:
                val = fk_param.read_text().strip()
                factor_keys = [k.strip().strip("'\"") for k in val.strip("[]").split(",") if k.strip()]
            except Exception:
                pass

    if not factor_keys or len(factor_keys) == 0:
        return "白盒因子"

    # 匹配已知精选集
    try:
        from core.domain.factors.registry import get_pool
        pool = get_pool()
        curated = pool.state.get("curated", {})
        if curated:
            fk_set = set(factor_keys)
            best_name, best_overlap = None, 0
            for name, entry in curated.items():
                overlap = len(fk_set & set(entry.get("flat_list", [])))
                if overlap > best_overlap:
                    best_overlap, best_name = overlap, name
            if best_name and best_overlap >= len(fk_set) * 0.8:
                return f"精选:{best_name}"
    except Exception:
        pass

    return f"自选{len(factor_keys)}因子"
def _make_display_name(m):
    """为模型生成唯一辨识名称，如: LGB_Alpha158_HS300_19-24"""
    import re
    # 模型类型缩写
    mc = m.get("model_class", "")
    if m.get("is_dual"):
        model_short = "LGB+XGB"
    elif "LGB" in mc:
        model_short = "LGB"
    elif "XGB" in mc:
        model_short = "XGB"
    elif "CatBoost" in mc:
        model_short = "CAT"
    else:
        model_short = mc.split(".")[-1] if mc else "?"

    # 因子集 (优先用 factor_label)
    fl = m.get("factor_label", "")
    factor_short = None
    if not fl or fl == "WhiteBoxHandler":
        pass
    elif fl.startswith("精选:"):
        factor_short = "精选"
    elif fl.startswith("自选"):
        factor_short = "自选"
    elif fl in ("Alpha158", "Alpha360"):
        factor_short = "α158" if fl == "Alpha158" else "α360"
    elif len(fl) > 8:
        factor_short = fl[:8]
    else:
        factor_short = fl

    if not factor_short:
        fs = m.get("factor_set", "")
        if "Alpha158" in fs:
            factor_short = "α158"
        elif "Alpha360" in fs:
            factor_short = "α360"
        elif "WhiteBox" in fs:
            factor_short = "WB"
        else:
            factor_short = fs.split(".")[-1] if fs else "?"

    # 股池缩写
    ins = (m.get("instruments") or "").lower()
    pool_map = {"csi300": "HS300", "hs300": "HS300", "csi500": "ZZ500",
                "a500": "稳定历史池500D", "stable_history_500d": "稳定历史池500D", "all": "ALL", "small_cap": "小市值"}
    pool_short = ""
    for k, v in pool_map.items():
        if k in ins:
            pool_short = v
            break
    if not pool_short:
        m2 = re.match(r'^list_(\d+)_stocks$', ins, re.I)
        if m2:
            pool_short = f"A{m2.group(1)}"
        else:
            pool_short = ins[:12] if ins else "?"

    # 训练年份范围
    tp = m.get("train_period", "")
    years_match = re.findall(r'(\d{4})', tp)
    if len(years_match) >= 2:
        year_range = f"{years_match[0][2:]}-{years_match[-1][2:]}"
    elif len(years_match) == 1:
        year_range = years_match[0][2:]
    else:
        year_range = ""

    label = m.get("label_period")
    label_suffix = f"_{label}d" if label else ""

    # 训练时间戳 (MMDD_HHMM)
    ts = m.get("trained_at", "")
    time_suffix = ""
    if ts:
        try:
            # ISO格式: "2026-07-15T14:30:00+00:00"
            t = ts[:16].replace("T", " ")
            time_suffix = f"@{t[5:7]}{t[8:10]}_{t[11:13]}{t[14:16]}"
        except Exception:
            pass

    parts = [model_short, factor_short, pool_short]
    if year_range:
        parts.append(year_range)
    name = "_".join(parts)
    if label_suffix and len(name) < 30:
        name += label_suffix
    if time_suffix:
        name += time_suffix
    return name


def _assert_run_path(mlruns_dir, experiment_id, run_id):
    """校验实验/运行标识为简单段并返回解析后的 run 目录, 防路径穿越 (../..)"""
    from fastapi import HTTPException
    for part in (experiment_id, run_id):
        if not part or part in (".", "..") or any(ch in part for ch in "/\\:"):
            raise HTTPException(400, f"非法模型标识: {part!r}")
    return (Path(mlruns_dir) / experiment_id / run_id).resolve()


# 端点文件通过 `from backend.api.qlib.common import *` 引用共享名字。
# `__all__` 显式列出 _ 开头名字, 让 import * 能导入它们 (否则 import * 会跳过 _ 前缀)。
__all__ = [
    "logger",
    "_sync_lock", "_train_jobs", "_train_lock", "_collect_job_id",
    "_OPTUNA_DIR",
    "_save_optuna_result", "_load_optuna_result",
    "_get_qlib_data_dir", "_resolve_instruments", "_build_filtered_instruments",
    "_filter_st_stocks", "_get_all_stock_codes",
    "_resolve_factor_label", "_make_display_name", "_assert_run_path",
    "is_excluded", "EXCLUDED_PREFIXES", "B_SHARE_PREFIXES",
    "fetch_tencent_kline_raw", "task_manager", "thread_pool",
    "uuid",
]
