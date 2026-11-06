"""
配置调优引擎 - 快扫对比 + 历史验证,找到最优筛选参数组合
"""

__all__ = ["ConfigTuner", "run_param_sweep"]

import logging
import datetime
import threading
import time
import pickle
import os
from concurrent.futures import as_completed
from core.shared.infrastructure.thread_pool import thread_pool
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

from core.domain.screening._screener import get_screener
from core.shared.persistence import today_date, now_strftime

logger = logging.getLogger(__name__)


SNAPSHOT_PREFIX = "snapshot"
from core.shared.anchor_cache import clear_mem_cache as _clear_mem_cache, get_mem_cache as _get_anchor_mem_cache


def _snapshot_cache_dir():
    p = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "snapshots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _snapshot_cache_key(eval_date, pool):
    d = eval_date.isoformat() if hasattr(eval_date, 'isoformat') else str(eval_date)
    return f"{d}_{pool}.pkl"


def _snapshot_redis_key(eval_date, pool):
    d = eval_date.isoformat() if hasattr(eval_date, 'isoformat') else str(eval_date)
    return f"{SNAPSHOT_PREFIX}:{pool}:{d}"


def _load_cached_snapshot(eval_date, pool, max_age_hours=24):
    """从 内存 → Redis → 磁盘 加载 snapshot"""
    # 0. 进程内存 (Phase 1→Phase 2 秒级复用)
    key = f"{pool}:{eval_date.isoformat()}"
    cache = _get_anchor_mem_cache()
    if key in cache:
        return cache[key]
    # 1. Redis
    from core.shared.infrastructure import redis_client as rds
    if rds.redis_enabled():
        cached = rds._get(_snapshot_redis_key(eval_date, pool))
        if cached is not None:
            _get_anchor_mem_cache()[key] = cached
            return cached

    # 2. 磁盘 pickle 回退
    p = _snapshot_cache_dir() / _snapshot_cache_key(eval_date, pool)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > max_age_hours * 3600:
        p.unlink()
        return None
    try:
        from core.shared.persistence import load_pickle
        snap = load_pickle(str(p))
        if snap:
            _get_anchor_mem_cache()[key] = snap
            if rds.redis_enabled():
                rds._set(_snapshot_redis_key(eval_date, pool), snap, expire=max_age_hours * 3600)
        return snap
    except Exception:
        return None


def _save_cached_snapshot(eval_date, pool, snapshot):
    # 进程内存（Phase 1→Phase 2 秒级复用）
    key = f"{pool}:{eval_date.isoformat()}"
    _get_anchor_mem_cache()[key] = snapshot
    from core.shared.infrastructure import redis_client as rds
    written = rds._set(_snapshot_redis_key(eval_date, pool), snapshot, expire=86400)
    if not written:
        p = _snapshot_cache_dir() / _snapshot_cache_key(eval_date, pool)
        try:
            from core.shared.persistence import save_pickle
            save_pickle(snapshot, str(p))
        except Exception:
            pass

# ══════════════════════════════════════════════
# 预定义配置模板
# ══════════════════════════════════════════════

WEIGHT_PRESETS = {
    "default": {"tech": 0.25, "qlib": 0.20, "sentiment": 0.05,
                "fundamental": 0.10, "flow": 0.05, "north_flow": 0.10,
                "size_turnover": 0.10, "momentum": 0.10, "volatility": 0.05,
                "illiq": 0.02, "overnight_gap": 0.02},
    "tech_heavy": {"tech": 0.35, "qlib": 0.20, "sentiment": 0.05,
                   "fundamental": 0.10, "flow": 0.05, "north_flow": 0.05,
                   "size_turnover": 0.05, "momentum": 0.10, "volatility": 0.05,
                   "illiq": 0.02, "overnight_gap": 0.02},
    "tech_only": {"tech": 0.40, "qlib": 0.0, "sentiment": 0.0,
                  "fundamental": 0.10, "flow": 0.10, "north_flow": 0.10,
                  "size_turnover": 0.10, "momentum": 0.15, "volatility": 0.05,
                  "illiq": 0.02, "overnight_gap": 0.02},
    "qlib_heavy": {"tech": 0.15, "qlib": 0.45, "sentiment": 0.05,
                   "fundamental": 0.10, "flow": 0.05, "north_flow": 0.05,
                   "size_turnover": 0.05, "momentum": 0.05, "volatility": 0.05,
                   "illiq": 0.02, "overnight_gap": 0.02},
    "balanced": {"tech": 0.20, "qlib": 0.15, "sentiment": 0.10,
                 "fundamental": 0.10, "flow": 0.05, "north_flow": 0.10,
                 "size_turnover": 0.10, "momentum": 0.10, "volatility": 0.10,
                 "illiq": 0.02, "overnight_gap": 0.02},
    "momentum_focused": {"tech": 0.15, "qlib": 0.10, "sentiment": 0.05,
                         "fundamental": 0.05, "flow": 0.05, "north_flow": 0.05,
                         "size_turnover": 0.10, "momentum": 0.35, "volatility": 0.10,
                         "illiq": 0.02, "overnight_gap": 0.02},
    "fundamental_only": {"tech": 0.10, "qlib": 0.0, "sentiment": 0.0,
                         "fundamental": 0.40, "flow": 0.10, "north_flow": 0.15,
                         "size_turnover": 0.15, "momentum": 0.05, "volatility": 0.05,
                         "illiq": 0.02, "overnight_gap": 0.02},
    "size_value": {"tech": 0.10, "qlib": 0.15, "sentiment": 0.05,
                   "fundamental": 0.20, "flow": 0.05, "north_flow": 0.10,
                   "size_turnover": 0.25, "momentum": 0.05, "volatility": 0.05,
                   "ht_factors": 0.12, "illiq": 0.02, "overnight_gap": 0.02},
    "ht_heavy": {"tech": 0.10, "qlib": 0.08, "sentiment": 0.05,
                 "fundamental": 0.05, "flow": 0.05, "north_flow": 0.05,
                 "size_turnover": 0.10, "momentum": 0.05, "volatility": 0.05,
                 "ht_factors": 0.42, "illiq": 0.02, "overnight_gap": 0.02},
}

SKIP_PRESETS = {
    "all_on": {"skip_tech": False, "skip_qlib": False, "skip_sent": False,
               "skip_fundamental": False, "skip_flow": False,
               "skip_north_flow": False, "skip_size_turnover": False,
               "skip_momentum": False, "skip_volatility": False,
               "skip_illiq": False, "skip_overnight_gap": False},
    "no_qlib": {"skip_tech": False, "skip_qlib": True, "skip_sent": False,
                "skip_fundamental": False, "skip_flow": False,
                "skip_north_flow": False, "skip_size_turnover": False,
                "skip_momentum": False, "skip_volatility": False,
               "skip_illiq": False, "skip_overnight_gap": False},
    "no_sentiment": {"skip_tech": False, "skip_qlib": False, "skip_sent": True,
                     "skip_fundamental": False, "skip_flow": False,
                     "skip_north_flow": False, "skip_size_turnover": False,
                     "skip_momentum": False, "skip_volatility": False,
               "skip_illiq": False, "skip_overnight_gap": False},
    "no_fundamental": {"skip_tech": False, "skip_qlib": False, "skip_sent": False,
                       "skip_fundamental": True, "skip_flow": False,
                       "skip_north_flow": False, "skip_size_turnover": False,
                       "skip_momentum": False, "skip_volatility": False,
               "skip_illiq": False, "skip_overnight_gap": False},
    "no_flow": {"skip_tech": False, "skip_qlib": False, "skip_sent": False,
                "skip_fundamental": False, "skip_flow": True,
                "skip_north_flow": False, "skip_size_turnover": False,
                "skip_momentum": False, "skip_volatility": False,
               "skip_illiq": False, "skip_overnight_gap": False},
    "tech_only": {"skip_tech": False, "skip_qlib": True, "skip_sent": True,
                  "skip_fundamental": True, "skip_flow": True,
                  "skip_north_flow": True, "skip_size_turnover": True,
                  "skip_momentum": False, "skip_volatility": False,
               "skip_illiq": False, "skip_overnight_gap": False},
    "core_only": {"skip_tech": False, "skip_qlib": False, "skip_sent": False,
                  "skip_fundamental": True, "skip_flow": True,
                  "skip_north_flow": False, "skip_size_turnover": False,
                  "skip_momentum": False, "skip_volatility": False,
               "skip_illiq": False, "skip_overnight_gap": False},
}

L0_THRESHOLD_PRESETS = [0.25, 0.30, 0.35, 0.40, 0.45]

# 信号源定义(供 Phase 1/2 调优使用)
SIGNAL_SOURCES = [
    {"key": "tech", "label": "技术指标", "skip_key": "skip_tech", "weight_key": "tech",
     "data_safety": "safe"},
    {"key": "qlib", "label": "Qlib模型", "skip_key": "skip_qlib", "weight_key": "qlib",
     "data_safety": "caveat"},
    {"key": "sentiment", "label": "情绪分析", "skip_key": "skip_sent", "weight_key": "sentiment",
     "data_safety": "caveat"},
    {"key": "fundamental", "label": "基本面", "skip_key": "skip_fundamental", "weight_key": "fundamental",
     "data_safety": "caveat"},
    {"key": "flow", "label": "资金流向", "skip_key": "skip_flow", "weight_key": "flow",
     "data_safety": "caveat"},
    {"key": "north_flow", "label": "北向资金", "skip_key": "skip_north_flow", "weight_key": "north_flow",
     "data_safety": "safe"},
    {"key": "size_turnover", "label": "小市值换手", "skip_key": "skip_size_turnover", "weight_key": "size_turnover",
     "data_safety": "safe"},
    {"key": "momentum", "label": "动量因子", "skip_key": "skip_momentum", "weight_key": "momentum",
     "data_safety": "safe"},
    {"key": "volatility", "label": "波动率", "skip_key": "skip_volatility", "weight_key": "volatility",
     "data_safety": "safe"},
    {"key": "ht_factors", "label": "华泰多因子", "skip_key": "skip_ht", "weight_key": "ht_factors",
     "data_safety": "safe"},
    {"key": "wb_industry", "label": "白盒-行业残差动量", "skip_key": "skip_wb_industry", "weight_key": "wb_industry",
     "data_safety": "safe"},
    {"key": "wb_divergence", "label": "白盒-量价背离", "skip_key": "skip_wb_divergence", "weight_key": "wb_divergence",
     "data_safety": "safe"},
    {"key": "wb_moneyflow", "label": "白盒-资金异动", "skip_key": "skip_wb_moneyflow", "weight_key": "wb_moneyflow",
     "data_safety": "caveat"},
    {"key": "wb_peg_adj", "label": "白盒-PEG调整估值", "skip_key": "skip_wb_peg_adj", "weight_key": "wb_peg_adj",
     "data_safety": "caveat"},
    {"key": "wb_earnings_mo", "label": "白盒-盈利动量", "skip_key": "skip_wb_earnings_mo", "weight_key": "wb_earnings_mo",
     "data_safety": "caveat"},
    {"key": "wb_smart_money", "label": "白盒-聪明钱跟随", "skip_key": "skip_wb_smart_money", "weight_key": "wb_smart_money",
     "data_safety": "safe"},
    {"key": "wb_vol_regime", "label": "白盒-波动率区间突破", "skip_key": "skip_wb_vol_regime", "weight_key": "wb_vol_regime",
     "data_safety": "safe"},
    {"key": "wb_turnover_price", "label": "白盒-换手率价格协同", "skip_key": "skip_wb_turnover_price", "weight_key": "wb_turnover_price",
     "data_safety": "safe"},
    {"key": "illiq", "label": "Amihud非流动性", "skip_key": "skip_illiq", "weight_key": "illiq",
     "data_safety": "safe"},
    {"key": "overnight_gap", "label": "隔夜跳空", "skip_key": "skip_overnight_gap", "weight_key": "overnight_gap",
     "data_safety": "safe"},
]

# 白盒因子参数变体 (Phase 1 做参数遍历)
WB_FACTOR_VARIANTS = [
    {
        "template_key": "wb_industry", "label": "白盒-行业残差动量",
        "skip_key": "skip_wb_industry", "weight_key": "wb_industry",
        "data_safety": "safe",
        "param_name": "N", "param_values": [5, 10, 20, 30],
        "base_skips": {"skip_wb_industry": False, "skip_wb_divergence": True, "skip_wb_moneyflow": True},
        "base_weights": {"wb_industry": 1.0},
        "extra_params": lambda n: {"wb_industry_n": n},
    },
    {
        "template_key": "wb_divergence", "label": "白盒-量价背离",
        "skip_key": "skip_wb_divergence", "weight_key": "wb_divergence",
        "data_safety": "safe",
        "param_name": "N", "param_values": [5, 10, 20],
        "base_skips": {"skip_wb_industry": True, "skip_wb_divergence": False, "skip_wb_moneyflow": True},
        "base_weights": {"wb_divergence": 1.0},
        "extra_params": lambda n: {"wb_divergence_n": n},
    },
    {
        "template_key": "wb_moneyflow", "label": "白盒-资金异动",
        "skip_key": "skip_wb_moneyflow", "weight_key": "wb_moneyflow",
        "data_safety": "caveat",
        "param_name": "N", "param_values": [3, 5, 10, 20],
        "base_skips": {"skip_wb_industry": True, "skip_wb_divergence": True, "skip_wb_moneyflow": False},
        "base_weights": {"wb_moneyflow": 1.0},
        "extra_params": lambda n: {"wb_moneyflow_n": n},
    },
    {
        "template_key": "wb_peg_adj", "label": "白盒-PEG调整估值",
        "skip_key": "skip_wb_peg_adj", "weight_key": "wb_peg_adj",
        "data_safety": "caveat",
        "param_name": "peg_w", "param_values": [0.5, 2.0, 4.0],
        "base_skips": {"skip_wb_peg_adj": False},
        "base_weights": {"wb_peg_adj": 1.0},
        "extra_params": lambda n: {},
    },
    {
        "template_key": "wb_earnings_mo", "label": "白盒-盈利动量",
        "skip_key": "skip_wb_earnings_mo", "weight_key": "wb_earnings_mo",
        "data_safety": "caveat",
        "param_name": "accel_w", "param_values": [2.0, 5.0, 10.0],
        "base_skips": {"skip_wb_earnings_mo": False},
        "base_weights": {"wb_earnings_mo": 1.0},
        "extra_params": lambda n: {},
    },
    {
        "template_key": "wb_smart_money", "label": "白盒-聪明钱跟随",
        "skip_key": "skip_wb_smart_money", "weight_key": "wb_smart_money",
        "data_safety": "safe",
        "param_name": "N", "param_values": [5, 10, 20],
        "base_skips": {"skip_wb_smart_money": False},
        "base_weights": {"wb_smart_money": 1.0},
        "extra_params": lambda n: {"wb_smart_money_n": n},
    },
    {
        "template_key": "wb_vol_regime", "label": "白盒-波动率区间突破",
        "skip_key": "skip_wb_vol_regime", "weight_key": "wb_vol_regime",
        "data_safety": "safe",
        "param_name": "N", "param_values": [10, 14, 20, 30],
        "base_skips": {"skip_wb_vol_regime": False},
        "base_weights": {"wb_vol_regime": 1.0},
        "extra_params": lambda n: {"wb_vol_regime_n": n},
    },
    {
        "template_key": "wb_turnover_price", "label": "白盒-换手率价格协同",
        "skip_key": "skip_wb_turnover_price", "weight_key": "wb_turnover_price",
        "data_safety": "safe",
        "param_name": "N", "param_values": [10, 20, 30],
        "base_skips": {"skip_wb_turnover_price": False},
        "base_weights": {"wb_turnover_price": 1.0},
        "extra_params": lambda n: {"wb_turnover_price_n": n},
    },
    {
        "template_key": "illiq", "label": "Amihud非流动性",
        "skip_key": "skip_illiq", "weight_key": "illiq",
        "data_safety": "safe",
        "param_name": "N", "param_values": [10, 20, 30],
        "base_skips": {"skip_illiq": False},
        "base_weights": {"illiq": 1.0},
        "extra_params": lambda n: {"illiq_n": n},
    },
    {
        "template_key": "overnight_gap", "label": "隔夜跳空",
        "skip_key": "skip_overnight_gap", "weight_key": "overnight_gap",
        "data_safety": "safe",
        "param_name": "N", "param_values": [10, 20, 30],
        "base_skips": {"skip_overnight_gap": False},
        "base_weights": {"overnight_gap": 1.0},
        "extra_params": lambda n: {"overnight_gap_n": n},
    },
]

# 白盒因子(Phase 1 单独检测)
WB_SIGNALS = [
    {"key": "wb_ind_residual",   "label": "白盒-行业残差动量", "ht_key": "wb_ind_residual", "signal_type": "continuous"},
    {"key": "wb_vp_divergence",  "label": "白盒-量价背离", "ht_key": "wb_vp_divergence", "signal_type": "continuous"},
    {"key": "wb_capital_anomaly","label": "白盒-资金异动", "ht_key": "wb_capital_anomaly", "signal_type": "continuous"},
]

# 华泰多因子子信号(Phase 1 单独检测) - 13连续 + 4离散
HT_CONTINUOUS_SIGNALS = [
    {"key": "ht_reversal_5",   "label": "华泰-短期反转(5日)", "ht_key": "ht_reversal_5", "signal_type": "continuous"},
    {"key": "ht_reversal_20",  "label": "华泰-中期反转(20日)", "ht_key": "ht_reversal_20", "signal_type": "continuous"},
    {"key": "ht_momentum_60",  "label": "华泰-长期动量(60日)", "ht_key": "ht_momentum_60", "signal_type": "continuous"},
    {"key": "ht_max_dd",       "label": "华泰-最大回撤(20日)", "ht_key": "ht_max_dd", "signal_type": "continuous"},
    {"key": "ht_skewness",     "label": "华泰-收益偏度", "ht_key": "ht_skewness", "signal_type": "continuous"},
    {"key": "ht_kurtosis",     "label": "华泰-收益峰度", "ht_key": "ht_kurtosis", "signal_type": "continuous"},
    {"key": "ht_abn_turnover", "label": "华泰-异常换手率", "ht_key": "ht_abn_turnover", "signal_type": "continuous"},
    {"key": "ht_turnover_chg", "label": "华泰-换手率变化", "ht_key": "ht_turnover_chg", "signal_type": "continuous"},
    {"key": "ht_illiquidity",  "label": "华泰-非流动性(Amihud)", "ht_key": "ht_illiquidity", "signal_type": "continuous"},
    {"key": "ht_vol_chg",      "label": "华泰-成交量变化", "ht_key": "ht_vol_chg", "signal_type": "continuous"},
    {"key": "ht_price_pos",    "label": "华泰-价格位置", "ht_key": "ht_price_pos", "signal_type": "continuous"},
]
HT_DISCRETE_SIGNALS = [
    {"key": "ht_reversal_buy",    "label": "华泰-超跌反弹", "ht_key": "ht_reversal_buy", "signal_type": "discrete"},
    {"key": "ht_ma_cross",        "label": "华泰-均线金叉", "ht_key": "ht_ma_cross", "signal_type": "discrete"},
    {"key": "ht_new_high",        "label": "华泰-新高突破", "ht_key": "ht_new_high", "signal_type": "discrete"},
    {"key": "ht_vol_price_div",   "label": "华泰-量价背离", "ht_key": "ht_vol_price_div", "signal_type": "discrete"},
]

# 技术子指标(Phase 1 单独检测)
TECH_SUB_SIGNALS = [
    {"key": "tech_macd", "label": "技术-MACD", "tech_preset": "macd_only"},
    {"key": "tech_rsi", "label": "技术-RSI", "tech_preset": "rsi_only"},
    {"key": "tech_ma", "label": "技术-MA均线", "tech_preset": "ma_only"},
    {"key": "tech_kdj", "label": "技术-KDJ", "tech_preset": "kdj_only"},
    {"key": "tech_bb", "label": "技术-布林带", "tech_preset": "bb_only"},
    {"key": "tech_volume", "label": "技术-成交量", "tech_preset": "volume_only"},
]

SIGNAL_VALIDITY_THRESHOLDS = {
    "rank_ic_min": 0.03,
    "hit_rate_20d_min": 0.52,
    "mean_20d_min": 0.005,
}


def get_presets():
    """返回所有预定义配置模板"""
    return {
        "weight_presets": {k: v for k, v in WEIGHT_PRESETS.items()},
        "skip_presets": {k: v for k, v in SKIP_PRESETS.items()},
        "l0_thresholds": L0_THRESHOLD_PRESETS,
        "tech_presets": ["all", "trend", "reversal", "momentum", "volume_only",
                         "macd_only", "rsi_only", "ma_only", "kdj_only", "bb_only"],
        "signal_sources": SIGNAL_SOURCES,
        "tech_sub_signals": TECH_SUB_SIGNALS,
        "ht_continuous_signals": HT_CONTINUOUS_SIGNALS,
        "ht_discrete_signals": HT_DISCRETE_SIGNALS,
        "wb_signals": WB_SIGNALS,
        "validity_thresholds": SIGNAL_VALIDITY_THRESHOLDS,
    }


def _config_label(config):
    """生成配置的可读标签：显示全部非零权重"""
    w = config.get("source_weights", {})
    tp = config.get("tech_preset", "all")
    KEY_ABBR = {"tech": "技", "qlib": "Q", "sentiment": "情", "fundamental": "基",
                "flow": "资", "north_flow": "北", "size_turnover": "小",
                "momentum": "动", "volatility": "波"}
    active = [(k, v) for k, v in w.items() if v > 0]
    active.sort(key=lambda x: x[1], reverse=True)
    parts = [f"{KEY_ABBR.get(k, k)}{v:.0%}" for k, v in active]
    if tp != "all":
        parts.append(tp)
    return " ".join(parts) if parts else "全均分"


class ScreeningConfigTuner:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def quick_sweep(self, configs: List[Dict], pool="all", top_n=50,
                    code_prefixes=None,
                    progress: Optional[Dict] = None,
                    regime=None) -> Dict:
        """快扫: 多组配置在当前数据上并行筛选,对比结果"""
        screener = get_screener()
        total = len(configs)
        results = []
        lock = threading.Lock()
        done = [0]

        # ── 预计算股票数据快照,供所有 sweeps 复用 ──
        if progress is not None:
            progress.update({"stage": "precompute", "current": 0, "total": 0,
                           "message": "预加载全部股票数据..."})
        snapshot = screener.precompute_snapshot(
            pool=pool, code_prefixes=code_prefixes, skip_tech=False)
        logger.info(f"[调优] 快照预计算完成: {len(snapshot)} 只, 开始并行快扫")

        def _update_progress(msg=None):
            if progress is None:
                return
            progress["current"] = done[0]
            progress["total"] = total
            if msg:
                progress["message"] = msg
            else:
                progress["message"] = f"快扫中 ({done[0]}/{total})"

        def _run_one(idx, cfg):
            try:
                r = screener.screen(
                    pool=pool, top_n=top_n, skip_l1=True, skip_l2=True, skip_debate=True,
                    code_prefixes=code_prefixes,
                    source_weights=cfg.get("source_weights"),
                    tech_preset=cfg.get("tech_preset", "all"),
                    skip_tech=cfg.get("skip_tech", False),
                    skip_qlib=cfg.get("skip_qlib", False),
                    skip_sent=cfg.get("skip_sent", False),
                    skip_fundamental=cfg.get("skip_fundamental", False),
                    skip_flow=cfg.get("skip_flow", False),
                    skip_north_flow=cfg.get("skip_north_flow", False),
                    skip_size_turnover=cfg.get("skip_size_turnover", False),
                    skip_momentum=cfg.get("skip_momentum", False),
                    skip_volatility=cfg.get("skip_volatility", False),
                    job_id=f"tuner_{idx}",
                    snapshot=snapshot,
                    regime=regime,
                )
                candidates = r.get("candidates", [])
                recommended = r.get("recommended", [])
                l0_weights = r.get("stages", {}).get("l0", {}).get("weights", {})
                scores = [c.get("l1_score", c.get("l0_score", 0)) for c in candidates]

                return {
                    "config": cfg,
                    "label": _config_label(cfg),
                    "candidate_count": len(candidates),
                    "recommended_count": len(recommended),
                    "score_mean": round(float(np.mean(scores)), 4) if scores else 0,
                    "score_std": round(float(np.std(scores)), 4) if scores else 0,
                    "score_max": round(float(max(scores)), 4) if scores else 0,
                    "score_min": round(float(min(scores)), 4) if scores else 0,
                    "high_score_ratio": round(
                        sum(1 for s in scores if s > 0.5) / max(len(scores), 1), 3
                    ),
                    "effective_weights": l0_weights,
                    "top_codes": [c["code"] for c in candidates[:10]],
                    "top_scores": [round(c.get("l1_score", c.get("l0_score", 0)), 4)
                                  for c in candidates[:10]],
                    "status": "done",
                }
            except Exception as e:
                logger.error(f"快扫配置 {idx} 失败: {e}")
                return {"config": cfg, "label": _config_label(cfg), "status": "error", "error": str(e)}
            finally:
                with lock:
                    done[0] += 1
                    _update_progress()

        logger.info(f"[调优] 快扫开始: {total} 组配置, pool={pool}, top_n={top_n}")
        if progress is not None:
            progress.update({"stage": "sweep", "current": 0, "total": total, "message": "快扫准备中..."})
        max_workers = min(16, max(4, total // 2))
        with thread_pool.default.borrow() as executor:
            futures = {executor.submit(_run_one, i, cfg): i for i, cfg in enumerate(configs)}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                except Exception as e:
                    logger.error(f"快扫任务异常: {e}")

        results.sort(key=lambda x: (
            x.get("high_score_ratio", 0),
            x.get("score_mean", 0),
            x.get("candidate_count", 0),
        ), reverse=True)

        logger.info(f"[调优] 快扫完成: {len(results)} 组结果")
        return {
            "total": total,
            "completed": len(results),
            "results": results,
            "best": results[0] if results else None,
        }

    def _precompute_snapshots_parallel(self, eval_dates, pool, code_prefixes,
                                        progress=None, label="调优",
                                        min_turnover=50_000_000, skip_limit_up=True):
        """并行预计算多个历史日期的 snapshot(磁盘缓存加速,跨 Phase 1/2 复用)"""
        snapshot_cache = {}
        lock = threading.Lock()
        done = [0]
        total = len(eval_dates)
        max_w = min(12, max(4, total // 2))  # 限制并发,每 snapshot 内部 20 线程

        def _fetch_one(d):
            # 先查磁盘缓存
            cached = _load_cached_snapshot(d, pool)
            if cached is not None:
                with lock:
                    done[0] += 1
                    if progress:
                        progress["current"] = done[0]
                        progress["total"] = total
                        progress["message"] = f"命中缓存 ({done[0]}/{total}) {d}"
                        progress["stage"] = "precompute"
                logger.info(f"[{label}] {d} 命中磁盘缓存, {len(cached)} 只 ({done[0]}/{total})")
                return d, cached

            try:
                snap = get_screener().precompute_snapshot(
                    pool=pool, code_prefixes=code_prefixes, skip_tech=False,
                    eval_date=d, min_turnover=min_turnover, skip_limit_up=skip_limit_up,
                    fast=True)
                if snap:
                    _save_cached_snapshot(d, pool, snap)
                with lock:
                    done[0] += 1
                    if progress:
                        progress["current"] = done[0]
                        progress["total"] = total
                        progress["message"] = f"预加载 ({done[0]}/{total}) {d}"
                        progress["stage"] = "precompute"
                logger.info(f"[{label}] {d} precompute 完成, {len(snap)} 只 ({done[0]}/{total})")
                return d, snap
            except Exception as e:
                logger.error(f"[{label}] {d} precompute 失败: {e}")
                with lock:
                    done[0] += 1
                return d, None

        logger.info(f"[{label}] 并行预计算 {total} 个历史日期快照 (workers={max_w})...")
        with thread_pool.default.borrow() as ex:
            futures = {ex.submit(_fetch_one, d): d for d in eval_dates}
            for future in as_completed(futures):
                d, snap = future.result()
                snapshot_cache[d] = snap

        ok = sum(1 for v in snapshot_cache.values() if v is not None)
        logger.info(f"[{label}] 快照预计算完成: {ok}/{total} 成功")
        return snapshot_cache

    def historical_validate(self, configs: List[Dict], eval_weeks=12,
                            date_start=None, date_end=None,
                            top_k=10, pool="all",
                            code_prefixes=None,
                            allow_all_signals=None,
                            skip_l1=False,
                            snapshot_cache: Dict = None,
                            top_n=None, date_step: int = 5,
                            progress: Optional[Dict] = None,
                            regime=None) -> Dict:
        """历史验证: 在多个历史日期回测配置表现,计算 forward return

        使用 config 自身的 skip 设置控制信号开关。
        注意: qlib/情绪/基本面/资金流 存在潜在前视偏差,报告中会标注提醒。
        snapshot_cache: 可选,外部预计算好的 {date: snapshot},避免重复 precompute
        top_n: 候选池大小,None 则为 top_k * 5
        """
        if top_n is None:
            top_n = top_k * 5
        screener = get_screener()
        eval_dates = self._get_eval_dates(weeks=eval_weeks, date_start=date_start, date_end=date_end, step=date_step)
        if not eval_dates:
            return {"error": "无法获取交易日列表"}

        total_jobs = len(configs) * len(eval_dates)
        results = []
        lock = threading.Lock()
        done = [0]
        completed_configs = [0]

        # ── 按日期顺序处理：一次只加载一个快照 (～0.8GB)，用完释放 ──
        # 不再预计算全部日期快照到内存——Redis/磁盘加载毫秒级，避免 49 日期 × 0.8GB = 40GB OOM
        if snapshot_cache is not None:
            logger.info(f"[调优] 使用外部快照缓存 ({len(snapshot_cache)} 日期)")
            _ext_snapshots = snapshot_cache
        else:
            _ext_snapshots = None

        logger.info(f"[调优] 开始验证 {total_jobs} 任务 (逐日期加载快照)")

        def _update_progress(msg=None):
            if progress is None:
                return
            progress["current"] = done[0]
            progress["total"] = total_jobs
            progress["config_done"] = completed_configs[0]
            progress["config_total"] = len(configs)
            if msg:
                progress["message"] = msg
            else:
                progress["message"] = f"验证中 配置{completed_configs[0]}/{len(configs)} 日期{done[0]}/{total_jobs}"

        def _run_one(cfg_idx, cfg, eval_date, snap):
            _t0 = time.time()
            try:

                _t1 = time.time()
                r = screener.screen(
                    pool=pool, top_n=top_n, skip_l1=skip_l1, skip_l2=True, skip_debate=True,
                    skip_qlib=cfg.get("skip_qlib", False),
                    skip_sent=cfg.get("skip_sent", False),
                    skip_fundamental=cfg.get("skip_fundamental", False),
                    skip_flow=cfg.get("skip_flow", False),
                    skip_north_flow=cfg.get("skip_north_flow", False),
                    skip_size_turnover=cfg.get("skip_size_turnover", False),
                    skip_tech=cfg.get("skip_tech", False),
                    skip_momentum=cfg.get("skip_momentum", False),
                    skip_volatility=cfg.get("skip_volatility", False),
                    code_prefixes=code_prefixes,
                    source_weights=cfg.get("source_weights"),
                    tech_preset=cfg.get("tech_preset", "all"),
                    eval_date=eval_date,
                    job_id=f"hist_{cfg_idx}_{eval_date.strftime('%Y%m%d')}",
                    snapshot=snap,
                    regime=regime,
                )
                logger.info(f"[调优] {eval_date} screen 完成, {len(r.get('candidates',[]))} 候选 ({time.time()-_t1:.1f}s)")
                candidates = r.get("candidates", [])
                top_codes = [c["code"] for c in candidates[:top_k]]
                top_scores = [c.get("l1_score", c.get("l0_score", 0)) for c in candidates[:top_k]]
                forward = self._compute_forward_returns(top_codes, eval_date)
                return {
                    "eval_date": eval_date.isoformat(),
                    "top_codes": top_codes,
                    "top_scores": top_scores,
                    "forward_5d": forward.get("5d"),
                    "forward_10d": forward.get("10d"),
                    "forward_20d": forward.get("20d"),
                    "per_code_returns": forward.get("per_code", {}),
                    "neutral_ic": self._neutral_ic_for(candidates, eval_date),
                    "candidate_count": len(candidates),
                }
            except Exception as e:
                logger.error(f"历史验证 {cfg_idx}/{eval_date} 失败: {e}")
                return {
                    "eval_date": eval_date.isoformat(),
                    "status": "error", "error": str(e),
                }
            finally:
                with lock:
                    done[0] += 1
                    _update_progress()

        logger.info(f"[调优] 历史验证开始: {len(configs)} 配置 x {len(eval_dates)} 日期 = {total_jobs}")
        if progress is not None:
            progress.update({"stage": "validate", "current": 0, "total": total_jobs,
                           "config_done": 0, "config_total": len(configs),
                           "message": "验证准备中..."})

        # ── 逐日期处理：每日期加载一个快照 → 所有配置并行验证 → 释放快照 ──
        # 内存峰值 = 1 个快照 (~0.8GB) + 并发结果，而非 49 个快照 (~40GB)
        config_date_results = {i: [] for i in range(len(configs))}
        for date_idx, eval_date in enumerate(eval_dates):
            _t_date = time.time()
            # 加载快照：优先外部缓存 → Redis/磁盘 → 重新计算
            snap = None
            if _ext_snapshots is not None:
                snap = _ext_snapshots.get(eval_date)
            if snap is None:
                snap = _load_cached_snapshot(eval_date, pool)
            if snap is None:
                try:
                    snap = screener.precompute_snapshot(
                        pool=pool, code_prefixes=code_prefixes, skip_tech=False,
                        eval_date=eval_date, fast=True)
                    if snap:
                        _save_cached_snapshot(eval_date, pool, snap)
                except Exception as e:
                    logger.error(f"[调优] {eval_date} 快照失败: {e}")

            if snap is None:
                logger.warning(f"[调优] {eval_date} 快照不可用，跳过")
                continue

            logger.info(f"[调优] 日期 {date_idx+1}/{len(eval_dates)}: {eval_date} "
                       f"快照={len(snap)}只 ({time.time()-_t_date:.1f}s)")

            # 所有配置对此日期并行验证（快照共享，纯CPU计算，多开worker）
            max_workers = min(16, max(4, len(configs) // 4))
            with thread_pool.default.borrow() as executor:
                futures = {
                    executor.submit(_run_one, cfg_idx, configs[cfg_idx], eval_date, snap): cfg_idx
                    for cfg_idx in range(len(configs))
                }
                for future in as_completed(futures):
                    cfg_idx = futures[future]
                    try:
                        r = future.result()
                        if r:
                            config_date_results[cfg_idx].append(r)
                    except Exception as e:
                        logger.error(f"历史验证任务异常: {e}")

            del snap
            with lock:
                done[0] += len(configs)
            if progress:
                progress["current"] = done[0]
                progress["message"] = f"日期 {date_idx+1}/{len(eval_dates)} {eval_date} 完成 ({time.time()-_t_date:.1f}s)"

        # ── 聚合各配置结果 ──
        config_results = []
        for cfg_idx, cfg in enumerate(configs):
            date_results = config_date_results[cfg_idx]
            date_results.sort(key=lambda x: x["eval_date"])
            agg = self._aggregate_validation(date_results)
            agg["config"] = cfg
            agg["label"] = _config_label(cfg)
            agg["date_count"] = len(date_results)
            agg["date_details"] = [
                {
                    "date": dr.get("eval_date", ""),
                    "codes": dr.get("top_codes", []),
                    "scores": dr.get("top_scores", []),
                    "ret_5d": dr.get("forward_5d"),
                    "ret_10d": dr.get("forward_10d"),
                    "ret_20d": dr.get("forward_20d"),
                    "candidate_count": dr.get("candidate_count", 0),
                }
                for dr in date_results
            ]
            config_results.append(agg)
            with lock:
                completed_configs[0] = cfg_idx + 1

        config_results.sort(key=lambda x: x.get("fitness", 0), reverse=True)
        for i, cr in enumerate(config_results):
            cr["rank"] = i + 1

        logger.info(f"[调优] 历史验证完成: {len(config_results)} 配置")
        return {
            "eval_dates": [d.isoformat() for d in eval_dates],
            "total_dates": len(eval_dates),
            "results": config_results,
            "best": config_results[0] if config_results else None,
        }

    def _get_eval_dates(self, weeks: int = None, date_start: str = None,
                        date_end: str = None, step: int = 5) -> List[datetime.date]:
        """获取历史交易日列表(每隔 step 个交易日取一个)

        - date_start/date_end (YYYY-MM-DD) 优先
        - 否则用 weeks 从 CURDATE 回推
        """
        from core.data import Database
        try:
            if date_start and date_end:
                rows = Database.fetchall(
                    "SELECT DISTINCT trade_date FROM daily_quotes "
                    "WHERE trade_date >= %s AND trade_date <= %s "
                    "ORDER BY trade_date",
                    (date_start, date_end),
                )
            elif weeks:
                rows = Database.fetchall(
                    "SELECT DISTINCT trade_date FROM daily_quotes "
                    "WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s WEEK) "
                    "ORDER BY trade_date",
                    (weeks,),
                )
            else:
                rows = Database.fetchall(
                    "SELECT DISTINCT trade_date FROM daily_quotes "
                    "WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK) "
                    "ORDER BY trade_date",
                )

            if not rows:
                return []
            all_dates = sorted(set(
                row["trade_date"] if isinstance(row["trade_date"], datetime.date)
                else row["trade_date"].date() if hasattr(row["trade_date"], "date")
                else datetime.date.fromisoformat(str(row["trade_date"])[:10])
                for row in rows
            ))
            return all_dates[::step]
        except Exception as e:
            logger.warning(f"获取交易日列表失败: {e}")
            today = today_date()
            if date_start and date_end:
                try:
                    start = datetime.date.fromisoformat(date_start)
                    end = datetime.date.fromisoformat(date_end)
                except ValueError:
                    start = today - datetime.timedelta(weeks=12) if weeks else today - datetime.timedelta(weeks=12)
                    end = today
            else:
                w = weeks or 12
                start = today - datetime.timedelta(weeks=w)
                end = today
            dates = []
            d = start
            while d <= end:
                if d.weekday() < 5:
                    dates.append(d)
                d += datetime.timedelta(days=1)
            return dates[::5]

    def _compute_forward_returns(self, codes: List[str], eval_date: datetime.date,
                                 periods=(5, 10, 20)) -> Dict:
        """计算 top-K 股票在后续 N 日的 forward return。
        优先查 daily_forward_ret 物化表 (ForwardReturnCache 预热后), 回退实时 SQL。"""
        per_code = {}
        if not codes:
            return {f"{p}d": 0.0 for p in periods} | {"per_code": {}}

        # 优先走物化表 (ForwardReturnCache 预热后数据, L0 内存缓存)
        from core.domain.factors.forward_returns import get_forward_cache
        fcache = get_forward_cache()
        try:
            raw = fcache.get_multi_period(codes, eval_date, periods=list(periods))
            if raw and any(v for v in raw.values()):
                for p in periods:
                    for code, val in raw.get(p, {}).items():
                        per_code.setdefault(code, {})[f"{p}d"] = round(val, 4)
                if per_code:
                    forward = {}
                    for p in periods:
                        key = f"{p}d"
                        vals = [v[key] for v in per_code.values() if key in v]
                        forward[key] = round(sum(vals) / max(len(vals), 1), 4) if vals else 0.0
                    forward["per_code"] = per_code
                    return forward
        except Exception:
            pass

        # 回退: 实时 SQL (原逻辑)
        from core.data import Database
        start = eval_date.isoformat()
        max_period = max(periods)
        end = (eval_date + datetime.timedelta(days=max_period + 10)).isoformat()
        ph = ",".join(["%s"] * len(codes))
        rows = Database.fetchall(
            f"SELECT code, trade_date, open, close FROM daily_quotes "
            f"WHERE code IN ({ph}) AND trade_date>=%s AND trade_date<=%s "
            f"ORDER BY code, trade_date",
            codes + [start, end],
        )
        code_rows: Dict[str, list] = {}
        for r in rows:
            code_rows.setdefault(r["code"], []).append(r)

        for code in codes:
            rows_c = code_rows.get(code, [])
            if len(rows_c) < 2:
                continue
            code_rets = {}
            for period in periods:
                entry_price = float(rows_c[1].get("open", 0) or 0)
                if entry_price <= 0:
                    entry_price = float(rows_c[0].get("close", 0) or 0)
                if entry_price <= 0:
                    continue
                idx = min(period, len(rows_c) - 1)
                exit_price = float(rows_c[idx].get("close", 0) or 0)
                if exit_price <= 0:
                    continue
                code_rets[f"{period}d"] = round((exit_price / entry_price) - 1, 4)
            if code_rets:
                per_code[code] = code_rets

        forward = {}
        for period in periods:
            key = f"{period}d"
            vals = [v[key] for v in per_code.values() if key in v]
            forward[key] = round(sum(vals) / max(len(vals), 1), 4) if vals else 0.0

        forward["per_code"] = per_code
        return forward

    def _cached_forward_returns(self, codes, eval_date, cache, lock):
        """带缓存的 forward return 查询，同日期多信号共享，避免重复 SQL"""
        if not codes:
            return {f"{p}d": 0.0 for p in (5, 10, 20)} | {"per_code": {}}
        date_str = eval_date.isoformat()
        with lock:
            missing = [c for c in codes if (date_str, c) not in cache]
        if missing:
            fwd = self._compute_forward_returns(missing, eval_date)
            with lock:
                for code, cr in fwd.get("per_code", {}).items():
                    cache[(date_str, code)] = cr
        with lock:
            per_code = {}
            for code in codes:
                cr = cache.get((date_str, code), {})
                if cr:
                    per_code[code] = cr
        forward = {}
        for period in (5, 10, 20):
            key = f"{period}d"
            vals = [v[key] for v in per_code.values() if key in v]
            forward[key] = round(sum(vals) / max(len(vals), 1), 4) if vals else 0.0
        forward["per_code"] = per_code
        return forward

    def _get_forward_return(self, code: str, eval_date: datetime.date,
                            period: int) -> Optional[float]:
        """单只查询（保留用于非批量场景）"""
        return self._compute_forward_returns([code], eval_date, (period,)).get(f"{period}d")

    def _aggregate_validation(self, date_results: List[Dict]) -> Dict:
        """聚合单个配置在所有历史日期的验证结果"""
        valid = [r for r in date_results if r.get("forward_5d") is not None]
        if not valid:
            return {"fitness": -999, "error": "无有效验证数据"}

        n = len(valid)
        ret_5d = [r.get("forward_5d", 0) or 0 for r in valid]
        ret_10d = [r.get("forward_10d", 0) or 0 for r in valid]
        ret_20d = [r.get("forward_20d", 0) or 0 for r in valid]

        mean_5d = float(np.mean(ret_5d))
        mean_10d = float(np.mean(ret_10d))
        mean_20d = float(np.mean(ret_20d))
        std_5d = float(np.std(ret_5d))
        std_10d = float(np.std(ret_10d))
        std_20d = float(np.std(ret_20d))
        hit_5d = sum(1 for r in ret_5d if r > 0) / n
        hit_10d = sum(1 for r in ret_10d if r > 0) / n
        hit_20d = sum(1 for r in ret_20d if r > 0) / n

        def _sharpe(mean_v, std_v):
            return round(mean_v / std_v, 4) if std_v > 1e-9 else 0.0

        from core.domain.factors.neutralize import time_weighted_mean
        neut_pairs = [(r.get("eval_date", ""), r.get("neutral_ic")) for r in valid if r.get("neutral_ic") is not None]
        if neut_pairs:
            neut_pairs.sort(key=lambda x: x[0])
            rank_ic = time_weighted_mean([v for _, v in neut_pairs], decay=0.90)
        else:
            # 降级路径: 逐日做中性化 IC 后时间加权, 而非直接 pool 全截面 Spearman
            try:
                daily_neut_ics = []
                from core.domain.factors.market_cap import get_market_cap_map
                from core.domain.factors.neutralize import cross_sectional_neutralize
                for r in valid:
                    codes = r.get("top_codes", [])
                    scores = r.get("top_scores", [])
                    per_code = r.get("per_code_returns", {})
                    if len(codes) < 20 or not per_code:
                        continue
                    fwd_map = {c: per_code.get(c, {}).get("20d") for c in codes}
                    fwd_rets = [fwd_map.get(c) for c in codes]
                    fwd_rets = [v for v in fwd_rets if v is not None]
                    valid_codes = [c for c in codes if fwd_map.get(c) is not None]
                    valid_scores = [s for c, s in zip(codes, scores) if fwd_map.get(c) is not None]
                    if len(valid_codes) < 10:
                        continue
                    mktcap_map = get_market_cap_map(valid_codes, r.get("eval_date"))
                    industry_map = self._industry_map_cached()
                    resid = cross_sectional_neutralize(valid_codes, valid_scores, mktcap_map, industry_map)
                    if resid is None:
                        continue
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(list(resid.values()), fwd_rets)
                    if corr is not None and not np.isnan(corr):
                        daily_neut_ics.append(float(corr))
                rank_ic = time_weighted_mean(daily_neut_ics, decay=0.90) if daily_neut_ics else 0.0
            except Exception:
                rank_ic = 0.0

        fitness = round(
            mean_20d * 0.4 + hit_20d * 0.3 + rank_ic * 0.2 + (1.0 - min(std_20d, 0.3) / 0.3) * 0.1,
            4,
        )

        return {
            "n_dates": n,
            "mean_5d": round(mean_5d, 4),
            "mean_10d": round(mean_10d, 4),
            "mean_20d": mean_20d,
            "std_5d": std_5d, "std_10d": std_10d, "std_20d": std_20d,
            "sharpe_5d": _sharpe(mean_5d, std_5d),
            "sharpe_10d": _sharpe(mean_10d, std_10d),
            "sharpe_20d": _sharpe(mean_20d, std_20d),
            "hit_rate_5d": round(hit_5d, 3),
            "hit_rate_10d": round(sum(1 for r in ret_10d if r > 0) / n, 3),
            "hit_rate_20d": round(hit_20d, 3),
            "rank_ic": round(rank_ic, 4),
            "fitness": fitness,
        }

    def _build_regime_breakdown(self, signal_date_results, date_regimes, thresholds):
        """按 (信号, 市态) 分组统计，产出交叉表"""
        REGIME_ORDER = ["trending_up", "trending_down", "ranging", "volatile"]
        REGIME_LABELS = {
            "trending_up": "趋势上涨", "trending_down": "趋势下跌",
            "ranging": "震荡盘整", "volatile": "高波动",
        }
        signals_breakdown = []
        regime_totals = {r: 0 for r in REGIME_ORDER}
        for date_str, rv in date_regimes.items():
            if rv in regime_totals:
                regime_totals[rv] += 1

        for key, sr in signal_date_results.items():
            info = sr["info"]
            date_results = sr["date_results"]
            # 按市态分组
            by_regime = {r: [] for r in REGIME_ORDER}
            for dr in date_results:
                rv = date_regimes.get(dr.get("eval_date", ""))
                if rv in by_regime:
                    by_regime[rv].append(dr)

            regime_metrics = {}
            for rv in REGIME_ORDER:
                rdrs = by_regime[rv]
                if len(rdrs) >= 3:  # 至少 3 个样本
                    agg = self._aggregate_validation(rdrs)
                    regime_metrics[rv] = {
                        "n_dates": agg.get("n_dates", 0),
                        "mean_20d": agg.get("mean_20d", 0) or 0,
                        "hit_rate_20d": agg.get("hit_rate_20d", 0) or 0,
                        "sharpe_20d": agg.get("sharpe_20d", 0) or 0,
                        "rank_ic": agg.get("rank_ic", 0) or 0,
                    }
                else:
                    regime_metrics[rv] = None

            signals_breakdown.append({
                "key": info["key"], "label": info["label"],
                "data_safety": info.get("data_safety", "safe"),
                "is_sub": sr["is_sub"],
                "regime_metrics": regime_metrics,
            })

        # 每个市态找最优信号
        best_per_regime = {}
        for rv in REGIME_ORDER:
            best = None
            for sb in signals_breakdown:
                rm = sb["regime_metrics"].get(rv)
                if rm and rm["n_dates"] >= 3:
                    if best is None or rm["mean_20d"] > best["mean_20d"]:
                        best = {"key": sb["key"], "label": sb["label"], **rm}
            best_per_regime[rv] = best

        return {
            "regime_order": REGIME_ORDER,
            "regime_labels": {r: REGIME_LABELS.get(r, r) for r in REGIME_ORDER},
            "regime_date_counts": regime_totals,
            "signals": signals_breakdown,
            "best_per_regime": best_per_regime,
        }

    def _industry_map_cached(self):
        if getattr(self, "_ind_map", None) is None:
            try:
                from core.data.fetcher import DataFetcher
                self._ind_map = DataFetcher().get_industry_map() or {}
            except Exception:
                self._ind_map = {}
        return self._ind_map

    def _neutral_ic_for(self, candidates, eval_date, fwd_cache=None, fwd_lock=None):
        """对全截面 candidates 的 l0_score 做市值+行业中性化后, 与 20 日 forward return 的当日 Rank IC。
        缺市值时退化为仅行业中性化; 截面不足返回 None (回退旧 pooled IC)。"""
        try:
            from core.domain.factors.market_cap import get_market_cap_map
            from core.domain.factors.neutralize import neutralized_daily_ic
            if not candidates or len(candidates) < 20:
                return None
            codes = [c["code"] for c in candidates]
            scores = [c.get("l0_score", 0) for c in candidates]
            if fwd_cache is not None:
                fwd = self._cached_forward_returns(codes, eval_date, fwd_cache, fwd_lock)
            else:
                fwd = self._compute_forward_returns(codes, eval_date)
            per = fwd.get("per_code", {})
            fwd_map = {c: per.get(c, {}).get("20d") for c in codes}
            mktcap = get_market_cap_map(codes, eval_date)
            return neutralized_daily_ic(codes, scores, fwd_map, mktcap, self._industry_map_cached())
        except Exception:
            return None

    def _calc_rank_ic(self, date_results: List[Dict]) -> float:
        """计算 Rank IC: 将所有日期的 (score, forward_return) 对池化后做 Spearman 秩相关"""
        all_scores = []
        all_rets = []
        for r in date_results:
            scores = r.get("top_scores", [])
            codes = r.get("top_codes", [])
            per_code = r.get("per_code_returns", {})
            if not scores or not per_code:
                continue
            for score, code in zip(scores, codes):
                code_ret = per_code.get(code, {}).get("20d")
                if code_ret is not None:
                    all_scores.append(score)
                    all_rets.append(code_ret)

        if len(all_scores) < 10:
            return 0.0

        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(all_scores, all_rets)
            if not (np.isnan(corr) or np.isinf(corr)):
                return round(float(corr), 4)
        except Exception:
            try:
                return round(float(pd.Series(all_scores).rank().corr(
                    pd.Series(all_rets).rank())), 4)
            except Exception:
                pass
        return 0.0


    def auto_tune(self, weights: Dict[str, float], skips: Dict[str, bool],
                  tech_presets: List[str], l0_thresholds: List[float],
                  pool="all", top_n=30, eval_weeks=12, top_k=10,
                  date_start=None, date_end=None,
                  code_prefixes=None,
                  progress: Optional[Dict] = None) -> Dict:
        """全自动调优: 生成配置 -> 快扫 -> 自动选优 -> 历史验证

        如果只传入了一组权重/skip, 自动展开 WEIGHT_PRESETS x SKIP_PRESETS
        产生多组有意义的对比配置."""

        # ── 智能展开: 单一配置时自动使用预设模板 ──
        if len(tech_presets) * len(l0_thresholds) <= 3:
            logger.info("[自动调优] 配置组合少,自动展开预设模板")
            weight_list = []
            skip_list = []

            # 把当前权重映射到最接近的预设类别
            has_qlib = not skips.get("skip_qlib", False)
            has_sent = not skips.get("skip_sent", False)
            tech_w = weights.get("tech", 0.20)

            preset_order = []
            if has_qlib and has_sent:
                preset_order = ["balanced", "qlib_heavy", "tech_heavy", "momentum_focused", "default"]
            elif has_qlib:
                preset_order = ["qlib_heavy", "tech_heavy", "balanced", "default"]
            elif has_sent:
                preset_order = ["balanced", "tech_heavy", "momentum_focused", "default"]
            else:
                preset_order = ["tech_heavy", "tech_only", "momentum_focused", "default"]

            for pname in preset_order:
                if pname in WEIGHT_PRESETS:
                    weight_list.append(dict(WEIGHT_PRESETS[pname]))

            # skip 预设: 从 all_on 开始,逐个关闭非核心信号做对比
            skip_preset_order = ["all_on", "no_fundamental", "no_flow",
                                 "core_only", "tech_only"] if not has_qlib else [
                                     "all_on", "no_fundamental", "no_flow", "core_only"]
            for sname in skip_preset_order:
                if sname in SKIP_PRESETS:
                    skip_list.append(dict(SKIP_PRESETS[sname]))

            # L0 阈值也展开
            if len(l0_thresholds) <= 1:
                l0_thresholds = [0.25, 0.30, 0.35, 0.40]

            if len(tech_presets) <= 1:
                tech_presets = ["all", "trend", "momentum"]

            configs = []
            for w in weight_list:
                for s in skip_list:
                    for tp in tech_presets:
                        for th in l0_thresholds:
                            cfg = {"source_weights": dict(w), "tech_preset": tp}
                            cfg.update(s)
                            configs.append(cfg)

            # 限制配置数量避免过载
            if len(configs) > 60:
                logger.info(f"[自动调优] 配置过多 ({len(configs)}), 截断为 60 组")
                import random
                random.seed(42)
                configs = random.sample(configs, 60)
        else:
            configs = []
            for tp in tech_presets:
                for th in l0_thresholds:
                    cfg = {
                        "source_weights": dict(weights),
                        "tech_preset": tp,
                    }
                    cfg.update(skips)
                    configs.append(cfg)

        logger.info(f"[自动调优] 生成 {len(configs)} 组配置, 开始快扫...")
        if progress is not None:
            progress.update({"stage": "sweep", "current": 0, "total": 0, "message": "生成配置完成,开始快扫..."})

        sweep_result = self.quick_sweep(configs, pool=pool, top_n=top_n,
                                        code_prefixes=code_prefixes, progress=progress)

        results = sweep_result.get("results", [])
        valid = [r for r in results if r.get("status") != "error"]
        if not valid:
            return {"error": "快扫无有效结果", "sweep": sweep_result}

        top3 = sorted(valid, key=lambda x: (
            x.get("high_score_ratio", 0),
            x.get("score_mean", 0),
        ), reverse=True)[:3]

        logger.info(f"[自动调优] 快扫完成, 自动选中 top-3: {[r['label'] for r in top3]}, 开始历史验证...")
        if progress is not None:
            progress.update({"stage": "selecting", "current": 0, "total": 0,
                           "message": f"快扫完成,自动选中最优3组: {', '.join(r['label'] for r in top3)}"})

        top3_configs = [r["config"] for r in top3]
        validate_result = self.historical_validate(top3_configs, eval_weeks=eval_weeks, top_k=top_k,
                                                    pool=pool, code_prefixes=code_prefixes,
                                                    date_start=date_start, date_end=date_end,
                                                    progress=progress)

        result = {
            "sweep": sweep_result,
            "top3_selected": [r["label"] for r in top3],
            "top3_scores": [r["score_mean"] for r in top3],
            "validate": validate_result,
        }

        # 生成报告
        try:
            report_path = self._generate_report(weights, skips, tech_presets, l0_thresholds, result)
            result["report_path"] = report_path
        except Exception as e:
            logger.warning(f"报告生成失败: {e}")

        return result

    # ══════════════════════════════════════════════
    # Phase 1: 信号有效性检测
    # ══════════════════════════════════════════════

    def _build_single_signal_skips(self, target_key):
        """构造只启用 target_key 信号的 skip dict"""
        skips = {"skip_tech": True, "skip_qlib": True, "skip_sent": True,
                 "skip_fundamental": True, "skip_flow": True,
                 "skip_north_flow": True, "skip_size_turnover": True,
                 "skip_momentum": True, "skip_volatility": True,
                 "skip_illiq": True, "skip_overnight_gap": True}
        for src in SIGNAL_SOURCES:
            if src["key"] == target_key:
                skips[src["skip_key"]] = False
                break
        return skips

    def _build_tech_sub_skips(self):
        """构造只启用技术子指标的 skip(其余信号全关)"""
        return {
            "skip_tech": False, "skip_qlib": True, "skip_sent": True,
            "skip_fundamental": True, "skip_flow": True,
            "skip_north_flow": True, "skip_size_turnover": True,
            "skip_momentum": True, "skip_volatility": True,
            "skip_illiq": True, "skip_overnight_gap": True,
        }

    def signal_validity_test(self, pool="all", eval_weeks=18,
                             date_start=None, date_end=None,
                             top_k=10, code_prefixes=None,
                             top_n=None, qlib_model=None,
                             thresholds=None, skip_tech_subs=False,
                             date_step=5, skip_financial_prefetch=True,
                             progress=None, regime=None,
                             signal_filter=None) -> Dict:
        """Phase 1: 逐个检测每个信号源的历史有效性

        内存优化v2: 按日期顺序处理,每次只保留一个 snapshot (~1GB),
        处理完即释放,而非全部日期同时加载(12+ GB)。skip_financial_prefetch 默认 True。

        skip_tech_subs: True 跳过 6 个技术子指标,只测 7 大类(省 ~50% 时间)
        signal_filter: 可选 list,只测指定的信号key; None=全部
        date_step: 每隔 N 个交易日取一个测试日期(默认5,调大更快)
        """
        try:
            import psutil
            def _mem():
                return psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            def _mem():
                return -1

        th = dict(SIGNAL_VALIDITY_THRESHOLDS)
        if thresholds:
            th.update(thresholds)
        screener = get_screener()
        eval_dates = self._get_eval_dates(weeks=eval_weeks, date_start=date_start, date_end=date_end, step=date_step)
        if not eval_dates:
            return {"error": "所选日期范围内无交易日数据", "eval_dates": [], "signals": []}

        actual_top_n = top_n or top_k * 5

        all_signals = []
        # 7 大信号源
        for src in SIGNAL_SOURCES:
            skips = self._build_single_signal_skips(src["key"])
            cfg = {
                "source_weights": {"tech": 1.0, "qlib": 1.0, "sentiment": 1.0,
                                   "fundamental": 1.0, "flow": 1.0,
                                   "momentum": 1.0, "volatility": 1.0},
                "tech_preset": "all",
            }
            cfg.update(skips)
            all_signals.append({"info": src, "config": cfg, "is_sub": False})
        # 6 个技术子指标(可选跳过,省 ~50% 时间)
        if not skip_tech_subs:
            for sub in TECH_SUB_SIGNALS:
                skips = self._build_tech_sub_skips()
                cfg = {
                    "source_weights": {"tech": 1.0},
                    "tech_preset": sub["tech_preset"],
                }
                cfg.update(skips)
                sub_info = {"key": sub["key"], "label": sub["label"],
                            "skip_key": "skip_tech", "weight_key": "tech",
                            "data_safety": "safe", "tech_preset": sub["tech_preset"]}
                all_signals.append({"info": sub_info, "config": cfg, "is_sub": True})
        # 白盒子信号单信号测试(通过 SIGNAL_SOURCES 已有 top-level 条目, 这里只做参数变体)

        # 华泰多因子子信号: 13个连续 + 4个离散
        ht_all = HT_CONTINUOUS_SIGNALS + HT_DISCRETE_SIGNALS
        for ht_sub in ht_all:
            skips = self._build_single_signal_skips("ht_factors")
            cfg = {"source_weights": {"ht_factors": 1.0},
                   "tech_preset": "all"}
            cfg.update(skips)
            sub_info = {"key": ht_sub["key"], "label": ht_sub["label"],
                       "skip_key": "skip_ht", "weight_key": "ht_factors",
                       "data_safety": "safe", "ht_key": ht_sub["ht_key"],
                       "signal_type": ht_sub.get("signal_type", "continuous")}
            all_signals.append({"info": sub_info, "config": cfg, "is_sub": True})

        # 6 个白盒因子参数变体
        for wb_var in WB_FACTOR_VARIANTS:
            for param_val in wb_var["param_values"]:
                skips = dict(wb_var["base_skips"])
                cfg = {
                    "source_weights": dict(wb_var["base_weights"]),
                    "tech_preset": "all",
                }
                extra = wb_var["extra_params"](param_val)
                cfg.update(extra)
                cfg.update(skips)
                var_key = f"{wb_var['template_key']}_n{param_val}"
                var_label = f"{wb_var['label']}({wb_var['param_name']}={param_val})"
                var_info = {
                    "key": var_key, "label": var_label,
                    "skip_key": wb_var["skip_key"], "weight_key": wb_var["weight_key"],
                    "data_safety": wb_var["data_safety"],
                    "template_key": wb_var["template_key"],
                    "param_value": param_val,
                    "is_wb_variant": True,
                }
                all_signals.append({"info": var_info, "config": cfg, "is_sub": True})

        # ── 信号过滤 ──
        if signal_filter:
            filter_set = set(signal_filter)
            before = len(all_signals)
            all_signals = [s for s in all_signals if s["info"]["key"] in filter_set]
            logger.info(f"[Phase1] 信号过滤: {before} → {len(all_signals)} (指定{len(filter_set)}个)")
            if not all_signals:
                return {"error": f"信号过滤后为空,有效key: {list(signal_filter)}", "signals": []}

        total_signals = len(all_signals)
        total_dates = len(eval_dates)
        logger.info(f"[Phase1] 内存优化模式: {total_dates} 日期 × {total_signals} 信号, "
                    f"启动内存 {_mem():.1f}GB")

        # 预取 Qlib 最佳模型（仅当需要测qlib信号时）
        best_qlib = None
        need_qlib = (not signal_filter) or ("qlib" in signal_filter)
        if need_qlib:
            try:
                best_qlib = screener._find_best_qlib_model()
                if best_qlib:
                    logger.info(f"[Phase1] Qlib 最佳模型: {best_qlib}")
                    if progress:
                        progress.update({"stage": "qlib", "current": 0, "total": 0,
                                       "message": f"批量加载 Qlib 预测 ({pool} 池, 预计 1-10 分钟)..."})
                    logger.info(f"[Phase1] 批量加载 {len(eval_dates)} 个日期的 Qlib 预测...")
                    t_qlib = time.time()
                    bulk_preds = screener._load_qlib_predictions_bulk(best_qlib, eval_dates, pool=pool)
                    if bulk_preds:
                        screener.set_qlib_predictions_by_date(bulk_preds)
                        logger.info(f"[Phase1] Qlib 批量加载完成: {len(bulk_preds)} 日期 ({time.time()-t_qlib:.0f}s)")
                    else:
                        logger.warning("[Phase1] Qlib 批量预测为空")
            except Exception as e:
                logger.warning(f"[Phase1] Qlib 模型查找跳过: {e}")

        # ── 财务预取(可选,默认跳过,省 500MB-1GB)──
        if not skip_financial_prefetch:
            if progress:
                progress.update({"stage": "pre_fin", "current": 0, "total": 0,
                               "message": "预取财务数据..."})
            try:
                first_snap = screener.precompute_snapshot(
                    pool=pool, code_prefixes=code_prefixes, skip_tech=False,
                    eval_date=eval_dates[0], fast=True)
                all_codes = sorted(first_snap.keys())
                del first_snap
                from core.domain.screening._screener import _prefetch_financials
                fin_cb = (lambda cur, tot: progress.update({"stage": "pre_fin", "current": cur, "total": tot,
                              "message": f"财务数据 ({cur}/{tot})"})) if progress else None
                fin_cache = _prefetch_financials(all_codes, progress_cb=fin_cb)
                screener.set_financial_cache(fin_cache)
                logger.info(f"[Phase1] 财务预取完成: {len(all_codes)} 只, 内存 {_mem():.1f}GB")
            except Exception as e:
                logger.warning(f"[Phase1] 财务预取跳过: {e}")

        # ── 按日期检测市场状态（用于分组统计）──
        date_regimes = {}
        try:
            from core.domain.regime.classifier import MarketRegimeClassifier
            from core.data.fetcher import DataFetcher
            from core.domain.screening.indicators import Analyzer
            from core.data import Database
            # 优先从 MySQL 拉沪深300日线 (比 HTTP 可靠且快)
            index_df = None
            if Database.is_available():
                rows = Database.fetchall(
                    "SELECT trade_date, open, high, low, close, volume, amount "
                    "FROM daily_quotes WHERE code='sh000300' ORDER BY trade_date"
                )
                if rows:
                    index_df = pd.DataFrame(rows)
                    logger.info(f"[Phase1] 从 MySQL 加载沪深300: {len(index_df)} 天")
            if index_df is None or index_df.empty:
                index_df = DataFetcher().get_index_daily("沪深300", days=365 * 2)
            if index_df is not None and not index_df.empty:
                if "trade_date" not in index_df.columns and "date" in index_df.columns:
                    index_df.rename(columns={"date": "trade_date"}, inplace=True)
                index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
                index_df = Analyzer.add_indicators(index_df)
                index_df = index_df.sort_values("trade_date").reset_index(drop=True)
                for d in eval_dates:
                    window = index_df[index_df["trade_date"] <= pd.Timestamp(d)]
                    if len(window) >= 60:
                        rr = MarketRegimeClassifier().classify_from_df(window)
                        date_regimes[d.isoformat()] = rr.regime.value
                regime_counts = {}
                for rv in date_regimes.values():
                    regime_counts[rv] = regime_counts.get(rv, 0) + 1
                logger.info(f"[Phase1] 市场状态分布: {regime_counts}")
            else:
                logger.warning("[Phase1] 无法获取沪深300指数数据,跳过分组")
        except Exception as e:
            logger.warning(f"[Phase1] 市场状态检测失败,跳过分组: {e}")

        # ── 按日期顺序处理,每次只保留一个 snapshot ──
        signal_date_results = {s["info"]["key"]: {"info": s["info"], "is_sub": s["is_sub"], "date_results": []}
                               for s in all_signals}

        for date_idx, eval_date in enumerate(eval_dates):
            date_str = eval_date.isoformat()
            t0 = time.time()
            # 当前日期的 forward return 缓存，避免多信号重复 SQL
            fwd_cache = {}
            fwd_lock = threading.Lock()
            logger.info(f"[Phase1] === 日期 {date_idx+1}/{total_dates}: {date_str} (内存 {_mem():.1f}GB) ===")
            if progress:
                progress.update({"stage": "phase1", "current": date_idx + 1, "total": total_dates,
                               "message": f"快照 {date_idx+1}/{total_dates}: {date_str}"})

            # 优先读缓存（预热或之前跑过的），miss 了再计算
            try:
                snap = _load_cached_snapshot(eval_date, pool)
                if snap is not None:
                    n_stocks = len(snap)
                    logger.info(f"[Phase1] {date_str} 命中缓存: {n_stocks} 只")
                else:
                    snap = screener.precompute_snapshot(
                        pool=pool, code_prefixes=code_prefixes, skip_tech=False,
                        eval_date=eval_date, fast=True)
                    n_stocks = len(snap)
                    if snap:
                        _save_cached_snapshot(eval_date, pool, snap)
                logger.info(f"[Phase1] {date_str} snapshot: {n_stocks} 只, "
                           f"内存 {_mem():.1f}GB ({time.time()-t0:.1f}s)")
            except Exception as e:
                logger.error(f"[Phase1] {date_str} snapshot 失败: {e}")
                continue

            # Qlib 预测已在循环前批量加载，无需逐日期加载

            # 并行测试所有信号(仅当前日期,共享同一个 snapshot)
            lock = threading.Lock()
            date_done = [0]

            def _test_one_on_date(entry):
                info = entry["info"]
                cfg = entry["config"]
                t1 = time.time()
                try:
                    r = screener.screen(
                        pool=pool, top_n=actual_top_n, skip_l1=True, skip_l2=True, skip_debate=True,
                        skip_qlib=cfg.get("skip_qlib", True),
                        skip_sent=cfg.get("skip_sent", True),
                        skip_fundamental=cfg.get("skip_fundamental", True),
                        skip_flow=cfg.get("skip_flow", True),
                        skip_north_flow=cfg.get("skip_north_flow", True),
                        skip_size_turnover=cfg.get("skip_size_turnover", True),
                        skip_tech=cfg.get("skip_tech", False),
                        skip_momentum=cfg.get("skip_momentum", False),
                        skip_volatility=cfg.get("skip_volatility", False),
                        skip_ht=cfg.get("skip_ht", True),
                        ht_key=info.get("ht_key"),
                        skip_wb_industry=cfg.get("skip_wb_industry", True),
                        skip_wb_divergence=cfg.get("skip_wb_divergence", True),
                        skip_wb_moneyflow=cfg.get("skip_wb_moneyflow", True),
                        wb_industry_n=cfg.get("wb_industry_n", 20),
                        wb_divergence_n=cfg.get("wb_divergence_n", 10),
                        wb_moneyflow_n=cfg.get("wb_moneyflow_n", 5),
                        code_prefixes=code_prefixes,
                        source_weights=cfg.get("source_weights"),
                        tech_preset=cfg.get("tech_preset", "all"),
                        eval_date=eval_date,
                        job_id=f"ph1_{info['key']}_{date_str}",
                        snapshot=snap,
                        regime=regime,
                    )
                    candidates = r.get("candidates", [])
                    top_codes = [c["code"] for c in candidates[:top_k]]
                    top_scores = [c.get("l1_score", c.get("l0_score", 0)) for c in candidates[:top_k]]
                    forward = self._cached_forward_returns(top_codes, eval_date, fwd_cache, fwd_lock)
                    n_cand = len(candidates)
                    logger.debug(f"[Phase1] {info['key']} @ {date_str}: {n_cand} 候选 ({time.time()-t1:.1f}s)")
                    return {
                        "eval_date": date_str,
                        "top_codes": top_codes,
                        "top_scores": top_scores,
                        "forward_5d": forward.get("5d"),
                        "forward_10d": forward.get("10d"),
                        "forward_20d": forward.get("20d"),
                        "per_code_returns": forward.get("per_code", {}),
                        "neutral_ic": self._neutral_ic_for(candidates, eval_date, fwd_cache, fwd_lock),
                        "candidate_count": n_cand,
                    }
                except Exception as e:
                    import traceback
                    logger.error(f"[Phase1] {info['key']} @ {date_str} 失败: {e}")
                    logger.error(f"[Phase1] Traceback:\n{traceback.format_exc()}")
                    return {"eval_date": date_str, "status": "error", "error": str(e)}
                finally:
                    with lock:
                        date_done[0] += 1
                        if progress:
                            progress["message"] = (
                                f"日期 {date_idx+1}/{total_dates} {date_str} "
                                f"信号 {date_done[0]}/{total_signals}"
                            )

            signal_workers = min(12, max(4, total_signals // 2))
            with thread_pool.default.borrow() as ex:
                futs = {ex.submit(_test_one_on_date, s): s for s in all_signals}
                for future in as_completed(futs):
                    entry = futs[future]
                    key = entry["info"]["key"]
                    try:
                        r = future.result()
                        if r:
                            signal_date_results[key]["date_results"].append(r)
                    except Exception as e:
                        logger.error(f"[Phase1] {key} @ {date_str} 任务异常: {e}")

            # 释放 snapshot（Qlib 预测批量加载后跨日期复用，不释放）
            del snap
            logger.info(f"[Phase1] {date_str} 完成, 内存 {_mem():.1f}GB ({time.time()-t0:.1f}s)")

        # ── 聚合每个信号的跨日期结果 ──
        results = []
        for key, sr in signal_date_results.items():
            info = sr["info"]
            date_results = sr["date_results"]
            date_results.sort(key=lambda x: x.get("eval_date", ""))
            agg = self._aggregate_validation(date_results)
            metrics = {
                "rank_ic": agg.get("rank_ic", 0) or 0,
                "mean_5d": agg.get("mean_5d", 0) or 0,
                "mean_10d": agg.get("mean_10d", 0) or 0,
                "mean_20d": agg.get("mean_20d", 0) or 0,
                "sharpe_5d": agg.get("sharpe_5d", 0) or 0,
                "sharpe_10d": agg.get("sharpe_10d", 0) or 0,
                "sharpe_20d": agg.get("sharpe_20d", 0) or 0,
                "hit_rate_5d": agg.get("hit_rate_5d", 0) or 0,
                "hit_rate_10d": agg.get("hit_rate_10d", 0) or 0,
                "hit_rate_20d": agg.get("hit_rate_20d", 0) or 0,
                "n_dates": agg.get("n_dates", 0),
            }
            # 单信号 Rank IC 无区分度(纯单源 top-K 分数雷同), 改用均值+胜率判据
            # 高收益豁免: 20日收益 >= 8% 且胜率 >= 70% 的信号即使其他指标差也保留
            passes = (metrics["hit_rate_20d"] >= th["hit_rate_20d_min"] and
                      metrics["mean_20d"] >= th["mean_20d_min"])
            exempt = (metrics["mean_20d"] >= 0.08 and
                      metrics["hit_rate_20d"] >= 0.70)
            if agg.get("error"):
                status = "无数据"
            elif metrics["n_dates"] == 0:
                status = "无数据"
            else:
                status = "有效" if (passes or exempt) else "失效"
            logger.info(f"[Phase1] {info['label']} -> {status} "
                       f"(rank_ic={metrics['rank_ic']:.4f}, hit_20d={metrics['hit_rate_20d']:.3f}, "
                       f"mean_20d={metrics['mean_20d']:.4f})")
            results.append({
                "key": info["key"], "label": info["label"],
                "data_safety": info["data_safety"], "status": status,
                "metrics": metrics, "is_sub": sr["is_sub"],
            })

        valid_keys = [r["key"] for r in results if r["status"] == "有效"]
        invalid_keys = [r["key"] for r in results if r["status"] == "失效"]
        no_data_keys = [r["key"] for r in results if r["status"] == "无数据"]

        # ── 市态分组统计 ──
        regime_breakdown = None
        if date_regimes:
            regime_breakdown = self._build_regime_breakdown(
                signal_date_results, date_regimes, th)

        # Phase 2 需自行计算快照,不再缓存全量
        self._phase1_snapshot_cache = None

        logger.info(f"[Phase1] 完成: 有效={len(valid_keys)} 失效={len(invalid_keys)} "
                    f"无数据={len(no_data_keys)}, 内存 {_mem():.1f}GB")

        # ── 白盒因子最优参数 ──
        best_params = {}
        for r in results:
            tk = r.get("template_key")
            if tk and r["status"] == "有效":
                pv = r.get("param_value")
                if tk not in best_params or (r["metrics"]["mean_20d"] > best_params[tk]["mean_20d"]):
                    best_params[tk] = {"key": r["key"], "label": r["label"],
                                       "param_value": pv, "mean_20d": r["metrics"]["mean_20d"]}

        return {
            "eval_dates": [d.isoformat() for d in eval_dates],
            "total_dates": total_dates,
            "signals": results,
            "regime_breakdown": regime_breakdown,
            "best_params": best_params,
            "summary": {
                "valid_count": len(valid_keys),
                "invalid_count": len(invalid_keys),
                "no_data_count": len(no_data_keys),
                "valid_keys": valid_keys,
                "invalid_keys": invalid_keys,
            },
        }

    def rolling_signal_screen(self, start: str, end: str, step_months=3, lookback_months=24,
                              pool="all", top_k=10, code_prefixes=None, skip_tech_subs=False,
                              date_step=5, write_pool=True, progress=None) -> Dict:
        """时序滚动信号筛选 (报告修正1): 每 step_months 用 [anchor-lookback, anchor] 的
        历史数据跑 Phase1 signal_validity_test, 选出当期有效信号, 写回动态因子池 (修正6闭环)。
        底层 _get_eval_dates(date_start/date_end) + forward return 按 eval_date 对齐, 严格样本外无前视。
        """
        from core.domain.factors.rolling import _gen_anchors
        s = datetime.date.fromisoformat(start)
        e = datetime.date.fromisoformat(end)
        anchors = _gen_anchors(s, e, step_months)

        per_anchor = {}
        timeline = {}
        final_active = []
        for i, anchor in enumerate(anchors):
            a_end = anchor.isoformat()
            a_start = (anchor - datetime.timedelta(days=int(lookback_months * 30.5))).isoformat()
            if progress is not None:
                progress.update({"stage": "rolling_signal", "anchor": a_end,
                                 "current": i + 1, "total": len(anchors)})
            res = self.signal_validity_test(
                pool=pool, date_start=a_start, date_end=a_end, top_k=top_k,
                code_prefixes=code_prefixes, skip_tech_subs=skip_tech_subs,
                date_step=date_step, progress=progress,
            )
            signals = res.get("signals", [])
            valid = [r["key"] for r in signals if r.get("status") == "有效"]
            per_anchor[a_end] = {"valid": valid, "summary": res.get("summary", {})}
            for k in valid:
                timeline.setdefault(k, []).append(a_end)
            if write_pool:
                try:
                    from core.domain.factors.registry import get_pool
                    # 白盒子指标写回动态池; 传统大类由 batch_add_from_phase1 兼容处理
                    get_pool().batch_add_from_phase1(signals)
                    final_active = valid
                except Exception as ex:
                    logger.warning(f"[RollingSignal] 写回池失败 {a_end}: {ex}")
            logger.info(f"[RollingSignal] anchor={a_end} 有效信号 {len(valid)}")

        return {
            "anchors": [a.isoformat() for a in anchors],
            "per_anchor": per_anchor,
            "signal_timeline": timeline,
            "final_active": final_active,
        }

    # ══════════════════════════════════════════════
    # 离散信号事件研究 (Phase 1 补充)
    # ══════════════════════════════════════════════

    def generate_regime_weights_from_breakdown(self, regime_breakdown: Dict) -> Dict:
        """根据 Phase 1 市态分组交叉表，自动生成各市态的权重配置。
        规则: 20日收益>0的信号按 (收益×胜率) 分配权重, 收益≤0的归零。
        同时根据技术子指标在各市态的表现推荐 tech_preset。"""
        if not regime_breakdown:
            return {"error": "无交叉表数据，请先跑 Phase 1"}

        SIGNAL_KEY_MAP = {
            "tech": "tech", "qlib": "qlib", "sentiment": "sentiment",
            "fundamental": "fundamental", "flow": "flow", "north_flow": "north_flow",
            "size_turnover": "size_turnover", "momentum": "momentum", "volatility": "volatility",
        }
        TECH_SUB_TO_PRESET = {
            "tech_macd": ("macd_only", "MACD"),
            "tech_rsi": ("rsi_only", "RSI"),
            "tech_ma": ("ma_only", "MA"),
            "tech_kdj": ("kdj_only", "KDJ"),
            "tech_bb": ("bb_only", "BB"),
        }

        regime_order = regime_breakdown.get("regime_order", [])
        signals_data = regime_breakdown.get("signals", [])

        generated = {}
        for rv in regime_order:
            scores = {}
            tech_sub_scores = {}  # 技术子指标各市态表现

            for sig in signals_data:
                key = sig["key"]
                rm = sig.get("regime_metrics", {}).get(rv)
                if not rm or rm["n_dates"] < 3:
                    continue

                mean = rm["mean_20d"]
                hit = rm["hit_rate_20d"]

                if key in TECH_SUB_TO_PRESET and mean > 0:
                    tech_sub_scores[key] = mean * max(hit, 0.4)

                if key in SIGNAL_KEY_MAP and mean > 0.001:
                    scores[SIGNAL_KEY_MAP[key]] = max(0.01, mean * max(hit, 0.4))
                elif key in SIGNAL_KEY_MAP:
                    scores[SIGNAL_KEY_MAP[key]] = 0.0  # 无效信号权重归零

            # 确保 9 个 key 都存在
            for k in SIGNAL_KEY_MAP.values():
                if k not in scores:
                    scores[k] = 0.0

            total = sum(v for v in scores.values() if v > 0)
            if total <= 0:
                # 全部信号都无效，回退均衡权重
                n = len(scores)
                weights = {k: round(1.0 / n, 3) for k in scores}
            else:
                weights = {k: round(scores[k] / total, 3) if scores[k] > 0 else 0.0
                          for k in scores}

            # 推荐 tech_preset
            best_sub = max(tech_sub_scores, key=tech_sub_scores.get, default=None)
            tech_preset = TECH_SUB_TO_PRESET.get(best_sub, (None, None))[0] if best_sub else "all"

            generated[rv] = {
                "weights": weights,
                "tech_preset": tech_preset or "all",
                "raw_scores": {k: round(v, 4) for k, v in scores.items()},
            }

        # ── 写入 regime_weights.py ──
        import os as _os
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(root, "core", "regime_weights.py")
        self._write_regime_weights_file(path, generated, regime_breakdown)

        return {"generated": generated, "saved_to": path}

    def _write_regime_weights_file(self, path, generated, breakdown):
        """生成 regime_weights.py 文件内容并写入磁盘"""
        REGIME_LABELS = {
            "trending_up": "趋势上涨", "trending_down": "趋势下跌",
            "ranging": "震荡盘整", "volatile": "高波动",
        }

        lines = ['"""', '动态权重映射表 — 自动生成 (Phase 1 数据驱动)', '"""',
                 'from core.domain.regime.classifier import Regime', '', '']

        # REGIME_WEIGHTS
        lines.append('REGIME_WEIGHTS = {')
        for rv in ["trending_up", "trending_down", "ranging", "volatile"]:
            cfg = generated.get(rv, {})
            w = cfg.get("weights", {})
            lines.append(f'    Regime.{rv.upper()}: {{')
            for k in ["tech", "qlib", "sentiment", "fundamental", "flow",
                       "north_flow", "size_turnover", "momentum", "volatility"]:
                lines.append(f'        "{k}": {w.get(k, 0.0)},')
            lines.append('    },')
        lines.append('}')
        lines.append('')

        # REGIME_TECH_PRESETS
        lines.append('REGIME_TECH_PRESETS = {')
        for rv in ["trending_up", "trending_down", "ranging", "volatile"]:
            cfg = generated.get(rv, {})
            tp = cfg.get("tech_preset", "all")
            lines.append(f'    Regime.{rv.upper()}: "{tp}",')
        lines.append('}')
        lines.append('')

        # REGIME_SKIPS
        lines.append('REGIME_SKIPS = {')
        for rv in ["trending_up", "trending_down", "ranging", "volatile"]:
            lines.append(f'    Regime.{rv.upper()}: {{}},')
        lines.append('}')
        lines.append('')

        # REGIME_LABELS
        lines.append('REGIME_LABELS = {')
        for rv in ["trending_up", "trending_down", "ranging", "volatile"]:
            lines.append(f'    Regime.{rv.upper()}: "{REGIME_LABELS.get(rv, rv)}",')
        lines.append('}')
        lines.append('')

        # REGIME_RATIONALE
        lines.append('REGIME_RATIONALE = {')
        for rv in ["trending_up", "trending_down", "ranging", "volatile"]:
            cfg = generated.get(rv, {})
            w = cfg.get("weights", {})
            active = [(k, v) for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True) if v > 0.05]
            parts = [f"{k}={v:.0%}" for k, v in active[:4]]
            rationale = f"自动生成: {' '.join(parts)}" if parts else "无有效信号,均衡分配"
            lines.append(f'    Regime.{rv.upper()}: "{rationale}",')
        lines.append('}')
        lines.append('')

        # helper functions
        lines.append('''
def get_regime_weights(regime, sector_bias=None):
    if isinstance(regime, str):
        regime = Regime(regime)
    base = dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.STEADY_FLAT]))
    if sector_bias == "偏多" and regime != Regime.VOLATILE_UP:
        base["north_flow"] = base.get("north_flow", 0.10) + 0.03
        base["momentum"] = base.get("momentum", 0.10) + 0.02
    elif sector_bias == "偏空" and regime != Regime.VOLATILE_DOWN:
        base["volatility"] = base.get("volatility", 0.10) + 0.03
        base["fundamental"] = base.get("fundamental", 0.10) + 0.02
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
''')

        from core.shared.persistence import save_text
        save_text("\n".join(lines), path)
        logger.info(f"[Regime] 权重已保存: {path}")

    def discrete_signal_test(self, tech_preset="macd_only",
                             pool="all", eval_weeks=52,
                             date_start=None, date_end=None,
                             date_step=5, code_prefixes=None,
                             progress=None) -> Dict:
        """事件研究法：衡量离散信号（金叉/超卖等）触发后的真实收益。

        与 Phase 1 不同，不要求信号有连续排序能力。
        只统计：信号触发后 N 日的胜率、均值收益、超额收益。

        tech_preset: "macd_only" / "rsi_only" / "kdj_only" / "bb_only" / "volume_only" / "ma_only"
        """
        from core.domain.signals.engine import SignalFusion

        preset_map = {
            "macd_only": ["MACD"], "rsi_only": ["RSI"],
            "kdj_only": ["KDJ"], "bb_only": ["BB"],
            "volume_only": ["VOLUME"], "ma_only": ["MA"],
        }
        indicators = preset_map.get(tech_preset, ["MACD"])
        signal_name = {"MACD": "MACD金叉/死叉", "RSI": "RSI超买/超卖",
                       "KDJ": "KDJ金叉/死叉", "BB": "布林带触轨",
                       "VOLUME": "放量异动", "MA": "MA均线排列"}.get(indicators[0], tech_preset)

        eval_dates = self._get_eval_dates(weeks=eval_weeks, date_start=date_start,
                                          date_end=date_end, step=date_step)
        if not eval_dates:
            return {"error": "无交易日数据"}

        screener = get_screener()

        # 单指标权重
        weights = {k: 1.0 if k in indicators else 0.0 for k in
                   ["MACD", "RSI", "MA", "KDJ", "BB", "VOLUME"]}

        # 收集所有触发事件的 forward return
        buy_returns = {"5d": [], "10d": [], "20d": []}
        sell_returns = {"5d": [], "10d": [], "20d": []}
        buy_dates = 0
        sell_dates = 0
        total_dates = 0
        trigger_counts = []

        for eval_date in eval_dates:
            snap = _load_cached_snapshot(eval_date, pool)
            if snap is None:
                snap = screener.precompute_snapshot(
                    pool=pool, code_prefixes=code_prefixes, skip_tech=False,
                    eval_date=eval_date, fast=True)
                if snap:
                    _save_cached_snapshot(eval_date, pool, snap)
            if not snap:
                continue
            total_dates += 1
            date_buy = []
            date_sell = []
            date_buy_codes = []
            date_sell_codes = []

            for code, (df, name, close, volume) in snap.items():
                if len(df) < 2:
                    continue
                try:
                    fr = SignalFusion(weights=weights).evaluate(df)
                except Exception:
                    continue

                vote = fr.indicator_votes.get(indicators[0], 0)
                if vote == 0:
                    continue

                if vote > 0:
                    date_buy_codes.append(code)
                else:
                    date_sell_codes.append(code)

            # 批量计算所有触发信号的股票的 forward return
            all_triggered = date_buy_codes + date_sell_codes
            fwd_all = self._compute_forward_returns(all_triggered, eval_date) if all_triggered else {}
            per_code = fwd_all.get("per_code", {})

            for code in date_buy_codes:
                cr = per_code.get(code, {})
                date_buy.append((cr.get("5d"), cr.get("10d"), cr.get("20d")))
            for code in date_sell_codes:
                cr = per_code.get(code, {})
                date_sell.append((cr.get("5d"), cr.get("10d"), cr.get("20d")))

            if date_buy:
                buy_dates += 1
                for i, period in enumerate(["5d", "10d", "20d"]):
                    vals = [r[i] for r in date_buy if r[i] is not None]
                    buy_returns[period].extend(vals)
                trigger_counts.append(len(date_buy))
            if date_sell:
                sell_dates += 1
                for i, period in enumerate(["5d", "10d", "20d"]):
                    vals = [r[i] for r in date_sell if r[i] is not None]
                    sell_returns[period].extend(vals)
                trigger_counts.append(len(date_sell))
            del snap  # 每日期释放快照，控制内存

        def _agg(returns_list, period_name=""):
            if not returns_list:
                return {"count": 0, "mean": 0, "win_rate": 0, "t_stat": 0, "median": 0}
            import numpy as np
            arr = np.array(returns_list)
            n = len(arr)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n > 1 else 1.0
            t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0
            win_rate = float((arr > 0).mean())
            median = float(np.median(arr))
            return {"count": n, "mean": mean, "win_rate": win_rate,
                    "t_stat": t_stat, "median": median,
                    "significant": abs(t_stat) > 1.96}

        result = {
            "signal": signal_name,
            "tech_preset": tech_preset,
            "eval_dates": len(eval_dates),
            "active_dates": total_dates,
            "buy": {
                "dates_with_signal": buy_dates,
                **{f"{p}d": _agg(buy_returns[f"{p}d"], p) for p in ["5", "10", "20"]},
            },
            "sell": {
                "dates_with_signal": sell_dates,
                **{f"{p}d": _agg(sell_returns[f"{p}d"], p) for p in ["5", "10", "20"]},
            },
            "avg_triggers_per_date": round(sum(trigger_counts) / max(len(trigger_counts), 1), 1),
        }

        # 简易结论
        b20 = result["buy"]["20d"]
        s20 = result["sell"]["20d"]
        if b20["count"] >= 30 and b20["mean"] > 0 and b20["t_stat"] > 1.96:
            result["verdict"] = f"有效：买入信号20日均收益 {b20['mean']*100:.2f}%，胜率 {b20['win_rate']*100:.0f}%，统计显著"
        elif b20["count"] < 10:
            result["verdict"] = f"信号过于稀疏：{buy_dates}个活跃日，日均仅 {result.get('avg_triggers_per_date', 0):.0f} 次触发"
        elif b20["mean"] <= 0:
            result["verdict"] = f"失效：买入信号20日均收益 {b20['mean']*100:.2f}% ≤ 0，无正向预测能力"
        else:
            result["verdict"] = f"不确定：样本{b20['count']}个，收益 {b20['mean']*100:.2f}%，但统计不显著 (t={b20['t_stat']:.2f})"

        return result

    # ══════════════════════════════════════════════
    # Phase 2: 权重优化(仅优化活跃信号)
    # ══════════════════════════════════════════════

    def _nelder_mead_optimize(self, active_list, base_skips, all_keys,
                              pool, opt_weeks, top_k, code_prefixes, progress, regime,
                              snapshot_cache=None):
        """Nelder-Mead 单纯形法搜索最优权重。log-ratio 变换处理 simplex 约束。
        单次评估 < 2s，30-50 次迭代收敛。"""
        import numpy as np
        from scipy.optimize import minimize

        n = len(active_list)
        if n < 2:
            return None

        # 构建评估函数: x (R^(n-1)) → fitness (越大越好)
        def _make_config_from_simplex(w_dict):
            return {
                "source_weights": {k: w_dict.get(k, 0.0) for k in all_keys},
                "tech_preset": "all",
                **base_skips,
            }

        def _simplex_to_weights(x):
            """log-ratio → simplex weights"""
            w = np.exp(np.append(x, 0))  # 最后一维固定为 0 (exp(0)=1)
            w = w / w.sum()
            return {active_list[i]: float(w[i]) for i in range(n)}

        # ── 预计算快照缓存(所有 N-M 评估共享, 避免重复计算)──
        import time as _time
        t0 = _time.time()
        if snapshot_cache is None:
            logger.info(f"[N-M] 兼容路径预计算 {opt_weeks} 周快照缓存...")
            eval_dates = self._get_eval_dates(weeks=opt_weeks, step=5)
            snap_cache = self._precompute_snapshots_parallel(
                eval_dates, pool, code_prefixes, label="N-M优化")
        else:
            snap_cache = snapshot_cache
            logger.info(
                f"[N-M] 使用L2注入的共享面板: {len(snap_cache)} 日期, "
                f"mode={getattr(snap_cache, 'backing_mode', 'injected')}"
            )
        logger.info(f"[N-M] 快照缓存就绪: {len(snap_cache)} 日期 ({_time.time()-t0:.1f}s)")

        # ── 用缓存快照 + skip_l1 加速评估 ──
        def _objective(x):
            w = _simplex_to_weights(x)
            cfg = _make_config_from_simplex(w)
            label = _config_label(cfg)
            try:
                r = self.historical_validate([cfg], eval_weeks=opt_weeks,
                                            top_k=top_k, pool=pool,
                                            code_prefixes=code_prefixes,
                                            skip_l1=True, snapshot_cache=snap_cache,
                                            progress=None, regime=regime)
                results = r.get("results", [])
                if results and results[0].get("fitness", -999) > -900:
                    f = results[0]["fitness"]
                    logger.info(f"[N-M] {label} fitness={f:.4f} ({_time.time()-t0:.1f}s)")
                    return -f
            except Exception as e:
                logger.debug(f"[N-M] eval 失败: {e}")
            return 999.0

        # 初始点: 等权重
        x0 = np.zeros(n - 1)
        logger.info(f"[N-M] 初始: {_config_label(_make_config_from_simplex(_simplex_to_weights(x0)))}")

        # 运行 Nelder-Mead (带进度, 评估数动态显示)
        nfev_counter = [0]
        def _objective_with_progress(x):
            nfev_counter[0] += 1
            if progress:
                progress.update({"stage": "optimize", "current": nfev_counter[0], "total": 40,
                               "message": f"Nelder-Mead 评估 {nfev_counter[0]} 次(~{nfev_counter[0]*3//60}分)..."})
            return _objective(x)

        try:
            res = minimize(_objective_with_progress, x0, method="Nelder-Mead",
                         options={"maxiter": 20, "xatol": 0.08, "fatol": 0.01})
            best_w = _simplex_to_weights(res.x)
            best_cfg = _make_config_from_simplex(best_w)
            label = _config_label(best_cfg)
            logger.info(f"[N-M] 最优: {label} fitness={-res.fun:.4f} (eval={res.nfev})")
            return best_cfg
        except Exception as e:
            logger.warning(f"[N-M] 优化失败: {e}")
            return None

    def phase2_weight_optimization(self, active_signals: List[str],
                                   pool="all", top_n=30, eval_weeks=12,
                                   date_start=None, date_end=None,
                                   top_k=10, code_prefixes=None,
                                   progress=None, regime=None,
                                   nm_snapshot_cache=None) -> Dict:
        """Phase 2: 基于有效信号源做权重精调 + 历史验证"""
        if not active_signals:
            return {"error": "active_signals 为空"}

        active_set = set(active_signals)
        all_keys = [s["key"] for s in SIGNAL_SOURCES]

        # 非活跃信号全部关闭
        base_skips = {}
        for src in SIGNAL_SOURCES:
            base_skips[src["skip_key"]] = src["key"] not in active_set

        # ── 智能权重搜索：预设 + 边界 + 稀疏随机 ──
        import numpy as np

        n_active = len(active_set)
        active_list = sorted(active_set)

        # 1. 预设模板（覆盖典型偏好）
        preset_weights = []
        for pname, pw in WEIGHT_PRESETS.items():
            filtered = {}
            for k in active_list:
                if pw.get(k, 0) > 0:
                    filtered[k] = pw[k]
            if filtered:
                total = sum(filtered.values())
                preset_weights.append({k: v / total for k, v in filtered.items()})

        # 2. 边界权重：单信号100%、双信号50/50（快速探测极值）
        edge_weights = []
        if n_active >= 1:
            for k in active_list:
                edge_weights.append({k: 1.0})
        if n_active >= 2:
            for i in range(n_active):
                for j in range(i + 1, min(i + 4, n_active)):
                    edge_weights.append({active_list[i]: 0.5, active_list[j]: 0.5})

        # 3. 稀疏随机采样（比之前减少 70%，两阶段验证会补偿精度）
        n_random = 10 if n_active <= 3 else (20 if n_active <= 5 else 40)
        random_weights = []
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            alpha = np.ones(n_active)
            w = rng.dirichlet(alpha)
            w = np.round(w * 100) / 100
            w = w / w.sum()
            random_weights.append({active_list[i]: float(w[i]) for i in range(n_active)})

        # 去重
        weight_configs = []
        seen = set()
        for w in preset_weights + edge_weights + random_weights:
            t = tuple(sorted((k, round(v, 4)) for k, v in w.items()))
            if t not in seen:
                seen.add(t)
                weight_configs.append(w)

        # 单信号无需搜索
        if n_active == 1:
            weight_configs = [{active_list[0]: 1.0}]

        # ── 纯优化路径: 单配置快速验证, Nelder-Mead 自动收敛 ──
        if n_active >= 2 and eval_weeks >= 8:
            logger.info(f"[Phase2] Nelder-Mead 优化 ({n_active} 信号, {eval_weeks}周)")
            if progress:
                progress.update({"stage": "optimize", "current": 0, "total": 0,
                               "message": f"Nelder-Mead 搜索最优权重..."})

            # 在短周期上做目标函数评估(每次 ~5s)，全周期上做最终验证
            opt_weeks = max(6, eval_weeks // 8)  # 8周→1周, 52周→6周
            opt_top_k = min(top_k, 10)

            best = self._nelder_mead_optimize(
                active_list, base_skips, all_keys,
                pool=pool, opt_weeks=opt_weeks, top_k=opt_top_k,
                code_prefixes=code_prefixes, progress=progress, regime=regime,
                snapshot_cache=nm_snapshot_cache)

            if best:
                logger.info(f"[Phase2] Nelder-Mead 完成, 最优: {_config_label(best)}")
                sweep_result = self.quick_sweep([best], pool=pool, top_n=max(15, top_n // 2),
                                                code_prefixes=code_prefixes, progress=progress,
                                                regime=regime)
                top_configs = [best]
                valid = [r for r in sweep_result.get("results", []) if r.get("status") != "error"]
                to_validate = [r for r in valid if r.get("candidate_count", 0) > 0]
            else:
                logger.warning("[Phase2] Nelder-Mead 失败, 回退随机搜索")
                sweep_result = {"total": 0, "completed": 0, "results": []}
                top_configs = []
        else:
            sweep_result = {"total": 0, "completed": 0, "results": []}
            top_configs = []

        if not top_configs:
            configs = weight_configs if n_active == 1 else (
                preset_weights + edge_weights + random_weights)[:30]
            configs = [{
                "source_weights": {k: w.get(k, 0.0) for k in all_keys},
                "tech_preset": "all",
                **base_skips,
            } for w in configs]

            sweep_result = self.quick_sweep(configs, pool=pool, top_n=max(15, top_n // 2),
                                            code_prefixes=code_prefixes, progress=progress,
                                            regime=regime)
            valid = [r for r in sweep_result.get("results", []) if r.get("status") != "error"]
            to_validate = [r for r in valid if r.get("candidate_count", 0) > 0]
            top_configs = [r["config"] for r in to_validate[:20]] if to_validate else []
            if progress:
                progress.update({"stage": "validate", "current": 0, "total": 0,
                               "message": f"随机搜索 {len(top_configs)} 组进验证"})

        if not top_configs:
            return {"error": "无有效配置", "sweep": sweep_result}

        validate_result = self.historical_validate(top_configs, eval_weeks=eval_weeks,
                                                    date_start=date_start, date_end=date_end,
                                                    top_k=top_k, pool=pool,
                                                    code_prefixes=code_prefixes,
                                                    snapshot_cache=None,
                                                    progress=progress,
                                                    regime=regime)

        n_total_cfgs = len(weight_configs)
        result = {
            "sweep": sweep_result,
            "n_validated": len(top_configs),
            "top3_selected": [r["label"] for r in (to_validate[:3] if to_validate else [])],
            "top3_scores": [r["score_mean"] for r in (to_validate[:3] if to_validate else [])],
            "validate": validate_result,
            "active_signals": active_signals,
        }

        try:
            report_path = self._generate_phase2_report(result)
            result["report_path"] = report_path
        except Exception as e:
            logger.warning(f"Phase2 报告生成失败: {e}")

        return result

    def _generate_phase2_report(self, result: Dict) -> str:
        """生成 Phase 2 调优报告 — 详尽版"""
        import os
        now_strftime_val = now_strftime('%Y%m%d_%H%M%S')
        filename = f"tuner_phase2_report_{now_strftime_val}.md"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, filename)

        KEY_LABELS = {
            "tech": "技术指标", "qlib": "Qlib模型", "sentiment": "情绪分析",
            "fundamental": "基本面", "flow": "资金流向",
            "north_flow": "北向资金", "size_turnover": "小市值换手",
            "momentum": "动量因子", "volatility": "波动率",
            "illiq": "Amihud非流动性", "overnight_gap": "隔夜跳空",
        }
        SKIP_LABELS = {
            "skip_tech": "技术指标", "skip_qlib": "Qlib模型", "skip_sent": "情绪分析",
            "skip_fundamental": "基本面", "skip_flow": "资金流向",
            "skip_north_flow": "北向资金", "skip_size_turnover": "小市值换手",
            "skip_momentum": "动量因子", "skip_volatility": "波动率",
            "skip_illiq": "Amihud非流动性", "skip_overnight_gap": "隔夜跳空",
        }
        KEY_TO_SKIP = {
            "tech": "skip_tech", "qlib": "skip_qlib", "sentiment": "skip_sent",
            "fundamental": "skip_fundamental", "flow": "skip_flow",
            "north_flow": "skip_north_flow", "size_turnover": "skip_size_turnover",
            "momentum": "skip_momentum", "volatility": "skip_volatility",
            "illiq": "skip_illiq", "overnight_gap": "skip_overnight_gap",
        }

        def _fmt_weight_table(cfg, effective_weights=None):
            parts = []
            sw = cfg.get("source_weights", {})
            ew = effective_weights or {}
            parts.append("| 信号源 | 输入权重 | 实际权重 | 开关 |")
            parts.append("|--------|----------|----------|------|")
            for k in ["tech", "qlib", "sentiment", "fundamental", "flow", "north_flow", "size_turnover", "momentum", "volatility"]:
                v_in = sw.get(k, 0)
                v_out = ew.get(k, 0)
                skipped = cfg.get(KEY_TO_SKIP[k], False)
                status = "❌" if skipped else "✅"
                in_str = f"**{v_in:.3f}**" if v_in > 0 else "0"
                out_str = f"**{v_out:.3f}**" if v_out > 0 else "0"
                parts.append(f"| {KEY_LABELS.get(k, k)} | {in_str} | {out_str} | {status} |")
            if cfg.get("tech_preset"):
                parts.append(f"| 技术预设 | `{cfg['tech_preset']}` | | |")
            return "\n".join(parts)

        def _fmt_codes_row(codes, scores):
            if not codes:
                return "-"
            pairs = [f"{c}({s:.3f})" for c, s in zip(codes[:10], scores[:10])]
            return " ".join(pairs)

        lines = []
        lines.append("# 信号权重优化报告 (Phase 2)")
        lines.append(f"**生成时间**: {now_strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # ═══════════════════════════════════════
        # 1. 参数概要
        # ═══════════════════════════════════════
        sweep = result.get("sweep", {})
        validate = result.get("validate", {})
        vresults = validate.get("results", [])
        v_edates = validate.get("eval_dates", [])

        lines.append("## 1. 参数概要")
        lines.append("")
        lines.append(f"| 参数 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 活跃信号 | {', '.join(result.get('active_signals', []))} |")
        lines.append(f"| 活跃信号数 | {len(result.get('active_signals', []))} |")
        lines.append(f"| 快扫配置总数 | {sweep.get('total', 0)} |")
        lines.append(f"| 快扫完成数 | {sweep.get('completed', 0)} |")
        lines.append(f"| 快扫失败数 | {sweep.get('total', 0) - sweep.get('completed', 0)} |")
        lines.append(f"| 验证配置数 | {len(vresults)} |")
        lines.append(f"| 验证日期数 | {validate.get('total_dates', 0)} |")
        if v_edates:
            lines.append(f"| 验证日期范围 | {v_edates[0]} ~ {v_edates[-1]} |")
        lines.append(f"| 评分方法 | L0 加权融合 (skip_l1=True) |")
        lines.append("")

        # ═══════════════════════════════════════
        # 2. 缩略词对照
        # ═══════════════════════════════════════
        lines.append("## 2. 缩略词与信号源说明")
        lines.append("")
        lines.append("| 缩略 | 信号源 | 数据来源 | 前视偏差风险 | 说明 |")
        lines.append("|------|--------|----------|-------------|------|")
        lines.append("| 技 | 技术指标 | K线价格衍生 (MACD/RSI/MA/KDJ/BB/量) | ✅ 无 | 纯价格计算，无外源数据 |")
        lines.append("| Q | Qlib模型 | LightGBM/ALSTM 预测 | ⚠️ 有 | 模型训练/预测日期可能与验证期重叠 |")
        lines.append("| 情 | 情绪分析 | NLP 新闻情感分类 | ⚠️ 有 | 新闻发布时间戳精确度有限 |")
        lines.append("| 基 | 基本面 | PE/ROE/PB 财报数据 | ⚠️ 有 | 财报发布日期 ≠ 数据生效日期 |")
        lines.append("| 资 | 资金流向 | 超大单/大单净流入 | ⚠️ 有 | 数据源时间对齐精度有限 |")
        lines.append("| 北 | 北向资金 | 沪深股通净买入 | ✅ 无 | 当日数据实时可得 |")
        lines.append("| 小 | 小市值换手 | 市值×换手率因子 | ✅ 无 | 纯量价衍生 |")
        lines.append("| 动 | 动量因子 | 多周期价格动量 | ✅ 无 | 纯价格计算 |")
        lines.append("| 波 | 波动率 | 历史波动率/ATR | ✅ 无 | 纯价格计算 |")
        lines.append("")
        lines.append("> ⚠️ **前视偏差提醒**: Qlib/情绪/基本面/资金流 四类信号源依赖外部数据，")
        lines.append("> 历史验证时无法保证数据时间戳与回看日期完全对齐。引擎未强制关闭这些信号，")
        lines.append("> 由用户自行判断风险。标注 ✅ 的信号源不存在此前视偏差。")
        lines.append("")

        # ═══════════════════════════════════════
        # 3. 快扫结果
        # ═══════════════════════════════════════
        lines.append("## 3. 快扫结果")
        lines.append("")
        lines.append(f"在当前市场截面数据上并行筛选 {sweep.get('total', 0)} 组权重配置。")
        lines.append(f"**注意**: 快扫仅使用单日截面数据，结果可能过拟合当日市场风格，")
        lines.append(f"最终结论应以历史验证 (第4节) 为准。")
        lines.append("")

        sweep_results = sweep.get("results", [])
        valid_sweep = [r for r in sweep_results if r.get("candidate_count", 0) > 0]
        zero_sweep = [r for r in sweep_results if r.get("candidate_count", 0) == 0]
        error_sweep = [r for r in sweep_results if r.get("status") == "error"]

        # 3.1 全部排名
        lines.append("### 3.1 全部配置排名")
        lines.append("")
        lines.append(f"共 {len(sweep_results)} 组，其中 {len(valid_sweep)} 组有候选，{len(zero_sweep)} 组无候选，{len(error_sweep)} 组异常。")
        lines.append("")
        lines.append("| 排名 | 配置标签 | 候选数 | 均分 | 标准差 | 最高分 | 最低分 | 高分率 | Top-5 代码 |")
        lines.append("|------|----------|--------|------|--------|--------|--------|--------|------------|")
        for i, r in enumerate(sweep_results):
            hr = r.get("high_score_ratio", 0) * 100
            top5 = " ".join(r.get("top_codes", [])[:5]) if r.get("top_codes") else "-"
            lines.append(f"| {i+1} | {r.get('label', '-')} | {r.get('candidate_count', 0)} | "
                       f"{r.get('score_mean', 0):.4f} | {r.get('score_std', 0):.4f} | "
                       f"{r.get('score_max', 0):.4f} | {r.get('score_min', 0):.4f} | "
                       f"{hr:.0f}% | {top5} |")
        lines.append("")

        # 3.2 得分分布统计
        if valid_sweep:
            means = [r["score_mean"] for r in valid_sweep]
            import numpy as np
            lines.append("### 3.2 快扫得分分布")
            lines.append("")
            lines.append("| 统计量 | 均分 | 最高分 | 高分率 |")
            lines.append("|--------|------|--------|--------|")
            lines.append(f"| 最大值 | {max(means):.4f} | {max(r.get('score_max',0) for r in valid_sweep):.4f} | {max(r.get('high_score_ratio',0)*100 for r in valid_sweep):.0f}% |")
            lines.append(f"| 75分位 | {np.percentile(means, 75):.4f} | - | - |")
            lines.append(f"| 中位数 | {np.percentile(means, 50):.4f} | - | - |")
            lines.append(f"| 25分位 | {np.percentile(means, 25):.4f} | - | - |")
            lines.append(f"| 最小值 | {min(means):.4f} | {min(r.get('score_max',0) for r in valid_sweep):.4f} | {min(r.get('high_score_ratio',0)*100 for r in valid_sweep):.0f}% |")
            lines.append(f"| 均值 | {np.mean(means):.4f} | - | - |")
            lines.append(f"| 标准差 | {np.std(means):.4f} | - | - |")
            lines.append("")

        # 3.3 零候选配置分析
        if zero_sweep:
            lines.append("### 3.3 零候选配置 (淘汰)")
            lines.append("")
            lines.append(f"以下 {len(zero_sweep)} 组配置未能筛出任何候选，直接被淘汰：")
            lines.append("")
            lines.append("| 配置标签 |")
            lines.append("|----------|")
            for r in zero_sweep:
                lines.append(f"| {r.get('label', '-')} |")
            lines.append("")

            # 分析零候选的共性
            q_count = sum(1 for r in zero_sweep if r.get("label", "").find("Q") >= 0)
            if q_count > 0:
                lines.append(f"其中 {q_count}/{len(zero_sweep)} 组包含 Qlib 信号 (Q)，")
                lines.append(f"说明当前 Qlib 预测数据缺失导致含 Qlib 的配置大面积失效。")
            lines.append("")

        # 3.4 Top-10 配置完整参数
        top10_sweep = sorted(valid_sweep, key=lambda r: (
            r.get("high_score_ratio", 0), r.get("score_mean", 0)
        ), reverse=True)[:10]
        if top10_sweep:
            lines.append("### 3.4 快扫 Top-10 完整参数")
            lines.append("")
            for i, r in enumerate(top10_sweep):
                lines.append(f"#### #{i+1} {r.get('label', '-')}")
                lines.append("")
                lines.append(f"- 候选数: {r.get('candidate_count', 0)} | 均分: {r.get('score_mean', 0):.4f} | "
                           f"标准差: {r.get('score_std', 0):.4f} | 最高: {r.get('score_max', 0):.4f} | "
                           f"最低: {r.get('score_min', 0):.4f} | 高分率: {r.get('high_score_ratio', 0)*100:.0f}%")
                lines.append("")
                lines.append(_fmt_weight_table(r.get("config", {}), r.get("effective_weights", {})))
                lines.append("")
                # Top candidate codes with scores
                codes = r.get("top_codes", [])
                scores = r.get("top_scores", [])
                if codes:
                    lines.append(f"**Top-{min(30, len(codes))} 候选代码及评分**:")
                    lines.append("")
                    for j, (c, s) in enumerate(zip(codes[:30], scores[:30])):
                        lines.append(f"| {j+1} | {c} | {s:.4f} |")
                    lines.append("")
                lines.append("---")
                lines.append("")

        # ═══════════════════════════════════════
        # 4. 历史验证结果
        # ═══════════════════════════════════════
        lines.append("## 4. 历史验证结果")
        lines.append("")
        if v_edates:
            lines.append(f"**验证日期**: {len(v_edates)} 个交易日截面")
            lines.append(f"({v_edates[0]} ~ {v_edates[-1]})")
        lines.append("")
        lines.append(f"每个日期用当日快照独立筛选 Top-K 股票，以次日开盘价入场，")
        lines.append(f"分别计算 5/10/20 个交易日的 forward return。")
        lines.append(f"各日期收益取 Top-K 股票均值，然后跨所有日期聚合统计。")
        lines.append("")

        if not vresults:
            lines.append("*无验证数据*\n")
        else:
            # 4.1 综合排名
            lines.append("### 4.1 综合排名 (按适应度)")
            lines.append("")
            lines.append("| 排名 | 配置 | 适应度 | 20日收益 | 20日胜率 | Sharpe | Rank IC | "
                       "10日收益 | 10日胜率 | 5日收益 | 5日胜率 | 有效日期 |")
            lines.append("|------|------|--------|----------|----------|--------|---------|"
                       "----------|----------|---------|---------|----------|")
            for r in vresults:
                lines.append(f"| {r.get('rank', '-')} | {r.get('label', '-')} | "
                           f"{r.get('fitness', 0):.4f} | "
                           f"{(r.get('mean_20d', 0)*100):.2f}% | "
                           f"{(r.get('hit_rate_20d', 0)*100):.0f}% | "
                           f"{r.get('sharpe_20d', 0):.2f} | "
                           f"{r.get('rank_ic', 0):.4f} | "
                           f"{(r.get('mean_10d', 0)*100):.2f}% | "
                           f"{(r.get('hit_rate_10d', 0)*100):.0f}% | "
                           f"{(r.get('mean_5d', 0)*100):.2f}% | "
                           f"{(r.get('hit_rate_5d', 0)*100):.0f}% | "
                           f"{r.get('n_dates', 0)} |")
            lines.append("")

            # 4.2 各周期分别排名
            periods = [
                ("5d", "5日", "mean_5d", "hit_rate_5d", "sharpe_5d", "std_5d"),
                ("10d", "10日", "mean_10d", "hit_rate_10d", "sharpe_10d", "std_10d"),
                ("20d", "20日", "mean_20d", "hit_rate_20d", "sharpe_20d", "std_20d"),
            ]
            for key, label, mean_key, hit_key, sharpe_key, std_key in periods:
                ranked = sorted(vresults, key=lambda r: r.get(mean_key, 0) or 0, reverse=True)
                lines.append(f"### 4.2.{['5d','10d','20d'].index(key)+1} {label}收益排名")
                lines.append("")
                lines.append("| 排名 | 配置 | 均收益 | 标准差 | 胜率 | Sharpe | 最大单日收益 | 最大单日亏损 |")
                lines.append("|------|------|--------|--------|------|--------|-------------|-------------|")
                for i, r in enumerate(ranked):
                    # 从 date_details 中找最大单日收益/亏损
                    dd = r.get("date_details", [])
                    max_ret = max((d.get(f"ret_{key}") or 0 for d in dd), default=0)
                    min_ret = min((d.get(f"ret_{key}") or 0 for d in dd), default=0)
                    lines.append(f"| {i+1} | {r.get('label', '-')} | "
                               f"{(r.get(mean_key, 0)*100):.2f}% | "
                               f"{(r.get(std_key, 0)*100):.2f}% | "
                               f"{(r.get(hit_key, 0)*100):.0f}% | "
                               f"{r.get(sharpe_key, 0):.2f} | "
                               f"{(max_ret*100):.2f}% | "
                               f"{(min_ret*100):.2f}% |")
                lines.append("")

            # 4.3 各配置逐日明细
            lines.append("### 4.3 各配置逐日明细")
            lines.append("")
            for i, r in enumerate(vresults):
                label = r.get("label", f"配置{i+1}")
                lines.append(f"#### 配置 #{i+1}: {label}")
                lines.append("")
                lines.append(f"综合: 20日均收益 {(r.get('mean_20d', 0)*100):.2f}% | "
                           f"胜率 {(r.get('hit_rate_20d', 0)*100):.0f}% | "
                           f"Sharpe {r.get('sharpe_20d', 0):.2f} | "
                           f"Rank IC {r.get('rank_ic', 0):.4f} | "
                           f"有效日期 {r.get('n_dates', 0)}")
                lines.append("")
                dd = r.get("date_details", [])
                if dd:
                    lines.append("| 日期 | Top-5 代码 | 5日收益 | 10日收益 | 20日收益 | 候选数 |")
                    lines.append("|------|-----------|---------|----------|----------|--------|")
                    for d in dd:
                        codes_str = " ".join(d.get("codes", [])[:5]) if d.get("codes") else "-"
                        r5 = f"{(d.get('ret_5d') or 0)*100:.2f}%" if d.get('ret_5d') is not None else "-"
                        r10 = f"{(d.get('ret_10d') or 0)*100:.2f}%" if d.get('ret_10d') is not None else "-"
                        r20 = f"{(d.get('ret_20d') or 0)*100:.2f}%" if d.get('ret_20d') is not None else "-"
                        lines.append(f"| {d.get('date', '-')} | {codes_str} | "
                                   f"{r5} | {r10} | {r20} | "
                                   f"{d.get('candidate_count', 0)} |")
                    lines.append("")
                lines.append("---")
                lines.append("")

            # 4.4 收益分布统计
            lines.append("### 4.4 收益分布统计 (跨日期)")
            lines.append("")
            for label_key in [("5d", "5日"), ("10d", "10日"), ("20d", "20日")]:
                key, lbl = label_key
                lines.append(f"**{lbl}收益分布**:")
                lines.append("")
                lines.append("| 配置 | 均值 | 标准差 | 最小值 | P25 | 中位数 | P75 | 最大值 | 胜率 |")
                lines.append("|------|------|--------|--------|-----|--------|-----|--------|------|")
                for r in vresults:
                    dd = r.get("date_details", [])
                    rets = [d.get(f"ret_{key}") for d in dd if d.get(f"ret_{key}") is not None]
                    if not rets:
                        continue
                    import numpy as np
                    arr = np.array(rets)
                    lines.append(f"| {r.get('label', '-')} | "
                               f"{np.mean(arr)*100:.2f}% | {np.std(arr)*100:.2f}% | "
                               f"{np.min(arr)*100:.2f}% | {np.percentile(arr,25)*100:.2f}% | "
                               f"{np.median(arr)*100:.2f}% | {np.percentile(arr,75)*100:.2f}% | "
                               f"{np.max(arr)*100:.2f}% | "
                               f"{(arr > 0).mean()*100:.0f}% |")
                lines.append("")

            # 4.5 配置间相关性
            if len(vresults) >= 2:
                lines.append("### 4.5 配置间收益相关性")
                lines.append("")
                lines.append("| | " + " | ".join(r.get("label", f"C{i}") for r in vresults) + " |")
                lines.append("|" + "|".join(["------"] * (len(vresults) + 1)) + "|")
                for i, r1 in enumerate(vresults):
                    dd1 = r1.get("date_details", [])
                    rets1 = {d["date"]: d.get("ret_20d") for d in dd1 if d.get("ret_20d") is not None}
                    row = [r1.get("label", f"C{i}")]
                    for j, r2 in enumerate(vresults):
                        dd2 = r2.get("date_details", [])
                        rets2 = {d["date"]: d.get("ret_20d") for d in dd2 if d.get("ret_20d") is not None}
                        common = sorted(set(rets1) & set(rets2))
                        if len(common) >= 10:
                            v1 = [rets1[d] for d in common]
                            v2 = [rets2[d] for d in common]
                            corr = np.corrcoef(v1, v2)[0, 1] if np.std(v1) > 0 and np.std(v2) > 0 else 0
                            row.append(f"{corr:.3f}")
                        else:
                            row.append("-")
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")

            # 4.6 股票频次分析
            lines.append("### 4.6 高频推荐股票")
            lines.append("")
            lines.append("统计各配置在所有验证日期中，哪些股票最常出现在 Top-K 推荐中：")
            lines.append("")
            for i, r in enumerate(vresults):
                dd = r.get("date_details", [])
                freq = {}
                for d in dd:
                    for code in d.get("codes", [])[:10]:
                        freq[code] = freq.get(code, 0) + 1
                top_codes = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
                if top_codes:
                    lines.append(f"**{r.get('label', '-')}** (共 {len(dd)} 日期):")
                    lines.append("")
                    lines.append("| 代码 | 出现次数 | 出现频率 |")
                    lines.append("|------|----------|----------|")
                    for code, cnt in top_codes:
                        lines.append(f"| {code} | {cnt} | {cnt/len(dd)*100:.0f}% |")
                    lines.append("")

        # ═══════════════════════════════════════
        # 5. 推荐配置
        # ═══════════════════════════════════════
        best_v = vresults[0] if vresults else None
        if best_v:
            best_cfg = best_v.get("config", {})
            lines.append("---")
            lines.append("## 5. 推荐配置")
            lines.append("")
            lines.append(f"**选择逻辑**: 按适应度 (fitness) 排序，综合考量 20日收益、胜率、Rank IC、稳定性。")
            lines.append(f"适应度 = mean_20d × 0.4 + hit_rate_20d × 0.3 + rank_ic × 0.2 + (1 - std_20d/0.3) × 0.1")
            lines.append("")
            lines.append(f"**配置名**: {best_v.get('label', '-')}")
            lines.append("")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| 20日均收益 | **{(best_v.get('mean_20d', 0)*100):.2f}%** |")
            lines.append(f"| 20日胜率 | **{(best_v.get('hit_rate_20d', 0)*100):.0f}%** |")
            lines.append(f"| 20日 Sharpe | **{best_v.get('sharpe_20d', 0):.2f}** |")
            lines.append(f"| 10日均收益 | {(best_v.get('mean_10d', 0)*100):.2f}% |")
            lines.append(f"| 10日胜率 | {(best_v.get('hit_rate_10d', 0)*100):.0f}% |")
            lines.append(f"| 5日均收益 | {(best_v.get('mean_5d', 0)*100):.2f}% |")
            lines.append(f"| 5日胜率 | {(best_v.get('hit_rate_5d', 0)*100):.0f}% |")
            lines.append(f"| Rank IC | {best_v.get('rank_ic', 0):.4f} |")
            lines.append(f"| 适应度 | {best_v.get('fitness', 0):.4f} |")
            lines.append(f"| 有效验证日期 | {best_v.get('n_dates', 0)} |")
            lines.append("")
            lines.append("### 权重与开关")
            lines.append("")
            lines.append(_fmt_weight_table(best_cfg))
            lines.append("")

            # 对比其他配置
            if len(vresults) >= 2:
                lines.append("### 与其他配置对比")
                lines.append("")
                lines.append("| 指标 | " + " | ".join(r.get("label", "-") for r in vresults) + " |")
                lines.append("|------|" + "|".join(["------"] * len(vresults)) + "|")
                metrics = [
                    ("20日均收益", "mean_20d", "%", True),
                    ("20日胜率", "hit_rate_20d", "%", True),
                    ("20日 Sharpe", "sharpe_20d", "", True),
                    ("10日均收益", "mean_10d", "%", True),
                    ("5日均收益", "mean_5d", "%", True),
                    ("Rank IC", "rank_ic", "", True),
                    ("适应度", "fitness", "", True),
                ]
                for m_name, m_key, unit, _ in metrics:
                    vals = []
                    for r in vresults:
                        v = r.get(m_key, 0) or 0
                        if unit == "%":
                            vals.append(f"{v*100:.2f}%")
                        else:
                            vals.append(f"{v:.4f}")
                    lines.append(f"| {m_name} | " + " | ".join(vals) + " |")
                lines.append("")

        lines.append("---")
        lines.append(f"*报告由 Apallg投研 Phase 2 权重优化引擎自动生成，共 {len(lines)} 行*")

        from core.shared.persistence import save_text
        save_text("\n".join(lines), path)
        logger.info(f"[Phase2] 详尽报告已生成: {path} ({len(lines)} 行)")
        return filename

    def _generate_report(self, weights: Dict[str, float], skips: Dict[str, bool],
                         tech_presets: List[str], l0_thresholds: List[float],
                         result: Dict) -> str:
        """生成调优报告 Markdown 文件到项目根目录"""
        import os
        now_strftime_val = now_strftime('%Y%m%d_%H%M%S')
        filename = f"tuner_report_{now_strftime_val}.md"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, filename)

        lines = []
        lines.append(f"# 信号融合配置调优报告")
        lines.append(f"**生成时间**: {now_strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## 输入参数")
        lines.append("")
        lines.append("### 信号权重")
        lines.append("| 信号源 | 权重 |")
        lines.append("|--------|------|")
        weight_labels = {s["weight_key"]: s["label"] for s in SIGNAL_SOURCES}
        for k, v in weights.items():
            lines.append(f"| {weight_labels.get(k, k)} | {v*100:.0f}% |")
        lines.append("")
        lines.append("### 信号开关")
        lines.append("| 信号源 | 状态 |")
        lines.append("|--------|------|")
        skip_labels = {s["skip_key"]: s["label"] for s in SIGNAL_SOURCES}
        for k, v in skips.items():
            lines.append(f"| {skip_labels.get(k, k)} | {'关闭' if v else '开启'} |")
        lines.append("")
        lines.append(f"- 技术预设: {', '.join(tech_presets)}")
        lines.append(f"- L0阈值: {', '.join(str(t) for t in l0_thresholds)}")
        lines.append("")

        # 快扫结果
        sweep = result.get("sweep", {})
        lines.append("## 快扫结果")
        lines.append(f"- 总配置数: {sweep.get('total', 0)}")
        lines.append(f"- 完成数: {sweep.get('completed', 0)}")
        best = sweep.get("best")
        if best:
            lines.append(f"- 最优配置: **{best['label']}** (均分 {best['score_mean']}, 高分率 {best.get('high_score_ratio', 0)*100:.0f}%)")
        lines.append("")
        lines.append("| 排名 | 配置 | 候选数 | 均分 | 最高分 | 高分率 |")
        lines.append("|------|------|--------|------|--------|--------|")
        for i, r in enumerate(sweep.get("results", [])[:10]):
            hr = r.get("high_score_ratio", 0) * 100
            lines.append(f"| {i+1} | {r.get('label', '-')} | {r.get('candidate_count', 0)} | "
                        f"{r.get('score_mean', 0)} | {r.get('score_max', 0)} | {hr:.0f}% |")
        lines.append("")

        # 自动选优
        lines.append("## 自动选中最优3组")
        for i, label in enumerate(result.get("top3_selected", [])):
            score = result.get("top3_scores", [])[i] if i < len(result.get("top3_scores", [])) else "-"
            lines.append(f"{i+1}. {label} (均分: {score})")
        lines.append("")

        # 历史验证
        validate = result.get("validate", {})
        lines.append(f"## 历史验证结果 ({validate.get('total_dates', 0)} 个日期)")
        lines.append("")
        vbest = validate.get("best")
        if vbest:
            lines.append(f"**最优配置: {vbest['label']}** - 适应度 {vbest.get('fitness', '-')}, "
                        f"20日收益 {vbest.get('mean_20d', 0)*100:.2f}%, "
                        f"胜率 {vbest.get('hit_rate_20d', 0)*100:.0f}%")
        lines.append("")
        lines.append("| 排名 | 配置 | 适应度 | 20日收益 | 20日胜率 | Rank IC | 5日收益 | 10日收益 | 日期数 |")
        lines.append("|------|------|--------|----------|----------|---------|---------|----------|--------|")
        for r in validate.get("results", []):
            lines.append(f"| {r.get('rank', '-')} | {r.get('label', '-')} | {r.get('fitness', '-')} | "
                        f"{(r.get('mean_20d', 0)*100):.2f}% | {(r.get('hit_rate_20d', 0)*100):.0f}% | "
                        f"{r.get('rank_ic', '-')} | {(r.get('mean_5d', 0) or 0)*100:.2f}% | "
                        f"{(r.get('mean_10d', 0) or 0)*100:.2f}% | {r.get('n_dates', '-')} |")
        lines.append("")
        lines.append("---")
        lines.append(f"*报告由 Apallg投研 配置调优引擎自动生成*")

        from core.shared.persistence import save_text
        save_text("\n".join(lines), path)
        logger.info(f"[调优] 报告已生成: {path}")
        return filename

    # ══════════════════════════════════════════════
    # Phase 3: 市场状态自适应 Walk-Forward 回测
    # ══════════════════════════════════════════════

    def regime_historical_replay(self, eval_weeks=52, date_start=None, date_end=None,
                                 date_step=5, index_name="沪深300",
                                 progress=None) -> Dict:
        """验证分类器质量: 各状态下后续20日指数实际涨跌是否匹配预期"""
        import time as _time
        eval_dates = self._get_eval_dates(weeks=eval_weeks, date_start=date_start,
                                          date_end=date_end, step=date_step)
        if not eval_dates:
            return {"error": "无交易日数据"}

        from core.data.fetcher import DataFetcher
        from core.domain.screening.indicators import Analyzer
        from core.domain.regime.classifier import MarketRegimeClassifier

        logger.info(f"[RegimeReplay] 获取 {index_name} 指数数据...")
        df = DataFetcher().get_index_daily(index_name, days=365 * 2)
        if df is None or df.empty:
            return {"error": f"无法获取{index_name}指数数据"}
        if "trade_date" not in df.columns and "date" in df.columns:
            df["trade_date"] = df["date"]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = Analyzer.add_indicators(df)
        df = df.sort_values("trade_date").reset_index(drop=True)

        classifier = MarketRegimeClassifier(index_name=index_name)

        per_date = []
        regime_stats = {}
        total = len(eval_dates)
        t0 = _time.time()

        for i, eval_date in enumerate(eval_dates):
            if progress:
                progress.update({"stage": "replay", "current": i + 1, "total": total,
                               "message": f"分类回放 {eval_date}"})

            window = df[df["trade_date"] <= pd.Timestamp(eval_date)]
            if len(window) < 60:
                continue

            rr = classifier.classify_from_df(window)
            idx_fwd = self._compute_forward_return_index(df, eval_date, 20)

            entry = {
                "date": eval_date.isoformat(),
                "regime": rr.regime.value,
                "confidence": rr.confidence,
                "sub_scores": rr.sub_scores,
                "index_forward_20d": idx_fwd,
            }
            per_date.append(entry)

            rg = rr.regime.value
            if rg not in regime_stats:
                regime_stats[rg] = {"count": 0, "returns": [], "confidences": []}
            regime_stats[rg]["count"] += 1
            if idx_fwd is not None:
                regime_stats[rg]["returns"].append(idx_fwd)
            regime_stats[rg]["confidences"].append(rr.confidence)

        # 汇总各状态的统计
        by_regime = {}
        import numpy as np
        for rg, st in regime_stats.items():
            rets = st["returns"]
            by_regime[rg] = {
                "count": st["count"],
                "pct_of_total": round(st["count"] / max(len(per_date), 1) * 100, 1),
                "mean_forward_20d": round(np.mean(rets) * 100, 2) if rets else 0,
                "median_forward_20d": round(np.median(rets) * 100, 2) if rets else 0,
                "hit_rate": round(sum(1 for r in rets if r > 0) / max(len(rets), 1), 3),
                "avg_confidence": round(np.mean(st["confidences"]), 3),
            }

        # 状态转移矩阵
        prev = None
        transitions = {}
        stays = {}
        for entry in per_date:
            rg = entry["regime"]
            stays[rg] = stays.get(rg, 0) + 1
            if prev and prev != rg:
                key = f"{prev}→{rg}"
                transitions[key] = transitions.get(key, 0) + 1
            prev = rg

        total_tx = max(sum(transitions.values()), 1)
        transitions_pct = {k: round(v / total_tx * 100, 1) for k, v in transitions.items()}

        avg_duration = round(len(per_date) / max(len(transitions), 1), 1)

        elapsed = _time.time() - t0
        logger.info(f"[RegimeReplay] 完成: {len(per_date)} 日期, {elapsed:.1f}s")
        return {
            "eval_dates": len(eval_dates),
            "valid_dates": len(per_date),
            "by_regime": by_regime,
            "transitions": {"avg_duration_days": avg_duration,
                           "counts": transitions,
                           "pct": transitions_pct},
            "per_date": per_date,
        }

    def walk_forward_regime_test(self, pool="all", eval_weeks=52,
                                 date_start=None, date_end=None,
                                 top_k=10, code_prefixes=None,
                                 date_step=5, min_turnover=5e7,
                                 progress=None) -> Dict:
        """Walk-forward回测: regime-aware动态权重 vs 固定权重,逐日对比

        每个交易日: 判断状态 → 用对应权重选股 → 计算N日后收益 → 对比基线
        """
        import time as _time
        import numpy as np

        screener = get_screener()
        eval_dates = self._get_eval_dates(weeks=eval_weeks, date_start=date_start,
                                          date_end=date_end, step=date_step)
        if not eval_dates:
            return {"error": "无交易日数据"}

        # 基线权重 — 使用 Phase 1 检测出的有效信号默认平均权重
        from core.domain.regime.weights import REGIME_LABELS, REGIME_RATIONALE
        baseline_weights = {"tech": 0.25, "qlib": 0.20, "sentiment": 0.05,
                           "fundamental": 0.10, "flow": 0.05,
                           "north_flow": 0.10, "size_turnover": 0.10,
                           "momentum": 0.10, "volatility": 0.05}

        total = len(eval_dates)
        per_date = []
        t0 = _time.time()

        for i, eval_date in enumerate(eval_dates):
            t_date = _time.time()
            date_str = eval_date.isoformat()
            if progress:
                progress.update({"stage": "wf_regime", "current": i + 1, "total": total,
                               "message": f"Walk-forward {date_str} ({i+1}/{total})"})

            # 快照优先读缓存
            try:
                snap = _load_cached_snapshot(eval_date, pool)
                if snap is None:
                    snap = screener.precompute_snapshot(
                        pool=pool, code_prefixes=code_prefixes, skip_tech=False,
                        eval_date=eval_date, fast=True, min_turnover=min_turnover)
                    if snap:
                        _save_cached_snapshot(eval_date, pool, snap)
                if not snap:
                    continue
            except Exception as e:
                logger.error(f"[WF] {date_str} snapshot 失败: {e}")
                continue

            # 加载 Qlib 预测
            try:
                best = screener._find_best_qlib_model()
                if best:
                    raw = screener._load_qlib_predictions(best, as_of=eval_date, pool=pool)
                    if raw:
                        screener.set_qlib_predictions_by_date({date_str: dict(raw)})
            except Exception:
                pass

            # —— regime-aware ——
            try:
                r_regime = screener.screen(
                    pool=pool, top_n=top_k * 5, skip_l1=True, skip_l2=True, skip_debate=True,
                    eval_date=eval_date, snapshot=snap, regime="auto",
                    job_id=f"wf_r_{date_str}",
                )
                r_cands = r_regime.get("candidates", [])
                r_codes = [c["code"] for c in r_cands[:top_k]]
                r_scores = [c.get("l1_score", c.get("l0_score", 0)) for c in r_cands[:top_k]]
                r_fwd = self._compute_forward_returns(r_codes, eval_date)
                r_regime_info = r_regime.get("regime", {})
            except Exception as e:
                logger.error(f"[WF] {date_str} regime-aware 失败: {e}")
                r_codes, r_scores, r_fwd, r_regime_info = [], [], {}, {}

            # —— baseline (固定权重) ——
            try:
                r_base = screener.screen(
                    pool=pool, top_n=top_k * 5, skip_l1=True, skip_l2=True, skip_debate=True,
                    source_weights=baseline_weights, tech_preset="all",
                    eval_date=eval_date, snapshot=snap,
                    job_id=f"wf_b_{date_str}",
                )
                b_cands = r_base.get("candidates", [])
                b_codes = [c["code"] for c in b_cands[:top_k]]
                b_scores = [c.get("l1_score", c.get("l0_score", 0)) for c in b_cands[:top_k]]
                b_fwd = self._compute_forward_returns(b_codes, eval_date)
            except Exception as e:
                logger.error(f"[WF] {date_str} baseline 失败: {e}")
                b_codes, b_scores, b_fwd = [], [], {}

            del snap
            screener.set_qlib_predictions_by_date({})

            entry = {
                "date": date_str,
                "regime": r_regime_info.get("regime", "unknown"),
                "regime_confidence": r_regime_info.get("confidence", 0),
                "r_codes": r_codes, "r_5d": r_fwd.get("5d"), "r_10d": r_fwd.get("10d"), "r_20d": r_fwd.get("20d"),
                "b_codes": b_codes, "b_5d": b_fwd.get("5d"), "b_10d": b_fwd.get("10d"), "b_20d": b_fwd.get("20d"),
            }
            per_date.append(entry)

            dt = _time.time() - t_date
            logger.info(f"[WF] {date_str} regime={entry['regime']} "
                       f"r_20d={entry['r_20d']} b_20d={entry['b_20d']} ({dt:.1f}s)")

        # —— 聚合统计 ——
        valid = [e for e in per_date if e["r_20d"] is not None and e["b_20d"] is not None]
        r_rets = [e["r_20d"] for e in valid]
        b_rets = [e["b_20d"] for e in valid]
        r_wins = sum(1 for e in valid if (e["r_20d"] or 0) > (e["b_20d"] or 0))
        b_wins = sum(1 for e in valid if (e["b_20d"] or 0) > (e["r_20d"] or 0))

        # 各状态分组统计
        by_regime = {}
        for e in valid:
            rg = e["regime"]
            if rg not in by_regime:
                by_regime[rg] = {"count": 0, "r_rets": [], "b_rets": []}
            by_regime[rg]["count"] += 1
            by_regime[rg]["r_rets"].append(e["r_20d"] or 0)
            by_regime[rg]["b_rets"].append(e["b_20d"] or 0)

        by_regime_summary = {}
        for rg, st in by_regime.items():
            r_mean = np.mean(st["r_rets"]) if st["r_rets"] else 0
            b_mean = np.mean(st["b_rets"]) if st["b_rets"] else 0
            by_regime_summary[rg] = {
                "count": st["count"],
                "regime_mean_20d": round(r_mean * 100, 2),
                "baseline_mean_20d": round(b_mean * 100, 2),
                "delta_bps": round((r_mean - b_mean) * 10000),
                "regime_hit_rate": round(sum(1 for r in st["r_rets"] if r > 0) / max(len(st["r_rets"]), 1), 3),
                "baseline_hit_rate": round(sum(1 for r in st["b_rets"] if r > 0) / max(len(st["b_rets"]), 1), 3),
            }

        elapsed = _time.time() - t0
        logger.info(f"[WF] 完成: {len(valid)}/{len(per_date)} 有效日期, {elapsed:.0f}s "
                    f"regime胜{r_wins} baseline胜{b_wins}")

        return {
            "total_dates": len(eval_dates),
            "valid_dates": len(valid),
            "summary": {
                "regime_mean_20d": round(np.mean(r_rets) * 100, 2) if r_rets else 0,
                "baseline_mean_20d": round(np.mean(b_rets) * 100, 2) if b_rets else 0,
                "regime_hit_rate": round(sum(1 for r in r_rets if r > 0) / max(len(r_rets), 1), 3),
                "baseline_hit_rate": round(sum(1 for r in b_rets if r > 0) / max(len(b_rets), 1), 3),
                "regime_wins": r_wins,
                "baseline_wins": b_wins,
            },
            "by_regime": by_regime_summary,
            "regime_labels": {k.value: v for k, v in REGIME_LABELS.items()},
            "regime_rationale": {k.value: v for k, v in REGIME_RATIONALE.items()},
            "per_date": per_date,
        }

    def _compute_forward_return_index(self, df, eval_date, period):
        """计算指数N日后的forward return"""
        try:
            df_sorted = df[df["trade_date"] > pd.Timestamp(eval_date)].sort_values("trade_date")
            if len(df_sorted) < period:
                return None
            idx = min(period - 1, len(df_sorted) - 1)
            entry_close = float(df_sorted.iloc[0]["open"])
            exit_close = float(df_sorted.iloc[idx]["close"])
            if entry_close <= 0 or exit_close <= 0:
                return None
            return round(float(exit_close / entry_close - 1), 4)
        except Exception:
            return None


_tuner = None


def get_tuner() -> ScreeningConfigTuner:
    global _tuner
    if _tuner is None:
        _tuner = ScreeningConfigTuner()
    return _tuner
