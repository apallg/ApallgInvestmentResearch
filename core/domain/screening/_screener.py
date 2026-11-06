"""
选股流水线 — 三级漏斗: L0廉价快筛 → L1策略精排 → L2 AI深度
"""

__all__ = ["BuySideScreener", "get_screener",
           "use_thread_local_screener", "clear_thread_local_screener"]

import ast
import logging
import datetime
import re
import threading
import time
from concurrent.futures import as_completed
from core.shared.infrastructure.thread_pool import thread_pool
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from core.shared import config
from core.shared.persistence import today_date, load_text, load_bytes, load_pickle, save_pickle
from core.domain.signals.meta import MetaSignalFusion
from core.domain.screening.scoring import compute_l0_matrix, rank_l1
from core.domain.screening.exclusion import (
    evaluate_decision_day_filter,
    is_excluded,
    prefilter_by_turnover,
    prefilter_by_index,
)
from core.domain.screening.indicators import Analyzer
from core.domain.screening.macro import apply_sector_bias, sector_bias_bonus
from core.domain.screening.score_pipeline import (
    SCORE_PIPELINE_VERSION,
    apply_score_step,
    build_legacy_processed_result,
    build_score_result,
    clone_score_result,
    score_fingerprint,
    value_fingerprint,
)
from core.domain.backtest.common import limit_pct_for_code

logger = logging.getLogger(__name__)


# ═══ L0 预计算缓存：同一 snapshot+eval_date 下结果不变，跨轮复用 ═══
_l0_precomp_cache = {}

# 每只 code 最新交易日 adj_factor (今日复权因子) — 用于 qfq→原始价还原 (close<2 低价过滤)
_adj_latest_cache = {}
_l0_precomp_lock = threading.Lock()
_L0_PRECOMP_MAXSIZE = 20

# ═══ Qlib 预测缓存：在线选股路径复用 dataset+model，避免每次请求重建 DatasetH ═══
_qlib_pred_cache = {}
_qlib_pred_lock = threading.Lock()
_QLIB_PRED_CACHE_TTL = 300  # 5 分钟 TTL，同日同模型预测结果不变


def _l0_precomp_put(cache, key, subkey, value, maxsize=_L0_PRECOMP_MAXSIZE):
    if len(cache) >= maxsize and key not in cache:
        oldest = next(iter(cache))
        del cache[oldest]
    cache.setdefault(key, {})[subkey] = value

# ══════════════════════════════════════════════
# L0 因子评分 + 数据采集
# ══════════════════════════════════════════════
from core.domain.screening._scoring_legacy import (
    _score_momentum, _score_volatility, _compute_ht_score,
    _prefetch_financials, _score_fundamentals,
    _score_capital_flow, fetch_and_save_capital_flow, backfill_capital_flow,
    fetch_and_save_north_flow, backfill_north_flow,
    fetch_and_save_fundamentals, _score_north_flow,
    _fetch_size_batch, _score_size_turnover, _parse_num,
    _score_illiq, _score_overnight_gap,
    _precompute_industry_returns, _score_industry_residual_momentum,
    _score_volume_price_divergence, _score_smart_money_flow,
    estimate_impact_cost, apply_impact_penalty,
)

_all_codes_cache = None
_all_codes_cache_time = 0.0
_CACHE_TTL = 3600  # 1小时


def _get_all_stock_codes():
    """获取全A股代码列表。Tushare优先，akshare回退，BaoStock兜底。（缓存1小时）"""
    global _all_codes_cache, _all_codes_cache_time
    import time
    now = time.time()
    if _all_codes_cache is not None and (now - _all_codes_cache_time) < _CACHE_TTL:
        return _all_codes_cache
    from core.data.sources.tushare import get_all_stock_codes
    codes = get_all_stock_codes()
    if codes:
        _all_codes_cache = codes
        _all_codes_cache_time = now
        return codes
    # 最终回退: BaoStock
    codes = []
    try:
        import baostock as bs
        bs.login()
        rs = bs.query_all_stock()
        while rs.next():
            row = rs.get_row_data()
            if len(row) >= 1:
                raw = row[0]
                if "." in raw:
                    raw = raw.split(".")[1]
                if raw.isdigit() and len(raw) == 6:
                    codes.append(raw)
        bs.logout()
    except Exception:
        pass
    return sorted(set(c for c in codes if isinstance(c, str) and len(c) == 6
                      and not is_excluded(c)[0]))


_stable_history_cache = {}


def _get_codes_with_min_days(codes, min_days=500, as_of=None):
    import time
    now = time.time()
    as_of_key = str(as_of)[:10] if as_of is not None else "latest"
    cache_key = (int(min_days), as_of_key)
    cached = _stable_history_cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return [c for c in codes if c in cached[1]]
    try:
        from core.data import Database
        if not Database.is_available():
            return [] if as_of is not None else codes
        if not codes:
            return []
        ph = ",".join(["%s"] * len(codes))
        date_clause = " AND trade_date < %s" if as_of is not None else ""
        params = list(codes)
        if as_of is not None:
            params.append(as_of_key)
        params.append(min_days)
        rows = Database.fetchall(
            f"SELECT code, COUNT(*) as cnt FROM daily_quotes "
            f"WHERE code IN ({ph}) AND open IS NOT NULL AND close IS NOT NULL{date_clause} "
            f"GROUP BY code HAVING COUNT(*) >= %s",
            params,
        )
        eligible = set(r["code"] for r in rows)
        _stable_history_cache[cache_key] = (time.time(), eligible)
        filtered = [c for c in codes if c in eligible]
        logger.info(f"[StableHistory500D] PIT过滤: {len(codes)} → {len(filtered)} 只 "
                    f"(决策日前>= {min_days}条, as_of={as_of_key})")
        return filtered
    except Exception as error:
        logger.warning(f"[StableHistory500D] PIT过滤失败: {error}")
        return [] if as_of is not None else codes


def filter_pool_codes(codes, pool, as_of=None):
    """按股票池过滤全市场代码 (指数/市值类池)。hot_sectors/custom/concept 由调用方处理。"""
    if pool in ("all", "hot_sectors", "sector_rotation", "concept_rotation", "custom", "concept"):
        return codes
    if pool in ("a500", "stable_history_500d"):
        return _get_codes_with_min_days(codes, 500, as_of=as_of)
    try:
        from core.data.sources.tushare import get_index_components, get_small_cap_codes
        if pool in ("csi300", "csi_all"):
            pool_codes = set(get_index_components("000300"))
        elif pool == "csi500":
            pool_codes = set(get_index_components("000905"))
        elif pool == "csi300+csi500":
            pool_codes = set(get_index_components("000300")) | set(get_index_components("000905"))
        elif pool == "small_cap":
            pool_codes = set(get_small_cap_codes())
        elif pool == "non_csi300":
            pool_codes = set(get_index_components("000300"))
            return [c for c in codes if c not in pool_codes]
        else:
            return codes
        return [c for c in codes if c in pool_codes]
    except Exception:
        return codes


def _instruments_count_to_str(n: int) -> str:
    if n <= 350:
        return "csi300"
    elif n <= 600:
        return "csi500"
    return "all"


def _resolve_instruments_param(raw: str, default: str, file_bytes: int = 0) -> str:
    """从 MLflow params 文件中解析 instruments 字符串.

    支持四种格式:
    - 完整列表: ['sh600519', 'sz000001', ...] → 按数量推断
    - 截断列表: MLflow 截断至 ~1000 字符 → 文件 >=900B 时保留默认值
    - 紧凑格式: list_4321_stocks → 提取 N 后推断
    - 普通字符串: csi300 / all → 直接使用
    """
    if raw.startswith("["):
        try:
            codes = ast.literal_eval(raw)
            return _instruments_count_to_str(len(codes))
        except Exception:
            last_comma = raw.rfind("', ")
            if last_comma > 0:
                try:
                    salvaged = raw[:last_comma] + "']"
                    codes = ast.literal_eval(salvaged)
                    n = len(codes)
                    if file_bytes >= 900:
                        return default
                    return _instruments_count_to_str(n)
                except Exception:
                    pass
            return default
    if raw.startswith("list_"):
        m = re.match(r"list_(\d+)_stocks", raw)
        if m:
            return _instruments_count_to_str(int(m.group(1)))
        return default
    return raw




def _read_exp_name(exp_dir):
    """读 MLflow experiment 目录 meta.yaml 的 name 字段, 失败返回空串"""
    try:
        meta = exp_dir / "meta.yaml"
        if meta.exists():
            for line in load_text(str(meta)).splitlines():
                if line.startswith("name:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _get_model_type(exp_name, artifacts_dir):
    """从实验名和 artifacts 推断模型类型。

    返回模型类型字符串: 'lightgbm', 'xgboost', 'catboost', 'linear',
    'lightgbm_xgboost'(双模型) 等。
    """
    # 双模型: 两个 params 文件都存在
    has_lgb = (artifacts_dir / "params_lgb.pkl").exists()
    has_xgb = (artifacts_dir / "params_xgb.pkl").exists()
    if has_lgb and has_xgb:
        return "lightgbm_xgboost"

    # 单模型: 从实验名/run名解析
    name = exp_name.lower()
    for mt in ("catboost", "lightgbm", "xgboost", "linear",
               "lstm", "gru", "transformer", "tcn", "tabnet"):
        if mt in name:
            return mt
    return "unknown"


def _get_regime_bonus(model_type, regime_str=None):
    """返回某模型类型在当前市场状态下的偏好乘数。"""
    if regime_str is None:
        return 1.0
    try:
        from core.shared import config as _cfg
        bonus_map = getattr(_cfg, "REGIME_MODEL_TYPE_BONUS", {})
        regime_bonus = bonus_map.get(regime_str, {})
    except Exception:
        return 1.0

    # 双模型取平均值
    if "_" in model_type:
        types = model_type.split("_")
        bonuses = [regime_bonus.get(t, 1.0) for t in types]
        return sum(bonuses) / len(bonuses)

    return regime_bonus.get(model_type, 1.0)


def _restore_trained_handler_kwargs(artifacts, params_dir):
    """还原 WhiteBox 模型训练时的 factor_keys/processors，保证预测特征与训练完全一致。

    优先读 artifacts/factor_keys.json（双模型，含 shared/infer_processors），
    回退 MLflow params 文件（单模型）。
    """
    import json as _json
    out = {}
    fk_json = artifacts / "factor_keys.json"
    if fk_json.exists():
        try:
            txt = load_bytes(str(fk_json)).decode("utf-8", errors="ignore")
            obj = _json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            if obj.get("custom_factor_keys"):
                out["factor_keys"] = obj["custom_factor_keys"]
            for k in ("shared_processors", "infer_processors", "learn_processors"):
                if obj.get(k):
                    out[k] = obj[k]
        except Exception:
            pass
    if params_dir.exists():
        if "factor_keys" not in out:
            pf = params_dir / "dataset.kwargs.handler.kwargs.factor_keys"
            if pf.exists():
                try:
                    val = load_text(str(pf)).strip()
                    keys = [k.strip().strip("'\"") for k in val.strip("[]").split(",") if k.strip()]
                    if keys:
                        out["factor_keys"] = keys
                except Exception:
                    pass
        for k in ("shared_processors", "infer_processors"):
            if k not in out:
                pf = params_dir / f"dataset.kwargs.handler.kwargs.{k}"
                if pf.exists():
                    try:
                        v = ast.literal_eval(load_text(str(pf)).strip())
                        if v:
                            out[k] = v
                    except Exception:
                        pass
    return out


def normalize_qlib_scores(scores, ref_date, industry_map=None, neutralize=None,
                          return_details=False, model_input_semantics="raw_model_score"):
    """按需中性化后做分位映射, 返回 {code: [-1,1] 分位分}"""
    from core.shared import config as app_config
    use_neutralize = (getattr(app_config, "QLIB_PRED_NEUTRALIZE", False)
                      if neutralize is None else bool(neutralize))
    neutralize_error = None
    neutralize_audit = {}
    if not use_neutralize:
        neutralized = scores
        neutralize_status = "disabled"
    elif len(scores) < 10:
        neutralized = scores
        neutralize_status = "skipped_small_universe"
    else:
        try:
            from core.domain.factors.market_cap import get_market_cap_map
            from core.domain.factors.neutralize import cross_sectional_neutralize
            codes = list(scores.keys())
            vals = [scores[c] for c in codes]
            mktcap_map = get_market_cap_map(codes, ref_date)
            if industry_map is None:
                from core.data.fetcher import DataFetcher
                industry_map = DataFetcher().get_industry_map() or {}
            neutralized, neutralize_audit = cross_sectional_neutralize(
                codes, vals, mktcap_map, industry_map, return_details=True)
            neutralize_status = neutralize_audit["status"]
            neutralize_audit["market_cap_fingerprint"] = score_fingerprint(mktcap_map)
            neutralize_audit["industry_fingerprint"] = value_fingerprint({
                code: industry_map.get(code) for code in sorted(codes)
            })
            logger.info(f"[Qlib] 预测分中性化: {len(codes)}只 status={neutralize_status}")
        except Exception as e:
            logger.warning(f"[Qlib] 预测分中性化失败, 用原始分: {e}")
            neutralized = scores
            neutralize_status = "fallback"
            neutralize_error = e
    result = build_score_result(
        scores,
        neutralized,
        neutralize_requested=use_neutralize,
        neutralize_status=neutralize_status,
        neutralize_error=neutralize_error,
        context={"ref_date": str(ref_date)[:10], **neutralize_audit},
        model_input_semantics=model_input_semantics,
    )
    return result if return_details else result["scores"]


class BuySideScreener:
    def __init__(self, fetcher=None, analyzer=None, qlib_predictions=None):
        from core.data.fetcher import DataFetcher
        self.fetcher = fetcher or DataFetcher()
        self.analyzer = analyzer or Analyzer()
        self.qlib_predictions = qlib_predictions
        self._qlib_by_date: Dict[str, Dict[str, float]] = {}
        self._qlib_score_details_by_date = {}
        self._last_score_result = None
        self._financial_cache: Dict[str, list] = {}
        self._jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._df_cache: Dict[str, pd.DataFrame] = {}

    def _get_adj_latest_map(self, codes):
        """批量取每只 code 最新交易日 adj_factor (今日复权因子) — 用于 qfq→原始价还原。
        历史快照的 qfq 价以今日为锚, 长期分红股在历史日的 qfq 价远低于真实成交价,
        <2元低价过滤需还原为原始价判断。模块级缓存, 进程内复用。
        """
        need = list(dict.fromkeys(codes))
        if not need:
            return _adj_latest_cache
        try:
            from core.data import Database
            if Database.is_available():
                ph = ",".join(["%s"] * len(need))
                rows = Database.fetchall(
                    f"SELECT t1.code, t1.adj_factor FROM daily_quotes t1 "
                    f"INNER JOIN (SELECT code, MAX(trade_date) md FROM daily_quotes "
                    f"WHERE code IN ({ph}) GROUP BY code) t2 "
                    f"ON t1.code = t2.code AND t1.trade_date = t2.md",
                    need)
                refreshed = {}
                for r in (rows or []):
                    if r.get("adj_factor"):
                        refreshed[str(r["code"]).strip()] = float(r["adj_factor"])
                with _l0_precomp_lock:
                    for code in need:
                        _adj_latest_cache.pop(str(code).strip(), None)
                    _adj_latest_cache.update(refreshed)
            else:
                with _l0_precomp_lock:
                    for code in need:
                        _adj_latest_cache.pop(str(code).strip(), None)
        except Exception as e:
            with _l0_precomp_lock:
                for code in need:
                    _adj_latest_cache.pop(str(code).strip(), None)
            logger.debug(f"[低价过滤] 最新复权因子获取失败: {e}")
        return _adj_latest_cache

    def _get_listing_dates_map(self, codes):
        try:
            from core.data import StockIdentityRepo
            return StockIdentityRepo.get_listing_dates(codes)
        except Exception as e:
            logger.warning("[上市日期] 稳定证券身份不可用，相关候选将按未知跳过: %s", e)
            return {}

    def _is_low_price(self, code, close, df):
        """低价股过滤 (<2元): 历史快照用 adj_factor 还原原始价再判断, 避免长期分红股 qfq 价被误杀。
        仅在已预加载今日 adj_factor 时还原 (历史路径); 实时路径 close=现价, 直接比较。
        """
        try:
            if close is None or close <= 0:
                return False
            adj_latest = _adj_latest_cache.get(str(code).strip())
            if adj_latest and adj_latest > 0 and df is not None and "adj_factor" in df.columns:
                adj_series = df["adj_factor"].dropna()
                if len(adj_series):
                    adj_row = float(adj_series.iloc[-1])
                    if adj_row > 0:
                        return close * adj_latest / adj_row < 2
            return close < 2
        except Exception:
            return close < 2

    def precompute_snapshot(self, pool="all", code_prefixes=None, concept_sectors=None,
                            custom_sectors=None, skip_tech=False, eval_date=None,
                            min_turnover=None, skip_limit_up=False, fast=False,
                            min_rows=100):
        """预计算全部候选股票的日K线+技术指标，供 sweeps 复用

        min_turnover: 覆盖最小日成交额阈值，None 则使用 config 默认值 (1000万)
        skip_limit_up: True 时不过滤涨停股（调优历史回测用）
        min_rows: 最小K线条数，低于此值的次新股直接跳过 (默认 100)
        """
        codes = _get_all_stock_codes()

        if code_prefixes:
            prefixes = tuple(str(p) for p in code_prefixes)
            codes = [c for c in codes if c.startswith(prefixes)]
        today = eval_date or today_date()
        codes = filter_pool_codes(codes, pool, as_of=today)
        cutoff_date = today - datetime.timedelta(days=60)
        is_historical = eval_date is not None
        end_date_str = today.strftime("%Y%m%d") if is_historical else None
        if is_historical:
            self._get_adj_latest_map(codes)  # 预加载今日复权因子, 供 close<2 原始价还原
        listing_dates = self._get_listing_dates_map(codes)

        snapshot = {}
        unknown_listing_codes = set()
        lock = threading.Lock()
        total = len(codes)
        # 缓存管理器只创建一次 (避免每只股票初始化 diskcache SQLite, cProfile 显示占 15% 耗时)
        _cm = None
        if is_historical and fast:
            from core.shared.infrastructure.cache import CacheManager
            _cm = CacheManager()

        # --- fast 模式批量预加载: 一次 SQL 替代 N 次逐只查询 ---
        if is_historical and fast and config.MYSQL_ENABLED:
            start_date = (today - datetime.timedelta(days=365)).strftime("%Y%m%d")
            end_date = end_date_str or today.strftime("%Y%m%d")

            def _daily_cache_key(code):
                anchor_factor = _adj_latest_cache.get(str(code).strip())
                return CacheManager.make_key(
                    "daily", "qfq", code, start_date, end_date,
                    format(anchor_factor, ".12g") if anchor_factor is not None else "anchor-unavailable",
                )

            missing_codes = []
            for code in codes:
                ck = _daily_cache_key(code)
                df = _cm.get_cached(ck)
                if df is not None and not df.empty:
                    if "trade_date" in df.columns:
                        df["_td"] = pd.to_datetime(df["trade_date"]).dt.date
                        df = df[df["_td"] <= today]
                        df = df.drop(columns=["_td"])
                    elif isinstance(df.index, pd.DatetimeIndex):
                        df = df[df.index <= pd.Timestamp(today)]
                    if len(df) >= min_rows:
                        continue
                missing_codes.append(code)

            if missing_codes:
                logger.info(f"[Snapshot] 缓存 miss {len(missing_codes)}/{len(codes)} 只, 批量加载中...")
                try:
                    from core.data import DailyQuotesRepo
                    batch_result = DailyQuotesRepo.get_range_batch(missing_codes, start_date, end_date)
                    for code, df in batch_result.items():
                        if df is not None and not df.empty:
                            ck = _daily_cache_key(code)
                            try:
                                _cm.set_redis(ck, df, expire=86400 * 3)
                            except Exception:
                                pass
                    logger.info(f"[Snapshot] 批量加载完成: {len(batch_result)} 只回写到 Redis")
                except Exception as e:
                    logger.warning(f"[Snapshot] 批量加载失败, 回退逐只查询: {e}")

        def _fetch_one(code):
            try:
                start_date = (today - datetime.timedelta(days=365)).strftime("%Y%m%d")
                end_date = end_date_str or today.strftime("%Y%m%d")
                df = None
                if is_historical and fast:
                    ck = _daily_cache_key(code)
                    df = _cm.get_cached(ck) if _cm else None  # Redis → diskcache
                    if df is not None and not df.empty:
                        if "trade_date" in df.columns:
                            df["_td"] = pd.to_datetime(df["trade_date"]).dt.date
                            df = df[df["_td"] <= today]
                            df = df.drop(columns=["_td"])
                        elif isinstance(df.index, pd.DatetimeIndex):
                            df = df[df.index <= pd.Timestamp(today)]
                        if len(df) < min_rows:
                            df = None
                    # 缓存已由批量预加载阶段填满, miss 的股票走最后兜底 (逐只 MySQL)
                    if df is None or df.empty:
                        if config.MYSQL_ENABLED:
                            try:
                                from core.data import DailyQuotesRepo
                                df = DailyQuotesRepo.get_range(code, start_date, end_date)
                            except Exception:
                                pass
                        if df is None or df.empty:
                            return
                else:
                    # 先走缓存链 (Redis → diskcache → MySQL缓存)
                    if is_historical:
                        df = self.fetcher.get_stock_daily(code, end_date=end_date_str)
                    else:
                        df = self.fetcher.get_stock_daily(code)
                    if df is not None and not df.empty:
                        pass
                    elif config.MYSQL_ENABLED:
                        try:
                            from core.data import DailyQuotesRepo
                            df = DailyQuotesRepo.get_range(code, start_date, end_date)
                        except Exception:
                            pass
                    if df is None or df.empty:
                        return

                if len(df) < min_rows:
                    return

                # MySQL Decimal → float (向量化: 比逐列 astype 快 3-5×)
                num_cols = [c for c in ("open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg") if c in df.columns]
                if num_cols:
                    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

                latest = df.iloc[-1]
                name = str(latest.get("name", ""))
                close = float(latest.get("close", 0) or 0)
                volume = float(latest.get("volume", 0) or 0)
                pct = float(latest.get("pct_chg", 0) or 0)
                # 补算 pct_chg (数据库可能未采集该字段)
                if pct == 0 and len(df) >= 2:
                    prev_close = float(df.iloc[-2].get("close", 0) or 0)
                    if prev_close > 0:
                        pct = (close / prev_close - 1) * 100
                        df.at[df.index[-1], "pct_chg"] = pct

                if name.upper().startswith("ST") or name.upper().startswith("*ST"):
                    return
                if close <= 0 or volume <= 0:
                    return
                # mask-first: 涨停股在因子计算前排除 (L1213 在 L1231 add_indicators 之前)
                # 涨停日的收盘价不可交易, 保留在因子窗口中将污染移动平均/相关性等计算
                if not skip_limit_up and pct >= limit_pct_for_code(code) * 100 - 0.5:
                    return
                _min_turnover = min_turnover if min_turnover is not None else getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 1e7)
                # 近20日日均成交额 (numpy, 避免 pd.Series.fillna 开销)
                amt_arr = df["amount"].values[-20:]
                avg_turnover = float(np.nanmean(amt_arr)) if len(amt_arr) > 0 and not np.all(np.isnan(amt_arr)) else 0.0
                if avg_turnover < _min_turnover:
                    return
                if self._is_low_price(code, close, df):
                    return

                listing_date = listing_dates.get(str(code).strip())
                if listing_date is None:
                    with lock:
                        unknown_listing_codes.add(code)
                    return
                if listing_date > cutoff_date:
                    return

                if not skip_tech:
                    df = self.analyzer.add_indicators(df)

                with lock:
                    snapshot[code] = (df, name, close, volume)
            except Exception:
                pass

        if len(snapshot) == 1:
            first_code = list(snapshot.keys())[0]
            logger.warning(f'[DEBUG] First pass: {first_code}, total codes: {len(codes)}, eval_date: {today}')

        if fast:
            with thread_pool.default.borrow() as pool:
                list(pool.map(_fetch_one, codes))
        else:
            logger.info(f"[Snapshot] 开始预计算 {total} 只股票")
            with thread_pool.default.borrow() as pool:
                list(pool.map(_fetch_one, codes))

        if unknown_listing_codes:
            logger.warning("[Snapshot] %s只证券缺少可信上市日期，已跳过", len(unknown_listing_codes))
        logger.info(f"[Snapshot] 预计算完成: {len(snapshot)}/{total} 只通过硬排除")
        return snapshot

    def batch_snapshots_from_master(self, master, eval_dates, skip_limit_up=False, min_turnover=None, progress_cb=None):
        """一遍遍历 master, 产出全部日期的快照 {date_key: {code: (df, name, close, volume)}}。
        并行处理 + 单股票 30s 超时，卡住的股票自动跳过。
        """
        import datetime as dt
        from concurrent.futures import as_completed, TimeoutError as FutTimeout
        today_ref = today_date()
        self._get_adj_latest_map(list(master.keys()))  # 预加载今日复权因子, 供 close<2 原始价还原
        listing_dates = self._get_listing_dates_map(list(master.keys()))
        _min_turnover = min_turnover if min_turnover is not None else getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 1e7)
        snapshots = {d.isoformat() if hasattr(d, 'isoformat') else str(d): {} for d in eval_dates}
        total_stocks = len(master)
        snap_lock = threading.Lock()
        stock_done = [0]
        skipped = [0]

        def _process_one(item):
            from bisect import bisect_right
            code, full_df = item
            if "trade_date" not in full_df.columns:
                return
            td = full_df["trade_date"]
            if hasattr(td.iloc[0], "date"):
                dates_col = [d.date() if hasattr(d, 'date') else d for d in td]
            else:
                dates_col = list(pd.to_datetime(td).dt.date)

            for eval_date in eval_dates:
                date_key = eval_date.isoformat() if hasattr(eval_date, 'isoformat') else str(eval_date)
                today = eval_date or today_ref
                cutoff_date = today - dt.timedelta(days=60)

                pos = bisect_right(dates_col, today)
                if pos < 100:
                    continue
                start_idx = max(0, pos - 260)
                if pos - start_idx < 100:
                    continue

                df = full_df.iloc[start_idx:pos].copy()
                if "trade_date" in df.columns:
                    df = df.set_index("trade_date")
                latest = df.iloc[-1]
                close = float(latest.get("close", 0) or 0)
                volume = float(latest.get("volume", 0) or 0)
                name = str(latest.get("name", ""))
                if name.upper().startswith("ST") or name.upper().startswith("*ST"):
                    continue
                if close <= 0 or volume <= 0:
                    continue

                pct = float(latest.get("pct_chg", 0) or 0)
                if pct == 0 and len(df) >= 2:
                    prev_close = float(df.iloc[-2].get("close", 0) or 0)
                    if prev_close > 0:
                        pct = (close / prev_close - 1) * 100
                if not skip_limit_up and pct >= limit_pct_for_code(code) * 100 - 0.5:
                    continue

                amt_arr = df["amount"].values[-20:]
                avg_turnover = float(np.nanmean(amt_arr)) if len(amt_arr) > 0 and not np.all(np.isnan(amt_arr)) else 0.0
                if avg_turnover < _min_turnover:
                    continue
                if self._is_low_price(code, close, df):
                    continue

                listing_date = listing_dates.get(str(code).strip())
                if listing_date is None or listing_date > cutoff_date:
                    continue

                with snap_lock:
                    snapshots[date_key][code] = (df, name, close, volume)

        BATCH_TIMEOUT = 420  # 总超时 7 分钟
        with thread_pool.default.borrow() as ex:
            futs = {ex.submit(_process_one, item): item[0] for item in master.items()}
            import time as _time
            _t0 = _time.time()
            for f in as_completed(futs):
                elapsed = _time.time() - _t0
                if elapsed > BATCH_TIMEOUT:
                    remaining = total_stocks - stock_done[0]
                    for uf in futs:
                        uf.cancel()
                    skipped[0] += remaining
                    break
                code = futs[f]
                stock_done[0] += 1
                try:
                    f.result(timeout=0)
                except Exception:
                    skipped[0] += 1
                if progress_cb and stock_done[0] % 200 == 0:
                    progress_cb({"stage": "snapshot", "current": stock_done[0], "total": total_stocks})
        # 最终进度回调
        if progress_cb:
            progress_cb({"stage": "snapshot", "current": total_stocks, "total": total_stocks})

        if skipped[0]:
            logger.warning(f"[batch_snapshots] {skipped[0]}/{total_stocks} 只股票超时或异常，已跳过")
        return snapshots


    def snapshot_from_master(self, master, eval_date, skip_limit_up=False, min_turnover=None):
        """从预计算的 master dict 切片生成单日期快照。

        master: {code: pd.DataFrame (全量日线 + add_indicators 已完成)}
        返回: {code: (df, name, close, volume)}
        """
        import datetime as dt
        today = eval_date or today_date()
        cutoff_date = today - dt.timedelta(days=60)
        self._get_adj_latest_map(list(master.keys()))  # 预加载今日复权因子, 供 close<2 原始价还原
        listing_dates = self._get_listing_dates_map(list(master.keys()))
        _min_turnover = min_turnover if min_turnover is not None else getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 1e7)
        snapshot = {}

        for code, full_df in master.items():
            if "trade_date" in full_df.columns:
                td = full_df["trade_date"]
                if hasattr(td.iloc[0], "date"):
                    mask = [d <= today for d in td]
                else:
                    mask = pd.to_datetime(td).dt.date <= today
                df = full_df[mask].tail(260).copy()
            elif isinstance(full_df.index, pd.DatetimeIndex):
                df = full_df[full_df.index <= pd.Timestamp(today)].tail(260).copy()
            else:
                continue

            if len(df) < 100:
                continue
            latest = df.iloc[-1]
            close = float(latest.get("close", 0) or 0)
            volume = float(latest.get("volume", 0) or 0)
            name = str(latest.get("name", ""))
            if name.upper().startswith("ST") or name.upper().startswith("*ST"):
                continue
            if close <= 0 or volume <= 0:
                continue
            pct = float(latest.get("pct_chg", 0) or 0)
            if pct == 0 and len(df) >= 2:
                prev_close = float(df.iloc[-2].get("close", 0) or 0)
                if prev_close > 0:
                    pct = (close / prev_close - 1) * 100
            if not skip_limit_up and pct >= limit_pct_for_code(code) * 100 - 0.5:
                continue
            amt_arr = df["amount"].values[-20:]
            avg_turnover = float(np.nanmean(amt_arr)) if len(amt_arr) > 0 and not np.all(np.isnan(amt_arr)) else 0.0
            if avg_turnover < _min_turnover:
                continue
            if self._is_low_price(code, close, df):
                continue
            listing_date = listing_dates.get(str(code).strip())
            if listing_date is None or listing_date > cutoff_date:
                continue
            snapshot[code] = (df, name, close, volume)

        return snapshot

    def set_qlib_predictions(self, predictions: Dict[str, float]):
        """注入 Qlib 预测结果 {code: score}，同步写入 Redis"""
        self.qlib_predictions = predictions
        from core.shared.infrastructure import redis_client as rds
        if rds.redis_enabled() and predictions:
            rds._set("qlib:predictions:latest", dict(predictions), expire=86400)

    def set_qlib_predictions_by_date(self, by_date: Dict[str, Dict[str, float]]):
        """注入按日期对齐的 Qlib 预测 {date_iso: {code: score}}，用于历史回测"""
        self._qlib_by_date = by_date
        self._qlib_score_details_by_date = {
            date_iso: details
            for date_iso, details in self._qlib_score_details_by_date.items()
            if date_iso in by_date
            and details.get("audit", {}).get("processed_fingerprint") == score_fingerprint(by_date[date_iso])
        }
        from core.shared.infrastructure import redis_client as rds
        if rds.redis_enabled() and by_date:
            for date_iso, scores in by_date.items():
                rds._set(f"qlib:predictions:{date_iso}", dict(scores), expire=86400)

    def set_financial_cache(self, cache: Dict[str, list]):
        """注入预取的财务数据缓存 {code: [(report_date, {指标: 值}), ...]}，用于历史回测"""
        self._financial_cache = cache

    # 技术指标预设
    TECH_PRESETS = {
        "all": ["MACD", "RSI", "MA", "KDJ", "BB", "VOLUME"],
        "trend": ["MACD", "MA", "BB"],       # 趋势类
        "reversal": ["RSI", "KDJ", "BB"],     # 反转类
        "momentum": ["MACD", "RSI", "VOLUME"], # 动量类
        "volume_only": ["VOLUME"],
        "macd_only": ["MACD"],
        "rsi_only": ["RSI"],
        "ma_only": ["MA"],
        "kdj_only": ["KDJ"],
        "bb_only": ["BB"],
    }

    def get_qlib_predictions(self, as_of=None, pool="csi300", qlib_model=None,
                             neutralize_qlib=None):
        model = qlib_model or self._find_best_qlib_model()
        if model is None:
            logger.warning("get_qlib_predictions: 无可用的 Qlib 模型")
            return (None, {})
        if neutralize_qlib is None:
            return (model, self._load_qlib_predictions(
                model, as_of=as_of, pool=pool))
        return (model, self._load_qlib_predictions(
            model, as_of=as_of, pool=pool,
            neutralize_qlib=neutralize_qlib))

    def screen(self, pool="all", top_n=None, debate_top=None, debate_agents=None,
               job_id=None, skip_debate=False,
               skip_l0=False, skip_l1=False, skip_l2=False,
               skip_tech=False, skip_qlib=False, skip_sent=False,
               skip_fundamental=False, skip_flow=False,
               skip_north_flow=False, skip_size_turnover=False,
               skip_momentum=False, skip_volatility=False,
               skip_illiq=False, skip_overnight_gap=False,
               skip_ht=False, ht_key=None,
               skip_wb_industry=False, skip_wb_divergence=False, skip_wb_moneyflow=False,
               wb_industry_n=20, wb_divergence_n=10, wb_moneyflow_n=5,
               momentum_n=20, volatility_n=20, illiq_n=20, overnight_gap_n=20,
               source_weights=None, tech_preset="all", qlib_model=None,
               hot_sector_count=10, custom_sectors=None,
               code_prefixes=None, concept_sectors=None,
               eval_date=None, snapshot=None, regime=None, macro_bias=True,
               macro_regime=True, smooth_regime=True, use_vol_switch=False,
               use_conditional_switch=True,
               state_models=None, neutralize_qlib=None,
               sector_fusion=None, factor_overlay=None,
               weight_method="fixed", skip_backfill=False,
               conditional_market_env=None, ranking_mode="legacy",
               decision_day_filter=True) -> Dict:
        """执行完整三级漏斗筛选

        source_weights: 可选 dict，覆盖 L0 信号源权重，如 {'tech': 0.3, 'qlib': 0.5, 'sentiment': 0.2}
        tech_preset: 技术指标预设 — "all" / "trend" / "reversal" / "momentum" / "volume_only"
        qlib_model: Qlib 模型 — (experiment_id, run_id) 元组，为 None 则用预注入预测
        code_prefixes: 可选 list，只保留匹配前缀的股票，如 ['600', '601', '000', '300', '688']
        concept_sectors: 可选 list，概念板块名称列表，pool="concept" 时使用
        regime: 市场状态 — None(不调节) / "auto"(自动检测) / "trending_up" / "trending_down" / "ranging" / "volatile"
        sector_fusion: 板块背离双模型分位融合 — None(按 config.SECTOR_FUSION_ENABLED) / True / False
        """
        if top_n is None:
            top_n = getattr(config, "BUY_SIDE_L1_TOP_N", 50)
        if debate_top is None:
            debate_top = getattr(config, "BUY_SIDE_DEBATE_TOP", 5)
        if debate_agents is None:
            debate_agents = getattr(config, "BUY_SIDE_DEBATE_AGENTS", 6)
        if source_weights is None:
            source_weights = {}
        self._last_fusion_info = None
        self._last_score_result = None
        user_weights = bool(source_weights)
        if user_weights:
            # 显式权重视为完整定义：未列出的信号源权重为 0，不再叠加默认值
            source_weights = dict(source_weights)
            for _sig in ("tech", "qlib", "sentiment", "fundamental", "flow", "north_flow",
                         "size_turnover", "momentum", "volatility", "ht_factors",
                         "wb_industry", "wb_divergence", "wb_moneyflow", "illiq", "overnight_gap"):
                source_weights.setdefault(_sig, 0.0)

        # ── 市场状态自适应权重 ──
        regime_result = None
        market_rr = None
        self._vol_switch_empty = False
        if regime:
            from core.domain.regime.classifier import Regime
            from core.domain.regime.weights import get_regime_weights, get_regime_tech_preset
            if regime == "auto":
                from core.domain.regime.unified import get_unified_regime
                # 实时/历史统一入口: RegimePredictor 6 状态 + 宏观修正 (实时含 EMA 平滑)
                rr = get_unified_regime(as_of=eval_date, apply_macro=macro_regime, smooth=smooth_regime,
                                        use_vol_switch=use_vol_switch)
                market_rr = rr
                macro_info = rr.details.get("macro") if isinstance(rr.details, dict) else None
                # 三档条件切换 (默认开): 高波+宽度低/超跌深→rebound; 低波+无超跌→trend; 中波→空仓(不用模型)
                if (use_vol_switch and use_conditional_switch and isinstance(rr.details, dict)
                        and rr.details.get("p_vol") is not None):
                    try:
                        env = None
                        if conditional_market_env:
                            key = str(eval_date or "")[:10]
                            env = conditional_market_env.get(key)
                            if env is None:
                                candidates = [d for d in conditional_market_env if d <= key]
                                if candidates:
                                    env = conditional_market_env[max(candidates)]
                        if env is not None:
                            from core.domain.regime.weights import get_vol_direction
                            _cond_dir = get_vol_direction(
                                rr.details.get("p_vol"), market_breadth=env[0],
                                oversold_degree=env[1], use_conditional=True)
                            rr.details["vol_direction"] = _cond_dir
                            if _cond_dir == "neutral":
                                self._vol_switch_empty = True
                                logger.info(f"[VolSwitch] {eval_date} 中波死亡区 → 空仓(不使用模型)")
                    except Exception as e:
                        logger.warning(f"[VolSwitch] 条件切换计算失败, 回退二值: {e}")
                regime_result = {"regime": rr.regime.value, "confidence": rr.confidence,
                                 "sub_scores": rr.sub_scores, "as_of": rr.as_of_date,
                                 "macro": macro_info,
                                 "vol_direction": rr.details.get("vol_direction") if isinstance(rr.details, dict) else None}
                if user_weights:
                    # 显式权重优先，regime 仅做展示不覆盖
                    logger.info(f"[Regime] 自动检测: {rr.regime.value} (confidence={rr.confidence:.2f}) "
                               f"macro={macro_info.get('action') if macro_info else 'n/a'} "
                               f"— 显式权重优先，不覆盖")
                else:
                    source_weights = get_regime_weights(rr.regime)
                    tech_preset = get_regime_tech_preset(rr.regime)
                    logger.info(f"[Regime] 自动检测: {rr.regime.value} (confidence={rr.confidence:.2f}) "
                               f"macro={macro_info.get('action') if macro_info else 'n/a'} "
                               f"→ tech_preset={tech_preset}")
            else:
                try:
                    rg = Regime(regime)
                    if not user_weights:
                        source_weights = get_regime_weights(rg)
                        tech_preset = get_regime_tech_preset(rg)
                    regime_result = {"regime": rg.value, "confidence": 1.0,
                                     "sub_scores": {}, "as_of": str(eval_date or "")}
                    logger.info(f"[Regime] 手动指定: {rg.value} → tech_preset={tech_preset}")
                except ValueError:
                    logger.warning(f"[Regime] 无效状态: {regime}")

        # ── 方向精选: 先确定活跃信号源集合（后续ICIR在此集合内计算）──
        _direction_applied = None
        if regime and not user_weights:
            try:
                from core.domain.regime.weights import get_regime_direction, filter_weights_by_direction
                from core.domain.regime.classifier import Regime
                _vol_dir = (market_rr.details.get("vol_direction")
                            if use_vol_switch and regime == "auto" and market_rr
                            and isinstance(market_rr.details, dict) else None)
                if _vol_dir:
                    _direction = _vol_dir
                else:
                    _dr_regime = market_rr.regime if regime == "auto" and market_rr else Regime(regime)
                    _direction = get_regime_direction(_dr_regime)
                if _direction and _direction != "neutral":
                    _before = {k: v for k, v in source_weights.items() if v > 0}
                    source_weights = filter_weights_by_direction(source_weights, _direction)
                    _dropped = [k for k in _before if source_weights.get(k, 0) == 0]
                    _direction_applied = _direction
                    if _dropped:
                        logger.info(f"[Direction] {_direction}: 停用不适配源 {_dropped}, "
                                    f"保留 {[k for k in _before if k not in _dropped]}")
            except Exception as e:
                logger.warning(f"[Direction] 过滤失败: {e}")

        _model_routing = None
        if state_models:
            from core.domain.regime.model_routing import validate_state_models, resolve_state_key
            state_models = validate_state_models(state_models, use_vol_switch)
            _routing_regime = (market_rr.regime.value if market_rr else
                               (regime_result or {}).get("regime"))
            _routing_direction = (market_rr.details.get("vol_direction")
                                  if use_vol_switch and market_rr and isinstance(market_rr.details, dict)
                                  else _direction_applied)
            if use_vol_switch and not _routing_direction:
                raise ValueError("波动率分类器未产生可用状态，拒绝默认选择模型")
            if _routing_direction == "neutral":
                # 三档切换: 中波死亡区 → 空仓, 不使用任何模型
                self._vol_switch_empty = True
                qlib_model = None
                _model_routing = {
                    "mode": "volatility",
                    "state": "neutral",
                    "regime": _routing_regime,
                    "direction": "neutral",
                    "experiment_id": None,
                    "run_id": None,
                    "empty": True,
                }
                logger.info(f"[ModelRouting] 中波死亡区 → 空仓(不使用模型)")
            else:
                if not _routing_direction and _routing_regime:
                    from core.domain.regime.weights import get_regime_direction
                    _routing_direction = get_regime_direction(_routing_regime)
                if not _routing_regime or not _routing_direction:
                    raise ValueError("状态模型映射需要先获取市场状态")
                _state_key = resolve_state_key(_routing_regime, _routing_direction, use_vol_switch)
                qlib_model = state_models[_state_key]
                _model_routing = {
                    "mode": "volatility" if use_vol_switch else "regime",
                    "state": _state_key,
                    "regime": _routing_regime,
                    "direction": _routing_direction,
                    "experiment_id": qlib_model[0],
                    "run_id": qlib_model[1],
                }
                logger.info(f"[ModelRouting] {_state_key} -> {qlib_model[0]}/{qlib_model[1]}")

        # ── 因子池门控: 因子池中标记为 inactive 的信号源权重归零 ──
        if not user_weights:
            try:
                from core.domain.factors.registry import get_pool
                pool_state = get_pool().state.get("traditional", {})
                _pool_disabled = []
                for sig_name in ("tech", "qlib", "sentiment", "fundamental", "flow",
                                 "north_flow", "size_turnover", "momentum", "volatility"):
                    entry = pool_state.get(sig_name, {})
                    if not entry.get("active", True) and source_weights.get(sig_name, 0) > 0:
                        source_weights[sig_name] = 0.0
                        _pool_disabled.append(sig_name)
                if _pool_disabled:
                    total = sum(source_weights.values())
                    if total > 0:
                        source_weights = {k: round(v / total, 3) for k, v in source_weights.items()}
                    logger.info(f"[FactorPool] 因子池已禁用: {_pool_disabled}, "
                                f"剩余活跃源: {[k for k,v in source_weights.items() if v>0]}")
            except Exception as e:
                logger.warning(f"[FactorPool] 门控检查失败: {e}")

        # ── ICIR 自适应权重（在方向过滤后的活跃源集合内计算）──
        if weight_method == "icir":
            try:
                from core.domain.factors.registry import get_pool
                icir_full = get_pool().get_icir_weights(as_of=eval_date)
                if icir_full:
                    if _direction_applied:
                        # 只在方向过滤允许的活跃源内计算ICIR
                        active = {k for k, v in source_weights.items() if v > 0}
                        icir_active = {k: v for k, v in icir_full.items() if k in active}
                        total = sum(icir_active.values())
                        if total > 0:
                            icir_active = {k: round(v / total, 3) for k, v in icir_active.items()}
                        source_weights = icir_active
                        logger.info(f"[ICIR] 方向约束({_direction_applied})下自适应权重: "
                                    f"{len(icir_active)}/{len(icir_full)} 个信号源")
                    else:
                        source_weights = icir_full
                        logger.info(f"[ICIR] 自适应权重: {len(icir_full)} 个信号源")
            except Exception as e:
                logger.warning(f"[ICIR] 权重计算失败，回退固定权重: {e}")

        _style_info = None
        if ranking_mode != "qlib_primary":
            try:
                from core.domain.timing.breadth import get_rsrs_momentum_scores
                rsrs = get_rsrs_momentum_scores(as_of=eval_date)
                valid = {k: v for k, v in rsrs.items() if v.get("score") is not None}
                if valid:
                    dominant = max(valid.items(), key=lambda x: x[1]["score"])
                    name, info = dominant
                    adjustments = {}
                    if name in ("上证50", "沪深300"):
                        st = source_weights.get("size_turnover", 0)
                        if st > 0:
                            source_weights["size_turnover"] = st * 0.5
                            adjustments["size_turnover"] = 0.5
                        mom = source_weights.get("momentum", 0)
                        if mom > 0:
                            source_weights["momentum"] = mom * 1.2
                            adjustments["momentum"] = 1.2
                        fund = source_weights.get("fundamental", 0)
                        if fund > 0:
                            source_weights["fundamental"] = fund * 1.15
                            adjustments["fundamental"] = 1.15
                        _style_info = {"style": "large_cap", "dominant_index": name,
                                       "score": info["score"], "adjustments": adjustments}
                    elif name in ("中证500", "国证2000"):
                        st = source_weights.get("size_turnover", 0)
                        if st > 0:
                            source_weights["size_turnover"] = st * 1.5
                            adjustments["size_turnover"] = 1.5
                        mom = source_weights.get("momentum", 0)
                        if mom > 0:
                            source_weights["momentum"] = mom * 0.8
                            adjustments["momentum"] = 0.8
                        fund = source_weights.get("fundamental", 0)
                        if fund > 0:
                            source_weights["fundamental"] = fund * 0.85
                            adjustments["fundamental"] = 0.85
                        _style_info = {"style": "small_cap", "dominant_index": name,
                                       "score": info["score"], "adjustments": adjustments}
                    if adjustments:
                        logger.info(f"[RSRS] {_style_info['style']}: {name}得分={info['score']:.4f} "
                                   f"→ 调节 {adjustments}")
            except Exception as e:
                logger.warning(f"[RSRS] 风格评分失败: {e}")

        tech_w = 0.0 if skip_tech else source_weights.get("tech", 0.25)
        qlib_w = 0.0 if skip_qlib else source_weights.get("qlib", 0.20)
        sent_w = 0.0 if skip_sent else source_weights.get("sentiment", 0.05)
        fundamental_w = 0.0 if skip_fundamental else source_weights.get("fundamental", 0.10)
        flow_w = 0.0 if skip_flow else source_weights.get("flow", 0.05)
        north_flow_w = 0.0 if skip_north_flow else source_weights.get("north_flow", 0.10)
        size_turnover_w = 0.0 if skip_size_turnover else source_weights.get("size_turnover", 0.10)
        momentum_w = 0.0 if skip_momentum else source_weights.get("momentum", 0.10)
        volatility_w = 0.0 if skip_volatility else source_weights.get("volatility", 0.05)
        ht_factors_w = 0.0 if skip_ht else source_weights.get("ht_factors", 0.10)
        wb_industry_w = 0.0 if skip_wb_industry else source_weights.get("wb_industry", 0.05)
        wb_divergence_w = 0.0 if skip_wb_divergence else source_weights.get("wb_divergence", 0.05)
        wb_moneyflow_w = 0.0 if skip_wb_moneyflow else source_weights.get("wb_moneyflow", 0.05)
        illiq_w = 0.0 if skip_illiq else source_weights.get("illiq", 0.02)
        overnight_gap_w = 0.0 if skip_overnight_gap else source_weights.get("overnight_gap", 0.02)

        # ── 配置驱动的信号源停用: BUY_SIDE_L0_DISABLED_SOURCES=illiq,overnight_gap,flow ──
        _disabled_raw = getattr(config, "BUY_SIDE_L0_DISABLED_SOURCES", "")
        _disabled_set = {s.strip() for s in _disabled_raw.split(",") if s.strip()} if _disabled_raw else set()
        if _disabled_set and not user_weights:
            if "tech" in _disabled_set:           skip_tech, tech_w = True, 0.0
            if "qlib" in _disabled_set:           skip_qlib, qlib_w = True, 0.0
            if "sentiment" in _disabled_set:      skip_sent, sent_w = True, 0.0
            if "fundamental" in _disabled_set:    skip_fundamental, fundamental_w = True, 0.0
            if "flow" in _disabled_set:           skip_flow, flow_w = True, 0.0
            if "north_flow" in _disabled_set:     skip_north_flow, north_flow_w = True, 0.0
            if "size_turnover" in _disabled_set:  skip_size_turnover, size_turnover_w = True, 0.0
            if "momentum" in _disabled_set:       skip_momentum, momentum_w = True, 0.0
            if "volatility" in _disabled_set:     skip_volatility, volatility_w = True, 0.0
            if "illiq" in _disabled_set:          skip_illiq, illiq_w = True, 0.0
            if "overnight_gap" in _disabled_set:  skip_overnight_gap, overnight_gap_w = True, 0.0
            if "ht_factors" in _disabled_set:     skip_ht, ht_factors_w = True, 0.0
            if "wb_industry" in _disabled_set:    skip_wb_industry, wb_industry_w = True, 0.0
            if "wb_divergence" in _disabled_set:  skip_wb_divergence, wb_divergence_w = True, 0.0
            if "wb_moneyflow" in _disabled_set:   skip_wb_moneyflow, wb_moneyflow_w = True, 0.0
            logger.info(f"[L0] 配置停用信号源 ({len(_disabled_set)}): {_disabled_set}")

        total_w = (tech_w + qlib_w + sent_w + fundamental_w + flow_w +
                   north_flow_w + size_turnover_w + momentum_w + volatility_w +
                   illiq_w + overnight_gap_w +
                   wb_industry_w + wb_divergence_w + wb_moneyflow_w)
        if total_w > 0:
            tech_w /= total_w; qlib_w /= total_w; sent_w /= total_w
            fundamental_w /= total_w; flow_w /= total_w
            north_flow_w /= total_w; size_turnover_w /= total_w
            momentum_w /= total_w; volatility_w /= total_w
            illiq_w /= total_w; overnight_gap_w /= total_w
            wb_industry_w /= total_w; wb_divergence_w /= total_w; wb_moneyflow_w /= total_w
        else:
            tech_w = 1.0 / 8; qlib_w = 1.0 / 8; sent_w = 1.0 / 8
            fundamental_w = 1.0 / 8; flow_w = 1.0 / 8
            illiq_w = 1.0 / 8; overnight_gap_w = 1.0 / 8

        # ── 宏观: 因子适配度 → 调节信号源权重 ──
        if macro_bias:
            try:
                from core.domain.macro.analyzer import MacroAnalyzer
                sig_weights = {
                    "momentum": momentum_w, "volatility": volatility_w,
                    "fundamental": fundamental_w, "flow": flow_w,
                    "size_turnover": size_turnover_w, "illiq": illiq_w,
                    "overnight_gap": overnight_gap_w,
                    "wb_industry": wb_industry_w, "wb_divergence": wb_divergence_w,
                    "wb_moneyflow": wb_moneyflow_w,
                }
                active_sigs = [k for k, v in sig_weights.items() if v > 0]
                if active_sigs:
                    m = MacroAnalyzer()
                    fitness = m.get_factor_macro_fitness(active_sigs, as_of=eval_date)
                    for sig in active_sigs:
                        fit = fitness.get(sig, {}).get("fitness", 0.5)
                        sig_weights[sig] *= fit
                    # 重新归一化到原总权重
                    orig_tw = sum([momentum_w, volatility_w, fundamental_w, flow_w,
                                   size_turnover_w, illiq_w, overnight_gap_w,
                                   wb_industry_w, wb_divergence_w, wb_moneyflow_w])
                    new_tw = sum(sig_weights.values())
                    if new_tw > 0 and orig_tw > 0:
                        scale = orig_tw / new_tw
                        for sig in active_sigs:
                            sig_weights[sig] *= scale
                    momentum_w, volatility_w, fundamental_w, flow_w = \
                        sig_weights["momentum"], sig_weights["volatility"], \
                        sig_weights["fundamental"], sig_weights["flow"]
                    size_turnover_w, illiq_w, overnight_gap_w = \
                        sig_weights["size_turnover"], sig_weights["illiq"], \
                        sig_weights["overnight_gap"]
                    wb_industry_w, wb_divergence_w, wb_moneyflow_w = \
                        sig_weights["wb_industry"], sig_weights["wb_divergence"], \
                        sig_weights["wb_moneyflow"]
                    logger.info(f"[Macro] 因子适配度调节完成: {dict((k, f'{v:.3f}') for k,v in sig_weights.items())}")
            except Exception as e:
                logger.warning(f"[Macro] 因子适配度调节失败: {e}")

        # ── 宏观数据集成: 风险调节 + 行业偏好 ──
        macro_risk_score = 0.0
        macro_sector_bias = {}
        if macro_bias:
            try:
                from core.domain.macro.analyzer import MacroAnalyzer
                m = MacroAnalyzer()
                risk = m.calculate_macro_risk_score(as_of=eval_date)
                macro_risk_score = risk.get("score", 0)
                if macro_risk_score > 0:
                    logger.info(f"[Macro] 风险分={macro_risk_score:.2f} level={risk.get('level')} "
                                f"signals={risk.get('signals', [])}")
                sector_signals = m.get_sector_rotation_signals(as_of=eval_date)
                for sig in sector_signals:
                    for sec in sig.get("sectors", []):
                        macro_sector_bias[sec] = 0.15  # 宏观偏多板块 +0.15 L0 加分
                if macro_sector_bias:
                    logger.info(f"[Macro] 行业偏好: {list(macro_sector_bias.keys())[:8]}... "
                                f"({len(macro_sector_bias)} 个板块)")
            except Exception as e:
                logger.warning(f"[Macro] 宏观数据获取失败: {e}")

        _etf_flow_bias = {}
        if ranking_mode != "qlib_primary":
            try:
                from core.data.sources.etf_flow import get_etf_flow_by_industry
                etf_flows = get_etf_flow_by_industry(as_of=str(eval_date) if eval_date else None, days=5)
                for industry, inflow in etf_flows.items():
                    if inflow > 1e9:
                        _etf_flow_bias[industry] = 0.15
                    elif inflow > 5e8:
                        _etf_flow_bias[industry] = 0.10
                    elif inflow > 1e8:
                        _etf_flow_bias[industry] = 0.06
                if _etf_flow_bias:
                    for ind, bonus in _etf_flow_bias.items():
                        macro_sector_bias[ind] = max(macro_sector_bias.get(ind, 0), bonus)
                    logger.info(f"[ETF Flow] 行业偏好: {list(_etf_flow_bias.keys())[:8]}... "
                               f"({len(_etf_flow_bias)} 个板块)")
            except Exception as e:
                logger.warning(f"[ETF Flow] 获取失败: {e}")

        tech_indicators = self.TECH_PRESETS.get(tech_preset, self.TECH_PRESETS["all"])

        result = {"job_id": job_id, "status": "running", "stages": {},
                  "progress": {"stage": "init", "current": 0, "total": 0}}
        if _model_routing:
            result["model_routing"] = _model_routing

        def _update_progress(stage, current, total):
            result["progress"] = {"stage": stage, "current": current, "total": total}

        with self._lock:
            self._jobs[job_id] = result
            # 防止内存泄漏: 保留最近 100 条, 超出时删最旧
            if len(self._jobs) > 100:
                oldest = sorted(self._jobs.keys(), key=lambda k: self._jobs[k].get("started_at", ""))[:len(self._jobs) - 100]
                for k in oldest:
                    del self._jobs[k]

        self._df_cache.clear()

        if self._vol_switch_empty:
            result.update({
                "status": "done", "vol_switch_empty": True,
                "candidates": [], "recommended": [], "ranking_mode": ranking_mode,
                "regime": regime_result, "direction": "neutral",
                "message": "中波区间，当前保持空仓，不生成买入候选。",
                "stages": {stage: {"skipped": True} for stage in ("l0", "l1", "l2")},
                "progress": {"stage": "done", "current": 0, "total": 0},
            })
            return result

        try:
            # ── L0: 廉价快筛 ──
            if skip_l0:
                result["stages"]["l0"] = {"skipped": True}
                codes = _get_all_stock_codes()
                candidates = [{"code": c, "name": c, "l0_score": 0.5,
                               "tech_score": 0, "qlib_score": 0,
                               "sentiment_score": 0, "close": 0, "volume": 0}
                              for c in codes[:max(200, top_n * 4)]]
                logger.info(f"[L0] 已跳过, 使用 {len(candidates)} 只候选")
            else:
                logger.info(f"[L0] 开始廉价快筛...")
                _update_progress("l0", 0, 0)
                l0_passed = self._l0_filter(pool, tech_w=tech_w, qlib_w=qlib_w, sent_w=sent_w,
                                           fundamental_w=fundamental_w, flow_w=flow_w,
                                           north_flow_w=north_flow_w, size_turnover_w=size_turnover_w,
                                           momentum_w=momentum_w, volatility_w=volatility_w,
                                           illiq_w=illiq_w, overnight_gap_w=overnight_gap_w,
                                           wb_industry_w=wb_industry_w, wb_divergence_w=wb_divergence_w,
                                           wb_moneyflow_w=wb_moneyflow_w,
                                           skip_tech=skip_tech, skip_qlib=skip_qlib,
                                           skip_sent=skip_sent, skip_fundamental=skip_fundamental,
                                           skip_flow=skip_flow, skip_north_flow=skip_north_flow,
                                           skip_size_turnover=skip_size_turnover,
                                           skip_momentum=skip_momentum, skip_volatility=skip_volatility,
                                           macro_risk_score=macro_risk_score, macro_sector_bias=macro_sector_bias,
                                           skip_illiq=skip_illiq, skip_overnight_gap=skip_overnight_gap,
                                           skip_ht=skip_ht, ht_factors_w=ht_factors_w, ht_key=ht_key,
                                           skip_wb_industry=skip_wb_industry,
                                           skip_wb_divergence=skip_wb_divergence,
                                           skip_wb_moneyflow=skip_wb_moneyflow,
                                           wb_industry_n=wb_industry_n,
                                           wb_divergence_n=wb_divergence_n,
                                           wb_moneyflow_n=wb_moneyflow_n,
                                           momentum_n=momentum_n, volatility_n=volatility_n,
                                           illiq_n=illiq_n, overnight_gap_n=overnight_gap_n,
                                           tech_indicators=tech_indicators,
                                           qlib_model=qlib_model,
                                           neutralize_qlib=neutralize_qlib,
                                           sector_fusion=sector_fusion, market_rr=market_rr,
                                           direction=_direction_applied,
                                           use_vol_switch=use_vol_switch,
                                           etf_flow_bias=_etf_flow_bias,
                                           progress_cb=lambda cur, tot: _update_progress("l0", cur, tot),
                                           hot_sector_count=hot_sector_count,
                                           custom_sectors=custom_sectors,
                                           code_prefixes=code_prefixes,
                                           concept_sectors=concept_sectors,
                                           eval_date=eval_date,
                                           snapshot=snapshot,
                                           factor_overlay=factor_overlay,
                                           top_n=top_n,
                                           ranking_mode=ranking_mode,
                                           decision_day_filter=decision_day_filter)
                result["stages"]["l0"] = {
                    "input_count": l0_passed.get("total_input", 0),
                    "passed": len(l0_passed.get("candidates", [])),
                    "excluded_reasons": l0_passed.get("excluded_reasons", {}),
                    "weights": l0_passed.get("weights", {}),
                    "sent_coverage": l0_passed.get("sent_coverage"),
                }
                if l0_passed.get("ranking_mode"):
                    result["ranking_mode"] = l0_passed["ranking_mode"]
                    result["auxiliary_only"] = l0_passed.get("auxiliary_only", [])
                    result["auxiliary_coverage"] = l0_passed.get("auxiliary_coverage", {})
                    result["qlib_coverage"] = l0_passed.get("qlib_coverage", 0)
                if l0_passed.get("score_processing"):
                    result["score_processing"] = l0_passed["score_processing"]
                candidates = l0_passed["candidates"]
                logger.info(f"[L0] 通过 {len(candidates)} 只")

            if not candidates:
                result["status"] = "done"
                result["recommended"] = []
                result["candidates"] = []
                return result

            # ── L1: 策略精排 ──
            if snapshot and not skip_l1:
                # 用 snapshot 预填充 df_cache，避免 L1 重复 HTTP 取数据
                for code, entry in snapshot.items():
                    self._df_cache[code] = entry[0]
            if skip_l1:
                result["stages"]["l1"] = {"skipped": True}
                ranked_all = candidates
                top_candidates = candidates[:top_n]
                for i, c in enumerate(top_candidates):
                    c["rank"] = i + 1
                    if "l1_score" not in c:
                        key = "qlib_score" if ranking_mode == "qlib_primary" else "l0_score"
                        c["l1_score"] = c.get(key, 0)
                logger.info(f"[L1] 已跳过, Top-{len(top_candidates)}")
            else:
                logger.info(f"[L1] 开始策略精排 ({len(candidates)} 只)...")
                _update_progress("l1", 0, min(len(candidates), 200))
                l1_ranked = self._l1_rank(candidates,
                                        progress_cb=lambda cur, tot: _update_progress("l1", cur, tot))
                result["stages"]["l1"] = {"ranked": min(len(l1_ranked), top_n)}
                ranked_all = l1_ranked
                top_candidates = l1_ranked[:top_n]
                for i, c in enumerate(top_candidates):
                    c["rank"] = i + 1
                logger.info(f"[L1] 完成, Top-{len(top_candidates)}")

            # ── 附加近5日主力净流入金额(万元)，供前端展示流入多少 ──
            if top_candidates and not skip_backfill and ranking_mode != "qlib_primary":
                try:
                    from core.data import Database
                    t_codes = [c["code"] for c in top_candidates]
                    ph = ",".join(["%s"] * len(t_codes))
                    date_cond = "AND trade_date <= %s" if eval_date else ""
                    rows = Database.fetchall(
                        f"SELECT code, SUM(main_net_flow) AS net5 FROM ("
                        f"SELECT code, main_net_flow, "
                        f"ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) rn "
                        f"FROM capital_flow_daily WHERE code IN ({ph}) {date_cond}"
                        f") t WHERE rn <= 5 GROUP BY code",
                        tuple(t_codes) + ((eval_date,) if eval_date else ()),
                    )
                    net_map = {r["code"]: float(r["net5"] or 0) for r in rows}
                    for c in top_candidates:
                        if c["code"] in net_map:
                            c["flow_net_5d"] = round(net_map[c["code"]], 1)
                except Exception as e:
                    logger.warning(f"[L1] 净流入金额附加失败: {e}")

            # ── 板块融合后处理: 归因/双路分/保送/单行业集中度 ──
            fusion_info = getattr(self, "_last_fusion_info", None)
            if fusion_info is not None and top_candidates:
                try:
                    ind_map_raw = self.fetcher.get_industry_map() or {}
                    attr = fusion_info.get("attr", {})
                    raw_scores = fusion_info.get("_raw_scores", {})
                    for c in top_candidates:
                        if c["code"] in attr:
                            c["qlib_attr"] = attr[c["code"]]
                            c["sector_divergent"] = True
                        if c["code"] in raw_scores:
                            c["qlib_score_raw"] = round(raw_scores[c["code"]], 3)
                            c["qlib_score_before_fusion"] = round(raw_scores[c["code"]], 3)

                    if ranking_mode == "qlib_primary":
                        escort = {item["code"]: item["direction"] for item in fusion_info.get("escort", [])}
                        for c in top_candidates:
                            if c["code"] in escort:
                                c["model_strong"] = escort[c["code"]]
                    else:
                        cap = max(2, int(top_n * 0.3))
                        ind_count = {}
                        kept, overflow = [], []
                        for c in top_candidates:
                            value = ind_map_raw.get(c["code"])
                            industry = value.get("industry") if isinstance(value, dict) else None
                            if industry:
                                ind_count[industry] = ind_count.get(industry, 0) + 1
                                if ind_count[industry] > cap:
                                    overflow.append(c)
                                    continue
                            kept.append(c)
                        if overflow:
                            top_codes = {c["code"] for c in top_candidates}
                            for c in ranked_all:
                                if len(kept) >= len(top_candidates):
                                    break
                                if c["code"] in top_codes:
                                    continue
                                value = ind_map_raw.get(c["code"])
                                industry = value.get("industry") if isinstance(value, dict) else None
                                if industry and ind_count.get(industry, 0) >= cap:
                                    continue
                                if industry:
                                    ind_count[industry] = ind_count.get(industry, 0) + 1
                                kept.append(c)
                            top_candidates = kept
                            for i, c in enumerate(top_candidates):
                                c["rank"] = i + 1

                        existing = {c["code"] for c in top_candidates}
                        cand_by_code = {c["code"]: c for c in ranked_all}
                        replaced = 0
                        for item in fusion_info.get("escort", []):
                            if item["code"] in existing:
                                source = next((c for c in top_candidates if c["code"] == item["code"]), None)
                                if source:
                                    source["model_strong"] = item["direction"]
                                continue
                            if replaced >= 10:
                                continue
                            source = cand_by_code.get(item["code"])
                            if source is None:
                                continue
                            source["model_strong"] = item["direction"]
                            source["escort"] = True
                            if source["code"] in attr:
                                source["qlib_attr"] = attr[source["code"]]
                                source["sector_divergent"] = True
                            swap_pos = len(top_candidates) - 1 - replaced
                            while swap_pos >= 0:
                                victim = top_candidates[swap_pos]
                                if not victim.get("escort") and not victim.get("sector_divergent"):
                                    source["rank"] = victim.get("rank", swap_pos + 1)
                                    top_candidates[swap_pos] = source
                                    existing.add(item["code"])
                                    existing.discard(victim["code"])
                                    replaced += 1
                                    break
                                swap_pos -= 1
                except Exception as e:
                    logger.warning(f"[SectorFusion] 后处理失败: {e}")

            # ── L2: AI 辩论 (仅前 debate_top 只) ──
            recommended = []
            if skip_l2 or skip_debate:
                result["stages"]["l2"] = {"skipped": True}
                # L2跳过时，把top_candidates直接作为推荐(无AI辩论增强)
                recommended = top_candidates[:debate_top]
                for rec in recommended:
                    rec["final_score"] = rec.get("l1_score", rec.get("l0_score", 0))
                    rec["confidence"] = None
                    rec["recommendation_level"] = "量化排名"
            elif top_candidates:
                debate_codes = top_candidates[:min(debate_top, len(top_candidates))]
                logger.info(f"[L2] AI 辩论: {len(debate_codes)} 只...")
                _update_progress("l2", 0, len(debate_codes))
                recommended = self._l2_debate(debate_codes, debate_agents=debate_agents,
                                             progress_cb=lambda cur, tot: _update_progress("l2", cur, tot),
                                             eval_date=eval_date)
                result["stages"]["l2"] = {"debated": len(debate_codes),
                                          "recommended": len(recommended)}
                for rec in recommended:
                    rec["rank"] = rec.get("rank", 0)
            else:
                result["stages"]["l2"] = {"skipped": True}

            result["status"] = "done"
            result["recommended"] = recommended
            result["candidates"] = top_candidates
            if fusion_info is not None:
                result["sector_fusion"] = {"divergent": fusion_info.get("divergent", {}),
                                           "models": fusion_info.get("models", {})}
            if regime_result:
                result["regime"] = regime_result
            if _direction_applied:
                result["direction"] = _direction_applied
            if _style_info:
                result["style"] = _style_info
            if _etf_flow_bias:
                result["etf_flow_bias"] = _etf_flow_bias

            # ── Barra 风格因子暴露检查 ──
            if getattr(config, "BARRA_RISK_ENABLED", True) and top_candidates and not skip_l0 and not skip_backfill:
                try:
                    from core.domain.portfolio.risk import (compute_factor_exposures,
                                                  compute_portfolio_exposure,
                                                  check_risk, format_risk_report)
                    top_codes = [c["code"] for c in top_candidates[:top_n]]
                    top_weights = {}
                    for i, c in enumerate(top_candidates[:top_n]):
                        w = c.get("l1_score", c.get("l0_score", 0))
                        top_weights[c["code"]] = max(w, 0.01)
                    w_sum = sum(top_weights.values())
                    if w_sum > 0:
                        top_weights = {k: v / w_sum for k, v in top_weights.items()}

                    exposures = compute_factor_exposures(top_codes, snapshot or {})
                    if exposures:
                        pf_exp = compute_portfolio_exposure(top_weights, exposures)
                        alerts = check_risk(pf_exp)
                        result["barra_risk"] = {
                            "exposures": pf_exp,
                            "alerts": {k: v for k, v in alerts.items() if v["level"] != "ok"},
                        }
                        report = format_risk_report(alerts)
                        if report:
                            logger.warning(report)
                            if getattr(config, "BARRA_RISK_NOTIFY", False):
                                try:
                                    from core.shared.infrastructure.notify import WeComNotifier
                                    WeComNotifier().send_alert(f"Barra因子暴露告警\n{report}")
                                except Exception:
                                    pass
                except Exception as e:
                    logger.debug(f"[Barra] 因子暴露检查跳过: {e}")

            _update_progress("done", 1, 1)

            # 回填股票名称（数据源的 K 线 DataFrame 不含 name 列）
            if not skip_backfill:
                self._backfill_names(result.get("candidates", []))
                self._backfill_names(result.get("recommended", []))

        except Exception as e:
            logger.exception(f"选股流程失败: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            _update_progress("error", 0, 0)

        return result

    def _backfill_names(self, candidates):
        """批量回填股票名称，跳过已有有效名称的条目"""
        if not candidates:
            return
        need_names = [c for c in candidates if not c.get("name")]
        if not need_names:
            return
        from core.data.fetcher import DataFetcher
        fetcher = DataFetcher()
        for c in need_names:
            try:
                name = fetcher.get_stock_name(c["code"])
                if name:
                    c["name"] = name
            except Exception as e:
                logger.debug(f"[Backfill] 名称回填失败 {c.get('code', '?')}: {e}")
    def _get_hot_sector_stocks(self, top_n: int = 5) -> tuple:
        """获取当前最活跃板块的成分股合集，返回 (codes, meta)
        优先通过 SectorScanner 获取成分股；若全部板块都拿不到成分股
        (如 EastMoney API 被墙)，则回退全A股并标注。"""
        from core.data.fetcher import DataFetcher as DF
        from core.domain.screening.stock_scanner import SectorScanner
        import akshare as ak
        fetcher = DF()
        scanner = SectorScanner()
        meta = {"sectors": {}, "total_in_pool": 0}
        try:
            perf = fetcher.get_sector_performance()
        except Exception:
            logger.warning("板块行情获取失败，回退全A股")
            return _get_all_stock_codes(), meta

        if perf is None or perf.empty:
            return _get_all_stock_codes(), meta

        if "涨跌幅" in perf.columns:
            perf = perf.copy()
            perf["abs_chg"] = perf["涨跌幅"].abs()
            perf = perf.sort_values("abs_chg", ascending=False)

        hot_names = perf["板块名称"].head(top_n).tolist()
        hot_chgs = perf["涨跌幅"].head(top_n).tolist() if "涨跌幅" in perf.columns else []

        # 通过 THS 板块摘要获取成分股数量 (不依赖被墙的 EastMoney)
        # 列顺序固定: 排名,板块名称,涨跌幅,总成交量,总成交额,换手率,上涨家数,下跌家数,...
        summary_count = {}
        try:
            summary = ak.stock_board_industry_summary_ths()
            if summary is not None and not summary.empty:
                cols = summary.columns.tolist()
                name_col = cols[1] if len(cols) > 1 else cols[0]
                up_col = cols[6] if len(cols) > 6 else "上涨家数"
                down_col = cols[7] if len(cols) > 7 else "下跌家数"
                for row in summary.itertuples(index=False):
                    sname = str(getattr(row, name_col, ''))
                    up = int(getattr(row, up_col, 0) or 0)
                    down = int(getattr(row, down_col, 0) or 0)
                    summary_count[sname] = up + down
        except Exception:
            pass

        all_stocks: set = set()
        sector_counts = {}
        all_zero = True
        for i, sector in enumerate(hot_names):
            chg = hot_chgs[i] if i < len(hot_chgs) else 0
            count = 0
            try:
                stocks = scanner.get_sector_stocks(sector)
                if stocks is not None and not stocks.empty:
                    code_col = "code" if "code" in stocks.columns else stocks.columns[0]
                    for raw in stocks[code_col]:
                        c = str(raw)
                        if "." in c:
                            c = c.split(".")[1]
                        c = c.strip()[:6]
                        if c.isdigit() and len(c) == 6:
                            all_stocks.add(c)
                            count += 1
            except Exception:
                pass
            if count > 0:
                all_zero = False
            # 若拿不到成分股，用 THS 摘要的数量
            actual_count = count if count > 0 else summary_count.get(sector, 0)
            sector_counts[sector] = {"涨跌幅": round(float(chg), 2), "成分股数": actual_count}

        # 所有热门板块都拿不到成分股 → 回退全A股
        if all_zero or not all_stocks:
            chg_info = ", ".join(f"{n}({c:+.2f}%)" for n, c in zip(hot_names, hot_chgs))
            logger.warning(f"[热门板块] Top-{top_n}: {chg_info} → 0只成分股，回退全A股")
            all_codes = _get_all_stock_codes()
            meta = {"sectors": sector_counts, "total_in_pool": len(all_codes), "fallback": True}
            return sorted(all_codes), meta

        chg_info = ", ".join(f"{n}({c:+.2f}%)" for n, c in zip(hot_names, hot_chgs))
        logger.info(f"[热门板块] Top-{top_n}: {chg_info} → 共 {len(all_stocks)} 只成分股")
        meta = {"sectors": sector_counts, "total_in_pool": len(all_stocks)}
        return sorted(all_stocks), meta

    def _compute_l0_matrix(self, snapshot, preloaded=None, as_of=None):
        """一次计算 snapshot 中所有股票的 13 个 L0 子分项。
        委托给 core.domain.screening.scoring.compute_l0_matrix (纯 L3 计算)。
        as_of: 决策日, 透传防止 DB 回退路径取未来数据。
        """
        return compute_l0_matrix(snapshot, preloaded, as_of=as_of)

    def _prefilter_by_turnover(self, codes, top_n, eval_date=None):
        if not codes or top_n <= 0:
            return codes
        from core.data import Database
        import datetime as _dt
        end_dt = eval_date or today_date()
        if hasattr(end_dt, 'strftime'):
            end_dt = end_dt.strftime("%Y-%m-%d")
        placeholders = ",".join(["%s"] * len(codes))
        rows = Database.fetchall(
            f"SELECT code, AVG(amount) as avg_amount FROM daily_quotes "
            f"WHERE code IN ({placeholders}) "
            f"AND trade_date >= DATE_SUB(%s, INTERVAL 30 DAY) "
            f"AND trade_date <= %s "
            f"GROUP BY code HAVING COUNT(*) >= 10 "
            f"ORDER BY avg_amount DESC LIMIT {int(top_n)}",
            codes + [end_dt, end_dt])
        turnover_data = {r["code"]: r["avg_amount"] for r in rows} if rows else {}
        return prefilter_by_turnover(codes, top_n, turnover_data)

    def _prefilter_by_index(self, codes, pool_spec, eval_date=None):
        """按指数成分预筛选—获取指数成分后委托纯函数过滤。"""
        from qlib.data import D
        valid = set()
        for name in pool_spec.split("+"):
            name = name.strip()
            try:
                inst = D.list_instruments(D.instruments(name), as_list=True)
                bare = {i[2:] for i in inst if len(i) >= 3 and i[:2] in ("sh", "sz")}
                valid.update(bare)
                logger.info(f"[L0] 预筛选: {name} → {len(bare)} 只成分股")
            except Exception as e:
                logger.warning(f"[L0] 预筛选: 获取 {name} 成分股失败: {e}")
        result = prefilter_by_index(codes, valid)
        logger.info(f"[L0] 预筛选: {len(codes)} → {len(result)} 只 (指数成分: {pool_spec})")
        return result

    def _select_qlib_top_candidates(self, codes, qlib_scores, top_n, eval_date=None,
                                    snapshot=None, skip_sent=False,
                                    skip_fundamental=False, skip_flow=False,
                                    skip_north_flow=False, progress_cb=None,
                                    hot_sector_meta=None, decision_day_filter=True):
        """按 Qlib 排序取可交易 Top-N，再为入选股票补充辅助字段。"""
        ranked_codes = []
        for code in codes:
            try:
                if code in qlib_scores and np.isfinite(float(qlib_scores[code])):
                    ranked_codes.append(code)
            except (TypeError, ValueError):
                continue
        ranked_codes.sort(key=lambda code: float(qlib_scores[code]), reverse=True)
        if not ranked_codes:
            raise RuntimeError("Qlib预测未覆盖当前股票池，已停止选股；辅助数据不会替代排序")

        score_date = eval_date or today_date()
        if isinstance(score_date, str):
            score_date = datetime.date.fromisoformat(score_date[:10])
        if eval_date is not None and decision_day_filter:
            self._get_adj_latest_map(ranked_codes)
        listing_dates = self._get_listing_dates_map(ranked_codes) if decision_day_filter else {}
        end_date_str = score_date.strftime("%Y%m%d")
        excluded = {"st": 0, "suspended": 0, "limit_up": 0, "low_liquidity": 0,
                    "low_price": 0, "new_listing": 0, "identity_unknown": 0, "data_error": 0,
                    "below_threshold": 0}
        candidates = []
        industry_map = {}
        try:
            industry_map = self.fetcher.get_industry_map() or {}
        except Exception as e:
            logger.warning(f"[QlibPrimary] 行业和名称映射加载失败: {e}")

        tls = threading.local()

        def _load_one(code):
            try:
                if snapshot is not None:
                    entry = snapshot.get(code)
                    if entry is None:
                        return code, None, "data_error"
                    df, name, close, volume = entry
                else:
                    if not hasattr(tls, "fetcher"):
                        from core.data.fetcher import DataFetcher
                        tls.fetcher = DataFetcher()
                    df = tls.fetcher.get_stock_daily(code, end_date=end_date_str)
                    if df is None or df.empty:
                        return code, None, "data_error"
                    info = industry_map.get(code, {})
                    name = info.get("name", "") if isinstance(info, dict) else ""
                    name = str(name)

                if df is None or df.empty:
                    return code, None, "data_error"
                name = str(name or "")
                min_turnover = getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 1e7)
                assessment = evaluate_decision_day_filter(
                    code, df, score_date, name=name, min_turnover=min_turnover,
                    latest_adj_factor=_adj_latest_cache.get(str(code).strip()),
                    limit_pct=limit_pct_for_code(code),
                    listing_date=listing_dates.get(code),
                )
                reason = assessment["reason"]
                if reason == "data_error" or (decision_day_filter and reason):
                    return code, None, reason
                close = assessment["close"]
                volume = assessment["volume"]

                impact_cost = 0.0
                if getattr(config, "BUY_SIDE_IMPACT_ENABLED", True):
                    impact_cost = estimate_impact_cost(close, volume)
                qlib_score = round(float(qlib_scores[code]), 6)
                candidate = {
                    "code": code, "name": name,
                    "qlib_score": qlib_score,
                    "l0_score": qlib_score,
                    "impact_cost": round(impact_cost, 6),
                    "fundamental_score": 0.0,
                    "flow_score": 0.0,
                    "north_flow_score": 0.0,
                    "sentiment_score": 0.0,
                    "etf_flow_score": 0.0,
                    "tech_score": 0.0,
                    "size_turnover_score": 0.0,
                    "momentum_score": 0.0,
                    "volatility_score": 0.0,
                    "illiq_score": 0.0,
                    "gap_score": 0.0,
                    "wb_industry_score": 0.0,
                    "wb_divergence_score": 0.0,
                    "wb_moneyflow_score": 0.0,
                    "custom_factor_score": 0.0,
                    "close": close, "volume": volume,
                }
                score_result = getattr(self, "_last_score_result", None) or {}
                score_row = score_result.get("ledger", {}).get(code)
                if score_row:
                    candidate.update({
                        "model_input_score": score_row.get("model_input_score"),
                        "model_input_rank": score_row.get("model_input_rank"),
                        "model_input_semantics": score_row.get("model_input_semantics"),
                        "raw_model_score": score_row.get("raw_model_score"),
                        "raw_rank": score_row.get("raw_rank"),
                        "neutralized_score": score_row.get("neutralized_score"),
                        "neutralized_rank": score_row.get("neutralized_rank"),
                        "rank_percentile": score_row.get("rank_percentile"),
                        "final_rank": score_row.get("final_rank"),
                        "score_trace": score_row.get("score_trace", {}),
                    })
                return code, (candidate, df), None
            except Exception:
                return code, None, "data_error"

        batch_size = max(50, top_n)
        examined = 0
        if progress_cb:
            progress_cb(0, len(ranked_codes))
        for start in range(0, len(ranked_codes), batch_size):
            batch = ranked_codes[start:start + batch_size]
            loaded = {}
            with thread_pool.default.borrow() as executor:
                futures = [executor.submit(_load_one, code) for code in batch]
                for future in as_completed(futures):
                    code, value, reason = future.result()
                    loaded[code] = (value, reason)
            for code in batch:
                examined += 1
                value, reason = loaded.get(code, (None, "data_error"))
                if reason:
                    excluded[reason] += 1
                elif value:
                    candidate, df = value
                    candidates.append(candidate)
                    self._df_cache[code] = df
                    if len(candidates) >= top_n:
                        break
            if progress_cb:
                progress_cb(examined, len(ranked_codes))
            if len(candidates) >= top_n:
                break

        selected_codes = [candidate["code"] for candidate in candidates]
        financial_cache = {}
        flow_cache = {}
        north_flow_data = {}
        sentiment_data = {}
        etf_flows = {}
        if selected_codes:
            try:
                from core.data import Database
                placeholders = ",".join(["%s"] * len(selected_codes))
                if not skip_fundamental:
                    rows = Database.fetchall(
                        f"SELECT f.code, f.report_date, GREATEST(f.ann_date, DATE(v.source_observed_at)) AS avail_date, "
                        f"f.roe, f.net_profit_yoy, f.revenue_yoy, f.debt_to_assets "
                        f"FROM fundamental_data f INNER JOIN ("
                        f"SELECT code, report_date, ann_date, MAX(source_observed_at) AS source_observed_at "
                        f"FROM fundamental_data_versions WHERE source_name='tushare_fina_indicator' "
                        f"AND version_kind='provider_observation' AND ann_date IS NOT NULL "
                        f"AND code IN ({placeholders}) "
                        f"GROUP BY code, report_date, ann_date"
                        f") v ON v.code=f.code AND v.report_date=f.report_date AND v.ann_date=f.ann_date "
                        f"WHERE f.code IN ({placeholders}) AND f.ann_date IS NOT NULL "
                        f"ORDER BY f.code, f.report_date",
                        tuple(selected_codes) + tuple(selected_codes),
                    )
                    for row in rows:
                        report_date = row["report_date"]
                        if hasattr(report_date, "date"):
                            report_date = report_date.date()
                        elif isinstance(report_date, str):
                            report_date = datetime.date.fromisoformat(report_date[:10])
                        metrics = {}
                        for field, label in (("roe", "净资产收益率"),
                                             ("net_profit_yoy", "净利润增长率"),
                                             ("revenue_yoy", "营业收入增长率"),
                                             ("debt_to_assets", "资产负债率")):
                            if row.get(field) is not None:
                                metrics[label] = float(row[field])
                        if metrics:
                            avail_date = row["avail_date"]
                            if hasattr(avail_date, "date"):
                                avail_date = avail_date.date()
                            elif isinstance(avail_date, str):
                                avail_date = datetime.date.fromisoformat(avail_date[:10])
                            financial_cache.setdefault(row["code"], []).append((report_date, avail_date, metrics))

                if not skip_flow:
                    date_clause = "AND f.trade_date <= %s" if eval_date else "AND f.trade_date <= CURDATE()"
                    args = tuple(selected_codes) + ((eval_date,) if eval_date else ())
                    rows = Database.fetchall(
                        f"SELECT code, main_net_flow, amount FROM ("
                        f"SELECT f.code, f.main_net_flow, q.amount, f.trade_date, "
                        f"ROW_NUMBER() OVER (PARTITION BY f.code ORDER BY f.trade_date DESC) rn "
                        f"FROM capital_flow_daily f "
                        f"LEFT JOIN daily_quotes q ON q.code=f.code AND q.trade_date=f.trade_date "
                        f"WHERE f.code IN ({placeholders}) {date_clause}"
                        f") t WHERE rn <= 20 ORDER BY code, rn",
                        args,
                    )
                    for row in rows:
                        flow_cache.setdefault(row["code"], []).append({
                            "main_net_flow": float(row.get("main_net_flow") or 0),
                            "amount": float(row.get("amount") or 0),
                        })

                if not skip_north_flow:
                    date_clause = "AND trade_date <= %s" if eval_date else "AND trade_date <= CURDATE()"
                    args = tuple(selected_codes) + ((eval_date,) if eval_date else ())
                    rows = Database.fetchall(
                        f"SELECT code, trade_date, net_flow FROM ("
                        f"SELECT code, trade_date, net_flow, "
                        f"ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) rn "
                        f"FROM north_flow WHERE code IN ({placeholders}) {date_clause}"
                        f") t WHERE rn <= 20 ORDER BY code, rn",
                        args,
                    )
                    for row in rows:
                        north_flow_data.setdefault(row["code"], []).append(
                            (row["trade_date"], float(row.get("net_flow") or 0)))

                if not skip_sent:
                    rows = Database.fetchall(
                        f"SELECT code, "
                        f"AVG(CASE WHEN trade_date >= DATE_SUB(%s, INTERVAL 7 DAY) THEN avg_score END) s_7d, "
                        f"AVG(CASE WHEN trade_date >= DATE_SUB(%s, INTERVAL 7 DAY) THEN avg_confidence END) conf_7d, "
                        f"AVG(CASE WHEN trade_date >= DATE_SUB(%s, INTERVAL 30 DAY) "
                        f"AND trade_date < DATE_SUB(%s, INTERVAL 7 DAY) THEN avg_score END) s_23d "
                        f"FROM sentiment_daily WHERE code IN ({placeholders}) "
                        f"AND trade_date >= DATE_SUB(%s, INTERVAL 30 DAY) "
                        f"AND trade_date <= %s GROUP BY code",
                        (score_date,) * 4 + tuple(selected_codes) + (score_date,) * 2,
                    )
                    for row in rows:
                        if row.get("s_7d") is None:
                            continue
                        raw = max(-1.0, min(1.0, float(row["s_7d"])))
                        confidence = float(row.get("conf_7d", 0.5) or 0.5)
                        if confidence < 0.35:
                            continue
                        score = raw * confidence
                        if row.get("s_23d") is not None:
                            score += (raw - float(row["s_23d"])) * 0.3
                        sentiment_data[row["code"]] = max(-1.0, min(1.0, score))
            except Exception as e:
                logger.warning(f"[QlibPrimary] 辅助数据批量加载失败: {e}")

            try:
                from core.data.sources.etf_flow import get_etf_flow_by_industry
                etf_flows = get_etf_flow_by_industry(as_of=score_date.isoformat(), days=5)
            except Exception as e:
                logger.warning(f"[QlibPrimary] ETF资金流加载失败: {e}")

        for candidate in candidates:
            code = candidate["code"]
            df = self._df_cache.get(code)
            info = industry_map.get(code, {})
            industry = info.get("industry", "") if isinstance(info, dict) else ""
            candidate["fundamental_score"] = round(
                _score_fundamentals(code, as_of=score_date, financial_cache=financial_cache), 3
            ) if not skip_fundamental else 0.0
            candidate["flow_score"] = round(
                _score_capital_flow(code, as_of=score_date, flow_cache=flow_cache), 3
            ) if not skip_flow else 0.0
            candidate["north_flow_score"] = round(
                _score_north_flow(code, df, north_flow_data, as_of=score_date), 3
            ) if not skip_north_flow else 0.0
            candidate["sentiment_score"] = round(sentiment_data.get(code, 0.0), 3)
            inflow = float(etf_flows.get(industry, 0) or 0)
            if inflow > 1e9:
                etf_score = 0.15
            elif inflow > 5e8:
                etf_score = 0.10
            elif inflow > 1e8:
                etf_score = 0.06
            elif inflow > 0:
                etf_score = 0.03
            elif inflow < -1e9:
                etf_score = -0.15
            elif inflow < -5e8:
                etf_score = -0.10
            elif inflow < -1e8:
                etf_score = -0.06
            elif inflow < 0:
                etf_score = -0.03
            else:
                etf_score = 0.0
            candidate["etf_flow_score"] = etf_score
            if flow_cache.get(code):
                candidate["flow_net_5d"] = round(sum(
                    row["main_net_flow"] for row in flow_cache[code][:5]
                ), 1)

        logger.info(f"[QlibPrimary] Qlib覆盖 {len(ranked_codes)} 只，检查 {examined} 只后取 Top-{len(candidates)}；辅助仅计算入选股票")
        coverage_total = len(candidates)
        industry_covered = sum(
            1 for candidate in candidates
            if isinstance(industry_map.get(candidate["code"]), dict)
            and industry_map[candidate["code"]].get("industry")
        )
        etf_covered = sum(
            1 for candidate in candidates
            if isinstance(industry_map.get(candidate["code"]), dict)
            and industry_map[candidate["code"]].get("industry") in etf_flows
        )
        auxiliary_coverage = {
            "industry": {
                "covered": industry_covered,
                "total": coverage_total,
                "rate": round(industry_covered / coverage_total, 3) if coverage_total else 0.0,
            },
            "etf_flow": {
                "covered": etf_covered,
                "total": coverage_total,
                "rate": round(etf_covered / coverage_total, 3) if coverage_total else 0.0,
            },
        }
        for source, data, skipped in (
            ("fundamental", financial_cache, skip_fundamental),
            ("flow", flow_cache, skip_flow),
            ("north_flow", north_flow_data, skip_north_flow),
            ("sentiment", sentiment_data, skip_sent),
        ):
            if skipped:
                continue
            covered = sum(1 for candidate in candidates if candidate["code"] in data)
            auxiliary_coverage[source] = {
                "covered": covered,
                "total": coverage_total,
                "rate": round(covered / coverage_total, 3) if coverage_total else 0.0,
            }
        result = {
            "total_input": len(codes),
            "candidates": candidates,
            "excluded_reasons": excluded,
            "weights": {
                "qlib": 1.0,
                "fundamental": 0.0,
                "flow": 0.0,
                "north_flow": 0.0,
                "sentiment": 0.0,
                "etf_flow": 0.0,
            },
            "ranking_mode": "qlib_primary",
            "auxiliary_only": ["fundamental", "flow", "north_flow", "sentiment", "etf_flow"],
            "auxiliary_coverage": auxiliary_coverage,
            "qlib_coverage": len(ranked_codes),
            "sent_coverage": round(len(sentiment_data) / len(candidates), 3) if candidates else 0.0,
        }
        if hot_sector_meta:
            result["pool_info"] = hot_sector_meta
        score_result = getattr(self, "_last_score_result", None)
        if score_result:
            result["score_processing"] = dict(score_result.get("audit", {}))
        return result

    def _l0_filter(self, pool="all", tech_w=0.20, qlib_w=0.20, sent_w=0.05,
                   fundamental_w=0.10, flow_w=0.05,
                   north_flow_w=0.10, size_turnover_w=0.10,
                   momentum_w=0.10, volatility_w=0.05,
                   illiq_w=0.02, overnight_gap_w=0.02,
                   skip_tech=False, skip_qlib=False, skip_sent=False,
                   skip_fundamental=False, skip_flow=False,
                   skip_north_flow=False, skip_size_turnover=False,
                   skip_momentum=False, skip_volatility=False,
                   skip_illiq=False, skip_overnight_gap=False,
                   skip_ht=False, ht_factors_w=0.10, ht_key=None,
                   skip_wb_industry=False, skip_wb_divergence=False, skip_wb_moneyflow=False,
                   wb_industry_w=0.05, wb_divergence_w=0.05, wb_moneyflow_w=0.05,
                   wb_industry_n=20, wb_divergence_n=10, wb_moneyflow_n=5,
                   momentum_n=20, volatility_n=20, illiq_n=20, overnight_gap_n=20,
                   tech_indicators=None, qlib_model=None, progress_cb=None,
                   hot_sector_count=10, custom_sectors=None, code_prefixes=None,
                   concept_sectors=None, eval_date=None, snapshot=None,
                   macro_risk_score=0.0, macro_sector_bias=None,
                   sector_fusion=None, market_rr=None, direction=None,
                   use_vol_switch=False, etf_flow_bias=None, factor_overlay=None,
                   neutralize_qlib=None, top_n=50, ranking_mode="legacy",
                   decision_day_filter=True) -> Dict:
        """L0: 硬排除规则 + 技术+Qlib+情绪初筛 (多线程并行)"""
        if tech_indicators is None:
            tech_indicators = self.TECH_PRESETS["all"]

        hot_sector_meta = None
        if snapshot is not None:
            codes = list(snapshot.keys())
            logger.info(f"[L0] 使用预计算快照: {len(codes)} 只 (跳过数据获取和硬排除)")
        else:
            codes = _get_all_stock_codes()
            # 代码前缀过滤
            if code_prefixes:
                prefixes = tuple(str(p) for p in code_prefixes)
                codes = [c for c in codes if c.startswith(prefixes)]
                logger.info(f"[L0] 代码前缀过滤: {prefixes} → {len(codes)} 只")
        if snapshot is None:
            if pool == "hot_sectors":
                hot_codes, hot_sector_meta = self._get_hot_sector_stocks(
                    top_n=hot_sector_count)
                codes = [c for c in codes if c in set(hot_codes)]
            elif pool in ("csi300", "csi_all", "csi500", "csi300+csi500", "a500", "stable_history_500d", "non_csi300", "small_cap"):
                codes = filter_pool_codes(codes, pool, as_of=eval_date)
            elif pool == "custom" and custom_sectors:
                try:
                    from core.domain.screening.stock_scanner import SectorScanner
                    scanner = SectorScanner()
                    custom_stocks = set()
                    for sector_name in custom_sectors:
                        try:
                            stocks_df = scanner.get_sector_stocks(sector_name)
                            if stocks_df is not None and not stocks_df.empty:
                                code_col = "code" if "code" in stocks_df.columns else stocks_df.columns[0]
                                for raw in stocks_df[code_col]:
                                    c2 = str(raw)
                                    if "." in c2:
                                        c2 = c2.split(".")[1]
                                    c2 = c2.strip()[:6]
                                    if c2.isdigit() and len(c2) == 6:
                                        custom_stocks.add(c2)
                        except Exception:
                            pass
                    codes = [c for c in codes if c in custom_stocks]
                    logger.info(f"[L0] 自定义板块池: {len(custom_sectors)}个板块 → {len(codes)}只股票")
                except Exception as e:
                    logger.warning(f"[L0] 自定义板块过滤失败: {e}，回退全A股")
            elif pool == "concept" and concept_sectors:
                try:
                    from core.domain.screening.stock_scanner import SectorScanner
                    scanner = SectorScanner()
                    concept_stocks = set()
                    for sector_name in concept_sectors:
                        try:
                            stocks_df = scanner.get_concept_sector_stocks(sector_name)
                            if stocks_df is not None and not stocks_df.empty:
                                code_col = "code" if "code" in stocks_df.columns else stocks_df.columns[0]
                                for raw in stocks_df[code_col]:
                                    c2 = str(raw)
                                    if "." in c2:
                                        c2 = c2.split(".")[1]
                                    c2 = c2.strip()[:6]
                                    if c2.isdigit() and len(c2) == 6:
                                        concept_stocks.add(c2)
                        except Exception:
                            pass
                    codes = [c for c in codes if c in concept_stocks]
                    logger.info(f"[L0] 自定义概念板块池: {len(concept_sectors)}个概念板块 → {len(codes)}只股票")
                except Exception as e:
                    logger.warning(f"[L0] 概念板块过滤失败: {e}，回退全A股")
            elif pool == "sector_rotation":
                try:
                    from core.domain.screening._scoring_legacy import _rank_sectors
                    from core.domain.screening.stock_scanner import SectorScanner
                    top_sectors = _rank_sectors(eval_date=eval_date, top_n=5)
                    sector_names = [s[0] for s in top_sectors]
                    scanner = SectorScanner()
                    rotation_codes = set()
                    for sname in sector_names:
                        try:
                            df = scanner.get_sector_stocks(sname)
                            if df is not None and not df.empty:
                                code_col = "code" if "code" in df.columns else df.columns[0]
                                for raw in df[code_col]:
                                    c2 = str(raw).split(".")[-1].strip()[:6]
                                    if c2.isdigit() and len(c2) == 6:
                                        rotation_codes.add(c2)
                        except Exception:
                            pass
                    codes = [c for c in codes if c in rotation_codes]
                    logger.info(f"[L0] 板块轮动: Top行业 {sector_names} → "
                               f"{len(rotation_codes)} 只成分股 → {len(codes)} 只筛选后")
                    hot_sector_meta = {"top_sectors": [(s[0], s[1]) for s in top_sectors]}
                except Exception as e:
                    logger.warning(f"[L0] 板块轮动失败: {e}，回退全A股")
            elif pool == "concept_rotation":
                try:
                    from core.domain.screening._scoring_legacy import _rank_concept_sectors
                    from core.domain.screening.stock_scanner import SectorScanner
                    top_concepts = _rank_concept_sectors(eval_date=eval_date, top_n=5)
                    concept_names = [s[0] for s in top_concepts]
                    scanner = SectorScanner()
                    rotation_codes = set()
                    for cname in concept_names:
                        try:
                            df = scanner.get_sector_stocks(cname, sector_type='concept')
                            if df is None or df.empty:
                                df = scanner.get_concept_sector_stocks(cname)
                            if df is not None and not df.empty:
                                code_col = "code" if "code" in df.columns else df.columns[0]
                                for raw in df[code_col]:
                                    c2 = str(raw).split(".")[-1].strip()[:6]
                                    if c2.isdigit() and len(c2) == 6:
                                        rotation_codes.add(c2)
                        except Exception:
                            pass
                    codes = [c for c in codes if c in rotation_codes]
                    logger.info(f"[L0] 概念轮动: Top概念 {concept_names} → "
                               f"{len(rotation_codes)} 只成分股 → {len(codes)} 只筛选后")
                    hot_sector_meta = {"top_concepts": [(s[0], s[1]) for s in top_concepts]}
                except Exception as e:
                    logger.warning(f"[L0] 概念轮动失败: {e}，回退全A股")

        total = len(codes)
        # ═══ 候选池预筛选 ═══
        # 指数成分作为基础池，可选叠加成交额 Top-N 扩充
        prefilter_pool = getattr(config, "BUY_SIDE_PREFILTER_POOL", "")
        prefilter_n = getattr(config, "BUY_SIDE_PREFILTER_N", 0)
        if prefilter_pool or prefilter_n > 0:
            index_codes = None
            if prefilter_pool:
                index_codes = set(self._prefilter_by_index(codes, prefilter_pool, eval_date))
            if prefilter_n > 0:
                extra = self._prefilter_by_turnover(codes, prefilter_n, eval_date)
                if index_codes:
                    extra = [c for c in extra if c not in index_codes]
                    codes = [c for c in codes if c in index_codes] + extra
                    logger.info(f"[L0] 预筛选: {total} → {len(codes)} 只 "
                                f"(指数{len(index_codes)} + 成交额Top{len(extra)})")
                else:
                    codes = extra
                    logger.info(f"[L0] 预筛选: {total} → {len(codes)} 只 (成交额Top{prefilter_n})")
            total = len(codes)
        # ═══ 先拿 Qlib 预测（权重检测需要知道 qlib_scores 是否为空）═══
        # 批量场景(带 eval_date)走预加载缓存；交互选股(无 eval_date)始终现场预测
        qlib_scores = {}
        score_result = None
        if ranking_mode == "qlib_primary" and skip_qlib:
            raise ValueError("当前选股只允许按Qlib预测排序，不能跳过Qlib")
        if not skip_qlib:
            if eval_date and self._qlib_by_date and not qlib_model:
                date_key = eval_date.isoformat() if hasattr(eval_date, 'isoformat') else str(eval_date)
                qlib_scores = self._qlib_by_date.get(date_key, {})
                if qlib_scores:
                    cached_details = self._qlib_score_details_by_date.get(date_key)
                    score_result = clone_score_result(cached_details) if cached_details else None
                    if score_result is None:
                        score_result = build_legacy_processed_result(
                            qlib_scores, {"source": "preloaded_processed_cache", "ref_date": date_key})
                    else:
                        score_result.setdefault("audit", {}).setdefault("context", {}).setdefault(
                            "ref_date", date_key
                        )
            if not qlib_scores and eval_date and not qlib_model:
                # 尝试从 Redis 恢复（Phase 1 批量加载后重启了也能用）
                from core.shared.infrastructure import redis_client as rds
                date_key = eval_date.isoformat() if hasattr(eval_date, 'isoformat') else str(eval_date)
                cached = rds._get(f"qlib:predictions:{date_key}")
                if cached and isinstance(cached, dict):
                    qlib_scores = cached
                    score_result = build_legacy_processed_result(
                        qlib_scores, {"source": "redis_processed_cache", "ref_date": date_key})
                    if not self._qlib_by_date:
                        self._qlib_by_date = {}
                    self._qlib_by_date[date_key] = cached
            if not qlib_scores:
                model = qlib_model
                if not model and direction:
                    model = self._find_model_by_direction(direction)
                    if model:
                        logger.info(f"[L0] 方向({direction})匹配模型: {model[0]}/{model[1]}")
                if not model and market_rr is not None:
                    regime_val = market_rr.regime if hasattr(market_rr, 'regime') else None
                    if regime_val:
                        model = self._find_best_qlib_model(regime=regime_val)
                if not model:
                    model = self._find_best_qlib_model()
                if model:
                    if not qlib_model:
                        logger.info(f"[L0] 自动选择 Qlib 模型: {model[0]}/{model[1]}")
                    loaded_scores = self._load_qlib_predictions(
                        model, as_of=eval_date, pool=pool,
                        neutralize_qlib=neutralize_qlib, return_details=True)
                    if isinstance(loaded_scores, dict) and "scores" in loaded_scores:
                        score_result = clone_score_result(loaded_scores)
                        qlib_scores = dict(score_result["scores"])
                    else:
                        qlib_scores = loaded_scores
                        if qlib_scores:
                            score_result = build_legacy_processed_result(
                                qlib_scores, {
                                    "source": "compatibility_loader",
                                    "ref_date": str(eval_date)[:10] if eval_date else None,
                                })
                else:
                    logger.info("[L0] 未找到已训练 Qlib 模型")
                if not qlib_scores and ranking_mode == "qlib_primary":
                    raise RuntimeError("Qlib预测不可用，已停止选股；辅助数据不会替代排序")
                if not qlib_scores:
                    logger.warning("[L0] Qlib 预测为空，回退权重转移")

        if ranking_mode == "qlib_primary" and macro_sector_bias and qlib_scores:
            try:
                qlib_scores = apply_sector_bias(
                    qlib_scores, macro_sector_bias, self.fetcher.get_industry_map() or {})
                if score_result is not None:
                    apply_score_step(score_result, "macro_sector_bias", qlib_scores)
            except Exception as e:
                logger.warning(f"[Macro] 选股行业加分失败: {e}")
                if score_result is not None:
                    apply_score_step(score_result, "macro_sector_bias", qlib_scores,
                                     status="fallback", detail=e)

        # ── 板块背离双模型融合 (开关: 入参优先于 config, 异常自动降级) ──
        fusion_info = None
        use_fusion = sector_fusion if sector_fusion is not None else getattr(config, "SECTOR_FUSION_ENABLED", False)
        if use_fusion and use_vol_switch:
            logger.info("[SectorFusion] use_vol_switch 开启, 跳过板块背离融合")
        if use_fusion and not use_vol_switch and qlib_scores and not skip_qlib:
            qlib_scores_raw = dict(qlib_scores)
            qlib_scores, f_info = self._apply_sector_fusion(
                qlib_scores, eval_date, pool, market_rr=market_rr,
                neutralize_qlib=neutralize_qlib)
            if score_result is not None:
                fusion_status = "applied" if f_info.get("attr") or f_info.get("divergent") else "no_change"
                apply_score_step(score_result, "sector_fusion", qlib_scores,
                                 status=fusion_status)
            if f_info.get("attr") or f_info.get("divergent"):
                fusion_info = f_info
                fusion_info["_raw_scores"] = qlib_scores_raw
        self._last_fusion_info = fusion_info
        self._last_score_result = score_result

        if ranking_mode == "qlib_primary" and qlib_scores and eval_date:
            context = (score_result or {}).get("audit", {}).get("context", {})
            prediction_date = context.get("prediction_date") or context.get("ref_date")
            decision_date = str(eval_date)[:10]
            if not prediction_date:
                raise RuntimeError("Qlib预测缺少实际预测日期，已停止选股")
            if str(prediction_date)[:10] != decision_date:
                raise RuntimeError(
                    f"Qlib预测日期 {str(prediction_date)[:10]} 与决策日 {decision_date} 不一致，"
                    "请先完成Qlib增量同步后重试"
                )

        if ranking_mode == "qlib_primary":
            return self._select_qlib_top_candidates(
                codes, qlib_scores, top_n, eval_date=eval_date, snapshot=snapshot,
                skip_sent=skip_sent, skip_fundamental=skip_fundamental,
                skip_flow=skip_flow, skip_north_flow=skip_north_flow,
                progress_cb=progress_cb, hot_sector_meta=hot_sector_meta,
                decision_day_filter=decision_day_filter,
            )

        # ═══ 自动检测空数据源并重新分配权重 ═══
        effective_tech_w = tech_w
        effective_qlib_w = qlib_w
        effective_sent_w = sent_w
        effective_fundamental_w = fundamental_w
        effective_flow_w = flow_w
        effective_north_flow_w = north_flow_w
        effective_size_turnover_w = size_turnover_w
        effective_momentum_w = momentum_w
        effective_volatility_w = volatility_w
        effective_ht_factors_w = ht_factors_w
        effective_wb_industry_w = wb_industry_w
        effective_wb_divergence_w = wb_divergence_w
        effective_wb_moneyflow_w = wb_moneyflow_w
        effective_illiq_w = illiq_w
        effective_overnight_gap_w = overnight_gap_w
        overlay_scores = (factor_overlay or {}).get("scores") or {}
        effective_overlay_w = max(0.0, min(float((factor_overlay or {}).get("weight", 0.0)), 0.5)) if overlay_scores else 0.0
        sent_coverage = 0.0

        if not skip_qlib and qlib_w > 0 and not qlib_scores:
            logger.info("[L0] Qlib 预测为空，权重重新分配至活跃源")
            _skip_map = {
                "tech": skip_tech, "momentum": skip_momentum,
                "volatility": skip_volatility, "sentiment": skip_sent,
                "fundamental": skip_fundamental, "flow": skip_flow,
                "north_flow": skip_north_flow, "size_turnover": skip_size_turnover,
                "illiq": skip_illiq, "overnight_gap": skip_overnight_gap,
                "wb_industry": skip_wb_industry, "wb_divergence": skip_wb_divergence,
                "wb_moneyflow": skip_wb_moneyflow, "ht_factors": skip_ht,
            }
            _src_weights = {
                "tech": effective_tech_w, "momentum": effective_momentum_w,
                "volatility": effective_volatility_w, "sentiment": effective_sent_w,
                "fundamental": effective_fundamental_w, "flow": effective_flow_w,
                "north_flow": effective_north_flow_w, "size_turnover": effective_size_turnover_w,
                "illiq": effective_illiq_w, "overnight_gap": effective_overnight_gap_w,
                "wb_industry": effective_wb_industry_w, "wb_divergence": effective_wb_divergence_w,
                "wb_moneyflow": effective_wb_moneyflow_w, "ht_factors": effective_ht_factors_w,
            }
            alt_w = sum(_w for _sig, _w in _src_weights.items()
                       if _w > 0 and not _skip_map.get(_sig, False))
            if alt_w == 0:
                logger.info("[L0] Qlib 权重无转移目标，回退到默认权重")
                _not_skip = lambda s: not _skip_map.get(s, False)
                _active = [
                    _not_skip("tech"), _not_skip("momentum"), _not_skip("volatility"),
                    _not_skip("sentiment"), _not_skip("fundamental"), _not_skip("flow"),
                    _not_skip("illiq"), _not_skip("overnight_gap"),
                ]
                n = sum(_active)
                w = 1.0 / n if n > 0 else 0
                effective_tech_w       = w if _active[0] else 0.0
                effective_momentum_w   = w if _active[1] else 0.0
                effective_volatility_w = w if _active[2] else 0.0
                effective_sent_w       = w if _active[3] else 0.0
                effective_fundamental_w = w if _active[4] else 0.0
                effective_flow_w       = w if _active[5] else 0.0
                effective_illiq_w      = w if _active[6] else 0.0
                effective_overnight_gap_w = w if _active[7] else 0.0
                effective_north_flow_w = effective_size_turnover_w = \
                    effective_wb_industry_w = effective_wb_divergence_w = \
                    effective_wb_moneyflow_w = effective_ht_factors_w = 0
            else:
                _new_w = {}
                for _sig, _w in _src_weights.items():
                    if _w > 0 and not _skip_map.get(_sig, False):
                        _new_w[_sig] = _w + qlib_w * (_w / alt_w)
                effective_tech_w = _new_w.get("tech", effective_tech_w)
                effective_momentum_w = _new_w.get("momentum", effective_momentum_w)
                effective_volatility_w = _new_w.get("volatility", effective_volatility_w)
                effective_sent_w = _new_w.get("sentiment", effective_sent_w)
                effective_fundamental_w = _new_w.get("fundamental", effective_fundamental_w)
                effective_flow_w = _new_w.get("flow", effective_flow_w)
                effective_north_flow_w = _new_w.get("north_flow", effective_north_flow_w)
                effective_size_turnover_w = _new_w.get("size_turnover", effective_size_turnover_w)
                effective_illiq_w = _new_w.get("illiq", effective_illiq_w)
                effective_overnight_gap_w = _new_w.get("overnight_gap", effective_overnight_gap_w)
                effective_wb_industry_w = _new_w.get("wb_industry", effective_wb_industry_w)
                effective_wb_divergence_w = _new_w.get("wb_divergence", effective_wb_divergence_w)
                effective_wb_moneyflow_w = _new_w.get("wb_moneyflow", effective_wb_moneyflow_w)
                effective_ht_factors_w = _new_w.get("ht_factors", effective_ht_factors_w)
            effective_qlib_w = 0

        if not skip_sent and sent_w > 0:
            sent_coverage = 0.0
            covered = 0
            quality_covered = 0
            try:
                from core.data import Database
                conn = Database.get_connection()
                try:
                    cur = conn.cursor()
                    sample = codes[:500]
                    placeholders = ",".join(["%s"] * len(sample))
                    ref_date = eval_date if eval_date else None
                    if ref_date:
                        cur.execute(
                            f"SELECT COUNT(DISTINCT code) as n FROM sentiment_daily "
                            f"WHERE code IN ({placeholders}) "
                            f"AND trade_date >= DATE_SUB(%s, INTERVAL 7 DAY)",
                            sample + [ref_date],
                        )
                    else:
                        cur.execute(
                            f"SELECT COUNT(DISTINCT code) as n FROM sentiment_daily "
                            f"WHERE code IN ({placeholders}) "
                            f"AND trade_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)",
                            sample,
                        )
                    row = cur.fetchone()
                    cur.close()
                    covered = row.get("n", 0) or 0 if row else 0
                    sent_coverage = covered / len(sample) if sample else 0
                    # 同时检查高置信度覆盖率 (复用同一连接)
                    cur2 = conn.cursor()
                    try:
                        if ref_date:
                            cur2.execute(
                                f"SELECT COUNT(DISTINCT code) as n FROM sentiment_daily "
                                f"WHERE code IN ({placeholders}) "
                                f"AND trade_date >= DATE_SUB(%s, INTERVAL 7 DAY) "
                                f"AND avg_confidence >= 0.35",
                                sample + [ref_date],
                            )
                        else:
                            cur2.execute(
                                f"SELECT COUNT(DISTINCT code) as n FROM sentiment_daily "
                                f"WHERE code IN ({placeholders}) "
                                f"AND trade_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) "
                                f"AND avg_confidence >= 0.35",
                                sample,
                            )
                        row2 = cur2.fetchone()
                        cur2.close()
                        quality_covered = row2.get("n", 0) or 0 if row2 else 0
                    except Exception:
                        pass
                    finally:
                        try:
                            cur2.close()
                        except Exception:
                            pass
                finally:
                    Database.return_connection(conn)
            except Exception:
                pass
            quality_coverage = quality_covered / len(sample) if sample else 0
            if sent_coverage < 0.10:
                logger.info(f"[L0] 情绪覆盖率仅 {sent_coverage:.0%}，权重转移至技术指标")
                effective_tech_w += sent_w
                effective_sent_w = 0
            else:
                logger.info(f"[L0] 情绪覆盖率 {sent_coverage:.0%} ({covered}/{len(sample)}), "
                           f"高置信度 {quality_coverage:.0%} ({quality_covered}/{len(sample)}), "
                           f"保留权重 sent={sent_w:.2f}")

        # ═══ 批量加载情绪数据（置信度加权） ═══
        # 不按权重门控：权重为0时仍加载用于前端展示
        sentiment_data = {}
        if not skip_sent:
            try:
                from core.data import Database
                batch_size = 500
                ref_date = eval_date if eval_date else None
                date_func = "%s" if ref_date else "CURDATE()"
                date_args = [ref_date] * 5 if ref_date else []
                for bs in range(0, len(codes), batch_size):
                    batch = codes[bs:bs + batch_size]
                    ph = ",".join(["%s"] * len(batch))
                    sql = (
                        f"SELECT code, "
                        f"AVG(CASE WHEN trade_date >= DATE_SUB({date_func}, INTERVAL 7 DAY) "
                        f"THEN avg_score END) as s_7d, "
                        f"AVG(CASE WHEN trade_date >= DATE_SUB({date_func}, INTERVAL 7 DAY) "
                        f"THEN avg_confidence END) as conf_7d, "
                        f"AVG(CASE WHEN trade_date >= DATE_SUB({date_func}, INTERVAL 30 DAY) "
                        f"AND trade_date < DATE_SUB({date_func}, INTERVAL 7 DAY) "
                        f"THEN avg_score END) as s_23d "
                        f"FROM sentiment_daily "
                        f"WHERE code IN ({ph}) "
                        f"AND trade_date >= DATE_SUB({date_func}, INTERVAL 30 DAY) "
                        f"GROUP BY code"
                    )
                    rows = Database.fetchall(sql, tuple(batch) + tuple(date_args))
                    for row in rows:
                        s7 = row.get("s_7d")
                        if s7 is None:
                            continue
                        raw_score = max(-1.0, min(1.0, float(s7)))
                        conf = float(row.get("conf_7d", 0.5) or 0.5)
                        if conf < 0.35:
                            continue
                        base = raw_score * conf
                        s23 = row.get("s_23d")
                        if s23 is not None:
                            trend = raw_score - float(s23)
                            base += trend * 0.3
                        sentiment_data[row["code"]] = max(-1.0, min(1.0, base))
                logger.info(f"[L0] 批量情绪加载: {len(sentiment_data)} 只 (含7d/30d趋势, 分 {((len(codes)-1)//batch_size)+1} 批)")
            except Exception:
                pass


        re_total = (effective_tech_w + effective_qlib_w + effective_sent_w +
                    effective_fundamental_w + effective_flow_w +
                    effective_north_flow_w + effective_size_turnover_w +
                    effective_momentum_w + effective_volatility_w +
                    effective_illiq_w + effective_overnight_gap_w +
                    effective_ht_factors_w +
                    effective_wb_industry_w + effective_wb_divergence_w + effective_wb_moneyflow_w + effective_overlay_w)
        if re_total > 0:
            effective_tech_w /= re_total
            effective_qlib_w /= re_total
            effective_sent_w /= re_total
            effective_fundamental_w /= re_total
            effective_flow_w /= re_total
            effective_north_flow_w /= re_total
            effective_size_turnover_w /= re_total
            effective_momentum_w /= re_total
            effective_volatility_w /= re_total
            effective_illiq_w /= re_total
            effective_overnight_gap_w /= re_total
            effective_ht_factors_w /= re_total
            effective_wb_industry_w /= re_total
            effective_wb_divergence_w /= re_total
            effective_wb_moneyflow_w /= re_total
            effective_overlay_w /= re_total

        all_weights = (effective_tech_w, effective_qlib_w, effective_sent_w,
                       effective_fundamental_w, effective_flow_w,
                       effective_north_flow_w, effective_size_turnover_w,
                       effective_momentum_w, effective_volatility_w,
                       effective_illiq_w, effective_overnight_gap_w,
                       effective_ht_factors_w, effective_overlay_w,
                       effective_wb_industry_w, effective_wb_divergence_w, effective_wb_moneyflow_w)
        active_sources = sum(1 for w in all_weights if w > 0)
        total_sources = 15 + (1 if effective_overlay_w > 0 else 0)
        min_score = getattr(config, "BUY_SIDE_L0_MIN_SCORE", 0.35) * (active_sources / total_sources) * 0.5
        # 宏观风险调节: danger → 提高门槛, warning → 轻微提高
        if macro_risk_score >= 0.7:
            min_score *= 1.5
            logger.info(f"[Macro] 宏观风险 danger({macro_risk_score:.2f}), L0阈值上调至 {min_score:.3f}")
        elif macro_risk_score >= 0.4:
            min_score *= 1.2
            logger.info(f"[Macro] 宏观风险 warning({macro_risk_score:.2f}), L0阈值上调至 {min_score:.3f}")
        if macro_sector_bias is None:
            macro_sector_bias = {}
        logger.info(f"[L0] 权重: tech={effective_tech_w:.2f} qlib={effective_qlib_w:.2f} "
                    f"sent={effective_sent_w:.2f} fund={effective_fundamental_w:.2f} "
                    f"flow={effective_flow_w:.2f} north={effective_north_flow_w:.2f} "
                    f"size={effective_size_turnover_w:.2f} mom={effective_momentum_w:.2f} "
                    f"vol={effective_volatility_w:.2f} illiq={effective_illiq_w:.2f} gap={effective_overnight_gap_w:.2f} "
                    f"| 阈值={min_score:.3f} | 活跃源={active_sources}/{total_sources}")

        # ── 宏观行业偏好: 提前加载行业映射 ──
        macro_industry_map = {}
        if macro_sector_bias:
            try:
                macro_industry_map = self.fetcher.get_industry_map()
                logger.info(f"[Macro] 行业映射: {len(macro_industry_map)} 只, "
                            f"偏好板块={len(macro_sector_bias)} 个")
            except Exception:
                pass

        # ═══ 预计算 indicator_weights ═══
        indicator_weights = {
            "MACD": 1.0 if "MACD" in tech_indicators else 0,
            "RSI": 0.8 if "RSI" in tech_indicators else 0,
            "MA": 0.7 if "MA" in tech_indicators else 0,
            "KDJ": 0.5 if "KDJ" in tech_indicators else 0,
            "BB": 0.6 if "BB" in tech_indicators else 0,
            "VOLUME": 0.3 if "VOLUME" in tech_indicators else 0,
        }
        # 根据实际活跃指标权重和归一化
        tech_weight_sum = sum(v for v in indicator_weights.values())
        if tech_weight_sum <= 0:
            tech_weight_sum = 1.0

        today = eval_date or today_date()
        cutoff_date = today - datetime.timedelta(days=60)
        listing_dates = {} if snapshot is not None else self._get_listing_dates_map(codes)
        # 历史回看模式：跳过情绪/qlib（避免前视偏差），调整数据获取截止日期
        is_historical = eval_date is not None
        if is_historical:
            end_date_str = today.strftime("%Y%m%d")

        # ═══ 多线程并行处理 ═══
        candidates = []
        excluded = {"st": 0, "suspended": 0, "limit_up": 0, "low_liquidity": 0,
                    "low_price": 0, "new_listing": 0, "identity_unknown": 0, "data_error": 0,
                    "below_threshold": 0}
        lock = threading.Lock()
        processed = [0]
        _tls = threading.local()

        def _score_l0(df, code, name, close, volume, close_arr, ret_arr, fetcher_for_fund):
            """计算所有L0子项评分 + 加权求和 + 宏观/冲击调整 → (candidate_dict|None, reason|df)"""
            tech_s = 0.0
            if not skip_tech:
                try:
                    from core.domain.signals.engine import get_fusion
                    fr = get_fusion(weights=indicator_weights).evaluate(df)
                    tech_s = fr.score / max(tech_weight_sum, 0.01)
                except Exception:
                    pass

            scores = {
                "tech": tech_s,
                "qlib": qlib_scores.get(code, 0.0),
                "sentiment": sentiment_data.get(code, 0.0) if not skip_sent else 0.0,
                "north_flow": _score_north_flow(code, df, north_flow_data, as_of=eval_date) if not skip_north_flow else 0.0,
                "size_turnover": _score_size_turnover(code, df, eval_date=eval_date) if not skip_size_turnover else 0.0,
                "momentum": _score_momentum(df, n=momentum_n, close_arr=close_arr) if not skip_momentum else 0.0,
                "volatility": _score_volatility(df, n=volatility_n, close_arr=close_arr, ret_arr=ret_arr) if not skip_volatility else 0.0,
                "ht": _compute_ht_score(df, ht_key) if not skip_ht else 0.0,
                "wb_industry": _score_industry_residual_momentum(code, df, n=wb_industry_n, industry_map=industry_map, industry_returns=industry_returns, close_arr=close_arr) if not skip_wb_industry else 0.0,
                "wb_divergence": _score_volume_price_divergence(df, n=wb_divergence_n, close_arr=close_arr) if not skip_wb_divergence else 0.0,
                "wb_moneyflow": _score_smart_money_flow(code, df, n=wb_moneyflow_n, flow_data=wb_flow_data, close_arr=close_arr) if not skip_wb_moneyflow else 0.0,
                "fundamental": _score_fundamentals(code, fetcher=fetcher_for_fund, as_of=today, financial_cache=self._financial_cache) if not skip_fundamental else 0.0,
                "flow": _score_capital_flow(code, fetcher=fetcher_for_fund, as_of=today, flow_cache=flow_cache) if not skip_flow else 0.0,
                "illiq": _score_illiq(df, n=illiq_n, close_arr=close_arr, ret_arr=ret_arr) if not skip_illiq else 0.0,
                "gap": _score_overnight_gap(df, n=overnight_gap_n, close_arr=close_arr) if not skip_overnight_gap else 0.0,
                "custom_factor": float(overlay_scores.get(code, 0.0) or 0.0),
            }

            l0 = (effective_tech_w * scores["tech"] + effective_qlib_w * scores["qlib"] +
                  effective_sent_w * scores["sentiment"] + effective_fundamental_w * scores["fundamental"] +
                  effective_flow_w * scores["flow"] + effective_north_flow_w * scores["north_flow"] +
                  effective_size_turnover_w * scores["size_turnover"] + effective_momentum_w * scores["momentum"] +
                  effective_volatility_w * scores["volatility"] + effective_illiq_w * scores["illiq"] +
                  effective_overnight_gap_w * scores["gap"] + effective_ht_factors_w * scores["ht"] +
                  effective_overlay_w * scores["custom_factor"] +
                  effective_wb_industry_w * scores["wb_industry"] + effective_wb_divergence_w * scores["wb_divergence"] +
                  effective_wb_moneyflow_w * scores["wb_moneyflow"])

            _etf_bonus = 0.0
            if macro_sector_bias and macro_industry_map:
                ind = macro_industry_map.get(code, {})
                if isinstance(ind, dict):
                    ind = ind.get("industry", "")
                if ind:
                    l0 += sector_bias_bonus(ind, macro_sector_bias)
                if ind and etf_flow_bias and ind in etf_flow_bias:
                    _etf_bonus = etf_flow_bias[ind]

            impact_cost = 0.0
            if getattr(config, "BUY_SIDE_IMPACT_ENABLED", True):
                impact_cost = estimate_impact_cost(close, volume)
                l0 = apply_impact_penalty(l0, impact_cost)

            if l0 >= min_score:
                return {
                    "code": code, "name": name,
                    "l0_score": round(l0, 3),
                    "impact_cost": round(impact_cost, 6),
                    "tech_score": round(scores["tech"], 3),
                    "qlib_score": round(scores["qlib"], 3),
                    "sentiment_score": round(scores["sentiment"], 3),
                    "north_flow_score": round(scores["north_flow"], 3),
                    "size_turnover_score": round(scores["size_turnover"], 3),
                    "fundamental_score": round(scores["fundamental"], 3),
                    "flow_score": round(scores["flow"], 3),
                    "illiq_score": round(scores["illiq"], 3),
                    "gap_score": round(scores["gap"], 3),
                    "custom_factor_score": round(scores["custom_factor"], 3),
                    "momentum_score": round(scores["momentum"], 3),
                    "volatility_score": round(scores["volatility"], 3),
                    "etf_flow_score": round(_etf_bonus, 3),
                    "wb_industry_score": round(scores["wb_industry"], 3),
                    "wb_divergence_score": round(scores["wb_divergence"], 3),
                    "wb_moneyflow_score": round(scores["wb_moneyflow"], 3),
                    "close": close, "volume": volume,
                }, df
            return None, "below_threshold"

        def _process_one(code):
            if snapshot is not None:
                entry = snapshot.get(code)
                if entry is None:
                    return None, "data_error"
                df, name, close, volume = entry
                close_arr = df["close"].astype(float)
                ret_arr = close_arr.pct_change(fill_method=None)
                return _score_l0(df, code, name, close, volume, close_arr, ret_arr, self.fetcher)

            # ── 无快照：完整流程 ──
            if not hasattr(_tls, 'fetcher'):
                from core.data.fetcher import DataFetcher
                _tls.fetcher = DataFetcher()
                _tls.analyzer = Analyzer()
            fetcher = _tls.fetcher
            analyzer = _tls.analyzer

            try:
                if is_historical:
                    df = fetcher.get_stock_daily(code, end_date=end_date_str)
                else:
                    df = fetcher.get_stock_daily(code)
            except Exception:
                return None, "data_error"
            if df is None or df.empty:
                return None, "data_error"

            latest = df.iloc[-1]
            name = str(latest.get("name", ""))
            close = float(latest.get("close", 0) or 0)
            volume = float(latest.get("volume", 0) or 0)
            pct = float(latest.get("pct_chg", 0) or 0)
            if pct == 0 and len(df) >= 2:
                prev_close = float(df.iloc[-2].get("close", 0) or 0)
                if prev_close > 0:
                    pct = (close / prev_close - 1) * 100
                    df.at[df.index[-1], "pct_chg"] = pct

            if name.upper().startswith("ST") or name.upper().startswith("*ST"):
                return None, "st"
            if close <= 0 or volume <= 0:
                return None, "suspended"
            if pct >= limit_pct_for_code(code) * 100 - 0.5:
                return None, "limit_up"
            min_turnover = getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 1e7)
            avg_turnover = float(df.tail(20)['amount'].fillna(0).mean())
            if avg_turnover < min_turnover:
                return None, "low_liquidity"
            if self._is_low_price(code, close, df):
                return None, "low_price"

            listing_date = listing_dates.get(code)
            if listing_date is None:
                return None, "identity_unknown"
            if pd.Timestamp(listing_date).date() > cutoff_date:
                return None, "new_listing"

            if not skip_tech:
                try:
                    df = analyzer.add_indicators(df)
                except Exception:
                    pass
            close_arr = df["close"].astype(float)
            ret_arr = close_arr.pct_change(fill_method=None)
            return _score_l0(df, code, name, close, volume, close_arr, ret_arr, fetcher)

        # ═══ 预加载北向资金 + 小市值缓存（一次，所有线程复用）═══
        north_flow_data = {}
        if not skip_north_flow and effective_north_flow_w > 0:
            nf_cache_key = (str(eval_date) if eval_date else "now", id(snapshot) if snapshot is not None else 0)
            with _l0_precomp_lock:
                if nf_cache_key in _l0_precomp_cache and "north_flow" in _l0_precomp_cache[nf_cache_key]:
                    north_flow_data = _l0_precomp_cache[nf_cache_key]["north_flow"]
            if north_flow_data:
                logger.info(f"[L0] 北向资金缓存命中: {len(north_flow_data)} 只")
            else:
                try:
                    from core.data import Database
                    batch_size = 500
                    for bs in range(0, len(codes), batch_size):
                        batch = codes[bs:bs + batch_size]
                        ph = ",".join(["%s"] * len(batch))
                        if eval_date:
                            rows = Database.fetchall(
                                f"SELECT code, net_flow FROM north_flow "
                                f"WHERE code IN ({ph}) AND trade_date = %s",
                                tuple(batch) + (eval_date,),
                            )
                        else:
                            rows = Database.fetchall(
                                f"SELECT code, net_flow FROM north_flow "
                                f"WHERE code IN ({ph}) AND trade_date = CURDATE()",
                                tuple(batch),
                            )
                        for r in rows:
                            north_flow_data[r["code"]] = float(r["net_flow"] or 0)
                    if north_flow_data:
                        logger.info(f"[L0] 北向资金预加载: {len(north_flow_data)} 只")
                    with _l0_precomp_lock:
                        _l0_precomp_put(_l0_precomp_cache, nf_cache_key, "north_flow", north_flow_data)
                except Exception as e:
                    logger.debug(f"[L0] 北向资金缓存写入失败: {e}")
        if not skip_size_turnover:
            _fetch_size_batch(codes, eval_date=eval_date)

        # ── 白盒因子预加载 ──
        industry_map = {}
        industry_returns = {}
        logger.info(f"[L0] 白盒预加载检查: skip_wb_industry={skip_wb_industry} skip_wb_divergence={skip_wb_divergence} skip_wb_moneyflow={skip_wb_moneyflow} eff_w={effective_wb_industry_w:.3f}/{effective_wb_divergence_w:.3f}/{effective_wb_moneyflow_w:.3f}")
        if not skip_wb_industry and effective_wb_industry_w > 0:
            ind_cache_key = (str(eval_date) if eval_date else "now", id(snapshot) if snapshot is not None else 0, wb_industry_n)
            with _l0_precomp_lock:
                if ind_cache_key in _l0_precomp_cache and "industry_returns" in _l0_precomp_cache[ind_cache_key]:
                    cached = _l0_precomp_cache[ind_cache_key]
                    industry_map = cached.get("industry_map", {})
                    industry_returns = cached.get("industry_returns", {})
            if industry_returns:
                logger.info(f"[L0] 行业收益缓存命中: {len(industry_returns)} 个行业 (n={wb_industry_n})")
            else:
                try:
                    industry_map = self.fetcher.get_industry_map()
                    if snapshot:
                        industry_returns = _precompute_industry_returns(snapshot, industry_map, n=wb_industry_n)
                        logger.info(f"[L0] 行业收益预计算: {len(industry_returns)} 个行业 (n={wb_industry_n})")
                        with _l0_precomp_lock:
                            _l0_precomp_put(_l0_precomp_cache, ind_cache_key, "industry_map", industry_map)
                            _l0_precomp_put(_l0_precomp_cache, ind_cache_key, "industry_returns", industry_returns)
                    else:
                        logger.info(f"[L0] 行业映射已加载: {len(industry_map)} 只, 但无snapshot跳过收益预计算")
                except Exception as e:
                    logger.error(f"[L0] 行业数据加载失败: {e}")

        wb_flow_data = {}
        if not skip_wb_moneyflow and effective_wb_moneyflow_w > 0:
            mf_cache_key = (str(eval_date) if eval_date else "now", id(snapshot) if snapshot is not None else 0)
            with _l0_precomp_lock:
                if mf_cache_key in _l0_precomp_cache and "money_flow" in _l0_precomp_cache[mf_cache_key]:
                    wb_flow_data = _l0_precomp_cache[mf_cache_key]["money_flow"]
            if wb_flow_data:
                logger.info(f"[L0] 资金异动缓存命中: {len(wb_flow_data)} 只")
            else:
                try:
                    from core.data import Database
                    batch_size = 500
                    for bs in range(0, len(codes), batch_size):
                        batch = codes[bs:bs + batch_size]
                        ph = ",".join(["%s"] * len(batch))
                        if eval_date:
                            rows = Database.fetchall(
                                f"SELECT code, main_net_pct FROM ("
                                f"SELECT code, main_net_pct, trade_date, "
                                f"ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) rn "
                                f"FROM capital_flow_daily "
                                f"WHERE code IN ({ph}) AND trade_date <= %s"
                                f") t WHERE rn <= 30",
                                tuple(batch) + (eval_date,),
                            )
                        else:
                            rows = Database.fetchall(
                                f"SELECT code, main_net_pct FROM ("
                                f"SELECT code, main_net_pct, trade_date, "
                                f"ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) rn "
                                f"FROM capital_flow_daily "
                                f"WHERE code IN ({ph}) AND trade_date <= CURDATE()"
                                f") t WHERE rn <= 30",
                                tuple(batch),
                            )
                        for r in rows:
                            code = r["code"]
                            pct = float(r.get("main_net_pct", 0) or 0)
                            if code not in wb_flow_data:
                                wb_flow_data[code] = []
                            wb_flow_data[code].append(pct)
                    # 反转列表让最早的在前面
                    for code in wb_flow_data:
                        wb_flow_data[code].reverse()
                    if wb_flow_data:
                        logger.info(f"[L0] 资金异动数据预加载: {len(wb_flow_data)} 只")
                    with _l0_precomp_lock:
                        _l0_precomp_put(_l0_precomp_cache, mf_cache_key, "money_flow", wb_flow_data)
                except Exception as e:
                    logger.debug(f"[L0] 资金流缓存写入失败: {e}")
        # ═══ 批量预加载财务数据（避免每只股票独立DB查询）═══
        if not skip_fundamental and effective_fundamental_w > 0:
            fc_key = (str(eval_date) if eval_date else "now", id(snapshot) if snapshot is not None else 0)
            with _l0_precomp_lock:
                if fc_key in _l0_precomp_cache and "financials" in _l0_precomp_cache[fc_key]:
                    self._financial_cache = _l0_precomp_cache[fc_key]["financials"]
            if self._financial_cache:
                logger.info(f"[L0] 财务缓存命中: {len(self._financial_cache)} 只")
            else:
                try:
                    self._financial_cache = _prefetch_financials(codes)
                    if self._financial_cache:
                        logger.info(f"[L0] 财务预加载: {len(self._financial_cache)} 只")
                    with _l0_precomp_lock:
                        _l0_precomp_put(_l0_precomp_cache, fc_key, "financials", self._financial_cache)
                except Exception as e:
                    logger.debug(f"[L0] 财务缓存写入失败: {e}")
        # ═══ 批量预加载资金流向（避免每只股票独立DB查询）═══
        flow_cache = {}
        if not skip_flow and effective_flow_w > 0:
            fl_key = (str(eval_date) if eval_date else "now", id(snapshot) if snapshot is not None else 0)
            with _l0_precomp_lock:
                if fl_key in _l0_precomp_cache and "capital_flow" in _l0_precomp_cache[fl_key]:
                    flow_cache = _l0_precomp_cache[fl_key]["capital_flow"]
            if flow_cache:
                logger.info(f"[L0] 资金流向缓存命中: {len(flow_cache)} 只")
            else:
                try:
                    from core.data import Database
                    batch_size = 500
                    for bs in range(0, len(codes), batch_size):
                        batch = codes[bs:bs + batch_size]
                        ph = ",".join(["%s"] * len(batch))
                        if eval_date:
                            rows = Database.fetchall(
                                f"SELECT f.code, f.main_net_flow, f.trade_date, q.amount "
                                f"FROM capital_flow_daily f "
                                f"LEFT JOIN daily_quotes q ON q.code=f.code AND q.trade_date=f.trade_date "
                                f"WHERE f.code IN ({ph}) AND f.trade_date <= %s "
                                f"ORDER BY f.code, f.trade_date DESC",
                                tuple(batch) + (eval_date,),
                            )
                        else:
                            rows = Database.fetchall(
                                f"SELECT f.code, f.main_net_flow, f.trade_date, q.amount "
                                f"FROM capital_flow_daily f "
                                f"LEFT JOIN daily_quotes q ON q.code=f.code AND q.trade_date=f.trade_date "
                                f"WHERE f.code IN ({ph}) "
                                f"ORDER BY f.code, f.trade_date DESC",
                                tuple(batch),
                            )
                        for r in rows:
                            code = r["code"]
                            if code not in flow_cache:
                                flow_cache[code] = []
                            flow_cache[code].append({
                                "main_net_flow": float(r.get("main_net_flow") or 0),
                                "amount": float(r.get("amount") or 0),
                            })
                    for code in flow_cache:
                        flow_cache[code] = flow_cache[code][:20]
                    if flow_cache:
                        logger.info(f"[L0] 资金流向预加载: {len(flow_cache)} 只")
                    with _l0_precomp_lock:
                        _l0_precomp_put(_l0_precomp_cache, fl_key, "capital_flow", flow_cache)
                except Exception as e:
                    logger.debug(f"[L0] 资金流向缓存写入失败: {e}")
        logger.info(f"[L0] 并行处理 {total} 只")

        with thread_pool.default.borrow() as executor:
            futures = {executor.submit(_process_one, code): code for code in codes}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    result = (None, None)

                with lock:
                    processed[0] += 1
                    cur = processed[0]
                    if isinstance(result[1], str) and result[1] in excluded:
                        excluded[result[1]] += 1
                    elif result[0]:
                        candidates.append(result[0])
                        if result[1] is not None and not isinstance(result[1], str):
                            self._df_cache[result[0]["code"]] = result[1]
                    if progress_cb and cur % 50 == 0:
                        progress_cb(cur, total)

        if progress_cb:
            progress_cb(total, total)
        candidates.sort(key=lambda x: x["l0_score"], reverse=True)
        below = total - sum(excluded.values()) - len(candidates)
        excluded["below_threshold"] = max(0, below)

        result = {
            "total_input": total,
            "candidates": candidates,
            "excluded_reasons": excluded,
            "weights": {
                "tech": round(effective_tech_w, 3),
                "qlib": round(effective_qlib_w, 3),
                "sentiment": round(effective_sent_w, 3),
                "fundamental": round(effective_fundamental_w, 3),
                "flow": round(effective_flow_w, 3),
                "north_flow": round(effective_north_flow_w, 3),
                "size_turnover": round(effective_size_turnover_w, 3),
                "momentum": round(effective_momentum_w, 3),
                "volatility": round(effective_volatility_w, 3),
                "wb_industry": round(effective_wb_industry_w, 3),
                "wb_divergence": round(effective_wb_divergence_w, 3),
                "wb_moneyflow": round(effective_wb_moneyflow_w, 3),
                "min_score": round(min_score, 3),
            },
        }
        if not skip_sent:
            result["sent_coverage"] = round(sent_coverage, 3)
        if hot_sector_meta:
            result["pool_info"] = hot_sector_meta
        return result

    def _l1_rank(self, candidates: List[Dict], progress_cb=None) -> List[Dict]:
        """L1 精排 — 委托给 rank_l1 (纯 L3 函数)。"""
        return rank_l1(candidates, progress_cb)

    def _find_best_qlib_model(self, regime=None):
        """扫描 mlruns 找最佳模型, 返回 (exp_id, run_id) 或 None。
        选择顺序:
        1. 达标模型 (ICIR≥QLIB_MODEL_MIN_ICIR 且 月度IC胜率≥QLIB_MODEL_MIN_IC_WIN_RATE):
           滚动训练 (rolling_* 实验) anchor 最新一轮优先, 同轮内 IC 最高
           regime 不为 None 时, 按市场状态偏好乘数调整 IC 排序
        2. 无达标者退回全局 IC 最高 (老模型无扩展指标时的兼容路径) + warning
        """
        from pathlib import Path
        from core.shared import config as app_config
        from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE, get_model_label_mode
        from core.adapters.qlib.models import _read_run_metric
        from core.shared.disabled_models import is_disabled

        mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", ""))
        if not mlruns_dir.exists():
            return None

        min_icir = float(getattr(app_config, "QLIB_MODEL_MIN_ICIR", 0.3))
        min_win = float(getattr(app_config, "QLIB_MODEL_MIN_IC_WIN_RATE", 0.55))

        best_ic = -999
        best_model = None
        fallback_dual = None
        regime_str = regime.value if hasattr(regime, 'value') else regime
        qualified = []  # (anchor, score, (exp_id, run_id), model_type), anchor 非滚动实验为 ""
        for exp_dir in mlruns_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "0":
                continue
            exp_name = _read_exp_name(exp_dir)
            anchor = exp_name.rsplit("_", 1)[-1] if exp_name.startswith("rolling_") else ""
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if is_disabled(exp_dir.name, run_dir.name):
                    continue
                if get_model_label_mode(run_dir) != TRADABLE_LABEL_MODE:
                    continue
                artifacts = run_dir / "artifacts"
                is_dual = (artifacts / "params_lgb.pkl").exists() and (artifacts / "params_xgb.pkl").exists()
                is_single = (artifacts / "params.pkl").exists()
                if not is_single and not is_dual:
                    continue
                # 单/双模型统一读 metrics/IC 比较
                ic = _read_run_metric(run_dir, "IC")
                if ic is not None:
                    if ic > best_ic:
                        best_ic = ic
                        best_model = (exp_dir.name, run_dir.name)
                    icir = _read_run_metric(run_dir, "ICIR")
                    win = _read_run_metric(run_dir, "IC_WIN_RATE_M")
                    if (icir is not None and win is not None
                            and icir >= min_icir and win >= min_win):
                        model_type = _get_model_type(exp_name, artifacts)
                        bonus = _get_regime_bonus(model_type, regime_str)
                        score = ic * bonus if regime_str else ic
                        qualified.append((anchor, score, (exp_dir.name, run_dir.name), model_type, bonus))
                elif is_dual and fallback_dual is None:
                    # 双模型无IC时的降级方案
                    fallback_dual = (exp_dir.name, run_dir.name)

        if qualified:
            # anchor 字典序最大 = 最新滚动轮次; 非滚动 ("") 排最后; 同 anchor 取 score 最高
            qualified.sort(key=lambda x: (x[0], x[1]), reverse=True)
            anchor, score, model, model_type, bonus = qualified[0]
            bonus_info = f" regime={regime_str} bonus={bonus:.2f}" if regime_str else ""
            logger.info(f"[Qlib] 达标模型: {model[0]}/{model[1]} score={score:.4f} "
                        f"type={model_type} anchor={anchor or '-'} "
                        f"({len(qualified)} 个达标){bonus_info}")
            return model

        if best_model is None:
            best_model = fallback_dual
        if best_model:
            logger.warning(f"[Qlib] 无达标模型(ICIR≥{min_icir} 且 月胜率≥{min_win}), "
                           f"退回IC最高: {best_model[0]}/{best_model[1]} IC={best_ic:.4f}")
        return best_model

    def _find_model_by_direction(self, direction):
        """按方向精选集匹配模型(训练因子集重合≥80%), 同方向取 IC 最优。返回 (exp_id, run_id) 或 None"""
        from pathlib import Path
        from core.shared import config as app_config
        from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE, get_model_label_mode
        from core.adapters.qlib.models import _read_run_metric
        from core.shared.disabled_models import is_disabled

        try:
            from core.domain.factors.registry import get_pool
            curated = set(get_pool().get_curated_factors(direction) or [])
        except Exception:
            curated = set()
        if not curated:
            return None

        mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", ""))
        if not mlruns_dir.exists():
            return None

        best = None
        best_ic = -999
        for exp_dir in mlruns_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "0":
                continue
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if is_disabled(exp_dir.name, run_dir.name):
                    continue
                if get_model_label_mode(run_dir) != TRADABLE_LABEL_MODE:
                    continue
                artifacts = run_dir / "artifacts"
                if not (artifacts / "params.pkl").exists() and not (artifacts / "params_lgb.pkl").exists():
                    continue
                kw = _restore_trained_handler_kwargs(artifacts, run_dir / "params")
                fk = set(kw.get("factor_keys") or [])
                if not fk or len(fk & curated) / len(fk) < 0.8:
                    continue
                ic = _read_run_metric(run_dir, "IC")
                ic_cmp = ic if ic is not None else -998
                if ic_cmp > best_ic:
                    best_ic = ic_cmp
                    best = (exp_dir.name, run_dir.name)
        if best:
            logger.info(f"[SectorFusion] 方向 {direction} 模型: {best[0]}/{best[1]} IC={best_ic:.4f}")
        return best

    def _apply_sector_fusion(self, qlib_scores, eval_date, pool, market_rr=None,
                             quotes_df=None, dir_scores=None, neutralize_qlib=None):
        """板块背离双模型分位融合。

        返回 (fused_scores, fusion_info)；任何异常降级返回原分。
        fusion_info: {"divergent": {行业: {...}}, "models": {方向: "exp/run"},
                      "attr": {code: 归因}, "escort": [保送候选]}
        quotes_df: 预拉取的全市场 daily_quotes (回测复用)。
        dir_scores: 预计算的 {方向: {code: score}} (回测复用 pred_cache, 避免现场预测)。
        """
        from core.shared import config as app_config
        info = {"divergent": {}, "models": {}, "attr": {}, "escort": []}
        try:
            from core.domain.regime.sector import get_divergent_sectors
            div = get_divergent_sectors(as_of=eval_date, market_regime=market_rr, quotes_df=quotes_df)
            if not div:
                return qlib_scores, info
            info["divergent"] = div

            # 每个背离方向: 匹配模型B → 全市场预测(分位分, 与A同标尺)
            b_scores_map = {}
            for direction in sorted({v["direction"] for v in div.values()}):
                if dir_scores and direction in dir_scores:
                    # 复用主循环已预加载的方向模型截面 (避免现场 DatasetH 预测)
                    # 归一化到分位分, 与 A 分及实时路径的 B 分同标尺
                    b_scores_map[direction] = normalize_qlib_scores(
                        dir_scores[direction], eval_date, neutralize=neutralize_qlib)
                    info["models"][direction] = direction
                    continue
                model = self._find_model_by_direction(direction)
                if not model:
                    logger.info(f"[SectorFusion] 方向 {direction} 无匹配模型, 该方向不融合")
                    continue
                scores_b = self._load_qlib_predictions(
                    model, as_of=eval_date, pool=pool,
                    neutralize_qlib=neutralize_qlib)
                if scores_b:
                    b_scores_map[direction] = scores_b
                    info["models"][direction] = f"{model[0]}/{model[1]}"
            if not b_scores_map:
                return qlib_scores, info

            ind_map_raw = self.fetcher.get_industry_map() or {}
            lam = float(getattr(app_config, "SECTOR_FUSION_LAMBDA", 0.2))
            fused = dict(qlib_scores)
            n_fused = 0
            for code, a in qlib_scores.items():
                v = ind_map_raw.get(code)
                ind = v.get("industry") if isinstance(v, dict) else None
                if not ind or ind not in div:
                    continue
                d = div[ind]
                b_scores = b_scores_map.get(d["direction"])
                if not b_scores or code not in b_scores:
                    continue
                c = d["confidence"]
                # 三档先验权重: c<0.5→0 / 0.5~0.7→0.3 / ≥0.7→0.5 (市场模型保底一半)
                w = 0.5 if c >= 0.7 else (0.3 if c >= 0.5 else 0.0)
                if w <= 0:
                    continue
                b = b_scores[code]
                penalty = lam * abs(a - b)
                fused[code] = max(-1.0, min(1.0, (1 - w) * a + w * b - penalty))
                info["attr"][code] = {
                    "sector": ind, "sector_regime": d["regime"],
                    "confidence": c, "direction": d["direction"],
                    "market_pct": round(a, 3), "sector_pct": round(b, 3),
                    "w_sector": w, "penalty": round(penalty, 3),
                }
                n_fused += 1

            # 保送候选: 各模型B在其背离板块内的分位 top3 (总量在候选组装处限 10)
            for direction, sb in b_scores_map.items():
                dir_inds = {ind for ind, d in div.items() if d["direction"] == direction}
                in_sector = []
                for code, s in sb.items():
                    v = ind_map_raw.get(code)
                    ind = v.get("industry") if isinstance(v, dict) else None
                    if ind in dir_inds:
                        in_sector.append((code, s, ind))
                for code, s, ind in sorted(in_sector, key=lambda x: -x[1])[:3]:
                    info["escort"].append({"code": code, "direction": direction,
                                           "sector": ind, "pct": round(s, 3)})

            logger.info(f"[SectorFusion] 融合 {n_fused} 只 ({len(div)} 背离板块, "
                        f"{len(b_scores_map)} 个方向模型), 保送候选 {len(info['escort'])}")
            return fused, info
        except Exception as e:
            logger.warning(f"[SectorFusion] 融合失败, 降级单模型: {e}")
            return qlib_scores, {"divergent": {}, "models": {}, "attr": {}, "escort": []}

    def _load_qlib_predictions(self, qlib_model, as_of=None, pool="csi300",
                               neutralize_qlib=None, return_details=False) -> dict:
        """根据模型标识加载/生成 Qlib 预测 {code: score}

        as_of: 可选 datetime.date，将因子计算截止日设为此日期，避免前视偏差。
        pool: 股票池 ("csi300"|"all"|"non_csi300")，决定 instruments 文件减少加载量。
        """
        import pickle
        from pathlib import Path
        from core.shared import config as app_config
        from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE, get_model_label_mode

        exp_id, run_id = qlib_model

        from core.shared.disabled_models import is_disabled
        if is_disabled(exp_id, run_id):
            logger.warning(f"[Qlib] 模型已停用: {exp_id}/{run_id}")
            return {}

        mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", ""))
        if get_model_label_mode(mlruns_dir / exp_id / run_id) != TRADABLE_LABEL_MODE:
            logger.warning(f"[Qlib] 拒绝非开盘到开盘标签模型: {exp_id}/{run_id}")
            return {}
        artifacts = mlruns_dir / exp_id / run_id / "artifacts"
        from core.adapters.qlib.fundval import require_run_contract, feature_contract
        trained_keys = require_run_contract(artifacts)
        uses_fundval = feature_contract(trained_keys)['requires_fundval']
        model_path = artifacts / "params.pkl"
        is_fusion = model_path.exists() and (artifacts / "params_lstm.pkl").exists()
        is_dual = not model_path.exists() and (artifacts / "params_lgb.pkl").exists()

        if not model_path.exists() and not is_dual:
            logger.warning(f"Qlib 模型不存在: {model_path}")
            return {}

        # ── as_of 给定(批量/历史场景)才读 DB 缓存；实时选股始终现场预测 ──
        import datetime as _dt
        model_prefix = f"{exp_id}/{run_id}"
        ref_date = as_of or today_date()
        requested_pool_codes = None
        if str(pool) != "all":
            pool_codes = set()
            pool_aliases = {"csi300": "csi300", "a500": "a500", "stable_history_500d": "a500", "small_cap": "small_cap"}
            for pool_name in str(pool).split("+"):
                pool_name = pool_name.strip()
                instrument_file = Path(app_config.QLIB_DATA_DIR) / "instruments" / f"{pool_aliases.get(pool_name, pool_name)}.txt"
                if not instrument_file.exists():
                    continue
                for line in load_text(str(instrument_file)).splitlines():
                    code = line.strip().split("\t")[0]
                    if code[:2].lower() in ("sh", "sz"):
                        code = code[2:]
                    if code:
                        pool_codes.add(code)
            if pool_codes:
                requested_pool_codes = pool_codes
        if as_of is not None and not uses_fundval:
            try:
                from core.data import Database
                # run_id 可能被前端截断, 用 LIKE 前缀匹配
                rows = Database.fetchall(
                    "SELECT model_id, code, score FROM qlib_predictions "
                    "WHERE model_id LIKE %s AND trade_date=%s",
                    (f"{model_prefix}%", ref_date),
                )
                if rows:
                    model_ids = sorted({str(r["model_id"]) for r in rows})
                    preferred = [f"{model_prefix}/dual", model_prefix] if is_dual else [model_prefix, f"{model_prefix}/dual"]
                    selected_model_id = next((value for value in preferred if value in model_ids), None)
                    if selected_model_id is None and len(model_ids) == 1:
                        selected_model_id = model_ids[0]
                    if selected_model_id is None:
                        logger.warning(f"[Qlib] DB预测命中多个模型标识, 拒绝模糊合并: {model_ids}")
                    else:
                        scores_db = {
                            r["code"]: float(r["score"])
                            for r in rows if str(r["model_id"]) == selected_model_id
                            and (requested_pool_codes is None or r["code"] in requested_pool_codes)
                        }
                        logger.info(f"[Qlib] 从DB加载预测: {len(scores_db)}只, date={ref_date}")
                        details = normalize_qlib_scores(
                            scores_db, ref_date, neutralize=neutralize_qlib,
                            return_details=True,
                            model_input_semantics=(
                                "daily_cross_sectional_rank_fusion" if is_fusion
                                else "raw_model_score"
                            ))
                        details["audit"]["context"].update({
                            "source": "database",
                            "model_id": selected_model_id,
                            "pool": str(pool),
                            "prediction_date": str(ref_date)[:10],
                            "pool_member_count": len(requested_pool_codes) if requested_pool_codes else None,
                        })
                        return details if return_details else details["scores"]
            except Exception as e:
                logger.warning(f"[Qlib] DB预测查询失败: {e}")

        # instruments 优先用股票池对应的文件（减少不必要的数据加载）
        instr_by_pool = {"csi300": "csi300", "non_csi300": "all", "all": "all", "a500": "a500", "stable_history_500d": "a500", "small_cap": "small_cap"}
        if "+" in str(pool):
            parts = str(pool).split("+"); combined = []
            for p in parts:
                p = p.strip()
                try:
                    fpath = Path(app_config.QLIB_DATA_DIR) / "instruments" / f"{instr_by_pool.get(p, p)}.txt"
                    if fpath.exists():
                        for line in load_text(str(fpath)).splitlines():
                            c = line.strip().split("\t")[0]
                            if c and c not in combined: combined.append(c)
                except Exception: pass
            instruments_str = combined if combined else "all"
        else:
            instruments_str = instr_by_pool.get(pool, "all")

        # 从 run params 提取 factor_set 和 instruments (覆盖默认)
        factor_set = "Alpha158"
        params_dir = mlruns_dir / exp_id / run_id / "params"
        if params_dir.exists():
            for pf in params_dir.iterdir():
                try:
                    if pf.name == "dataset.kwargs.handler.class":
                        factor_set = load_text(str(pf)).strip()
                    elif pf.name == "dataset.kwargs.handler.kwargs.instruments":
                        raw = load_text(str(pf)).strip()
                        prev = instruments_str
                        instruments_str = _resolve_instruments_param(raw, instruments_str, len(raw))
                        if instruments_str != prev:
                            logger.info(f"[Qlib] instruments 解析: {instruments_str} (raw={raw[:80]}...)")
                except Exception:
                    pass

        # WhiteBox 模型必须还原训练时的 factor_keys/processors，否则特征数与训练不一致预测报错
        extra_handler_kwargs = {}
        if "WhiteBox" in factor_set:
            extra_handler_kwargs = _restore_trained_handler_kwargs(artifacts, params_dir)
            fk = extra_handler_kwargs.get("factor_keys")
            if fk:
                procs = [k for k in extra_handler_kwargs if k != "factor_keys"]
                logger.info(f"[Qlib] 还原训练因子集: {len(fk)} 个因子, processors={procs}")
            else:
                logger.warning("[Qlib] WhiteBox 模型未找到 factor_keys，预测可能因特征数不符失败")

        try:
            from core.adapters.qlib.config import init_qlib
            init_qlib()

            import datetime
            ref_date = as_of if as_of else today_date()
            end_date = ref_date.strftime("%Y-%m-%d")
            # 钉到 Qlib 日历中 ≤ ref_date 的最新交易日
            try:
                from qlib.data import D
                cal = D.calendar(end_time=end_date)
                if len(cal):
                    end_date = pd.Timestamp(cal[-1]).strftime("%Y-%m-%d")
            except Exception:
                pass
            start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            if as_of is None:
                logger.info(f"[Qlib] 现场预测: 最新交易日 {end_date} 数据 → 预测下一交易日")

            # 检查预测缓存（同日同模型同池子结果不变，跳过 2-4 分钟 DatasetH 重建）
            _pred_cache_key = None
            if as_of is None:
                neutralize_cache_flag = (getattr(app_config, "QLIB_PRED_NEUTRALIZE", False)
                                         if neutralize_qlib is None else bool(neutralize_qlib))
                from core.adapters.qlib.fundval import current_feature_contract
                import json as _json
                feature_signature = _json.dumps(current_feature_contract(trained_keys), sort_keys=True)
                _pred_cache_key = (SCORE_PIPELINE_VERSION, exp_id, run_id, pool,
                                   end_date, neutralize_cache_flag, feature_signature)
                with _qlib_pred_lock:
                    entry = _qlib_pred_cache.get(_pred_cache_key)
                    if entry and time.time() - entry[0] < _QLIB_PRED_CACHE_TTL:
                        cached = entry[1]
                        cached_scores = cached.get("scores", {}) if isinstance(cached, dict) else {}
                        logger.info(f"[Qlib] 预测缓存命中: {len(cached_scores)}只, date={end_date}")
                        cached_copy = clone_score_result(cached)
                        return cached_copy if return_details else dict(cached_scores)

            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": factor_set,
                        "module_path": "core.adapters.qlib.custom_handler" if "WhiteBox" in factor_set else "qlib.contrib.data.handler",
                        "kwargs": {
                            "instruments": instruments_str,
                            "start_time": start_date,
                            "end_time": end_date,
                            "fit_start_time": start_date,
                            "fit_end_time": end_date,
                            **extra_handler_kwargs,
                        },
                    },
                    "segments": {"test": (start_date, end_date)},
                },
            }

            def _extract_scores(pred):
                s = {}
                if hasattr(pred, "reset_index"):
                    pred_df = pred.reset_index()
                    if len(pred_df.columns) >= 3:
                        pred_df.columns = ["datetime", "instrument", "score"]
                    if "datetime" in pred_df.columns:
                        # 按日期升序，同一股票以最新交易日截面为准
                        pred_df = pred_df.sort_values("datetime")
                    for row in pred_df.itertuples(index=False):
                        inst = str(row.instrument)
                        if inst.startswith("sh") or inst.startswith("sz"):
                            inst = inst[2:]
                        s[inst] = float(row.score)
                elif hasattr(pred, "items"):
                    for k, v in pred.items():
                        inst = str(k)
                        if inst.startswith("sh") or inst.startswith("sz"):
                            inst = inst[2:]
                        s[inst] = float(v)
                return s

            if is_fusion:
                from core.adapters.qlib.lstm_fusion import predict_lgb_lstm_fusion
                pred, _ = predict_lgb_lstm_fusion(
                    exp_id, run_id, end_date, end_date, instruments=instruments_str)
                scores = _extract_scores(pred)
                logger.info(f"[Qlib] LGB+LSTM 真实秩融合: {len(scores)}只")
            elif is_dual:
                # 双模型：LGB + XGB 等权融合
                from core.adapters.qlib.config import build_dataset_cached
                dataset = build_dataset_cached(dataset_config)
                from core.shared.persistence import load_pickle
                model_lgb = load_pickle(str(artifacts / "params_lgb.pkl"))
                model_xgb = load_pickle(str(artifacts / "params_xgb.pkl"))
                if model_lgb is None or model_xgb is None:
                    return {}
                pred_lgb = model_lgb.predict(dataset)
                pred_xgb = model_xgb.predict(dataset)
                scores_lgb = _extract_scores(pred_lgb)
                scores_xgb = _extract_scores(pred_xgb)
                # 按训练时记录的 IC 权重融合 (老模型无记录退回等权)
                from core.adapters.qlib.models import dual_fusion_weights_from_run, _read_run_metric
                run_dir = mlruns_dir / exp_id / run_id
                w_lgb, w_xgb = dual_fusion_weights_from_run(run_dir)
                # 可选 linear 第三模型 (训练时记录了 LINEAR_W 才参与)
                scores_lin = {}
                w_lin = _read_run_metric(run_dir, "LINEAR_W") or 0.0
                lin_path = artifacts / "params_linear.pkl"
                if w_lin > 0 and lin_path.exists():
                    try:
                        from core.shared.persistence import load_pickle
                        model_lin = load_pickle(str(lin_path))
                        scores_lin = _extract_scores(model_lin.predict(dataset))
                    except Exception as e:
                        logger.warning(f"[Qlib] Linear 预测失败, 双模型融合: {e}")
                        w_lin = 0.0
                all_codes = set(scores_lgb.keys()) | set(scores_xgb.keys())
                scores = {}
                for code in all_codes:
                    s_lgb = scores_lgb.get(code, 0)
                    s_xgb = scores_xgb.get(code, 0)
                    base = w_lgb * s_lgb + w_xgb * s_xgb
                    if w_lin > 0 and code in scores_lin:
                        base = (1 - w_lin) * base + w_lin * scores_lin[code]
                    scores[code] = base
                logger.info(f"[Qlib] 多模型融合: LGB={len(scores_lgb)}只 XGB={len(scores_xgb)}只 "
                            f"→ {len(scores)}只 (w_lgb={w_lgb:.2f} w_xgb={w_xgb:.2f} w_lin={w_lin:.2f})")
            else:
                from core.adapters.qlib.config import build_dataset_cached
                dataset = build_dataset_cached(dataset_config)
                from core.shared.persistence import load_pickle
                model = load_pickle(str(model_path))
                if model is None:
                    return {}
                pred = model.predict(dataset)
                scores = _extract_scores(pred)

            if scores:
                result = normalize_qlib_scores(
                    scores, ref_date, neutralize=neutralize_qlib,
                    return_details=True,
                    model_input_semantics=(
                        "daily_cross_sectional_rank_fusion" if is_fusion
                        else "raw_model_score"
                    ))
                result["audit"]["context"].update({
                    "source": "live_model",
                    "model_id": model_prefix,
                    "pool": str(pool),
                    "prediction_date": end_date,
                })
                # 写入预测缓存（在线路径，同日内复用）
                if _pred_cache_key and result["scores"]:
                    with _qlib_pred_lock:
                        _qlib_pred_cache[_pred_cache_key] = (time.time(), clone_score_result(result))
                return result if return_details else result["scores"]
        except Exception as e:
            logger.warning(f"Qlib 预测加载失败: {e}")

        return {}

    def _load_qlib_predictions_bulk(self, qlib_model, eval_dates, pool="csi300",
                                    neutralize_qlib=None) -> dict:
        """一次性加载多个日期的 Qlib 预测，返回 {date_iso: {code: score}}

        eval_dates: 需要预测的日期列表 (datetime.date)
        只做一次数据加载，避免每日期重复初始化 dataset（省 600s×N）
        """
        if not eval_dates:
            return {}
        import pickle
        from pathlib import Path
        from core.shared import config as app_config
        from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE, get_model_label_mode

        exp_id, run_id = qlib_model
        mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", ""))
        if get_model_label_mode(mlruns_dir / exp_id / run_id) != TRADABLE_LABEL_MODE:
            logger.warning(f"[Qlib Bulk] 拒绝非开盘到开盘标签模型: {exp_id}/{run_id}")
            return {}
        artifacts_dir = mlruns_dir / exp_id / run_id / "artifacts"
        model_path = artifacts_dir / "params.pkl"
        is_dual = (not model_path.exists() and
                   (artifacts_dir / "params_lgb.pkl").exists() and
                   (artifacts_dir / "params_xgb.pkl").exists())
        if not model_path.exists() and not is_dual:
            return {}
        if is_dual:
            logger.info(f"[Qlib Bulk] 双模型检测到, 回退逐日融合加载 ({len(eval_dates)}个日期)")
            result = {}
            details_by_date = {}
            for d in eval_dates:
                try:
                    details = self._load_qlib_predictions(
                        qlib_model, as_of=d, pool=pool,
                        neutralize_qlib=neutralize_qlib, return_details=True)
                    scores = details.get("scores", {}) if isinstance(details, dict) else {}
                    if scores:
                        date_key = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                        result[date_key] = scores
                        details_by_date[date_key] = details
                except Exception as e:
                    logger.warning(f"[Qlib Bulk] 日期 {d} 预测失败: {e}")
            self._qlib_score_details_by_date = details_by_date
            return result

        instr_by_pool = {"csi300": "csi300", "non_csi300": "all", "all": "all", "a500": "a500", "stable_history_500d": "a500", "small_cap": "small_cap"}
        if "+" in str(pool):
            parts = str(pool).split("+"); combined = []
            for p in parts:
                p = p.strip()
                try:
                    fpath = Path(app_config.QLIB_DATA_DIR) / "instruments" / f"{instr_by_pool.get(p, p)}.txt"
                    if fpath.exists():
                        for line in load_text(str(fpath)).splitlines():
                            c = line.strip().split("\t")[0]
                            if c and c not in combined: combined.append(c)
                except Exception: pass
            instruments_str = combined if combined else "all"
        else:
            instruments_str = instr_by_pool.get(pool, "all")
        factor_set = "Alpha158"

        params_dir = mlruns_dir / exp_id / run_id / "params"
        if params_dir.exists():
            for pf in params_dir.iterdir():
                try:
                    if pf.name == "dataset.kwargs.handler.class":
                        factor_set = load_text(str(pf)).strip()
                    elif pf.name == "dataset.kwargs.handler.kwargs.instruments":
                        raw = load_text(str(pf)).strip()
                        instruments_str = _resolve_instruments_param(raw, instruments_str, len(raw))
                except Exception:
                    pass

        # WhiteBox 模型还原训练时的 factor_keys/processors（与 _load_qlib_predictions 一致）
        extra_handler_kwargs = {}
        if "WhiteBox" in factor_set:
            extra_handler_kwargs = _restore_trained_handler_kwargs(
                mlruns_dir / exp_id / run_id / "artifacts", params_dir)

        try:
            from core.shared.persistence import load_pickle
            model = load_pickle(str(model_path))
            if model is None:
                return {}

            from core.adapters.qlib.config import init_qlib
            init_qlib()

            sorted_dates = sorted(eval_dates)
            start_date = (sorted_dates[0] - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = sorted_dates[-1].strftime("%Y-%m-%d")

            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": factor_set,
                        "module_path": "core.adapters.qlib.custom_handler" if "WhiteBox" in factor_set else "qlib.contrib.data.handler",
                        "kwargs": {
                            "instruments": instruments_str,
                            "start_time": start_date,
                            "end_time": end_date,
                            "fit_start_time": start_date,
                            "fit_end_time": end_date,
                            **extra_handler_kwargs,
                        },
                    },
                    "segments": {"test": (start_date, end_date)},
                },
            }

            from core.adapters.qlib.config import build_dataset_cached
            dataset = build_dataset_cached(dataset_config)
            pred = model.predict(dataset)

            # 将预测结果按日期分组
            by_date = {}
            if hasattr(pred, "reset_index"):
                pred_df = pred.reset_index()
                if len(pred_df.columns) >= 3:
                    pred_df.columns = ["datetime", "instrument", "score"]
                for row in pred_df.itertuples(index=False):
                    dt = row.datetime
                    if hasattr(dt, "strftime"):
                        date_str = dt.strftime("%Y-%m-%d")
                    else:
                        date_str = str(dt)[:10]
                    inst = str(row.instrument)
                    if inst.startswith("sh") or inst.startswith("sz"):
                        inst = inst[2:]
                    s = float(row.score)
                    if date_str not in by_date:
                        by_date[date_str] = {}
                    if inst not in by_date[date_str] or abs(s) > abs(by_date[date_str][inst]):
                        by_date[date_str][inst] = s
            elif hasattr(pred, "items"):
                for k, v in pred.items():
                    inst = str(k)
                    if inst.startswith("sh") or inst.startswith("sz"):
                        inst = inst[2:]
                    date_str = end_date
                    if date_str not in by_date:
                        by_date[date_str] = {}
                    by_date[date_str][inst] = float(v)
            else:
                logger.warning(f"[Qlib Bulk] pred 类型无法处理: {type(pred)}")
                return {}

            # 归一化 (与单日期路径一致: 逐日中性化 → 分位映射)
            result = {}
            details_by_date = {}
            for date_str, scores in by_date.items():
                if scores:
                    try:
                        d = datetime.date.fromisoformat(date_str)
                    except ValueError:
                        d = sorted_dates[-1]
                    details = normalize_qlib_scores(
                        scores, d, neutralize=neutralize_qlib,
                        return_details=True)
                    details["audit"]["context"].update({
                        "source": "bulk_model",
                        "model_id": f"{exp_id}/{run_id}",
                        "pool": str(pool),
                    })
                    result[date_str] = details["scores"]
                    details_by_date[date_str] = details

            self._qlib_score_details_by_date = details_by_date

            logger.info(f"[Qlib Bulk] 一次加载 {len(eval_dates)} 日期 → 命中 {len(result)} 日期, "
                       f"instruments={instruments_str}")
            return result
        except Exception as e:
            logger.warning(f"Qlib 批量预测加载失败: {e}")
            return {}

    def _l2_debate(self, top_candidates: List[Dict], debate_agents=6, progress_cb=None, eval_date=None) -> List[Dict]:
        """L2: 仅对 Top-N 启动 AI 辩论；progress_cb(current, total) 回报进度。
        eval_date 给定时 (历史回测) 用截止当日的数据, 避免用未来 K 线/财务喂 LLM。"""
        total = len(top_candidates)
        from agents.panel import DebatePanel
        from core.shared.infrastructure.llm import LLMClient
        from core.shared.infrastructure.cache import CacheManager

        cache = CacheManager()
        llm = LLMClient()
        panel = DebatePanel(llm=llm, cache=cache, agent_count=debate_agents)

        recommended = []
        for idx, candidate in enumerate(top_candidates):
            if progress_cb:
                progress_cb(idx + 1, total)
            code = candidate["code"]
            name = candidate.get("name", code)
            try:
                df = None if eval_date else self._df_cache.get(code)
                if df is None:
                    if eval_date is not None:
                        ed = eval_date.strftime("%Y%m%d") if hasattr(eval_date, "strftime") else str(eval_date).replace("-", "")
                        df = self.fetcher.get_stock_daily(code, end_date=ed)
                    else:
                        df = self.fetcher.get_stock_daily(code)
                    if df is not None and not df.empty:
                        df = self.analyzer.add_indicators(df)
                if df is None or df.empty:
                    continue
                financial = self.fetcher.get_stock_financial(code) or {}

                ctx = self.analyzer.get_analysis_context(df, financial, indicators_added=True)
                ctx += f"\n\n股票名称: {name}\n股票代码: {code}"
                ctx += f"\n选股综合分: {candidate.get('l1_score', 0)}"
                ctx += f"\n策略投票: {candidate.get('buy_count',0)}买/{candidate.get('sell_count',0)}卖"
                ctx += f"\n技术分: {candidate.get('tech_score',0)}, Qlib分: {candidate.get('qlib_score',0)}"

                debate_result = panel.debate(name, code, ctx)
                final = debate_result.get("最终决议", "")

                risk_score, risk_level = 5, "中风险"
                opportunity_score = 5
                import re
                rm = re.search(r"风险评分:\s*(\d+)/10\s*\n\s*风险等级:\s*(高风险|中风险|低风险)", final)
                if rm:
                    risk_score = int(rm.group(1))
                    risk_level = rm.group(2)
                om = re.search(r"机会评分:\s*(\d+)/10", final)
                if om:
                    opportunity_score = int(om.group(1))

                if opportunity_score >= 7 and risk_score <= 5:
                    level = "★★★ 强烈推荐"
                elif opportunity_score >= 6 and risk_score <= 6:
                    level = "★★ 推荐"
                elif opportunity_score >= 5:
                    level = "★ 关注"
                else:
                    level = "暂不推荐"

                fusion = MetaSignalFusion(mode="buy_side")
                fusion.add("tech_fusion", candidate.get("tech_score", 0), 0.8)
                fusion.add("qlib", candidate.get("qlib_score", 0),
                           getattr(config, "QLIB_DEFAULT_IC", 0.5))
                fusion.add("strategy_cluster", candidate.get("strategy_score", 0), 0.7)
                fusion.add("sentiment", candidate.get("sentiment_score", 0), 0.5)
                debate_norm = (opportunity_score - risk_score) / 10
                fusion.add("ai_debate", debate_norm, 0.85)
                fused = fusion.fuse()

                candidate["debate_summary"] = final[:500]
                agent_reports = []
                for k, v in debate_result.items():
                    if k == "最终决议":
                        continue
                    if v and len(str(v)) > 10:
                        agent_reports.append({"name": k, "content": str(v)})
                candidate["debate_detail"] = {
                    "final": final,
                    "agents": agent_reports,
                }
                candidate["risk_score"] = risk_score
                candidate["opportunity_score"] = opportunity_score
                candidate["recommendation_level"] = level
                candidate["final_score"] = fused.score
                candidate["confidence"] = fused.confidence
                candidate["signal_breakdown"] = {
                    v.source: {
                        "score": round(v.score, 2),
                        "direction": "bullish" if v.direction > 0 else "bearish" if v.direction < 0 else "neutral",
                        "detail": v.detail.get("reason", "") if v.detail else "",
                    }
                    for v in fused.votes
                }
                recommended.append(candidate)
            except Exception as e:
                logger.error(f"L2 辩论失败 {code}: {e}")
                continue

        recommended.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        for i, rec in enumerate(recommended):
            rec["rank"] = i + 1
        return recommended



_tls = threading.local()

def get_screener() -> BuySideScreener:
    """返回 BuySideScreener。多线程模式下每线程独立实例（避免 _df_cache 竞争）。"""
    tls = getattr(_tls, 'screener', None)
    if tls is not None:
        return tls
    inst = BuySideScreener()
    _tls.screener = inst
    return inst


def use_thread_local_screener():
    """多线程批量任务入口：每个线程创建独立 BuySideScreener 实例"""
    _tls.screener = BuySideScreener()


def clear_thread_local_screener():
    """清理线程本地 screener"""
    _tls.screener = None
