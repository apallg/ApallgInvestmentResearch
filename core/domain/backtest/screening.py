"""
选股管线回测引擎

直接调 BuySideScreener.screen(eval_date=date) 获取完整多信号融合评分（l0_score），
保证回测逻辑与选股页面 100% 一致。

与 direction_backtest.py 的区别：
- direction_backtest: 市场状态→方向模型→Qlib原始预测分→排序 (仅测Qlib能力)
- screening_backtest: 市场状态→regime权重→多信号融合→ICIR→因子池门控→l0_score→排序 (实盘管线)
"""

import logging
import datetime as dt
import time
import numpy as np
import pandas as pd

from core.domain.backtest.common import (
    lookup_price, get_benchmark_return_fast,
    precompute_limit_up, precompute_limit_down, initial_build,
)
from core.domain.backtest.metrics import compute_backtest_metrics
from core.domain.portfolio.allocator import (
    PositionAllocator,
    AllocationConfig,
    build_allocation_config,
)
from core.shared import get_freq_meta

logger = logging.getLogger(__name__)


class ScreeningBacktestRunner:

    def __init__(self):
        from core.adapters.qlib.config import init_qlib
        init_qlib()
        self._timing_model = None
        self._screener = None

    def _get_timing_model(self):
        if self._timing_model is None:
            from core.domain.timing.model import get_model
            self._timing_model = get_model()
        return self._timing_model

    def _get_screener(self):
        if self._screener is None:
            from core.domain.screening._screener import get_screener
            self._screener = get_screener()
        return self._screener

    def run(self, start, end, pool="a500", top_n=10, freq="week",
            use_timing=True, macro_bias=False, sector_fusion=None,
            source_weights=None, qlib_model=None, code_prefixes=None,
            factor_overlay=None,
            progress_cb=None, cancel_cb=None,
            commission_rate=None, slippage_rate=None, stamp_duty=None,
            stop_loss_pct=None, stop_loss_warn=None, stop_loss_full=None,
            take_profit_half=None, take_profit_full=None,
            max_position_pct=None, max_sector_pct=None,
            cooldown_stop_weeks=None, cooldown_profit_weeks=None):
        self._code_prefixes = code_prefixes
        self._buy_cost_rate = (commission_rate if commission_rate is not None else 0.0003) + (slippage_rate if slippage_rate is not None else 0.0001)
        self._sell_cost_rate = (commission_rate if commission_rate is not None else 0.0003) + (stamp_duty if stamp_duty is not None else 0.0005) + (slippage_rate if slippage_rate is not None else 0.0001)
        self._stop_loss = stop_loss_pct if stop_loss_pct is not None else -0.08
        self._max_position_pct = max_position_pct if max_position_pct is not None else 0.20
        self._max_sector_pct = max_sector_pct if max_sector_pct is not None else 0.30

        freq_days = get_freq_meta(freq).days
        alloc_cfg = build_allocation_config(
            freq,
            score_key="l0_score",
            max_single_pct=self._max_position_pct,
            max_sector_pct=self._max_sector_pct,
            single_floor_pct=0.10,
            target_sum=0.95,
            stop_loss_warn=stop_loss_warn if stop_loss_warn is not None else (stop_loss_pct if stop_loss_pct is not None else -0.05),
            stop_loss_full=stop_loss_full if stop_loss_full is not None else
                (stop_loss_pct if stop_loss_pct is not None else -0.10),
            take_profit_half=take_profit_half if take_profit_half is not None else 0.15,
            take_profit_full=take_profit_full if take_profit_full is not None else 0.30,
            max_turnover=0.30,
            commission_rate=self._buy_cost_rate - (slippage_rate if slippage_rate is not None else 0.0001),
            slippage_rate=slippage_rate if slippage_rate is not None else 0.0001,
            stamp_duty=stamp_duty if stamp_duty is not None else 0.0005,
            cash_yield_annual=0.03,
            frequency_days=freq_days,
            cooldown_stop_weeks=cooldown_stop_weeks,
            cooldown_profit_weeks=cooldown_profit_weeks,
        )
        self._allocator = PositionAllocator(alloc_cfg)
        t0 = time.time()
        start_d = dt.date.fromisoformat(start)
        end_d = dt.date.fromisoformat(end)
        screener = self._get_screener()

        # 加载每状态的止盈止损配置 (对齐 DirectionBacktest)
        STATE_NAME_TO_IDX = {n: i for i, n in enumerate(
            ["volatile_up","volatile_down","volatile_flat","steady_up","steady_down","steady_flat"])}
        self._regime_stops = {}
        try:
            from core.shared.persistence import load_json
            cfg_path = __import__('pathlib').Path('config/regime_ratio.json')
            cfg = load_json(str(cfg_path))
            if cfg:
                stops = cfg.get('stops', {})
                for k, v in stops.items():
                    idx = int(k)
                    self._regime_stops[idx] = {
                        'warn': -abs(float(v.get('warn', 5))) / 100,
                        'full': -abs(float(v.get('full', 10))) / 100,
                        'profitHalf': float(v.get('profitHalf', 15)) / 100,
                        'profitFull': float(v.get('profitFull', 30)) / 100,
                    }
        except Exception:
            pass

        from core.shared.infrastructure.executors import run_batch
        _saved_limit = getattr(run_batch, "_force_limit", None)
        try:
            run_batch._force_limit = 8  # 回测降并发，减少 GIL 争抢

            # ── 1. 生成回测日期序列 ──
            pd_freq = get_freq_meta(freq).pd_freq
            dates = pd.date_range(start, end, freq=pd_freq)
            dates = [d.date() for d in dates if d.date() >= start_d and d.date() <= end_d]
            if len(dates) < 2:
                raise RuntimeError("回测周期太短")
            n_periods = len(dates)
            logger.info(f"[ScrBacktest] {n_periods} 个调仓日, {start} → {end}")
            self._overlay_coverages = []

            # ── 2. 预加载指数日线（基准+择时共享）──
            index_df_full = self._prefetch_index_data(start_d, end_d)

            # ── 3. 基准价格映射 ──
            bench_price_map = {}
            if index_df_full is not None and len(index_df_full) >= 2:
                bench_dates = index_df_full['trade_date'].astype(str).str[:10]
                mask = (bench_dates >= start) & (bench_dates <= end)
                bench_price_map = dict(zip(bench_dates[mask], index_df_full['close'][mask].astype(float)))

            # ── 4. 批量预计算择时比例 ──
            timing_ratios = {}
            if use_timing:
                timing_ratios = self._precompute_timing_ratios(
                    dates, index_df_full, start_d)

            # ── 5. 预加载全池价格缓存（一次 MySQL 批量拉取，避免逐期重查）──
            price_cache = {}
            self._limit_up_cache = {}
            self._limit_down_cache = {}
            self._st_codes = self._load_st_codes()
            pool_codes = self._get_pool_codes(pool, code_prefixes, before_date=start_d)
            if pool_codes:
                self._load_price_batch(start, end, pool_codes, price_cache)
                logger.info(f"[ScrBacktest] 价格缓存预加载: {len(price_cache)} 只")

            # ── 6. 加载行业映射 ──
            sector_map = self._load_sector_map()

            # ── 7. 主循环 ──
            periods = []
            equity = [1.0]
            benchmark_equity = [1.0]
            equity_dates = [dates[0].isoformat()]
            holdings = {}
            entry_prices = {}
            cash = 0.0
            cooldown = {}
            half_sold = set()
            trades_log = []
            n_top_hold = max(top_n * 2, top_n + 5)
            freq_days = get_freq_meta(freq).days

            first_valid_idx = None  # 第一个有候选股的日期下标

            for i, d in enumerate(dates):
                if cancel_cb:
                    cancel_cb()
                d_str = d.isoformat()

                screen_result = self._screen_at_date(
                    screener, d, pool, top_n, macro_bias,
                    sector_fusion, source_weights, qlib_model, factor_overlay)
                if progress_cb:
                    progress_cb(i + 1, n_periods, d_str)

                candidates = screen_result.get("candidates", [])

                # 冷冻期递减
                expired = [c for c, v in cooldown.items() if v <= 0]
                for c in expired:
                    del cooldown[c]
                for c in list(cooldown.keys()):
                    cooldown[c] -= 1

                # 无候选且尚未找到首个有效日期 → 跳过
                if not candidates and first_valid_idx is None:
                    logger.warning(f"[ScrBacktest] {d_str} 无候选股，跳过（可能非交易日）")
                    continue

                # 无候选但已有持仓 → 保持不动，仅算持有收益
                if not candidates:
                    logger.warning(f"[ScrBacktest] {d_str} 选股返回空，保持现有持仓不变")
                    held_ret = 0.0
                    prev_d = dates[i - 1]
                    prev_d_str = prev_d.isoformat()[:10]
                    total_start = cash + sum(
                        lookup_price(price_cache.get(code), prev_d_str, 'close') or 0
                        for code in list(holdings.keys()))
                    for code, init_w in list(holdings.items()):
                        df = price_cache.get(code)
                        if df is not None:
                            p_prev = lookup_price(df, prev_d_str, 'close')
                            p_curr = lookup_price(df, d_str, 'close')
                            ep = entry_prices.get(code)
                            if p_prev and p_curr and ep and ep > 0 and total_start > 0:
                                cur_w = init_w * (p_prev / ep) / total_start
                                held_ret += cur_w * (p_curr / p_prev - 1)
                    bench_ret = get_benchmark_return_fast(bench_price_map, d, dates, i)
                    equity.append(equity[-1] * (1 + held_ret))
                    equity_dates.append(d_str)
                    benchmark_equity.append(benchmark_equity[-1] * (1 + bench_ret))
                    periods.append({
                        "date": d_str, "regime": "unknown", "direction": "",
                        "confidence": 0, "timing_ratio": round(timing_ratios.get(d, 1.0), 3),
                        "top_stocks": [], "period_return": round(held_ret, 6),
                        "benchmark_return": round(bench_ret, 6), "n_holdings": len(holdings),
                    })
                    if (i + 1) % 10 == 0:
                        logger.info(f"[ScrBacktest] {i+1}/{n_periods} (跳过空候选)")
                    continue

                # 记录第一个有效日期
                if first_valid_idx is None:
                    first_valid_idx = i
                    equity_dates[0] = d_str
                    logger.info(f"[ScrBacktest] 首个有效日期: {d_str} (跳过前 {i} 个)")
                    is_first_valid = True
                else:
                    is_first_valid = False

                # 更新价格缓存（新出现的股票）
                all_codes = set()
                for c in candidates:
                    code = str(c.get("code", "")).strip()
                    if code:
                        all_codes.add(code)
                for code in list(holdings.keys()):
                    all_codes.add(code)
                missing = [c for c in all_codes if c not in price_cache]
                if missing:
                    self._load_price_batch(start, end, missing, price_cache)

                # 按 l0_score 排序取 top
                sorted_cands = sorted(candidates, key=lambda x: x.get("l0_score", 0) or 0, reverse=True)
                buy_codes = [str(c["code"]).strip() for c in sorted_cands[:top_n]]
                buy_scores = {str(c["code"]).strip(): (c.get("l0_score", 0) or 0) for c in sorted_cands[:top_n]}
                sell_threshold = set(str(c["code"]).strip() for c in sorted_cands[:n_top_hold])

                # 择时仓位
                timing_ratio = timing_ratios.get(d, 1.0)

                # 按市场状态覆盖止盈止损 (对齐 DirectionBacktest)
                regime_info = screen_result.get("regime", {})
                regime_name = regime_info.get("regime", "")
                ri = STATE_NAME_TO_IDX.get(regime_name)
                if ri is not None:
                    s = self._regime_stops.get(ri)
                    if s is not None:
                        self._allocator.config.stop_loss_warn = s['warn']
                        self._allocator.config.stop_loss_full = s['full']
                        self._allocator.config.take_profit_half = s['profitHalf']
                        self._allocator.config.take_profit_full = s['profitFull']

                # 基准收益
                bench_ret = get_benchmark_return_fast(bench_price_map, d, dates, i)

                # 模拟当期
                if is_first_valid:
                    holdings, entry_prices, cash, half_sold = self._initial_build(
                        buy_codes, buy_scores, top_n, price_cache, d_str, sector_map)
                else:
                    if i > 0:
                        prev_d = dates[i - 1]
                        holdings, entry_prices, cash, cooldown, half_sold, daily_trades = \
                            self._daily_risk_check(
                                holdings, entry_prices, cash, cooldown, half_sold,
                                prev_d, d, price_cache)
                        if daily_trades:
                            trades_log.extend(daily_trades)

                    total_end, holdings, entry_prices, cash, cooldown, half_sold, trades = self._simulate_period_fast(
                        holdings, entry_prices, cash, cooldown, half_sold,
                        buy_codes, buy_scores, sell_threshold,
                        d, top_n, price_cache, timing_ratio, sector_map)

                    cash *= (1.0 + self._allocator.config.cash_yield_annual) ** (freq_days / 365)

                    equity.append(total_end)
                    equity_dates.append(d_str)
                    benchmark_equity.append(benchmark_equity[-1] * (1 + bench_ret))

                    period_return = total_end / equity[-2] - 1 if equity[-2] > 0 else 0
                    periods.append({
                        "date": d_str,
                        "regime": regime_info.get("regime", "unknown"),
                        "direction": regime_info.get("direction", ""),
                        "confidence": round(regime_info.get("confidence", 0), 3),
                        "timing_ratio": round(timing_ratio, 3),
                        "top_stocks": buy_codes[:5],
                        "period_return": round(period_return, 6),
                        "benchmark_return": round(bench_ret, 6),
                        "n_holdings": len(holdings),
                    })
                    if trades:
                        trades_log.extend(trades)

                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - t0
                        logger.info(f"[ScrBacktest] {i+1}/{n_periods} 期完成, 耗时 {elapsed:.0f}s")

            elapsed = time.time() - t0

            # 清理：如果首期被跳过，equity_dates[0] 可能为空
            if equity_dates and equity_dates[0] == "":
                equity_dates = equity_dates[1:] if len(equity_dates) > 1 else []
            if not periods:
                raise RuntimeError(
                    f"回测区间 {start}~{end} 内无有效交易日，"
                    f"所有调仓日均无候选股。请检查日期范围是否为节假日或选股管线数据是否就绪。")

            logger.info(f"[ScrBacktest] 完成, {n_periods} 个日期, {len(periods)} 期有效, 耗时 {elapsed:.1f}s")


            result = self._compute_metrics(
                equity, benchmark_equity, equity_dates, periods, trades_log, freq)
            if factor_overlay:
                coverages = [v for v in self._overlay_coverages if v is not None]
                result["overlay_coverage"] = sum(coverages) / len(coverages) if coverages else 0.0
            return result
        finally:
            run_batch._force_limit = _saved_limit

    # ══════════════════════════════════════════════
    # 选股管线调用
    # ══════════════════════════════════════════════

    def _screen_at_date(self, screener, eval_date, pool, top_n,
                        macro_bias, sector_fusion, source_weights, qlib_model,
                        factor_overlay=None):
        d_str = eval_date.isoformat()
        try:
            snapshot = None
            if factor_overlay:
                snapshot = screener.precompute_snapshot(
                    pool=pool, eval_date=eval_date, skip_limit_up=True, fast=True)
            overlay = self._prepare_factor_overlay(
                screener, eval_date, pool, factor_overlay, snapshot=snapshot)
            if factor_overlay:
                self._overlay_coverages.append(overlay.get("coverage", 0.0) if overlay else 0.0)
            result = screener.screen(
                pool=pool,
                top_n=top_n * 2,
                eval_date=eval_date,
                regime="auto",
                skip_debate=True,
                macro_bias=macro_bias,
                sector_fusion=sector_fusion,
                source_weights=source_weights,
                qlib_model=qlib_model,
                code_prefixes=self._code_prefixes,
                snapshot=snapshot,
                factor_overlay=overlay,
            )
            return result
        except Exception as e:
            logger.warning(f"[ScrBacktest] {d_str} 选股失败: {e}")
            return {"candidates": [], "regime": {}}

    def _prepare_factor_overlay(self, screener, eval_date, pool, factor_overlay, snapshot=None):
        if not factor_overlay:
            return None
        expr = factor_overlay.get("expr")
        if not expr:
            return None
        try:
            from core.domain.factors.expression import batch_eval_all, parse_factor
            from core.domain.factors.pool_overlay import build_factor_overlay
            if snapshot is None:
                snapshot = screener.precompute_snapshot(
                    pool=pool, eval_date=eval_date, skip_limit_up=True, fast=True)
            if not snapshot:
                return {"expr": expr, "weight": factor_overlay.get("weight", 0.15), "scores": {}, "coverage": 0.0}
            raw = batch_eval_all({0: parse_factor(expr)}, snapshot).get(0, {})
            if factor_overlay.get("risk_neutralize", True) and len(raw) >= 10:
                try:
                    from core.domain.factors.market_cap import get_market_cap_map
                    from core.domain.factors.neutralize import cross_sectional_neutralize
                    codes = list(raw)
                    mktcap_map = get_market_cap_map(codes, eval_date)
                    industry_map = screener.fetcher.get_industry_map() or {}
                    raw = cross_sectional_neutralize(
                        codes, [raw[code] for code in codes], mktcap_map, industry_map)
                except Exception as exc:
                    logger.warning(f"[ScrBacktest] 候选因子中性化失败，使用原始分: {exc}")
            overlay = build_factor_overlay(
                expr, raw, weight=factor_overlay.get("weight", 0.15), source="candidate")
            overlay["coverage"] = len(overlay["scores"]) / max(len(snapshot), 1)
            return overlay
        except Exception as exc:
            logger.warning(f"[ScrBacktest] 候选因子评分失败: {exc}")
            return {"expr": expr, "weight": factor_overlay.get("weight", 0.15), "scores": {}, "coverage": 0.0, "error": str(exc)}

    # ══════════════════════════════════════════════
    # 数据预加载
    # ══════════════════════════════════════════════

    def _prefetch_index_data(self, start_d, end_d):
        try:
            from core.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.get_index_daily("沪深300", days=2000)
            if df is not None and len(df) >= 60:
                df = df.sort_values('date').reset_index(drop=True)
                if 'date' in df.columns and 'trade_date' not in df.columns:
                    df = df.rename(columns={'date': 'trade_date'})
                logger.info(f"[ScrBacktest] 指数预加载: {len(df)} 条")
                return df
        except Exception as e:
            logger.warning(f"[ScrBacktest] 指数预加载失败: {e}")
        return None

    def _precompute_timing_ratios(self, dates, index_df, start_d):
        ratios = {}
        tm = self._get_timing_model()
        if tm.lgb is None or index_df is None or len(index_df) < 100:
            return ratios

        try:
            from core.domain.timing.breadth import get_breadth_history
            from core.domain.timing.model import _make_features, _get_regime_features

            min_d = (dates[0] - dt.timedelta(days=365 * 2)).isoformat()
            max_d = dates[-1].isoformat()
            breadth_df = get_breadth_history(min_d, max_d)
            if breadth_df is not None and not breadth_df.empty:
                breadth_df = breadth_df.reset_index()
            else:
                logger.warning("[ScrBacktest] 市场宽度数据为空，择时特征将跳过")
                return ratios

            hmm_features = None
            use_hmm = any(c.startswith("hmm_") for c in tm.feature_names)

            feature_cache = {}
            n = len(dates)
            for i, d in enumerate(dates):
                cache_key = d.isoformat()[:7]
                if cache_key in feature_cache:
                    ratios[d] = feature_cache[cache_key]
                    continue

                ratio = 1.0
                try:
                    idx = index_df[index_df['trade_date'] <= str(d)].rename(columns={'trade_date': 'date'})
                    if len(idx) < 60:
                        ratios[d] = 1.0
                        continue

                    df = _make_features(idx, breadth_df)
                    df["date"] = pd.to_datetime(df["date"])

                    if use_hmm:
                        hmm_feat = _get_regime_features(idx)
                        if hmm_feat is not None and "date" in hmm_feat.columns:
                            hmm_feat["date"] = pd.to_datetime(hmm_feat["date"])
                            df = df.merge(hmm_feat, on="date", how="left")
                            for c in hmm_feat.columns:
                                if c != "date" and c in df.columns:
                                    df[c] = df[c].fillna(0)

                    for c in tm.feature_names:
                        if c not in df.columns:
                            df[c] = 0

                    latest = df.dropna(subset=tm.feature_names)
                    if latest.empty:
                        ratios[d] = 1.0
                        continue

                    X = latest[tm.feature_names].iloc[-1:]
                    from core.domain.timing.model import prob_to_ratio
                    lgb_prob = float(tm.lgb.predict_proba(X)[:, 1][0])
                    xgb_prob = float(tm.xgb.predict_proba(X)[:, 1][0])
                    prob = (lgb_prob + xgb_prob) / 2
                    ratio = prob_to_ratio(prob)
                except Exception:
                    ratio = 1.0

                ratios[d] = ratio
                feature_cache[cache_key] = ratio

                if (i + 1) % 20 == 0:
                    logger.info(f"[ScrBacktest] 择时预计算 {i+1}/{n}")

            avg_ratio = sum(ratios.values()) / max(len(ratios), 1)
            logger.info(f"[ScrBacktest] 择时预计算完成, {len(ratios)} 期, 均值 {avg_ratio:.2%}")
        except Exception as e:
            logger.warning(f"[ScrBacktest] 择时预计算失败: {e}")

        return ratios

    def _load_st_codes(self):
        """加载 ST 股代码集合 (name 含 ST, 涨跌幅 5%)。"""
        try:
            from core.data.fetcher import DataFetcher
            ind_map = DataFetcher().get_industry_map() or {}
            return {c for c, v in ind_map.items()
                    if isinstance(v, dict) and 'ST' in str(v.get('name', '')).upper()}
        except Exception:
            return set()

    def _load_price_batch(self, start, end, codes, price_cache):
        if not codes:
            return

        start_str = start.replace('-', '') if isinstance(start, str) else start.strftime('%Y%m%d')
        end_str = end.replace('-', '') if isinstance(end, str) else end.strftime('%Y%m%d')

        # 先查 Redis
        from core.shared.infrastructure.cache import CacheManager
        cm = CacheManager()
        missing = []
        for code in codes:
            ck = CacheManager.make_key("daily", "qfq_adj", code)
            df = cm.get_redis(ck)
            if df is not None and not df.empty:
                try:
                    dates_in_cache = pd.to_datetime(df['trade_date'])
                    req_start = pd.Timestamp(start_str[:4] + '-' + start_str[4:6] + '-' + start_str[6:8])
                    req_end = pd.Timestamp(end_str[:4] + '-' + end_str[4:6] + '-' + end_str[6:8])
                    if dates_in_cache.min() <= req_start and dates_in_cache.max() >= req_end:
                        price_cache[code] = df
                        continue
                except Exception:
                    pass
            missing.append(code)

        if missing:
            try:
                from core.data import Database
                if Database.is_available():
                    import polars as pl
                    batch_size = 500
                    all_rows = []
                    for batch_start in range(0, len(missing), batch_size):
                        batch = missing[batch_start:batch_start + batch_size]
                        places = ','.join(['%s'] * len(batch))
                        rows = Database.fetchall(
                            "SELECT code, trade_date, open, high, low, close, volume, amount, "
                            "pct_chg, turnover, adj_factor FROM daily_quotes "
                            f"WHERE trade_date >= %s AND trade_date <= %s AND code IN ({places})",
                            [start_str, end_str] + batch)
                        if rows:
                            all_rows.extend(rows)

                    if all_rows:
                        df_all = pl.from_dicts(all_rows, infer_schema_length=len(all_rows) + 1)
                        grouped = df_all.group_by("code", maintain_order=True)
                        for name, group in grouped:
                            code = name[0] if isinstance(name, tuple) else name
                            pdf = group.to_pandas()
                            ck = CacheManager.make_key("daily", "qfq_adj", code)
                            cm.set_redis(ck, pdf, expire=86400 * 7)
                            price_cache[code] = pdf
            except Exception as e:
                logger.warning(f"[ScrBacktest] 批量加载失败: {e}")

        new_in_cache = {c: price_cache[c] for c in codes if c in price_cache}
        if new_in_cache:
            precompute_limit_up(new_in_cache, self._limit_up_cache, self._st_codes)
            precompute_limit_down(new_in_cache, self._limit_down_cache, self._st_codes)
            self._check_price_consistency(new_in_cache)

    @staticmethod
    def _check_price_consistency(price_cache_sample):
        """检测价格是否存在复权不一致 (除权除息日假跌幅 >15%)"""
        import numpy as np
        warnings_list = []
        for code, df in list(price_cache_sample.items())[:20]:
            try:
                df_sorted = df.sort_values('trade_date')
                closes = df_sorted['close'].astype(float).values
                for i in range(1, len(closes)):
                    if closes[i-1] > 0:
                        ret = closes[i] / closes[i-1] - 1
                        if ret < -0.15:
                            date_str = str(df_sorted.iloc[i]['trade_date'])[:10]
                            warnings_list.append(f"{code} @ {date_str}: {ret:.1%}")
                            break
            except Exception:
                pass
        if warnings_list:
            logger.warning(
                f"[ScrBacktest] 检测到 {len(warnings_list)} 只股票存在 >15% 单日跌幅, "
                f"可能是未经复权处理的原始价格 (除权除息), 回测结果可能失真. "
                f"示例: {warnings_list[:3]}"
            )

    # ══════════════════════════════════════════════
    # 池代码获取
    # ══════════════════════════════════════════════

    @staticmethod
    def _filter_by_list_date(codes, before_date):
        """排除在 before_date 之后才上市的股票，消除幸存者偏差"""
        if not codes or before_date is None:
            return codes
        try:
            from core.data import Database
            if not Database.is_available():
                return codes
            before_str = before_date.isoformat() if hasattr(before_date, 'isoformat') else str(before_date)
            ph = ",".join(["%s"] * len(codes))
            rows = Database.fetchall(
                f"SELECT code, MIN(trade_date) AS first_date FROM daily_quotes "
                f"WHERE code IN ({ph}) GROUP BY code",
                list(codes))
            first_dates = {}
            for r in rows:
                fd = str(r.get("first_date", ""))[:10]
                if fd:
                    first_dates[r["code"]] = fd
            filtered = [c for c in codes
                       if c not in first_dates or first_dates[c] <= before_str]
            excluded = len(codes) - len(filtered)
            if excluded > 0:
                logger.info(f"[ScrBacktest] 排除 {excluded} 只 {before_str} 之后上市的股票")
            return filtered
        except Exception as e:
            logger.warning(f"[ScrBacktest] 上市日期过滤失败: {e}")
            return codes

    @staticmethod
    def _get_pool_codes(pool, code_prefixes=None, before_date=None):
        try:
            from core.domain.screening._screener import _get_all_stock_codes, filter_pool_codes
            codes = _get_all_stock_codes()
            if before_date is not None:
                codes = ScreeningBacktestRunner._filter_by_list_date(codes, before_date)
            if code_prefixes:
                prefixes = tuple(str(p) for p in code_prefixes)
                codes = [c for c in codes if c.startswith(prefixes)]
            codes = filter_pool_codes(codes, pool, as_of=before_date)
            return codes
        except Exception as e:
            logger.warning(f"[ScrBacktest] 获取池代码失败: {e}")
            return []

    # ══════════════════════════════════════════════
    # 交易模拟（复用 direction_backtest 逻辑）
    # ══════════════════════════════════════════════

    def _simulate_period_fast(self, holdings, entry_prices, cash, cooldown, half_sold,
                              buy_codes, buy_scores, sell_threshold,
                              curr_date, top_n,
                              price_cache, timing_ratio, sector_map):
        curr_str = curr_date.isoformat()[:10]
        trades = []

        current_mv = {}
        total_mv = 0.0
        for code, init_w in list(holdings.items()):
            df = price_cache.get(code)
            if df is None:
                current_mv[code] = init_w
                total_mv += init_w
                continue
            p_curr = lookup_price(df, curr_str, 'close')
            ep = entry_prices.get(code)
            if p_curr and p_curr > 0 and ep and ep > 0:
                mv = init_w * (p_curr / ep)
            else:
                mv = init_w
            current_mv[code] = mv
            total_mv += mv

        total_start = cash + total_mv
        if total_start <= 0:
            return total_start, holdings, entry_prices, cash, cooldown, half_sold, trades

        current_weights = {c: mv / total_start for c, mv in current_mv.items()}

        risk_info = []
        for code, ep in entry_prices.items():
            if ep and ep > 0:
                df = price_cache.get(code)
                p_low = lookup_price(df, curr_str, 'low') if df is not None else None
                p_high = lookup_price(df, curr_str, 'high') if df is not None else None
                risk_info.append({
                    "code": code, "entry_price": ep,
                    "p_low": p_low, "p_high": p_high,
                })
        risk_actions = self._allocator.classify_risk(
            risk_info, half_sold=half_sold, use_intraday_low_high=True)

        target_weights = self._allocator.allocate_weights_only(
            [{"code": c, "l0_score": buy_scores.get(c, 0)} for c in buy_codes],
            current_weights, sector_map, cooldown)
        target_weights = self._allocator._normalize_with_timing(
            target_weights, timing_ratio)

        sell_proceeds = 0.0
        limit_down_skipped = 0
        for code, cur_w in list(current_weights.items()):
            if self._is_limit_down(code, curr_str):
                limit_down_skipped += 1
                continue
            action = risk_actions.get(code)
            should_sell = False
            reason = ""
            if action and action.action == "full":
                should_sell = True
                reason = action.reason
            elif action and action.action == "half":
                if "止损" in action.reason:
                    half_value = holdings[code] * (1 + self._allocator.config.stop_loss_warn) / 2
                elif "止盈" in action.reason:
                    half_value = holdings[code] * (1 + self._allocator.config.take_profit_half) / 2
                else:
                    half_value = cur_w * total_start / 2
                proceeds = half_value * (1 - self._sell_cost_rate)
                sell_proceeds += proceeds
                holdings[code] = holdings[code] / 2
                current_mv[code] = current_mv[code] / 2
                half_sold.add(code)
                trades.append({"date": curr_str, "code": code, "action": "sell",
                               "reason": action.reason, "value": half_value,
                               "cost": half_value * self._sell_cost_rate})
            elif code not in sell_threshold:
                should_sell = True
                reason = "不在持有区"
            else:
                tgt = target_weights.get(code, 0)
                if cur_w > tgt * 1.20 and cur_w > 0.02:
                    should_sell = True
                    reason = f"超配 (当前{cur_w:.1%} > 目标{tgt:.1%}×1.2)"
            if should_sell:
                if "止损" in reason:
                    sell_value = holdings[code] * (1 + self._allocator.config.stop_loss_full)
                elif "止盈" in reason:
                    sell_value = holdings[code] * (1 + self._allocator.config.take_profit_full)
                else:
                    sell_value = cur_w * total_start
                proceeds = sell_value * (1 - self._sell_cost_rate)
                sell_proceeds += proceeds
                trades.append({"date": curr_str, "code": code, "action": "sell",
                               "reason": reason, "value": sell_value,
                               "cost": sell_value * self._sell_cost_rate})
                del holdings[code]
                entry_prices.pop(code, None)
                is_stop = "止损" in reason
                cooldown[code] = (self._allocator.config.cooldown_stop_periods
                                  if is_stop else self._allocator.config.cooldown_profit_periods)
                half_sold.discard(code)
                current_mv.pop(code, None)

        cash += sell_proceeds
        total_post_sell = cash + sum(current_mv.values())
        if total_post_sell <= 0:
            logger.warning(f"[ScrBacktest] {curr_str} 组合净值归零, 强制清仓")
            return 0.0, {}, {}, 0.0, cooldown, half_sold, trades

        current_weights = {c: mv / total_post_sell for c, mv in current_mv.items()}

        target_weights = self._allocator._apply_turnover_limit(
            target_weights, current_weights)

        remaining = cash
        limit_up_skipped = 0
        for code, tgt_w in target_weights.items():
            if code in cooldown and cooldown[code] > 0:
                continue
            cur_w = current_weights.get(code, 0)
            if tgt_w > cur_w * 1.12:
                df = price_cache.get(code)
                if self._is_limit_up(code, curr_str):
                    limit_up_skipped += 1
                    continue
                buy_value = (tgt_w - cur_w) * total_post_sell
                cost = buy_value * (1 + self._buy_cost_rate)
                if cost > remaining * 0.8:
                    cost = remaining * 0.8
                    buy_value = cost / (1 + self._buy_cost_rate)
                remaining -= cost
                if remaining < 0:
                    break
                buy_price = lookup_price(df, curr_str, 'close') if df is not None else None
                if code in holdings:
                    old_cost = holdings[code]
                    old_ep = entry_prices.get(code)
                    holdings[code] = old_cost + buy_value
                    if old_ep and old_ep > 0 and buy_price and buy_price > 0:
                        entry_prices[code] = (old_cost + buy_value) / (old_cost / old_ep + buy_value / buy_price)
                else:
                    holdings[code] = buy_value
                    if buy_price is not None:
                        entry_prices[code] = buy_price
                trades.append({"date": curr_str, "code": code, "action": "buy",
                               "reason": f"新进/加仓 (目标{tgt_w:.1%})",
                               "value": buy_value, "cost": buy_value * self._buy_cost_rate})

        if limit_up_skipped > 0:
            logger.info(f"[ScrBacktest] {curr_str} 涨停跳过: {limit_up_skipped} 只")
        if limit_down_skipped > 0:
            logger.info(f"[ScrBacktest] {curr_str} 跌停跳过(卖): {limit_down_skipped} 只")

        cash = remaining

        total_end = cash
        for code, init_w in holdings.items():
            df = price_cache.get(code)
            if df is not None:
                p_curr = lookup_price(df, curr_str, 'close')
                ep = entry_prices.get(code)
                if p_curr and p_curr > 0 and ep and ep > 0:
                    total_end += init_w * (p_curr / ep)
                else:
                    total_end += init_w
            else:
                total_end += init_w

        return total_end, holdings, entry_prices, cash, cooldown, half_sold, trades

    def _is_limit_up(self, code, date_str):
        date_key = date_str.replace('-', '')[:8]
        return self._limit_up_cache.get(code, {}).get(date_key, False)

    def _is_limit_down(self, code, date_str):
        date_key = date_str.replace('-', '')[:8]
        return self._limit_down_cache.get(code, {}).get(date_key, False)

    @staticmethod
    def _lookup_price(df, date_str, col):
        return lookup_price(df, date_str, col)

    @staticmethod
    def _get_benchmark_return_fast(price_map, date, all_dates, idx):
        return get_benchmark_return_fast(price_map, date, all_dates, idx)

    def _load_sector_map(self):
        try:
            from core.data.fetcher import DataFetcher
            return DataFetcher().get_industry_map() or {}
        except Exception:
            return {}

    def _initial_build(self, codes, scores, top_n, price_cache, date_str, sector_map):
        return initial_build(
            codes, scores, top_n, price_cache, date_str, sector_map,
            self._allocator, self._limit_up_cache, self._buy_cost_rate,
            score_key="l0_score", log_label="[ScrBacktest]",
        )

    # ══════════════════════════════════════════════
    # 日频风险检查（双档阶梯止损止盈）
    # ══════════════════════════════════════════════

    def _daily_risk_check(self, holdings, entry_prices, cash, cooldown, half_sold,
                          prev_date, curr_date, price_cache):
        trading_days = self._get_trading_days_between(prev_date, curr_date)
        if not trading_days:
            return holdings, entry_prices, cash, cooldown, half_sold, []

        trades = []
        for day in trading_days:
            day_str = day.isoformat()[:10]
            risk_info = []
            for code, init_w in list(holdings.items()):
                ep = entry_prices.get(code)
                if not ep or ep <= 0:
                    continue
                df = price_cache.get(code)
                if df is None:
                    continue
                if self._is_limit_down(code, day_str):
                    continue
                risk_info.append({
                    "code": code, "entry_price": ep,
                    "p_low": lookup_price(df, day_str, 'low'),
                    "p_high": lookup_price(df, day_str, 'high'),
                })
            if not risk_info:
                continue

            actions = self._allocator.classify_risk(
                risk_info, half_sold=half_sold, use_intraday_low_high=True)

            for code, action in actions.items():
                init_w = holdings.get(code, 0)
                ep = entry_prices.get(code)
                df = price_cache.get(code)
                p_close = lookup_price(df, day_str, 'close') if df is not None else None
                mv = init_w * (p_close / ep) if p_close and ep and ep > 0 else init_w

                if action.action == "full":
                    if "止损" in action.reason:
                        sell_value = init_w * (1 + self._allocator.config.stop_loss_full)
                    elif "止盈" in action.reason:
                        sell_value = init_w * (1 + self._allocator.config.take_profit_full)
                    else:
                        sell_value = mv
                    proceeds = sell_value * (1 - self._sell_cost_rate)
                    cash += proceeds
                    trades.append({"date": day_str, "code": code, "action": "sell",
                                   "reason": action.reason, "value": sell_value,
                                   "cost": sell_value * self._sell_cost_rate})
                    del holdings[code]
                    entry_prices.pop(code, None)
                    cooldown[code] = action.new_cooldown
                    half_sold.discard(code)
                elif action.action == "half":
                    if "止损" in action.reason:
                        half_value = init_w * (1 + self._allocator.config.stop_loss_warn) / 2
                    elif "止盈" in action.reason:
                        half_value = init_w * (1 + self._allocator.config.take_profit_half) / 2
                    else:
                        half_value = mv / 2
                    proceeds = half_value * (1 - self._sell_cost_rate)
                    cash += proceeds
                    holdings[code] = init_w / 2
                    half_sold.add(code)
                    trades.append({"date": day_str, "code": code, "action": "sell",
                                   "reason": action.reason, "value": half_value,
                                   "cost": half_value * self._sell_cost_rate})

        return holdings, entry_prices, cash, cooldown, half_sold, trades

    def _get_trading_days_between(self, prev_date, curr_date):
        try:
            from core.shared.trading_time import get_trading_days_between
            return get_trading_days_between(prev_date, curr_date)
        except Exception:
            return []

    # ══════════════════════════════════════════════
    # 指标计算
    # ══════════════════════════════════════════════

    def _compute_metrics(self, equity, benchmark_equity, equity_dates,
                         periods, trades_log, freq):
        """回测绩效计算 — 委托给 compute_backtest_metrics (纯 L3 函数)。"""
        return compute_backtest_metrics(
            equity, benchmark_equity, equity_dates,
            periods, trades_log, freq,
            profit_factor_cap=999.0,
        )
