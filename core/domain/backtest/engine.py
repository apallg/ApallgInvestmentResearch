"""
策略方向回测引擎

交易逻辑:
- 每周检测市场状态(HMM/规则) → 选择方向模型 → 取该模型预测值
- 按预测分加权分配权重 (score² / sum(score²))，经仓位上限+行业分散+换手率约束
- 买入区 Top-N, 持有区 Top-N×2, 之外卖出
- 止损: 日内最低价触及 -8% 即触发 (非收盘价等)
- 择时模型叠加仓位系数，未投入资金计入现金
- 卖出后 4 期冷冻期，避免反复交易
- 成本: 买入 0.03% 佣金 + 卖出 0.08% (含印花税0.05%) + 双边各 0.01% 滑点
- 单只 ≤ 20%, 单行业 ≤ 30%, 单边换手 ≤ 30%

风控约束:
- 权重漂移修正: 卖出回款 = 当前市值 × (1 - 费率)，非初始权重
- 现金账户: 择时降仓剩余资金显式追踪，参与下期分配
- 止损用最低价: low / entry_price - 1 ≤ -8%，日内真实可触发

性能优化:
- 指数日线数据全区间一次性预加载，HMM/体制分类/择时共享，避免逐期 HTTP 回退
- 择时比例全期批量预计算，复用单次 index_df + breadth 拉取
"""
import logging
import datetime as dt
import json
import math
import os
import time
import numpy as np
import pandas as pd

from core.domain.backtest.common import (
    limit_pct_for_code, lookup_price, norm_code, get_benchmark_return_fast, equal_weight_pool_return,
    precompute_limit_up, precompute_limit_down, initial_build,
)
from core.domain.backtest.metrics import compute_backtest_metrics
from core.domain.portfolio.allocator import (
    PositionAllocator,
    RiskAction,
    build_allocation_config,
    classify_dynamic_trailing,
    classify_atr_stop,
    classify_risk_action,
    compute_atr,
)
from core.domain.screening.macro import apply_macro_sector_bias, apply_sector_fusion
from core.domain.screening.exclusion import evaluate_decision_day_filter
from core.shared import get_freq_meta, cooldown_periods

logger = logging.getLogger(__name__)

__all__ = ["DirectionBacktestRunner"]

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "direction_models.json")
MODEL_CACHE_TTL = 3600


def _freq_stride_days(freq_meta):
    """调仓间隔的交易日数 (stride=True 时 days 即交易日, 否则日历天×5/7 近似)"""
    if freq_meta.stride:
        return freq_meta.days
    return max(1, round(freq_meta.days * 5 / 7))


def _backtest_decision_dates(start, end, decision_end, pd_freq, freq_meta, index_df_full):
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    decision_end_d = dt.date.fromisoformat(decision_end or end)
    if not start_d <= decision_end_d <= end_d:
        raise ValueError("decision_end必须位于start和end之间")

    dates = [value.date() for value in pd.date_range(start, end, freq=pd_freq)]
    dates = [value for value in dates if start_d <= value <= end_d]
    if index_df_full is None:
        return [value for value in dates if value <= decision_end_d]

    if freq_meta.stride:
        all_trading = sorted(pd.to_datetime(index_df_full["trade_date"]).dt.date)
        all_trading = [value for value in all_trading if start_d <= value <= decision_end_d]
        return all_trading[::_freq_stride_days(freq_meta)]

    trading_dates = set(
        pd.to_datetime(index_df_full["trade_date"]).dt.strftime("%Y-%m-%d")
    )
    snapped = []
    seen = set()
    for value in dates:
        value_str = value.isoformat()
        if value_str in trading_dates and value_str not in seen:
            snapped.append(value)
            seen.add(value_str)
            continue
        for offset in range(1, 7):
            previous = value - dt.timedelta(days=offset)
            previous_str = previous.isoformat()
            if previous_str in trading_dates and previous_str not in seen:
                snapped.append(previous)
                seen.add(previous_str)
                break
    return [value for value in snapped if start_d <= value <= decision_end_d]


class DirectionBacktestRunner:

    def __init__(self, latest_adj_factor_loader=None):
        from core.adapters.qlib.config import init_qlib
        init_qlib()
        self._timing_model = None
        self._latest_adj_factor_loader = latest_adj_factor_loader
        self._latest_adj_factors = {}

    def _get_timing_model(self):
        if self._timing_model is None:
            from core.domain.timing.model import get_model
            self._timing_model = get_model()
        return self._timing_model

    def run(self, start, end, pool="stable_history_500d", top_n=5, freq="week",
            sizing="equal", target_exposure=0.95, max_one_way_turnover=0.30,
            directions=None, use_hmm=False, use_timing=False, cash_yield=False,
            full_position=True, code_prefixes=None,
            use_rule_based=False, use_crf=False, use_vol_switch=False,
            vol_label_horizon=10,
            use_conditional_switch=True,
            state_models=None,
            fixed_model=None,
            commission_rate=None, slippage_rate=None, stamp_duty=None,
            stop_loss_pct=None, take_profit_pct=None,
            stop_loss_warn=None, stop_loss_full=None,
            take_profit_half=None, take_profit_full=None,
            max_position_pct=None, max_sector_pct=None,
            stop_source="off", dynamic_lookback=20, dynamic_trail=0.05,
            atr_period=14, atr_mult=2.0,
            macro_bias=False, macro_regime=False, sector_fusion=False,
            neutralize_qlib=False,
            decision_day_filter=True,
            cooldown_stop_weeks=None, cooldown_profit_weeks=None,
            decision_policy=None, decision_overlays=None,
            conditional_market_env=None, benchmark_membership_by_date=None,
            security_references_by_date=None, decision_end=None):
        self._buy_cost_rate = (commission_rate if commission_rate is not None else 0.0003) + (slippage_rate if slippage_rate is not None else 0.0001)
        self._sell_cost_rate = (commission_rate if commission_rate is not None else 0.0003) + (stamp_duty if stamp_duty is not None else 0.0005) + (slippage_rate if slippage_rate is not None else 0.0001)
        # 阶梯式止损: warn=半仓(不冷冻) full=清仓(冷冻4周)
        self._stop_loss_warn = stop_loss_warn if stop_loss_warn is not None else (stop_loss_pct if stop_loss_pct is not None else -0.05)
        self._stop_loss_full = stop_loss_full if stop_loss_full is not None else (stop_loss_pct if stop_loss_pct is not None else -0.10)
        # 阶梯式止盈: half=半仓(不冷冻) full=清仓(冷冻4周)
        self._take_profit_half = take_profit_half if take_profit_half is not None else (take_profit_pct if take_profit_pct is not None else 0.15)
        self._take_profit_full = take_profit_full if take_profit_full is not None else (take_profit_pct if take_profit_pct is not None else 0.30)
        self._stop_source = stop_source
        self._dynamic_lookback = int(dynamic_lookback)
        self._dynamic_trail = float(dynamic_trail)
        self._atr_period = int(atr_period)
        self._atr_mult = float(atr_mult)
        if stop_source == "off":
            self._stop_loss_warn = -999.0
            self._stop_loss_full = -999.0
            self._take_profit_half = 999.0
            self._take_profit_full = 999.0
        # 保留旧引用供外部诊断日志使用
        self._stop_loss = self._stop_loss_full
        self._take_profit = self._take_profit_full
        self._max_position_pct = max_position_pct if max_position_pct is not None else 0.20
        self._max_sector_pct = max_sector_pct if max_sector_pct is not None else 0.30
        if not 0 < float(target_exposure) <= 1:
            raise ValueError("target_exposure必须在(0,1]范围")
        if not 0 <= float(max_one_way_turnover) <= 1:
            raise ValueError("max_one_way_turnover必须在[0,1]范围")
        self._cash_yield = cash_yield
        decision_periods = []
        decision_filter_excluded = {
            "st": 0, "suspended": 0, "limit_up": 0, "low_liquidity": 0,
            "low_price": 0, "new_listing": 0, "data_error": 0,
        }
        decision_filter_examined = 0
        decision_filter_shortfalls = 0

        _freq_meta = get_freq_meta(freq)
        freq_days = _freq_meta.days

        # 因果 CRF 平滑 (回测实例级状态, 不污染全局 _prior_probs)
        self._crf_prior = []
        self._crf_trans = None
        self._crf_window = 4  # 窗口调仓期数 (≈20交易日)
        self._use_conditional_switch = bool(use_conditional_switch)
        self._cond_neutral = False
        self._cond_env = conditional_market_env or {}
        if use_crf:
            try:
                import numpy as np
                from core.domain.regime.unified import _get_transition_matrix
                stride = _freq_stride_days(_freq_meta)
                self._crf_trans = np.linalg.matrix_power(_get_transition_matrix(), stride)
                logger.info(f"[DirBacktest] 因果 CRF 平滑启用 (调仓间隔≈{stride}交易日, 窗口{self._crf_window}期)")
            except Exception as e:
                logger.warning(f"[DirBacktest] CRF 转移矩阵初始化失败, 关闭平滑: {e}")
                self._crf_trans = None

        alloc_cfg = build_allocation_config(
            freq,
            sizing=sizing,
            score_key="score",
            max_single_pct=self._max_position_pct,
            max_sector_pct=self._max_sector_pct,
            single_floor_pct=0.10,
            target_sum=float(target_exposure),
            stop_loss_warn=self._stop_loss_warn,
            stop_loss_full=self._stop_loss_full,
            take_profit_half=self._take_profit_half,
            take_profit_full=self._take_profit_full,
            max_turnover=float(max_one_way_turnover),
            commission_rate=commission_rate if commission_rate is not None else 0.0003,
            slippage_rate=slippage_rate if slippage_rate is not None else 0.0001,
            stamp_duty=stamp_duty if stamp_duty is not None else 0.0005,
            cash_yield_annual=0.03 if cash_yield else 0.0,
            frequency_days=freq_days,
            cooldown_stop_weeks=cooldown_stop_weeks,
            cooldown_profit_weeks=cooldown_profit_weeks,
        )
        self._allocator = PositionAllocator(alloc_cfg)
        if neutralize_qlib is None:
            from core.shared import config as app_config
            neutralize_qlib = bool(getattr(app_config, "QLIB_PRED_NEUTRALIZE", False))
        else:
            neutralize_qlib = bool(neutralize_qlib)
        neutralize_applied = bool(neutralize_qlib)
        neutralize_fallback_count = 0
        score_status_counts = {}
        t0 = time.time()
        if directions is None:
            directions = ["trend_momentum"]
        from core.domain.regime.model_routing import normalize_model_ref, validate_state_models
        state_models = validate_state_models(state_models, use_vol_switch)
        fixed_model = normalize_model_ref(fixed_model)
        if state_models and fixed_model:
            raise ValueError("固定单模型与状态模型映射不能同时启用")
        decision_end = decision_end or end
        decision_start = dt.date.fromisoformat(start)
        decision_end_date = dt.date.fromisoformat(decision_end)
        data_end = dt.date.fromisoformat(end)
        if not decision_start <= decision_end_date <= data_end:
            raise ValueError("decision_end必须位于start和数据截止日end之间")
        cache_key = (start, end, decision_end, pool, tuple(directions), tuple(code_prefixes or ()),
                     tuple(sorted(state_models.items())), fixed_model)
        cache_valid = getattr(self, '_cached_run_key', None) == cache_key
        if not cache_valid:
            self._cached_run_key = None
        start_d = dt.date.fromisoformat(start)
        end_d = dt.date.fromisoformat(end)

        # 加载每状态的止盈止损配置 (独立参数, 非倍数)
        self._regime_stops = {}
        STATE_NAME_TO_IDX = {n: i for i, n in enumerate(
            ["volatile_up","volatile_down","volatile_flat","steady_up","steady_down","steady_flat"])}
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

        # ── 1. 加载方向模型 ──
        direction_models = ({"fixed": fixed_model} if fixed_model else
                            state_models or self._load_models_cached(directions))
        if len(direction_models) < 1:
            raise RuntimeError("未找到方向模型，请先在 AI 实验室训练")
        from pathlib import Path
        from core.adapters.qlib.model_contract import (
            TRADABLE_LABEL_MODE,
            get_model_label_mode,
        )
        from core.shared import config as app_config
        mlruns_dir = Path(str(app_config.QLIB_MLFLOW_URI).replace("file:", ""))
        incompatible = []
        model_score_semantics = {}
        for model_state, (exp_id, run_id) in direction_models.items():
            run_path = mlruns_dir / exp_id / run_id
            label_mode = get_model_label_mode(run_path)
            model_score_semantics[model_state] = (
                "daily_cross_sectional_rank_fusion"
                if (run_path / "artifacts" / "params_lstm.pkl").exists()
                else "raw_model_score"
            )
            if label_mode != TRADABLE_LABEL_MODE:
                incompatible.append(f"{model_state}={exp_id}/{run_id[:8]}({label_mode or 'unknown'})")
        if incompatible:
            raise RuntimeError("策略回测仅允许开盘到开盘标签模型，请先重训: " + ", ".join(incompatible))
        if (state_models or fixed_model) and sector_fusion:
            logger.info("[DirBacktest] 显式模型已启用，关闭板块背离二次路由")
            sector_fusion = False

        # ── 2. 预加载 Qlib 预测 ──
        prediction_errors = {}
        if cache_valid and getattr(self, '_cached_pred_cache', None) is not None:
            pred_cache = {k: v for k, v in self._cached_pred_cache.items() if k in direction_models}
            logger.info(f"[DirBacktest] 复用缓存预测: {list(pred_cache.keys())}")
        else:
            from core.adapters.qlib.models import QlibModelManager
            mgr = QlibModelManager()
            pred_cache = {}
            model_predictions = {}
            for dir_name, (exp_id, run_id) in direction_models.items():
                try:
                    from core.shared.disabled_models import is_disabled
                    if is_disabled(exp_id, run_id):
                        logger.warning(f"[DirBacktest] {dir_name}: 模型已停用, 跳过")
                        continue
                    model_id = (exp_id, run_id)
                    if model_id in model_predictions:
                        pred = model_predictions[model_id]
                        logger.info(f"[DirBacktest] {dir_name}: 复用同 run 预测 {exp_id}/{run_id}")
                    else:
                        pred, _ = mgr.predict_dual(exp_id, run_id,
                                                   predict_start=start, predict_end=decision_end,
                                                   instruments=pool, dataset=None)
                        model_predictions[model_id] = pred
                    pred_cache[dir_name] = pred
                    logger.info(f"[DirBacktest] {dir_name}: {len(pred)} 条预测")
                except Exception as e:
                    prediction_errors[dir_name] = f"{type(e).__name__}: {e}"
                    logger.warning(f"[DirBacktest] {dir_name}: 加载失败: {e}")

        if not pred_cache:
            detail = "; ".join(
                f"{name}={reason}" for name, reason in prediction_errors.items())
            raise RuntimeError(
                "无法加载任何方向模型的预测"
                + (f": {detail}" if detail else "")
            )
        self._cached_pred_cache = pred_cache

        # ── 2b. 预加载行业映射 (归一化时复用, 避免逐日查 DB) ──
        _norm_industry_map = None
        try:
            from core.data.fetcher import DataFetcher
            _norm_industry_map = DataFetcher().get_industry_map() or {}
        except Exception:
            pass

        # ── 3. 生成回测日期序列 ──
        pd_freq = _freq_meta.pd_freq

        # 频次相关参数
        self._freq_days_per_period = _freq_meta.days

        # 冷冻期: 显式周数参数 > freq 默认 (止损2周 / 止盈1周 换算为期数)
        self._cooldown_stop, self._cooldown_profit = cooldown_periods(
            freq, cooldown_stop_weeks, cooldown_profit_weeks)

        # ── 4. 预加载价格缓存 ──
        all_codes = set()
        for pred_df in pred_cache.values():
            try:
                codes = pred_df.index.get_level_values('instrument').unique().tolist()
                all_codes.update(norm_code(c) for c in codes)
            except Exception:
                pass
        if benchmark_membership_by_date is not None:
            for member_codes in benchmark_membership_by_date.values():
                all_codes.update(norm_code(code) for code in member_codes)
        if code_prefixes:
            prefixes = tuple(str(p) for p in code_prefixes)
            all_codes = {c for c in all_codes if c.startswith(prefixes)}
            logger.info(f"[DirBacktest] 代码前缀过滤: {prefixes} → {len(all_codes)} 只")
        price_cache = (self._cached_price_cache if cache_valid
                       and getattr(self, '_cached_price_cache', None) is not None
                       else self._preload_price_cache(
                           start, end, list(all_codes),
                           history_observations=500 if pool == "stable_history_500d" else None,
                           security_references_by_date=security_references_by_date,
                       ))
        self._cached_price_cache = price_cache

        # ── 5. ★ 预加载指数日线（全区间一次性拉取，HMM/体制/基准/择时/日频止损共享）──
        index_df_full = (self._cached_index_df if cache_valid
                        and getattr(self, '_cached_index_df', None) is not None
                        else self._prefetch_index_data(start_d, end_d))
        self._cached_index_df = index_df_full
        self._cached_run_key = cache_key
        self._index_df = index_df_full  # 日频止损止盈需要交易日列表

        # ── 5b. 调仓日假期对齐 ──
        dates = _backtest_decision_dates(
            start, end, decision_end, pd_freq, _freq_meta, index_df_full,
        )
        if _freq_meta.stride and index_df_full is not None:
            logger.info(f"[DirBacktest] {freq}频: {len(dates)} 个调仓日 "
                       f"({dates[0].isoformat() if dates else '?'} ~ {dates[-1].isoformat() if dates else '?'})")
        elif index_df_full is not None:
            logger.info(f"[DirBacktest] 假期对齐后 {len(dates)} 个调仓日")
        if len(dates) < 2:
            raise RuntimeError("决策窗口内假期对齐后回测周期太短")
        benchmark_membership_by_date, security_references_by_date = (
            self._materialize_point_in_time_context(
                dates, benchmark_membership_by_date, security_references_by_date,
            )
        )
        self._require_point_in_time_context(
            dates, benchmark_membership_by_date, security_references_by_date,
        )

        # ── 6. 基准数据 ──
        bench_price_map = {}
        bench_open_map = {}
        if index_df_full is not None and len(index_df_full) >= 2:
            bench_dates = index_df_full['trade_date'].astype(str).str[:10]
            mask = (bench_dates >= start) & (bench_dates <= end)
            bench_price_map = dict(zip(bench_dates[mask], index_df_full['close'][mask].astype(float)))
            if 'open' in index_df_full.columns:
                bench_open_map = dict(zip(bench_dates[mask], index_df_full['open'][mask].astype(float)))

        # ── 7. 初始化 Regime 分类器 ──
        # 统一入口 get_unified_regime 内部处理: RegimePredictor(主) → RuleBased(回退) + 宏观修正。
        # use_rule_based=True 时强制走确定性规则 (不加载监督模型, 可复现)。
        rule_classifier = None
        if use_rule_based:
            from core.domain.regime.classifier import RuleBasedRegimeClassifier
            rule_classifier = RuleBasedRegimeClassifier(index_df_full)
            logger.info("[DirBacktest] 使用确定性规则分类器 (RuleBased)")
        else:
            logger.info("[DirBacktest] 使用统一入口 (RegimePredictor + RuleBased 回退 + 宏观修正)")

        # ── 7.5 预加载市场宽度 (regime 特征) ──
        # 实盘 breadth 特征用真实 up_ratio, 回测若传 None 则 predictor 填 0 → train/serve skew
        breadth_df = None
        try:
            from core.domain.timing.breadth import get_breadth_history
            _bd = get_breadth_history((start_d - dt.timedelta(days=365 * 2)).isoformat(), end)
            if _bd is not None and not _bd.empty:
                breadth_df = _bd.reset_index()
        except Exception as e:
            logger.warning(f"[DirBacktest] 市场宽度预加载失败: {e}")

        # ── 7.6 条件切换环境特征 (宽度/超跌, 按调仓日 point-in-time) ──
        if use_vol_switch and self._use_conditional_switch and self._cond_env:
            logger.info(f"[DirBacktest] 条件切换环境特征: {len(self._cond_env)} 个交易日")

        # ── 8. ★ 批量预计算择时比例 ──
        timing_ratios = {}
        if use_timing and not full_position:
            timing_ratios = self._precompute_timing_ratios(
                dates, index_df_full, start_d)
        if full_position:
            logger.info("[DirBacktest] 满仓模式: 忽略择时仓位, timing_ratio=1.0")

        # ── 9. 加载行业映射 ──
        sector_map = _norm_industry_map or self._load_sector_map()

        # ── 9.5 预拉取全市场日线 (板块融合复用, 避免逐调仓日全市场重算) ──
        sector_quotes_df = None
        if sector_fusion and not use_vol_switch:
            try:
                from core.domain.regime.sector import prefetch_sector_quotes
                from core.shared import config as _cfg
                _lb = int(getattr(_cfg, "REGIME_LOOKBACK_DAYS", 252))
                pre_start = dates[0] - dt.timedelta(days=int(_lb * 1.6) + 30)
                sector_quotes_df = prefetch_sector_quotes(pre_start, dates[-1])
                if sector_quotes_df is not None:
                    logger.info(f"[DirBacktest] 预拉取全市场日线 {pre_start}~{dates[-1]} "
                                f"({sector_quotes_df.height} 行)")
            except Exception as e:
                logger.warning(f"[DirBacktest] 预拉取全市场日线失败: {e}")

        # ── 10. 主循环 ──
        periods = []
        equity = [1.0]
        benchmark_equity = [1.0]
        secondary_benchmark_equity = [1.0]
        equity_dates = [dates[0].isoformat()]
        holdings = {}
        entry_prices = {}
        entry_dates = {}  # {code: 买入日期} — 用于 T+1 约束 (当日买入不可当日卖出)
        cash = 0.0
        cooldown = {}
        trades_log = []
        _hold_mult = _freq_meta.hold_mult
        n_top_hold = max(top_n * _hold_mult, top_n + 5)
        n_periods = len(dates)
        half_sold = {}  # {code: True} 已触发过半仓止盈/止损, 下次只检查清仓阈值
        prev_d = None

        for i, d in enumerate(dates):
            d_str = d.isoformat()
            self._cond_neutral = False
            decision_members = (
                set(benchmark_membership_by_date[d_str])
                if benchmark_membership_by_date is not None else None
            )
            if security_references_by_date is not None:
                references = security_references_by_date[d_str]
                decision_sector_map = {
                    code: {
                        "name": row.get("name", ""),
                        "industry": row.get("industry", ""),
                    }
                    for code, row in references.items()
                }
            else:
                decision_sector_map = sector_map

            # 10a. 体制分类 — 统一入口 (预加载 index_df + breadth)
            regime_val, direction, confidence, detected_model_state = self._classify_regime(
                d, rule_classifier, index_df_full, use_rule_based, use_vol_switch, breadth_df,
                vol_label_horizon, macro_regime)
            if state_models and not detected_model_state:
                raise RuntimeError(f"{d} 未获得可用的模型路由状态")
            model_state = "fixed" if fixed_model else (detected_model_state if state_models else direction)

            # 下一交易日 (T+1 开盘成交) — 提前计算, 供基准收益窗口对齐
            next_d = self._next_trading_day(d)
            next_str = next_d.isoformat()[:10] if next_d else d_str

            # 10a.5 主基准使用同池等权，与组合净值估值时点对齐。沪深300只作辅助基准。
            bench_ret = 0.0
            csi300_ret = 0.0
            benchmark_coverage = 0.0
            if i > 0:
                previous_exec = self._next_trading_day(dates[i - 1])
                previous_exec_str = previous_exec.isoformat()[:10] if previous_exec else dates[i - 1].isoformat()
                membership_date = dates[i - 1].isoformat()
                membership_codes = None
                if benchmark_membership_by_date is not None:
                    membership_codes = benchmark_membership_by_date.get(membership_date, [])
                current_mark_str = next_str if next_d else d_str
                start_col = 'open' if i == 1 else 'close'
                pool_benchmark = equal_weight_pool_return(
                    price_cache, previous_exec_str, current_mark_str,
                    start_col=start_col, end_col='close', codes=membership_codes)
                bench_ret = pool_benchmark["return"]
                benchmark_coverage = pool_benchmark["coverage"]
                start_map = bench_open_map if start_col == 'open' else bench_price_map
                _b_prev = start_map.get(previous_exec_str)
                _b_curr = bench_price_map.get(current_mark_str)
                csi300_ret = (_b_curr / _b_prev - 1) if (_b_prev and _b_curr and _b_prev > 0) else 0.0

            # 10b. 取预测 (缺失模型时按优先级兜底)
            pred_df = pred_cache.get(model_state)
            prediction_model_state = model_state
            if pred_df is None and not state_models and not fixed_model:
                for fallback_dir in ["trend_momentum", "volume_breakout", "oversold_rebound"]:
                    if fallback_dir in pred_cache:
                        pred_df = pred_cache[fallback_dir]
                        direction = fallback_dir
                        prediction_model_state = fallback_dir
                        break
            if pred_df is None:
                benchmark_equity.append(benchmark_equity[-1] * (1 + bench_ret))
                secondary_benchmark_equity.append(secondary_benchmark_equity[-1] * (1 + csi300_ret))
                equity.append(equity[-1])
                equity_dates.append(d_str)
                continue
            try:
                day_pred = pred_df.loc[d_str]
            except KeyError:
                day_pred = self._nearest_pred(pred_df, d)
                if day_pred is None:
                    benchmark_equity.append(benchmark_equity[-1] * (1 + bench_ret))
                    secondary_benchmark_equity.append(secondary_benchmark_equity[-1] * (1 + csi300_ret))
                    equity.append(equity[-1])
                    equity_dates.append(d_str)
                    continue

            scores = day_pred.iloc[:, 0] if hasattr(day_pred, 'columns') else day_pred
            scores = scores.dropna()
            raw_dict = {}
            for c, v in scores.items():
                code = norm_code(c)
                if code not in raw_dict and (decision_members is None or code in decision_members):
                    raw_dict[code] = float(v)
            # 中性化 + 分位映射, 对齐 screener 管线和实盘交易路径
            try:
                from core.domain.screening._screener import normalize_qlib_scores
                score_result = normalize_qlib_scores(
                    raw_dict, d, decision_sector_map, neutralize=neutralize_qlib,
                    return_details=True,
                    model_input_semantics=model_score_semantics.get(
                        prediction_model_state, "model_input_score_unknown"),
                )
                norm_dict = score_result["scores"]
                neutralize_status = score_result["audit"]["neutralize_status"]
                score_status_counts[neutralize_status] = score_status_counts.get(neutralize_status, 0) + 1
                if neutralize_status not in ("applied", "partial"):
                    neutralize_applied = False
                if neutralize_status == "fallback":
                    neutralize_fallback_count += 1
            except Exception as e:
                from core.domain.screening.score_pipeline import apply_score_step, build_score_result
                score_result = build_score_result(
                    raw_dict, raw_dict, neutralize_requested=neutralize_qlib,
                    neutralize_status="fallback", neutralize_error=e,
                    context={"ref_date": str(d)[:10], "source": "backtest"},
                    model_input_semantics=model_score_semantics.get(
                        prediction_model_state, "model_input_score_unknown"))
                apply_score_step(score_result, "pipeline_fallback", raw_dict,
                                 status="fallback", detail=e)
                norm_dict = raw_dict
                neutralize_applied = False
                neutralize_fallback_count += 1
                score_status_counts["fallback"] = score_status_counts.get("fallback", 0) + 1
                logger.warning(f"[DirBacktest] 预测分处理中性化失败, date={d}: {e}")

            # 宏观修正 + 板块背离融合 (与选股页面共用代码)
            if macro_bias:
                try:
                    norm_dict, _ = apply_macro_sector_bias(norm_dict, d, decision_sector_map)
                    from core.domain.screening.score_pipeline import apply_score_step
                    apply_score_step(score_result, "macro_sector_bias", norm_dict)
                except Exception as e:
                    from core.domain.screening.score_pipeline import apply_score_step
                    apply_score_step(score_result, "macro_sector_bias", norm_dict,
                                     status="fallback", detail=e)
            if sector_fusion and not use_vol_switch:
                try:
                    dir_scores = {}
                    for dir_name, pred_df in pred_cache.items():
                        try:
                            dp = pred_df.loc[d_str]
                        except KeyError:
                            dp = self._nearest_pred(pred_df, d)
                        if dp is None:
                            continue
                        try:
                            s = dp.iloc[:, 0] if hasattr(dp, 'columns') else dp
                            s = s.dropna()
                            dir_scores[dir_name] = {norm_code(c): float(v) for c, v in s.items()}
                        except Exception:
                            pass
                    norm_dict, _ = apply_sector_fusion(
                        norm_dict, d, pool, quotes_df=sector_quotes_df,
                        dir_scores=dir_scores, neutralize_qlib=neutralize_qlib)
                    from core.domain.screening.score_pipeline import apply_score_step
                    apply_score_step(score_result, "sector_fusion", norm_dict)
                except Exception as e:
                    from core.domain.screening.score_pipeline import apply_score_step
                    apply_score_step(score_result, "sector_fusion", norm_dict,
                                     status="fallback", detail=e)

            norm_scores = pd.Series(norm_dict).sort_values(ascending=False)
            norm_scores = norm_scores[norm_scores.index.isin(all_codes)]

            candidate_limit = n_top_hold
            if decision_policy is not None:
                candidate_limit = max(
                    candidate_limit,
                    int(decision_policy.get("candidate_pool", 50)),
                )
            if decision_day_filter:
                eligible, excluded, examined = self._filter_decision_day_ranking(
                    norm_scores.index.tolist(), d, candidate_limit,
                    price_cache, decision_sector_map,
                    min_history_observations=500 if pool == "stable_history_500d" else None,
                )
                norm_scores = norm_scores.reindex(eligible).dropna()
                decision_filter_examined += examined
                for reason, count in excluded.items():
                    decision_filter_excluded[reason] += count
                if len(eligible) < top_n:
                    decision_filter_shortfalls += 1

            if decision_policy is None:
                buy_codes = norm_scores.head(top_n).index.tolist()
                buy_scores = {c: norm_scores[c] for c in buy_codes}
                sell_threshold = set(norm_scores.head(n_top_hold).index.tolist())
            else:
                from core.domain.portfolio.decision import build_candidate_decision
                overlay = (decision_overlays or {}).get(d_str, {})
                candidate_pool = max(
                    top_n,
                    n_top_hold,
                    int(decision_policy.get("candidate_pool", 50)),
                )
                decision_candidates = [
                    {
                        "code": code,
                        "qlib_score": float(score),
                        "score": float(score),
                        "overlay_delta": overlay.get(code, 0),
                    }
                    for code, score in norm_scores.head(candidate_pool).items()
                ]
                decision_result = build_candidate_decision(
                    decision_candidates,
                    top_n,
                    decision_policy,
                    current_codes=set(holdings),
                )
                eligible = [
                    row for row in decision_result["ranked_candidates"]
                    if not row.get("hard_excluded")
                ]
                buy_codes = [row["code"] for row in decision_result["selected"]]
                buy_scores = {row["code"]: row["utility_score"] for row in decision_result["selected"]}
                sell_threshold = {row["code"] for row in eligible[:n_top_hold]}
                decision_periods.append({
                    "date": d_str,
                    "fingerprint": decision_result["fingerprint"],
                    "summary": decision_result["summary"],
                    "selected": buy_codes,
                })

            # 10c. 冷冻期递减
            expired = [c for c, v in cooldown.items() if v <= 0]
            for c in expired:
                del cooldown[c]
            for c in list(cooldown.keys()):
                cooldown[c] -= 1

            # 10d. 择时仓位系数
            timing_ratio = timing_ratios.get(d, 1.0)
            # 三档切换: 中波死亡区 → 空仓(不使用模型)
            if self._cond_neutral:
                timing_ratio = 0.0
                self._cond_neutral = False

            # 10e.5 止盈止损来源: regime 时按市场状态覆盖 (off/fixed/dynamic 保持初始值)
            if self._stop_source == "regime":
                ri = STATE_NAME_TO_IDX.get(regime_val, 2)
                s = self._regime_stops.get(ri)
                if s is not None:
                    self._stop_loss_warn = s['warn']
                    self._stop_loss_full = s['full']
                    self._take_profit_half = s['profitHalf']
                    self._take_profit_full = s['profitFull']
                    self._allocator.config.stop_loss_warn = s['warn']
                    self._allocator.config.stop_loss_full = s['full']
                    self._allocator.config.take_profit_half = s['profitHalf']
                    self._allocator.config.take_profit_full = s['profitFull']

            # 10f. 日频止损止盈 — 逐日检查上一调仓日至今日之间的交易日
            if i > 0 and prev_d is not None and self._index_df is not None:
                holdings, entry_dates, entry_prices, cash, cooldown, half_sold, daily_trades = \
                    self._daily_risk_check(holdings, entry_dates, entry_prices, cash, cooldown, half_sold,
                                          prev_d, d, price_cache)
                if daily_trades:
                    trades_log.extend(daily_trades)

            # 10g. 模拟当期
            if i > 0:
                if next_d is None:
                    # 无下一交易日 (回测区间止于最新交易日): 无法 T+1 成交, 跳过本期调仓,
                    # 仅按当前持仓在 d_str 收盘估值, 避免末期末 T+0 同日出成交语义漂移
                    total_end = cash
                    for code, init_w in holdings.items():
                        df = price_cache.get(code)
                        if df is not None:
                            p_curr = lookup_price(df, d_str, 'close')
                            ep = entry_prices.get(code)
                            if p_curr and p_curr > 0 and ep and ep > 0:
                                total_end += init_w * (p_curr / ep)
                            else:
                                total_end += init_w
                    trades = []
                else:
                    total_end, holdings, entry_dates, entry_prices, cash, cooldown, half_sold, trades = self._simulate_period_fast(
                        holdings, entry_dates, entry_prices, cash, cooldown, half_sold,
                        buy_codes, buy_scores, sell_threshold,
                        d, top_n, price_cache, timing_ratio, decision_sector_map, next_str)

                equity.append(total_end)
                equity_dates.append(d_str)
                benchmark_equity.append(benchmark_equity[-1] * (1 + bench_ret))
                secondary_benchmark_equity.append(secondary_benchmark_equity[-1] * (1 + csi300_ret))

                period_return = total_end / equity[-2] - 1 if equity[-2] > 0 else 0
                periods.append({
                    "date": d_str,
                    "regime": regime_val,
                    "direction": direction,
                    "model_state": model_state,
                    "confidence": round(confidence, 3),
                    "timing_ratio": round(timing_ratio, 3),
                    "top_stocks": buy_codes[:5],
                    "score_audit": {
                        "source_fingerprint": score_result["audit"].get("source_fingerprint"),
                        "neutralize_status": score_result["audit"].get("neutralize_status"),
                        "selected": {
                            code: score_result["ledger"][code]
                            for code in buy_codes[:5] if code in score_result["ledger"]
                        },
                    },
                    "period_return": round(period_return, 6),
                    "benchmark_return": round(bench_ret, 6),
                    "benchmark_coverage": round(benchmark_coverage, 4),
                    "secondary_benchmark_return": round(csi300_ret, 6),
                    "n_holdings": len(holdings),
                })
                if trades:
                    trades_log.extend(trades)

                if (i + 1) % 20 == 0:
                    logger.info(f"[DirBacktest] {i + 1}/{n_periods} 期完成")
            else:
                holdings, entry_prices, cash, half_sold = self._initial_build(
                    buy_codes, buy_scores, top_n, price_cache, d_str, decision_sector_map, next_str)
                for _c in entry_prices:
                    entry_dates[_c] = next_str

            prev_d = d

        elapsed = time.time() - t0
        logger.info(f"[DirBacktest] 模拟完成, {n_periods} 期, 耗时 {elapsed:.1f}s")

        # 诊断：体制/方向分布
        regime_counts = {}
        direction_counts = {}
        model_state_counts = {}
        for p in periods:
            r = p["regime"]
            regime_counts[r] = regime_counts.get(r, 0) + 1
            d = p["direction"]
            direction_counts[d] = direction_counts.get(d, 0) + 1
            ms = p.get("model_state", d)
            model_state_counts[ms] = model_state_counts.get(ms, 0) + 1
        logger.info(f"[DirBacktest] 体制分布: {regime_counts}")
        logger.info(f"[DirBacktest] 方向分布: {direction_counts}")
        logger.info(f"[DirBacktest] 状态模型使用: {model_state_counts}")
        if timing_ratios:
            ratios = list(timing_ratios.values())
            logger.info(f"[DirBacktest] 择时仓位: 均值 {sum(ratios)/len(ratios):.2%}, "
                       f"最小 {min(ratios):.2%}, 最大 {max(ratios):.2%}, "
                       f"<80% 的期数 {sum(1 for r in ratios if r < 0.8)}")

        metrics = self._compute_metrics(
            equity, benchmark_equity, equity_dates, periods, trades_log, freq)
        metrics["benchmark_definition"] = {
            "primary": "stable_history_500d_equal_weight",
            "secondary": "csi300",
            "timing": "same_portfolio_valuation_window",
            "membership_source": (
                "universe_membership_history"
                if benchmark_membership_by_date is not None else "prediction_universe_legacy"
            ),
        }
        metrics["run_window"] = {
            "start": str(start)[:10], "end": str(end)[:10],
            "decision_end": str(decision_end)[:10],
        }
        metrics["portfolio_contract"] = {
            "sizing": sizing,
            "target_exposure": float(target_exposure),
            "max_position_pct": float(self._max_position_pct),
            "max_sector_pct": float(self._max_sector_pct),
            "max_one_way_turnover": float(max_one_way_turnover),
        }
        metrics["secondary_benchmark_curve"] = [
            {"date": date, "value": round(float(value), 6)}
            for date, value in zip(equity_dates, secondary_benchmark_equity)
        ]
        if len(secondary_benchmark_equity) > 1:
            metrics["summary"]["annual_csi300"] = round(float(
                secondary_benchmark_equity[-1] ** (_freq_meta.periods_per_year / (len(secondary_benchmark_equity) - 1)) - 1
            ), 4)
            metrics["summary"]["csi300_return"] = round(float(secondary_benchmark_equity[-1] - 1), 4)
        neutralize_applied = bool(
            neutralize_qlib
            and (score_status_counts.get("applied") or score_status_counts.get("partial"))
            and sum(count for status, count in score_status_counts.items()
                    if status not in ("applied", "partial")) == 0
        )
        metrics["model_routing"] = {
            "mode": "fixed" if fixed_model else ("volatility" if use_vol_switch else "regime"),
            "explicit": bool(state_models or fixed_model),
            "models": {key: list(value) for key, value in direction_models.items()},
            "usage": model_state_counts,
        }
        metrics["score_processing"] = {
            "neutralize_qlib": neutralize_qlib,
            "neutralize_qlib_applied": neutralize_applied,
            "neutralize_qlib_fallback_count": neutralize_fallback_count,
            "neutralize_status_counts": score_status_counts,
            "pipeline_version": "score-ledger-v2",
            "model_input_score_available": bool(score_status_counts),
            "raw_model_score_available": (
                bool(score_status_counts)
                and all(value == "raw_model_score" for value in model_score_semantics.values())
            ),
            "model_input_semantics": sorted(set(model_score_semantics.values())),
            "score_semantics": "cross_sectional_rank_not_expected_return",
            "percentile_mapping": True,
            "macro_sector_bias": bool(macro_bias),
            "macro_regime": bool(macro_regime),
            "decision_day_filter": bool(decision_day_filter),
            "decision_day_filter_examined": decision_filter_examined,
            "decision_day_filter_excluded": decision_filter_excluded,
            "decision_day_filter_shortfalls": decision_filter_shortfalls,
        }
        metrics["decision_policy"] = {
            "enabled": decision_policy is not None,
            "policy": decision_policy,
            "periods": decision_periods,
        }
        return metrics

    @staticmethod
    def _materialize_point_in_time_context(dates, membership_by_date, references_by_date):
        date_keys = [value.isoformat() if hasattr(value, "isoformat") else str(value)[:10]
                     for value in dates]

        def materialize(source, label):
            if source is None:
                return None
            normalized = {str(key)[:10]: value for key, value in source.items()}
            source_dates = sorted(normalized)
            result = {}
            for decision_date in date_keys:
                eligible = [value for value in source_dates if value <= decision_date]
                if not eligible:
                    raise RuntimeError(f"{label}缺少不晚于决策日的快照: {decision_date}")
                result[decision_date] = normalized[eligible[-1]]
            return result

        return (
            materialize(membership_by_date, "PIT股票池成员"),
            materialize(references_by_date, "PIT证券状态"),
        )

    @staticmethod
    def _require_point_in_time_context(dates, membership_by_date, references_by_date):
        date_keys = [value.isoformat() if hasattr(value, "isoformat") else str(value)[:10]
                     for value in dates]
        if membership_by_date is not None:
            missing = [value for value in date_keys if value not in membership_by_date]
            if missing:
                raise RuntimeError(f"PIT股票池成员缺少决策日: {missing[0]}")
        if references_by_date is not None:
            missing = [value for value in date_keys if value not in references_by_date]
            if missing:
                raise RuntimeError(f"PIT证券状态缺少决策日: {missing[0]}")
            if membership_by_date is not None:
                for value in date_keys:
                    absent = set(membership_by_date[value]) - set(references_by_date[value])
                    if absent:
                        raise RuntimeError(
                            f"PIT证券状态覆盖不完整: {value} 缺少 {len(absent)} 只"
                        )
                    unauthoritative = [
                        code for code in membership_by_date[value]
                        if references_by_date[value][code].get("source") != "pit_import"
                    ]
                    if unauthoritative:
                        raise RuntimeError(
                            f"PIT证券状态来源不权威: {value} 有 {len(unauthoritative)} 只"
                        )

    # ══════════════════════════════════════════════
    # 预加载 & 预计算
    # ══════════════════════════════════════════════

    def _prefetch_index_data(self, start_d, end_d):
        """一次性拉取沪深300日线，覆盖回测区间 + 前 2000 天（HMM 训练需要）"""
        try:
            from core.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.get_index_daily("沪深300", days=2000)
            if df is not None and len(df) >= 60:
                df = df.sort_values('date').reset_index(drop=True)
                # 统一列名: date → trade_date (与 HMM/择时特征 pipeline 一致)
                if 'date' in df.columns and 'trade_date' not in df.columns:
                    df = df.rename(columns={'date': 'trade_date'})
                # 确保日期为 Timestamp 类型, 避免字符串比较导致 df_slice 为空
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                logger.info(f"[DirBacktest] 指数数据预加载: {len(df)} 条 "
                           f"({df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]})")
                return df
        except Exception as e:
            logger.warning(f"[DirBacktest] 指数数据预加载失败: {e}")
        return None

    def _precompute_timing_ratios(self, dates, index_df, start_d):
        """批量预计算所有回测日期的择时比例 — 优先用 RegimePredictor"""
        ratios = {}

        # 优先用 RegimePredictor 的仓位系数 (V1 Tree+Linear 融合)
        try:
            from core.domain.regime.predictor import get_regime_predictor
            from core.domain.regime.predictor_v2 import get_regime_predictor_v2
            rp = get_regime_predictor()
            rp_v2 = get_regime_predictor_v2()

            if rp._is_loaded():
                for d in dates:
                    ratio = 1.0
                    try:
                        df_slice = index_df[index_df['trade_date'] <= str(d)].rename(columns={'trade_date': 'date'})
                        if len(df_slice) >= 60:
                            # V1 基础仓位
                            probs = rp.predict_from_df(df_slice)
                            if probs is not None:
                                from core.domain.regime.predictor import _soft_position_ratio
                                pos_base = _soft_position_ratio(probs)
                                # V1 隐含波动率缩放
                                ANCHOR_VOL = {
                                    "volatile_up": 0.225, "volatile_down": 0.209, "volatile_flat": 0.228,
                                    "steady_up": 0.112, "steady_down": 0.121, "steady_flat": 0.117,
                                }
                                STATE_NAMES = ["volatile_up","volatile_down","volatile_flat","steady_up","steady_down","steady_flat"]
                                implied_vol = sum(probs[i] * ANCHOR_VOL[s] for i, s in enumerate(STATE_NAMES))
                                vol_scalar = 1.0 / max(1.0, implied_vol / 0.17)
                                vol_scalar = max(0.70, min(1.0, vol_scalar))
                                ratio = max(0.15, min(1.00, pos_base * vol_scalar))
                    except Exception:
                        pass
                    ratios[d] = ratio
                logger.info(f"[DirBacktest] RegimePredictor 仓位预计算完成, "
                           f"avg={sum(ratios.values())/max(len(ratios),1):.2f}")
                return ratios
        except Exception as e:
            logger.warning(f"[DirBacktest] RegimePredictor 仓位失败: {e}")

        # 回退: 用旧的 IndexTimingModel
        tm = self._get_timing_model()
        if tm.lgb is None or index_df is None or len(index_df) < 100:
            return ratios

        try:
            from core.domain.timing.breadth import get_breadth_history

            min_d = (dates[0] - dt.timedelta(days=365 * 2)).isoformat()
            max_d = dates[-1].isoformat()
            breadth_df = get_breadth_history(min_d, max_d)
            if breadth_df is not None and not breadth_df.empty:
                breadth_df = breadth_df.reset_index()

            from core.domain.timing.model import _make_features, _get_regime_features
            use_hmm = any(c.startswith("hmm_") for c in tm.feature_names)

            n = len(dates)
            for i, d in enumerate(dates):

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

                if (i + 1) % 20 == 0:
                    logger.info(f"[DirBacktest] 择时预计算 {i + 1}/{n}")

            avg_ratio = sum(ratios.values()) / max(len(ratios), 1)
            logger.info(f"[DirBacktest] 择时预计算完成, {len(ratios)} 期, "
                       f"均值 {avg_ratio:.2%}")
        except Exception as e:
            logger.warning(f"[DirBacktest] 择时预计算失败: {e}")

        return ratios

    def _classify_regime(self, date, rule_classifier, index_df, use_rule_based=False, use_vol_switch=False,
                         breadth_df=None, vol_label_horizon=10, macro_regime=True):
        """体制分类 — 统一入口: RegimePredictor(主) → RuleBased(回退) + 宏观修正 + 因果 CRF 平滑"""
        from core.domain.regime.weights import get_regime_direction

        # use_rule_based=True: 确定性规则 (可复现, 不加载监督模型)
        if use_rule_based and rule_classifier is not None:
            try:
                rr = rule_classifier.classify(as_of=date)
                if macro_regime:
                    from core.domain.regime.unified import apply_macro_regime_correction
                    rr = apply_macro_regime_correction(rr, date)
                direction = get_regime_direction(rr.regime)
                return rr.regime.value, direction, rr.confidence, (None if use_vol_switch else rr.regime.value)
            except Exception as e:
                logger.warning(f"[DirBacktest] 规则分类失败 {date}: {e}")

        # 默认: 统一入口 (主模型 + 宏观修正 + 规则回退)
        try:
            from core.domain.regime.unified import get_unified_regime
            rr = get_unified_regime(as_of=date, index_df=index_df, breadth_df=breadth_df,
                                    apply_macro=macro_regime, use_vol_switch=use_vol_switch,
                                    vol_label_horizon=vol_label_horizon)
            regime_val = rr.regime.value
            confidence = rr.confidence
            # 因果 CRF 平滑 (与实盘一致, 仅用过去信息, 不引入前视)
            if self._crf_trans is not None and rr.sub_scores:
                from core.domain.regime.unified import crf_smooth_sequence, STATE_NAMES
                self._crf_prior.append([rr.sub_scores.get(s, 0.0) for s in STATE_NAMES])
                if len(self._crf_prior) > self._crf_window:
                    self._crf_prior = self._crf_prior[-self._crf_window:]
                smoothed = crf_smooth_sequence(self._crf_prior, self._crf_trans)
                if smoothed:
                    regime_val = max(smoothed, key=smoothed.get)
                    confidence = smoothed[regime_val]
            # 方向模型选择: 波动率切换(use_vol_switch) 或 6 状态映射(默认)
            if use_vol_switch:
                vol_dir = rr.details.get("vol_direction") if isinstance(rr.details, dict) else None
                if vol_dir:
                    direction = vol_dir
                    model_state = "high_vol" if vol_dir == "oversold_rebound" else "low_vol"
                    # 三档条件切换 (OOS 实验结论): 高波+宽度低/超跌深→rebound; 低波+无超跌→trend; 中波→neutral(降仓)
                    if getattr(self, "_use_conditional_switch", False) and self._cond_env:
                        p_vol = rr.details.get("p_vol")
                        date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)[:10]
                        env = self._cond_env.get(date_str)
                        if p_vol is not None and env is not None:
                            from core.domain.regime.weights import get_vol_direction
                            new_dir = get_vol_direction(
                                p_vol, market_breadth=env[0], oversold_degree=env[1],
                                use_conditional=True)
                            if new_dir == "neutral":
                                self._cond_neutral = True
                                new_dir = "trend_momentum"  # 空仓期仍须有模型引用; 仓位已置 0
                            direction = new_dir
                            model_state = "high_vol" if new_dir == "oversold_rebound" else "low_vol"
                else:
                    logger.warning(f"[DirBacktest] VolRegimeClassifier 不可用 {date}, 回退 6 状态方向映射")
                    direction = get_regime_direction(regime_val)
                    model_state = None
            else:
                direction = get_regime_direction(regime_val)
                model_state = regime_val
            return regime_val, direction, confidence, model_state
        except Exception as e:
            logger.warning(f"[DirBacktest] 统一入口失败 {date}: {e}, 回退规则")
            if rule_classifier is not None:
                try:
                    rr = rule_classifier.classify(as_of=date)
                    if macro_regime:
                        from core.domain.regime.unified import apply_macro_regime_correction
                        rr = apply_macro_regime_correction(rr, date)
                    direction = get_regime_direction(rr.regime)
                    return rr.regime.value, direction, rr.confidence, (None if use_vol_switch else rr.regime.value)
                except Exception:
                    pass

        return "unknown", "trend_momentum", 0, None

    @staticmethod
    def _nearest_pred(pred_df, target_date):
        """取最近一个不晚于 target_date 的预测日期"""
        try:
            available = sorted(pd.unique(pred_df.index.get_level_values('datetime')))
            closest = None
            for ad in available:
                ad_d = ad.date() if hasattr(ad, 'date') else pd.Timestamp(ad).date()
                if ad_d <= target_date:
                    closest = ad
            if closest is not None:
                return pred_df.loc[closest]
        except Exception:
            pass
        return None

    # ══════════════════════════════════════════════
    # 模型缓存
    # ══════════════════════════════════════════════

    @staticmethod
    def _validate_cached_run(exp_id, run_id):
        """校验缓存中的 run 是否仍然有效（模型文件存在）"""
        from pathlib import Path
        from core.shared import config as _app_config
        _base = Path(_app_config.QLIB_MLFLOW_URI.replace("file:", ""))
        if not _base.exists():
            _base = Path("mlruns")
        artifacts = _base / exp_id / run_id / "artifacts"
        return ((artifacts / "params_lgb.pkl").exists()
                or (artifacts / "params_xgb.pkl").exists()
                or (artifacts / "params.pkl").exists())

    def _load_models_cached(self, directions):
        from pathlib import Path

        def _validate_cache(cached):
            from core.shared import config as _app_config
            from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE, get_model_label_mode

            if not cached or not isinstance(cached, dict):
                return None
            if set(cached.keys()) != set(directions) or len(cached) < 1:
                return None
            for dir_name, (exp_id, run_id) in cached.items():
                from core.shared.disabled_models import is_disabled
                if is_disabled(exp_id, run_id):
                    logger.warning(f"[DirBacktest] 缓存模型 {dir_name}={exp_id}/{run_id[:8]} 已停用, 缓存失效")
                    return None
                if not self._validate_cached_run(exp_id, run_id):
                    logger.warning(f"[DirBacktest] 缓存模型 {dir_name}={exp_id}/{run_id[:8]} 文件缺失, 跳过")
                    return None
                base_dir = Path(_app_config.QLIB_MLFLOW_URI.replace("file:", ""))
                if not base_dir.exists():
                    base_dir = Path("mlruns")
                run_dir = base_dir / exp_id / run_id
                if get_model_label_mode(run_dir) != TRADABLE_LABEL_MODE:
                    logger.warning(f"[DirBacktest] 缓存模型 {dir_name}={exp_id}/{run_id[:8]} 不是开盘到开盘标签, 缓存失效")
                    return None
            return cached

        try:
            from core.data import SettingsRepo
            cached = SettingsRepo.get("direction_models")
            valid = _validate_cache(cached)
            if valid:
                logger.info(f"[DirBacktest] 命中 MySQL 模型缓存 ({len(valid)} 个方向)")
                return valid
            if cached:
                SettingsRepo.set("direction_models", {})
        except Exception:
            pass

        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            mtime = os.path.getmtime(MODEL_CACHE_FILE)
            if time.time() - mtime < MODEL_CACHE_TTL:
                from core.shared.persistence import load_json
                cached = load_json(MODEL_CACHE_FILE)
                valid = _validate_cache(cached)
                if valid:
                    logger.info(f"[DirBacktest] 命中文件模型缓存 ({len(valid)} 个方向)")
                    return valid
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        direction_models = {}
        used = set()
        for dir_name in directions:
            exclude_list = [m for m_list in direction_models.values() for m in ["/".join(m_list)]]
            best = self._find_best_model_for_direction(dir_name, exclude=exclude_list)
            if best:
                direction_models[dir_name] = list(best)
                used.add(f"{best[0]}/{best[1]}")
                logger.info(f"[DirBacktest] {dir_name}: {best[0]}/{best[1]}")
            else:
                # 兜底: 不限排除，取任意匹配模型
                best = self._find_best_model_for_direction(dir_name)
                if best:
                    direction_models[dir_name] = list(best)
                    logger.info(f"[DirBacktest] {dir_name}(fallback): {best[0]}/{best[1]}")
                elif direction_models:
                    # 二次兜底: 复用已加载的模型，确保 regime 轮动能真正切换方向
                    first_dir = next(iter(direction_models))
                    reused = direction_models[first_dir]
                    direction_models[dir_name] = list(reused)
                    logger.warning(f"[DirBacktest] {dir_name} 无专用模型, 复用 {first_dir}: {reused[0]}/{reused[1]}")

        if len(direction_models) >= 1:
            try:
                from core.data import SettingsRepo
                SettingsRepo.set("direction_models", direction_models)
            except Exception:
                try:
                    from core.shared.persistence import save_json
                    save_json(direction_models, MODEL_CACHE_FILE)
                except OSError:
                    pass

        return direction_models

    # ══════════════════════════════════════════════
    # 价格缓存 & 模拟
    # ══════════════════════════════════════════════

    def _preload_price_cache(self, start, end, codes, history_observations=None,
                             security_references_by_date=None):
        price_cache = {}
        self._limit_up_cache = {}
        self._limit_down_cache = {}
        self._latest_adj_factors = {}
        if not codes:
            return price_cache

        start_date = dt.date.fromisoformat(start[:10]) if isinstance(start, str) else start
        lookback_days = max(120, int(history_observations * 1.6) + 30) if history_observations else 120
        load_start = start_date - dt.timedelta(days=lookback_days)
        start_str = load_start.strftime('%Y%m%d')
        end_str = end.replace('-', '') if isinstance(end, str) else end.strftime('%Y%m%d')

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
            logger.info(f"[DirBacktest] Redis 命中 {len(codes) - len(missing)}/{len(codes)}, "
                        f"MySQL 补加载 {len(missing)} 只")
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
                logger.warning(f"[DirBacktest] 批量加载失败: {e}", exc_info=True)

        if self._latest_adj_factor_loader is not None:
            try:
                self._latest_adj_factors = self._latest_adj_factor_loader(codes) or {}
            except Exception as e:
                logger.warning(f"[DirBacktest] 最新复权因子加载失败: {e}")

        st_codes = set() if security_references_by_date is not None else self._load_st_codes()
        st_codes_by_date = None
        if security_references_by_date is not None:
            st_codes_by_date = {
                str(value).replace("-", "")[:8]: {
                    code for code, row in references.items() if row.get("is_st")
                }
                for value, references in security_references_by_date.items()
            }
        precompute_limit_up(
            price_cache, self._limit_up_cache, st_codes,
            st_codes_by_date=st_codes_by_date,
        )
        precompute_limit_down(
            price_cache, self._limit_down_cache, st_codes,
            st_codes_by_date=st_codes_by_date,
        )
        logger.info(f"[DirBacktest] 价格缓存就绪: {len(price_cache)} 只, "
                    f"涨停缓存: {sum(len(v) for v in self._limit_up_cache.values())} 条, "
                    f"跌停缓存: {sum(len(v) for v in self._limit_down_cache.values())} 条")
        return price_cache

    def _filter_decision_day_ranking(self, ranked_codes, decision_date, target_count,
                                     price_cache, sector_map, min_history_observations=None):
        eligible = []
        excluded = {
            "st": 0, "suspended": 0, "limit_up": 0, "low_liquidity": 0,
            "low_price": 0, "new_listing": 0, "insufficient_history": 0,
            "data_error": 0,
        }
        examined = 0
        min_turnover = 10000000
        try:
            from core.shared import config as app_config
            min_turnover = getattr(app_config, "BUY_SIDE_L0_MIN_TURNOVER", min_turnover)
        except Exception:
            pass
        try:
            from core.data import StockIdentityRepo
            listing_dates = StockIdentityRepo.get_listing_dates(ranked_codes)
        except Exception as error:
            logger.warning("[DirBacktest] 证券上市日期不可用，相关候选将按未知排除: %s", error)
            listing_dates = {}
        excluded["identity_unknown"] = 0
        for code in ranked_codes:
            examined += 1
            info = sector_map.get(code, {})
            name = info.get("name", "") if isinstance(info, dict) else ""
            assessment = evaluate_decision_day_filter(
                code, price_cache.get(code), decision_date,
                name=name, min_turnover=min_turnover,
                latest_adj_factor=self._latest_adj_factors.get(code),
                limit_pct=limit_pct_for_code(code),
                min_history_observations=min_history_observations,
                listing_date=listing_dates.get(code),
            )
            reason = assessment["reason"]
            if reason:
                excluded[reason] += 1
                continue
            eligible.append(code)
            if len(eligible) >= target_count:
                break
        return eligible, excluded, examined

    def _load_sector_map(self):
        try:
            from core.data.fetcher import DataFetcher
            return DataFetcher().get_industry_map() or {}
        except Exception:
            return {}

    def _load_st_codes(self):
        """加载 ST 股代码集合 (name 含 ST, 涨跌幅 5%)。"""
        try:
            from core.data.fetcher import DataFetcher
            ind_map = DataFetcher().get_industry_map() or {}
            return {c for c, v in ind_map.items()
                    if isinstance(v, dict) and 'ST' in str(v.get('name', '')).upper()}
        except Exception:
            return set()

    def _get_trading_days_between(self, start_date, end_date):
        """获取 start_date (不含) 到 end_date (含) 之间的交易日列表"""
        if self._index_df is None:
            return []
        dates = pd.to_datetime(self._index_df['trade_date'])
        mask = (dates > pd.Timestamp(start_date)) & (dates < pd.Timestamp(end_date))
        return sorted(dates[mask].dt.date.tolist())

    def _next_trading_day(self, d):
        """返回 d 之后第一个交易日 date；无则返回 None (最后一个调仓日回退)。"""
        if self._index_df is None:
            return None
        dates = pd.to_datetime(self._index_df['trade_date'])
        future = dates[dates > pd.Timestamp(d)]
        return future.iloc[0].date() if len(future) else None

    def _daily_risk_check(self, holdings, entry_dates, entry_prices, cash, cooldown, half_sold,
                          prev_date, curr_date, price_cache):
        """日频阶梯止损止盈: 逐日检查。
        止损: -5%半仓(不冷冻) → -10%清仓(冷冻4周)
        止盈: +15%半仓(不冷冻) → +30%清仓(冷冻4周)
        """
        trading_days = self._get_trading_days_between(prev_date, curr_date)
        if not trading_days:
            return holdings, entry_dates, entry_prices, cash, cooldown, half_sold, []

        trades = []
        for day in trading_days:
            day_str = day.isoformat()[:10]
            risk_info = []
            for code, init_w in list(holdings.items()):
                ep = entry_prices.get(code)
                if not ep or ep <= 0:
                    continue
                # A 股 T+1: 当日买入的股份当日不可卖出, 买入日跳过止损检查
                if entry_dates.get(code) == day_str:
                    continue
                df = price_cache.get(code)
                if df is None:
                    continue
                p_close = lookup_price(df, day_str, 'close')
                if p_close is None or p_close <= 0:
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

            if self._stop_source == "dynamic":
                actions = self._dynamic_risk_actions(holdings, entry_prices, day_str, price_cache, half_sold)
            elif self._stop_source == "atr":
                actions = self._atr_risk_actions(holdings, entry_prices, day_str, price_cache, half_sold)
                eligible_codes = {item["code"] for item in risk_info}
                actions = {code: action for code, action in actions.items()
                           if code in eligible_codes}
            else:
                actions = self._allocator.classify_risk(
                    risk_info, half_sold=half_sold, use_intraday_low_high=True)

            for code, ra in actions.items():
                init_w = holdings.get(code, 0)
                ep = entry_prices.get(code)
                df = price_cache.get(code)
                p_close = lookup_price(df, day_str, 'close') if df is not None else None
                mv = init_w * (p_close / ep) if p_close and ep and ep > 0 else init_w

                if ra.action == "full":
                    fill_price = None
                    if ra.trigger_price is not None:
                        p_open = lookup_price(df, day_str, 'open') if df is not None else None
                        fill_price = p_open if p_open and p_open <= ra.trigger_price else ra.trigger_price
                        sell_value = init_w * (fill_price / ep)
                    else:
                        # fixed 止损/止盈按当日实际价成交 (mv 基于当日收盘价), 反映跳空/深跌而非理论止损价
                        sell_value = mv
                    proceeds = sell_value * (1 - self._sell_cost_rate)
                    cash += proceeds
                    trade = {"date": day_str, "code": code, "action": "sell",
                             "reason": ra.reason, "value": sell_value,
                             "cost": sell_value - proceeds,
                             "realized_return": sell_value / init_w - 1 if init_w else None}
                    if ra.trigger_price is not None:
                        trade["trigger_price"] = ra.trigger_price
                        trade["fill_price"] = fill_price
                        trade["gap_slippage"] = min(0.0, fill_price / ra.trigger_price - 1)
                    trades.append(trade)
                    del holdings[code]
                    entry_prices.pop(code, None)
                    entry_dates.pop(code, None)
                    cooldown[code] = ra.new_cooldown
                    half_sold.discard(code)
                elif ra.action == "half":
                    if "止损" in ra.reason:
                        half_value = init_w * (1 + self._allocator.config.stop_loss_warn) / 2
                    elif "止盈" in ra.reason:
                        half_value = init_w * (1 + self._allocator.config.take_profit_half) / 2
                    else:
                        half_value = mv / 2
                    proceeds = half_value * (1 - self._sell_cost_rate)
                    cash += proceeds
                    holdings[code] = init_w / 2
                    half_sold.add(code)
                    trades.append({"date": day_str, "code": code, "action": "sell",
                                   "reason": ra.reason, "value": half_value,
                                   "cost": half_value - proceeds,
                                   "realized_return": half_value * 2 / init_w - 1
                                   if init_w else None})

        return holdings, entry_dates, entry_prices, cash, cooldown, half_sold, trades

    def _dynamic_risk_actions(self, holdings, entry_prices, date_str, price_cache, half_sold):
        """动态 trailing stop 判定, 返回 {code: RiskAction}"""
        actions = {}
        for code in list(holdings.keys()):
            if code not in entry_prices:
                continue
            df = price_cache.get(code)
            if df is None:
                continue
            df_sorted = df.sort_values('trade_date')
            past = df_sorted[df_sorted['trade_date'] < date_str]
            if past.empty:
                continue
            high_series = past['high'].astype(float).tolist()
            current_low = lookup_price(df, date_str, 'low')
            action, reason = classify_dynamic_trailing(
                high_series, current_low,
                lookback=self._dynamic_lookback, trail_full=self._dynamic_trail,
                already_half=code in half_sold)
            if action:
                new_cooldown = self._cooldown_stop if action == "full" else 0
                actions[code] = RiskAction(code, action, reason, new_cooldown)
        return actions

    def _atr_risk_actions(self, holdings, entry_prices, date_str, price_cache, half_sold):
        actions = {}
        for code in list(holdings.keys()):
            ep = entry_prices.get(code)
            if not ep or ep <= 0:
                continue
            df = price_cache.get(code)
            if df is None:
                continue
            df_sorted = df.sort_values('trade_date')
            past = df_sorted[df_sorted['trade_date'] < date_str]
            atr = compute_atr(past['high'], past['low'], past['close'], self._atr_period)
            current_low = lookup_price(df, date_str, 'low')
            current_high = lookup_price(df, date_str, 'high')
            if atr is None:
                fallback = self._allocator.classify_risk([{
                    "code": code, "entry_price": ep,
                    "p_low": current_low, "p_high": current_high,
                }], half_sold=half_sold, use_intraday_low_high=True)
                if code in fallback:
                    actions[code] = fallback[code]
                continue
            action, reason = classify_atr_stop(ep, atr, self._atr_mult, current_low)
            if action:
                stop_price = ep - self._atr_mult * atr
                actions[code] = RiskAction(
                    code, action, reason, self._cooldown_stop, stop_price)
                continue
            pnl_high = current_high / ep - 1 if current_high and current_high > 0 else 0
            action, reason = classify_risk_action(
                pnl_high, -999.0, -999.0,
                self._take_profit_half, self._take_profit_full,
                already_half=code in half_sold)
            if action:
                cooldown = self._cooldown_profit if action == "full" else 0
                actions[code] = RiskAction(code, action, reason, cooldown)
        return actions

    def _simulate_period_fast(self, holdings, entry_dates, entry_prices, cash, cooldown, half_sold,
                              buy_codes, buy_scores, sell_threshold,
                              curr_date, top_n,
                              price_cache, timing_ratio, sector_map, next_str=None):
        curr_str = curr_date.isoformat()[:10]
        if next_str is None:
            next_str = curr_str
        trades = []

        # ── 1. 按当前收盘价计算持仓市值 ──
        current_mv = {}
        total_mv = 0.0
        for code, init_w in list(holdings.items()):
            df = price_cache.get(code)
            if df is None:
                current_mv[code] = init_w
                total_mv += init_w
                continue
            p_curr = lookup_price(df, next_str, 'open')
            ep = entry_prices.get(code)
            if p_curr and p_curr > 0 and ep and ep > 0:
                mv = init_w * (p_curr / ep)
            else:
                mv = init_w
            current_mv[code] = mv
            total_mv += mv

        total_start = cash + total_mv
        if total_start <= 0:
            logger.warning(f"[DirBacktest] {curr_str} 组合净值归零, 强制清仓")
            return 0.0, {}, {}, {}, 0.0, cooldown, {}, trades

        # ── 2. 归一化为当前组合权重 ──
        current_weights = {c: mv / total_start for c, mv in current_mv.items()}

        # ── 3. 阶梯止损止盈检查 (用日内最低/最高价) ──
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
        if self._stop_source == "dynamic":
            risk_actions = self._dynamic_risk_actions(holdings, entry_prices, curr_str, price_cache, half_sold)
        elif self._stop_source == "atr":
            risk_actions = self._atr_risk_actions(holdings, entry_prices, curr_str, price_cache, half_sold)
        else:
            risk_actions = self._allocator.classify_risk(
                risk_info, half_sold=half_sold, use_intraday_low_high=True)

        # ── 5. 计算目标权重 ──
        target_weights = self._allocator.allocate_weights_only(
            [{"code": c, "score": buy_scores.get(c, 0)} for c in buy_codes],
            current_weights, sector_map, cooldown)
        target_weights = self._allocator._normalize_with_timing(
            target_weights, timing_ratio)

        # ── 6. 卖出: 阶梯止损止盈 / 不在持有区 / 超配 ──
        sell_proceeds = 0.0
        limit_down_skipped = 0
        for code, cur_w in list(current_weights.items()):
            if self._is_limit_down(code, next_str):
                limit_down_skipped += 1
                continue
            ra = risk_actions.get(code)
            action = ra.action if ra else None
            reason = ra.reason if ra else ""

            if action == "full":
                fill_price = None
                if ra and ra.trigger_price is not None:
                    p_open = lookup_price(price_cache.get(code), next_str, 'open')
                    fill_price = p_open if p_open and entry_prices.get(code) else None
                    sell_value = holdings[code] * (p_open / entry_prices[code]) \
                        if p_open and entry_prices.get(code) else cur_w * total_start
                elif "止损" in reason or "止盈" in reason:
                    # fixed 止损/止盈按调仓日实际收盘价成交, 反映跳空/深跌而非理论止损价
                    _p = lookup_price(price_cache.get(code), next_str, 'close')
                    sell_value = holdings[code] * (_p / entry_prices[code]) \
                        if _p and entry_prices.get(code) else cur_w * total_start
                else:
                    sell_value = cur_w * total_start
                proceeds = sell_value * (1 - self._sell_cost_rate)
                sell_proceeds += proceeds
                trade = {"date": next_str, "code": code, "action": "sell",
                         "reason": reason, "value": sell_value,
                         "cost": sell_value - proceeds,
                         "realized_return": sell_value / holdings[code] - 1
                         if holdings.get(code) else None}
                if ra and ra.trigger_price is not None and fill_price is not None:
                    trade["trigger_price"] = ra.trigger_price
                    trade["fill_price"] = fill_price
                    trade["gap_slippage"] = min(0.0, fill_price / ra.trigger_price - 1)
                trades.append(trade)
                del holdings[code]
                entry_prices.pop(code, None)
                entry_dates.pop(code, None)
                cooldown[code] = ra.new_cooldown if ra else (self._cooldown_stop if "止损" in reason else self._cooldown_profit)
                half_sold.discard(code)
                current_mv.pop(code, None)
            elif action == "half":
                original_value = holdings[code]
                if "止损" in reason:
                    half_value = holdings[code] * (1 + self._allocator.config.stop_loss_warn) / 2
                elif "止盈" in reason:
                    half_value = holdings[code] * (1 + self._allocator.config.take_profit_half) / 2
                else:
                    half_value = cur_w * total_start / 2
                proceeds = half_value * (1 - self._sell_cost_rate)
                sell_proceeds += proceeds
                holdings[code] = holdings[code] / 2
                current_mv[code] = current_mv[code] / 2
                half_sold.add(code)
                trades.append({"date": next_str, "code": code, "action": "sell",
                               "reason": reason, "value": half_value,
                               "cost": half_value - proceeds,
                               "realized_return": half_value * 2 / original_value - 1
                               if original_value else None})
            elif code not in sell_threshold:
                sell_value = cur_w * total_start
                proceeds = sell_value * (1 - self._sell_cost_rate)
                sell_proceeds += proceeds
                trades.append({"date": next_str, "code": code, "action": "sell",
                               "reason": "不在持有区", "value": sell_value,
                               "cost": sell_value - proceeds})
                del holdings[code]
                entry_prices.pop(code, None)
                entry_dates.pop(code, None)
                half_sold.discard(code)
                current_mv.pop(code, None)
            else:
                tgt = target_weights.get(code, 0)
                if cur_w > tgt * 1.20 and cur_w > 0.02:
                    sell_value = cur_w * total_start
                    proceeds = sell_value * (1 - self._sell_cost_rate)
                    sell_proceeds += proceeds
                    trades.append({"date": next_str, "code": code, "action": "sell",
                                   "reason": f"超配 (当前{cur_w:.1%} > 目标{tgt:.1%}×1.2)",
                                   "value": sell_value, "cost": sell_value - proceeds})
                    del holdings[code]
                    entry_prices.pop(code, None)
                    entry_dates.pop(code, None)
                    half_sold.discard(code)
                    current_mv.pop(code, None)

        cash += sell_proceeds

        # ── 7. 重算当前权重 (卖出后) ──
        total_post_sell = cash + sum(current_mv.values())
        if total_post_sell <= 0:
            return total_post_sell, holdings, entry_dates, entry_prices, cash, cooldown, half_sold, trades
        current_weights = {c: mv / total_post_sell for c, mv in current_mv.items()}

        # ── 8. 应用换手率约束 ──
        target_weights = self._allocator._apply_turnover_limit(
            target_weights, current_weights)

        # ── 9. 买入: 目标权重中未持有 or 低配 ──
        remaining = cash
        limit_up_skipped = 0
        unavailable_skipped = 0
        for code, tgt_w in target_weights.items():
            if code in cooldown and cooldown[code] > 0:
                continue
            cur_w = current_weights.get(code, 0)
            if tgt_w > cur_w * 1.12:
                # 涨停板流动性检查: 当日涨停封板的股票现实中无法买入
                df = price_cache.get(code)
                if self._is_limit_up(code, next_str):
                    limit_up_skipped += 1
                    continue
                buy_price = lookup_price(df, next_str, 'open') if df is not None else None
                if buy_price is None or buy_price <= 0:
                    unavailable_skipped += 1
                    continue
                buy_value = (tgt_w - cur_w) * total_post_sell
                cost = buy_value * (1 + self._buy_cost_rate)
                if cost > remaining * 0.8:
                    cost = remaining * 0.8
                    buy_value = cost / (1 + self._buy_cost_rate)
                remaining -= cost
                if remaining < 0:
                    break
                if code in holdings:
                    old_cost = holdings[code]
                    old_ep = entry_prices.get(code)
                    holdings[code] = old_cost + buy_value
                    if old_ep and old_ep > 0 and buy_price and buy_price > 0:
                        entry_prices[code] = (old_cost + buy_value) / (old_cost / old_ep + buy_value / buy_price)
                    half_sold.discard(code)  # 重新加仓, 清除半仓标记
                else:
                    holdings[code] = buy_value
                    if buy_price is not None:
                        entry_prices[code] = buy_price
                        entry_dates[code] = next_str
                trades.append({"date": next_str, "code": code, "action": "buy",
                               "reason": f"新进/加仓 (目标{tgt_w:.1%})",
                               "value": buy_value, "cost": cost - buy_value})

        if limit_up_skipped > 0:
            logger.info(f"[DirBacktest] {curr_str} 涨停跳过: {limit_up_skipped} 只")
        if unavailable_skipped > 0:
            logger.info(f"[DirBacktest] {curr_str} T+1无有效开盘价跳过: {unavailable_skipped} 只")
        if limit_down_skipped > 0:
            logger.info(f"[DirBacktest] {curr_str} 跌停跳过(卖): {limit_down_skipped} 只")

        cash = remaining

        # ── 10. 现金收益 (闲置资金买货基/短债, 年化3%) ──
        if self._cash_yield and cash > 0:
            cash += cash * (1.03 ** (self._freq_days_per_period / 365) - 1)

        # ── 11. 期末总资产 = 现金 + 持仓市值 ──
        total_end = cash
        for code, init_w in holdings.items():
            df = price_cache.get(code)
            if df is not None:
                p_curr = lookup_price(df, next_str, 'close')
                ep = entry_prices.get(code)
                if p_curr and p_curr > 0 and ep and ep > 0:
                    total_end += init_w * (p_curr / ep)
                else:
                    total_end += init_w
            else:
                total_end += init_w

        return total_end, holdings, entry_dates, entry_prices, cash, cooldown, half_sold, trades

    def _is_limit_up(self, code, date_str):
        date_key = date_str.replace('-', '')[:8]
        return self._limit_up_cache.get(code, {}).get(date_key, False)

    def _is_limit_down(self, code, date_str):
        date_key = date_str.replace('-', '')[:8]
        return self._limit_down_cache.get(code, {}).get(date_key, False)

    def _initial_build(self, codes, scores, top_n, price_cache, date_str, sector_map, next_str=None):
        return initial_build(
            codes, scores, top_n, price_cache, date_str, sector_map,
            self._allocator, self._limit_up_cache, self._buy_cost_rate,
            score_key="score", log_label="[DirBacktest]", next_str=next_str,
        )

    # ══════════════════════════════════════════════
    # 指标计算 & 模型发现
    # ══════════════════════════════════════════════

    def _compute_metrics(self, equity, benchmark_equity, equity_dates,
                         periods, trades_log, freq):
        """回测绩效计算 — 委托给 compute_backtest_metrics (纯 L3 函数)。"""
        return compute_backtest_metrics(
            equity, benchmark_equity, equity_dates,
            periods, trades_log, freq,
            profit_factor_cap=None,
        )

    def _find_best_model_for_direction(self, direction_name, exclude=None):
        from core.shared import config as app_config
        from pathlib import Path
        from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE, get_model_label_mode
        from core.shared.disabled_models import is_disabled
        mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", ""))
        if not mlruns_dir.exists():
            mlruns_dir = Path("mlruns")
        if not mlruns_dir.exists():
            return None

        exclude_set = set(exclude) if exclude else set()
        best_ic = -999
        best_model = None
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
                run_key = f"{exp_dir.name}/{run_dir.name}"
                if run_key in exclude_set:
                    continue
                artifacts_dir = run_dir / "artifacts"
                if not (artifacts_dir / "params_lgb.pkl").exists() and \
                   not (artifacts_dir / "params_xgb.pkl").exists():
                    continue
                label = self._resolve_direction_labels(run_dir)
                if direction_name not in label:
                    continue
                metrics_dir = run_dir / "metrics"
                ic_val = None
                from core.shared.persistence import load_text
                for ic_name in ["Rank_IC", "IC"]:
                    ic_file = metrics_dir / ic_name
                    if ic_file.exists():
                        try:
                            txt = load_text(str(ic_file))
                            parts = txt.strip().split()
                            if len(parts) >= 2:
                                ic_val = float(parts[1])
                        except Exception:
                            pass
                if ic_val is None or not (ic_val > -999):
                    for vname in ["XGB_VALID_IC", "LGB_VALID_IC"]:
                        vf = metrics_dir / vname
                        if vf.exists():
                            try:
                                txt = load_text(str(vf))
                                parts = txt.strip().split()
                                if len(parts) >= 2:
                                    v = float(parts[1])
                                    if v > -999 and not math.isnan(v):
                                        ic_val = v
                                        break
                            except Exception:
                                pass
                if ic_val is not None and ic_val > best_ic and not math.isnan(ic_val):
                    best_ic = ic_val
                    best_model = (exp_dir.name, run_dir.name)
        if best_model:
            logger.info(f"[DirBacktest] {direction_name}: {best_model[0]}/{best_model[1]} IC={best_ic:.4f}")
        return best_model

    @staticmethod
    def _resolve_direction_labels(run_dir):
        """返回该模型匹配的所有方向标签集合。

        两阶段匹配:
        1. 模型因子主要在精选集中 (专训模型): fk_set * 0.8
        2. 模型覆盖了精选集的大部分因子 (通用模型): curated_set * 0.5
        """
        import json
        from core.shared.persistence import load_pickle
        fk_json = run_dir / "artifacts" / "factor_keys.json"
        factor_keys = None
        if fk_json.exists():
            try:
                raw = load_pickle(str(fk_json))
                data = json.loads(raw) if isinstance(raw, str) else raw
                factor_keys = data.get("custom_factor_keys", [])
            except Exception:
                try:
                    from core.shared.persistence import load_text
                    data = json.loads(load_text(str(fk_json)))
                    factor_keys = data.get("custom_factor_keys", [])
                except Exception:
                    pass
        if not factor_keys:
            return set()
        try:
            from core.domain.factors.registry import get_pool
            pool = get_pool()
            curated = pool.state.get("curated", {})
            if curated:
                fk_set = set(factor_keys)
                matched = set()
                # 策略1: 模型因子主要在精选集中
                for name in curated:
                    cur_list = curated[name].get("flat_list", [])
                    if not cur_list:
                        continue
                    overlap = len(fk_set & set(cur_list))
                    if overlap >= len(fk_set) * 0.8:
                        matched.add(name)
                # 策略2: 模型覆盖了精选集的大部分因子 (覆盖率>=50%)
                if not matched:
                    for name in curated:
                        cur_list = curated[name].get("flat_list", [])
                        if not cur_list:
                            continue
                        cur_set = set(cur_list)
                        overlap = len(fk_set & cur_set)
                        ratio = overlap / len(cur_set)
                        if ratio >= 0.5:
                            matched.add(name)
                return matched
        except Exception:
            pass
        return set()
