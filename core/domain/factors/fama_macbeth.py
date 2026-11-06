"""Fama-MacBeth (1973) 两阶段回归 — 检验因子定价能力

Stage 1: 时间序列 — 每只股票对因子做滚动回归估计 beta
Stage 2: 截面 — 每期收益对 beta 回归得到风险溢价 lambda
汇总: t 检验判断 lambda 是否显著非零

因 A 股日频数据可得，采用每日截面简化版：
  每天截面上，forward_return ~ factor1 + factor2 + ... + ε
  时间序列上对每个因子的系数做 t 检验（Newey-West 调整）
"""
import datetime
import logging
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def newey_west_t(series: np.ndarray, max_lag: int = 5) -> float:
    """Newey-West t-statistic"""
    n = len(series)
    if n < 10:
        return 0.0
    mu = np.mean(series)
    se = np.std(series, ddof=1) / np.sqrt(n)
    if se < 1e-10:
        return 99.0 if mu > 0 else -99.0
    t_ols = mu / se
    adj = 1.0
    for lag in range(1, min(max_lag + 1, n - 1)):
        rho = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        if not np.isnan(rho):
            adj += 2 * (1 - lag / (max_lag + 1)) * rho
    adj = max(adj, 1.0)
    return t_ols / np.sqrt(adj)


# 传统因子: 无表达式, 直接从 DataFrame 计算
_TRADITIONAL_SCORERS = {
    "momentum": lambda d: float(d["close"].iloc[-1] / d["close"].iloc[-21] - 1) if len(d) >= 21 else 0,
    "volatility": lambda d: float(d["close"].pct_change(fill_method=None).tail(20).std()) if len(d) >= 20 else 0,
    "size": lambda d: float(np.log1p(d["amount"].tail(20).mean())) if len(d) >= 5 else 0,
    "turnover": lambda d: float(d["volume"].tail(20).mean() / max(d.get("流通股本", [1e9])[-1], 1)) if len(d) >= 20 else 0,
    "illiq": lambda d: float((abs(d["close"].pct_change(fill_method=None)) / d["amount"]).tail(20).mean()) if len(d) >= 20 and "amount" in d.columns else 0,
}

# L0 矩阵因子映射: factor_key -> l0_column
_L0_FACTOR_MAP = {
    "momentum": "momentum_score",
    "volatility": "volatility_score",
    "tech": "tech_score",
    "north_flow": "north_flow_score",
    "size_turnover": "size_turnover_score",
}


def run_fama_macbeth(
    factor_keys: List[str],
    pool: str = "a500",
    start: datetime.date = None,
    end: datetime.date = None,
    eval_weeks: int = 52,
    forward_days: int = 20,
) -> List[dict]:
    """每日截面 Fama-MacBeth (优化版)

    优化点:
    1. 传统因子从 L0 矩阵批量取值, 省掉逐股 DataFrame 计算
    2. 白盒因子用实际表达式求值 (修复了之前用硬编码表达式的 bug)
    3. 去掉未使用的市值/行业查询
    """
    from core.domain.screening._screener import get_screener
    from core.domain.factors.forward_returns import get_forward_cache
    from core.shared.trading_time import is_trading_day
    from core.domain.factors.registry import get_pool

    if start is None:
        raise ValueError("start 参数为必填项, L3 禁止 date.today_date()")
    if end is None:
        raise ValueError("end 参数为必填项, L3 禁止 date.today_date()")

    screener = get_screener()
    fwd_cache = get_forward_cache()

    # 获取因子元信息: 表达式 + L0 映射
    pool_mgr = get_pool()
    all_factors = pool_mgr.get_all_factors()
    factor_meta = []  # [(key, expr, is_traditional, scorer)]
    for fk in factor_keys:
        info = all_factors.get(fk, {})
        expr = info.get("expr")
        l0_col = _L0_FACTOR_MAP.get(fk)
        traditional = _TRADITIONAL_SCORERS.get(fk)
        factor_meta.append((fk, expr, l0_col, traditional))

    # 采样评估日期
    d = start
    eval_dates = []
    while d <= end:
        if is_trading_day(d):
            eval_dates.append(d)
        d += datetime.timedelta(days=5)
    if len(eval_dates) < 20:
        return [{"error": "评估日期不足(需>=20个截面)"}]

    coef_series = defaultdict(list)
    n_factors = len(factor_keys)
    needs_tech = any(l0_col or traditional for _, _, l0_col, traditional in factor_meta)

    for i, dt in enumerate(eval_dates):
        try:
            snap = screener.precompute_snapshot(
                pool=pool, eval_date=dt, fast=True, skip_limit_up=True,
                skip_tech=not needs_tech,
            )
            if not snap or len(snap) < 30:
                continue
            codes = list(snap.keys())
            n_codes = len(codes)

            # 一次计算 L0 矩阵, 所有传统因子共享
            l0 = screener._compute_l0_matrix(snapshot=snap, as_of=dt)

            # 因子值矩阵 (n_stocks × n_factors)
            X = np.zeros((n_codes, n_factors))
            for j, (fk, expr, l0_col, traditional) in enumerate(factor_meta):
                if l0_col:
                    for ci, code in enumerate(codes):
                        X[ci, j] = l0.get(code, {}).get(l0_col, 0)
                elif expr:
                    from core.domain.factors.expression import evaluate as qlib_eval
                    for ci, code in enumerate(codes):
                        df = snap[code][0]
                        try:
                            v = qlib_eval(expr, df)
                            X[ci, j] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0
                        except Exception:
                            pass
                elif traditional:
                    for ci, code in enumerate(codes):
                        try:
                            X[ci, j] = traditional(snap[code][0])
                        except Exception:
                            pass

            # forward return
            fwd_vals = fwd_cache.get_list(codes, dt, forward_days)
            fwd = dict(zip(codes, fwd_vals))
            y = np.array([fwd.get(c, 0.0) for c in codes])

            # OLS: y = X @ beta
            X_design = np.column_stack([np.ones(n_codes), X])
            try:
                beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
                for j, (fk, _, _, _) in enumerate(factor_meta):
                    if not np.isnan(beta[j + 1]):
                        coef_series[fk].append(float(beta[j + 1]))
            except Exception:
                pass

            if i % 20 == 0:
                logger.info(f"[FM] {i+1}/{len(eval_dates)} 截面完成")
        except Exception as e:
            logger.debug(f"[FM] {dt} 跳过: {e}")

    # 汇总结果
    from scipy import stats as _st
    results = []
    for fk, _, _, _ in factor_meta:
        series = np.array(coef_series.get(fk, []))
        if len(series) < 20:
            results.append({"factor": fk, "risk_premium": 0, "t_stat": 0, "p_value": 1, "significant": False, "n": len(series)})
            continue
        mu = float(np.mean(series))
        t = float(newey_west_t(series))
        p = float(2 * _st.t.sf(abs(t), df=max(1, len(series) - 1)))
        results.append({
            "factor": fk, "risk_premium": round(mu * 10000, 2),
            "t_stat": round(t, 2), "p_value": round(p, 4),
            "significant": p < 0.05, "n": len(series),
        })

    results.sort(key=lambda x: abs(x["t_stat"]), reverse=True)
    return results
