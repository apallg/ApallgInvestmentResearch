"""
截面中性化 — 对因子分数按 log(市值) + 行业哑变量做 OLS 取残差, 剥离市值/行业风格暴露。
无 statsmodels 依赖, 用 numpy.linalg.lstsq。缺市值/行业时自动退化, 不阻断。
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["cross_sectional_neutralize", "neutralized_daily_ic",
           "time_weighted_mean", "orthogonalize_residual"]


def _industry_of(v):
    if isinstance(v, dict):
        return v.get("industry", "") or "__NA__"
    return v or "__NA__"


def time_weighted_mean(values, decay=0.97):
    """按新近度指数衰减加权求均值。values 按时间升序 (末尾最新)。
    w_i = decay ** (n-1-i), 最新权重最大。decay=1.0 退化为等权。"""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return 0.0
    if decay >= 1.0:
        return float(np.mean(vals))
    w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    return float(np.average(vals, weights=w))


def orthogonalize_residual(target, factor_matrix):
    """把 target 向量对 factor_matrix (每列一个已有因子) 做 OLS 取残差。
    含截距项。剥离已有因子能解释的部分, 残差即 target 的增量信息。
    factor_matrix 为空 → 原样返回 (去均值)。"""
    y = np.asarray(target, dtype=float)
    n = len(y)
    if factor_matrix is None or len(factor_matrix) == 0:
        return y - y.mean()
    cols = [np.ones(n)]
    for f in factor_matrix:
        arr = np.asarray(f, dtype=float)
        if arr.shape[0] == n and np.std(arr) > 1e-12:
            cols.append(arr)
    if len(cols) == 1:
        return y - y.mean()
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    except Exception:
        return y - y.mean()


def cross_sectional_neutralize(codes, scores, mktcap_map, industry_map, extra_factors=None,
                               return_details=False):
    """对齐 codes 的 scores 做截面中性化, 返回 {code: residual}。
    - mktcap_map: {code: total_mv}, 覆盖率不足则市值维不参与
    - industry_map: {code: {"industry": str}} 或 {code: str}, 行业过多/过少则不参与
    - extra_factors: {name: {code: value}} 附加回归列 (如动量/换手/波动),
      每列覆盖率不足则不参与; 缺失值用该列中位数填充
    无任何有效解释变量 → 原样返回 (residual = score)。"""
    n = len(codes)
    audit = {
        "status": "skipped_small_universe" if n < 10 else "no_effect_missing_exposures",
        "sample_count": n,
        "market_cap_coverage": 0.0,
        "industry_coverage": 0.0,
        "industry_count": 0,
        "active_exposures": [],
    }
    if n < 10:
        result = {c: float(s) for c, s in zip(codes, scores)}
        return (result, audit) if return_details else result

    y = np.asarray(scores, dtype=float)
    cols = [np.ones(n)]

    mv = np.array([float((mktcap_map or {}).get(c, 0) or 0) for c in codes])
    audit["market_cap_coverage"] = float((mv > 0).sum() / n) if n else 0.0
    if (mv > 0).sum() >= max(10, int(0.6 * n)):
        logmv = np.where(mv > 0, np.log(np.where(mv > 0, mv, 1.0)), np.nan)
        med = np.nanmedian(logmv)
        logmv = np.where(np.isnan(logmv), med, logmv)
        cols.append(logmv)
        audit["active_exposures"].append("market_cap")

    inds = [_industry_of((industry_map or {}).get(c)) for c in codes]
    uniq = [u for u in sorted(set(inds)) if u != "__NA__"]
    audit["industry_coverage"] = float(sum(value != "__NA__" for value in inds) / n) if n else 0.0
    audit["industry_count"] = len(uniq)
    if 2 <= len(uniq) <= max(2, n // 3):
        for u in uniq[:-1]:  # 丢弃一个作基准
            cols.append(np.array([1.0 if x == u else 0.0 for x in inds]))
        audit["active_exposures"].append("industry")

    if extra_factors:
        for _name, fmap in extra_factors.items():
            arr = np.array([float((fmap or {}).get(c, np.nan)) for c in codes])
            if np.isfinite(arr).sum() < max(10, int(0.6 * n)):
                continue
            med = np.nanmedian(arr)
            arr = np.where(np.isfinite(arr), arr, med)
            cols.append(arr)
            audit["active_exposures"].append(f"extra:{_name}")

    if len(cols) == 1:
        result = {c: float(s) for c, s in zip(codes, scores)}
        return (result, audit) if return_details else result

    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        primary = {value for value in audit["active_exposures"] if not value.startswith("extra:")}
        audit["status"] = "applied" if primary == {"market_cap", "industry"} else "partial"
    except Exception:
        resid = y
        audit["status"] = "fallback"
    result = {c: float(r) for c, r in zip(codes, resid)}
    return (result, audit) if return_details else result


def neutralized_daily_ic(codes, scores, forward_ret_map, mktcap_map, industry_map):
    """全截面中性化后与 forward return 做 Spearman 秩相关, 返回当日 IC (float 或 None)。"""
    resid = cross_sectional_neutralize(codes, scores, mktcap_map, industry_map)
    xs, ys = [], []
    for c in codes:
        r = (forward_ret_map or {}).get(c)
        if r is None or c not in resid:
            continue
        xs.append(resid[c])
        ys.append(r)
    if len(xs) < 10:
        return None
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(xs, ys)
        if corr is None or np.isnan(corr) or np.isinf(corr):
            return None
        return float(corr)
    except Exception:
        return None
