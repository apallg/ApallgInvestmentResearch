"""Portfolio Sort — 分组回测检验因子单调性和经济显著性

按因子值将股票分5组(Q1最低~Q5最高)，计算各组持有期收益，检验：
- 单调性：Q1<Q2<Q3<Q4<Q5 的收益排序是否成立
- 多空收益：Q5-Q1 的年化收益、夏普比、最大回撤
- 胜率：Q5跑赢Q1的日期占比

参考《因子投资方法与实践》第4章。
"""
import datetime
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)


@dataclass
class SortResult:
    factor_key: str
    factor_label: str
    forward_days: int
    n_dates: int
    quintile_returns: Dict[int, List[float]] = field(default_factory=dict)
    long_short_returns: List[float] = field(default_factory=lambda: [])
    annual_return: float = 0.0
    annual_vol: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    monotonicity_score: float = 0.0
    hit_rate: float = 0.0
    mean_spread: float = 0.0


def _quintile_buckets(scores: np.ndarray) -> np.ndarray:
    """等分5组，0=最低, 4=最高"""
    n = len(scores)
    if n < 5:
        return np.zeros(n, dtype=int)
    idx = np.argsort(scores)
    buckets = np.zeros(n, dtype=int)
    for q in range(5):
        start = int(n * q / 5)
        end = int(n * (q + 1) / 5)
        buckets[idx[start:end]] = q
    return buckets


def _max_drawdown(returns: List[float]) -> float:
    cum = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cum)
    return float(np.max((peak - cum) / peak))


def _monotonicity(quintile_avg: Dict[int, float]) -> float:
    """检查 Q0<Q1<Q2<Q3<Q4 的单调性。返回0-1分，1=完美单调"""
    vals = [quintile_avg.get(q, 0) for q in range(5)]
    pairs = sum(1 for i in range(4) if vals[i] < vals[i + 1])
    return pairs / 4


def run_single_sort(
    factor_key: str,
    factor_label: str,
    forward_days: int,
    eval_dates: List[datetime.date],
    get_scores: callable,   # (eval_date) -> {code: factor_value}
    get_forward: callable,  # (eval_date, codes) -> {code: forward_return}
) -> SortResult:
    """对单个因子做分组回测"""
    result = SortResult(
        factor_key=factor_key,
        factor_label=factor_label,
        forward_days=forward_days,
        n_dates=len(eval_dates),
    )
    for q in range(5):
        result.quintile_returns[q] = []

    for d in eval_dates:
        try:
            scores = get_scores(d)
            if not scores or len(scores) < 5:
                continue
            codes = list(scores.keys())
            vals = np.array([scores[c] for c in codes])
            fwd = get_forward(d, codes)
            if not fwd:
                continue

            buckets = _quintile_buckets(vals)
            rets = []
            for c, q in zip(codes, buckets):
                if c in fwd:
                    rets.append((q, fwd[c]))
            if not rets:
                continue

            for q in range(5):
                q_rets = [r for b, r in rets if b == q]
                if q_rets:
                    result.quintile_returns[q].append(np.mean(q_rets))

            q5_r = [r for b, r in rets if b == 4]
            q0_r = [r for b, r in rets if b == 0]
            if q5_r and q0_r:
                result.long_short_returns.append(np.mean(q5_r) - np.mean(q0_r))
        except Exception:
            pass

    ls = np.array(result.long_short_returns)
    if len(ls) > 0:
        result.annual_return = float(np.mean(ls) * 252)
        result.annual_vol = float(np.std(ls) * np.sqrt(252))
        result.sharpe = result.annual_return / result.annual_vol if result.annual_vol > 0 else 0
        result.max_drawdown = _max_drawdown(result.long_short_returns)
        result.hit_rate = float(np.mean([1 if r > 0 else 0 for r in ls]))
        result.mean_spread = float(np.mean(ls))

    q_means = {q: np.mean(vals) if vals else 0 for q, vals in result.quintile_returns.items()}
    result.monotonicity_score = _monotonicity(q_means)
    return result


def run_batch_sort(
    factors: List[dict],
    pool: str = "a500",
    start: datetime.date = None,
    end: datetime.date = None,
    eval_weeks: int = 18,
    forward_days: int = 20,
    top_k: int = 30,
    code_prefixes: Optional[List[str]] = None,
    progress_cb: callable = None,
    snapshots=None,
) -> List[SortResult]:
    """批量分组回测"""
    from core.domain.screening._screener import get_screener
    from core.domain.factors.forward_returns import get_forward_cache
    from core.shared.trading_time import is_trading_day

    screener = get_screener()
    fwd_cache = get_forward_cache()

    if start is None:
        raise ValueError("start 参数为必填项, L3 禁止 date.today_date()")
    if end is None:
        raise ValueError("end 参数为必填项, L3 禁止 date.today_date()")

    # 采样评估日期
    d = start
    eval_dates = []
    while d <= end:
        if is_trading_day(d):
            eval_dates.append(d)
        d += datetime.timedelta(days=5)
    if not eval_dates:
        return []

    results = []
    total = len(factors)
    for i, f in enumerate(factors):
        key = f.get("key", "")
        label = f.get("label", key)
        logger.info(f"[PortSort] {key} ({i+1}/{total})")

        # 按日期获取因子值和forward return
        date_scores = {}
        date_codes = set()

        for d in eval_dates:
            try:
                if snapshots is None:
                    snap = screener.precompute_snapshot(
                        pool=pool, eval_date=d, fast=True, skip_limit_up=True,
                    )
                else:
                    dkey = d.isoformat() if hasattr(d, "isoformat") else str(d)
                    snap = snapshots[dkey]
                if not snap:
                    continue
                codes = list(snap.keys())
                scores = {}
                if "score" in f:
                    scores = {c: f["score"](snap[c][0]) for c in codes}
                else:
                    # 从 _compute_l0_matrix 取对应因子分
                    matrix = screener._compute_l0_matrix(snapshot=snap, as_of=d)
                    if matrix:
                        # matrix 是 {code: {score_name: value}}, 转换为 {code: score}
                        score_field = key + "_score"
                        scores = {}
                        for c, sv in matrix.items():
                            if score_field in sv:
                                scores[c] = sv[score_field]
                            elif key in sv:
                                scores[c] = sv[key]
                if scores:
                    date_scores[d] = scores
                    date_codes.update(scores.keys())
            except Exception as e:
                logger.debug(f"[PortSort] {key} {d} 跳过: {e}")

        result = run_single_sort(
            factor_key=key,
            factor_label=label,
            forward_days=forward_days,
            eval_dates=[d for d in eval_dates if d in date_scores],
            get_scores=lambda d, ds=date_scores: ds.get(d, {}),
            get_forward=lambda d, codes, fc=fwd_cache, fd=forward_days: fc.get_list(codes, d, fd),
        )
        results.append(result)

        if progress_cb:
            progress_cb(i + 1, total)

    return results
