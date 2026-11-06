"""因子拥挤度监控 — 多空收益波动率/偏度/相关性聚合

参考《因子投资方法与实践》第10章 + Arnott et al. (2019)
"""
import datetime
import logging
import numpy as np
from enum import StrEnum
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["CrowdingStatus", "check_crowding", "run_crowding_check",
           "monthly_crowding_report"]


class CrowdingStatus(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    CROWDED = "crowded"


def _rolling_std(series: List[float], window: int = 20) -> float:
    if len(series) < window:
        return 0.0
    return float(np.std(series[-window:]))


def _rolling_skew(series: List[float], window: int = 60) -> float:
    if len(series) < window:
        return 0.0
    from scipy import stats as _st
    return float(_st.skew(series[-window:]))


def check_crowding(
    factor_key: str,
    ls_returns: List[float],
    eval_date: datetime.date = None,
) -> dict:
    """检查单个因子的拥挤度状态

    返回: {"status": "normal"|"warning"|"crowded", "metrics": {...}}
    """
    result = {"factor": factor_key, "status": CrowdingStatus.NORMAL, "metrics": {}}
    if len(ls_returns) < 30:
        result["status"] = CrowdingStatus.NORMAL
        result["metrics"]["reason"] = "数据不足"
        return result

    vol_20d = _rolling_std(ls_returns, 20)
    vol_60d = _rolling_std(ls_returns, 60)
    skew_60d = _rolling_skew(ls_returns, 60)

    result["metrics"] = {
        "vol_20d": round(vol_20d, 6),
        "vol_60d": round(vol_60d, 6),
        "skew_60d": round(skew_60d, 3),
    }

    warnings = 0
    # 短期波动率急剧上升 (>1.5x 长期)
    if vol_60d > 0 and vol_20d > 1.5 * vol_60d:
        warnings += 2
    # 偏度大幅为负 (< -1.0)
    if skew_60d < -1.0:
        warnings += 2
    # 偏度中度为负 (< -0.5)
    elif skew_60d < -0.5:
        warnings += 1

    if warnings >= 3:
        result["status"] = CrowdingStatus.CROWDED
    elif warnings >= 1:
        result["status"] = CrowdingStatus.WARNING
    else:
        result["status"] = CrowdingStatus.NORMAL

    return result


def run_crowding_check(
    factor_infos: List[dict],
    pool: str = "a500",
    eval_weeks: int = 26,
    forward_days: int = 20,
    as_of: datetime.date = None,
) -> List[dict]:
    """批量拥挤度检测 — 用 batch_eval_all 一次求所有因子值, 然后分组回测得到 LS 收益, 再调 check_crowding

    300 因子 × 30 截面 ≈ 2-3 分钟 (取决于快照缓存命中率)
    """
    from core.domain.screening._screener import get_screener
    from core.domain.factors.forward_returns import get_forward_cache
    from core.shared.trading_time import is_trading_day
    from core.domain.factors.portfolio_sort import _quintile_buckets
    from core.domain.factors.expression import parse_factor, batch_eval_all

    screener = get_screener()
    fwd_cache = get_forward_cache()

    if as_of is None:
        raise ValueError("as_of 参数为必填项, L3 禁止 date.today_date()")
    end = as_of
    start = end - datetime.timedelta(days=eval_weeks * 7 + 30)

    d = start
    eval_dates = []
    while d <= end:
        if is_trading_day(d):
            eval_dates.append(d)
        d += datetime.timedelta(days=5)

    if len(eval_dates) < 10:
        return [{"factor": f.get("key"), "status": "normal", "metrics": {"reason": "评估日期不足"}} for f in factor_infos]

    # 预解析表达式 AST (共享, 一次)
    ast_map = {}      # {key: (inner_ast, outer_op)}
    no_expr_keys = []  # 无表达式的传统因子
    for fi in factor_infos:
        fk = fi.get("key", "")
        expr = fi.get("expr")
        if expr:
            try:
                ast_map[fk] = parse_factor(expr)
            except Exception:
                no_expr_keys.append(fk)
        else:
            no_expr_keys.append(fk)

    # 收集每期 LS 收益
    ls_collect = defaultdict(list)  # {factor_key: [ls_return_per_date]}

    for i, dt in enumerate(eval_dates):
        try:
            snap = screener.precompute_snapshot(
                pool=pool, eval_date=dt, fast=True, skip_limit_up=True,
                skip_tech=not no_expr_keys,
            )
            if not snap or len(snap) < 30:
                continue
            codes = list(snap.keys())
            fwd = fwd_cache.get_list(codes, dt, forward_days)

            # 白盒因子: batch_eval_all 一次求所有
            if ast_map:
                all_scores = batch_eval_all(ast_map, snap)
                for fk, scores in all_scores.items():
                    if len(scores) < 30:
                        continue
                    vals = np.array(list(scores.values()))
                    code_list = list(scores.keys())
                    buckets = _quintile_buckets(vals)
                    q5_r = [fwd.get(c, 0) for c, b in zip(code_list, buckets) if b == 4]
                    q0_r = [fwd.get(c, 0) for c, b in zip(code_list, buckets) if b == 0]
                    if q5_r and q0_r:
                        ls_collect[fk].append(np.mean(q5_r) - np.mean(q0_r))

            # 无表达式因子: 从 L0 矩阵取
            if no_expr_keys:
                l0 = screener._compute_l0_matrix(snapshot=snap, as_of=dt)
                score_map = {
                    "momentum": "momentum_score",
                    "volatility": "volatility_score",
                    "tech": "tech_score",
                    "size_turnover": "size_turnover_score",
                    "north_flow": "north_flow_score",
                    "size": "size_turnover_score",
                    "illiq": "size_turnover_score",
                    "turnover": "size_turnover_score",
                    "sentiment": "sentiment_score",
                    "flow": "flow_score",
                    "wb_industry": "wb_industry_score",
                    "wb_divergence": "wb_divergence_score",
                    "wb_moneyflow": "wb_moneyflow_score",
                }
                for fk in no_expr_keys:
                    col = score_map.get(fk, fk)
                    scores = {c: l0.get(c, {}).get(col, 0) for c in codes}
                    vals = np.array(list(scores.values()))
                    if len(vals) < 30:
                        continue
                    buckets = _quintile_buckets(vals)
                    q5_r = [fwd.get(c, 0) for c, b in zip(codes, buckets) if b == 4]
                    q0_r = [fwd.get(c, 0) for c, b in zip(codes, buckets) if b == 0]
                    if q5_r and q0_r:
                        ls_collect[fk].append(np.mean(q5_r) - np.mean(q0_r))

        except Exception:
            pass

    # 逐因子调 check_crowding
    results = []
    for fi in factor_infos:
        fk = fi.get("key", "")
        ls_returns = ls_collect.get(fk, [])
        cr = check_crowding(fk, ls_returns)
        results.append(cr)

    return results


def monthly_crowding_report(ls_returns_map: Dict[str, List[float]], as_of: datetime.date = None) -> str:
    """生成月度拥挤度 Markdown 报告"""
    if as_of is None:
        raise ValueError("as_of 参数为必填项, L3 禁止 date.today_date()")
    lines = ["# 因子拥挤度月报", "", f"生成时间: {as_of}", ""]
    lines.append("| 因子 | 状态 | 20日波动率 | 60日波动率 | 60日偏度 |")
    lines.append("|------|------|-----------|-----------|---------|")

    all_results = []
    status_icon = {CrowdingStatus.NORMAL: "🟢", CrowdingStatus.WARNING: "🟡", CrowdingStatus.CROWDED: "🔴"}
    for fk, returns in sorted(ls_returns_map.items()):
        r = check_crowding(fk, returns)
        all_results.append(r)
        m = r["metrics"]
        icon = status_icon.get(r["status"], "⚪")
        lines.append(f"| {fk} | {icon} {r['status']} | {m.get('vol_20d','?')} | {m.get('vol_60d','?')} | {m.get('skew_60d','?')} |")

    crowded_count = sum(1 for r in all_results if r["status"] == CrowdingStatus.CROWDED)
    warning_count = sum(1 for r in all_results if r["status"] == CrowdingStatus.WARNING)
    lines.append("")
    lines.append(f"**汇总**: {len(all_results)} 个因子, 🔴拥挤 {crowded_count}, 🟡关注 {warning_count}")

    return "\n".join(lines)
