"""
选股 L0 打分矩阵 — 纯计算, 不依赖 DB/文件/时间函数

compute_l0_matrix: 一次计算 snapshot 中所有股票的 13 个 L0 子分项,
供因子校验和选股管线复用。
"""
import logging

logger = logging.getLogger(__name__)


def compute_l0_matrix(snapshot, preloaded=None, as_of=None):
    """一次计算 snapshot 中所有股票的 13 个 L0 子分项, 返回 {code: {score_name: value}}。

    Args:
        snapshot: {code: (df, name, close, volume)} dict, 全部为内存数据
        preloaded: {'north_flow': dict, 'sentiment': dict, 'flow': dict,
                    'wb_flow': dict, 'financial': dict,
                    'industry_map': dict, 'industry_returns': dict}
        as_of: 决策日。透传给北向/市值/资金流/基本面打分, 保证 DB 回退路径不取未来数据。

    Returns:
        {code: {tech_score, momentum_score, ..., close, volume}}
    """
    from core.domain.signals.engine import get_fusion
    from core.domain.screening._scoring_legacy import (
        _score_volatility, _score_momentum,
        _score_industry_residual_momentum, _score_volume_price_divergence,
        _score_smart_money_flow, _score_north_flow, _score_size_turnover,
        _score_capital_flow, _score_fundamentals,
        _score_illiq, _score_overnight_gap,
    )

    indicator_weights = {"MACD": 1.0, "RSI": 0.8, "MA": 0.7, "KDJ": 0.5, "BB": 0.6, "VOLUME": 0.3}
    tw_sum = sum(indicator_weights.values()) or 1.0
    p = preloaded or {}
    north_flow_data = p.get("north_flow", {})
    sentiment_data = p.get("sentiment", {})
    flow_data = p.get("flow", {})
    wb_flow_data = p.get("wb_flow", {})
    industry_map = p.get("industry_map", {})
    financial_cache = p.get("financial", {})
    industry_returns = p.get("industry_returns", {})

    matrix = {}
    for code, entry in snapshot.items():
        try:
            df, name, close, volume = entry
            tech_score = 0.0
            try:
                fr = get_fusion(weights=indicator_weights).evaluate(df)
                tech_score = fr.score / tw_sum
            except Exception:
                logger.debug(f"[L0Matrix] tech_score calc failed for {code}")

            matrix[code] = {
                "tech_score":         round(tech_score, 3),
                "momentum_score":     round(_score_momentum(df, n=20), 3),
                "volatility_score":   round(_score_volatility(df, n=20), 3),
                "wb_industry_score":  round(_score_industry_residual_momentum(code, df, n=20, industry_map=industry_map, industry_returns=industry_returns), 3),
                "wb_divergence_score": round(_score_volume_price_divergence(df, n=20), 3),
                "wb_moneyflow_score": round(_score_smart_money_flow(code, df, n=20, flow_data=wb_flow_data), 3),
                "north_flow_score":   round(_score_north_flow(code, df, north_flow_data, as_of=as_of), 3),
                "size_turnover_score": round(_score_size_turnover(code, df, eval_date=as_of), 3),
                "sentiment_score":    round(sentiment_data.get(code, 0.0), 3),
                "flow_score":         round(_score_capital_flow(code, fetcher=None, as_of=as_of, flow_cache=flow_data), 3),
                "fundamental_score":  round(_score_fundamentals(code, fetcher=None, as_of=as_of, financial_cache=financial_cache), 3),
                "illiq_score":        round(_score_illiq(df, n=20), 3),
                "gap_score":          round(_score_overnight_gap(df, n=20), 3),
                "close": close, "volume": volume,
            }
        except Exception:
            pass
    return matrix


def rank_l1(candidates, progress_cb=None):
    """L1 精排：直通上游 L0 分数。纯 L3 函数。"""
    import logging
    logger = logging.getLogger(__name__)
    total = len(candidates)
    if progress_cb:
        progress_cb(0, total)

    for c in candidates:
        c["l1_score"] = round(c.get("l0_score", 0.0), 3)
        c["strategy_score"] = 0
        c["buy_count"] = 0
        c["sell_count"] = 0
        c["total_strategies"] = 0

    logger.info(f"[L1] {total} 只 (直通 L0)")
    if progress_cb:
        progress_cb(total, total)
    return sorted(candidates, key=lambda x: x["l1_score"], reverse=True)
