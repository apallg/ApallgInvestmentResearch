"""散户热门板块热度 — 纯计算 (L3, 零 I/O)

将个股资金结构 (散户/中户/大户/机构四档净流入) + 换手率 + 涨跌幅,
按板块聚合后加权成"散户热度分", 输出板块排行。

资金档位语义 (tushare moneyflow, 金额单位万元):
  sm  小单   <5万    散户
  md  中单   5-20万  中户
  lg  大单   20-100万 大户
  elg 超大单 >100万   机构
  main_net_flow = lg + elg (主力, tushare 定义)
"""
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "retail_inflow": 0.5,   # 散户流入强度 (散户净额 / 流通市值)
    "turnover": 0.2,        # 平均换手率
    "pct_chg": 0.2,         # 平均涨跌幅
    "advance_ratio": 0.1,   # 上涨家数占比
}

FLOW_COLS = ["sm_net_flow", "md_net_flow", "lg_net_flow", "elg_net_flow", "main_net_flow"]


def _zscore(s):
    s = s.astype(float)
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def compute_retail_heat(stock_df, sector_map, weights=None, min_stocks=3):
    """计算板块散户热度排行。

    stock_df: DataFrame, 每行一只股票, 需含列:
        code, sm_net_flow, md_net_flow, lg_net_flow, elg_net_flow,
        main_net_flow, turnover_rate, pct_chg, circ_mv
    sector_map: {code: [sector_name, ...]} (一只股票可属多个概念板块)
    weights: 维度权重 dict, 缺省用 DEFAULT_WEIGHTS

    返回: DataFrame[sector, heat_score, ...各维度原始值] 按热度降序
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    df = stock_df.copy()
    # circ_mv 缺失或为 0 的股票不参与散户流入强度计算 (避免除零)
    df["_mv"] = df["circ_mv"].astype(float).replace(0, np.nan)

    # 展开 code -> sector 长表
    rows = []
    for code, sectors in sector_map.items():
        for sn in sectors:
            rows.append((code, sn))
    if not rows:
        return pd.DataFrame(columns=["sector", "heat_score"])
    map_df = pd.DataFrame(rows, columns=["code", "sector"])

    merged = df.merge(map_df, on="code", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["sector", "heat_score"])

    # 散户流入强度 = 散户净额 / 流通市值 (无量纲, 去除大票偏差)
    merged["_retail_strength"] = merged["sm_net_flow"].astype(float) / merged["_mv"]

    agg = merged.groupby("sector").agg(
        stock_count=("code", "count"),
        sm_net=("sm_net_flow", "sum"),
        md_net=("md_net_flow", "sum"),
        lg_net=("lg_net_flow", "sum"),
        elg_net=("elg_net_flow", "sum"),
        main_net=("main_net_flow", "sum"),
        retail_strength=("_retail_strength", "mean"),
        avg_turnover=("turnover_rate", "mean"),
        avg_pct_chg=("pct_chg", "mean"),
        advance_ratio=("pct_chg", lambda s: (s.astype(float) > 0).mean()),
    ).reset_index()

    # 成分股过少的板块剔除
    agg = agg[agg["stock_count"] >= min_stocks].copy()

    score = (
        w["retail_inflow"] * _zscore(agg["retail_strength"])
        + w["turnover"] * _zscore(agg["avg_turnover"])
        + w["pct_chg"] * _zscore(agg["avg_pct_chg"])
        + w["advance_ratio"] * _zscore(agg["advance_ratio"])
    )
    agg["heat_score"] = score.round(4)

    out = agg.sort_values("heat_score", ascending=False).reset_index(drop=True)
    return out
