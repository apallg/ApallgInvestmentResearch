import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERT_ROLES = ("trend", "rebound")


def load_expert_factor_keys(role, path=None):
    if role not in EXPERT_ROLES:
        raise ValueError(f"不支持的专家角色: {role}")
    if path is None:
        from core.domain.factors.registry import get_pool
        curated = get_pool().state.get("curated", {})
    else:
        curated = json.loads(Path(path).read_text(encoding="utf-8"))["curated"]
    if not curated:
        raise RuntimeError("精选因子池为空")
    core = list(curated["core_base"]["flat_list"])
    satellite = curated[f"{role}_satellite"]["flat_list"]
    keys = list(dict.fromkeys(core + list(satellite)))
    if not keys:
        raise RuntimeError(f"{role} 专家冻结因子为空")
    return keys


def compute_state_affinity(frame, role):
    if role not in EXPERT_ROLES:
        raise ValueError(f"不支持的专家角色: {role}")
    trend = frame["trend_strength"].rank(pct=True)
    oversold = frame["oversold_degree"].rank(pct=True)
    breadth = frame["market_breadth"].rank(pct=True)
    vol = frame["realized_vol_percentile"].rank(pct=True)
    if role == "trend":
        affinity = 0.5 * trend + 0.3 * breadth + 0.2 * (1.0 - vol)
        formula = "0.5*trend+0.3*breadth+0.2*(1-vol)"
    else:
        affinity = 0.4 * oversold + 0.3 * vol + 0.3 * (1.0 - breadth)
        formula = "0.4*oversold+0.3*vol+0.3*(1-breadth)"
    return affinity, formula


def build_state_subset_profile(market, train_period, role):
    dates = pd.to_datetime(market["date"])
    frame = market.loc[
        (dates >= pd.Timestamp(train_period[0])) &
        (dates <= pd.Timestamp(train_period[1]))
    ].copy()
    if frame.empty:
        raise RuntimeError(f"{role} 专家的状态子样本为空")
    affinity, formula = compute_state_affinity(frame, role)
    threshold = float(affinity.quantile(0.5))
    date_strings = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    selected = date_strings.loc[affinity >= threshold].tolist()
    if len(selected) < 20:
        raise RuntimeError(f"{role} 专家状态子样本不足: {len(selected)}")
    profile = {
        "training_mode": "state_subset",
        "expert_role": role,
        "state_subset_version": 1,
        "state_subset_formula": formula,
        "state_subset_threshold": threshold,
        "state_subset_total_dates": int(len(frame)),
        "state_subset_n_dates": int(len(selected)),
        "state_subset_ratio": float(len(selected) / len(frame)),
        "selected_dates": selected,
    }
    return selected, profile


def load_causal_market_features(start, end):
    from core.data.connection import Database

    query = """SELECT trade_date,
                      AVG(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) AS breadth,
                      AVG(pct_chg) AS mean_ret
                 FROM daily_quotes
                WHERE trade_date BETWEEN %s AND %s
                  AND pct_chg IS NOT NULL
                GROUP BY trade_date ORDER BY trade_date"""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rows = []
    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(days=180), end_ts)
        rows.extend(Database.fetchall(query, (
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    if not rows:
        raise RuntimeError("状态子样本训练所需的 daily_quotes 市场数据为空")
    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    frame["breadth"] = pd.to_numeric(frame["breadth"], errors="coerce").fillna(0.5)
    frame["mean_ret"] = pd.to_numeric(frame["mean_ret"], errors="coerce").fillna(0.0)
    market_ret = frame["mean_ret"] / 100.0
    market_level = (1.0 + market_ret).cumprod()
    realized = market_level.pct_change(fill_method=None).rolling(20, min_periods=10).std().fillna(0.0)
    frame["realized_vol_percentile"] = realized.rolling(252, min_periods=30).apply(
        lambda values: float(np.mean(values <= values[-1])), raw=True
    ).fillna(0.5)
    ma20 = market_level.rolling(20, min_periods=5).mean()
    ma60 = market_level.rolling(60, min_periods=20).mean()
    frame["trend_strength"] = ((ma20 / ma60.replace(0, np.nan)) - 1.0).fillna(0.0)
    frame["oversold_degree"] = (-market_level.pct_change(20, fill_method=None)).clip(-1, 1).fillna(0.0)
    frame["market_breadth"] = frame["breadth"].clip(0, 1)
    return frame.reset_index().rename(columns={"trade_date": "date"})


def prepare_state_subset_training(train_period, role):
    feature_start = (pd.Timestamp(train_period[0]) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    market = load_causal_market_features(feature_start, train_period[1])
    selected, profile = build_state_subset_profile(market, train_period, role)
    profile["market_feature_start"] = feature_start
    profile["market_feature_end"] = train_period[1]
    return selected, profile
