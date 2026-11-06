"""诊断: 宏观行业修正在预测池中是否真的改变任何股票 + 中性化改变了哪些 top picks。"""
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.domain.backtest.common import norm_code
from core.domain.screening._screener import normalize_qlib_scores
from core.domain.screening.macro import apply_macro_sector_bias
from core.domain.macro.analyzer import MacroAnalyzer, SECTOR_ROTATION_RULES

path = "data/qlib_data/prediction_cache/lgb_lstm/44bafc51ae00cc50e9ec8fef550e3bc7b02bd07d8272cb0437a09efe548a6150.pkl"
p = pickle.load(open(path, "rb"))["prediction"]

from core.data.fetcher import DataFetcher
industry_map = DataFetcher().get_industry_map() or {}
print("industry_map size:", len(industry_map))
inds = set()
for v in industry_map.values():
    if isinstance(v, dict):
        inds.add(v.get("industry", ""))
print("industry names sample:", sorted(inds)[:40])

m = MacroAnalyzer()
for as_of in ("2025-07-01", "2025-09-30", "2026-01-05", "2026-03-02", "2026-06-01"):
    sigs = m.get_sector_rotation_signals(as_of=pd.Timestamp(as_of))
    fav = sorted({s for sig in sigs for s in sig.get("sectors", [])})
    matched = [s for s in fav if s in inds]
    print(f"{as_of}: signals={len(sigs)} fav_industries={fav[:8]} 在行业映射中命中={matched}")

# 检查股票池行业分布 vs 宏观偏好行业的交集
day = p.xs("2025-07-01", level="datetime")
scores = day.dropna()
num = {norm_code(k): float(v) for k, v in scores.items()}
codes = [c for c in num if len(c) == 6]
stock_inds = set()
for c in codes:
    v = industry_map.get(c)
    if isinstance(v, dict):
        stock_inds.add(v.get("industry", ""))
print()
print("股票池行业数:", len(stock_inds))
print("股票池行业 sample:", sorted(stock_inds)[:40])

# 中性化对 top10 的改动
raw_dict = {c: num[c] for c in codes}
neu_dict = normalize_qlib_scores(raw_dict, pd.Timestamp("2025-07-01"), industry_map, neutralize=True)
raw_top = sorted(raw_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]
neu_top = sorted(neu_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]
raw_set = {c for c, _ in raw_top}
neu_set = {c for c, _ in neu_top}
print()
print("raw top10:", [c for c, _ in raw_top])
print("neu top10:", [c for c, _ in neu_top])
print("交集:", len(raw_set & neu_set), "/10")
