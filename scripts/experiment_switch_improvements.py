
# -*- coding: utf-8 -*-
"""改进步骤对照实验: 单模型 / 二值切换 / 三档+空仓 / 改进1-4 (vol60稳定输入·前5日均值确认·死亡区重定义·半仓渐变)
阈值只在 w1 上选择, w2/w3/final 为独立验证。"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

ROOT = Path(r"C:/Users/zhj/Desktop/AShareSignalAdvisor/Documents/Apallg投研V2")

# ── 每日 vol60 分位 + 日间档位 (point-in-time) ──
from core.data.fetcher import DataFetcher
idx = DataFetcher().get_index_daily("沪深300", days=365 * 10)
idx = idx.sort_values("date").reset_index(drop=True)
idx["date"] = pd.to_datetime(idx["date"])
rets = idx["close"].astype(float).pct_change(fill_method=None)
idx["vol60"] = rets.rolling(60).std() * np.sqrt(250)
idx["vol60_pct250"] = idx["vol60"].rolling(250, min_periods=60).apply(lambda x: float((x <= x[-1]).mean()), raw=True)
pos = idx.set_index("date")

def tier60(p, lo=0.40, hi=0.60):
    if pd.isna(p): return np.nan
    if p > hi: return "high"
    if p < lo: return "low"
    return "mid"

pos["tier60"] = pos["vol60_pct250"].apply(lambda p: tier60(p))
# 日间档位变化次数 (滚动10日, 含当日)
pos["tier_chg"] = (pos["tier60"] != pos["tier60"].shift(1)).astype(float)
pos["instab10"] = pos["tier_chg"].rolling(10, min_periods=5).mean()  # 近10日档位变化率
pos["vol60_pct5"] = pos["vol60_pct250"].rolling(5, min_periods=3).mean()  # 前5日均值

# ── 228 期决策数据 ──
data = json.load(open(ROOT / "data/experiments/expert_router_oos_h10_matured.json", encoding="utf-8"))
WINDOWS = ("w1", "w2", "w3")
rows = []
for wid in WINDOWS:
    w = data["windows"][wid]
    for mode in ("full", "weighted", "subset"):
        for rec in w["router"][mode]["daily"]:
            rows.append({"window": wid, "mode": mode, **rec})
fo = data.get("final_oos")
if fo:
    for rec in fo["router"]["weighted"]["daily"]:
        rows.append({"window": "final", "mode": "weighted", **rec})
df = pd.DataFrame(rows)
for col in ("trend_net", "rebound_net", "future_vol_probability", "market_breadth", "oversold_degree"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["trend_net", "rebound_net", "future_vol_probability"]).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# 各日期特征
df["v60"] = df["date"].map(pos["vol60_pct250"])
df["v60_5"] = df["date"].map(pos["vol60_pct5"])
df["instab"] = df["date"].map(pos["instab10"])
df["tier"] = df["v60"].apply(lambda p: tier60(p))
df = df.dropna(subset=["v60", "instab"])

# ── 决策规则: 返回 (model_choice, pos) ──
def decide(rule, r, th_instab=None):
    p = r["future_vol_probability"]
    v, v5, inst = r["v60"], r["v60_5"], r["instab"]
    tier = r["tier"]
    if rule == "fixed_trend":
        return "trend", 1.0
    if rule == "fixed_rebound":
        return "rebound", 1.0
    if rule == "binary":
        return ("rebound" if p > 0.5 else "trend"), 1.0
    if rule == "model_3tier":
        if p > 0.59: return "rebound", 1.0
        if p < 0.36: return "trend", 1.0
        return "trend", 0.0
    if rule == "rule60_3tier":
        if tier == "high": return "rebound", 1.0
        if tier == "low": return "trend", 1.0
        return "trend", 0.0
    if rule == "rule60_confirm5":
        t5 = tier60(v5)
        if t5 == "high": return "rebound", 1.0
        if t5 == "low": return "trend", 1.0
        return "trend", 0.0
    if rule == "rule60_half":
        if tier == "high": return "rebound", 1.0
        if tier == "low": return "trend", 1.0
        return "trend", 0.5
    if rule == "rule60_instab_cash":
        if tier == "high": return "rebound", 1.0
        if tier == "low": return "trend", 1.0
        # 中波: 不稳定→空仓; 稳定→trend (死亡区=快速变化的中波)
        return ("trend", 0.0) if inst > th_instab else ("trend", 1.0)
    if rule == "rule60_instab_half":
        if tier == "high": return "rebound", 1.0
        if tier == "low": return "trend", 1.0
        return ("trend", 0.5) if inst > th_instab else ("trend", 1.0)
    raise ValueError(rule)

RULES = ["fixed_trend", "fixed_rebound", "binary", "model_3tier", "rule60_3tier",
         "rule60_confirm5", "rule60_half", "rule60_instab_cash", "rule60_instab_half"]

# ── 不稳定阈值: 只在 w1 上选 (Calmar 最大) ──
w1 = df[df["window"] == "w1"]
best_th = {}
for rule in ("rule60_instab_cash", "rule60_instab_half"):
    best = (None, -1e9)
    for th in (0.15, 0.25, 0.35, 0.45):
        rets = []
        for _, r in w1.iterrows():
            model, posn = decide(rule, r, th)
            net = r["rebound_net"] if model == "rebound" else r["trend_net"]
            rets.append(net * posn)
        rets = np.array(rets)
        tot = np.prod(1 + rets) - 1
        cum = np.cumprod(1 + rets)
        mdd = (cum / np.maximum.accumulate(cum) - 1).min()
        cal = tot / abs(mdd) if mdd < 0 else 0
        if cal > best[1]:
            best = (th, cal)
    best_th[rule] = best[0]
    print(f"[w1 选阈] {rule}: th={best[0]} (Calmar={best[1]:.2f})")

# ── 全样本评估 ──
def evaluate(rule, th=None):
    out = []
    for _, r in df.iterrows():
        model, posn = decide(rule, r, th)
        net = r["rebound_net"] if model == "rebound" else r["trend_net"]
        out.append({"date": r["date"], "window": r["window"], "ret": net * posn,
                    "dec": f"{model}:{posn}"})
    o = pd.DataFrame(out)
    rets = o["ret"].values
    cum = np.cumprod(1 + rets)
    mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    tot = cum[-1] - 1
    return {
        "rule": rule, "n": len(rets), "mean": rets.mean(), "total": tot,
        "win": (rets > 0).mean(), "mdd": mdd,
        "calmar": tot / abs(mdd) if mdd < 0 else np.nan,
        "turnover": (o["dec"].values[1:] != o["dec"].values[:-1]).mean(),
        "cash_frac": (o["dec"].str.contains(":0.0")).mean(),
        "half_frac": (o["dec"].str.contains(":0.5")).mean(),
        "rets": rets,
    }

from scipy import stats as st
results = {}
for rule in RULES:
    th = best_th.get(rule)
    results[rule] = evaluate(rule, th)
    print(f"eval {rule} th={th}")

bin_rets = results["binary"]["rets"]
print("\n=== 全样本 228 期对照 ===")
hdr = f"{'规则':<22}{'均值/期':>9}{'累计':>8}{'胜率':>7}{'回撤':>8}{'Calmar':>8}{'换手':>7}{'空仓':>6}{'半仓':>6}  vs binary"
print(hdr)
for rule in RULES:
    r = results[rule]
    t, p = st.ttest_rel(r["rets"], bin_rets)
    sig = f"t={t:+.2f} p={p:.3f}"
    print(f"{rule:<22}{r['mean']*100:+7.2f}%{r['total']*100:+7.1f}%{r['win']*100:6.1f}%{r['mdd']*100:7.1f}%{r['calmar']:8.2f}{r['turnover']*100:6.1f}%{r['cash_frac']*100:5.0f}%{r['half_frac']*100:5.0f}%  {sig}")

print("\n=== 分窗口均值/期 (w2+w3 = 独立验证集) ===")
print(f"{'规则':<22}{'w1':>9}{'w2':>9}{'w3':>9}{'final':>9}{'w2+w3':>9}")
for rule in RULES:
    r = results[rule]
    sub = df.copy(); sub["ret"] = r["rets"]
    g = sub.groupby("window")["ret"].mean() * 100
    v23 = sub[sub["window"].isin(["w2", "w3"])]["ret"].mean() * 100
    print(f"{rule:<22}{g.get('w1', float('nan')):+8.2f}%{g.get('w2', float('nan')):+8.2f}%{g.get('w3', float('nan')):+8.2f}%{g.get('final', float('nan')):+8.2f}%{v23:+8.2f}%")

# 汇总 JSON
import json as _json
summary = {rule: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "rets"}
           for rule, r in results.items()}
summary["best_th"] = best_th
Path(ROOT / "data/experiments/switch_improvement_experiment.json").write_text(
    _json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n结果已存: data/experiments/switch_improvement_experiment.json")
