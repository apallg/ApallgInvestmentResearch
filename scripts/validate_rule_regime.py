"""
规则 6 状态 → 模型 IC 区分度检验
"""
import sys
sys.path.insert(0, ".")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
from core.shared import config
from core.data import Database

LABEL_HORIZON = 10

RUN_MAP = {
    "41270c3636cc45d8906f771b8d241168": "trend (19-25)",
    "d69ef77d9339495a974365f947af1728": "trend (22-25)",
    "055472a7f75d4535a106e6e87bd110c5": "oversold (19-25)",
    "ffcd48579a3544b7b7ad7b58feaa89ad": "oversold (22-25)",
    "50f3a7dbc2d84fea8ff02214d4ca2d10": "volume (19-25)",
    "6c522f68c7974bc38a2594a07ad08b23": "volume (22-25)",
}

# ── 1. 加载预测 ──
mlruns = Path(config.QLIB_MLFLOW_URI.replace("file:", ""))
preds = {}
for p in mlruns.glob("*/**/artifacts/pred.pkl"):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if not hasattr(obj, "index"):
        continue
    idx = obj.index
    dmin = str(idx.get_level_values(0).min())[:10]
    if not dmin.startswith("2026-03"):
        continue
    run_id = p.parent.parent.name
    if run_id not in RUN_MAP:
        continue
    df = obj.reset_index()
    df.columns = ["trade_date", "instrument", "score"]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["code"] = df["instrument"].str.replace("^sh|^sz", "", regex=True)
    preds[run_id] = df
print(f"加载 {len(preds)} 个模型预测")

# ── 2. 算规则 regime ──
# 用 32 只大盘股做代理指数
CODES = ['600519','601318','600036','000858','600900','601166','600276','600887',
         '601398','000333','600030','601328','600809','000651','002415','600585',
         '600104','002714','601288','601988','600031','000725','600690','601857',
         '300750','002594','601899','600048','601668','000568','000002','600309']

code_list = ",".join([f"'{c}'" for c in CODES])
all_dates = sorted(set().union(*[set(df["trade_date"]) for df in preds.values()]))
d_first = (min(all_dates) - pd.Timedelta(days=300)).strftime("%Y-%m-%d")  # 多取300天算分位数
d_last = max(all_dates).strftime("%Y-%m-%d")

rows = Database.fetchall(
    f"SELECT code, trade_date, close FROM daily_quotes "
    f"WHERE code IN ({code_list}) AND trade_date >= %s AND trade_date <= %s "
    f"ORDER BY code, trade_date",
    (d_first, d_last)
)
price_df = pd.DataFrame(rows)
price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
price_df["close"] = price_df["close"].astype(float)
price_df = price_df.sort_values(["code", "trade_date"])
price_df["ret"] = price_df.groupby("code")["close"].pct_change()

idx = price_df.groupby("trade_date")["ret"].mean().reset_index(name="mkt_ret")
idx = idx.sort_values("trade_date")
idx["ret_20d"] = idx["mkt_ret"].rolling(20).sum()
idx["vol_20d"] = idx["mkt_ret"].rolling(20).std() * np.sqrt(252)
idx["vol_pct"] = idx["vol_20d"].rolling(252).rank(pct=True)

# 规则分类
def classify(row):
    ret = row.get("ret_20d")
    pct = row.get("vol_pct")
    if pd.isna(ret) or pd.isna(pct):
        return None
    vs = "volatile" if pct > 0.7 else "steady"
    if ret > 0.02:
        ds = "up"
    elif ret < -0.02:
        ds = "down"
    else:
        ds = "flat"
    return f"{vs}_{ds}"

idx["regime"] = idx.apply(classify, axis=1)
rule_labels = dict(zip(idx["trade_date"].dt.strftime("%Y-%m-%d"), idx["regime"]))
print(f"规则标注: {len(rule_labels)} 天")

# ── 3. Forward returns ──
def get_fwd_returns(dates, codes):
    d_min = min(dates).strftime("%Y-%m-%d")
    d_max = (max(dates) + pd.Timedelta(days=LABEL_HORIZON + 5)).strftime("%Y-%m-%d")
    all_rows = []
    for i in range(0, len(codes), 500):
        batch = codes[i:i+500]
        ph = ",".join(["%s"]*len(batch))
        r = Database.fetchall(
            f"SELECT code, trade_date, close FROM daily_quotes "
            f"WHERE trade_date>=%s AND trade_date<=%s AND code IN ({ph})",
            (d_min, d_max, *batch))
        all_rows.extend(r)
    pdf = pd.DataFrame(all_rows)
    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"])
    pdf["code"] = pdf["code"].str.replace("^sh|^sz", "", regex=True)
    pdf = pdf.sort_values(["code","trade_date"])
    pdf["fwd_close"] = pdf.groupby("code")["close"].shift(-LABEL_HORIZON)
    pdf["fwd_ret"] = pdf["fwd_close"] / pdf["close"] - 1
    return pdf[["code","trade_date","fwd_ret"]].dropna()

all_codes = sorted(set().union(*[set(df["code"].unique()) for df in preds.values()]))
fwd = get_fwd_returns(all_dates, all_codes)
print(f"Forward returns: {fwd['trade_date'].nunique()} 天")

# ── 4. 逐模型 × regime IC ──
print()
print("=" * 75)
print(f"{'模型':25s} {'总体IC':>8s} {'显著regime数':>10s} {'最佳regime':>18s} {'最差regime':>18s}")
print("-" * 75)

SIX = ["volatile_up","volatile_down","volatile_flat","steady_up","steady_down","steady_flat"]
all_model_ics = {}  # model -> {regime: [daily_ics]}

for run_id, pred_df in sorted(preds.items()):
    strategy = RUN_MAP[run_id]
    m = pred_df.merge(fwd, on=["code","trade_date"], how="inner")
    m["regime"] = m["trade_date"].apply(lambda d: rule_labels.get(d.strftime("%Y-%m-%d")))

    def daily_ic(g):
        if len(g) < 10: return np.nan
        return g["score"].corr(g["fwd_ret"])

    daily = m.groupby(["trade_date","regime"]).apply(daily_ic, include_groups=False).reset_index(name="IC")
    daily = daily.dropna(subset=["IC"])

    overall = daily.groupby("trade_date")["IC"].mean()
    mean_ic = overall.mean()

    regime_ics = {}
    for r in SIX:
        vals = daily[daily["regime"] == r]["IC"].values
        if len(vals) >= 5:
            regime_ics[r] = vals

    # Kruskal-Wallis
    groups = list(regime_ics.values())
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
    else:
        h, p = 0, 1

    # 最佳/最差
    regime_means = {r: np.mean(v) for r, v in regime_ics.items()}
    if regime_means:
        best_r = max(regime_means, key=regime_means.get)
        worst_r = min(regime_means, key=regime_means.get)
    else:
        best_r = worst_r = "-"

    sig_mark = " **" if p < 0.05 else (" *" if p < 0.10 else "")
    print(f"{strategy:25s} {mean_ic:+8.4f} {len(groups):4d}reg{p:6.0%}{sig_mark}  "
          f"{best_r:18s} {worst_r:18s}")

    all_model_ics[strategy] = regime_ics

# ── 5. 详细: 每个模型 × regime ──
print()
print("=" * 75)
print("逐模型 × Regime 详细 IC")
print("=" * 75)

for strategy, regime_ics in sorted(all_model_ics.items()):
    print(f"\n{strategy}:")
    print(f"  {'Regime':20s} {'N':>4s} {'Mean IC':>8s} {'Std':>7s} {'Win%':>6s}")
    print(f"  {'-'*45}")
    for r in SIX:
        if r in regime_ics:
            ic = regime_ics[r]
            print(f"  {r:20s} {len(ic):4d} {np.mean(ic):+8.4f} {np.std(ic):7.4f} "
                  f"{(ic>0).mean():6.0%}")

# ── 6. 每个 regime 下最佳策略 ──
print()
print("=" * 75)
print("各 Regime 最佳策略")
print("=" * 75)

for regime in SIX:
    entries = []
    for strategy, regime_ics in all_model_ics.items():
        if regime in regime_ics:
            entries.append((strategy, np.mean(regime_ics[regime]), len(regime_ics[regime])))
    if entries:
        entries.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{regime}:")
        for i, (s, ic, n) in enumerate(entries[:3]):
            print(f"  {i+1}. {s:25s} IC={ic:+.4f} (n={n}d)")

# ── 7. 和 V1 对比 ──
print()
print("=" * 75)
print("规则 vs V1: regime 区分度对比")
print("=" * 75)

# 统计有几个模型在不同 regime 间显著
rule_sig = sum(1 for _, ric in all_model_ics.items()
               if len(ric) >= 2 and stats.kruskal(*list(ric.values()))[1] < 0.10)
v1_sig = 1  # 之前的结果,只有1个显著

print(f"  规则: {rule_sig}/6 模型 regime 间 IC 显著不同 (p<0.10)")
print(f"  V1:   {v1_sig}/6 模型 regime 间 IC 显著不同 (p<0.10)")

# 效应量对比
rule_etas = []
for _, ric in all_model_ics.items():
    means = [np.mean(v) for v in ric.values()]
    if len(means) >= 2:
        ss = sum((m - np.mean(means))**2 for m in means)
        rule_etas.append(ss / (ss + 1))

v1_etas = [0.000, 0.002, 0.001, 0.002, 0.000, 0.000]  # 之前的eta2值
print(f"  规则 eta2: {np.mean(rule_etas):.3f} (均值)")
print(f"  V1   eta2: {np.mean(v1_etas):.3f} (均值)")

print()
if rule_sig > v1_sig or np.mean(rule_etas) > np.mean(v1_etas) * 2:
    print(">>> 规则 regime 的区分度优于 V1")
else:
    print(">>> 规则 regime 的区分度没有明显改善")
    print("    '市场状态'这个维度本身可能无法解释策略表现的波动")
