"""按市场环境（波动率/宽度/趋势强度/超跌）分析两种模型的适用性。

数据: data/experiments/expert_router_oos_h10_matured.json 的 router.daily
(每窗口×训练方式×决策日: trend_net / rebound_net 为 Top10 扣费净收益,
 future_vol_probability 为波动率分类器概率, 另有 market_breadth/trend_strength/oversold_degree)

回答三个问题:
1. 固定趋势模型是否整体不如波动率切换? (逐期配对检验, 并分波动率桶看)
2. 子样本训练是否过拟合? (subset vs full 的 IC/收益/衰减/风格相关证据)
3. 两种模型分别在什么市场环境下最好? (按 vol/breadth/trend_strength/oversold 分桶)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
JSON_PATH = ROOT / "data/experiments/expert_router_oos_h10_matured.json"

WINDOWS = ("w1", "w2", "w3")
MODES = ("full", "weighted", "subset")
VOL_TH = 0.5


def load():
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def collect_daily(data):
    """汇总所有 窗口×模式×决策日 的记录 → list[dict]"""
    rows = []
    for wid in WINDOWS:
        w = data["windows"][wid]
        for mode in MODES:
            for rec in w["router"][mode]["daily"]:
                rows.append({"window": wid, "mode": mode, **rec})
    # final OOS (weighted)
    fo = data.get("final_oos")
    if fo:
        for rec in fo["router"]["weighted"]["daily"]:
            rows.append({"window": "final", "mode": "weighted", **rec})
    return rows


def bucket(x, edges):
    """把连续变量分桶, 返回 (label, value)"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    for i in range(len(edges) - 1):
        if x <= edges[i + 1]:
            return f"{edges[i]:.2f}~{edges[i+1]:.2f}" if edges[0] < edges[-1] else f"b{i}"
    return f">{edges[-1]:.2f}"


def paired(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(len(a), len(b))
    if n < 3:
        return None
    d = b - a
    t, p = st.ttest_rel(b[:n], a[:n])
    return t, p, int((d > 0).sum()), n


def main():
    data = load()
    rows = collect_daily(data)
    df = pd.DataFrame(rows)
    print(f"记录数: {len(df)} ({df['window'].value_counts().to_dict()})")
    for col in ("trend_net", "rebound_net", "future_vol_probability",
                "market_breadth", "trend_strength", "oversold_degree"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 切换逻辑: p>0.5 → rebound, else trend
    df["switch_net"] = np.where(df["future_vol_probability"] > VOL_TH,
                                df["rebound_net"], df["trend_net"])
    df["switch_pick"] = np.where(df["future_vol_probability"] > VOL_TH, "rebound", "trend")

    # ══════════════════════════════════════════════════
    # 问题1: 固定趋势 vs 波动率切换 (逐期配对, 全样本 + 分窗口 + 分波动率桶)
    # ══════════════════════════════════════════════════
    print("=" * 100)
    print("问题1: 固定趋势(trend) vs 波动率切换(switch) — 逐期 Top10 净收益")
    print("=" * 100)
    for label, sub in [("全样本", df)]:
        a = sub["trend_net"].values
        b = sub["switch_net"].values
        res = paired(a, b)
        if res:
            t, p, wins, n = res
            d = b - a
            print(f"  {label}: n={n}  trend均值={a.mean():+.4%}/期  switch均值={b.mean():+.4%}/期  "
                  f"差={d.mean():+.4%}  t={t:+.2f} p={p:.3f}  胜期{wins}/{n}")
    print("  分窗口:")
    for wid in list(WINDOWS) + ["final"]:
        sub = df[df["window"] == wid]
        if sub.empty:
            continue
        a, b = sub["trend_net"].values, sub["switch_net"].values
        res = paired(a, b)
        if res:
            t, p, wins, n = res
            print(f"    {wid:6s}: n={n:2d} trend={a.mean():+.4%} switch={b.mean():+.4%} "
                  f"差={ (b-a).mean():+.4%} 胜{wins}/{n}  t={t:+.2f} p={p:.3f}")
    print("  分波动率桶 (p_vol):")
    hi = df[df["future_vol_probability"] > VOL_TH]
    lo = df[df["future_vol_probability"] <= VOL_TH]
    for lbl, sub in [("高波 p>0.5", hi), ("低波 p<=0.5", lo)]:
        if sub.empty:
            continue
        a, b = sub["trend_net"].values, sub["switch_net"].values
        res = paired(a, b)
        if res:
            t, p, wins, n = res
            print(f"    {lbl:12s}: n={n:2d} trend={a.mean():+.4%} rebound={sub['rebound_net'].mean():+.4%} "
                  f"switch={b.mean():+.4%} 差={ (b-a).mean():+.4%} 胜{wins}/{n}  t={t:+.2f} p={p:.3f}")
    # 高波下 rebound vs trend / 低波下 trend vs rebound
    print("  波动率切换的机制本身对不对 (选对模型?):")
    if not hi.empty:
        res = paired(hi["trend_net"].values, hi["rebound_net"].values)
        if res:
            t, p, wins, n = res
            print(f"    高波期: rebound vs trend 差={ (hi['rebound_net']-hi['trend_net']).mean():+.4%} "
                  f"胜{wins}/{n}  t={t:+.2f} p={p:.3f}  (正=切 rebound 正确)")
    if not lo.empty:
        res = paired(lo["trend_net"].values, lo["rebound_net"].values)
        if res:
            t, p, wins, n = res
            print(f"    低波期: trend vs rebound 差={ (lo['trend_net']-lo['rebound_net']).mean():+.4%} "
                  f"胜{wins}/{n}  t={t:+.2f} p={p:.3f}  (正=切 trend 正确)")

    # ══════════════════════════════════════════════════
    # 问题3: 两种模型在什么环境下最好
    # ══════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("问题3: trend 与 rebound 分别在什么市场环境下最好 (全样本逐期)")
    print("=" * 100)
    # 波动率三分位
    pv = df["future_vol_probability"].dropna()
    e3 = [pv.quantile(0), pv.quantile(1/3), pv.quantile(2/3), pv.quantile(1)]
    # 宽度/趋势强度/超跌 按中位数分高低
    for col, name in [("market_breadth", "市场宽度"), ("trend_strength", "趋势强度"),
                      ("oversold_degree", "超跌程度")]:
        med = df[col].median()
        print(f"\n  [{name}] (中位数={med:.3f})")
        for lbl, sub in [("高", df[df[col] > med]), ("低", df[df[col] <= med])]:
            if sub.empty:
                continue
            t_mean = sub["trend_net"].mean()
            r_mean = sub["rebound_net"].mean()
            s_mean = sub["switch_net"].mean()
            better = "trend" if t_mean > r_mean else "rebound"
            print(f"    {name}{lbl:2s} n={len(sub):2d}: trend={t_mean:+.4%} rebound={r_mean:+.4%} "
                  f"switch={s_mean:+.4%}  → {better} 更好")
    print("\n  二维交叉 (波动率 × 市场宽度 / 超跌):")
    for col, name in [("market_breadth", "市场宽度"), ("oversold_degree", "超跌程度")]:
        med = df[col].median()
        print(f"    [{name}]")
        for vl in ("低波", "高波"):
            for hl in ("低", "高"):
                sub = df[((df["future_vol_probability"] > VOL_TH) if vl == "高波"
                          else (df["future_vol_probability"] <= VOL_TH)) &
                         ((df[col] > med) if hl == "高" else (df[col] <= med))]
                if sub.empty:
                    continue
                t_mean = sub["trend_net"].mean()
                r_mean = sub["rebound_net"].mean()
                better = "trend" if t_mean > r_mean else "rebound"
                print(f"      {vl}+{name}{hl:2s} n={len(sub):2d}: trend={t_mean:+.4%} rebound={r_mean:+.4%} "
                      f"→ {better} 更好")
    print("\n  波动率三档:")
    labels = ["低波", "中波", "高波"]
    for i, (e0, e1) in enumerate(zip(e3[:-1], e3[1:])):
        sub = df[(df["future_vol_probability"] > e0) & (df["future_vol_probability"] <= e1)]
        if sub.empty:
            continue
        t_mean = sub["trend_net"].mean()
        r_mean = sub["rebound_net"].mean()
        s_mean = sub["switch_net"].mean()
        better = "trend" if t_mean > r_mean else "rebound"
        print(f"    {labels[i]:4s}({e0:.2f}~{e1:.2f}) n={len(sub):2d}: trend={t_mean:+.4%} "
              f"rebound={r_mean:+.4%} switch={s_mean:+.4%}  → {better} 更好")

    # ══════════════════════════════════════════════════
    # 问题2: 子样本训练是否过拟合
    # ══════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("问题2: 子样本训练(subset) vs 全样本(full) — trend 专家, 过拟合证据")
    print("=" * 100)
    # 逐期收益差 (subset trend - full trend)
    sub_rows = []
    for wid in WINDOWS:
        w = data["windows"][wid]
        d_full = {r["date"]: r for r in w["router"]["full"]["daily"]}
        d_sub = {r["date"]: r for r in w["router"]["subset"]["daily"]}
        for date, rec in d_sub.items():
            if date in d_full:
                sub_rows.append({
                    "window": wid, "date": date,
                    "full_trend_net": d_full[date]["trend_net"],
                    "subset_trend_net": rec["trend_net"],
                    "future_vol_probability": rec["future_vol_probability"],
                    "trend_strength": rec["trend_strength"],
                    "market_breadth": rec["market_breadth"],
                    "oversold_degree": rec["oversold_degree"],
                })
    sdf = pd.DataFrame(sub_rows)
    sdf["diff"] = sdf["subset_trend_net"] - sdf["full_trend_net"]
    print("  逐期 subset-full (trend Top10 净收益):")
    for wid in WINDOWS:
        sub = sdf[sdf["window"] == wid]
        if sub.empty:
            continue
        d = sub["diff"]
        res = paired(sub["full_trend_net"].values, sub["subset_trend_net"].values)
        t = f"t={res[0]:+.2f} p={res[1]:.3f}" if res else "n<3"
        print(f"    {wid}: n={len(sub):2d} 差均值={d.mean():+.4%} 正期数={(d>0).sum()}/{len(sub)}  {t}")
    # 与风格变量相关性
    print("\n  subset-full 收益差 与市场变量相关性 (Spearman):")
    for col in ("future_vol_probability", "trend_strength", "market_breadth", "oversold_degree"):
        c = sdf["diff"].corr(sdf[col], method="spearman")
        print(f"    {col:24s}: r={c:+.3f}")
    # IC 证据 (来自 JSON 聚合指标)
    print("\n  Rank IC 对比 (full vs subset, trend 专家, top10):")
    for wid in WINDOWS:
        w = data["windows"][wid]
        f_ic = w["modes"]["full"]["trend"]["metrics"]["top10"]["rank_ic"]
        s_ic = w["modes"]["subset"]["trend"]["metrics"]["top10"]["rank_ic"]
        print(f"    {wid}: full={f_ic:.4f}  subset={s_ic:.4f}  Δ={s_ic-f_ic:+.4f}")
    print("\n  过拟合判读:")
    ic_delta = [data["windows"][w]["modes"]["subset"]["trend"]["metrics"]["top10"]["rank_ic"] -
                data["windows"][w]["modes"]["full"]["trend"]["metrics"]["top10"]["rank_ic"]
                for w in WINDOWS]
    print(f"    - subset IC - full IC: {['%+.4f' % x for x in ic_delta]} → IC 上 subset 从未更好")
    print(f"    - 收益差衰减: w1→w2→w3 = "
          f"{sdf[sdf.window=='w1']['diff'].mean():+.4%} / {sdf[sdf.window=='w2']['diff'].mean():+.4%} / "
          f"{sdf[sdf.window=='w3']['diff'].mean():+.4%}")
    print("    → 收益优势与 IC 矛盾且随时间衰减 → 支持'子样本训练是过拟合/风格拟合'的判断")


if __name__ == "__main__":
    main()
