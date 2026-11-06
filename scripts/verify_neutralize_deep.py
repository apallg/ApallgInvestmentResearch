"""深度验证: 行业中性化 (市值+行业 OLS 残差) 在选股/回测中是否有效。

相比 verify_neutralize_macro.py 的加强点:
1. 多窗口: w2 (2024-07-01~2025-06-30) + w3 (2025-07-01~2026-06-30) 全部缓存预测 (9 个模型文件)
2. 多变体: pct(关) / neu_mc(仅市值) / neu_ind(仅行业) / neu(市值+行业) / raw
3. 多指标: 截面 IC/ICIR、Top5/10/20 净收益(换手率调整成本)、多空十分位、命中率
4. 机制诊断: 被剥离的风格分量 (raw-neu) 的 IC、top10 的市值/行业暴露、与 raw top10 重合度
5. 统计: 配对 t、符号检验、paired bootstrap 95% CI
"""
import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.domain.backtest.common import norm_code
from core.domain.factors.forward_returns import get_forward_cache
from core.domain.factors.neutralize import cross_sectional_neutralize
from core.domain.factors.market_cap import get_market_cap_map
from core.domain.signals.meta import MetaSignalFusion

WINDOWS = {
    "w2": ("2024-07-01", "2025-06-30"),
    "w3": ("2025-07-01", "2026-06-30"),
}
BUY_COST, SELL_COST = 0.0004, 0.0009
VARIANT_ORDER = ("raw", "pct", "neu_mc", "neu_ind", "neu")


def load_predictions():
    files = sorted(glob.glob(str(ROOT / "data/qlib_data/prediction_cache/lgb_lstm/*.pkl")))
    out = []
    for f in files:
        try:
            c = pickle.load(open(f, "rb"))
            p = c["prediction"]
        except Exception:
            continue
        d0 = str(p.index.get_level_values(0).min())[:10]
        d1 = str(p.index.get_level_values(0).max())[:10]
        for wid, (s0, s1) in WINDOWS.items():
            if d0 <= s0 and d1 >= s1:
                out.append({"file": Path(f).name, "window": wid, "pred": p})
                break
    return out


def trading_days(start, end):
    from core.data import Database
    rows = Database.fetchall(
        "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date",
        [start, end])
    return [r["trade_date"] for r in rows]


def rank_ic(x, y):
    if len(x) < 30:
        return None
    xr = pd.Series(x).rank()
    yr = pd.Series(y).rank()
    c = xr.corr(yr)
    return float(c) if c is not None and np.isfinite(c) else None


def topn_return(scores, fwd_map, top_n):
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rets = [fwd_map.get(c) for c, _ in order]
    rets = [r for r in rets if r is not None]
    return float(np.mean(rets)) if rets else None


def decile_ls(scores, fwd_map):
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(order)
    if n < 40:
        return None
    k = max(10, n // 10)
    top = [fwd_map.get(c) for c, _ in order[:k]]
    bot = [fwd_map.get(c) for c, _ in order[-k:]]
    top = [r for r in top if r is not None]
    bot = [r for r in bot if r is not None]
    if not top or not bot:
        return None
    return float(np.mean(top) - np.mean(bot))


def top10_exposure(scores, mc_map, ind_map, all_codes):
    """top10 的市值暴露(对数市值相对全截面 z) 与 行业数"""
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
    codes = [c for c, _ in top]
    mv_all = np.array([float(mc_map.get(c, 0) or 0) for c in all_codes])
    ok = mv_all > 0
    if ok.sum() < 30:
        return None, None, None
    lv_all = np.log(np.where(ok, mv_all, np.nan))
    med = np.nanmedian(lv_all)
    lv_all = np.where(np.isnan(lv_all), med, lv_all)
    mu, sd = lv_all.mean(), lv_all.std() + 1e-9
    mvs = [float(mc_map.get(c, 0) or 0) for c in codes]
    lv = np.log([m for m in mvs if m > 0]) if any(m > 0 for m in mvs) else []
    z = float(np.mean((np.array(lv) - mu) / sd)) if len(lv) >= 5 else None
    inds = []
    for c in codes:
        v = ind_map.get(c)
        inds.append(v.get("industry", "") if isinstance(v, dict) else "")
    n_ind = len({i for i in inds if i})
    return z, n_ind, codes


def per_file_analysis(entry, industry_map, fc, max_dates):
    p = entry["pred"]
    wid = entry["window"]
    s0, s1 = WINDOWS[wid]
    days = trading_days(s0, s1)
    rebal = days[::10][:max_dates]
    fwd_cache = {}
    mc_cache = {}
    stats = {v: {"ic": [], "t5": [], "t10": [], "t20": [], "t10_net": [], "t20_net": [],
                  "ls": [], "hit10": [], "turn": []}
             for v in VARIANT_ORDER}
    exps = {"z_lmv": {v: [] for v in VARIANT_ORDER}, "n_ind": {v: [] for v in VARIANT_ORDER},
            "overlap10": []}
    ic_style = []
    corr_rn = []

    for d in rebal:
        d_str = str(d)[:10]
        try:
            day = p.xs(d_str, level="datetime")
        except Exception:
            continue
        scores = day.dropna()
        num = {norm_code(k): float(v) for k, v in scores.items()}
        codes = [c for c in num if len(c) == 6]
        if len(codes) < 100:
            continue
        if d_str not in fwd_cache:
            fwd_cache[d_str] = fc.get_batch(codes, pd.Timestamp(d_str), 10)
        fwd_map = fwd_cache[d_str]
        y = [fwd_map.get(c) for c in codes]
        valid = [i for i, r in enumerate(y) if r is not None and np.isfinite(r)]
        if len(valid) < 100:
            continue
        codes_v = [codes[i] for i in valid]
        raw_v = [num[c] for c in codes_v]
        y_v = [y[i] for i in valid]
        raw_dict = {c: num[c] for c in codes_v}
        y_dict = {c: y_v[i] for i, c in enumerate(codes_v)}

        # 市值映射 (按日期缓存)
        if d_str not in mc_cache:
            mc_cache[d_str] = get_market_cap_map(codes_v, pd.Timestamp(d_str))
        mc_map = mc_cache[d_str]

        # ── 变体 ──
        variants = {}
        # raw
        variants["raw"] = raw_dict
        # pct: 仅分位
        pct = {c: MetaSignalFusion.qlib_to_score(v, raw_v) for c, v in raw_dict.items()}
        variants["pct"] = pct
        # 中性化变体 (先中性化再分位, 与管线一致)
        neu_mc = cross_sectional_neutralize(codes_v, raw_v, mc_map, {})
        variants["neu_mc"] = {c: MetaSignalFusion.qlib_to_score(v, list(neu_mc.values()))
                              for c, v in neu_mc.items()}
        neu_ind = cross_sectional_neutralize(codes_v, raw_v, {}, industry_map)
        variants["neu_ind"] = {c: MetaSignalFusion.qlib_to_score(v, list(neu_ind.values()))
                               for c, v in neu_ind.items()}
        neu = cross_sectional_neutralize(codes_v, raw_v, mc_map, industry_map)
        variants["neu"] = {c: MetaSignalFusion.qlib_to_score(v, list(neu.values()))
                           for c, v in neu.items()}

        for v in VARIANT_ORDER:
            s = variants[v]
            sv = [s[c] for c in codes_v]
            ic = rank_ic(sv, y_v)
            if ic is not None:
                stats[v]["ic"].append(ic)
            for key, n in (("t5", 5), ("t10", 10), ("t20", 20)):
                r_ = topn_return(s, y_dict, n)
                if r_ is not None:
                    stats[v][key].append(r_)
            ls = decile_ls(s, y_dict)
            if ls is not None:
                stats[v]["ls"].append(ls)
            top10 = [c for c, _ in sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:10]]
            hits = [1 for c in top10 if (y_dict.get(c) or 0) > 0]
            stats[v]["hit10"].append(len(hits) / 10)
            # 换手 (与上一期 top10 重合度) → 净收益 (按实际换手扣费)
            prev = getattr(per_file_analysis, "_prev_" + v + "_" + wid, None)
            if prev is not None:
                turn = 1 - len(set(prev) & set(top10)) / 10
                stats[v]["turn"].append(turn)
                cost = turn * (BUY_COST + SELL_COST)
                r10_ = stats[v]["t10"][-1]
                r20_ = stats[v]["t20"][-1]
                if r10_ is not None:
                    stats[v]["t10_net"].append(r10_ - cost)
                if r20_ is not None:
                    stats[v]["t20_net"].append(r20_ - cost)
            setattr(per_file_analysis, "_prev_" + v + "_" + wid, set(top10))
            # 风格暴露 (top10 对数市值 z、行业数)
            z_lmv, ni, _ = top10_exposure(s, mc_map, industry_map, codes_v)
            if z_lmv is not None:
                exps["z_lmv"][v].append(z_lmv)
            exps["n_ind"][v].append(ni)
        raw_top10 = {c for c, _ in sorted(variants["raw"].items(), key=lambda kv: kv[1], reverse=True)[:10]}
        neu_top10 = {c for c, _ in sorted(variants["neu"].items(), key=lambda kv: kv[1], reverse=True)[:10]}
        exps["overlap10"].append(len(raw_top10 & neu_top10))

        # 机制: 被剥离分量 (raw - neu_residual, 用残差前的拟合值) 的 IC
        resid = cross_sectional_neutralize(codes_v, raw_v, mc_map, industry_map)
        fit = {c: raw_dict[c] - resid.get(c, raw_dict[c]) for c in codes_v}
        fit_v = [fit[c] for c in codes_v]
        ic_fit = rank_ic(fit_v, y_v)
        if ic_fit is not None:
            ic_style.append(ic_fit)
        # raw 与 neu 分数相关性
        rn = pd.Series(raw_v).corr(pd.Series([variants["neu"][c] for c in codes_v]))
        if rn is not None and np.isfinite(rn):
            corr_rn.append(float(rn))

    return stats, exps, ic_style, corr_rn, len(rebal)


def paired_test(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(len(a), len(b))
    if n < 5:
        return None
    a, b = a[:n], b[:n]
    d = b - a
    from scipy import stats as st
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n) + 1e-12)
    p = 2 * (1 - st.t.cdf(abs(t), df=n - 1))
    return t, p, int((d > 0).sum()), n


def bootstrap_ci(a, b, n_boot=2000, seed=7):
    a = np.asarray(a, float)[: min(len(a), len(b))]
    b = np.asarray(b, float)[: len(a)]
    d = b - a
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        means[i] = d[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dates", type=int, default=25)
    args = ap.parse_args()

    from core.data.fetcher import DataFetcher
    industry_map = DataFetcher().get_industry_map() or {}
    print("行业映射:", len(industry_map), "只", flush=True)

    entries = load_predictions()
    print("覆盖窗口的预测文件:", len(entries))
    for e in entries:
        print("  ", e["window"], e["file"][:12], flush=True)

    fc = get_forward_cache()
    all_res = []  # (wid, file, n_periods, stats, exps, ic_style, corr_rn)
    for entry in entries:
        stats, exps, ic_style, corr_rn, n_per = per_file_analysis(entry, industry_map, fc, args.max_dates)
        all_res.append((entry["window"], entry["file"], n_per, stats, exps, ic_style, corr_rn))

    # ── 汇总输出 ──
    print()
    print("=" * 110)
    print("深度验证: 行业中性化 (市值+行业) — 多窗口多模型汇总")
    print("=" * 110)
    agg = {v: {"ic": [], "t5": [], "t10": [], "t20": [], "t10_net": [], "t20_net": [],
                "ls": [], "hit10": []} for v in VARIANT_ORDER}
    for wid, f, n_per, stats, exps, ic_style, corr_rn in all_res:
        print(f"\n[{wid}] {f[:12]}  ({n_per} 期)")
        print(f"  {'变体':8s} {'meanIC':>8s} {'ICIR':>7s} {'Top5':>9s} {'Top10':>9s} {'Top20':>9s} "
              f"{'多空':>9s} {'命中率':>7s} {'top10市值z':>9s} {'行业数':>6s} {'raw重合':>7s}")
        overlap = exps["overlap10"]
        for v in VARIANT_ORDER:
            s = stats[v]
            ic = np.array(s["ic"])
            icir = ic.mean() / (ic.std() + 1e-9) if len(ic) else 0
            m5 = np.mean(s["t5"]) if s["t5"] else float("nan")
            m10 = np.mean(s["t10"]) if s["t10"] else float("nan")
            m20 = np.mean(s["t20"]) if s["t20"] else float("nan")
            mls = np.mean(s["ls"]) if s["ls"] else float("nan")
            hit = np.mean(s["hit10"]) if s["hit10"] else float("nan")
            z_lmv = np.mean(exps["z_lmv"][v]) if exps["z_lmv"][v] else float("nan")
            n_ind = np.mean(exps["n_ind"][v]) if exps["n_ind"][v] else float("nan")
            ov = np.mean(overlap) if overlap else float("nan")
            print(f"  {v:8s} {ic.mean():>8.4f} {icir:>7.2f} {m5:>9.2%} {m10:>9.2%} {m20:>9.2%} "
                  f"{mls:>9.2%} {hit:>7.1%} {z_lmv:>9.2f} {n_ind:>6.1f} {ov:>7.1f}")
        for v in VARIANT_ORDER:
            for k in ("ic", "t5", "t10", "t20", "t10_net", "t20_net", "ls", "hit10"):
                agg[v][k].extend(stats[v][k])
        print(f"  被剥离风格分量IC: mean={np.mean(ic_style):+.4f} (n={len(ic_style)})   "
              f"raw×neu相关: {np.mean(corr_rn):.3f}")

    # 汇总统计: pct vs neu
    print()
    print("-" * 110)
    print("汇总 (全部窗口×模型×期数 pooled):")
    for k in ("ic", "t5", "t10", "t10_net", "t20", "t20_net", "ls"):
        a = np.array(agg["pct"][k])
        b = np.array(agg["neu"][k])
        if len(a) and len(b):
            res = paired_test(a, b)
            ci = bootstrap_ci(a, b)
            txt = f"  {k:8s}: pct={a.mean():+.4f} neu={b.mean():+.4f} Δ={b.mean()-a.mean():+.4f}"
            if res:
                t, p, wins, n = res
                txt += f"  t={t:+.2f} p={p:.3f} 胜期{wins}/{n}  bootCI[{ci[0]:+.4f},{ci[1]:+.4f}]"
            print(txt)

    # ── 按文件粒度的显著性 (每个文件独立配对检验, 避免 pooled 伪重复) ──
    print()
    print("-" * 110)
    print("按文件粒度: 每个文件 25 期的独立配对检验 (pct vs neu)")
    print(f"  {'窗口':4s} {'文件':12s} {'ΔIC':>8s} {'ΔTop5':>9s} {'ΔTop10':>9s} {'ΔTop20':>9s} {'Top10 t':>7s} {'p':>6s}")
    file_delta_t10 = []
    for wid, f, n_per, stats, exps, ic_style, corr_rn in all_res:
        a_ic = np.array(stats["pct"]["ic"]); b_ic = np.array(stats["neu"]["ic"])
        a5 = np.array(stats["pct"]["t5"]); b5 = np.array(stats["neu"]["t5"])
        a10 = np.array(stats["pct"]["t10"]); b10 = np.array(stats["neu"]["t10"])
        a20 = np.array(stats["pct"]["t20"]); b20 = np.array(stats["neu"]["t20"])
        res = paired_test(a10, b10)
        t = res[0] if res else float("nan")
        p = res[1] if res else float("nan")
        file_delta_t10.append(b10.mean() - a10.mean())
        print(f"  {wid:4s} {f[:12]:12s} {b_ic.mean()-a_ic.mean():>+8.4f} {b5.mean()-a5.mean():>+9.2%} "
              f"{b10.mean()-a10.mean():>+9.2%} {b20.mean()-a20.mean():>+9.2%} {t:>7.2f} {p:>6.3f}")
    fd = np.array(file_delta_t10)
    if len(fd) >= 3:
        from scipy import stats as st2
        t_meta, p_meta = st2.ttest_1samp(fd, 0)
        print(f"\n  ΔTop10 跨文件单样本 t 检验: 均值={fd.mean():+.4%}/期, t={t_meta:+.2f}, p={p_meta:.4f}, "
              f"正文件数 {int((fd > 0).sum())}/{len(fd)}")

    print()
    print("判读:")
    a = np.array(agg["pct"]["t10"]); b = np.array(agg["neu"]["t10"])
    d10 = b.mean() - a.mean() if len(a) and len(b) else float("nan")
    a5 = np.array(agg["pct"]["t5"]); b5 = np.array(agg["neu"]["t5"])
    d5 = b5.mean() - a5.mean() if len(a5) and len(b5) else float("nan")
    aic = np.array(agg["pct"]["ic"]); bic = np.array(agg["neu"]["ic"])
    dic = bic.mean() - aic.mean() if len(aic) and len(bic) else float("nan")
    a_net = np.array(agg["pct"]["t10_net"]); b_net = np.array(agg["neu"]["t10_net"])
    d10_net = b_net.mean() - a_net.mean() if len(a_net) and len(b_net) else float("nan")
    a20 = np.array(agg["pct"]["t20"]); b20 = np.array(agg["neu"]["t20"])
    a20n = np.array(agg["pct"]["t20_net"]); b20n = np.array(agg["neu"]["t20_net"])
    d20 = b20.mean() - a20.mean() if len(a20) and len(b20) else float("nan")
    d20n = b20n.mean() - a20n.mean() if len(a20n) and len(b20n) else float("nan")
    print(f"  中性化 vs 仅分位: ΔIC={dic:+.4f}, ΔTop5毛={d5:+.4%}, ΔTop10毛={d10:+.4%}, "
          f"ΔTop10净={d10_net:+.4%}, ΔTop20毛={d20:+.4%}, ΔTop20净={d20n:+.4%}")
    if len(a) and len(b):
        res = paired_test(a, b)
        if res and res[1] < 0.05:
            print("  → Top10 毛收益配对检验显著 (p<0.05)")
        else:
            print("  → Top10 毛收益配对检验未达显著")
    if len(a_net) and len(b_net):
        res = paired_test(a_net, b_net)
        if res:
            print(f"  → Top10 净收益(含换手费)配对检验: Δ={d10_net:+.4%}/期, t={res[0]:+.2f}, p={res[1]:.3f}")
    if len(a20n) and len(b20n):
        res = paired_test(a20n, b20n)
        if res:
            print(f"  → Top20 净收益配对检验: Δ={d20n:+.4%}/期, t={res[0]:+.2f}, p={res[1]:.3f}")
    if len(aic) and len(bic) and dic < 0:
        print("  → IC 口径仍为负贡献 (与组合口径分离), 建议以组合指标为准")


if __name__ == "__main__":
    main()
