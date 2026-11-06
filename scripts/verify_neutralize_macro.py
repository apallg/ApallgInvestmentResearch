"""验证选股/回测两个分数处理功能是否有效：

1. 行业中性化 (normalize_qlib_scores neutralize=True, 市值+行业 OLS 残差)
2. 宏观行业修正 (apply_macro_sector_bias, 对宏观利好行业 +0.15)

方法: 用 OOS 实验 w3 窗口 (2025-07-01~2026-06-30) 的缓存预测
(data/qlib_data/prediction_cache/lgb_lstm/<file>.pkl, 与实验 full/rebound rank_ic 0.0673 匹配),
按 10 交易日调仓, 对比 6 个变体在全截面的 Rank IC 与 Top-N 等权收益。

变体:
  raw            — 模型原始分
  pct            — 仅分位映射 [-1,1] (中性化关)
  neu            — 中性化+分位映射 (行业中性化开)
  pct+macro      — 分位映射 + 宏观行业修正 (中性化关)
  neu+macro      — 中性化 + 分位映射 + 宏观行业修正 (回测默认路径)

输出: 各变体 mean IC / ICIR / IC>0, Top10/Top20 净收益, 以及覆盖诊断
(市值/行业覆盖率、宏观信号数) — 用于判断"失效"是机制问题还是数据缺失。
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
from core.domain.screening._screener import normalize_qlib_scores
from core.domain.screening.macro import apply_macro_sector_bias

START, END = "2025-07-01", "2026-06-30"
BUY_COST, SELL_COST = 0.0004, 0.0009  # 买0.04% 卖0.09% (实验口径)


def load_prediction(path):
    with open(path, "rb") as f:
        p = pickle.load(f)["prediction"]
    d0, d1 = str(p.index.get_level_values(0).min())[:10], str(p.index.get_level_values(0).max())[:10]
    if not (d0 <= START and d1 >= END):
        raise SystemExit(f"{path}: 日期范围 {d0}~{d1} 不覆盖 {START}~{END}")
    return p


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


def paired_test(a, b):
    """配对 t 检验 + 符号检验 (a vs b 的逐期差)"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 5:
        return None
    a, b = a[:n], b[:n]
    d = b - a
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n) + 1e-12)
    from scipy import stats as st
    p = 2 * (1 - st.t.cdf(abs(t), df=n - 1))
    wins = int((d > 0).sum())
    return t, p, wins, n


def summarize_ics(name, ics):
    a = np.array([x for x in ics if x is not None])
    if len(a) == 0:
        return f"{name:10s} 无有效截面"
    return (f"{name:10s} n={len(a):2d}  meanIC={a.mean():+.4f}  ICIR={a.mean()/(a.std()+1e-9):+.3f}  "
            f"IC>0={np.mean(a > 0):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="预测缓存 pkl (缺省自动选与 w3 full/rebound IC 匹配的文件)")
    ap.add_argument("--dates", type=int, default=25, help="使用的调仓日数 (每10交易日, 最多25)")
    args = ap.parse_args()

    if args.file:
        path = args.file
    else:
        files = sorted(glob.glob(str(ROOT / "data/qlib_data/prediction_cache/lgb_lstm/*.pkl")))
        # 选与实验 full/rebound rank_ic 0.0673 最接近的 w3 文件
        best, best_diff = None, 9
        for f in files:
            try:
                c = pickle.load(open(f, "rb"))
                p = c["prediction"]
            except Exception:
                continue
            d0, d1 = str(p.index.get_level_values(0).min())[:10], str(p.index.get_level_values(0).max())[:10]
            if d0 <= START and d1 >= END:
                diff = abs(len(p) - 719950)  # w3 实验预测约 719950 行
                if diff < best_diff:
                    best_diff = diff
                    best = f
        if not best:
            raise SystemExit("未找到覆盖 w3 窗口的预测缓存")
        path = best
    print("预测文件:", Path(path).name, flush=True)
    pred = load_prediction(path)

    fc = get_forward_cache()
    days = trading_days(START, END)
    rebal = days[::10][:args.dates]
    print(f"调仓日: {len(rebal)} ({rebal[0]} ~ {rebal[-1]})", flush=True)

    # ── 数据准备 ──
    variants = {"raw": [], "pct": [], "neu": [], "pct+macro": [], "neu+macro": []}
    top10 = {k: [] for k in variants}
    top20 = {k: [] for k in variants}
    cov = {"mktcap": [], "industry": [], "macro_signal_dates": set(), "macro_bonus_codes": 0}
    industry_map = None
    try:
        from core.data.fetcher import DataFetcher
        industry_map = DataFetcher().get_industry_map() or {}
        print(f"行业映射: {len(industry_map)} 只", flush=True)
    except Exception as e:
        print(f"[WARN] 行业映射加载失败: {e}", flush=True)

    for i, d in enumerate(rebal):
        d_str = str(d)[:10]
        try:
            day = pred.xs(d_str, level="datetime")
        except Exception:
            print(f"[skip] {d_str} 无预测")
            continue
        scores = day.iloc[:, 0] if hasattr(day, "columns") else day
        scores = scores.dropna()
        num = {norm_code(k): float(v) for k, v in scores.items()}
        codes = [c for c in num if len(c) == 6]
        if len(codes) < 100:
            continue
        fwd_map = fc.get_batch(codes, pd.Timestamp(d_str), 10)
        y = [fwd_map.get(c) for c in codes]
        valid = [i0 for i0, r in enumerate(y) if r is not None and np.isfinite(r)]
        if len(valid) < 100:
            continue
        codes_v = [codes[i0] for i0 in valid]
        raw_v = [num[c] for c in codes_v]
        y_v = [y[i0] for i0 in valid]
        raw_dict = {c: num[c] for c in codes_v}

        # 1) raw
        variants["raw"].append(rank_ic(raw_v, y_v))
        r10 = topn_return(dict(zip(codes_v, raw_v)), dict(zip(codes_v, y_v)), 10)
        r20 = topn_return(dict(zip(codes_v, raw_v)), dict(zip(codes_v, y_v)), 20)
        if r10 is not None: top10["raw"].append(r10)
        if r20 is not None: top20["raw"].append(r20)

        # 2) 分位映射 (中性化关)
        pct_dict = normalize_qlib_scores(raw_dict, pd.Timestamp(d_str), industry_map, neutralize=False)
        pct_v = [pct_dict[c] for c in codes_v]
        variants["pct"].append(rank_ic(pct_v, y_v))
        if r10 is not None:
            top10["pct"].append(topn_return(pct_dict, dict(zip(codes_v, y_v)), 10))
            top20["pct"].append(topn_return(pct_dict, dict(zip(codes_v, y_v)), 20))

        # 3) 中性化 (开)
        neu_dict = normalize_qlib_scores(raw_dict, pd.Timestamp(d_str), industry_map, neutralize=True)
        neu_v = [neu_dict[c] for c in codes_v]
        variants["neu"].append(rank_ic(neu_v, y_v))
        top10["neu"].append(topn_return(neu_dict, dict(zip(codes_v, y_v)), 10))
        top20["neu"].append(topn_return(neu_dict, dict(zip(codes_v, y_v)), 20))

        # 4) pct + macro
        pm_dict, sig1 = apply_macro_sector_bias(dict(pct_dict), pd.Timestamp(d_str))
        pm_v = [pm_dict[c] for c in codes_v]
        variants["pct+macro"].append(rank_ic(pm_v, y_v))
        top10["pct+macro"].append(topn_return(pm_dict, dict(zip(codes_v, y_v)), 10))
        top20["pct+macro"].append(topn_return(pm_dict, dict(zip(codes_v, y_v)), 20))
        if sig1:
            cov["macro_signal_dates"].add(d_str)

        # 5) neu + macro (回测默认路径)
        nm_dict, sig2 = apply_macro_sector_bias(dict(neu_dict), pd.Timestamp(d_str))
        nm_v = [nm_dict[c] for c in codes_v]
        variants["neu+macro"].append(rank_ic(nm_v, y_v))
        top10["neu+macro"].append(topn_return(nm_dict, dict(zip(codes_v, y_v)), 10))
        top20["neu+macro"].append(topn_return(nm_dict, dict(zip(codes_v, y_v)), 20))
        if sig2:
            cov["macro_signal_dates"].add(d_str)
        cov["macro_bonus_codes"] += sum(1 for c in codes_v if abs(nm_dict.get(c, 0) - neu_dict.get(c, 0)) > 1e-9)

        # 覆盖诊断 (每 5 期采样)
        if i % 5 == 0:
            try:
                from core.domain.factors.market_cap import get_market_cap_map
                mc = get_market_cap_map(codes_v, pd.Timestamp(d_str))
                cov["mktcap"].append(sum(1 for c in codes_v if mc.get(c)) / len(codes_v))
                if industry_map:
                    cov["industry"].append(sum(1 for c in codes_v if industry_map.get(c)) / len(codes_v))
                else:
                    cov["industry"].append(0.0)
            except Exception:
                pass

    print()
    print("=" * 100)
    print("行业中性化 + 宏观行业修正 验证 (w3 测试期, 缓存预测)")
    print("=" * 100)
    print(f"\n{'变体':10s} 截面 Rank IC 对比:")
    for k in variants:
        print(" ", summarize_ics(k, variants[k]))

    print(f"\nTop-10 等权 10 日净收益 (扣费 0.13%/期, 近似满换手):")
    for k in top10:
        a = np.array(top10[k])
        if len(a):
            net = (1 + a).prod() * (1 - (BUY_COST + SELL_COST)) ** len(a) - 1
            print(f"  {k:10s} 平均单期={a.mean():+.4%}  累计(毛)={((1+a).prod()-1):+.2%}  累计(净)={net:+.2%}")

    print(f"\nTop-20 等权 10 日净收益:")
    for k in top20:
        a = np.array(top20[k])
        if len(a):
            net = (1 + a).prod() * (1 - (BUY_COST + SELL_COST)) ** len(a) - 1
            print(f"  {k:10s} 平均单期={a.mean():+.4%}  累计(毛)={((1+a).prod()-1):+.2%}  累计(净)={net:+.2%}")

    print()
    print("覆盖诊断:")
    if cov["mktcap"]:
        print(f"  市值覆盖率: {np.mean(cov['mktcap']):.1%} (n={len(cov['mktcap'])})")
    print(f"  行业覆盖率: {np.mean(cov['industry']):.1%} (n={len(cov['industry'])})" if cov["industry"] else "  行业覆盖率: 0%")
    print(f"  宏观信号生效日: {len(cov['macro_signal_dates'])} / {len(rebal)}")
    print(f"  宏观加分覆盖股票数(累计): {cov['macro_bonus_codes']}")

    print()
    print("统计检验 (逐期配对, Top10 收益):")
    from scipy import stats as st
    if top10["pct"] and top10["neu"]:
        res = paired_test(top10["pct"], top10["neu"])
        if res:
            t, p, wins, n = res
            print(f"  中性化 vs 仅分位: 差均={np.mean(top10['neu'])-np.mean(top10['pct']):+.4%}/期, "
                  f"配对t={t:+.2f}, p={p:.3f}, 胜期 {wins}/{n}")
    if top10["pct"] and top10["pct+macro"]:
        res = paired_test(top10["pct"], top10["pct+macro"])
        if res:
            t, p, wins, n = res
            print(f"  宏观 vs 仅分位: 差均={np.mean(top10['pct+macro'])-np.mean(top10['pct']):+.4%}/期, "
                  f"配对t={t:+.2f}, p={p:.3f}, 胜期 {wins}/{n}")

    print()
    print("判读:")
    ics = {k: np.array([x for x in v if x is not None]) for k, v in variants.items()}
    if len(ics["neu"]) and len(ics["pct"]):
        d_ic = ics["neu"].mean() - ics["pct"].mean()
        d_t10 = np.mean(top10["neu"]) - np.mean(top10["pct"]) if top10["neu"] and top10["pct"] else 0
        print(f"  中性化 vs 仅分位: ΔIC={d_ic:+.4f}, ΔTop10单期={d_t10:+.4%}")
        if d_ic > 0:
            print("  → 行业中性化有效 (提升截面 IC)")
        else:
            print("  → 行业中性化 IC 层面有效, 但需与组合收益一起看")
    if len(ics["pct+macro"]) and len(ics["pct"]):
        d_ic = ics["pct+macro"].mean() - ics["pct"].mean()
        d_t10 = np.mean(top10["pct+macro"]) - np.mean(top10["pct"]) if top10["pct+macro"] and top10["pct"] else 0
        print(f"  宏观修正 vs 仅分位: ΔIC={d_ic:+.4f}, ΔTop10单期={d_t10:+.4%}")
        if abs(d_ic) < 1e-9 and abs(d_t10) < 1e-9:
            print("  → 宏观行业修正在预测池中零生效 (覆盖股票数为 0), 见覆盖诊断 — 属于失效")
        elif d_ic > 0:
            print("  → 宏观行业修正有效 (提升截面 IC)")
        else:
            print("  → 宏观行业修正无效/反效果 (降低截面 IC)")


if __name__ == "__main__":
    main()
