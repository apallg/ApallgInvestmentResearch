"""实验 5: 中性化验证 — 非中性化 IC vs 市值+行业中性化 IC

用 trend 模型分数, 逐日对比「原始分数」和「中性化后分数」与未来收益的 Rank IC,
验证中性化是否提升 IC / ICIR。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

EXP_ID, RUN_ID = "656749665990832865", "1fa5eecd2bb04b4ebc1b37f4a5e18884"


def _cross_section(pred, d_str):
    from core.domain.backtest.common import norm_code
    try:
        day = pred.loc[d_str]
    except Exception:
        try:
            day = pred.xs(d_str, level="datetime")
        except Exception:
            return {}
    if hasattr(day, "iloc"):
        scores = day.iloc[:, 0] if day.ndim > 1 else day
        return {norm_code(c): float(v) for c, v in scores.items()}
    return {}


def _rank_ic(scores, fwd_map):
    common = [c for c in scores if c in fwd_map]
    if len(common) < 30:
        return None
    x = np.array([scores[c] for c in common])
    y = np.array([fwd_map[c] for c in common])
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def main():
    from core.adapters.qlib.models import QlibModelManager
    from core.domain.factors.validation import OverfittingChecker
    from core.domain.factors.market_cap import get_market_cap_map
    from core.domain.factors.neutralize import cross_sectional_neutralize
    from core.data.fetcher import DataFetcher

    mgr = QlibModelManager()
    checker = OverfittingChecker()
    industry_map = DataFetcher().get_industry_map() or {}

    pred, _ = mgr.predict_dual(EXP_ID, RUN_ID, predict_start="2026-01-01",
                               predict_end="2026-06-30", instruments="a500")
    try:
        dates = sorted(set(pred.index.get_level_values("datetime")))
    except Exception:
        dates = sorted(set(pred.index))
    dates = dates[::5]

    raw_ics, neu_ics = [], []
    for d in dates:
        d_str = str(d)[:10]
        scores = _cross_section(pred, d_str)
        codes = sorted(scores.keys())
        if len(codes) < 30:
            continue
        fwd = checker._batch_forward_returns(codes, 10, pd.Timestamp(d_str))
        fwd_map = {c: r for c, r in zip(codes, fwd)} if fwd is not None else {}
        # 非中性化 IC
        ic_raw = _rank_ic(scores, fwd_map)
        # 中性化 IC
        mktcap_map = get_market_cap_map(codes, pd.Timestamp(d_str))
        resid = cross_sectional_neutralize(codes, [scores[c] for c in codes], mktcap_map, industry_map)
        ic_neu = _rank_ic(resid, fwd_map)
        if ic_raw is not None and ic_neu is not None:
            raw_ics.append(ic_raw)
            neu_ics.append(ic_neu)

    if not raw_ics:
        print("无有效截面")
        return
    a, b = np.array(raw_ics), np.array(neu_ics)
    print(f"有效截面: {len(a)}\n")
    print(f"{'方式':14s} {'平均RankIC':>10s} {'ICIR':>8s} {'IC>0占比':>10s}")
    print(f"{'非中性化':14s} {a.mean():>10.4f} {a.mean()/(a.std()+1e-9):>8.3f} {np.mean(a>0):>10.2%}")
    print(f"{'市值+行业中性化':14s} {b.mean():>10.4f} {b.mean()/(b.std()+1e-9):>8.3f} {np.mean(b>0):>10.2%}")
    print(f"\n中性化提升: 平均IC {b.mean()-a.mean():+.4f}, ICIR {b.mean()/(b.std()+1e-9) - a.mean()/(a.std()+1e-9):+.3f}")


if __name__ == "__main__":
    main()
