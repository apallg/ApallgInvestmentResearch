"""实验 1 (v2): 前视偏差验证 — trend 模型三段 IC 对比

trend 模型 train=2019~2025-12, valid=2026-01~06。
对比「训练期(2025) / 验证期(2026上半年) / 样本外(2026下半年)」的 Rank IC,
看是否从训练期到样本外明显退化 (过拟合/前视)。
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
        out = {}
        for c, v in scores.items():
            out[norm_code(c)] = float(v)
        return out
    return {}


def _rank_ic(scores, fwd_map):
    common = [c for c in scores if c in fwd_map]
    if len(common) < 30:
        return None
    x = np.array([scores[c] for c in common])
    y = np.array([fwd_map[c] for c in common])
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def _ic_over_period(mgr, checker, start, end, instruments, forward_days=10):
    pred, _ = mgr.predict_dual(EXP_ID, RUN_ID, predict_start=start, predict_end=end, instruments=instruments)
    try:
        dates = sorted(set(pred.index.get_level_values("datetime")))
    except Exception:
        dates = sorted(set(pred.index))
    dates = dates[::5]  # 采样, 每5个交易日算一次, 控制耗时
    ics = []
    for d in dates:
        d_str = str(d)[:10]
        scores = _cross_section(pred, d_str)
        codes = sorted(scores.keys())
        if len(codes) < 30:
            continue
        fwd = checker._batch_forward_returns(codes, forward_days, pd.Timestamp(d_str))
        fwd_map = {c: r for c, r in zip(codes, fwd)} if fwd is not None else {}
        ic = _rank_ic(scores, fwd_map)
        if ic is not None:
            ics.append(ic)
    return ics


def main():
    from core.adapters.qlib.models import QlibModelManager
    from core.domain.factors.validation import OverfittingChecker

    mgr = QlibModelManager()
    checker = OverfittingChecker()
    instruments = "a500"

    periods = [
        ("训练期 2025", "2025-01-01", "2025-12-31"),
        ("验证期 2026上", "2026-01-01", "2026-06-30"),
        ("样本外 2026下", "2026-07-01", "2026-08-14"),
    ]
    print(f"{'时期':16s} {'截面数':>6s} {'平均RankIC':>10s} {'ICIR':>8s} {'IC>0占比':>10s}")
    for label, s, e in periods:
        try:
            ics = _ic_over_period(mgr, checker, s, e, instruments, forward_days=10)
        except Exception as ex:
            print(f"{label:16s} 失败: {ex}")
            continue
        if not ics:
            print(f"{label:16s} 无有效截面")
            continue
        a = np.array(ics)
        icir = a.mean() / (a.std() + 1e-9)
        print(f"{label:16s} {len(a):>6d} {a.mean():>10.4f} {icir:>8.3f} {np.mean(a > 0):>10.2%}")


if __name__ == "__main__":
    main()
