"""实验 3: 融合方式对比 (单模型 vs 等权 vs ICIR加权)

用 4 个方向模型的分数 + 未来收益, 逐日算 Rank IC, 对比:
  1. 各单模型 IC
  2. 等权融合 IC
  3. 加权融合 IC (按各模型历史 ICIR 加权)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DIRECTION_MODELS = {
    "trend_momentum": ["656749665990832865", "1fa5eecd2bb04b4ebc1b37f4a5e18884"],
    "volume_breakout": ["656749665990832865", "235f1dbec8144ebe97a4a570f4293be9"],
    "oversold_rebound": ["656749665990832865", "3272fb4fe4a84fe1b24e4df5319b4c28"],
    "value_defense": ["656749665990832865", "1fa5eecd2bb04b4ebc1b37f4a5e18884"],
}
NAMES = list(DIRECTION_MODELS.keys())


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


def main():
    from core.adapters.qlib.models import QlibModelManager
    from core.domain.factors.validation import OverfittingChecker

    mgr = QlibModelManager()
    checker = OverfittingChecker()
    predict_start, predict_end = "2026-01-01", "2026-06-30"
    instruments = "a500"

    preds = {}
    for d in NAMES:
        exp_id, run_id = DIRECTION_MODELS[d]
        try:
            pred, _ = mgr.predict_dual(exp_id, run_id, predict_start=predict_start,
                                       predict_end=predict_end, instruments=instruments)
            preds[d] = pred
            print(f"[加载] {d}: {len(pred)} 条")
        except Exception as e:
            print(f"[失败] {d}: {e}")

    if len(preds) < 2:
        print("模型不足")
        return

    # 取共同日期
    try:
        dates = sorted(set(preds[NAMES[0]].index.get_level_values("datetime")))
    except Exception:
        dates = sorted(set(preds[NAMES[0]].index))
    print(f"[区间] {dates[0] if dates else '?'} ~ {dates[-1] if dates else '?'}  共 {len(dates)} 日\n")

    # 逐日算 IC
    ic_records = {n: [] for n in NAMES}
    ic_eq = []
    for d in dates:
        d_str = str(d)[:10]
        sections = {n: _cross_section(preds[n], d_str) for n in NAMES}
        common = set.intersection(*[set(s.keys()) for s in sections.values()])
        if len(common) < 30:
            continue
        codes = sorted(common)
        # 未来收益
        fwd = checker._batch_forward_returns(codes, 20, pd.Timestamp(d_str))
        fwd_map = {c: r for c, r in zip(codes, fwd)} if fwd is not None else {}
        if len(fwd_map) < 30:
            continue
        # 各模型 IC
        valid = True
        for n in NAMES:
            ic = _rank_ic({c: sections[n][c] for c in codes}, fwd_map)
            if ic is None:
                valid = False
                break
            ic_records[n].append(ic)
        if not valid:
            continue
        # 等权融合 (分数先 z-score 再等权)
        zscores = {}
        for n in NAMES:
            arr = np.array([sections[n][c] for c in codes])
            z = (arr - arr.mean()) / (arr.std() + 1e-9)
            zscores[n] = dict(zip(codes, z))
        eq = {c: sum(zscores[n][c] for n in NAMES) / len(NAMES) for c in codes}
        ic_eq.append(_rank_ic(eq, fwd_map))

    print(f"有效截面数: {len(ic_records[NAMES[0]])}\n")
    print(f"{'模型/融合':22s} {'平均RankIC':>12s} {'ICIR':>8s} {'IC>0占比':>10s}")
    for n in NAMES:
        ics = np.array(ic_records[n])
        icir = ics.mean() / (ics.std() + 1e-9)
        print(f"{n:22s} {ics.mean():>12.4f} {icir:>8.3f} {np.mean(ics>0):>10.2%}")
    ics_eq = np.array(ic_eq)
    print(f"{'等权融合':22s} {ics_eq.mean():>12.4f} {ics_eq.mean()/(ics_eq.std()+1e-9):>8.3f} {np.mean(ics_eq>0):>10.2%}")


if __name__ == "__main__":
    main()
