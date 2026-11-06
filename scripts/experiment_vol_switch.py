"""实验: 波动率切换 vs 单模型 vs 6状态切换

三种策略的选股 Rank IC 对比:
  1. 单模型: 固定 trend_momentum
  2. 6状态切换: RuleBased 规则 → REGIME_DIRECTION 选模型
  3. 波动率切换: VolRegimeClassifier 预测 volatile 概率, 高波用 oversold, 低波用 trend
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
    from core.domain.regime.classifier import RuleBasedRegimeClassifier
    from core.domain.regime.weights import get_regime_direction
    from core.domain.regime.vol_predictor import get_vol_classifier
    from core.data.fetcher import DataFetcher

    mgr = QlibModelManager()
    checker = OverfittingChecker()
    vol_cls = get_vol_classifier()

    # 1. predict 模型分数 (只需 trend + oversold, 切换只需这两个)
    preds = {}
    for d in ["trend_momentum", "oversold_rebound", "volume_breakout"]:
        exp_id, run_id = DIRECTION_MODELS[d]
        pred, _ = mgr.predict_dual(exp_id, run_id, predict_start="2026-01-01",
                                   predict_end="2026-06-30", instruments="a500")
        preds[d] = pred
        print(f"[加载] {d}: {len(pred)} 条")

    # 2. 指数数据
    index_df = DataFetcher().get_index_daily("沪深300", days=365 * 3)
    index_df = index_df.sort_values("date").reset_index(drop=True)
    if "trade_date" not in index_df.columns and "date" in index_df.columns:
        index_df["trade_date"] = index_df["date"]
    regime_cls = RuleBasedRegimeClassifier(index_df)

    try:
        dates = sorted(set(preds["trend_momentum"].index.get_level_values("datetime")))
    except Exception:
        dates = sorted(set(preds["trend_momentum"].index))
    dates = dates[::5]

    single_ics, switch_ics, vol_ics = [], [], []
    vol_choices = []
    for d in dates:
        d_str = str(d)[:10]
        # 6状态 → 方向模型
        rr = regime_cls.classify(as_of=pd.Timestamp(d_str))
        direction = get_regime_direction(rr.regime)
        # 波动率 → volatile 概率 (point-in-time)
        sub = index_df[index_df["trade_date"] <= pd.Timestamp(d_str)]
        p_vol = vol_cls.predict_proba(sub)
        vol_direction = "oversold_rebound" if (p_vol is not None and p_vol > 0.5) else "trend_momentum"
        vol_choices.append((d_str, round(p_vol, 2) if p_vol is not None else None, vol_direction))

        codes = sorted(_cross_section(preds["trend_momentum"], d_str).keys())
        if len(codes) < 30:
            continue
        fwd = checker._batch_forward_returns(codes, 10, pd.Timestamp(d_str))
        fwd_map = {c: r for c, r in zip(codes, fwd)} if fwd is not None else {}

        ic_single = _rank_ic(_cross_section(preds["trend_momentum"], d_str), fwd_map)
        ic_switch = _rank_ic(_cross_section(preds[direction], d_str), fwd_map)
        ic_vol = _rank_ic(_cross_section(preds[vol_direction], d_str), fwd_map)

        if ic_single is not None and ic_switch is not None and ic_vol is not None:
            single_ics.append(ic_single)
            switch_ics.append(ic_switch)
            vol_ics.append(ic_vol)

    a, b, c = np.array(single_ics), np.array(switch_ics), np.array(vol_ics)
    print(f"\n有效截面: {len(a)}")
    print(f"波动率切换: 高波(oversold) {sum(1 for x in vol_choices if x[2]=='oversold_rebound')} 次, "
          f"低波(trend) {sum(1 for x in vol_choices if x[2]=='trend_momentum')} 次\n")
    print(f"{'策略':14s} {'平均RankIC':>10s} {'ICIR':>8s} {'IC>0占比':>10s}")
    print(f"{'单模型(trend)':14s} {a.mean():>10.4f} {a.mean()/(a.std()+1e-9):>8.3f} {np.mean(a>0):>10.2%}")
    print(f"{'6状态切换':14s} {b.mean():>10.4f} {b.mean()/(b.std()+1e-9):>8.3f} {np.mean(b>0):>10.2%}")
    print(f"{'波动率切换':14s} {c.mean():>10.4f} {c.mean()/(c.std()+1e-9):>8.3f} {np.mean(c>0):>10.2%}")
    print(f"\n波动率切换 vs 单模型: {c.mean()-a.mean():+.4f};  vs 6状态切换: {c.mean()-b.mean():+.4f}")


if __name__ == "__main__":
    main()
