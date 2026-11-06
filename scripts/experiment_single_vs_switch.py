"""实验: 单模型(trend) vs 市场状态切换选模型

对比两种策略的选股 Rank IC:
  1. 单模型: 每期固定用 trend_momentum 模型分数
  2. 切换:   每期根据市场状态(6状态)用 REGIME_DIRECTION 映射选方向模型分数

市场状态用 RuleBasedRegimeClassifier (确定性规则, point-in-time 可算历史状态)。
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
    from core.data.fetcher import DataFetcher

    mgr = QlibModelManager()
    checker = OverfittingChecker()

    # 1. predict 4 模型分数
    preds = {}
    for d in DIRECTION_MODELS:
        exp_id, run_id = DIRECTION_MODELS[d]
        pred, _ = mgr.predict_dual(exp_id, run_id, predict_start="2026-01-01",
                                   predict_end="2026-06-30", instruments="a500")
        preds[d] = pred
        print(f"[加载] {d}: {len(pred)} 条")

    # 2. 指数数据 (算市场状态)
    index_df = DataFetcher().get_index_daily("沪深300", days=365 * 3)
    if "trade_date" not in index_df.columns and "date" in index_df.columns:
        index_df["trade_date"] = index_df["date"]
    regime_cls = RuleBasedRegimeClassifier(index_df)

    # 3. 逐日对比
    try:
        dates = sorted(set(preds["trend_momentum"].index.get_level_values("datetime")))
    except Exception:
        dates = sorted(set(preds["trend_momentum"].index))
    dates = dates[::5]

    single_ics, switch_ics = [], []
    switch_choices = {}
    for d in dates:
        d_str = str(d)[:10]
        # 市场状态 → 方向模型
        rr = regime_cls.classify(as_of=pd.Timestamp(d_str))
        direction = get_regime_direction(rr.regime)
        switch_choices[direction] = switch_choices.get(direction, 0) + 1

        codes = sorted(_cross_section(preds["trend_momentum"], d_str).keys())
        if len(codes) < 30:
            continue
        fwd = checker._batch_forward_returns(codes, 10, pd.Timestamp(d_str))
        fwd_map = {c: r for c, r in zip(codes, fwd)} if fwd is not None else {}

        # 单模型: 固定 trend
        s_single = _cross_section(preds["trend_momentum"], d_str)
        ic_single = _rank_ic(s_single, fwd_map)
        # 切换: 用 market 状态选的方向模型
        s_switch = _cross_section(preds[direction], d_str)
        ic_switch = _rank_ic(s_switch, fwd_map)

        if ic_single is not None and ic_switch is not None:
            single_ics.append(ic_single)
            switch_ics.append(ic_switch)

    a, b = np.array(single_ics), np.array(switch_ics)
    print(f"\n有效截面: {len(a)}")
    print(f"市场状态切换选择的方向模型分布: {switch_choices}\n")
    print(f"{'策略':16s} {'平均RankIC':>10s} {'ICIR':>8s} {'IC>0占比':>10s}")
    print(f"{'单模型(trend)':14s} {a.mean():>10.4f} {a.mean()/(a.std()+1e-9):>8.3f} {np.mean(a>0):>10.2%}")
    print(f"{'市场状态切换':14s} {b.mean():>10.4f} {b.mean()/(b.std()+1e-9):>8.3f} {np.mean(b>0):>10.2%}")
    print(f"\n单模型 vs 切换 差值: 平均IC {a.mean()-b.mean():+.4f}, "
          f"单模型胜率(逐截面IC更大) {np.mean(a>b):.2%}")


if __name__ == "__main__":
    main()
