"""验证「波动择时」效果: 用 VolRegimeClassifier 的 volatile 概率驱动仓位, 对比满仓持有沪深300。

方案:
  1. 满仓基准 (buy & hold)
  2. 硬阈值: p_vol > 0.5 → 半仓
  3. 软缩放: 仓位 = 1 - 0.6 * p_vol
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def _metrics(daily_ret):
    """年化收益 / 最大回撤 / 夏普 (从日收益序列)"""
    eq = (1 + daily_ret).cumprod()
    ann = eq.iloc[-1] ** (252 / len(daily_ret)) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252)
    return ann, mdd, sharpe


def main():
    from core.data.fetcher import DataFetcher
    from core.domain.regime.vol_predictor import get_vol_classifier

    m = get_vol_classifier()
    df = DataFetcher().get_index_daily("沪深300", days=365 * 2).sort_values("date").reset_index(drop=True)
    te = pd.Timestamp(m.last_train_date)

    close = df["close"].astype(float)
    ret = close.pct_change(fill_method=None)

    positions = {"满仓": [], "硬阈值": [], "软缩放": []}
    p_vols = []
    for i in range(len(df)):
        if i < 200:
            positions["满仓"].append(1.0); positions["硬阈值"].append(1.0); positions["软缩放"].append(1.0)
            p_vols.append(np.nan)
            continue
        p = m.predict_proba(df.iloc[:i + 1])
        if p is None:
            p = 0.5
        p_vols.append(p)
        positions["满仓"].append(1.0)
        positions["硬阈值"].append(0.5 if p > 0.5 else 1.0)
        positions["软缩放"].append(max(0.4, 1.0 - 0.6 * p))

    # 只统计样本外 (te 之后), T+1 持仓
    oos_mask = (df["date"] > te).values
    print(f"样本外区间: {te.date()} 之后, n={oos_mask.sum()} 交易日")
    print(f"volatile 概率均值={np.nanmean(p_vols):.3f}\n")

    print(f"{'方案':8s} {'年化收益':>10s} {'最大回撤':>10s} {'夏普':>8s}")
    for name in ["满仓", "硬阈值", "软缩放"]:
        pos = pd.Series(positions[name])
        # 仓位作用于下一日收益 (t 日决定仓位, t+1 日收益)
        strat_ret = pos.shift(1) * ret
        strat_ret = strat_ret[oos_mask & pos.shift(1).notna()]
        ann, mdd, sharpe = _metrics(strat_ret.fillna(0))
        print(f"{name:8s} {ann:>10.2%} {mdd:>10.2%} {sharpe:>8.2f}")


if __name__ == "__main__":
    main()
