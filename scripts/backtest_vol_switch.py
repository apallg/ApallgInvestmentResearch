"""回测验证: 波动率切换 vs 6状态切换 的完整交易效果。

调用 DirectionBacktestRunner.run() 两次, 对比 use_vol_switch=False/True 的绩效。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

METRICS = ["annual_return", "annual_excess", "sharpe", "max_drawdown",
           "win_rate", "excess_win_rate", "calmar", "total_return"]


def run_backtest(use_vol_switch):
    from core.domain.backtest.engine import DirectionBacktestRunner
    runner = DirectionBacktestRunner()
    return runner.run(
        start="2026-01-01", end="2026-08-14",
        pool="a500", top_n=5, freq="week",
        use_rule_based=False, use_crf=True, use_vol_switch=use_vol_switch,
        macro_bias=True, sector_fusion=True,
        stop_source="fixed",
    )


def _summary(r):
    s = r.get("summary", {}) if isinstance(r, dict) else {}
    return s


def main():
    print("回测区间 2026-01-01 ~ 2026-08-14, a500, Top5, 周频\n")

    print("=== 6 状态切换 (use_vol_switch=False) ===")
    r1 = run_backtest(False)
    s1 = _summary(r1)
    for k in METRICS:
        if k in s1:
            print(f"  {k:18s} {s1[k]}")

    print("\n=== 波动率切换 (use_vol_switch=True) ===")
    r2 = run_backtest(True)
    s2 = _summary(r2)
    for k in METRICS:
        if k in s2:
            print(f"  {k:18s} {s2[k]}")

    print("\n=== 对比 (波动率切换 - 6状态切换) ===")
    for k in METRICS:
        if k in s1 and k in s2:
            d = s2[k] - s1[k]
            print(f"  {k:18s} {d:+.4f}")


if __name__ == "__main__":
    main()
