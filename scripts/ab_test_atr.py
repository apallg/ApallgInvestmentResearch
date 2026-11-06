import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atr_research_common import METRICS, print_diagnostics, require_oos_or_override


def run_backtest(runner, stop_source, start, end, atr_period, atr_mult):
    return runner.run(
        start=start, end=end,
        pool="a500", top_n=5, freq="week",
        use_rule_based=False, use_crf=True, use_vol_switch=False,
        macro_bias=True, sector_fusion=True,
        stop_source=stop_source,
        atr_period=atr_period, atr_mult=atr_mult,
    )


def _summary(r):
    return r.get("summary", {}) if isinstance(r, dict) else {}


def main():
    from core.domain.backtest.engine import DirectionBacktestRunner

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("--mult", type=float, default=2.0)
    parser.add_argument("--allow-seen-data", action="store_true")
    args = parser.parse_args()

    runner = DirectionBacktestRunner()
    require_oos_or_override(runner, args.start, args.end, args.allow_seen_data)
    print(f"回测区间 {args.start} ~ {args.end}, a500, Top5, 周频\n")

    print("=== 固定百分比止损 (fixed) ===")
    r1 = run_backtest(runner, "fixed", args.start, args.end, args.period, args.mult)
    s1 = _summary(r1)
    for k in METRICS:
        if k in s1:
            print(f"  {k:18s} {s1[k]}")
    print_diagnostics(r1)

    print(f"\n=== ATR 止损 (atr, 周期{args.period}, 倍数{args.mult}) ===")
    r2 = run_backtest(runner, "atr", args.start, args.end, args.period, args.mult)
    s2 = _summary(r2)
    for k in METRICS:
        if k in s2:
            print(f"  {k:18s} {s2[k]}")
    print_diagnostics(r2)

    print("\n=== 对比 (ATR - fixed) ===")
    for k in METRICS:
        if k in s1 and k in s2:
            d = s2[k] - s1[k]
            print(f"  {k:18s} {d:+.4f}")


if __name__ == "__main__":
    main()
