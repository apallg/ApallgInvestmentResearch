import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.domain.backtest.engine import DirectionBacktestRunner
from atr_research_common import print_diagnostics, require_oos_or_override

K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
PERIOD_GRID = [7, 14, 21]


def run_once(runner, stop_source, start, end, atr_period=14, atr_mult=2.0):
    r = runner.run(
        start=start, end=end,
        pool="a500", top_n=5, freq="week",
        use_rule_based=False, use_crf=True, use_vol_switch=False,
        macro_bias=True, sector_fusion=True,
        stop_source=stop_source,
        atr_period=atr_period, atr_mult=atr_mult,
    )
    return r.get("summary", {}) if isinstance(r, dict) else {}


def fmt(s):
    return (f"ann={s.get('annual_return', 0):.3f} sharpe={s.get('sharpe', 0):.2f} "
            f"mdd={s.get('max_drawdown', 0):.3f} calmar={s.get('calmar', 0):.2f} "
            f"total={s.get('total_return', 0):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--allow-seen-data", action="store_true")
    args = parser.parse_args()

    runner = DirectionBacktestRunner()
    require_oos_or_override(runner, args.start, args.end, args.allow_seen_data)

    print("=== baseline: fixed ===")
    r_fixed = runner.run(
        start=args.start, end=args.end, pool="a500", top_n=5, freq="week",
        use_rule_based=False, use_crf=True, use_vol_switch=False,
        macro_bias=True, sector_fusion=True, stop_source="fixed")
    s_fixed = r_fixed.get("summary", {}) if isinstance(r_fixed, dict) else {}
    print(f"  {fmt(s_fixed)}")
    print_diagnostics(r_fixed)

    print("\n=== ATR sweep (period x k) ===")
    rows = []
    for period in PERIOD_GRID:
        for k in K_GRID:
            s = run_once(runner, "atr", args.start, args.end,
                         atr_period=period, atr_mult=k)
            rows.append((period, k, s))
            print(f"  period={period:2d} k={k:.1f}  {fmt(s)}")

    print("\n=== top 5 by sharpe ===")
    rows.sort(key=lambda r: -(r[2].get("sharpe", 0)))
    for period, k, s in rows[:5]:
        print(f"  period={period:2d} k={k:.1f}  {fmt(s)}")


if __name__ == "__main__":
    main()
