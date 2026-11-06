# -*- coding: utf-8 -*-
"""
THS 历史分时回填 — 1分钟分时 → 5分钟K线 → ths_minute_bars
数据源: 同花顺 v2/time/hs_{code}/{yyyymmdd}.js（历史分时回放，全历史可用）

用法:
  py -3 scripts/ths_5min_backfill.py --code 000001 --start 2025-04-01 --end 2025-05-30
  py -3 scripts/ths_5min_backfill.py --code 000001 --start 2024-01-01 --end 2025-05-30 --interval 0.4
  py -3 scripts/ths_5min_backfill.py --daily --code 000001 --years 2022 2023 2024 2025

说明: --interval 为请求间隔秒(默认0.5, 同花顺限流较严建议>=0.4)；周末/节假日自动跳过。
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.ths_collector import THSCollector
from core.data.repos.intraday import IntradayDataRepo


def main():
    ap = argparse.ArgumentParser(description="THS 历史分时/日K回填")
    ap.add_argument("--code", default="000001")
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--daily", action="store_true", help="回填日K(按年)")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = ap.parse_args()

    c = THSCollector()
    end = date.today() if not args.end else date.fromisoformat(args.end)

    if not args.daily:
        start = date.fromisoformat(args.start)
        print(f"[5min] 回填 {args.code} {start} ~ {end} ...")
        result = c.backfill_5min(args.code, start, end, interval=args.interval)
        total = sum(result.values())
        print(f"[5min] 完成: {len(result)} 个有数据交易日, 共 {total} 根 5 分钟K线")
        for d in sorted(result):
            print(f"        {d}: {result[d]} 根")
        rows = IntradayDataRepo.get_history_bars(args.code, start, end)
        print(f"[5min] 读回验证: ths_minute_bars 区间内共 {len(rows)} 行")
        if rows:
            print(f"        首行 {rows[0]['bar_time']} O={rows[0]['open']} H={rows[0]['high']} "
                  f"L={rows[0]['low']} C={rows[0]['close']} V={rows[0]['volume']}")
            print(f"        末行 {rows[-1]['bar_time']} C={rows[-1]['close']}")
    else:
        print(f"[daily] 回填 {args.code} 日K 年份 {args.years} ...")
        result = c.backfill_daily(args.code, args.years)
        print(f"[daily] 完成: {result}")
        rows = IntradayDataRepo.get_daily_kline(
            args.code, f"{min(args.years)}-01-01", f"{max(args.years)}-12-31"
        )
        print(f"[daily] 读回验证: ths_daily_kline 共 {len(rows)} 行")


if __name__ == "__main__":
    main()
