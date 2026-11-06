"""
遗传编程因子挖掘 — 研究脚本入口。

进化表达式树, 适应度 = 五因子中性化后的截面 RankIC 均值。
产出 Top-N 因子入库 (带 qlib expr), 打印训练/验证 IC 报告。

用法:
  python scripts/mine_genetic_factors.py --pool csi300 \
      --start 2020-01-01 --end 2024-06-30 \
      --valid-start 2024-07-01 --valid-end 2025-06-30 \
      --pop 200 --gen 20 --top 10

说明:
  - 训练区间进化 (允许过拟合), 验证区间算样本外 IC 作最终筛选
  - 五因子中性化: 市值 + 行业 + 动量 + 波动 + 换手
  - --dry-run 只跑不写因子池
"""
import sys
import os
import math
import argparse
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="csi300")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--valid-start", default="2024-07-01")
    ap.add_argument("--valid-end", default="2025-06-30")
    ap.add_argument("--step", type=int, default=5, help="每隔N个交易日取一个评估日")
    ap.add_argument("--forward-days", type=int, default=20)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="只跑不写因子池")
    args = ap.parse_args()

    from core.domain.factors.validation import OverfittingChecker
    from core.domain.factors.genetic import GeneticFactorMiner, build_mining_context
    from core.domain.screening._screener import get_screener

    checker = OverfittingChecker()
    train_dates = checker._dates_between(
        dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end), args.step)
    valid_dates = checker._dates_between(
        dt.date.fromisoformat(args.valid_start), dt.date.fromisoformat(args.valid_end), args.step)
    print(f"训练区间 {len(train_dates)} 评估日, 验证区间 {len(valid_dates)} 评估日")

    screener = get_screener()
    snapshots, mktcap_maps, extra_factors = build_mining_context(
        screener, args.pool, list(train_dates) + list(valid_dates),
        risk_neutralize=True,
        progress_cb=lambda done, total, dkey: print(f"  [{done}/{total}] {dkey}"),
    )
    if not snapshots:
        print("无可用快照, 退出")
        return 1

    from core.data.fetcher import DataFetcher
    try:
        industry_map = DataFetcher().get_industry_map() or {}
    except Exception:
        industry_map = {}

    miner = GeneticFactorMiner(
        snapshots, forward_days=args.forward_days,
        mktcap_maps=mktcap_maps, industry_map=industry_map, extra_factors=extra_factors,
    )
    result = miner.mine(
        population=args.pop, generations=args.gen, top_n=args.top,
        train_dates=[d.isoformat() for d in train_dates],
        valid_dates=[d.isoformat() for d in valid_dates],
        seed=args.seed,
    )

    print("\n=== 遗传挖矿结果 (按验证 IC 降序) ===")
    print(f"{'#':<4}{'验证IC':>9}{'训练IC':>9}  表达式")
    for i, r in enumerate(result):
        vi = r["valid_ic"]
        ti = r["train_ic"]
        vs = f"{vi * 100:.2f}%" if math.isfinite(vi) else "  nan"
        ts = f"{ti * 100:.2f}%" if math.isfinite(ti) else "  nan"
        print(f"{i:<4}{vs:>9}{ts:>9}  {r['expr']}")

    if args.dry_run:
        print("\n[dry-run] 未入库")
        return 0

    from core.domain.factors.registry import get_pool
    pool = get_pool()
    added = 0
    for i, r in enumerate(result):
        key = f"gp_{i:03d}"
        metrics = {
            "train_ic": round(r["train_ic"], 4) if math.isfinite(r["train_ic"]) else None,
            "valid_ic": round(r["valid_ic"], 4) if math.isfinite(r["valid_ic"]) else None,
        }
        pool.add_factor(
            key=key, label=f"GP-{r['expr'][:36]}", expr=r["expr"],
            metrics=metrics, category="white_box",
        )
        added += 1
    print(f"\n已入库 {added} 个遗传因子到因子池 white_box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
