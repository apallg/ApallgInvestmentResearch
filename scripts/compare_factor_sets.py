"""
因子集对照实验 — 同一候选池下, 不同「类内排序方式」产出的因子集对比。

目的: 定位「为什么类内插入顺序(uniform) 比 ICIR 排序更稳定」。

生成两组 (同一候选池、同一 target_n):
  A. uniform — 每类均匀配额, 类内按注册/插入顺序 (改后的精选, 不做数据排序)
  B. icir    — 每类均匀配额, 类内按历史 ICIR 降序 + L2/L3/L4 综合评分 (原精选排序)

输出: 两套 flat_list、类别分布、重叠/差异因子, 落盘 JSON 供训练对比。

用法:
  python scripts/compare_factor_sets.py trend_momentum              # 打印对比, 不保存
  python scripts/compare_factor_sets.py trend_momentum --save A     # 把 uniform(插入顺序) 保存为 curated
  python scripts/compare_factor_sets.py trend_momentum --save B     # 把 icir(排序) 保存为 curated
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json


def _build_sets(direction):
    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter
    from core.domain.factors.selector import curate_direction, group_by_category

    pool_mgr = get_pool()
    active = pool_mgr.get_active_factors()
    active_keys = list(active.keys())
    adapter = PublicFactorAdapter()

    name_a, flat_a, cnt_a = curate_direction(
        direction_name=direction, active_keys=active_keys,
        adapter=adapter, validation_results=None, save=False,
        regime_aware=True, pool_active=active, method="uniform",
    )
    name_b, flat_b, cnt_b = curate_direction(
        direction_name=direction, active_keys=active_keys,
        adapter=adapter, validation_results=None, save=False,
        regime_aware=True, pool_active=active, method="icir",
    )

    cats = group_by_category(active_keys, adapter, pool_active=active)
    return {
        "A": {"label": "uniform(插入顺序)", "flat": flat_a, "cnt": cnt_a},
        "B": {"label": "icir(排序)", "flat": flat_b, "cnt": cnt_b},
    }, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", nargs="?", default="trend_momentum")
    ap.add_argument("--save", choices=["A", "B"], default=None,
                    help="把 A(uniform) 或 B(icir) 保存为 curated[direction]")
    args = ap.parse_args()

    sets, cats = _build_sets(args.direction)

    for tag in ("A", "B"):
        s = sets[tag]
        print(f"=== {tag}. {s['label']}: {s['cnt']} 因子 ===")
        for c in cats:
            in_c = [k for k in s["flat"] if k in cats.get(c, [])]
            if in_c:
                print(f"  [{c}] {len(in_c)}: {' '.join(in_c)}")
        print()

    fa = sets["A"]["flat"]
    fb = sets["B"]["flat"]
    sa, sb = set(fa), set(fb)
    only_a = [k for k in fa if k not in sb]
    only_b = [k for k in fb if k not in sa]
    overlap = [k for k in fa if k in sb]
    print(f"重叠: {len(overlap)}  仅A(插入顺序): {len(only_a)}  仅B(icir排序): {len(only_b)}")
    print(f"\n仅 A 独有: {' '.join(only_a)}")
    print(f"\n仅 B 独有: {' '.join(only_b)}")

    out = {
        "direction": args.direction,
        "A": {"label": sets["A"]["label"], "flat": fa},
        "B": {"label": sets["B"]["label"], "flat": fb},
        "overlap": overlap, "only_A": only_a, "only_B": only_b,
    }
    os.makedirs("logs", exist_ok=True)
    path = f"logs/compare_factor_sets_{args.direction}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘 {path}")

    if args.save:
        from core.domain.factors.registry import get_pool
        s = sets[args.save]
        pool = get_pool()
        pool.save_curated_set(
            name=args.direction,
            factors_by_category={"方向精选": [(k, 0.0, {}, {}) for k in s["flat"]]},
            config={
                "description": f"对照实验 {s['label']}",
                "direction": args.direction,
                "categories": [],
                "target_n": s["cnt"],
                "method": f"compare_{args.save}",
            },
            stats={"total": s["cnt"], "method": f"compare_{args.save}",
                   "note": f"对照实验: {s['label']}"},
        )
        print(f"已保存 curated[{args.direction}] = {s['label']} ({s['cnt']} 因子)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
