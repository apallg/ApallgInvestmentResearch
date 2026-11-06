"""类别均衡 + 类内 ICIR 排序: 6 类各 8 个 (对齐 uniform), 类内按带符号 ICIR 降序。

对比 uniform(类内插入顺序) 与 split/gain(类内排序但整类淘汰成长/规模)。
核心问题: 在「类别均衡」前提下, 类内排序(ICIR) vs 类内插入顺序, 谁好。

用法: python scripts/curate_balanced_icir.py [--dry-run]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

PLACEHOLDERS = {"flow", "size", "tech", "value", "quality", "test_f1", "momentum"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target-n", type=int, default=48)
    args = ap.parse_args()

    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter
    from core.domain.factors.selector import (
        DIRECTION_PRESETS, group_by_category, _dedupe_by_expression, _build_direction_icir_map)

    direction = "trend_momentum"
    preset = DIRECTION_PRESETS[direction]
    pool = get_pool()
    active = pool.get_active_factors()
    active_keys = _dedupe_by_expression(list(active.keys()))
    adapter = PublicFactorAdapter()
    category_groups = group_by_category(active_keys, adapter, pool_active=active)

    all_keys = [k for c in preset["categories"] for k in category_groups.get(c, [])]
    icir_map, _, _, icir_source = _build_direction_icir_map(direction, all_keys, True)
    print(f"ICIR source: {icir_source}")

    per_cat = args.target_n // len(preset["categories"])
    taken = []
    report = {}
    for cat in preset["categories"]:
        keys = category_groups.get(cat, [])
        keys = [k for k in keys if k not in PLACEHOLDERS]
        ranked = sorted(keys, key=lambda k: icir_map.get(k, -999.0), reverse=True)
        top = ranked[:per_cat]
        taken.extend(top)
        report[cat] = [(k, icir_map.get(k, 0)) for k in top]
        print(f"[{cat}] 候选 {len(keys)} → 取 {len(top)}: " + ", ".join(top))

    print(f"\n合计 {len(taken)} 因子 (类别均衡, 类内 ICIR 排序)")

    if args.dry_run:
        print("[dry-run] 未保存")
        return 0

    factors_by_category = {c: [(k, 0.0, {}, {"icir": icir_map.get(k, 0)}) for k, _ in report[c]]
                           for c in preset["categories"]}
    pool.save_curated_set(
        name="trend_momentum_balanced_icir",
        factors_by_category=factors_by_category,
        config={
            "description": "类别均衡(6类各8) + 类内ICIR排序",
            "direction": direction,
            "categories": preset["categories"],
            "target_n": args.target_n,
            "method": "balanced_icir",
            "icir_source": icir_source,
        },
        stats={"total": len(taken), "method": "balanced_icir",
               "note": "每类强制配额 + 类内带符号ICIR降序"},
    )
    print(f"\n已保存 curated set 'trend_momentum_balanced_icir': {len(taken)} 因子")
    return 0


if __name__ == "__main__":
    sys.exit(main())
