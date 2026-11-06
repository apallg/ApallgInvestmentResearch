"""
在 gain 排名 32 因子集基础上，补回「基本面增速」和「市值规模」两类因子。

验证假设: gain 排名整类淘汰成长/规模类是否导致 RankIC 退步。
组成: gain32(价格动量8+趋势强度8+高阶8+成交量8) + 成长3 + 规模3 = 38 因子。

用法:
  python scripts/add_fundamental_to_gain.py            # 组合并保存 trend_momentum
  python scripts/add_fundamental_to_gain.py --dry-run  # 只校验+打印, 不保存
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


# 顺序对齐 DIRECTION_PRESETS 6 类: 价格动量/趋势强度/成长/规模/高阶特征/成交量特征
MIXED_SETS = {
    "价格动量": [
        "a191_mom_21", "a191_mom_22", "a191_mom_26", "a191_mom_15",
        "a191_mom_25", "AROON", "a191_mom_11", "a101_a021",
    ],
    "趋势强度": [
        "a158_ma10_60", "a191_trd_19", "a191_trd_20", "a191_trd_14",
        "a191_trd_10", "a101_a076", "a158_low20", "a158_ma5_20",
    ],
    "成长": [
        "fund_rev_yoy", "fund_profit_yoy", "fund_roe",
    ],
    "规模": [
        "fund_size", "fund_float_mv", "fund_turnover",
    ],
    "高阶特征": [
        "fund_asset_turnover", "fund_earn_stb", "a191_hof_23", "a191_hof_05",
        "a191_hof_11", "a101_a089", "a158_corr_pv20", "a191_hof_04",
    ],
    "成交量特征": [
        "avg_turnover_20d", "a191_amt_23", "a158_vol20", "a101_a039",
        "a191_liq_09", "a191_amt_08", "alpha3", "a191_liq_13",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只校验+打印, 不保存")
    args = ap.parse_args()

    from core.domain.factors.registry import get_pool
    from core.adapters.qlib.factor_resolver import resolve_factor_expression

    pool = get_pool()
    active = pool.get_active_factors()

    flat = []
    problems = []
    for dim, keys in MIXED_SETS.items():
        for key in keys:
            if key in flat:
                problems.append(f"{key}: 重复")
                continue
            flat.append(key)
            if key not in active:
                problems.append(f"{key}: 不在活跃因子池")
            expr, skip = resolve_factor_expression(key)
            if not expr:
                problems.append(f"{key}: 无法解析 ({skip})")

    print(f"=== 混合因子集 [gain32 + 成长/规模] ===")
    for dim, keys in MIXED_SETS.items():
        print(f"  {dim}: {len(keys)} 个  ->  {' '.join(keys)}")
    print(f"\n合计: {len(flat)} 个因子")

    if problems:
        print("\n⚠️ 问题因子:")
        for p in problems:
            print(f"  - {p}")
        if not args.dry_run:
            print("\n存在无法解析的因子, 中止保存。")
            return 1

    if args.dry_run:
        print("\n[dry-run] 未保存")
        return 0

    factors_by_category = {
        dim: [(key, 0.0, {}, {}) for key in keys]
        for dim, keys in MIXED_SETS.items()
    }
    pool.save_curated_set(
        name="trend_momentum",
        factors_by_category=factors_by_category,
        config={
            "description": "gain32 + 基本面增速(成长) + 市值规模: 验证类型补全是否回升 RankIC",
            "direction": "trend_momentum",
            "categories": list(MIXED_SETS.keys()),
            "target_n": len(flat),
            "method": "gain32_plus_fundamental",
        },
        stats={
            "total": len(flat),
            "method": "gain32_plus_fundamental",
            "note": "gain排名32因子 + 补回成长3(fund_rev_yoy/profit_yoy/roe) + 规模3(fund_size/float_mv/turnover)",
        },
    )
    print(f"\n已保存 curated set 'trend_momentum': {len(flat)} 因子 (method=gain32_plus_fundamental)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
