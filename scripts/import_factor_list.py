"""
从「趋势动量_因子清单.md」导入因子集，覆盖 curated.trend_momentum。

来源: 桌面导出的 uniform 48 因子集 (selector.DIRECTION_PRESETS uniform 模式产物)。
7 个占位符 (flow/size/tech/value/quality/test_f1/momentum) 无表达式, 训练时自动跳过。

用法:
  python scripts/import_factor_list.py            # 保存
  python scripts/import_factor_list.py --dry-run  # 只校验+打印, 不保存
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


FACTOR_LIST = {
    "价格动量": ["MOM", "IMOM", "IMom", "AROON", "CF_RET", "cf_rev", "halpha", "max_n5", "a101_a052"],
    "趋势强度": ["a101_a055", "a101_a059", "a101_a060"],
    "成长": ["SUE", "Alpha7", "Alpha8", "Alpha11", "FSUE_n5", "roic_n5", "FSUE_n10", "FSUE_n20"],
    "规模": ["SP_n5", "ICV_n5", "SP_n10", "SP_n20", "SP_n60", "mktcap", "ICV_n10", "ICV_n20"],
    "高阶特征": ["a101_a056", "a101_a058", "a101_a061", "a101_a062",
               "flow", "size", "tech", "value", "quality", "test_f1", "momentum"],
    "成交量特征": ["CR", "PVT", "beta", "Alpha6", "CF_REV", "Mkt_n5", "alpha1", "alpha3"],
    "波动率特征": ["a101_a054"],
}

PLACEHOLDERS = {"flow", "size", "tech", "value", "quality", "test_f1", "momentum"}


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
    skipped_placeholders = []
    for dim, keys in FACTOR_LIST.items():
        for key in keys:
            if key in flat:
                problems.append(f"{key}: 重复")
                continue
            flat.append(key)
            if key in PLACEHOLDERS:
                skipped_placeholders.append(key)
                continue
            if key not in active:
                problems.append(f"{key}: 不在活跃因子池")
            expr, skip = resolve_factor_expression(key)
            if not expr:
                problems.append(f"{key}: 无法解析 ({skip})")

    print(f"=== 导入因子集 [趋势动量_因子清单.md] ===")
    for dim, keys in FACTOR_LIST.items():
        print(f"  {dim}: {len(keys)} 个  ->  {' '.join(keys)}")
    print(f"\n合计: {len(flat)} 个 key (含 {len(skipped_placeholders)} 个占位符, 训练自动跳过)")
    print(f"实际可训练因子: {len(flat) - len(skipped_placeholders)} 个")

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
        for dim, keys in FACTOR_LIST.items()
    }
    pool.save_curated_set(
        name="trend_momentum",
        factors_by_category=factors_by_category,
        config={
            "description": "追涨不抄底，牛市/上涨趋势中动量+成长因子主导 (uniform 48 因子集)",
            "direction": "trend_momentum",
            "categories": list(FACTOR_LIST.keys()),
            "target_n": len(flat),
            "method": "uniform_48",
        },
        stats={
            "total": len(flat),
            "method": "uniform_48",
            "note": f"从桌面因子清单导入, 含 {len(skipped_placeholders)} 个占位符 (训练跳过)",
        },
    )
    print(f"\n已保存 curated set 'trend_momentum': {len(flat)} key (method=uniform_48)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
