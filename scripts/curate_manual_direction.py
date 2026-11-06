"""
手工组合策略方向因子集 — 凭经济逻辑, 不看 IC/数据。

趋势动量 (trend_momentum) 设计思路:
  追涨不抄底 — 识别处于上涨趋势中的股票, 期望趋势延续。

  五个维度 (缺一不可):
    1. 价格动量   — 趋势的核心信号: 股票过去 N 日在涨 (多时间尺度, 避免超短反转噪声)
    2. 趋势强度   — 趋势的方向与力度: 均线多头排列 / 创新高 / 趋势拟合度
    3. 成交量确认 — 趋势需要资金支撑: 放量 + 量价同向 + 流动性加速
    4. 成长       — 趋势市中成长股占优: 营收/盈利增速 + 盈利质量 (真实 MySQL 财务)
    5. 规模/活跃   — 小盘弹性 + 换手活跃度

  全部选用有明确表达式、可被 Qlib 解析的因子 (Alpha191/158/101 + FundVal 真实基本面),
  不用 *_n5/n20/n60/n120 这类空 label 的数据挖掘扩展因子。

用法:
  python scripts/curate_manual_direction.py            # 组合并保存 trend_momentum
  python scripts/curate_manual_direction.py --dry-run  # 只校验+打印, 不保存
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


# 每个因子: (key, 含义注释)
MANUAL_SETS = {
    "trend_momentum": {
        "label": "趋势动量",
        "desc": "追涨不抄底，牛市/上涨趋势中动量+成长因子主导 (人工经济逻辑组合)",
        "factors": {
            "价格动量": [
                ("a191_mom_05",  "20日收益 (月动量, 核心)"),
                ("a191_mom_06",  "40日收益 (季动量)"),
                ("a191_mom_08",  "120日收益 (半年动量)"),
                ("a191_mom_11",  "20日动量隔10 (Jegadeesh-Titman经典, 跳近期反转)"),
                ("a191_mom_13",  "10日加权收益 (近期权重高)"),
                ("a191_mom_14",  "20日夏普比 (动量质量: 收益/风险)"),
                ("a191_mom_15",  "60日夏普比 (中期动量质量)"),
                ("a191_mom_22",  "20日盈亏比 (涨幅/跌幅, 趋势健康度)"),
                ("a191_mom_24",  "20日胜率 (上涨天数占比)"),
            ],
            "趋势强度": [
                ("a158_ma5_20",   "MA5/MA20 (短期均线金叉)"),
                ("a158_ma10_60",  "MA10/MA60 (中期均线多头)"),
                ("a158_resi20",   "价格偏离20日均线 (趋势延伸)"),
                ("a158_resi60",   "价格偏离60日均线 (长期趋势)"),
                ("a158_high20",   "距20日高点距离 (逼近新高)"),
                ("a191_trd_06",   "多头排列度 (均线多头排列计数)"),
                ("a191_trd_10",   "60日新高 (突破长期箱体)"),
                ("a191_trd_15",   "Aroon-Up (趋势启动强度)"),
                ("a191_trd_18",   "趋势R方 (价格线性趋势拟合度)"),
                ("a101_a065",     "Alpha65新高动量 (创20日新高程度)"),
            ],
            "成交量确认": [
                ("a191_amt_03",   "20日均量比 (相对60日放量)"),
                ("a191_amt_08",   "OBV动量 (量价同向累积)"),
                ("a191_amt_16",   "5日量价相关 (涨跌与量的同步性)"),
                ("a191_amt_21",   "量增价涨比 (健康放量占比)"),
                ("a101_a038",     "Alpha38放量突破 (价量双突破)"),
                ("a101_a075",     "Alpha75均线交叉+量 (金叉且放量)"),
                ("a101_a077",     "Alpha77趋势动量 (量价趋势共振)"),
                ("a191_liq_14",   "流动性动量 (换手率加速)"),
            ],
            "成长": [
                ("fund_rev_yoy",    "营收同比增速 (FundVal真实财务)"),
                ("fund_profit_yoy", "净利润同比增速 (FundVal真实财务)"),
                ("fund_roe",        "ROE (盈利质量, FundVal真实财务)"),
            ],
            "规模/活跃": [
                ("fund_size",       "总市值 (小盘弹性, FundVal真实财务)"),
                ("fund_float_mv",   "流通市值 (小盘弹性, FundVal真实财务)"),
                ("fund_turnover",   "换手率 (交易活跃度, FundVal真实财务)"),
            ],
        },
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", nargs="?", default="trend_momentum")
    ap.add_argument("--dry-run", action="store_true", help="只校验+打印, 不保存")
    args = ap.parse_args()

    from core.domain.factors.registry import get_pool
    from core.adapters.qlib.factor_resolver import resolve_factor_expression

    spec = MANUAL_SETS[args.direction]
    pool = get_pool()
    active = pool.get_active_factors()

    # 校验: 是否活跃 + 是否可解析
    flat = []
    problems = []
    for dim, items in spec["factors"].items():
        for key, note in items:
            flat.append(key)
            if key not in active:
                problems.append(f"{key}: 不在活跃因子池")
            expr, skip = resolve_factor_expression(key)
            if not expr:
                problems.append(f"{key}: 无法解析 ({skip})")

    print(f"=== 手工组合 [{args.direction}] {spec['label']} ===")
    print(f"维度分布:")
    for dim, items in spec["factors"].items():
        print(f"  {dim}: {len(items)} 个")
        for key, note in items:
            mark = "✓" if key not in [p.split(':')[0] for p in problems] else "✗"
            print(f"    {mark} {key:<18} {note}")
    print(f"\n合计: {len(flat)} 个因子")

    if problems:
        print("\n⚠️ 问题因子:")
        for p in problems:
            print(f"  - {p}")
        if not args.dry_run:
            print("\n存在无法解析的因子, 中止保存。请修正清单。")
            return 1

    if args.dry_run:
        print("\n[dry-run] 未保存")
        return 0

    # 保存为 curated set (覆盖 trend_momentum)
    factors_by_category = {
        dim: [(key, 0.0, {}, {}) for key, _ in items]
        for dim, items in spec["factors"].items()
    }
    pool.save_curated_set(
        name=args.direction,
        factors_by_category=factors_by_category,
        config={
            "description": spec["desc"],
            "direction": args.direction,
            "categories": list(spec["factors"].keys()),
            "target_n": len(flat),
            "method": "manual_expert",
        },
        stats={
            "total": len(flat),
            "method": "manual_expert",
            "note": "凭经济逻辑手工组合, 不看IC/数据; 动量+趋势+量价+成长+规模五维",
        },
    )
    print(f"\n已保存 curated set '{args.direction}': {len(flat)} 因子 (method=manual_expert)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
