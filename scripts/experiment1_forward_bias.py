"""实验 1: 前视偏差验证 (训练期 IC vs 样本外 IC)

用 trend_momentum 因子集, 分别算「训练期」(2024~2025) 和「样本外」(2026) 的 Rank IC,
对比是否样本外明显退化 (说明因子集选择有过拟合/前视)。
"""
import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np


def main():
    from core.domain.factors.adapter import PublicFactorAdapter
    from core.domain.factors.validation import OverfittingChecker

    d = json.load(open("core/domain/factors/factor_pool_state.json", encoding="utf-8"))
    factors = d["curated"]["trend_momentum"].get("flat_list", [])
    # 取 5 个代表性因子 (覆盖不同类别): 动量/基本面/Alpha/资金流/技术
    factors = ["MOM", "SUE", "Alpha6", "flow", "tech"]

    adapter = PublicFactorAdapter()
    checker = OverfittingChecker()
    ctx = {"pool": "all", "top_k": 10}

    train_dates = checker._get_eval_dates(weeks=8, anchor_date=dt.date(2025, 6, 30))
    oos_dates = checker._get_eval_dates(weeks=8, anchor_date=dt.date(2026, 8, 14))
    print(f"训练期 {len(train_dates)} 天(截至2025-06), 样本外 {len(oos_dates)} 天(截至2026-08)\n")

    print(f"{'因子':18s} {'训练期IC':>10s} {'样本外IC':>10s} {'退化':>8s}")
    train_ics, oos_ics = [], []
    for key in factors:
        v = adapter.to_variant(key)
        if not v:
            continue
        try:
            ic_train = checker._calc_rank_ic(v, train_dates, ctx)
            ic_oos = checker._calc_rank_ic(v, oos_dates, ctx)
        except Exception as e:
            print(f"{key:18s} 失败: {e}")
            continue
        if ic_train is not None and ic_oos is not None:
            train_ics.append(ic_train)
            oos_ics.append(ic_oos)
            print(f"{key:18s} {ic_train:>10.4f} {ic_oos:>10.4f} {ic_oos - ic_train:>+8.4f}")

    if train_ics:
        mt, mo = np.mean(train_ics), np.mean(oos_ics)
        print(f"\n平均: 训练期IC={mt:.4f}  样本外IC={mo:.4f}  退化={mo - mt:+.4f}")
        if mo < mt * 0.5:
            print("结论: 样本外 IC 大幅退化 (>50%), 因子集存在前视/过拟合")
        elif mo < mt:
            print("结论: 样本外 IC 轻度退化, 有一定过拟合但不算严重")
        else:
            print("结论: 样本外 IC 未退化, 因子集无明显前视")


if __name__ == "__main__":
    main()
