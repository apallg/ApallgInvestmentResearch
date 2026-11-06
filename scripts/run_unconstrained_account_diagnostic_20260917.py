import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOUR = ROOT / "data/experiments/router_four_groups_20260917.json"
HMM = ROOT / "data/experiments/hmm_point_in_time_route_20260916.json"
OUT = ROOT / "data/experiments/unconstrained_account_diagnostic_20260917.json"
REPORT = ROOT / "docs/unconstrained_account_diagnostic_20260917.md"


def _bootstrap(values, block_size, draws=5000):
    values = np.asarray(values, dtype=float)
    blocks = [values[i:i + block_size] for i in range(0, len(values), block_size)]
    rng = np.random.default_rng(20260917 + block_size)
    means = []
    for _ in range(draws):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])[:len(values)]
        means.append(float(sample.mean()))
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "lower": float(low), "upper": float(high), "n": int(len(values)), "block_size": block_size}


def main():
    sys.path.insert(0, str(ROOT))
    import scripts.run_router_four_groups_20260917 as replay
    from scripts.run_contextual_top5_weekly_corrected import _account as simple_account

    four = json.loads(FOUR.read_text(encoding="utf-8"))
    hmm = json.loads(HMM.read_text(encoding="utf-8"))
    trend = replay._load_prediction(replay.TREND_RUN)
    rebound = replay._load_prediction(replay.REBOUND_RUN)
    common = sorted(set(trend.index.get_level_values("date")) & set(rebound.index.get_level_values("date")))
    common = [d for d in common if pd.Timestamp("2025-07-01") <= d <= pd.Timestamp("2026-08-31")]
    weekly = replay._weekly_dates(common)
    execution_map = {d: common[common.index(d) + 1] for d in weekly if d in common and common.index(d) + 1 < len(common)}
    decision_dates = [d for d in weekly if d in execution_map]
    next_exec = [execution_map[d] for d in decision_dates]
    all_codes = set(trend.index.get_level_values("code")) & set(rebound.index.get_level_values("code"))
    opens = replay._load_open(all_codes, next_exec + next_exec[1:])
    t_week, t_baskets = replay._expert_weekly(trend, opens, decision_dates, next_exec)
    r_week, r_baskets = replay._expert_weekly(rebound, opens, decision_dates, next_exec)
    fusion_baskets, _ = replay._fusion_baskets(trend, rebound, opens, decision_dates, next_exec)
    t_route = t_week.merge(r_week, on=["date", "execution", "following_execution"], suffixes=("_t", "_r")).set_index("date")
    hmm_trace = {str(pd.Timestamp(row["date"]).date()): row["route_hmm_pit"] for row in hmm["weekly_trace"]}
    x_trace = {str(pd.Timestamp(row["date"]).date()): row["route_X"] for row in four["routes"]}
    baskets = {
        "T_fixed_trend": t_baskets,
        "R_fixed_rebound": r_baskets,
        "F_static_fusion": fusion_baskets,
        "X_g0_volatility": {key: (r_baskets if x_trace.get(key) == "rebound" else t_baskets).get(key, []) for key in x_trace},
        "H0_point_in_time_HMM": {key: (r_baskets if hmm_trace.get(key) == "rebound" else t_baskets).get(key, []) for key in hmm_trace},
    }
    accounts = {}
    for name, basket in baskets.items():
        account = simple_account(t_route, basket, opens, next_exec)
        accounts[name] = {key: value for key, value in account.items() if key != "daily"}
    frames = {name: pd.DataFrame(simple_account(t_route, basket, opens, next_exec)["daily"]).set_index("date") for name, basket in baskets.items()}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    increments = {}
    for left, right in [("T_fixed_trend", "H0_point_in_time_HMM"), ("T_fixed_trend", "X_g0_volatility"), ("T_fixed_trend", "F_static_fusion")]:
        common_idx = frames[left].index.intersection(frames[right].index)
        delta = (frames[right].nav.loc[common_idx].pct_change(fill_method=None) - frames[left].nav.loc[common_idx].pct_change(fill_method=None)).dropna()
        increments[f"{right}_minus_{left}"] = {"ci20": _bootstrap(delta.to_numpy(), 20), "ci40": _bootstrap(delta.to_numpy(), 40), "positive_share": float((delta > 0).mean()), "n": int(len(delta))}
    result = {"status": "unconstrained_equal_weight_account_diagnostic", "contract": {"top_n": 5, "buy_cost": replay.BUY_COST, "sell_cost": replay.SELL_COST, "constraints": "none_except_equal_weight_basket_and_available_open"}, "accounts": accounts, "increments": increments}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 无组合约束等权账户诊断", "", "状态：`unconstrained_equal_weight_account_diagnostic`。使用同一预测和成交开盘，但移除行业、换手、整手和涨跌停限制，仅保留等权 Top5 和费用，用于隔离选股/路由信号本身。结果仍是开发期诊断。", "", "|方案|终值|收益|最大回撤|平均换手|", "|---|---:|---:|---:|---:|"]
    for name, x in accounts.items():
        lines.append(f"|{name}|{x['final_nav']:.6f}|{x['return']:.2%}|{x['max_drawdown']:.2%}|{x['mean_turnover']:.3f}|")
    lines += ["", "|比较|平均周期增量|20周块95%区间|40周块95%区间|正增量占比|", "|---|---:|---:|---:|---:|"]
    for name, x in increments.items():
        lines.append(f"|{name}|{x['ci20']['mean']:.5f}|[{x['ci20']['lower']:.5f}, {x['ci20']['upper']:.5f}]|[{x['ci40']['lower']:.5f}, {x['ci40']['upper']:.5f}]|{x['positive_share']:.1%}|")
    lines += ["", "## 结论", "", "- 若 HMM 在无约束账户中仍低于固定趋势，说明主要问题来自专家选择/持仓排序，而非行业和换手约束。", "- 若 HMM 只在无约束账户中改善，说明实际账户差异主要来自换手、整手或可成交性，路由仍不能直接上线。", "- 本诊断不替代真实约束账户；生产判断继续以约束账户和独立确认期为准。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
