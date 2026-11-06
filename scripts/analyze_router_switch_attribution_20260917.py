import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/experiments/router_four_groups_20260917.json"
TRACE_SOURCE = ROOT / "data/experiments/contextual_top5_weekly_corrected_20260916.json"
OUT = ROOT / "data/experiments/router_switch_attribution_20260917.json"
REPORT = ROOT / "docs/router_switch_attribution_20260917.md"


def main():
    sys.path.insert(0, str(ROOT))
    import scripts.run_router_four_groups_20260917 as replay

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    trace_payload = json.loads(TRACE_SOURCE.read_text(encoding="utf-8"))
    trace = trace_payload["weekly_trace"]
    x_routes = [row["route_X"] for row in payload["routes"]]
    switch_indices = [i for i in range(1, len(x_routes)) if x_routes[i] != x_routes[i - 1]]
    switch_rows = []
    for i in switch_indices:
        switch_rows.append({"index": i, "date": payload["routes"][i]["date"], "from": x_routes[i - 1], "to": x_routes[i]})

    trend = replay._load_prediction(replay.TREND_RUN)
    rebound = replay._load_prediction(replay.REBOUND_RUN)
    common_dates = sorted(set(trend.index.get_level_values("date")) & set(rebound.index.get_level_values("date")))
    common_dates = [d for d in common_dates if pd.Timestamp("2025-07-01") <= d <= pd.Timestamp("2026-08-31")]
    weekly = replay._weekly_dates(common_dates)
    execution_map = {d: common_dates[common_dates.index(d) + 1] for d in weekly if d in common_dates and common_dates.index(d) + 1 < len(common_dates)}
    decision_dates = [d for d in weekly if d in execution_map]
    next_exec = [execution_map[d] for d in decision_dates]
    all_codes = set(trend.index.get_level_values("code")) & set(rebound.index.get_level_values("code"))
    dates_for_data = next_exec + next_exec[1:]
    opens = replay._load_open(all_codes, dates_for_data)
    trade_data = replay._load_trade_data(all_codes, dates_for_data)
    industry_map = replay._load_industry_map()
    t_week, t_baskets = replay._expert_weekly(trend, opens, decision_dates, next_exec)
    r_week, r_baskets = replay._expert_weekly(rebound, opens, decision_dates, next_exec)
    merged = t_week.merge(r_week, on=["date", "execution", "following_execution"], suffixes=("_t", "_r"))
    route_index = merged.set_index("date")
    x_baskets = payload["baskets"]["X_g0_volatility"]
    base = replay._account_constrained(route_index, x_baskets, trade_data, industry_map)
    counterfactuals = []
    for switch in switch_rows:
        routes = list(x_routes)
        routes[switch["index"]] = switch["from"]
        baskets = {str(date.date()): (r_baskets if routes[i] == "rebound" else t_baskets).get(str(date.date()), []) for i, date in enumerate(route_index.index)}
        account = replay._account_constrained(route_index, baskets, trade_data, industry_map)
        counterfactuals.append({**switch, "base_final_nav": base["final_nav"], "cancel_final_nav": account["final_nav"], "cancel_minus_base_nav": float(account["final_nav"] - base["final_nav"]), "cancel_minus_base_return": float(account["final_nav"] / base["final_nav"] - 1.0), "cancel_max_drawdown": account["max_drawdown"]})

    trace_by_date = {str(pd.Timestamp(row["date"]).date()): row for row in trace}
    fork_rows = []
    for i, route in enumerate(x_routes):
        row = trace_by_date.get(payload["routes"][i]["date"])
        if not row:
            continue
        trend_net = float(row["net_trend"])
        rebound_net = float(row["net_rebound"])
        selected = trend_net if route == "trend" else rebound_net
        oracle = max(trend_net, rebound_net)
        fork_rows.append({"date": payload["routes"][i]["date"], "route": route, "trend_net": trend_net, "rebound_net": rebound_net, "selected_net": selected, "oracle_net": oracle, "regret": oracle - selected, "correct": bool(selected == oracle)})
    fork = pd.DataFrame(fork_rows)
    oos = fork.iloc[26:] if len(fork) > 26 else fork.iloc[0:0]
    result = {"status": "router_switch_attribution_diagnostic", "base": {key: value for key, value in base.items() if key != "daily"}, "switch_count": len(switch_rows), "switches": switch_rows, "counterfactuals": counterfactuals, "decision_fork": {"n": int(len(fork)), "oos_n": int(len(oos)), "oos_accuracy": float(oos.correct.mean()) if len(oos) else None, "oos_mean_regret": float(oos.regret.mean()) if len(oos) else None, "oos_mean_selected_net": float(oos.selected_net.mean()) if len(oos) else None, "oos_mean_oracle_net": float(oos.oracle_net.mean()) if len(oos) else None}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 路由切换归因", "", "状态：`router_switch_attribution_diagnostic`。取消切换采用单期反事实：仅把切换周改为继续上一专家，其余路由日程保持不变。该结果用于判断增量是否集中在少数事件，不构成可交易策略。", "", f"基线 X 终值：{base['final_nav']:.6f}；实际切换次数：{len(switch_rows)}。", "", "## 单次取消切换", "", "| 日期 | 原路由 | 取消后路由 | 取消后终值 | 相对基线复利差 |", "|---|---|---|---:|---:|"]
    for row in counterfactuals:
        lines.append(f"| {row['date']} | {row['from']}→{row['to']} | 继续 {row['from']} | {row['cancel_final_nav']:.6f} | {row['cancel_minus_base_return']:.2%} |")
    lines += ["", "## 同期决策分叉（信号层）", "", f"OOS 周数：{result['decision_fork']['oos_n']}；路由选中事后最佳专家比例：{result['decision_fork']['oos_accuracy']:.2%}；平均选择后悔：{result['decision_fork']['oos_mean_regret']:.5f}；平均事后最优净收益：{result['decision_fork']['oos_mean_oracle_net']:.5f}。", "", "## 判定", "", "- 路由只有 5 次切换，不能满足至少 20 次事件的泛化门槛。", "- 单次取消切换只表示该事件在当前账户路径上的局部影响，不能把各事件差值相加。", "- 即使某次取消切换改善终值，也不能说明状态模型可预测；必须在冻结后的新周期重复。", "- 当前继续使用固定趋势研究基线，X 只保留影子记录。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
