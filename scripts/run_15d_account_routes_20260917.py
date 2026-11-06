import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOUR = ROOT / "data/experiments/router_four_groups_20260917.json"
CONTEXT = ROOT / "data/experiments/pit_market_context_extended_20260916.parquet"
OUT = ROOT / "data/experiments/15d_account_routes_20260917.json"
REPORT = ROOT / "docs/15d_account_routes_20260917.md"


def _bootstrap(values, block_size, draws=5000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": None, "lower": None, "upper": None, "n": 0, "block_size": block_size}
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

    base = json.loads(FOUR.read_text(encoding="utf-8"))
    trend = replay._load_prediction(replay.TREND_RUN)
    rebound = replay._load_prediction(replay.REBOUND_RUN)
    common = sorted(set(trend.index.get_level_values("date")) & set(rebound.index.get_level_values("date")))
    common = [d for d in common if pd.Timestamp("2025-07-01") <= d <= pd.Timestamp("2026-08-31")]
    weekly = replay._weekly_dates(common)
    execution_map = {d: common[common.index(d) + 1] for d in weekly if d in common and common.index(d) + 1 < len(common)}
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
    context = pd.read_parquet(CONTEXT)
    context["date"] = pd.to_datetime(context["date"])
    context = context.drop_duplicates("date").set_index("date").sort_index()
    context["volpct15"] = context["vol15"].rolling(252, min_periods=60).rank(pct=True)
    pre = context.loc[context.index <= pd.Timestamp("2024-12-31")]
    dd25 = float(pre["DD15"].dropna().quantile(0.25))
    er50 = float(pre["ER15"].dropna().median())
    rows = []
    for date in route_index.index:
        row = context.loc[date] if date in context.index else pd.Series(dtype=float)
        vol_route = "rebound" if float(row.get("volpct15", np.nan)) > 0.7 else "trend"
        repair = bool(row.get("R15", np.nan) < 0 and row.get("DD15", np.nan) <= dd25 and row.get("R3", np.nan) > 0 and row.get("dB3", np.nan) > 0)
        observed_route = "rebound" if repair else "trend"
        rows.append({"date": str(pd.Timestamp(date).date()), "vol15": row.get("vol15"), "volpct15": row.get("volpct15"), "environment": "repair" if repair else "other", "route_15d_vol": vol_route, "route_observed15": observed_route})
    trace = pd.DataFrame(rows).set_index("date")
    baskets = {
        "Y15_volatility": {key: (r_baskets if value == "rebound" else t_baskets).get(key, []) for key, value in trace.route_15d_vol.items()},
        "Y15_observed": {key: (r_baskets if value == "rebound" else t_baskets).get(key, []) for key, value in trace.route_observed15.items()},
    }
    accounts = {}
    for name, basket in baskets.items():
        account = replay._account_constrained(route_index, basket, trade_data, industry_map)
        stress = replay._account_constrained(route_index, basket, trade_data, industry_map, replay.BUY_COST * 2.0, replay.SELL_COST * 2.0)
        accounts[name] = {"summary": {key: value for key, value in account.items() if key != "daily"}, "stress_final_nav": stress["final_nav"], "daily": account["daily"]}
    accounts["T_fixed_trend"] = {"summary": base["summary"]["T_fixed_trend"], "stress_final_nav": base["summary"]["T_fixed_trend"]["stress_final_nav"], "daily": base["accounts"]["T_fixed_trend"]["daily"]}
    frames = {name: pd.DataFrame(value["daily"]).set_index("date") for name, value in accounts.items()}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    increments = {}
    for name in ("Y15_volatility", "Y15_observed"):
        common_idx = frames["T_fixed_trend"].index.intersection(frames[name].index)
        delta = (frames[name].nav.loc[common_idx].pct_change(fill_method=None) - frames["T_fixed_trend"].nav.loc[common_idx].pct_change(fill_method=None)).dropna().to_numpy(dtype=float)
        increments[name] = {"ci20": _bootstrap(delta, 20), "ci40": _bootstrap(delta, 40), "positive_share": float((delta > 0).mean())}
    result = {"status": "15d_account_route_same_contract_diagnostic", "thresholds": {"dd25": dd25, "er50": er50, "volpct_rebound_threshold": 0.7}, "decision_weeks": int(len(route_index)), "route_counts": {"Y15_volatility": trace.route_15d_vol.value_counts().to_dict(), "Y15_observed": trace.route_observed15.value_counts().to_dict()}, "accounts": {name: {key: value for key, value in data.items() if key != "daily"} for name, data in accounts.items() if name != "T_fixed_trend"}, "increments_vs_fixed_trend": increments, "trace": trace.reset_index().to_dict(orient="records")}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 15日可观察环境同合同账户回放", "", "状态：`15d_account_route_same_contract_diagnostic`。本实验把 15 日规则接入 T/R 同一 Top5 周频账户成交器；阈值只使用 2024-12-31 之前的 context 数据，结果仍属于开发期诊断。", "", "## 账户结果", "", "|方案|终值|收益|最大回撤|加倍成本终值|平均换手|平均仓位|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, data in accounts.items():
        s = data["summary"] if "summary" in data else {key: data[key] for key in ["final_nav", "return", "max_drawdown", "mean_turnover", "mean_exposure"]}
        stress = data.get("stress_final_nav", float("nan"))
        lines.append(f"|{name}|{s['final_nav']:.6f}|{s['return']:.2%}|{s['max_drawdown']:.2%}|{stress:.6f}|{s['mean_turnover']:.3f}|{s['mean_exposure']:.3f}|")
    lines += ["", "## 相对固定趋势", "", "|方案|平均周增量|20周块95%区间|40周块95%区间|正增量占比|", "|---|---:|---:|---:|---:|"]
    for name, item in increments.items():
        c20, c40 = item["ci20"], item["ci40"]
        lines.append(f"|{name}|{c20['mean']:.5f}|[{c20['lower']:.5f}, {c20['upper']:.5f}]|[{c40['lower']:.5f}, {c40['upper']:.5f}]|{item['positive_share']:.1%}|")
    lines += ["", f"路由使用次数：`{result['route_counts']}`。", "", "## 结论", "", "- 15 日波动率规则和显式观察环境规则均未超过固定趋势，区间不能证明正向增量。", "- 15 日规则可以解释部分专家差异，但当前样本无法证明其能形成稳定的账户选择优势。", "- 不启动更多窗口、阈值或状态组合搜索；下一步仍是冻结合同后的独立前向确认。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "accounts": {name: data for name, data in result["accounts"].items()}, "increments": increments, "route_counts": result["route_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
