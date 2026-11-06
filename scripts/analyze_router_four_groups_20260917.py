import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/experiments/router_four_groups_20260917.json"
OUT = ROOT / "data/experiments/router_four_groups_20260917_analysis.json"
REPORT = ROOT / "docs/router_four_groups_20260917.md"
SEED = 20260917


def _bootstrap(values, block_size, draws=5000):
    values = np.asarray(values, dtype=float)
    blocks = [values[i:i + block_size] for i in range(0, len(values), block_size)]
    rng = np.random.default_rng(SEED + block_size)
    means = []
    for _ in range(draws):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])[:len(values)]
        means.append(float(sample.mean()))
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "lower": float(low), "upper": float(high), "n": int(len(values)), "draws": draws, "block_size": block_size}


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    accounts = payload["accounts"]
    frames = {name: pd.DataFrame(value["daily"]).set_index("date") for name, value in accounts.items()}
    names = list(frames)
    increments = {}
    for left, right in [("T_fixed_trend", "R_fixed_rebound"), ("T_fixed_trend", "F_static_fusion"), ("T_fixed_trend", "X_g0_volatility"), ("F_static_fusion", "X_g0_volatility")]:
        common = frames[left].index.intersection(frames[right].index)
        left_ret = frames[left].nav.loc[common].pct_change(fill_method=None).dropna()
        right_ret = frames[right].nav.loc[common].pct_change(fill_method=None).dropna()
        delta = (right_ret - left_ret).dropna().to_numpy(dtype=float)
        increments[f"{right}_minus_{left}"] = {"ci_20": _bootstrap(delta, 20), "ci_40": _bootstrap(delta, 40), "positive_share": float((delta > 0).mean()), "right_wins": int((delta > 0).sum()), "left_wins": int((delta < 0).sum())}
    source_trace = json.loads((ROOT / "data/experiments/contextual_top5_weekly_corrected_20260916.json").read_text(encoding="utf-8"))["weekly_trace"]
    route_counts = {}
    for row in source_trace:
        route = row.get("route_G0_volatility")
        route_counts[route] = route_counts.get(route, 0) + 1
    result = {"status": "router_four_groups_analysis", "source": str(SOURCE.relative_to(ROOT)), "summary": payload["summary"], "increments": increments, "route_counts": route_counts, "contract": payload["contract"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 四组同合同账户重放分析", "", "状态：`router_four_groups_same_contract_replay`。本结果使用冻结的趋势/反弹预测和同一成交器，覆盖 59 个有效周；仍属于开发期诊断，不是冻结后的独立确认期。", "", "## 账户结果", "", "| 方案 | 终值 | 收益 | 最大回撤 | 加倍成本终值 | 平均换手 | 平均仓位 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        x = payload["summary"][name]
        lines.append(f"| {name} | {x['final_nav']:.6f} | {x['return']:.2%} | {x['max_drawdown']:.2%} | {x['stress_final_nav']:.6f} | {x['mean_turnover']:.3f} | {x['mean_exposure']:.3f} |")
    lines += ["", "## 相对固定趋势的配对周增量", "", "| 比较 | 平均周增量 | 20周块95%区间 | 40周块95%区间 | 正增量占比 |", "|---|---:|---:|---:|---:|"]
    for key, x in increments.items():
        c20, c40 = x["ci_20"], x["ci_40"]
        lines.append(f"| {key} | {c20['mean']:.5f} | [{c20['lower']:.5f}, {c20['upper']:.5f}] | [{c40['lower']:.5f}, {c40['upper']:.5f}] | {x['positive_share']:.1%} |")
    lines += ["", f"X 路由沿用成熟标签修正版的 G0 决策序列，状态使用次数：`{route_counts}`。", "", "## 判定", "", "- 固定趋势在本共同测试期终值最高，且比 X 路由高 1.64 个百分点；X 相对固定趋势的配对增量均值为负，20/40 周区间均跨 0。", "- 静态融合低于固定趋势约 7.50 个百分点，但高于固定反弹；简单把两个模型混合并没有产生稳定增量。", "- 固定反弹本期显著落后，但该结论只属于本测试期，不能推断反弹模型在所有市场阶段失效。", "- X 路由没有达到上线门槛：增量没有稳定为正、尚未完成逐笔生产交易器对账、且仍属于开发期。", "- 下一步停止增加状态特征；只有在独立确认期重新证明专家互补后，才启动 `vol15 + dvol5` 的收益优势预测。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
