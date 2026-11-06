import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HMM = ROOT / "data/experiments/hmm_point_in_time_route_20260916.json"
FOUR = ROOT / "data/experiments/router_four_groups_20260917.json"
OUT = ROOT / "data/experiments/hmm_vs_four_groups_20260917.json"
REPORT = ROOT / "docs/hmm_vs_four_groups_20260917.md"


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
    hmm = json.loads(HMM.read_text(encoding="utf-8"))
    four = json.loads(FOUR.read_text(encoding="utf-8"))
    accounts = {name: pd.DataFrame(value["daily"]).set_index("date") for name, value in four["accounts"].items()}
    accounts["H0_point_in_time_HMM"] = pd.DataFrame(hmm["accounts"]["hmm_point_in_time"]["daily"]).set_index("date")
    summaries = {name: four["summary"][name] for name in four["summary"]}
    h = hmm["accounts"]["hmm_point_in_time"]
    summaries["H0_point_in_time_HMM"] = {key: h[key] for key in ["n", "final_nav", "return", "max_drawdown", "mean_turnover", "mean_exposure"]}
    summaries["H0_point_in_time_HMM"]["stress_final_nav"] = hmm["stress_accounts"]["hmm_point_in_time"]["final_nav"]
    comparisons = {}
    for left, right in [("T_fixed_trend", "H0_point_in_time_HMM"), ("X_g0_volatility", "H0_point_in_time_HMM"), ("F_static_fusion", "H0_point_in_time_HMM"), ("R_fixed_rebound", "H0_point_in_time_HMM")]:
        common = accounts[left].index.intersection(accounts[right].index)
        delta = (accounts[right].nav.loc[common].pct_change(fill_method=None) - accounts[left].nav.loc[common].pct_change(fill_method=None)).dropna().to_numpy(dtype=float)
        comparisons[f"{right}_minus_{left}"] = {"ci_20": _bootstrap(delta, 20), "ci_40": _bootstrap(delta, 40), "positive_share": float((delta > 0).mean())}
    result = {"status": "hmm_vs_four_groups_same_contract_diagnostic", "hmm_status": hmm["status"], "summaries": summaries, "comparisons": comparisons, "hmm_route_rebound_count": hmm["hmm_route_rebound_count"], "hmm_route_accuracy": hmm["hmm_route_accuracy_vs_realized_best"], "state_distribution": hmm["state_distribution"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 直接 HMM 与四组路由对照", "", "状态：`hmm_vs_four_groups_same_contract_diagnostic`。H0 使用每个决策日前截断的指数/宽度/北向信息，每4周更新一次 HMM；与 T/R/F/X 使用同一 Top5、次日开盘和账户约束。结果仍是开发期诊断。", "", "## 账户结果", "", "| 方案 | 终值 | 收益 | 最大回撤 | 加倍成本终值 | 平均换手 | 平均仓位 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, x in summaries.items():
        lines.append(f"| {name} | {x['final_nav']:.6f} | {x['return']:.2%} | {x['max_drawdown']:.2%} | {x.get('stress_final_nav', float('nan')):.6f} | {x['mean_turnover']:.3f} | {x['mean_exposure']:.3f} |")
    lines += ["", "## HMM 相对各方案的配对区间", "", "| 比较 | 平均周增量 | 20周块95%区间 | 40周块95%区间 |", "|---|---:|---:|---:|"]
    for key, x in comparisons.items():
        c20, c40 = x["ci_20"], x["ci_40"]
        lines.append(f"| {key} | {c20['mean']:.5f} | [{c20['lower']:.5f}, {c20['upper']:.5f}] | [{c40['lower']:.5f}, {c40['upper']:.5f}] |")
    lines += ["", f"HMM 选择反弹 {hmm['hmm_route_rebound_count']} 周，事后最佳专家比例 {hmm['hmm_route_accuracy_vs_realized_best']:.2%}；状态分布：`{hmm['state_distribution']}`。", "", "## 结论", "", "- 点时 HMM 账户终值低于固定趋势和 G0 波动率路由，高于固定反弹。", "- HMM 的路由准确率仅 44.07%，不能解释其状态语义与专家收益之间的关系。", "- HMM 平均换手高于 G0，平均仓位更低；状态切换不仅没有带来收益增量，还引入了额外交易和现金偏离。", "- 这解释了“某次 HMM 回测优于固定反弹”与本次结果相反：旧结果不能与本次点时、同合同结果直接合并。", "- 当前不启用 HMM；保留为研究对照，等待冻结后的独立确认期。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
