import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
result_dir = ROOT / "data/experiments/stock_group_split_p1_pilot_20261020_qfqfix"
frame = pd.read_parquet(result_dir / "predictions.parquet")
summary_path = result_dir / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
else:
    summary = {}
    indexed = frame.set_index(["date", "instrument"]).sort_index()
    indexed.index = indexed.index.set_names(["datetime", "instrument"])
    predictions = {}
    labels = {}
    for arm in ("A", "B", "C"):
        arm_frame = indexed.loc[indexed["arm"] == arm]
        predictions[arm] = arm_frame["prediction"]
        labels[arm] = arm_frame["label"]
        valid = arm_frame.dropna(subset=["prediction", "label", "group_id"])
        daily_ic = valid.groupby(level="datetime").apply(lambda part: part["prediction"].corr(part["label"], method="spearman")).dropna()
        summary[arm] = {
            "n_rows": int(len(valid)),
            "n_dates": int(valid.index.get_level_values("datetime").nunique()),
            "rank_ic_mean": float(daily_ic.mean()) if len(daily_ic) else None,
            "unknown_share": float((valid["group_id"] == 6).mean()) if len(valid) else None,
            "group_counts": {str(k): int(v) for k, v in valid["group_id"].value_counts().sort_index().items()},
        }
    paired = {}
    for arm in ("A", "B", "C"):
        common_dates = sorted(set(predictions[arm].dropna().index.get_level_values("datetime")) & set(labels[arm].dropna().index.get_level_values("datetime")))
        returns = []
        for signal_date in common_dates[::10]:
            score = predictions[arm].xs(signal_date, level="datetime").dropna().nlargest(5).index
            returns.append(float(labels[arm].xs(signal_date, level="datetime").reindex(score).mean()))
        paired[arm] = {"signals": len(returns), "mean_return": float(np.mean(returns)) if returns else None, "returns": returns}
    summary["paired_top5_10d_gross_diagnostic"] = paired
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
manifest_path = result_dir / "manifest.json"
if not manifest_path.exists():
    coverage = json.loads((ROOT / "data/experiments/stock_context_p1_w8_qfqfix_20261020/coverage.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "p1_context_pilot_predictions_complete",
        "evidence_stage": "single_incomplete_historical_fold_diagnostic",
        "date_frozen": "2026-10-20",
        "reference_run_id": "656749665990832865/5501ed88f12145f0a454d68d92dbad76",
        "factor_count": 40,
        "taxonomy": "G1 six broad industries plus unknown; historical effective intervals, applied strictly after in_date",
        "context_panel_rows": coverage["rows"],
        "context_features": coverage["feature_columns"],
        "arms": {"A": "40 original features, unified model", "B": "same plus P1 context features, unified model", "C": "same features, six group experts with B fallback for unknown or expert failure"},
        "train": ["2023-01-01", "2025-12-31"],
        "early_stopping": ["2026-01-01", "2026-03-31"],
        "calibration": ["2026-04-01", "2026-06-30"],
        "test": ["2026-07-01", "2026-09-15"],
        "label": "tradable_open_to_open, horizon 10, label_norm none",
        "seed": 42,
        "calibration": "separate 2026Q2, non-negative strictly positive slopes; non-positive group slope falls back to pooled slope to preserve within-group order",
        "lstm_epochs": 80,
        "expert_failures": {},
        "full_plan_decision_eligible": False,
        "limitations": ["one seed and W8 only", "industry source lacks complete publication-vintage history", "no weekly fee-adjusted account ledger", "full 8-window matrix not run"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
sample = frame.pivot(index=["date", "instrument"], columns="arm", values="prediction")
complete = sample.dropna(subset=["A", "B", "C"])
same_samples = len(complete) == int(summary["A"]["n_rows"]) == int(summary["B"]["n_rows"]) == int(summary["C"]["n_rows"])
returns = summary["paired_top5_10d_gross_diagnostic"]
arm_returns = {arm: np.asarray(returns[arm]["returns"], dtype=float) for arm in ("A", "B", "C")}
effects = {
    "B_minus_A": float(np.mean(arm_returns["B"] - arm_returns["A"])),
    "C_minus_B": float(np.mean(arm_returns["C"] - arm_returns["B"])),
    "C_minus_A": float(np.mean(arm_returns["C"] - arm_returns["A"])),
}
block_stats = {
    arm: {
        "signals": int(len(values)),
        "profitable_signals": int(np.sum(values > 0)),
        "mean_gross_return": float(np.mean(values)),
        "returns": [float(value) for value in values],
    }
    for arm, values in arm_returns.items()
}
decision = {
    "decision": "inconclusive",
    "evidence_stage": "single_incomplete_historical_fold_diagnostic",
    "data_pass": False,
    "same_samples": bool(same_samples),
    "pit_pass": False,
    "execution_pass": False,
    "c_minus_b_effect": effects["C_minus_B"],
    "c_minus_b_adjusted_ci": None,
    "c_minus_a_adjusted_ci": None,
    "drawdown_delta_pp": None,
    "cost_stress_pass": None,
    "fold_stability_pass": None,
    "fallback_share": float(summary["C"]["unknown_share"]),
    "complete_prediction_rows": int(len(complete)),
    "test_dates": int(summary["C"]["n_dates"]),
    "non_overlapping_signal_blocks": int(len(arm_returns["C"])),
    "reason": "P1 source coverage fails the registered 98% gate for liquidity and market/industry relations; industry publication vintage and weekly account ledger remain incomplete; one W8 fold and one seed cannot satisfy the registered decision gates.",
}
(result_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [
    "# 股票分类拆分 W8 P1 上下文单折预跑",
    "",
    "状态：证据不足。该预跑不满足完整实验决策门槛。早停期为 2026Q1，独立校准期为 2026Q2，测试期为 2026-07-01 至 2026-09-15。标签为 10 日可交易开盘到开盘收益；Top5 按每 10 个交易日取一次信号，表内为毛收益标签篮子，不是费用后连续账户净值。B/C 使用股票身份、分组、市值、流动性、上市年龄、板块、市场关系、行业相对收益及缺失指示特征。",
    "",
    f"A/B/C 完整共同预测样本为 {len(complete):,} 行、55 个测试日；每臂另有 348 个相同缺预测行。未知行业占比为 {summary['C']['unknown_share']:.2%}。",
    "",
    "| 方案 | RankIC均值 | Top5信号盈利块 | 六块平均毛收益 |",
    "|---|---:|---:|---:|",
]
for arm in ("A", "B", "C"):
    stats = block_stats[arm]
    lines.append(f"| {arm} | {summary[arm]['rank_ic_mean']:.4f} | {stats['profitable_signals']}/6 | {stats['mean_gross_return']:.2%} |")
lines.extend([
    "",
    f"配对均值：B−A {effects['B_minus_A']:.2%}，C−B {effects['C_minus_B']:.2%}，C−A {effects['C_minus_A']:.2%}。逐块差异和置信区间不用于显著性判断：仅 6 个信号块，且本预跑没有实际账户成交、费用、回撤及滑点压力账本。",
    "",
    "P1源缺失指示率：市值0.12%、流动性12.81%、市场关系28.33%、行业关系27.11%；其中2022/2023市场关系缺失率分别为51.43%/52.03%，行业关系为50.89%/51.44%，流动性为44.38%/35.01%。这未达到方案预先冻结的核心新增字段98%覆盖门槛。该单折只用于链路与训练可行性检查；行业来源缺少发布时点版本，样本量和执行账本也不足，结论为证据不足。不得据此决定拆分或淘汰拆分。",
    "",
    "A/B/C 的每日预测与标签在 `predictions.parquet`，预注册字段和限制见 `manifest.json`，机器判定见 `decision.json`。",
])
(result_dir / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"decision": decision, "block_stats": block_stats, "effects": effects}, ensure_ascii=False))
