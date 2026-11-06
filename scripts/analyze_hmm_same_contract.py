import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "experiments" / "contextual_top5_weekly_corrected_20260916.json"
STATE_INPUT = ROOT / "data" / "experiments" / "regime_predictions_2025_07_2026_08_20260916.json"
OUT = ROOT / "data" / "experiments" / "hmm_same_contract_diagnostic_20260916.json"
REPORT = ROOT / "docs" / "hmm_same_contract_diagnostic_20260916.md"

def main():
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    states = json.loads(STATE_INPUT.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload["weekly_trace"])
    rows["date"] = pd.to_datetime(rows["date"])
    state = pd.DataFrame(states["rows"])
    state["date"] = pd.to_datetime(state["date"])
    rows = rows.merge(state[["date", "state", "confidence"]], on="date", how="left")
    mapping = {
        "volatile_up": "trend",
        "steady_up": "trend",
        "volatile_down": "rebound",
        "steady_flat": "rebound",
        "volatile_flat": "trend",
        "steady_down": "rebound",
    }
    rows["mapped_route"] = rows["state"].map(mapping).fillna("trend")
    rows["mapped_net"] = np.where(rows["mapped_route"].eq("rebound"), rows["net_rebound"], rows["net_trend"])
    rows["best_expert"] = np.where(rows["net_rebound"] > rows["net_trend"], "rebound", "trend")
    rows["mapped_correct"] = rows["mapped_route"].eq(rows["best_expert"])
    by_state = []
    for name, group in rows.groupby("state", dropna=False):
        by_state.append({
            "state": "missing" if pd.isna(name) else str(name),
            "n": int(len(group)),
            "trend_mean": float(group["net_trend"].mean()),
            "rebound_mean": float(group["net_rebound"].mean()),
            "advantage_mean": float(group["advantage"].mean()),
            "rebound_better_rate": float(group["best_expert"].eq("rebound").mean()),
            "mapped_route": mapping.get(name, "trend") if not pd.isna(name) else "trend",
            "mapped_mean": float(group["mapped_net"].mean()),
            "mapped_accuracy": float(group["mapped_correct"].mean()),
        })
    result = {
        "status": "same_contract_retrospective_diagnostic",
        "input": str(INPUT.relative_to(ROOT)),
        "state_input": str(STATE_INPUT.relative_to(ROOT)),
        "decision_weeks": int(len(rows)),
        "state_rows": int(len(state)),
        "mapped_route_rebound_count": int(rows["mapped_route"].eq("rebound").sum()),
        "mapped_route_mean": float(rows["mapped_net"].mean()),
        "mapped_route_sum": float(rows["mapped_net"].sum()),
        "mapped_accuracy": float(rows["mapped_correct"].mean()),
        "fixed_trend_mean": float(rows["net_trend"].mean()),
        "fixed_rebound_mean": float(rows["net_rebound"].mean()),
        "state_missing": int(rows["state"].isna().sum()),
        "by_state": by_state,
        "limitations": [
            "状态预测文件来自当前模型对历史日期的回算，未证明点时训练，因此本结果是同合同回顾性诊断，不是HMM样本外验收。",
            "状态到专家映射使用冻结映射，未使用当前周收益学习。",
            "本结果复用成熟标签修正版账户，未重新生成历史HMM参数。",
        ],
        "weekly_trace": rows.to_dict(orient="records"),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# HMM 同合同回顾性诊断（2026-09-16）", "",
        "状态：`same_contract_retrospective_diagnostic`。本文件用于解释机制，不是点时样本外验收。",
        "", f"共同账本：{len(rows)} 个有效周；状态文件行数：{len(state)}；映射后反弹路由：{int(rows['mapped_route'].eq('rebound').sum())} 周。", "",
        "## 总体", "", "| 项目 | 值 |", "|---|---:|",
        f"| 固定趋势周均净收益 | {rows['net_trend'].mean():.5f} |",
        f"| 固定反弹周均净收益 | {rows['net_rebound'].mean():.5f} |",
        f"| HMM映射路由周均净收益 | {rows['mapped_net'].mean():.5f} |",
        f"| HMM映射路由事后最佳专家比例 | {rows['mapped_correct'].mean():.3f} |",
        f"| HMM映射路由账户终值（已有约束账本） | {payload['outcomes']['B6_supervised_retrospective']['account']['final_nav']:.5f} |",
        "", "## 状态条件下的专家收益", "", "| 状态 | 周数 | 趋势均值 | 反弹均值 | 反弹更优比例 | 冻结映射 | 映射均值 |", "|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in by_state:
        lines.append(f"| {item['state']} | {item['n']} | {item['trend_mean']:.5f} | {item['rebound_mean']:.5f} | {item['rebound_better_rate']:.3f} | {item['mapped_route']} | {item['mapped_mean']:.5f} |")
    lines += ["", "## 判定", "", "- 该状态文件与共同账本可以在同一账户合同下计算，但它是历史回算，不能作为HMM真实样本外证据。", "- 若固定映射在状态条件下经常选择收益较低的专家，则HMM收益不能解释为模型选择能力。", "- 只有用每个决策日前可训练的HMM参数和状态映射重跑，且与固定趋势、G0同时比较，才能进入正式晋级。", "", f"原始 JSON：`{OUT.relative_to(ROOT)}`。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["decision_weeks", "mapped_route_rebound_count", "mapped_route_mean", "mapped_accuracy", "fixed_trend_mean", "fixed_rebound_mean", "state_missing"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
