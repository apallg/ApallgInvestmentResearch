import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_monthly_group_research import OUTPUT


def main():
    decision = json.loads((OUTPUT / "decision.json").read_text(encoding="utf-8"))
    complete = decision["status"] == "complete"
    gates = {}
    for name, result in decision["comparisons"].items():
        lower = result["bootstrap"].get("2", {}).get("family_adjusted_ci", [None])[0]
        stable = (result["mean_top5_gross_difference"] >= 0.001 and result["positive_full_folds"] >= 5
                  and sum(value > 0 for value in result["by_seed"].values()) >= 2)
        gates[name] = {"historical_screening_gate": "passed" if complete and stable and not name.endswith("_interaction")
                       else "failed" if complete else "pending",
                       "positive_full_folds": result["positive_full_folds"],
                       "adjusted_interval_positive": lower is not None and lower > 0,
                       "deployment_eligible": False}
    report = {"status": "historical_research_complete" if complete else "partial_engineering_run",
              "strict_pit_verified": False, "trading_execution_verified": False, "forward_26_weeks_complete": False,
              "gates": gates, "candidate_comparisons": [name for name, value in gates.items()
                                                  if value["historical_screening_gate"] == "passed"],
              "evidence_index": ["manifest.json", "decision.json", "report.md", "daily_metrics.csv",
                                 "nonoverlap_10d_baskets.csv", "W*_input_audit.json", "W*_*_training.json"]}
    (OUTPUT / "assessment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 月K与分类方案证据判断", "", f"状态：{report['status']}。主要判断按七个完整折；W8只作敏感性。", "",
             "| 对照 | 候选筛选 | 校正区间下界>0 |", "|---|---|---|"]
    lines += [f"| {name} | {value['historical_screening_gate']} | {value['adjusted_interval_positive']} |"
              for name, value in gates.items()]
    lines += ["", "所有历史窗口已被研究接触，行业发布时间版本未认证；标签篮子毛收益不能当作成交账户净收益。",
              "候选须完成融合模型、等训练时间预算统一集成和前瞻确认，不能据此切换生产模型。"]
    (OUTPUT / "assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "candidates": report["candidate_comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
