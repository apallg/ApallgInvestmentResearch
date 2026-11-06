import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    execution = json.loads((output / "execution_readiness.json").read_text(encoding="utf-8"))
    fusion_path = output / "same_date_fusion/same_date_fusion_42.json"
    fusion = json.loads(fusion_path.read_text(encoding="utf-8")) if fusion_path.exists() else None
    fusion_complete = bool(fusion and fusion.get("status") == "diagnostic_same_date_safe_retrain"
                           and (output / "same_date_fusion/same_date_lgb_42.txt").exists()
                           and (output / "same_date_fusion/same_date_lstm_42.pkl").exists())
    complete = decision["status"] == "complete" and decision["seed_count"] == 3
    gates = {}
    for name, result in decision["comparisons"].items():
        lower = result["bootstrap"].get("2", {}).get("family_adjusted_ci", [None])[0]
        full_folds_positive = sum(result["by_fold"][f"W{number}"] > 0 for number in range(1, 8))
        screening = (result["mean_top5_gross_difference"] >= 0.001
                     and full_folds_positive >= 5
                     and sum(value > 0 for value in result["by_seed"].values()) >= 2)
        gates[name] = {
            "historical_screening_gate": "passed" if complete and screening else "failed" if complete else "pending",
            "positive_full_folds": full_folds_positive,
            "adjusted_interval_positive": lower is not None and lower > 0,
            "trading_gate": "unavailable",
            "deployment_eligible": False,
        }
    evidence = {
        "manifest": "manifest.json",
        "old_artifact_audit": "old_artifact_audit.json",
        "execution_readiness": "execution_readiness.json",
        "feature_probe": "feature_probe.json",
        "training": "W{1..8}_{42,2026,3407}_training.json",
        "predictions": "W{1..8}_{42,2026,3407}_{D,L,W,M,U,G6,GI,GS}.parquet",
        "daily_metrics": "daily_metrics.csv",
        "nonoverlap_baskets": "nonoverlap_10d_baskets.csv",
        "paired_statistics": "decision.json",
        "group_contributions": "group_contributions_summary.csv",
    }
    if fusion_complete:
        evidence["same_date_fusion"] = "same_date_fusion/same_date_fusion_42.json"
        evidence["same_date_common_predictions"] = "same_date_fusion/same_date_pair_42.parquet"
    assessment = {
        "status": "historical_research_complete" if complete else "partial_engineering_run",
        "source_panel_end": manifest["source_panel_end"],
        "source_is_current_generation": manifest["source_is_current_generation"],
        "strict_pit_verified": False,
        "trading_execution_verified": False,
        "old_new_fusion_comparison_complete": fusion_complete,
        "old_new_fusion_diagnostic": fusion if fusion_complete else None,
        "forward_26_weeks_complete": False,
        "gates": gates,
        "execution_evidence": execution,
        "evidence_index": evidence,
    }
    (output / "assessment.json").write_text(json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 三项研究结论与证据", "", f"状态：{assessment['status']}。历史 LightGBM 预测研究与融合模型、交易账户和前瞻验证分别判断。", "",
             f"研究输入截至 {manifest['source_panel_end']}；当前代际：{manifest['source_is_current_generation']}；严格历史 PIT：未认证。", "",
             "| 对照 | 完整折正向数／7 | 历史筛选门槛 | Bonferroni 区间下界 > 0 | 交易与上线 |", "|---|---:|---|---|---|"]
    for name, value in gates.items():
        lines.append(f"| {name} | {value['positive_full_folds']}/7 | {value['historical_screening_gate']} | {value['adjusted_interval_positive']} | 无合格成交证据 |")
    lines.extend(["", "W8 仅有 5 个非重叠篮子，属于部分折；筛选门槛按 W1–W7 七个完整折计数。原始 `report.md` 的“正向折数”包含 W8，仅用于展示。"])
    if fusion_complete:
        lines.append("")
        lines.append(f"旧模型同日期诊断：{fusion['common_days']} 个共同交易日，旧／新 RankIC {fusion['old_common_rank_ic']:.4f}／{fusion['new_common_rank_ic']:.4f}；每日重叠 Top5 十日标签毛收益差 {fusion['daily_top5_gross_difference'] * 10000:+.1f}bp，但 {fusion['nonoverlap_10d_blocks']} 个非重叠篮子均值差 {fusion['nonoverlap_top5_gross_difference'] * 10000:+.1f}bp。旧模型标签边界交叠、输入代际不同，不能归因或作为上线证据。")
    lines.extend(["", "旧模型工件的训练和验证标签跨边界，且缺少当前 FundVal 输入契约；其已保存预测仅作诊断。更新训练日期的融合比较尚未完成。",
                  "历史 ST/证券状态快照不足，不能计算可靠的逐笔成交、成本、净收益或回撤。候选尚未冻结，26 周前瞻观察未开始。",
                  "收益表中的十日数值均为未来标签篮子的毛收益，不能视为账户净收益或上线证明。", "", "## 证据索引", ""])
    lines.extend(f"- {name}: `{path}`" for name, path in evidence.items())
    (output / "assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": assessment["status"], "assessment": str(output / "assessment.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
