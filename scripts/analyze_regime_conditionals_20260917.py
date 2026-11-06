import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HMM = ROOT / "data/experiments/hmm_point_in_time_route_20260916.json"
CONTEXT = ROOT / "data/experiments/contextual_top5_weekly_corrected_20260916.json"
OUT = ROOT / "data/experiments/regime_conditionals_20260917.json"
REPORT = ROOT / "docs/regime_conditionals_20260917.md"


def _table(frame, column):
    rows = []
    for key, group in frame.groupby(column, dropna=False, observed=True):
        rows.append({"group": "NA" if pd.isna(key) else str(key), "n": int(len(group)), "trend_mean": float(group.net_trend.mean()), "rebound_mean": float(group.net_rebound.mean()), "advantage_trend_minus_rebound": float(group.advantage.mean()), "trend_best_share": float((group.best == "trend").mean()), "route_trend_share": float((group.route == "trend").mean())})
    return rows


def main():
    hmm = pd.DataFrame(json.loads(HMM.read_text(encoding="utf-8"))["weekly_trace"])
    context = pd.DataFrame(json.loads(CONTEXT.read_text(encoding="utf-8"))["weekly_trace"])
    hmm["date"] = pd.to_datetime(hmm["date"])
    context["date"] = pd.to_datetime(context["date"])
    cols = ["date", "vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3", "continuation3"]
    frame = hmm.merge(context[cols], on="date", how="left")
    frame["advantage"] = frame["net_trend"] - frame["net_rebound"]
    frame["best"] = np.where(frame["advantage"] > 0, "trend", "rebound")
    frame["route"] = frame["route_hmm_pit"]
    frame = frame.iloc[26:].copy()
    frame["vol_quartile"] = pd.qcut(frame["vol15"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    frame["breadth_tertile"] = pd.qcut(frame["B20"], 3, labels=["low", "mid", "high"], duplicates="drop")
    tables = {key: _table(frame, key) for key in ["state", "route", "vol_quartile", "breadth_tertile"]}
    result = {"status": "regime_conditional_expert_diagnostic", "oos_n": int(len(frame)), "tables": tables, "hmm_state_route_rule": "steady_up/volatile_up -> trend; other states -> rebound", "limitations": ["条件分组只使用33个路由OOS周，分组样本很小。", "net_trend/net_rebound是已成熟的下一持有期结果，只用于诊断，不作为当期特征。", "本分析不构成独立确认期或生产路由验证。"]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 市场状态与专家收益条件分解", "", "状态：`regime_conditional_expert_diagnostic`。只分析 HMM 点时路由的 33 个 OOS 周；趋势/反弹净收益是事后成熟结果，用于解释状态为何选错，不作为特征。"]
    titles = {"state": "HMM 状态", "route": "HMM 路由", "vol_quartile": "15日波动率分位", "breadth_tertile": "站上MA20比例分位"}
    for key, rows in tables.items():
        lines += ["", f"## {titles[key]}", "", "| 分组 | n | 趋势均值 | 反弹均值 | 趋势-反弹 | 趋势为事后最佳 | 实际路由选趋势 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['group']} | {row['n']} | {row['trend_mean']:.4f} | {row['rebound_mean']:.4f} | {row['advantage_trend_minus_rebound']:.4f} | {row['trend_best_share']:.1%} | {row['route_trend_share']:.1%} |")
    lines += ["", "## 结论", "", "- HMM 的 steady_up 状态样本中，反弹均值略高于趋势；把 up 状态固定映射为趋势并不成立。", "- volatile_up 样本显示趋势明显占优，但只有3周；volatile_down 只有2周且趋势也占优，说明“上涨/下跌”方向不能直接决定专家。", "- 波动率分位呈现低波偏反弹、高波偏趋势的线索，但只有33周，且中间分位方向反复，不能直接上线。", "- B20 中低分位更偏趋势、高分位两模型接近，单独使用宽度也不能稳定区分。", "- 下一阶段如果要训练路由，目标应是下一持有期 `趋势净收益-反弹净收益`，先用 vol15、dvol5 和一个宽度变量做正则线性模型；继续预测 HMM 状态没有收益目标上的必要。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
