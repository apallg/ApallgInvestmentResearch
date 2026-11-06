import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    if decision["status"] != "complete":
        raise RuntimeError("分组贡献只能在八折三种子齐全后计算")
    baskets = pd.read_csv(output / "nonoverlap_10d_baskets.csv")
    dates = baskets.loc[(baskets["seed"] == "ensemble") & (baskets["arm"] == "U"), ["fold", "date"]]
    seeds = (42, 2026, 3407)
    rows = []
    for fold, days in dates.groupby("fold"):
        wanted = set(pd.to_datetime(days["date"]))
        for arm, group_column in (("U", "coarse_group"), ("G6", "coarse_group"),
                                  ("GI", "industry_group"), ("GS", "style_group")):
            frames = [pd.read_parquet(output / f"{fold}_{seed}_{arm}.parquet") for seed in seeds]
            scores = pd.concat([frame["score"] for frame in frames], axis=1).mean(axis=1)
            panel = frames[0][["label", group_column, "fallback"]].copy()
            panel["score"] = scores
            for day, section in panel.groupby(level="datetime"):
                if day not in wanted:
                    continue
                selected = section.reset_index().sort_values(["score", "instrument"], ascending=[False, True], kind="stable").head(5)
                for group, part in selected.groupby(group_column, dropna=False):
                    rows.append({"fold": fold, "date": str(day.date()), "arm": arm,
                                 "group": int(group) if pd.notna(group) else -1,
                                 "selected_count": len(part), "contribution_to_top5_gross": part["label"].sum() / 5,
                                 "mean_selected_label": part["label"].mean(),
                                 "fallback_selected": int(part["fallback"].sum())})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("没有非重叠篮子可供分组归因")
    frame.to_csv(output / "group_contributions_daily.csv", index=False)
    summary = frame.groupby(["arm", "group"], as_index=False).agg(
        selected_count=("selected_count", "sum"),
        gross_contribution_sum=("contribution_to_top5_gross", "sum"),
        fallback_selected=("fallback_selected", "sum"))
    summary["signal_blocks"] = len(dates)
    summary["mean_gross_contribution_per_basket"] = summary["gross_contribution_sum"] / len(dates)
    summary.to_csv(output / "group_contributions_summary.csv", index=False)
    print(json.dumps({"status": "complete", "rows": len(frame), "summary": str(output / "group_contributions_summary.csv")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
