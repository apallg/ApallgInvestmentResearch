import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_multiscale_group_research import COMPARISONS, OUTPUT, ROOT


def main(args):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    arms = ("D", "L", "W", "M", "U", "G6", "GI", "GS")
    available = []
    for fold in args.folds:
        if all((OUTPUT / f"{fold}_{seed}_{arm}.parquet").exists() for seed in args.seeds for arm in arms):
            available.append(fold)
    if not available:
        raise RuntimeError("尚无完成的共同折/种子/方案预测")
    daily_rows = []
    block_rows = []
    for fold in available:
        window = manifest["folds"][fold]
        dates = calendar[(calendar >= pd.Timestamp(window["test"][0])) & (calendar <= pd.Timestamp(window["test"][1]))]
        signals = set()
        for day in dates[::10]:
            location = calendar.get_loc(day)
            if location + 11 < len(calendar) and calendar[location + 11] <= dates[-1]:
                signals.add(day)
        scores = {}
        label = None
        for seed in args.seeds:
            for arm in arms:
                path = OUTPUT / f"{fold}_{seed}_{arm}.parquet"
                frame = pd.read_parquet(path).sort_index()
                if label is None:
                    label = frame["label"]
                if not frame.index.equals(label.index) or not np.allclose(frame["label"], label, equal_nan=True, rtol=0, atol=1e-7):
                    raise RuntimeError(f"股票日或标签不一致: {fold} {seed} {arm}")
                if frame["score"].isna().any() or not np.isfinite(frame["score"]).all():
                    raise RuntimeError(f"预测分数有缺失: {fold} {seed} {arm}")
                scores[(str(seed), arm)] = frame["score"]
        for arm in arms:
            scores[("ensemble", arm)] = pd.concat([scores[(str(seed), arm)] for seed in args.seeds], axis=1).mean(axis=1)
        for (seed, arm), score in scores.items():
            together = pd.concat({"score": score, "label": label}, axis=1)
            for day, frame in together.groupby(level="datetime", sort=True):
                if len(frame) < 50:
                    raise RuntimeError(f"共同选股截面不足50只: {day}")
                ranked = frame.reset_index().sort_values(["score", "instrument"], ascending=[False, True], kind="stable")
                ic = frame["score"].corr(frame["label"], method="spearman")
                row = {"fold": fold, "seed": seed, "arm": arm, "date": day.strftime("%Y-%m-%d"),
                       "stocks": len(frame), "rank_ic": ic, "top5_gross": ranked.head(5)["label"].mean(),
                       "top50_gross": ranked.head(50)["label"].mean(), "universe_gross": frame["label"].mean()}
                daily_rows.append(row)
                if day in signals:
                    block_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    blocks = pd.DataFrame(block_rows)
    daily.to_csv(OUTPUT / "daily_metrics.csv", index=False)
    blocks.to_csv(OUTPUT / "nonoverlap_10d_baskets.csv", index=False)
    summary = {}
    ensemble = blocks.loc[blocks["seed"] == "ensemble"]
    for arm in arms:
        part = ensemble.loc[ensemble["arm"] == arm]
        all_days = daily.loc[(daily["seed"] == "ensemble") & (daily["arm"] == arm)]
        summary[arm] = {"signal_blocks": len(part), "mean_top5_gross": float(part["top5_gross"].mean()),
                        "mean_top5_excess": float((part["top5_gross"] - part["universe_gross"]).mean()),
                        "mean_rank_ic": float(all_days["rank_ic"].mean()),
                        "profitable_basket_share": float((part["top5_gross"] > 0).mean())}
    comparisons = {}
    count = len(COMPARISONS)
    for left, right in COMPARISONS:
        left_rows = ensemble.loc[ensemble["arm"] == left].set_index(["fold", "date"])["top5_gross"]
        right_rows = ensemble.loc[ensemble["arm"] == right].set_index(["fold", "date"])["top5_gross"]
        if not left_rows.index.equals(right_rows.index):
            raise RuntimeError(f"配对评价日不一致: {left} {right}")
        difference = left_rows - right_rows
        by_fold = difference.groupby(level="fold").mean()
        seed_differences = {}
        for seed in args.seeds:
            left_seed = blocks.loc[(blocks["arm"] == left) & (blocks["seed"] == str(seed))].set_index(["fold", "date"])["top5_gross"]
            right_seed = blocks.loc[(blocks["arm"] == right) & (blocks["seed"] == str(seed))].set_index(["fold", "date"])["top5_gross"]
            seed_differences[str(seed)] = float((left_seed - right_seed).mean())
        bootstrap = {}
        for width in (2, 4, 6):
            groups = [part.to_numpy() for _, part in difference.groupby(level="fold")]
            if any(len(part) < width for part in groups):
                bootstrap[str(width)] = {"status": "insufficient_blocks"}
                continue
            generator = np.random.default_rng(20261103 + width)
            resampled = np.empty(10000)
            for iteration in range(10000):
                values = []
                for part in groups:
                    starts = generator.integers(0, len(part), size=int(np.ceil(len(part) / width)))
                    selected = np.concatenate([part[(np.arange(width) + start) % len(part)] for start in starts])[:len(part)]
                    values.append(selected)
                resampled[iteration] = np.concatenate(values).mean()
            bootstrap[str(width)] = {"ci95": np.quantile(resampled, [0.025, 0.975]).tolist(),
                                     "family_adjusted_ci": np.quantile(resampled, [0.025 / count, 1 - 0.025 / count]).tolist()}
        comparisons[f"{left}-{right}"] = {"mean_top5_gross_difference": float(difference.mean()),
                                            "positive_folds": int((by_fold > 0).sum()),
                                            "by_fold": by_fold.to_dict(), "by_seed": seed_differences,
                                            "bootstrap": bootstrap}
    result = {"status": "complete" if len(available) == 8 and len(args.seeds) == 3 else "partial",
              "research_only": True, "strict_pit_verified": False, "execution_verified": False,
              "deployment_eligible": False, "completed_folds": available,
              "seed_count": len(args.seeds), "summary": summary, "comparisons": comparisons,
              "conclusion_allowed": len(available) == 8 and len(args.seeds) == 3,
              "source_panel_end": manifest["source_panel_end"]}
    (OUTPUT / "decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = ["# 多周期与分组模型历史研究", "", f"状态：{result['status']}；仅历史研究，尚无成交账户证据。", "",
              "| 方案 | 每日RankIC | Top5十日篮子平均毛收益 | 非重叠篮子数 |",
              "|---|---:|---:|---:|"]
    report.extend(f"| {arm} | {value['mean_rank_ic']:.4f} | {value['mean_top5_gross']:.2%} | {value['signal_blocks']} |" for arm, value in summary.items())
    report.extend(["", "| 对照 | Top5十日毛收益差 | 95%区间 | 多重校正区间 | 正向折数 |",
                   "|---|---:|---:|---:|---:|"])
    for name, value in comparisons.items():
        stats = value["bootstrap"].get("2", {})
        ci = stats.get("ci95")
        adjusted = stats.get("family_adjusted_ci")
        ci_text = f"[{ci[0]:.2%}, {ci[1]:.2%}]" if ci else "样本不足"
        adjusted_text = f"[{adjusted[0]:.2%}, {adjusted[1]:.2%}]" if adjusted else "样本不足"
        report.append(f"| {name} | {value['mean_top5_gross_difference']:.2%} | {ci_text} | {adjusted_text} | {value['positive_folds']} |")
    report.extend(["", f"原始研究面板截至{manifest['source_panel_end']}，历史行业发布版本未认证；收益为未来标签篮子，不是扣费账户净收益。",
                   "所有历史窗口已被既有研究接触，不能作为前瞻确认。"])
    if result["status"] == "partial":
        report.append("当前仅为工程预跑；折数或种子不足，不能据此判断改善。")
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "folds": available, "report": str(OUTPUT / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    args = parser.parse_args()
    if set(args.folds) - {f"W{i}" for i in range(1, 9)} or set(args.seeds) - {42, 2026, 3407}:
        raise ValueError("折或种子不在冻结契约内")
    main(args)
