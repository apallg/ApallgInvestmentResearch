import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_monthly_group_research import ARMS, COMPARISONS, OUTPUT, ROOT, SEEDS


def analyze(args):
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    available = [fold for fold in args.folds if all((OUTPUT / f"{fold}_{seed}_{arm}.parquet").exists()
                                                  for seed in args.seeds for arm in ARMS)]
    if not available:
        raise RuntimeError("尚无共同折、种子、方案预测")
    daily_rows = []
    block_rows = []
    for fold in available:
        window = manifest["folds"][fold]
        dates = calendar[(calendar >= window["test"][0]) & (calendar <= window["test"][1])]
        signals = {day for day in dates[::10] if calendar.get_loc(day) + 11 < len(calendar)
                   and calendar[calendar.get_loc(day) + 11] <= dates[-1]}
        scores = {}
        label = None
        for seed in args.seeds:
            for arm in ARMS:
                frame = pd.read_parquet(OUTPUT / f"{fold}_{seed}_{arm}.parquet").sort_index()
                if label is None:
                    label = frame["label"]
                if not frame.index.equals(label.index) or not np.allclose(frame["label"], label, equal_nan=True, rtol=0, atol=1e-7):
                    raise RuntimeError(f"共同股票日或标签不一致: {fold} {seed} {arm}")
                if frame["score"].isna().any() or not np.isfinite(frame["score"]).all():
                    raise RuntimeError(f"预测缺失: {fold} {seed} {arm}")
                scores[(str(seed), arm)] = frame["score"]
        for arm in ARMS:
            scores[("ensemble", arm)] = pd.concat([scores[(str(seed), arm)] for seed in args.seeds], axis=1).mean(axis=1)
        for (seed, arm), score in scores.items():
            frame = pd.concat({"score": score, "label": label}, axis=1)
            for day, section in frame.groupby(level="datetime", sort=True):
                ranked = section.reset_index().sort_values(["score", "instrument"], ascending=[False, True], kind="stable")
                row = {"fold": fold, "seed": seed, "arm": arm, "date": str(day.date()), "stocks": len(section),
                       "rank_ic": section["score"].corr(section["label"], method="spearman"),
                       "top5_gross": ranked.head(5)["label"].mean(),
                       "top50_gross": ranked.head(50)["label"].mean(), "universe_gross": section["label"].mean()}
                daily_rows.append(row)
                if day in signals:
                    block_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    blocks = pd.DataFrame(block_rows)
    daily.to_csv(OUTPUT / "daily_metrics.csv", index=False)
    blocks.to_csv(OUTPUT / "nonoverlap_10d_baskets.csv", index=False)
    primary = blocks.loc[blocks["fold"].isin(manifest["primary_folds"])]
    ensemble = primary.loc[primary["seed"] == "ensemble"]
    summary = {}
    for arm in ARMS:
        part = ensemble.loc[ensemble["arm"] == arm]
        all_days = daily.loc[(daily["fold"].isin(manifest["primary_folds"])) &
                             (daily["seed"] == "ensemble") & (daily["arm"] == arm)]
        summary[arm] = {"signal_blocks": len(part), "mean_top5_gross": float(part["top5_gross"].mean()),
                        "mean_top5_excess": float((part["top5_gross"] - part["universe_gross"]).mean()),
                        "mean_rank_ic": float(all_days["rank_ic"].mean())}
    comparisons = {}
    definitions = {f"{left}-{right}": ((left, 1), (right, -1)) for left, right in COMPARISONS}
    for name in ("SL4", "PB4", "IC6", "GS8"):
        definitions[f"{name}_interaction"] = ((f"{name}_M", 1), ("UM", -1), (f"{name}_0", -1), ("U0", 1))
    if len(definitions) != 16:
        raise RuntimeError("预注册比较家族不是16项")
    for name, terms in definitions.items():
        pairs = []
        seed_means = {}
        for seed in ("ensemble", *map(str, args.seeds)):
            series = None
            for arm, sign in terms:
                value = primary.loc[(primary["seed"] == seed) & (primary["arm"] == arm)].set_index(["fold", "date"])["top5_gross"]
                if series is None:
                    series = sign * value
                else:
                    if not series.index.equals(value.index):
                        raise RuntimeError(f"配对评价日不同: {name}")
                    series = series + sign * value
            if seed == "ensemble":
                pairs = series
            else:
                seed_means[seed] = float(series.mean())
        by_fold = pairs.groupby(level="fold").mean()
        bootstrap = {}
        for width in (2, 4, 6):
            groups = [part.to_numpy() for _, part in pairs.groupby(level="fold")]
            if any(len(part) < width for part in groups):
                bootstrap[str(width)] = {"status": "insufficient_blocks"}
                continue
            generator = np.random.default_rng(20261103 + width)
            resampled = np.empty(10000)
            for iteration in range(10000):
                samples = []
                for part in groups:
                    starts = generator.integers(0, len(part), size=int(np.ceil(len(part) / width)))
                    selected = np.concatenate([part[(np.arange(width) + start) % len(part)] for start in starts])[:len(part)]
                    samples.append(selected)
                resampled[iteration] = np.concatenate(samples).mean()
            bootstrap[str(width)] = {"ci95": np.quantile(resampled, [0.025, 0.975]).tolist(),
                                     "family_adjusted_ci": np.quantile(resampled, [0.025 / 16, 1 - 0.025 / 16]).tolist()}
        sensitivity = blocks.loc[(blocks["fold"] == "W8") & (blocks["seed"] == "ensemble")]
        extra = None
        if not sensitivity.empty:
            value = None
            for arm, sign in terms:
                part = sensitivity.loc[sensitivity["arm"] == arm].set_index("date")["top5_gross"]
                value = sign * part if value is None else value + sign * part
            extra = float(value.mean())
        comparisons[name] = {"mean_top5_gross_difference": float(pairs.mean()),
                             "positive_full_folds": int((by_fold > 0).sum()), "by_fold": by_fold.to_dict(),
                             "by_seed": seed_means, "w8_sensitivity_mean": extra, "bootstrap": bootstrap}
    complete = set(available) == {f"W{i}" for i in range(1, 9)} and set(args.seeds) == set(SEEDS)
    report = {"status": "complete" if complete else "partial", "research_only": True,
              "strict_pit_verified": False, "execution_verified": False, "deployment_eligible": False,
              "completed_folds": available, "seed_count": len(args.seeds), "summary": summary,
              "comparisons": comparisons, "source_panel_end": manifest["source_panel_end"]}
    (OUTPUT / "decision.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 月K与替代分类研究", "", f"状态：{report['status']}；主要评价仅 W1–W7，W8 单列敏感性。", "",
             "| 方案 | 十日Top5标签毛收益 | 每日RankIC |", "|---|---:|---:|"]
    lines += [f"| {arm} | {summary[arm]['mean_top5_gross']:.2%} | {summary[arm]['mean_rank_ic']:.4f} |" for arm in ARMS]
    lines += ["", "| 比较 | 差值 | 16项校正区间 | 完整折为正 | W8敏感性 |", "|---|---:|---:|---:|---:|"]
    for name, value in comparisons.items():
        interval = value["bootstrap"]["2"].get("family_adjusted_ci")
        ci = f"[{interval[0]:.2%}, {interval[1]:.2%}]" if interval else "样本不足"
        extra = f"{value['w8_sensitivity_mean']:.2%}" if value["w8_sensitivity_mean"] is not None else "—"
        lines.append(f"| {name} | {value['mean_top5_gross_difference']:.2%} | {ci} | {value['positive_full_folds']}/7 | {extra} |")
    lines += ["", "历史研究窗口已经被接触，行业来源发布时间未认证，收益不是交易账户净收益。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "folds": available, "report": str(OUTPUT / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=[f"W{i}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if set(args.folds) - {f"W{i}" for i in range(1, 9)} or set(args.seeds) - set(SEEDS):
        raise ValueError("折或种子不在冻结契约内")
    analyze(args)
