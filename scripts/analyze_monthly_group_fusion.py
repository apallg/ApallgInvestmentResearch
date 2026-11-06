import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_monthly_group_research import OUTPUT, ROOT, SEEDS


def main():
    source = OUTPUT / "fusion_confirmation"
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    arms = ("D", "DM", "UM", "SL4_M")
    rows = []
    daily_rows = []
    for fold in (f"W{i}" for i in range(1, 9)):
        if not all((source / f"{fold}_{seed}_{arm}.parquet").exists() for seed in SEEDS for arm in arms):
            raise RuntimeError(f"融合矩阵尚未完成: {fold}")
        window = manifest["folds"][fold]
        calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
        dates = calendar[(calendar >= window["test"][0]) & (calendar <= window["test"][1])]
        signals = {day for day in dates[::10] if calendar.get_loc(day) + 11 < len(calendar)
                   and calendar[calendar.get_loc(day) + 11] <= dates[-1]}
        scores = {}
        label = None
        for seed in SEEDS:
            for arm in arms:
                frame = pd.read_parquet(source / f"{fold}_{seed}_{arm}.parquet").sort_index()
                if label is None:
                    label = frame["label"]
                if not frame.index.equals(label.index) or not np.allclose(frame["label"], label, rtol=0, atol=1e-7):
                    raise RuntimeError(f"融合预测股票日或标签不一致: {fold} {seed} {arm}")
                if not np.isfinite(frame["score"]).all():
                    raise RuntimeError(f"融合预测非有限值: {fold} {seed} {arm}")
                scores[(str(seed), arm)] = frame["score"]
        for arm in arms:
            scores[("ensemble", arm)] = pd.concat([scores[(str(seed), arm)] for seed in SEEDS], axis=1).mean(axis=1)
        for (seed, arm), score in scores.items():
            frame = pd.DataFrame({"score": score, "label": label})
            for date, part in frame.groupby(level="datetime", sort=True):
                daily_rows.append({"fold": fold, "seed": seed, "arm": arm, "date": str(date.date()),
                                   "stocks": len(part), "rank_ic": part["score"].corr(part["label"], method="spearman")})
            for date in signals:
                if date not in frame.index.get_level_values("datetime"):
                    continue
                part = frame.xs(date, level="datetime")
                ranked = part.reset_index().sort_values(["score", "instrument"], ascending=[False, True], kind="stable")
                rows.append({"fold": fold, "seed": seed, "arm": arm, "date": str(date.date()),
                             "top5_gross": float(ranked.head(5)["label"].mean()),
                             "top50_gross": float(ranked.head(50)["label"].mean()),
                             "universe_gross": float(part["label"].mean()), "stocks": len(part)})
    baskets = pd.DataFrame(rows)
    baskets.to_csv(source / "nonoverlap_10d_baskets.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(source / "daily_rank_ic.csv", index=False)
    primary = baskets.loc[baskets["fold"].isin(manifest["primary_folds"])]
    result = {"status": "complete", "research_only": True, "strict_pit_verified": False,
              "trading_execution_verified": False, "forward_26_weeks_complete": False,
              "folds": list(manifest["folds"]), "seeds": list(SEEDS), "comparisons": {}}
    for left, right in (("DM", "D"), ("SL4_M", "UM")):
        name = f"{left}-{right}"
        by_seed = {}
        by_fold = None
        for seed in ("ensemble", *map(str, SEEDS)):
            pair = primary.loc[(primary["seed"] == seed) & (primary["arm"].isin((left, right)))]
            pivot = pair.pivot(index=["fold", "date"], columns="arm", values="top5_gross")
            difference = pivot[left] - pivot[right]
            by_seed[seed] = float(difference.mean())
            if seed == "ensemble":
                by_fold = difference.groupby(level="fold").mean()
                main_difference = difference
        bootstrap = {}
        groups = [part.to_numpy() for _, part in main_difference.groupby(level="fold")]
        for width in (2, 4, 6):
            generator = np.random.default_rng(20261103 + width)
            values = np.empty(10000)
            for iteration in range(10000):
                selected = []
                for part in groups:
                    starts = generator.integers(0, len(part), size=int(np.ceil(len(part) / width)))
                    selected.append(np.concatenate([part[(np.arange(width) + start) % len(part)] for start in starts])[:len(part)])
                values[iteration] = np.concatenate(selected).mean()
            bootstrap[str(width)] = {"ci95": np.quantile(values, [0.025, 0.975]).tolist(),
                                     "family_adjusted_ci": np.quantile(values, [0.025 / 2, 1 - 0.025 / 2]).tolist()}
        extra = baskets.loc[(baskets["fold"] == "W8") & (baskets["seed"] == "ensemble")].pivot(
            index="date", columns="arm", values="top5_gross")
        result["comparisons"][name] = {"mean_top5_gross_difference": by_seed["ensemble"],
                                        "by_seed": by_seed, "by_fold": by_fold.to_dict(),
                                        "positive_full_folds": int((by_fold > 0).sum()),
                                        "positive_seeds": sum(by_seed[str(seed)] > 0 for seed in SEEDS),
                                        "w8_sensitivity_mean": float((extra[left] - extra[right]).mean()),
                                        "bootstrap": bootstrap}
    (source / "decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "comparisons": {key: {"difference": value["mean_top5_gross_difference"],
                       "positive_full_folds": value["positive_full_folds"], "positive_seeds": value["positive_seeds"],
                       "adjusted_ci": value["bootstrap"]["2"]["family_adjusted_ci"]}
                       for key, value in result["comparisons"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
