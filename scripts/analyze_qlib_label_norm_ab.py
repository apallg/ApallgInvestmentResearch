import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "experiments" / "qlib_label_norm_ab_20261015"
REPORT_PATH = OUT_DIR / "report.json"
MLRUNS = ROOT / "mlruns"
NORMS = ("csrank", "none", "cszscore")
WINDOWS = ("w1", "w2", "w3", "w4", "w5")


def block_bootstrap(folds, n_boot=10000, block_len=20, seed=20261016):
    rng = np.random.default_rng(seed)
    values = []
    for fold in folds:
        if not len(fold):
            continue
        if isinstance(fold[0], dict):
            fold = [row["delta"] if "delta" in row else row["v"] for row in fold]
        values.append(np.asarray(fold, dtype=float))
    if not values:
        return {"mean": None, "ci95": [None, None], "positive_day_ratio": None}
    sizes = [len(x) for x in values]
    total = sum(sizes)
    mean = sum(float(x.sum()) for x in values) / total
    draws = np.empty(n_boot)
    for i in range(n_boot):
        total_sum = 0.0
        for x in values:
            picked = []
            while len(picked) < len(x):
                start = int(rng.integers(0, max(1, len(x) - block_len + 1)))
                picked.extend(x[start:min(start + block_len, len(x))].tolist())
            total_sum += float(np.sum(picked[:len(x)]))
        draws[i] = total_sum / total
    return {
        "mean": mean,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "positive_day_ratio": float(np.mean(np.concatenate(values) > 0)),
        "n_days": total,
        "block_len_days": block_len,
        "bootstrap_replicates": n_boot,
    }


def find_report_path(window, norm):
    exp_name = f"label_norm_ab_20261015_{window}_{norm}"
    for exp_dir in MLRUNS.iterdir():
        meta = exp_dir / "meta.yaml"
        if not meta.is_file() or not re.search(rf"(?m)^name:\s*{re.escape(exp_name)}\s*$", meta.read_text(encoding="utf-8")):
            continue
        matches = list(exp_dir.glob("*/artifacts/portfolio_analysis/report_normal_1day.pkl"))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"找不到唯一回测产物: {exp_name}")


def main():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    reports = {}
    for window in WINDOWS:
        reports[window] = {}
        for norm in NORMS:
            path = find_report_path(window, norm)
            frame = pd.read_pickle(path)
            frame.index = pd.to_datetime(frame.index)
            reports[window][norm] = frame

    costed = {}
    for norm in NORMS:
        by_window = {}
        for window in WINDOWS:
            frame = reports[window][norm]
            daily = frame.iloc[1:]
            excess = daily["return"].astype(float) - daily["bench"].astype(float)
            by_window[window] = {
                "n_days": int(len(excess)),
                "mean_daily_excess_after_cost": float(excess.mean()),
                "information_ratio_daily": float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) else None,
                "cumulative_return": float(frame["account"].iloc[-1] / frame["account"].iloc[0] - 1),
                "benchmark_cumulative_return": float((1 + daily["bench"].astype(float)).prod() - 1),
                "max_drawdown": float((frame["account"] / frame["account"].cummax() - 1).min()),
                "turnover": float(frame["total_turnover"].iloc[-1]),
                "cost": float(frame["total_cost"].iloc[-1]),
                "artifact": str(find_report_path(window, norm).relative_to(ROOT)),
            }
        costed[norm] = by_window

    paired = {}
    for norm in ("none", "cszscore"):
        folds = []
        window_delta = {}
        for window in WINDOWS:
            base = reports[window]["csrank"].iloc[1:]
            arm = reports[window][norm].iloc[1:]
            base_excess = base["return"].astype(float) - base["bench"].astype(float)
            arm_excess = arm["return"].astype(float) - arm["bench"].astype(float)
            aligned = pd.concat([arm_excess.rename("arm"), base_excess.rename("base")], axis=1).dropna()
            delta = (aligned["arm"] - aligned["base"]).to_numpy()
            folds.append(delta)
            window_delta[window] = {
                "mean_daily_excess_delta": float(np.mean(delta)),
                "arm_cumulative_return": costed[norm][window]["cumulative_return"],
                "csrank_cumulative_return": costed["csrank"][window]["cumulative_return"],
                "cumulative_return_delta": float(costed[norm][window]["cumulative_return"] - costed["csrank"][window]["cumulative_return"]),
            }
        paired[norm] = {
            "daily_excess_delta_after_cost": block_bootstrap(folds),
            "windows_won_by_mean_daily_excess": sum(v["mean_daily_excess_delta"] > 0 for v in window_delta.values()),
            "window_count": len(window_delta),
            "by_window": window_delta,
        }

    rows_by_window = {
        window: {
            norm: json.loads((OUT_DIR / f"{window}_{norm}_daily.json").read_text(encoding="utf-8"))
            for norm in NORMS
        }
        for window in WINDOWS
    }
    all_dates = sorted({row["date"] for window in WINDOWS for row in rows_by_window[window]["csrank"]})
    nonoverlap_dates = set(all_dates[::10])
    nonoverlap = {
        "sampling": "在全部连续 OOS 交易日中每隔 10 个交易日取一个信号日；每个样本对应 10 个交易日的 tradable_open_to_open 标签收益，避免标签收益窗口相互重叠。",
        "signal_dates": sorted(nonoverlap_dates),
        "by_norm": {},
        "comparisons_vs_csrank": {},
    }
    for norm in NORMS:
        nonoverlap["by_norm"][norm] = {}
        for top_n in (5, 10, 20, 50):
            metric = f"top{top_n}_excess"
            values = [
                row[metric]
                for window in WINDOWS
                for row in rows_by_window[window][norm]
                if row["date"] in nonoverlap_dates
            ]
            nonoverlap["by_norm"][norm][metric] = {"mean": float(np.mean(values)), "n_signal_dates": len(values)}
    for norm in ("none", "cszscore"):
        nonoverlap["comparisons_vs_csrank"][norm] = {}
        for top_n in (5, 10, 20, 50):
            metric = f"top{top_n}_excess"
            folds = []
            by_window = {}
            for window in WINDOWS:
                arm_map = {row["date"]: row for row in rows_by_window[window][norm]}
                base_map = {row["date"]: row for row in rows_by_window[window]["csrank"]}
                dates = sorted(nonoverlap_dates.intersection(arm_map).intersection(base_map))
                delta = np.asarray([arm_map[date][metric] - base_map[date][metric] for date in dates], dtype=float)
                folds.append(delta)
                by_window[window] = {
                    "n_signal_dates": len(dates),
                    "mean_excess_delta": float(delta.mean()) if len(delta) else None,
                }
            stat = block_bootstrap(folds, block_len=2, seed=20261016 + top_n)
            stat["positive_signal_date_ratio"] = stat.pop("positive_day_ratio")
            stat["block_len_signal_dates"] = 2
            stat["by_window"] = by_window
            nonoverlap["comparisons_vs_csrank"][norm][metric] = stat

    report["costed_backtest_analysis"] = {
        "metric": "daily portfolio return minus benchmark return, after costs; equal-date paired comparison",
        "role": "exploratory diagnostic only because the simulated TopkDropout strategy rebalances daily while the label horizon is 10 trading days",
        "per_arm_by_window": costed,
        "comparisons_vs_csrank": paired,
        "method": "20-trading-day moving-block bootstrap stratified within each non-overlapping OOS window, 10000 replicates; CI is exploratory and does not correct for testing multiple candidate labels.",
    }
    report["non_overlapping_horizon_analysis"] = nonoverlap
    report["primary_outcome"] = {
        "metric": "Top5 mean 10-trading-day label excess over the same-day universe mean, sampled at globally spaced 10-trading-day signal dates",
        "primary_comparison": "none - csrank",
        "n_signal_dates": len(nonoverlap_dates),
        "result": nonoverlap["comparisons_vs_csrank"]["none"]["top5_excess"],
        "interpretation": "confidence interval includes zero; this experiment does not establish a reliable winner between none and csrank",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# Qlib 标签归一化对照实验结果",
        "",
        "## 实验结论摘要",
        "",
        "| 窗口 | 日期 | csrank 成本后收益 | none 成本后收益 | cszscore 成本后收益 | none-csrank Top50 10日超额差 | none-csrank 成本后日均超额差 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        period = report["contract"]["windows"][window]["test"]
        row = paired["none"]["by_window"][window]
        vals = [costed[n][window]["cumulative_return"] for n in NORMS]
        top50_delta = report["results"][window]["none"]["top50_excess_mean"] - report["results"][window]["csrank"]["top50_excess_mean"]
        lines.append(f"| {window} | {period[0]} 至 {period[1]} | {vals[0]:.2%} | {vals[1]:.2%} | {vals[2]:.2%} | {top50_delta:+.2%} | {row['mean_daily_excess_delta']:+.3%} |")
    lines += ["", "## 汇总指标", "", "| 指标 | none - csrank | cszscore - csrank |", "|---|---:|---:|"]
    for label, key in (("Top5 日均 10 日超额收益", "top5_excess_delta_vs_csrank"), ("Top10 日均 10 日超额收益", "top10_excess_delta_vs_csrank"), ("Top20 日均 10 日超额收益", "top20_excess_delta_vs_csrank"), ("Top50 日均 10 日超额收益", "top50_excess_delta_vs_csrank"), ("RankIC", "rank_ic_delta_vs_csrank")):
        n = report["comparisons_vs_csrank"]["none"][key]
        z = report["comparisons_vs_csrank"]["cszscore"][key]
        lines.append(f"| {label} | {n['mean']:+.3%} [{n['ci95'][0]:+.3%}, {n['ci95'][1]:+.3%}] | {z['mean']:+.3%} [{z['ci95'][0]:+.3%}, {z['ci95'][1]:+.3%}] |")
    lines.append(f"| 成本后组合日均超额收益差 | {paired['none']['daily_excess_delta_after_cost']['mean']:+.3%} [{paired['none']['daily_excess_delta_after_cost']['ci95'][0]:+.3%}, {paired['none']['daily_excess_delta_after_cost']['ci95'][1]:+.3%}] | {paired['cszscore']['daily_excess_delta_after_cost']['mean']:+.3%} [{paired['cszscore']['daily_excess_delta_after_cost']['ci95'][0]:+.3%}, {paired['cszscore']['daily_excess_delta_after_cost']['ci95'][1]:+.3%}] |")
    lines += [
        "",
        "## 非重叠 10 日持有期核对",
        "",
        f"从 539 个连续 OOS 交易日中每隔 10 日抽一个信号日，共 {len(nonoverlap_dates)} 个不重叠的 10 日前瞻收益样本；bootstrap 在各窗口内按 2 个信号日（约 20 个交易日）成块重采样。",
        "",
        "| TopN 超额收益差（相对 csrank） | none | cszscore |",
        "|---|---:|---:|",
    ]
    for top_n in (5, 10, 20, 50):
        key = f"top{top_n}_excess"
        a = nonoverlap["comparisons_vs_csrank"]["none"][key]
        b = nonoverlap["comparisons_vs_csrank"]["cszscore"][key]
        lines.append(f"| Top{top_n} | {a['mean']:+.2%} [{a['ci95'][0]:+.2%}, {a['ci95'][1]:+.2%}] | {b['mean']:+.2%} [{b['ci95'][0]:+.2%}, {b['ci95'][1]:+.2%}] |")
    lines += [
        "",
        f"完整日频指标中，`none` 在 {paired['none']['windows_won_by_mean_daily_excess']}/{paired['none']['window_count']} 个窗口的成本后日均超额高于 `csrank`；个别窗口的差距很小。非重叠 10 日核对用于判断这一优势能否跨持有周期稳健复现。",
        "",
        "## 解读与限制",
        "",
        "结论：目前没有足够稳健的数据证明 `none` 优于 `csrank`。完整日频 OOS 汇总和日频 Qlib Top50 成本后回测偏向 `none`，但回测频率与 10 日标签持有周期不完全匹配；更匹配的非重叠 10 日 Top5 比较中，`none - csrank` 为 +1.57 个百分点，95% 区间 [-0.70, +4.14]，跨零。按现有证据，二者暂不能判出可靠赢家。",
        "诊断组 `cszscore` 在非重叠 Top5 的差值为 +3.83 个百分点，区间 [+1.49, +7.27]，表现值得继续验证；不过它是额外比较，且仍只有一个随机种子，尚不能据此直接改生产默认值。RankIC 衡量全截面顺序相关，与有限持仓、换手和交易成本后的组合收益不是同一个目标，因此排序不同并不矛盾。",
        "",
        "每个 OOS 窗口只训练一个随机种子；股票池冻结为当前 Qlib a500 主板 2986 只，历史比较有幸存者偏差；Top50 Qlib 日频回测不等于生产方向回测的完整持仓、止损和其他过滤链。完整日频 TopN 标签收益窗口重叠；非重叠抽样减少时间重叠，但只有 54 个信号日期，仍需多个随机种子和按历史成分股构造的无幸存者偏差股票池复核。",
        "",
        "统计口径：TopN 使用同日共同覆盖行计算 TopN 均值减全池均值；RankIC 是 Spearman 截面相关；回测比较使用 Qlib 报告中的组合日收益减基准日收益，成本已计入，并按相同日期配对。",
        "",
    ]
    md_path = OUT_DIR / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
