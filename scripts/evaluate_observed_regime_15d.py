# -*- coding: utf-8 -*-
"""Evaluate point-in-time 15-day observed-environment routing.

This is a diagnostic experiment over the existing matured weighted expert
outcomes. It deliberately excludes the three-date final OOS block and does
not claim a production account result.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "data/experiments/expert_router_oos_h10_matured.json"
OUT_JSON = ROOT / "data/experiments/observed_regime_15d_report.json"
OUT_MD = ROOT / "docs/observed_regime_15d_report.md"
WINDOWS = ("w1", "w2", "w3")


def load_rows():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for window in WINDOWS:
        period = data["windows"][window]
        for item in period["router"]["weighted"]["daily"]:
            rows.append({"window": window, "train": period["period"]["train"], **item})
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ("trend_net", "rebound_net", "future_vol_probability", "market_breadth"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return data, frame.dropna(subset=["trend_net", "rebound_net", "future_vol_probability"])


def add_index_features(frame):
    from core.data.fetcher import DataFetcher

    index = DataFetcher().get_index_daily("沪深300", days=365 * 10)
    index["date"] = pd.to_datetime(index["date"])
    index = index.sort_values("date").drop_duplicates("date")
    close = pd.to_numeric(index["close"], errors="coerce")
    ret = close.pct_change(fill_method=None)
    index["r15"] = close / close.shift(15) - 1
    index["r3"] = close / close.shift(3) - 1
    index["er15"] = index["r15"].abs() / ret.abs().rolling(15).sum()
    index["dd15"] = close / close.rolling(15).max() - 1
    index["vol15"] = ret.rolling(15).std() * np.sqrt(252)
    index["volpct15"] = index["vol15"].rolling(252, min_periods=60).rank(pct=True)
    lookup = index.set_index("date")
    for col in ("r15", "r3", "er15", "dd15", "volpct15"):
        frame[col] = frame["date"].map(lookup[col])
    frame["breadth_change"] = frame.groupby("window")["market_breadth"].diff().fillna(0.0)
    return frame, index


def thresholds(index, train_range):
    start, end = map(pd.Timestamp, train_range)
    train = index[(index["date"] >= start) & (index["date"] <= end)]
    return {
        "dd25": float(train["dd15"].quantile(0.25)),
        "er50": float(train["er15"].median()),
    }


def classify(row, limits):
    lim = limits[row["window"]]
    if (
        row["r15"] < 0
        and row["dd15"] <= lim["dd25"]
        and row["r3"] > 0
        and row["breadth_change"] > 0
    ):
        return "repair"
    if row["r15"] > 0 and row["er15"] >= lim["er50"] and row["market_breadth"] >= 0.5:
        return "up"
    if row["r15"] < 0 and row["er15"] >= lim["er50"] and row["market_breadth"] < 0.5:
        return "down"
    return "range"


def max_drawdown(values):
    curve = np.cumprod(1.0 + np.asarray(values, dtype=float))
    return float(np.min(curve / np.maximum.accumulate(curve) - 1.0))


def summary(values):
    values = np.asarray(values, dtype=float)
    curve = np.cumprod(1.0 + values)
    return {
        "n": int(len(values)),
        "mean_per_period": float(values.mean()),
        "median_per_period": float(np.median(values)),
        "compound_total": float(curve[-1] - 1.0),
        "win_rate": float(np.mean(values > 0)),
        "max_drawdown": max_drawdown(values),
    }


def bootstrap_ci(values, seed=20260916, block=4, n_boot=5000):
    values = np.asarray(values, dtype=float)
    blocks = [values[i : i + block] for i in range(0, len(values), block)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        means.append(float(np.mean(sample[: len(values)])))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    source, frame = load_rows()
    frame, index = add_index_features(frame)
    limits = {w: thresholds(index, source["windows"][w]["period"]["train"]) for w in WINDOWS}
    frame["environment"] = frame.apply(lambda r: classify(r, limits), axis=1)
    frame["binary"] = np.where(frame["future_vol_probability"] > 0.5, frame["rebound_net"], frame["trend_net"])
    frame["rule15_vol"] = np.where(frame["volpct15"] > 0.7, frame["rebound_net"], frame["trend_net"])
    frame["observed15"] = np.where(frame["environment"] == "repair", frame["rebound_net"], frame["trend_net"])
    frame["fixed_trend"] = frame["trend_net"]
    frame["fixed_rebound"] = frame["rebound_net"]

    cases = ("fixed_trend", "fixed_rebound", "binary", "rule15_vol", "observed15")
    results = {case: summary(frame[case]) for case in cases}
    for case in cases:
        results[case]["mean_ci_block4"] = bootstrap_ci(frame[case].to_numpy())
    for case in cases:
        if case == "binary":
            continue
        delta = frame[case].to_numpy() - frame["binary"].to_numpy()
        t, p = stats.ttest_rel(frame[case], frame["binary"])
        results[case]["delta_vs_binary_mean"] = float(delta.mean())
        results[case]["delta_vs_binary_ci_block4"] = bootstrap_ci(delta, seed=20260917)
        results[case]["paired_t"] = None if np.isnan(t) else float(t)
        results[case]["paired_t_p"] = None if np.isnan(p) else float(p)

    env = {}
    for name, group in frame.groupby("environment", sort=True):
        delta = group["rebound_net"] - group["trend_net"]
        env[name] = {
            "n": int(len(group)),
            "trend_mean": float(group["trend_net"].mean()),
            "rebound_mean": float(group["rebound_net"].mean()),
            "rebound_minus_trend_mean": float(delta.mean()),
            "rebound_win_rate": float(np.mean(delta > 0)),
            "delta_ci_block4": bootstrap_ci(delta.to_numpy(), seed=20260918),
            "by_window": {
                w: {
                    "n": int(len(g)),
                    "delta_mean": float((g["rebound_net"] - g["trend_net"]).mean()),
                }
                for w, g in group.groupby("window", sort=True)
            },
        }

    by_window = {}
    for window, group in frame.groupby("window", sort=True):
        by_window[window] = {
            "n": int(len(group)),
            "environment_counts": group["environment"].value_counts().to_dict(),
            "cases": {case: summary(group[case]) for case in cases},
        }

    payload = {
        "schema_version": 1,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_experiment_fingerprint": source.get("experiment_fingerprint"),
        "sample": {"windows": WINDOWS, "weighted_rows": int(len(frame)), "unique_dates": int(frame["date"].nunique()), "final_oos_included": False},
        "limits_by_window": limits,
        "cases": results,
        "environment": env,
        "by_window": by_window,
        "notes": [
            "weighted rows only; full/weighted/subset same-date rows are not treated as independent",
            "trend_net and rebound_net are matured expert basket outcomes, not a continuous account replay",
            "2026 final_oos has three dates and is excluded from the primary estimate",
            "15-day rules use fixed feature definitions and train-period thresholds; no test-period threshold search",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 15日可观察市场环境路由实验报告",
        "",
        "## 结论",
        "",
        "在 3 个互不重叠测试窗口、75 个 weighted 调仓日上，15 日可观察环境规则没有超过波动率模型切换。它在前两个窗口的“下跌后修复”环境中曾有明显反弹模型优势，但第三窗口方向反转，因此目前只能作为诊断特征，不能确认成生产路由。",
        "",
        "## 样本与边界",
        "",
        f"- 样本：w1/w2/w3 各25日，共 {len(frame)} 行、{frame['date'].nunique()} 个唯一决策日；只使用 weighted，排除了 final-OOS 的3个日期。",
        "- 数据来自已有成熟专家结果；收益是同日专家篮子扣费结果，不是独立连续账户回放。",
        "- 每个窗口的 DD15 下四分位和 ER15 中位数只用该窗口训练期计算；没有用测试期标签调阈值。",
        "- 因子和专家来自既有实验冻结版本，早期窗口不能当作全新端到端生产 OOS。",
        "",
        "## 总体结果",
        "",
        "|方案|均值/期|区块Bootstrap 95% CI|复合总收益|最大回撤|胜率|相对波动率切换均值|相对差95% CI|配对t检验p|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        r = results[case]
        ci = r["mean_ci_block4"]
        if case == "binary":
            delta = "—"; dci = "—"; p = "—"
        else:
            delta = f"{r['delta_vs_binary_mean'] * 100:+.2f}%"
            dci = f"[{r['delta_vs_binary_ci_block4'][0] * 100:+.2f}%, {r['delta_vs_binary_ci_block4'][1] * 100:+.2f}%]"
            p = "n/a" if r["paired_t_p"] is None else f"{r['paired_t_p']:.3f}"
        lines.append(f"|{case}|{r['mean_per_period'] * 100:+.2f}%|[{ci[0] * 100:+.2f}%, {ci[1] * 100:+.2f}%]|{r['compound_total'] * 100:+.1f}%|{r['max_drawdown'] * 100:+.1f}%|{r['win_rate'] * 100:.1f}%|{delta}|{dci}|{p}|")
    lines += [
        "",
        "复合收益和回撤只用于这批专家篮子结果的直观比较；不能替代含持仓路径、切换换手和涨跌停约束的连续账户确认。",
        "",
        "## 环境条件下的模型优势",
        "",
        "|环境|样本数|趋势均值|反弹均值|反弹-趋势|反弹胜率|按窗口的优势差|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in env.items():
        by_w = ", ".join(f"{w}:{v['delta_mean'] * 100:+.2f}% (n={v['n']})" for w, v in r["by_window"].items())
        lines.append(f"|{name}|{r['n']}|{r['trend_mean'] * 100:+.2f}%|{r['rebound_mean'] * 100:+.2f}%|{r['rebound_minus_trend_mean'] * 100:+.2f}%|{r['rebound_win_rate'] * 100:.1f}%|{by_w}|")
    lines += [
        "",
        "关键观察：repair 环境的反弹相对优势为 w1 +2.29%、w2 +4.91%，但 w3 为 -3.24%；range 环境 w1 +0.71%、w2 -0.12%、w3 +1.16%。这不是稳定的状态条件优势。",
        "",
        "## 分窗口结果",
        "",
        "|窗口|环境分布|fixed trend|binary|rule15 vol|observed15|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for w, item in by_window.items():
        c = ", ".join(f"{k}:{v}" for k, v in item["environment_counts"].items())
        vals = [item["cases"][x]["mean_per_period"] * 100 for x in ("fixed_trend", "binary", "rule15_vol", "observed15")]
        lines.append(f"|{w}|{c}|{vals[0]:+.2f}%|{vals[1]:+.2f}%|{vals[2]:+.2f}%|{vals[3]:+.2f}%|")
    lines += [
        "",
        "## 决策",
        "",
        "1. 15日规则本轮不晋级为生产路由；保留为可解释诊断和后续前向特征。",
        "2. 波动率切换作为当前参照，但不能把本报告的 75 日专家篮子结果直接等同于完整回测账户优势。",
        "3. 若继续研究，应先做同一模型集合、同一 TopN、同一连续账户的纯路由回放；只有在至少一个未参与选择的确认窗口中，repair 条件的相对优势不再反转，才考虑部署反弹门控。",
        "4. 不在这批数据上继续搜索 10/15/20/30 日窗口、多个阈值和更多状态；那会把本次诊断变成回测挑选。",
        "",
        f"原始结构化结果：`{OUT_JSON.relative_to(ROOT)}`。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "rows": len(frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
