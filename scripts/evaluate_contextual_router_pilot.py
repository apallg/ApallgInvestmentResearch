import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "experiments" / "expert_router_oos_h10_matured.json"
CONTEXT = ROOT / "data" / "experiments" / "pit_market_context_20260916.parquet"
OUT = ROOT / "data" / "experiments" / "contextual_router_pilot_20260916.json"
REPORT = ROOT / "docs" / "contextual_router_pilot_20260916.md"
SEED = 20260916
MIN_TRAIN = 26
MIN_LEAF = 8
KAPPA = 0.001

GROUPS = {
    "G0_volatility": ["vol15", "dvol5"],
    "G1_market_structure": [
        "vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3",
    ],
    "G2_strength_and_diffusion": [
        "vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3", "continuation3",
    ],
    "G3_plus_disagreement": [
        "vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3", "continuation3",
        "expert_disagreement",
    ],
}


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(values, block_size=2, draws=5000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"lower": None, "upper": None, "mean": None, "n": 0}
    rng = np.random.default_rng(SEED)
    blocks = [values[i:i + block_size] for i in range(0, len(values), block_size)]
    samples = []
    for _ in range(draws):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        samples.append(np.concatenate([blocks[i] for i in picked])[:len(values)].mean())
    q = np.quantile(samples, [0.025, 0.975])
    return {"lower": float(q[0]), "upper": float(q[1]), "mean": float(values.mean()), "n": int(len(values))}


def _run(rows, features):
    frame = rows.sort_values("date").reset_index(drop=True).copy()
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce")
    frame["advantage"] = pd.to_numeric(frame["advantage"], errors="coerce")
    frame = frame.dropna(subset=features + ["advantage"]).reset_index(drop=True)
    decisions = []
    for i in range(MIN_TRAIN, len(frame)):
        train = frame.iloc[:i]
        test = frame.iloc[i]
        model = DecisionTreeRegressor(
            max_depth=2, min_samples_leaf=MIN_LEAF, random_state=SEED,
        )
        model.fit(train[features].to_numpy(), train["advantage"].to_numpy())
        predicted = float(model.predict(test[features].to_numpy().reshape(1, -1))[0])
        route = "rebound" if predicted > KAPPA else "trend"
        realized = float(test["advantage"] if route == "rebound" else 0.0)
        decisions.append({
            "date": str(test["date"].date()),
            "predicted_advantage": predicted,
            "realized_advantage_if_route": realized,
            "actual_advantage": float(test["advantage"]),
            "actual_winner": int(test["target"]),
            "route": route,
            "correct": int((route == "rebound") == bool(test["target"])),
        })
    result = pd.DataFrame(decisions)
    if result.empty:
        return {"n_train_min": MIN_TRAIN, "n_test": 0, "features": features, "metrics": {}}
    selected = result["realized_advantage_if_route"].to_numpy()
    actual = result["actual_advantage"].to_numpy()
    metrics = {
        "n_test": int(len(result)),
        "rebound_route_count": int((result["route"] == "rebound").sum()),
        "route_rate": float((result["route"] == "rebound").mean()),
        "direction_accuracy": float(result["correct"].mean()),
        "selected_mean_advantage": float(selected.mean()),
        "selected_cumulative_simple_sum": float(selected.sum()),
        "oracle_mean_advantage": float(np.maximum(actual, 0.0).mean()),
        "fixed_trend_mean_advantage": 0.0,
        "selected_ci_4week_block": _bootstrap(selected, 2),
        "selected_ci_8week_block": _bootstrap(selected, 4),
    }
    return {
        "n_train_min": MIN_TRAIN,
        "features": features,
        "metrics": metrics,
        "decision_trace": result.to_dict(orient="records"),
    }


def _paired_summary(left, right):
    a = pd.DataFrame(left["decision_trace"])[["date", "realized_advantage_if_route"]]
    b = pd.DataFrame(right["decision_trace"])[["date", "realized_advantage_if_route"]]
    merged = a.merge(b, on="date", suffixes=("_left", "_right"))
    delta = merged["realized_advantage_if_route_right"] - merged["realized_advantage_if_route_left"]
    out = _bootstrap(delta.to_numpy(), 2)
    out["right_mean"] = float(merged["realized_advantage_if_route_right"].mean())
    out["left_mean"] = float(merged["realized_advantage_if_route_left"].mean())
    out["right_wins"] = int((delta > 0).sum())
    out["left_wins"] = int((delta < 0).sum())
    merged["quarter"] = pd.to_datetime(merged["date"]).dt.to_period("Q").astype(str)
    out["quarterly"] = []
    for quarter, group in merged.groupby("quarter", sort=True):
        d = group["realized_advantage_if_route_right"] - group["realized_advantage_if_route_left"]
        out["quarterly"].append({
            "quarter": quarter,
            "n": int(len(group)),
            "left_mean": float(group["realized_advantage_if_route_left"].mean()),
            "right_mean": float(group["realized_advantage_if_route_right"].mean()),
            "delta_mean": float(d.mean()),
        })
    return out


def main():
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = pd.DataFrame(source["router_history"]["weighted"])
    rows["date"] = pd.to_datetime(rows["date"])
    context = pd.read_parquet(CONTEXT)
    context["date"] = pd.to_datetime(context["date"])
    context["dvol5"] = context["vol15"] - context["vol15"].shift(5)
    rows = rows.drop(columns=["vol15", "dvol5"], errors="ignore").merge(
        context[["date", "vol15", "dvol5", "R15", "ER15", "DD15", "R3", "B20", "dB3", "continuation3"]],
        on="date", how="left",
    )
    results = {name: _run(rows, features) for name, features in GROUPS.items()}
    comparisons = {
        "G1_minus_G0": _paired_summary(results["G0_volatility"], results["G1_market_structure"]),
        "G2_minus_G1": _paired_summary(results["G1_market_structure"], results["G2_strength_and_diffusion"]),
        "G3_minus_G2": _paired_summary(results["G2_strength_and_diffusion"], results["G3_plus_disagreement"]),
    }
    payload = {
        "schema_version": 1,
        "status": "pilot_diagnostic_not_full_protocol",
        "source": str(INPUT.relative_to(ROOT)),
        "source_sha256": _hash_file(INPUT),
        "context_source": str(CONTEXT.relative_to(ROOT)),
        "context_sha256": _hash_file(CONTEXT),
        "source_mode": "weighted",
        "source_rows": int(len(rows)),
        "source_period": [rows.date.min().strftime("%Y-%m-%d"), rows.date.max().strftime("%Y-%m-%d")],
        "source_contract": source["config"],
        "seed": SEED,
        "tree": {"max_depth": 2, "min_samples_leaf": MIN_LEAF},
        "kappa": KAPPA,
        "results": results,
        "comparisons": comparisons,
        "limitations": [
            "源标签是既有 Top10、约10日调仓的 weighted 专家优势，不是本协议 Top5 周频连续账户收益。",
            "市场特征使用项目 Qlib a500.txt PIT 有效区间计算；这不是已核实的历史指数成分序列。行业持续性尚未接入，G2 使用 continuation3 强弱延续代理。",
            "结果仅用于判断信息增量线索，不能用于生产切换或证明完整协议通过。",
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 上下文专家路由先导实验（2026-09-16）", "",
        "状态：`pilot_diagnostic_not_full_protocol`。本报告使用既有成熟专家标签做信息增量诊断，不是 Top5 周频连续账户验收。", "",
        f"数据：weighted 模式 {len(rows)} 个决策日，{payload['source_period'][0]} 至 {payload['source_period'][1]}；市场上下文来自 a500.txt PIT 有效区间 Qlib 行情；最小滚动训练 {MIN_TRAIN} 日。",
        f"模型：最大深度2、叶节点最少{MIN_LEAF}条，随机种子{SEED}；预测优势大于 {KAPPA:.3f} 才选反弹，否则选趋势。", "",
        "## 结果", "",
        "| 组 | 测试日 | 反弹选择次数 | 方向准确率 | 路由后平均优势 | 4周块95%区间 | 8周块95%区间 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        m = item["metrics"]
        c4, c8 = m["selected_ci_4week_block"], m["selected_ci_8week_block"]
        lines.append(
            f"| {name} | {m['n_test']} | {m['rebound_route_count']} | {m['direction_accuracy']:.3f} "
            f"| {m['selected_mean_advantage']:.5f} | [{c4['lower']:.5f}, {c4['upper']:.5f}] "
            f"| [{c8['lower']:.5f}, {c8['upper']:.5f}] |"
        )
    lines += ["", "## 逐组增量（配对同一批测试日）", "",
              "| 增量 | 左组平均 | 右组平均 | 右减左 | 4周块95%区间 | 右组胜出日 | 左组胜出日 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in comparisons.items():
        lines.append(
            f"| {name} | {item['left_mean']:.5f} | {item['right_mean']:.5f} "
            f"| {item['mean']:.5f} | [{item['lower']:.5f}, {item['upper']:.5f}] "
            f"| {item['right_wins']} | {item['left_wins']} |"
        )
    lines += ["", "### 按季度的增量", "", "| 增量 | 季度 | 日期数 | 左组均值 | 右组均值 | 右减左 |", "|---|---|---:|---:|---:|---:|"]
    for name, item in comparisons.items():
        for q in item["quarterly"]:
            lines.append(
                f"| {name} | {q['quarter']} | {q['n']} | {q['left_mean']:.5f} "
                f"| {q['right_mean']:.5f} | {q['delta_mean']:.5f} |"
            )
    lines += [
        "", "## 解读规则", "",
        "路由后平均优势是选反弹时使用实际反弹减趋势优势、选趋势时记0的简单平均；它不是账户收益，也没有重复计算交易成本。" ,
        "只有在完整协议的共同 Top5 周频账户中，新增组相对 B6 的净收益、回撤和成本压力同时通过，才可把该信息用于候选晋级。", "",
        "## 预登记猜想的当前判定", "",
        "- H0（两专家有互补性）：成熟标签中实际优势均值为正且存在明显正负差异，但 G0 路由后均值仅约等于固定反弹基准；判定为“部分支持，尚不足以证明可交易互补”。",
        "- H1（市场结构比单独波动率有增量）：G1-G0 点估计为正，但 4 周块95%区间跨0；判定为“证据不足”，不是通过。",
        "- H2（强弱/扩散有额外信息）：G2-G1 点估计为负，区间上界为0；在该先导样本上“不支持”，需用完整 PIT 特征复核。",
        "- H3（模型分歧有额外信息）：G3-G2 点估计接近0且区间跨0；判定为“证据不足/当前未见增量”。",
        "- H4（新路由优于六状态参照）：本先导没有 B6 连续账户，判定为“未检验”，不得推出通过或失败。",
        "",
        "## 数据边界", "",
        "- 源数据不是本轮预登记的历史指数成分 PIT 股票池、Top5、周频和连续账户合同；a500.txt 仅提供工具有效区间。",
        "- G2 已生成日频强弱延续代理，但尚未加入 PIT 行业持续性；当前标签仍为旧 Top10/约10日合同。",
        "- 因此本报告可以否定“现有少量特征在这批成熟标签上完全没有增量”的说法，但不能证明新路由已经有效。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: item["metrics"] for name, item in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
