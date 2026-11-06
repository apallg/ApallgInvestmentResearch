"""模拟候选改法: 在 OOS 实验 228 期逐期数据上对比

现有:  binary   p_vol>0.5 → rebound, else trend   (一刀切)
候选:
  A      中波死亡区持现金; 高波→rebound, 低波→trend
  B      同 A, 但中波死亡区半仓 trend (少亏档)
  C      A + 确认门: 高波仅在(宽度低或超跌深)才切 rebound; 低波仅在无超跌才用 trend
  D      C 但中波死亡区半仓 trend

指标: 均值/期, 标准差, 夏普(mean/std), 累计(复利), 最大回撤, 相对 fixed 胜期
注意: 阈值与规则在同期数据上选取 → 属样本内; 正式采纳前需独立 OOS/引擎 A/B。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
JSON_PATH = ROOT / "data/experiments/expert_router_oos_h10_matured.json"

WINDOWS = ("w1", "w2", "w3")
MODES = ("full", "weighted", "subset")

# 阈值 (来自实验数据的样本分位/中位数, 点时间可用)
P_MID_LO, P_MID_HI = 0.36, 0.59   # 波动率三档边界 (0.12~0.36 / 0.36~0.59 / 0.59~1.00)
BR_TH = 0.40                       # 市场宽度中位数 ≈ 0.399
OS_TH = -0.01                      # 超跌程度中位数 ≈ -0.010


def load_daily():
    d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for wid in WINDOWS:
        w = d["windows"][wid]
        for mode in MODES:
            for rec in w["router"][mode]["daily"]:
                rows.append({"window": wid, "mode": mode, **rec})
    fo = d.get("final_oos")
    if fo:
        for rec in fo["router"]["weighted"]["daily"]:
            rows.append({"window": "final", "mode": "weighted", **rec})
    df = pd.DataFrame(rows)
    for col in ("trend_net", "rebound_net", "future_vol_probability",
                "market_breadth", "oversold_degree"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rule_pick(rule, row):
    p = row["future_vol_probability"]
    br = row["market_breadth"]
    os_ = row["oversold_degree"]
    if rule == "fixed":
        return "trend", 1.0
    if rule == "binary":
        return ("rebound" if p > 0.5 else "trend"), 1.0
    if rule in ("A", "B"):
        if p > P_MID_HI:
            return "rebound", 1.0
        if p < P_MID_LO:
            return "trend", 1.0
        return ("trend", 0.5) if rule == "B" else ("cash", 0.0)
    if rule in ("C", "D"):
        if p > P_MID_HI:
            pick = "rebound" if (br <= BR_TH or os_ >= OS_TH) else "trend"
            return pick, 1.0
        if p < P_MID_LO:
            pick = "rebound" if os_ <= OS_TH else "trend"
            return pick, 1.0
        return ("trend", 0.5) if rule == "D" else ("cash", 0.0)
    if rule == "M":  # 只选模型, 不做仓位建议: 中波被迫二选一时选 trend(少亏)
        if p > P_MID_HI:
            return ("rebound" if (br <= BR_TH or os_ >= OS_TH) else "trend"), 1.0
        if p < P_MID_LO:
            return ("rebound" if os_ <= OS_TH else "trend"), 1.0
        return ("trend", 1.0)
    raise ValueError(rule)


def sim(df, rule):
    nets = []
    picks = []
    for _, row in df.iterrows():
        pick, scale = rule_pick(rule, row)
        net = row["rebound_net"] if pick == "rebound" else row["trend_net"] if pick == "trend" else 0.0
        nets.append(net * scale)
        picks.append(pick)
    nets = np.asarray(nets)
    cum = np.cumprod(1 + nets)
    peak = np.maximum.accumulate(cum)
    dd = (cum / peak - 1).min()
    sharpe = nets.mean() / (nets.std(ddof=1) + 1e-12)
    return {
        "rule": rule, "n": len(nets), "mean": nets.mean(), "std": nets.std(ddof=1),
        "sharpe": sharpe, "total": cum[-1] - 1, "maxdd": dd,
        "win_vs_fixed": int((nets - df["trend_net"].values > 0).sum()),
        "picks": pd.Series(picks).value_counts().to_dict(),
    }


def main():
    df = load_daily()
    print(f"记录: {len(df)} 期 ({df['window'].value_counts().to_dict()})")
    print(f"阈值: 中波带 [{P_MID_LO},{P_MID_HI}]  宽度≤{BR_TH}  超跌≥{OS_TH}")
    print()
    rules = ["fixed", "binary", "M", "A", "B", "C", "D"]
    print(f"{'规则':8s} {'均值/期':>9s} {'std':>8s} {'夏普':>6s} {'累计':>9s} {'最大回撤':>9s} {'胜fixed':>8s}  选型分布")
    print("-" * 110)
    res = {}
    for rule in rules:
        r = sim(df, rule)
        res[rule] = r
        print(f"{rule:8s} {r['mean']:>+9.4%} {r['std']:>8.3%} {r['sharpe']:>6.2f} "
              f"{r['total']:>+9.2%} {r['maxdd']:>9.2%} {r['win_vs_fixed']:>8d}/{len(df)}  {r['picks']}")
    print()
    print("分窗口 (binary vs A vs C):")
    for wid in list(WINDOWS) + ["final"]:
        sub = df[df["window"] == wid]
        if sub.empty:
            continue
        b = sim(sub, "binary")
        a = sim(sub, "A")
        c = sim(sub, "C")
        print(f"  {wid:6s}: binary={b['mean']:+.4%}/期 累计{b['total']:+.2%} | "
              f"A={a['mean']:+.4%}/期 累计{a['total']:+.2%} | "
              f"C={c['mean']:+.4%}/期 累计{c['total']:+.2%} | "
              f"ΔA={a['mean']-b['mean']:+.4%} ΔC={c['mean']-b['mean']:+.4%}")
    print()
    print("判读: M = 只选模型不降仓(用户要求口径); A/B/C/D = 含死亡区降仓的对照。规则阈值来自同期数据, 样本内结果, 采纳前需独立 OOS")


if __name__ == "__main__":
    main()
