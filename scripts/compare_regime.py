"""
确定性规则 vs V1 分类器 对比
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from collections import Counter
from core.data import Database

# ── 1. 数据准备 ──
# 用大盘股代理指数
CODES = ['600519','601318','600036','000858','600900','601166','600276','600887',
         '601398','000333','600030','601328','600809','000651','002415','600585',
         '600104','002714','601288','601988','600031','000725','600690','601857',
         '300750','002594','601899','600048','601668','000568','000002','600309']

code_list = ",".join([f"'{c}'" for c in CODES])
rows = Database.fetchall(
    f"SELECT code, trade_date, close FROM daily_quotes "
    f"WHERE code IN ({code_list}) AND trade_date >= '2023-01-01' AND trade_date <= '2026-08-09' "
    f"ORDER BY code, trade_date"
)
df = pd.DataFrame(rows)
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["close"] = df["close"].astype(float)

# 等权指数
df = df.sort_values(["code", "trade_date"])
df["ret"] = df.groupby("code")["close"].pct_change()
idx = df.groupby("trade_date")["ret"].mean().reset_index(name="mkt_ret")
idx = idx.sort_values("trade_date")

# ── 2. 确定性规则分类器 ──
def deterministic_regime(idx_df):
    """纯规则: 20日收益方向 × 波动率分位数"""
    df = idx_df.copy()
    df["ret_20d"] = df["mkt_ret"].rolling(20).sum()
    df["vol_20d"] = df["mkt_ret"].rolling(20).std() * np.sqrt(252)
    df["vol_pct"] = df["vol_20d"].rolling(252).rank(pct=True)  # 过去1年分位数

    results = []
    for _, row in df.iterrows():
        ret = row.get("ret_20d")
        vol = row.get("vol_20d")
        pct = row.get("vol_pct")

        if pd.isna(ret) or pd.isna(vol) or pd.isna(pct):
            results.append(None)
            continue

        # 波动率
        vol_state = "volatile" if pct > 0.7 else "steady"
        # 方向
        if ret > 0.02:
            dir_state = "up"
        elif ret < -0.02:
            dir_state = "down"
        else:
            dir_state = "flat"

        results.append(f"{vol_state}_{dir_state}")

    df["rule_regime"] = results
    return df

idx = deterministic_regime(idx)
idx = idx.dropna(subset=["rule_regime"])

# ── 3. V1 分类器标注 ──
print("V1 Predictor 标注中...")
from core.domain.regime.predictor import get_regime_predictor
predictor = get_regime_predictor()

# 只标注 2024 年后的 (V1 需要预热)
test_dates = idx[idx["trade_date"] >= "2024-06-01"]["trade_date"].dt.strftime("%Y-%m-%d").unique()
sample_dates = sorted(test_dates)[::3]  # 每3天抽1天, 节省时间
print(f"  抽样 {len(sample_dates)} 天")

v1_labels = {}
for i, d in enumerate(sample_dates):
    try:
        r = predictor.predict_regime(as_of=d)
        if r and r.get("top_state"):
            v1_labels[d] = r["top_state"]
    except Exception:
        pass
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(sample_dates)}")

print(f"  V1 成功标注: {len(v1_labels)} 天")

# ── 4. 对比 ──
# 取两者共同覆盖的日期
idx["date_str"] = idx["trade_date"].dt.strftime("%Y-%m-%d")
idx["v1_regime"] = idx["date_str"].map(v1_labels)
common = idx.dropna(subset=["v1_regime", "rule_regime"])
print(f"\n共同覆盖: {len(common)} 天")

# 4a. 状态数
SIX_STATES = ["volatile_up", "volatile_down", "volatile_flat",
              "steady_up", "steady_down", "steady_flat"]

print("\n" + "=" * 70)
print("1. 状态覆盖度对比")
print("=" * 70)

rule_counts = Counter(common["rule_regime"])
v1_counts = Counter(common["v1_regime"])

print(f"\n{'状态':20s} {'规则':>6s} {'V1':>6s}")
print("-" * 35)
for s in sorted(set(list(rule_counts.keys()) + list(v1_counts.keys()))):
    rc = rule_counts.get(s, 0)
    vc = v1_counts.get(s, 0)
    pct_rule = rc / len(common) * 100
    pct_v1 = vc / len(common) * 100
    print(f"{s:20s} {rc:4d}({pct_rule:4.0f}%) {vc:4d}({pct_v1:4.0f}%)")

print(f"\n有效状态数: 规则={len(rule_counts)}  V1={len(v1_counts)}")

# 集中度 (最大状态占比)
rule_top = max(rule_counts.values()) / len(common)
v1_top = max(v1_counts.values()) / len(common)
print(f"最大状态占比: 规则={rule_top:.0%}  V1={v1_top:.0%}")
print(f"(越接近100%越退化)")

# 4b. 波动率区分度
print("\n" + "=" * 70)
print("2. 波动率区分度对比")
print("=" * 70)
print(f"(不同状态之间的 vol_20d 差异越大 → 区分度越好)")
print()

print(f"--- 规则分类器 ---")
print(f"{'状态':20s} {'N':>4s} {'Vol均值':>8s} {'Vol范围':>15s}")
print("-" * 50)
rule_vols = {}
for s in sorted(rule_counts.keys()):
    sub = common[common["rule_regime"] == s]
    vol_mean = sub["vol_20d"].mean()
    vol_min = sub["vol_20d"].min()
    vol_max = sub["vol_20d"].max()
    rule_vols[s] = vol_mean
    print(f"{s:20s} {len(sub):4d} {vol_mean:8.1%}  [{vol_min:.1%}, {vol_max:.1%}]")

# 计算 vol 的组间方差 / 总方差
rule_vol_values = [rule_vols[s] for s in rule_vols]
if len(rule_vol_values) > 1:
    print(f"  组间vol差异: max-min = {max(rule_vol_values)-min(rule_vol_values):.1%}")

print(f"\n--- V1 分类器 ---")
print(f"{'状态':20s} {'N':>4s} {'Vol均值':>8s} {'Vol范围':>15s}")
print("-" * 50)
v1_vols = {}
for s in sorted(v1_counts.keys()):
    sub = common[common["v1_regime"] == s]
    vol_mean = sub["vol_20d"].mean()
    vol_min = sub["vol_20d"].min()
    vol_max = sub["vol_20d"].max()
    v1_vols[s] = vol_mean
    print(f"{s:20s} {len(sub):4d} {vol_mean:8.1%}  [{vol_min:.1%}, {vol_max:.1%}]")

v1_vol_values = [v1_vols[s] for s in v1_vols if len(common[common["v1_regime"]==s]) >= 3]
if len(v1_vol_values) > 1:
    print(f"  组间vol差异: max-min = {max(v1_vol_values)-min(v1_vol_values):.1%}")

# 4c. 方向区分度
print("\n" + "=" * 70)
print("3. 方向区分度对比")
print("=" * 70)

print(f"\n--- 规则分类器 ---")
print(f"{'状态':20s} {'N':>4s} {'涨%':>5s} {'跌%':>5s} {'平%':>5s}")
print("-" * 50)
for s in sorted(rule_counts.keys()):
    sub = common[common["rule_regime"] == s]
    up = (sub["ret_20d"] > 0.02).mean()
    dn = (sub["ret_20d"] < -0.02).mean()
    flat = (sub["ret_20d"].abs() <= 0.02).mean()
    print(f"{s:20s} {len(sub):4d} {up:5.0%} {dn:5.0%} {flat:5.0%}")

print(f"\n--- V1 分类器 ---")
print(f"{'状态':20s} {'N':>4s} {'涨%':>5s} {'跌%':>5s} {'平%':>5s}")
print("-" * 50)
for s in sorted(v1_counts.keys()):
    sub = common[common["v1_regime"] == s]
    up = (sub["ret_20d"] > 0.02).mean()
    dn = (sub["ret_20d"] < -0.02).mean()
    flat = (sub["ret_20d"].abs() <= 0.02).mean()
    print(f"{s:20s} {len(sub):4d} {up:5.0%} {dn:5.0%} {flat:5.0%}")

# 4d. 状态切换频率
print("\n" + "=" * 70)
print("4. 状态切换频率")
print("=" * 70)

for label, col in [("规则", "rule_regime"), ("V1", "v1_regime")]:
    seq = common[col].values
    switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    rate = switches / (len(seq) - 1) if len(seq) > 1 else 0
    avg_days = 1 / rate if rate > 0 else float("inf")
    print(f"  {label}: {switches}次切换 / {len(seq)-1}次机会 = {rate:.1%}  (平均{avg_days:.0f}天切换一次)")

# ── 5. 综合评分 ──
print("\n" + "=" * 70)
print("5. 综合评分")
print("=" * 70)

def score(name, counts, vols, n_common):
    s = {}
    # 覆盖率: 有效状态数 / 6
    s["coverage"] = min(len([c for c in counts.values() if c >= 3]) / 6, 1.0)
    # 均匀度: 1 - 最大状态占比
    s["balance"] = 1.0 - max(counts.values()) / n_common
    # 波动率区分度
    valid_vols = [v for k, v in vols.items() if counts.get(k, 0) >= 3]
    if len(valid_vols) >= 2:
        s["vol_disc"] = min((max(valid_vols) - min(valid_vols)) / 0.15, 1.0)
    else:
        s["vol_disc"] = 0
    # 稳定性 (切换频率适中)
    total = sum(counts.values())
    return s, sum(s.values()) / 3

s_rule, avg_rule = score("规则", rule_counts, rule_vols, len(common))
s_v1, avg_v1 = score("V1", v1_counts, v1_vols, len(common))

print(f"\n{'指标':20s} {'规则':>8s} {'V1':>8s}")
print("-" * 40)
print(f"{'状态覆盖(/6)':20s} {s_rule['coverage']:8.0%} {s_v1['coverage']:8.0%}")
print(f"{'分布均匀度':20s} {s_rule['balance']:8.0%} {s_v1['balance']:8.0%}")
print(f"{'波动率区分度':20s} {s_rule['vol_disc']:8.0%} {s_v1['vol_disc']:8.0%}")
print(f"{'综合':20s} {avg_rule:8.0%} {avg_v1:8.0%}")

print()
if avg_rule > avg_v1 * 1.3:
    print(">>> 确定性规则显著优于 V1 分类器")
    print("    建议: 直接替换, 用规则做 regime 分类")
elif avg_rule > avg_v1:
    print(">>> 确定性规则略优于 V1 分类器")
    print("    建议: 替换, 同时保留 V1 做对照观察")
else:
    print(">>> V1 分类器不差于规则方案")
    print("    建议: 继续排查 position_ratio 公式问题")
