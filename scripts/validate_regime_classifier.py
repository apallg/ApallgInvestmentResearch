"""
验证市态分类器本身是否准确：每个 regime 下的真实市场特征是否和名称一致。
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from core.data import Database

# 1. 拿所有交易日 (2024-2026), 算市场特征
print("拉取全量市场数据...")
mkt_rows = Database.fetchall(
    "SELECT trade_date, AVG(pct_chg) as mkt_ret, STDDEV(pct_chg) as cross_std "
    "FROM daily_quotes WHERE trade_date >= '2024-01-01' AND trade_date <= '2026-08-09' "
    "GROUP BY trade_date ORDER BY trade_date"
)
mkt = pd.DataFrame(mkt_rows)
mkt["trade_date"] = pd.to_datetime(mkt["trade_date"])
mkt["mkt_ret"] = mkt["mkt_ret"].astype(float) / 100  # 转为小数
mkt = mkt.sort_values("trade_date")

# 滚动特征
mkt["ret_20d"] = mkt["mkt_ret"].rolling(20).sum()  # 近似20日累计收益
mkt["vol_20d"] = mkt["mkt_ret"].rolling(20).std() * np.sqrt(252)
mkt["ret_5d"] = mkt["mkt_ret"].rolling(5).sum()
mkt = mkt.dropna(subset=["vol_20d"])  # 去掉前20天无rolling数据的

print(f"交易日: {len(mkt)} 天 (有rolling数据的)")
print(f"市场特征: ret_20d mean={mkt['ret_20d'].mean():+.2%} vol_20d mean={mkt['vol_20d'].mean():.1%}")

# 2. 拿 regime 标签
regime_rows = Database.fetchall(
    "SELECT trade_date, regime FROM market_regime_history ORDER BY trade_date"
)
regime_map = {str(r["trade_date"])[:10]: r["regime"] for r in regime_rows}
print(f"Regime 标签: {len(regime_map)} 天")

# Join
mkt["regime"] = mkt["trade_date"].apply(
    lambda d: regime_map.get(d.strftime("%Y-%m-%d"), None)
)
labeled = mkt.dropna(subset=["regime"])
print(f"有 regime 标签的: {len(labeled)} 天")
print(f"Regime 分布: {labeled['regime'].value_counts().to_dict()}")
print()

# 3. 按 regime 分析
print("=" * 85)
print(f"{'Regime':20s} {'N':>4s} {'Ret20d':>8s} {'Vol(年)':>8s} "
      f"{'Up%':>5s} {'Dn%':>5s} {'Flat%':>5s} {'预期特征'}")
print("-" * 85)

all_vols = labeled["vol_20d"]
vol_median = np.median(all_vols)
print(f"全市场 vol 中位数: {vol_median:.1%}")

regime_stats = {}
for regime in sorted(labeled["regime"].unique()):
    sub = labeled[labeled["regime"] == regime]
    n = len(sub)
    ret_20d = sub["ret_20d"].mean()
    vol = sub["vol_20d"].mean()
    up_pct = (sub["ret_20d"] > 0.02).mean()  # 20日涨超2%
    dn_pct = (sub["ret_20d"] < -0.02).mean()
    flat_pct = (sub["ret_20d"].abs() <= 0.02).mean()
    vol_high = vol > vol_median

    # 推断实际方向
    if up_pct >= 0.4:
        actual_dir = "up"
    elif dn_pct >= 0.4:
        actual_dir = "down"
    else:
        actual_dir = "flat"

    actual_vol = "high" if vol_high else "low"
    expected = f"{actual_vol}波+{actual_dir}"

    print(f"{regime:20s} {n:4d} {ret_20d:+8.2%} {vol:8.2%} "
          f"{up_pct:5.0%} {dn_pct:5.0%} {flat_pct:5.0%}  -> {expected}")

    regime_stats[regime] = {
        "n": n, "ret_20d": ret_20d, "vol": vol,
        "up%": up_pct, "dn%": dn_pct, "flat%": flat_pct,
        "vol_high": vol_high, "actual_dir": actual_dir,
    }

# 4. 一致性检验 (旧4状态命名)
print()
print("=" * 85)
print("名称 vs 实际特征 一致性")
print("=" * 85)

# 旧4状态的预期
old_checks = {
    "trending_up": "涨",
    "trending_down": "跌",
    "ranging": "平",
    "volatile": "高波",
}

for regime, expected_desc in old_checks.items():
    if regime not in regime_stats:
        continue
    s = regime_stats[regime]

    issues = []
    if regime == "trending_up" and s["actual_dir"] != "up":
        issues.append(f"名称含'up', 实际涨日仅 {s['up%']:.0%}")
    if regime == "trending_down" and s["actual_dir"] != "down":
        issues.append(f"名称含'down', 实际跌日仅 {s['dn%']:.0%}")
    if regime == "ranging" and s["actual_dir"] != "flat":
        issues.append(f"名称含'ranging', 实际平盘 {s['flat%']:.0%} 方向={s['actual_dir']}")
    if regime == "volatile" and not s["vol_high"]:
        issues.append(f"名称含'volatile', 实际 vol={s['vol']:.2%} <= 中位数 {vol_median:.1%}")

    if issues:
        print(f"[MISMATCH] {regime}:")
        for i in issues:
            print(f"    {i}")
    else:
        print(f"[OK] {regime}: 名称与实际特征一致")

# 5. 如果全部是平盘 — 说明分类器区分不出方向
print()
n_up = sum(1 for s in regime_stats.values() if s["actual_dir"] == "up")
n_down = sum(1 for s in regime_stats.values() if s["actual_dir"] == "down")
n_flat = sum(1 for s in regime_stats.values() if s["actual_dir"] == "flat")

if n_up == 0 and n_down == 0:
    print("!!! 所有 regime 的实际方向都是'平盘'")
    print("市态分类器无法区分上涨和下跌, 方向维度完全退化")
if n_flat >= 4:
    print("!!! 大部分 regime 方向为'平盘'")
    print("分类器对方向的敏感度不足, 可能因为 ret_20d 阈值设太高或市场本身偏平")

# 6. V1 predictor 实时标注
print()
print("=" * 85)
print("V1 Predictor 2026年标注结果")
print("=" * 85)

from core.domain.regime.predictor import get_regime_predictor
predictor = get_regime_predictor()

# 取2026年有regime标签的日期
dates_2026 = sorted([
    d.strftime("%Y-%m-%d") for d in labeled[labeled["trade_date"] >= "2026-01-01"]["trade_date"]
])

v1_labels = {}
for d in dates_2026:
    try:
        r = predictor.predict_regime(as_of=d)
        if r and r.get("top_state"):
            v1_labels[d] = r["top_state"]
    except Exception:
        pass

if v1_labels:
    print(f"成功标注 {len(v1_labels)} 天")
    for regime, cnt in Counter(v1_labels.values()).most_common():
        print(f"  {regime}: {cnt}天 ({cnt/max(len(v1_labels),1)*100:.0f}%)")

    # 用 V1 标注看市场特征
    v1_features = []
    for d, regime in v1_labels.items():
        row = labeled[labeled["trade_date"] == d]
        if not row.empty:
            v1_features.append({
                "regime": regime,
                "ret_20d": row["ret_20d"].iloc[0],
                "vol_20d": row["vol_20d"].iloc[0],
            })
    if v1_features:
        v1_df = pd.DataFrame(v1_features)
        print()
        print(f"  {'Regime':20s} {'N':>4s} {'Ret20d':>8s} {'Vol(年)':>8s}")
        print(f"  {'-'*50}")
        for regime in sorted(v1_df["regime"].unique()):
            sub = v1_df[v1_df["regime"] == regime]
            print(f"  {regime:20s} {len(sub):4d} {sub['ret_20d'].mean():+8.2%} "
                  f"{sub['vol_20d'].mean():8.2%}")

print()
print("结论:")
print("如果同一 regime 名称下波动率/方向与名称一致 → 分类器有效")
print("如果所有 regime 波动率/方向差不多 → 分类器退化了")
