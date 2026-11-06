"""
验证 vol_scalar 与模型 IC 的相关性
如果 vol_scalar 能解释 IC 波动 → 择时模型有效 → 不需要复杂 regime 分类
"""
import sys
sys.path.insert(0, ".")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from core.shared import config
from core.data import Database

LABEL_HORIZON = 10

RUN_MAP = {
    "d69ef77d9339495a974365f947af1728": "trend_momentum (22-25)",
    "41270c3636cc45d8906f771b8d241168": "trend_momentum (19-25)",
    "50f3a7dbc2d84fea8ff02214d4ca2d10": "volume_breakout (19-25)",
    "055472a7f75d4535a106e6e87bd110c5": "oversold_rebound (19-25)",
    "ffcd48579a3544b7b7ad7b58feaa89ad": "oversold_rebound (22-25)",
    "6c522f68c7974bc38a2594a07ad08b23": "volume_breakout (22-25)",
}


def load_test_preds():
    mlruns = Path(config.QLIB_MLFLOW_URI.replace("file:", ""))
    preds = {}
    for p in mlruns.glob("*/**/artifacts/pred.pkl"):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "index"):
            continue
        idx = obj.index
        dmin = str(idx.get_level_values(0).min())[:10]
        dmax = str(idx.get_level_values(0).max())[:10]
        if dmin.startswith("2026-03") and dmax.startswith("2026-08"):
            run_id = p.parent.parent.name
            if run_id not in RUN_MAP:
                continue
            df = obj.reset_index()
            df.columns = ["trade_date", "instrument", "score"]
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["code"] = df["instrument"].str.replace("^sh|^sz", "", regex=True)
            preds[run_id] = df
    return preds


def get_forward_returns(dates, codes):
    d_min = min(dates).strftime("%Y-%m-%d")
    d_max = (max(dates) + pd.Timedelta(days=LABEL_HORIZON + 5)).strftime("%Y-%m-%d")
    all_rows = []
    for i in range(0, len(codes), 500):
        batch = codes[i:i+500]
        ph = ",".join(["%s"] * len(batch))
        rows = Database.fetchall(
            f"SELECT code, trade_date, close FROM daily_quotes "
            f"WHERE trade_date >= %s AND trade_date <= %s AND code IN ({ph})",
            (d_min, d_max, *batch)
        )
        all_rows.extend(rows)
    price_df = pd.DataFrame(all_rows)
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
    price_df["code"] = price_df["code"].str.replace("^sh|^sz", "", regex=True)
    price_df = price_df.sort_values(["code", "trade_date"])
    price_df["close_fwd"] = price_df.groupby("code")["close"].shift(-LABEL_HORIZON)
    price_df["fwd_ret"] = price_df["close_fwd"] / price_df["close"] - 1
    return price_df[["code", "trade_date", "fwd_ret"]].dropna(subset=["fwd_ret"])


def get_vol_scalars(dates):
    """用 RegimePredictorV2 获取每个日期的 vol_scalar"""
    from core.domain.regime.predictor_v2 import RegimePredictorV2
    predictor = RegimePredictorV2()
    scalars = {}
    for d in sorted(set(d.strftime("%Y-%m-%d") for d in dates)):
        try:
            result = predictor.predict_regime(as_of=d)
            if result and result.get("vol_scalar") is not None:
                scalars[d] = float(result["vol_scalar"])
        except Exception:
            continue
    return scalars


def daily_ic(group):
    if len(group) < 10:
        return np.nan
    return group["score"].corr(group["fwd_ret"])


def main():
    print("=" * 65)
    print("vol_scalar x 模型 IC 相关性验证")
    print("核心问题: 波动率能否解释模型表现的变化?")
    print("=" * 65)

    # 1. 加载
    print("\n[1/4] 加载预测...")
    preds = load_test_preds()
    all_codes = sorted(set().union(*[set(df["code"].unique()) for df in preds.values()]))
    all_dates = set().union(*[set(df["trade_date"]) for df in preds.values()])
    returns_df = get_forward_returns(list(all_dates), all_codes)

    # 2. vol_scalar
    print("\n[2/4] 获取 vol_scalar...")
    vol_scalars = get_vol_scalars(all_dates)
    print(f"  标注 {len(vol_scalars)} 天")

    # 分布
    vals = list(vol_scalars.values())
    print(f"  vol_scalar: min={min(vals):.3f}  max={max(vals):.3f}  "
          f"mean={np.mean(vals):.3f}  median={np.median(vals):.3f}")

    low_vol = sum(1 for v in vals if v >= 0.95)
    mid_vol = sum(1 for v in vals if 0.80 <= v < 0.95)
    high_vol = sum(1 for v in vals if v < 0.80)
    print(f"  低波(>=0.95): {low_vol}天 中波(0.80-0.95): {mid_vol}天  高波(<0.80): {high_vol}天")

    # 3. 计算逐日 IC
    print("\n[3/4] 计算逐日 IC...")
    all_daily_ic = {}  # strategy -> {date: IC}
    for run_id, pred_df in sorted(preds.items()):
        strategy = RUN_MAP[run_id]
        merged = pred_df.merge(returns_df, on=["code", "trade_date"], how="inner")
        groups = merged.groupby("trade_date")
        daily = groups.apply(daily_ic, include_groups=False).reset_index(name="IC")
        daily = daily.dropna(subset=["IC"])
        all_daily_ic[strategy] = dict(zip(
            daily["trade_date"].dt.strftime("%Y-%m-%d"), daily["IC"]
        ))

    # 4. 相关性检验
    print("\n[4/4] 相关性检验...")
    print("=" * 65)

    # 合并所有模型的数据
    # 对每个日期, 取所有模型 IC 的均值作为 "整体模型表现"
    dates_common = sorted(set(vol_scalars.keys()) &
                          set().union(*[set(d.keys()) for d in all_daily_ic.values()]))

    print(f"\n  公共日期: {len(dates_common)} 天")

    # 对每个模型单独检验
    print(f"\n  {'模型':30s} {'Pearson r':>10s} {'p值':>8s} {'Spearman r':>10s} {'p值':>8s} {'判断'}")
    print(f"  {'-'*75}")

    sig_count = 0
    for strategy, ic_dict in sorted(all_daily_ic.items()):
        ic_series = []
        vol_series = []
        for d in dates_common:
            if d in ic_dict and d in vol_scalars:
                ic_series.append(ic_dict[d])
                vol_series.append(vol_scalars[d])

        if len(ic_series) < 20:
            continue

        # Pearson (线性相关)
        r_pearson, p_pearson = stats.pearsonr(vol_series, ic_series)
        # Spearman (单调相关, 更稳健)
        r_spearman, p_spearman = stats.spearmanr(vol_series, ic_series)

        verdict = ""
        if p_spearman < 0.05:
            direction = "正相关" if r_spearman > 0 else "负相关"
            verdict = f"显著 {direction}"
            sig_count += 1
        elif p_spearman < 0.10:
            verdict = "边缘显著"
        else:
            verdict = "不显著"

        print(f"  {strategy:30s} {r_pearson:+10.4f} {p_pearson:8.4f} "
              f"{r_spearman:+10.4f} {p_spearman:8.4f} {verdict}")

    # 5. 整体模型 IC vs vol_scalar
    print(f"\n  --- 整体 ---")
    pooled_ic = []
    pooled_vol = []
    for d in dates_common:
        day_ics = [all_daily_ic[s][d] for s in all_daily_ic if d in all_daily_ic[s]]
        if day_ics and d in vol_scalars:
            pooled_ic.append(np.mean(day_ics))
            pooled_vol.append(vol_scalars[d])

    r_p, p_p = stats.pearsonr(pooled_vol, pooled_ic)
    r_s, p_s = stats.spearmanr(pooled_vol, pooled_ic)
    print(f"  {'整体 (6模型均值)':30s} {r_p:+10.4f} {p_p:8.4f} "
          f"{r_s:+10.4f} {p_s:8.4f} {'显著' if p_s < 0.05 else '不显著'}")

    # 6. 按 vol_scalar 分组看 IC
    print(f"\n  按 vol_scalar 分组:")
    bins = [(0.70, 0.80, "高波 <0.80"), (0.80, 0.90, "中波 0.80-0.90"),
            (0.90, 0.95, "中低波 0.90-0.95"), (0.95, 1.01, "低波 >=0.95")]
    for lo, hi, label in bins:
        group_dates = [d for d in dates_common if lo <= vol_scalars[d] < hi]
        if len(group_dates) < 3:
            continue
        group_ics = []
        for d in group_dates:
            day_ics = [all_daily_ic[s][d] for s in all_daily_ic if d in all_daily_ic[s]]
            if day_ics:
                group_ics.append(np.mean(day_ics))
        print(f"    {label:20s}: {len(group_dates):3d}d  Mean IC={np.mean(group_ics):+.4f}  "
              f"Std={np.std(group_ics):.4f}  Win%={(np.array(group_ics)>0).mean():.0%}")

    # 7. 结论
    print("\n" + "=" * 65)
    print("结论")
    print("=" * 65)

    if sig_count >= 3:
        print(f"\n  [{sig_count}/6 模型显著] vol_scalar 能稳定解释模型 IC 变化")
        print(f"  建议: 用 vol_scalar 做仓位管理 + 策略权重调整, 放弃 regime 分类")
    elif sig_count >= 1:
        print(f"\n  [{sig_count}/6 模型显著] vol_scalar 有一定解释力但不够稳定")
        print(f"  建议: vol_scalar 用于仓位管理, regime 分类继续观察")
    else:
        print(f"\n  [0/6 模型显著] vol_scalar 与模型 IC 无明显关系")
        print(f"  波动率不是当前市场下模型表现的主要驱动因素")

    if r_s > 0:
        print(f"  整体 Spearman r={r_s:+.3f} → vol_scalar 越高(波动越低), 模型 IC 越高")
    else:
        print(f"  整体 Spearman r={r_s:+.3f} → 方向与预期相反或无关")


if __name__ == "__main__":
    main()
