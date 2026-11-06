"""
验证择时模型 position_ratio 与模型 IC 的关系
position_ratio = 多维信号融合输出 (HMM状态概率 + 波动率 + 信号权重)
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


def get_position_ratios(dates):
    """用 RegimePredictor (V1) 获取完整的择时信号"""
    from core.domain.regime.predictor import get_regime_predictor
    predictor = get_regime_predictor()
    results = {}
    for d in sorted(set(d.strftime("%Y-%m-%d") for d in dates)):
        try:
            r = predictor.predict_regime(as_of=d)
            if r and r.get("position_ratio") is not None:
                results[d] = {
                    "position_ratio": float(r["position_ratio"]),
                    "confidence": float(r.get("confidence", 0)),
                    "top_state": r.get("top_state", ""),
                }
        except Exception:
            continue
    return results


def daily_ic(group):
    if len(group) < 10:
        return np.nan
    return group["score"].corr(group["fwd_ret"])


def main():
    print("=" * 65)
    print("择时模型 position_ratio x 模型 IC 验证")
    print("=" * 65)

    # 1. 加载
    print("\n[1/4] 加载预测...")
    preds = load_test_preds()
    all_codes = sorted(set().union(*[set(df["code"].unique()) for df in preds.values()]))
    all_dates = set().union(*[set(df["trade_date"]) for df in preds.values()])
    returns_df = get_forward_returns(list(all_dates), all_codes)

    # 2. position_ratio
    print("\n[2/4] 获取择时信号...")
    timing_signals = get_position_ratios(all_dates)
    print(f"  标注 {len(timing_signals)} 天")

    pos_vals = [v["position_ratio"] for v in timing_signals.values()]
    print(f"  position_ratio: min={min(pos_vals):.3f}  max={max(pos_vals):.3f}  "
          f"mean={np.mean(pos_vals):.3f}  median={np.median(pos_vals):.3f}")

    # 按仓位区间统计
    bins = [(0.2, 0.4, "轻仓 0.2-0.4"), (0.4, 0.6, "半仓 0.4-0.6"),
            (0.6, 0.8, "重仓 0.6-0.8"), (0.8, 1.0, "满仓 0.8-1.0")]
    for lo, hi, label in bins:
        cnt = sum(1 for v in pos_vals if lo <= v < hi)
        print(f"    {label}: {cnt}天 ({cnt/max(len(pos_vals),1)*100:.0f}%)")
    print(f"  仓位均值 = 择时模型建议的平均仓位")

    # 3. 计算逐日 IC
    print("\n[3/4] 计算逐日 IC...")
    all_daily_ic = {}
    for run_id, pred_df in sorted(preds.items()):
        strategy = RUN_MAP[run_id]
        merged = pred_df.merge(returns_df, on=["code", "trade_date"], how="inner")
        groups = merged.groupby("trade_date")
        daily = groups.apply(daily_ic, include_groups=False).reset_index(name="IC")
        daily = daily.dropna(subset=["IC"])
        all_daily_ic[strategy] = dict(zip(
            daily["trade_date"].dt.strftime("%Y-%m-%d"), daily["IC"]
        ))

    # 4. 相关性
    print("\n[4/4] 相关性检验...")
    print("=" * 65)

    dates_common = sorted(set(timing_signals.keys()) &
                          set().union(*[set(d.keys()) for d in all_daily_ic.values()]))
    print(f"\n  公共日期: {len(dates_common)} 天")

    # 逐模型
    print(f"\n  {'模型':30s} {'Pearson r':>10s} {'p值':>8s} {'Spearman r':>10s} {'p值':>8s} {'判断'}")
    print(f"  {'-'*75}")

    sig_count = 0
    model_results = []
    for strategy, ic_dict in sorted(all_daily_ic.items()):
        ic_list, pos_list = [], []
        for d in dates_common:
            if d in ic_dict and d in timing_signals:
                ic_list.append(ic_dict[d])
                pos_list.append(timing_signals[d]["position_ratio"])

        if len(ic_list) < 20:
            continue

        r_p, p_p = stats.pearsonr(pos_list, ic_list)
        r_s, p_s = stats.spearmanr(pos_list, ic_list)

        if p_s < 0.05:
            verdict = f"显著 {'正' if r_s > 0 else '负'}相关"
            sig_count += 1
        elif p_s < 0.10:
            verdict = "边缘显著"
        else:
            verdict = "不显著"

        print(f"  {strategy:30s} {r_p:+10.4f} {p_p:8.4f} "
              f"{r_s:+10.4f} {p_s:8.4f} {verdict}")
        model_results.append((strategy, r_s, p_s))

    # 整体
    pooled_ic, pooled_pos = [], []
    for d in dates_common:
        day_ics = [all_daily_ic[s][d] for s in all_daily_ic if d in all_daily_ic[s]]
        if day_ics and d in timing_signals:
            pooled_ic.append(np.mean(day_ics))
            pooled_pos.append(timing_signals[d]["position_ratio"])

    r_p_all, p_p_all = stats.pearsonr(pooled_pos, pooled_ic)
    r_s_all, p_s_all = stats.spearmanr(pooled_pos, pooled_ic)

    print(f"  {'整体 (6模型均值)':30s} {r_p_all:+10.4f} {p_p_all:8.4f} "
          f"{r_s_all:+10.4f} {p_s_all:8.4f} {'显著' if p_s_all < 0.05 else '不显著'}")

    # 5. 按仓位分组
    print(f"\n  按 position_ratio 分组 (择时信号 → 实际 IC):")
    bins2 = [(0.2, 0.5, "轻仓 <0.5"), (0.5, 0.7, "中等 0.5-0.7"),
             (0.7, 0.9, "重仓 0.7-0.9"), (0.9, 1.01, "满仓 >=0.9")]
    for lo, hi, label in bins2:
        group_ics = []
        for d in dates_common:
            pos = timing_signals[d]["position_ratio"]
            if lo <= pos < hi:
                day_ics = [all_daily_ic[s][d] for s in all_daily_ic if d in all_daily_ic[s]]
                if day_ics:
                    group_ics.append(np.mean(day_ics))
        if len(group_ics) >= 3:
            print(f"    {label:15s}: {len(group_ics):3d}d  "
                  f"Mean IC={np.mean(group_ics):+.4f}  Std={np.std(group_ics):.4f}  "
                  f"Win%={(np.array(group_ics)>0).mean():.0%}")

    # 6. 择时信号的择时能力
    print(f"\n  择时模型仓位建议分布 vs 实际 IC 表现:")
    high_pos_dates = [d for d in dates_common if timing_signals[d]["position_ratio"] >= 0.7]
    low_pos_dates = [d for d in dates_common if timing_signals[d]["position_ratio"] < 0.7]

    if high_pos_dates and low_pos_dates:
        high_ics = []
        for d in high_pos_dates:
            day_ics = [all_daily_ic[s][d] for s in all_daily_ic if d in all_daily_ic[s]]
            if day_ics:
                high_ics.append(np.mean(day_ics))
        low_ics = []
        for d in low_pos_dates:
            day_ics = [all_daily_ic[s][d] for s in all_daily_ic if d in all_daily_ic[s]]
            if day_ics:
                low_ics.append(np.mean(day_ics))

        if high_ics and low_ics:
            u_stat, p_val = stats.mannwhitneyu(high_ics, low_ics, alternative="greater")
            print(f"    高仓位建议(>=0.7): {len(high_pos_dates)}d  Mean IC={np.mean(high_ics):+.4f}  Win%={(np.array(high_ics)>0).mean():.0%}")
            print(f"    低仓位建议(<0.7) : {len(low_pos_dates)}d  Mean IC={np.mean(low_ics):+.4f}  Win%={(np.array(low_ics)>0).mean():.0%}")
            print(f"    高仓位 > 低仓位? p={p_val:.4f} {'** 显著' if p_val < 0.05 else '(不显著)'}")
            print(f"    如果择时有效: 高仓位日 IC 应显著高于低仓位日")

    # 7. 结论
    print("\n" + "=" * 65)
    print("结论")
    print("=" * 65)

    if sig_count >= 3 and p_s_all < 0.05:
        print(f"\n  [择时有效] position_ratio 能稳定预测模型 IC 变化")
        print(f"  建议: 用择时信号驱动仓位管理和策略选择")
    elif sig_count >= 1 or (p_s_all < 0.10):
        print(f"\n  [择时有一定效果] {sig_count}/6 模型显著, 整体 p={p_s_all:.3f}")
        print(f"  建议: 择时模型可用但需进一步调优")
    else:
        print(f"\n  [择时无效] position_ratio 与模型 IC 无显著关系")
        print(f"  整体 Spearman r={r_s_all:+.3f}, p={p_s_all:.3f}")
        print(f"  当前择时模型的输出无法预测策略表现")


if __name__ == "__main__":
    main()
