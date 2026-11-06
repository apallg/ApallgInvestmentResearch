"""
Regime 有效性验证脚本
核心问题: 不同 regime 下，同一策略的 IC 分布是否有显著差异？
如果没差异 → regime 标签是噪声 → regime 分治方案不可行
"""
import sys
sys.path.insert(0, ".")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats

from core.shared import config
from core.data import Database

LABEL_HORIZON = 10
MIN_SAMPLES = 5  # 单个 regime 最少天数才参与检验

# 6 个测试期模型的 strategy 映射 (按时间戳匹配训练日志)
RUN_MAP = {
    "d69ef77d9339495a974365f947af1728": "trend_momentum (22-25)",
    "41270c3636cc45d8906f771b8d241168": "trend_momentum (19-25)",
    "50f3a7dbc2d84fea8ff02214d4ca2d10": "volume_breakout (19-25)",
    "055472a7f75d4535a106e6e87bd110c5": "oversold_rebound (19-25)",
    "ffcd48579a3544b7b7ad7b58feaa89ad": "oversold_rebound (22-25)",
    "6c522f68c7974bc38a2594a07ad08b23": "volume_breakout (22-25)",
}

REGIME_NAMES = [
    "volatile_up", "volatile_down", "volatile_flat",
    "steady_up", "steady_down", "steady_flat",
]


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


def get_regime_labels(dates):
    """V1 predictor 按日期标注 regime"""
    from core.domain.regime.predictor import get_regime_predictor
    predictor = get_regime_predictor()
    labels = {}
    for d in sorted(set(d.strftime("%Y-%m-%d") for d in dates)):
        try:
            result = predictor.predict_regime(as_of=d)
            if result and result.get("top_state"):
                labels[d] = result["top_state"]
        except Exception:
            continue
    return labels


def daily_ic(group):
    if len(group) < 10:
        return np.nan
    return group["score"].corr(group["fwd_ret"])


def main():
    print("=" * 65)
    print("Regime 有效性验证")
    print("核心问题: 不同 regime 下的 IC 是否有显著差异?")
    print("=" * 65)

    # 1. 加载数据
    print("\n[1/4] 加载预测数据...")
    preds = load_test_preds()
    print(f"  加载 {len(preds)} 个模型")

    all_codes = sorted(set().union(*[set(df["code"].unique()) for df in preds.values()]))
    all_dates = set().union(*[set(df["trade_date"]) for df in preds.values()])

    returns_df = get_forward_returns(list(all_dates), all_codes)
    print(f"  Forward returns: {returns_df['trade_date'].nunique()} 天")

    # 2. Regime 标注
    print("\n[2/4] 标注 regime 标签...")
    regime_labels = get_regime_labels(all_dates)
    print(f"  标注 {len(regime_labels)} 天")
    counts = Counter(regime_labels.values())
    for r, c in counts.most_common():
        print(f"    {r}: {c}天 ({c/max(len(regime_labels),1)*100:.0f}%)")

    # 3. 计算 IC × regime
    print("\n[3/4] 计算逐日 IC...")
    model_ic_data = {}  # model_name -> DataFrame[trade_date, regime, IC]

    for run_id, pred_df in sorted(preds.items()):
        strategy = RUN_MAP[run_id]
        merged = pred_df.merge(returns_df, on=["code", "trade_date"], how="inner")
        merged["regime"] = merged["trade_date"].apply(
            lambda d: regime_labels.get(d.strftime("%Y-%m-%d"), "unknown")
        )
        merged = merged[merged["regime"] != "unknown"]

        groups = merged.groupby(["trade_date", "regime"])
        daily = groups.apply(daily_ic, include_groups=False).reset_index(name="IC")
        daily = daily.dropna(subset=["IC"])
        daily["model"] = strategy
        model_ic_data[strategy] = daily

    # 4. 统计检验
    print("\n[4/4] 统计检验...")
    print("=" * 65)

    all_results = []
    for strategy, daily in model_ic_data.items():
        print(f"\n--- {strategy} ---")
        print(f"  总天数: {daily['trade_date'].nunique()}")

        # 按 regime 分组
        regime_groups = {}
        for r in REGIME_NAMES:
            group_ic = daily[daily["regime"] == r]["IC"]
            if len(group_ic) >= MIN_SAMPLES:
                regime_groups[r] = group_ic.values

        if len(regime_groups) < 2:
            print(f"  有效 regime 不足 2 个, 无法检验")
            continue

        # 描述统计
        print(f"  {'Regime':20s} {'N':>4s} {'Mean IC':>8s} {'Std':>7s} {'Win%':>6s}")
        print(f"  {'-'*45}")
        for r in REGIME_NAMES:
            if r in regime_groups:
                ic = regime_groups[r]
                print(f"  {r:20s} {len(ic):4d} {np.mean(ic):+8.4f} {np.std(ic):7.4f} "
                      f"{(ic > 0).mean():6.0%}")

        # Kruskal-Wallis H-test (非参数, 不假设正态)
        groups_list = list(regime_groups.values())
        h_stat, p_value = stats.kruskal(*groups_list)
        print(f"\n  Kruskal-Wallis H-test: H={h_stat:.3f}, p={p_value:.4f}")

        if p_value < 0.05:
            print(f"  => 显著! 不同 regime 的 IC 分布存在差异")
        elif p_value < 0.10:
            print(f"  => 边缘显著, 需要更多数据确认")
        else:
            print(f"  => 不显著, regime 标签未能区分 IC 表现")

        # 两两比较
        print(f"\n  两两 Mann-Whitney U 检验 (p<0.05):")
        sig_pairs = []
        reg_names = sorted(regime_groups.keys())
        for i in range(len(reg_names)):
            for j in range(i + 1, len(reg_names)):
                r1, r2 = reg_names[i], reg_names[j]
                u_stat, p = stats.mannwhitneyu(regime_groups[r1], regime_groups[r2],
                                               alternative="two-sided")
                if p < 0.05:
                    sig_pairs.append((r1, r2, p))
                    mean_diff = np.mean(regime_groups[r1]) - np.mean(regime_groups[r2])
                    print(f"    {r1} vs {r2}: p={p:.4f}  diff={mean_diff:+.4f} **")

        if not sig_pairs:
            print(f"    无显著差异对")

        all_results.append({
            "strategy": strategy,
            "h_stat": h_stat,
            "p_value": p_value,
            "n_regimes": len(regime_groups),
            "regime_means": {r: np.mean(v) for r, v in regime_groups.items()},
            "sig_pairs": len(sig_pairs),
        })

    # 5. 汇总判断
    print("\n" + "=" * 65)
    print("汇总判断")
    print("=" * 65)

    sig_count = sum(1 for r in all_results if r["p_value"] < 0.10)
    print(f"\n  {sig_count}/{len(all_results)} 个模型在不同 regime 间 IC 有差异")

    if sig_count == 0:
        print("\n  [结论] Regime 标签大概率是噪声。")
        print("  当前 regime 分类器无法区分策略表现, regime 分治方案暂时不可行。")
        print("  建议: 先修 regime 分类器 (重新训练/更换特征), 或改用确定性指标替代。")
    elif sig_count < len(all_results):
        print(f"\n  [结论] 部分模型对 regime 敏感, 结论不一致。")
        print(f"  需要看具体哪个 regime 对哪个策略有区分度, 针对性使用。")
    else:
        print("\n  [结论] Regime 分类有效, 可以用于策略切换。")

    # 6. 效应量
    print(f"\n  效应量 (eta-squared 近似):")
    for r in all_results:
        n = sum(1 for v in r["regime_means"].values())
        if n >= 2:
            means = list(r["regime_means"].values())
            grand_mean = np.mean(means)
            ss_between = sum((m - grand_mean)**2 for m in means)
            ss_total = ss_between + 0.01  # 无组内数据时的粗略近似
            eta2 = ss_between / (ss_between + 1) if ss_between > 0 else 0
            desc = "大" if eta2 > 0.14 else ("中" if eta2 > 0.06 else "小")
            print(f"    {r['strategy']:30s} eta2~{eta2:.3f} ({desc}效应) p={r['p_value']:.4f}")


if __name__ == "__main__":
    main()
