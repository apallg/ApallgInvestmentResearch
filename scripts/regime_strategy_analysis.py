"""
策略 × Regime IC 分解分析
从 MLflow 加载预测，用 RegimePredictorV2 实时标注。
"""
import sys
sys.path.insert(0, ".")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from core.shared import config
from core.data import Database

LABEL_HORIZON = 10

# MLflow run_id → strategy 映射 (按 end_time 匹配训练时间戳)
# end_time 从 meta.yaml 提取
RUN_STRATEGY = {
    "d69ef77d9339495a974365f947af1728": "trend_momentum (22-25)",
    "41270c3636cc45d8906f771b8d241168": "trend_momentum (19-25)",
    "50f3a7dbc2d84fea8ff02214d4ca2d10": "volume_breakout (19-25)",
    "055472a7f75d4535a106e6e87bd110c5": "oversold_rebound (19-25)",
    "ffcd48579a3544b7b7ad7b58feaa89ad": "oversold_rebound (22-25)",
    "6c522f68c7974bc38a2594a07ad08b23": "volume_breakout (22-25)",
}

# regime → 预期最优策略
EXPECTED = {
    "volatile_up": "trend_momentum",
    "steady_up": "trend_momentum",
    "volatile_down": "oversold_rebound",
    "steady_flat": "oversold_rebound",
    "volatile_flat": "volume_breakout",
    "steady_down": "volume_breakout",
}

TRENDING_REGIMES = {"volatile_up", "volatile_down", "steady_up", "steady_down"}
RANGING_REGIMES = {"volatile_flat", "steady_flat"}


def load_test_preds():
    """加载所有 test 期 pred.pkl"""
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
            df = obj.reset_index()
            df.columns = ["trade_date", "instrument", "score"]
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["code"] = df["instrument"].str.replace("^sh|^sz", "", regex=True)
            preds[run_id] = df
    return preds


def get_forward_returns(dates, codes):
    """从 daily_quotes 计算 10日 forward return"""
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
    """用 RegimePredictor (V1 分类器) 生成 6 状态 regime 标签"""
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
    print("=" * 70)
    print("策略 × Regime IC 分解")
    print(f"标签周期: {LABEL_HORIZON}日")
    print("=" * 70)

    # 1. 加载预测
    preds = load_test_preds()
    print(f"\n加载 {len(preds)} 个测试期预测模型")

    # 2. 获取 returns
    all_codes = set()
    all_dates = set()
    for df in preds.values():
        all_codes.update(df["code"].unique())
        all_dates.update(df["trade_date"])
    all_codes = sorted(all_codes)
    print(f"覆盖 {len(all_codes)} 只股票, {len(all_dates)} 个交易日")

    returns_df = get_forward_returns(list(all_dates), all_codes)
    print(f"Forward returns: {len(returns_df)} 条, "
          f"{returns_df['trade_date'].min().date()} ~ {returns_df['trade_date'].max().date()}")

    # 3. 生成 regime 标签
    print("\n生成 regime 标签...")
    regime_labels = get_regime_labels(all_dates)
    print(f"标注 {len(regime_labels)} 天")
    from collections import Counter
    for r, c in Counter(regime_labels.values()).most_common():
        print(f"  {r}: {c}天 ({c/max(len(regime_labels),1)*100:.0f}%)")

    # 4. Join 并计算
    print("\n" + "=" * 70)
    print("逐模型 × Regime IC 分解")
    print("=" * 70)

    all_results = []

    for run_id, pred_df in sorted(preds.items()):
        strategy = RUN_STRATEGY.get(run_id, f"unknown ({run_id[:8]})")
        strat_key = strategy.split(" (")[0]  # "trend_momentum" etc

        merged = pred_df.merge(returns_df, on=["code", "trade_date"], how="inner")
        merged["regime"] = merged["trade_date"].apply(
            lambda d: regime_labels.get(d.strftime("%Y-%m-%d"), "unknown")
        )
        merged = merged[merged["regime"] != "unknown"]
        if len(merged) < 100:
            continue

        # 逐日 × regime IC
        groups = merged.groupby(["trade_date", "regime"])
        daily_ic_df = groups.apply(daily_ic, include_groups=False).reset_index(name="IC")
        daily_ic_df = daily_ic_df.dropna(subset=["IC"])

        # 整体
        overall = daily_ic_df.groupby("trade_date")["IC"].mean().reset_index()
        overall_mean = overall["IC"].mean()
        overall_std = overall["IC"].std()
        overall_icir = overall_mean / overall_std if overall_std and overall_std > 0 else np.nan

        # 按 regime
        regime_stats = daily_ic_df.groupby("regime").agg(
            n_days=("IC", "count"),
            mean_IC=("IC", "mean"),
            std_IC=("IC", "std"),
            win_rate=("IC", lambda x: (x > 0).mean()),
        ).reset_index()
        regime_stats["ICIR"] = regime_stats["mean_IC"] / regime_stats["std_IC"].replace(0, np.nan)

        print(f"\n{strategy}")
        print(f"  整体: IC={overall_mean:+.4f}  ICIR={overall_icir:+.3f}  "
              f"n={overall['trade_date'].nunique()}d")
        for _, row in regime_stats.iterrows():
            exp = EXPECTED.get(row["regime"], "")
            match = "[MATCH]" if exp == strat_key else ("[MIS]" if exp and exp != strat_key else "")
            print(f"    {row['regime']:20s} | {int(row['n_days']):3d}d | "
                  f"IC={row['mean_IC']:+.4f} | ICIR={row['ICIR']:+.3f} | "
                  f"WR={row['win_rate']:.0%}  {match}")

        all_results.append({
            "strategy": strategy,
            "strat_key": strat_key,
            "overall_IC": overall_mean,
            "overall_ICIR": overall_icir,
            "by_regime": dict(zip(regime_stats["regime"], regime_stats["mean_IC"])),
        })

    # 5. 汇总排名
    print("\n" + "=" * 70)
    print("各 Regime 最佳策略排名")
    print("=" * 70)

    regime_best = defaultdict(list)
    for r in all_results:
        for regime, ic in r["by_regime"].items():
            regime_best[regime].append((r["strategy"], ic))

    for regime in sorted(regime_best.keys()):
        entries = sorted(regime_best[regime], key=lambda x: x[1], reverse=True)
        exp = EXPECTED.get(regime, "?")
        print(f"\n{regime} (预期: {exp})")
        for i, (strat, ic) in enumerate(entries):
            flag = " [EXPECTED]" if strat.split(" (")[0] == exp else ""
            print(f"  {i+1}. {strat:28s} IC={ic:+.4f}{flag}")

    # 6. 核心假设验证
    print("\n" + "=" * 70)
    print("假设验证: 趋势策略在趋势市更优, 震荡策略在震荡市更优")
    print("=" * 70)

    for r in all_results:
        trend_ics = [r["by_regime"].get(tr, np.nan) for tr in TRENDING_REGIMES]
        range_ics = [r["by_regime"].get(rg, np.nan) for rg in RANGING_REGIMES]
        t_avg = np.nanmean(trend_ics)
        r_avg = np.nanmean(range_ics)
        diff = t_avg - r_avg

        if r["strat_key"] == "trend_momentum":
            verdict = "OK: trend>range" if diff > 0 else "FAIL"
        elif r["strat_key"] == "oversold_rebound":
            verdict = "OK: range>trend" if r_avg > t_avg else "FAIL"
        else:
            verdict = ""

        print(f"  {r['strategy']:28s} 趋势={t_avg:+.4f}  震荡={r_avg:+.4f}  "
              f"diff={diff:+.4f}  {verdict}")


if __name__ == "__main__":
    main()
