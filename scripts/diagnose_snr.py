"""
分析模型 IC 的信噪比，判断"验证无效"的根因在模型端还是择时端。
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
mlruns = Path(config.QLIB_MLFLOW_URI.replace("file:", ""))

RUN_MAP = {
    "41270c3636cc45d8906f771b8d241168": "trend (19-25)",
    "d69ef77d9339495a974365f947af1728": "trend (22-25)",
    "055472a7f75d4535a106e6e87bd110c5": "oversold (19-25)",
    "ffcd48579a3544b7b7ad7b58feaa89ad": "oversold (22-25)",
    "50f3a7dbc2d84fea8ff02214d4ca2d10": "volume (19-25)",
    "6c522f68c7974bc38a2594a07ad08b23": "volume (22-25)",
}


def load_data():
    preds = {}
    for p in mlruns.glob("*/**/artifacts/pred.pkl"):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "index"):
            continue
        idx = obj.index
        dmin = str(idx.get_level_values(0).min())[:10]
        if not dmin.startswith("2026-03"):
            continue
        run_id = p.parent.parent.name
        if run_id not in RUN_MAP:
            continue
        df = obj.reset_index()
        df.columns = ["trade_date", "instrument", "score"]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["code"] = df["instrument"].str.replace("^sh|^sz", "", regex=True)
        preds[run_id] = df

    all_codes = sorted(set().union(*[set(df["code"].unique()) for df in preds.values()]))
    all_dates = sorted(set().union(*[set(df["trade_date"]) for df in preds.values()]))
    d_min = min(all_dates).strftime("%Y-%m-%d")
    d_max = (max(all_dates) + pd.Timedelta(days=LABEL_HORIZON + 5)).strftime("%Y-%m-%d")

    all_rows = []
    for i in range(0, len(all_codes), 500):
        batch = all_codes[i:i + 500]
        ph = ",".join(["%s"] * len(batch))
        rows = Database.fetchall(
            f"SELECT code, trade_date, close FROM daily_quotes "
            f"WHERE trade_date>=%s AND trade_date<=%s AND code IN ({ph})",
            (d_min, d_max, *batch))
        all_rows.extend(rows)

    price_df = pd.DataFrame(all_rows)
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
    price_df["code"] = price_df["code"].str.replace("^sh|^sz", "", regex=True)
    price_df = price_df.sort_values(["code", "trade_date"])
    price_df["close_fwd"] = price_df.groupby("code")["close"].shift(-LABEL_HORIZON)
    price_df["fwd_ret"] = price_df["close_fwd"] / price_df["close"] - 1
    price_df = price_df.dropna(subset=["fwd_ret"])

    return preds, price_df


def daily_ic(group):
    if len(group) < 10:
        return np.nan
    return group["score"].corr(group["fwd_ret"])


def main():
    preds, price_df = load_data()

    print("=" * 65)
    print("Part 1: 模型 IC 自身是否可靠? (信噪比)")
    print("=" * 65)
    print()

    all_ic_stats = {}
    best_ic_vals = None
    best_strategy = ""

    for run_id, pred_df in sorted(preds.items()):
        strategy = RUN_MAP[run_id]
        merged = pred_df.merge(
            price_df[["code", "trade_date", "fwd_ret"]],
            on=["code", "trade_date"], how="inner")

        daily = merged.groupby("trade_date").apply(
            daily_ic, include_groups=False).dropna()

        n = len(daily)
        mean_ic = daily.mean()
        std_ic = daily.std()
        t_stat = mean_ic / (std_ic / np.sqrt(n))
        p_val = stats.t.sf(t_stat, n - 1)
        icir = mean_ic / std_ic if std_ic > 0 else 0
        cv = std_ic / abs(mean_ic) if abs(mean_ic) > 0 else float("inf")

        # IC 的 95% CI
        se = std_ic / np.sqrt(n)
        ci_lo = mean_ic - 1.96 * se
        ci_hi = mean_ic + 1.96 * se

        if p_val < 0.05:
            verdict = "显著 > 0 [模型有 alpha]"
        elif p_val < 0.10:
            verdict = "边缘显著 [待观察]"
        else:
            verdict = "不显著 [可能是噪声]"

        print(f"{strategy}:")
        print(f"  IC: {mean_ic:+.4f} ± {std_ic:.4f}  n={n}d  t={t_stat:+.2f}  p={p_val:.4f}")
        print(f"  ICIR={icir:+.3f}  WinRate={(daily > 0).mean():.0%}  CV={cv:.1f}x")
        print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  => {verdict}")
        print()

        all_ic_stats[strategy] = {"mean": mean_ic, "std": std_ic, "n": n, "t": t_stat}

        if strategy == "trend (19-25)":
            best_ic_vals = daily.values
            best_strategy = strategy

    # Part 2: 模拟 — 如果有一个完美的择时信号，在当前噪声水平下能检测到吗?
    print("=" * 65)
    print("Part 2: 择时信号可检测性模拟")
    print("=" * 65)
    print()

    if best_ic_vals is not None:
        ic_std = np.std(best_ic_vals)
        print(f"基准: {best_strategy}  IC std={ic_std:.4f}  n={len(best_ic_vals)}")
        print()
        print("如果一个择时信号能将 IC 分成高低两组 (组间差异 = d),")
        print(f"在当前 IC 噪声水平 (std={ic_std:.4f}) 下, 检测到差异需要多少天?")
        print()
        print(f"  {'组间差异':>10s}  {'每组20天':>8s}  {'每组50天':>8s}  {'每组100天':>8s}  {'每组200天':>8s}")
        print(f"  {'-'*50}")

        for effect_size in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
            row = f"  {effect_size:+.3f} ({effect_size*100:.1f}%)"
            for group_n in [20, 50, 100, 200]:
                np.random.seed(42)
                sig_count = 0
                n_sim = 3000
                for _ in range(n_sim):
                    g1 = np.random.normal(0, ic_std, group_n)
                    g2 = np.random.normal(effect_size, ic_std, group_n)
                    _, p = stats.ttest_ind(g1, g2)
                    if p < 0.05:
                        sig_count += 1
                power = sig_count / n_sim
                marker = " *" if power > 0.8 else ""
                row += f"  {power:6.0%}{marker}"
            print(row)

        print()
        print("  * = power > 80%, 可稳定检测")
        print()

        # Part 3: 当前择时信号的实际区分度
        print("=" * 65)
        print("Part 3: 当前择时信号的实际区分度")
        print("=" * 65)
        print()

        from core.domain.regime.predictor import get_regime_predictor
        predictor = get_regime_predictor()

        # 取 best model 的 merged data
        for run_id, pred_df in sorted(preds.items()):
            if "41270c36" in run_id:
                merged = pred_df.merge(
                    price_df[["code", "trade_date", "fwd_ret"]],
                    on=["code", "trade_date"], how="inner")
                break

        pos_ratios = {}
        dates_list = sorted(set(merged["trade_date"].dt.strftime("%Y-%m-%d")))
        for d in dates_list:
            try:
                r = predictor.predict_regime(as_of=d)
                if r and r.get("position_ratio"):
                    pos_ratios[d] = float(r["position_ratio"])
            except Exception:
                pass

        vals = np.array(list(pos_ratios.values()))
        signal_range = vals.max() - vals.min()
        signal_std = vals.std()

        print(f"position_ratio: min={vals.min():.3f}  max={vals.max():.3f}  "
              f"range={signal_range:.3f}  std={signal_std:.4f}")
        print()

        # 按 position_ratio 分两组
        daily_ic_dict = dict(zip(
            merged["trade_date"].dt.strftime("%Y-%m-%d"),
            merged.groupby("trade_date").apply(daily_ic, include_groups=False)
        ))

        # 高仓位 vs 低仓位
        median_pos = np.median(vals)
        high_dates = [d for d in pos_ratios if pos_ratios[d] >= median_pos]
        low_dates = [d for d in pos_ratios if pos_ratios[d] < median_pos]

        high_ics = [daily_ic_dict[d] for d in high_dates if d in daily_ic_dict
                    and not np.isnan(daily_ic_dict[d])]
        low_ics = [daily_ic_dict[d] for d in low_dates if d in daily_ic_dict
                   and not np.isnan(daily_ic_dict[d])]

        actual_diff = np.mean(high_ics) - np.mean(low_ics)
        print(f"按 position_ratio 中位数 ({median_pos:.3f}) 分组:")
        print(f"  高仓位: {len(high_ics)}d  IC={np.mean(high_ics):+.4f} ± {np.std(high_ics):.4f}")
        print(f"  低仓位: {len(low_ics)}d  IC={np.mean(low_ics):+.4f} ± {np.std(low_ics):.4f}")
        print(f"  实际差异: {actual_diff:+.4f}")
        print()

        if abs(actual_diff) > 0 and len(high_ics) > 5 and len(low_ics) > 5:
            _, p_actual = stats.ttest_ind(high_ics, low_ics)
            print(f"  t 检验 p = {p_actual:.4f}  {'(显著)' if p_actual < 0.05 else '(不显著)'}")

        # Part 4: 综合判断
        print()
        print("=" * 65)
        print("综合判断")
        print("=" * 65)
        print()

        # 模型端
        model_ok = all_ic_stats[best_strategy]["t"] > 2.0
        print(f"[模型端] {best_strategy}:")
        print(f"  IC 显著 > 0 ? {'是' if model_ok else '否'} "
              f"(t={all_ic_stats[best_strategy]['t']:.1f})")
        print(f"  IC 的日间波动 std={all_ic_stats[best_strategy]['std']:.4f}")
        print()

        # 择时端
        timing_ok = signal_range > 0.2
        print(f"[择时端] position_ratio:")
        print(f"  变化范围 > 0.2 ? {'是' if timing_ok else '否'} "
              f"(range={signal_range:.3f})")
        print()

        # 交互
        print(f"[交互] 检测条件:")
        n_available = len(high_ics) + len(low_ics)
        # 从 Part 2 的模拟表估算: 在 std=ic_std, diff=|actual_diff|, n=n_available/2 时 power
        np.random.seed(42)
        sig_count = 0
        group_n_eff = min(len(high_ics), len(low_ics))
        for _ in range(3000):
            g1 = np.random.normal(0, ic_std, group_n_eff)
            g2 = np.random.normal(abs(actual_diff), ic_std, group_n_eff)
            _, p = stats.ttest_ind(g1, g2)
            if p < 0.05:
                sig_count += 1
        detect_power = sig_count / 3000
        print(f"  日 IC 噪声 std = {ic_std:.4f}")
        print(f"  择时信号区分度 = {abs(actual_diff):.4f}")
        print(f"  每组有效天数 = {group_n_eff}")
        print(f"  检测 power = {detect_power:.0%}  (需 > 80% 才能可靠检测)")
        print()

        if model_ok and not timing_ok:
            print(">>> 结论: 模型有效, 择时信号退化")
            print("    模型能稳定产生 alpha, 但择时的 position_ratio 变化太小,")
            print("    无法区分好日和坏日。问题在择时端。")
            print("    建议: 修择时模型, 或直接用静态仓位跑。")
        elif not model_ok and timing_ok:
            print(">>> 结论: 择时有效, 模型 alpha 不稳定")
            print("    问题在模型端。建议: 提升模型预测质量。")
        elif not model_ok and not timing_ok:
            print(">>> 结论: 两端都有问题")
            print(f"    模型 IC 不显著 (t={all_ic_stats[best_strategy]['t']:.1f}),")
            print(f"    择时信号无变化 (range={signal_range:.3f})。")
        else:
            print(">>> 结论: 两端都 OK, 但样本不足")
            print(f"    需要更多数据来检测择时效果 (当前 power={detect_power:.0%})。")


if __name__ == "__main__":
    main()
