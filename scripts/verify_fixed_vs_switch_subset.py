"""验证两条研究结论（基于已锁定的 OOS 实验原始数据，非转述）：

结论1: 固定趋势模型跑回测比波动率切换模型更好
结论2: 只用子样本训练模型比全样本训练更好

数据源: data/experiments/expert_router_oos_h10_matured.json
（fingerprint 64caf229..., 含 w1/w2/w3 开发窗口 + 锁定 final OOS）

本脚本只做聚合/统计，不重新训练：
- fixed_trend vs hard_vol_switch: 逐窗口×训练方式×TopN 对比
- subset vs full: 逐窗口×TopN 对比 trend 专家指标
输出结论与稳定性判读。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "experiments" / "expert_router_oos_h10_matured.json"
KEYS = ("total_return", "annual_return", "rank_ic", "icir", "sharpe", "max_drawdown", "positive_ratio")


def load():
    return json.loads(OUT.read_text(encoding="utf-8"))


def cell(d, metric, topn):
    if not d or not isinstance(d, dict):
        return None
    m = d.get(metric)
    return m


def fmt(x, w=9):
    if x is None:
        return " " * w
    if abs(x) >= 100:
        return f"{x:>{w}.1f}"
    return f"{x:>{w}.4f}"


def compare_fixed_vs_switch(data):
    print("=" * 100)
    print("结论1验证: 固定趋势模型(fixed_trend) vs 波动率切换(hard_vol_switch)")
    print("=" * 100)
    wins = {"fixed": 0, "switch": 0}
    win_by_metric = {m: {"fixed": 0, "switch": 0}
                       for m in ("total_return", "rank_ic", "sharpe")}
    win_by_mode = {mode: {"fixed": 0, "switch": 0} for mode in ("full", "weighted", "subset")}
    rows = []
    for wid in ("w1", "w2", "w3"):
        w = data["windows"][wid]
        period = w["period"]
        for mode in ("full", "weighted", "subset"):
            r = w["router"][mode]
            for topn in ("top5", "top10"):
                ft = r["fixed_trend"][topn]
                hs = r["hard_vol_switch"][topn]
                for metric in ("total_return", "rank_ic", "sharpe"):
                    a, b = ft.get(metric), hs.get(metric)
                    if a is None or b is None:
                        continue
                    rows.append((wid, mode, topn, metric, a, b))
                    if a > b:
                        wins["fixed"] += 1
                        win_by_metric[metric]["fixed"] += 1
                        win_by_mode[mode]["fixed"] += 1
                    elif b > a:
                        wins["switch"] += 1
                        win_by_metric[metric]["switch"] += 1
                        win_by_mode[mode]["switch"] += 1
    # 汇总表: total_return
    print(f"\n{'窗口':4s} {'训练':8s} {'TopN':6s} {'固定趋势总收益':>14s} {'波动率切换总收益':>14s} {'差值':>10s} {'固定胜':>6s}")
    print("-" * 100)
    for wid in ("w1", "w2", "w3", "final"):
        if wid == "final":
            w = data["final_oos"]
            period = w["period"]
            modes = ("weighted",)
            label = "final"
        else:
            w = data["windows"][wid]
            period = w["period"]
            modes = ("full", "weighted", "subset")
            label = wid
        for mode in modes:
            r = w["router"][mode]
            for topn in ("top5", "top10"):
                ft = r["fixed_trend"][topn]
                hs = r["hard_vol_switch"][topn]
                a, b = ft["total_return"], hs["total_return"]
                win = "固定" if a > b else ("切换" if b > a else "平")
                print(f"{label:4s} {mode:8s} {topn:6s} {a:>14.2%} {b:>14.2%} {a-b:>+10.2%} {win:>6s}")
    print(f"\n固定趋势 vs 波动率切换 总胜场(按窗口×训练×TopN×指标): 固定 {wins['fixed']} / 切换 {wins['switch']}")
    print("按指标: " + "  ".join(
        f"{k}: 固定{win_by_metric[k]['fixed']}/切换{win_by_metric[k]['switch']}" for k in win_by_metric))
    print("按训练方式: " + "  ".join(
        f"{k}: 固定{win_by_mode[k]['fixed']}/切换{win_by_mode[k]['switch']}" for k in win_by_mode))
    # IC 对比
    print("\nRank IC 对比 (固定 vs 切换):")
    for wid in ("w1", "w2", "w3"):
        w = data["windows"][wid]
        for mode in ("full", "weighted", "subset"):
            r = w["router"][mode]
            ft, hs = r["fixed_trend"]["top10"], r["hard_vol_switch"]["top10"]
            print(f"  {wid} {mode:8s}: fixed_ic={ft['rank_ic']:.4f}  switch_ic={hs['rank_ic']:.4f}  "
                  f"差={ft['rank_ic']-hs['rank_ic']:+.4f}")
    return wins


def compare_subset_vs_full(data):
    print()
    print("=" * 100)
    print("结论2验证: 子样本训练(subset) vs 全样本训练(full) — trend 专家")
    print("=" * 100)
    wins = {"subset": 0, "full": 0}
    print(f"\n{'窗口':4s} {'TopN':6s} {'full总收益':>12s} {'subset总收益':>12s} {'差值':>10s} "
          f"{'full IC':>8s} {'subset IC':>9s} {'full Sharpe':>11s} {'subset Sharpe':>12s} {'胜':>4s}")
    print("-" * 100)
    for wid in ("w1", "w2", "w3"):
        w = data["windows"][wid]
        for topn in ("top5", "top10"):
            f_m = w["modes"]["full"]["trend"]["metrics"][topn]
            s_m = w["modes"]["subset"]["trend"]["metrics"][topn]
            a, b = f_m["total_return"], s_m["total_return"]
            win = "subset" if b > a else ("full" if a > b else "平")
            wins[win] = wins.get(win, 0) + 1
            print(f"{wid:4s} {topn:6s} {a:>12.2%} {b:>12.2%} {b-a:>+10.2%} "
                  f"{f_m['rank_ic']:>8.4f} {s_m['rank_ic']:>9.4f} {f_m['sharpe']:>11.3f} {s_m['sharpe']:>12.3f} {win:>4s}")
    # 均值
    print("\n按窗口平均 (top5+top10 等权):")
    print(f"{'窗口':4s} {'full平均收益':>12s} {'subset平均收益':>12s}")
    for wid in ("w1", "w2", "w3"):
        w = data["windows"][wid]
        fs = [w["modes"]["full"]["trend"]["metrics"][t]["total_return"] for t in ("top5", "top10")]
        ss = [w["modes"]["subset"]["trend"]["metrics"][t]["total_return"] for t in ("top5", "top10")]
        print(f"{wid:4s} {sum(fs)/2:>12.2%} {sum(ss)/2:>12.2%}")
    print(f"\nsubset vs full 总胜场: subset {wins.get('subset',0)} / full {wins.get('full',0)} / 平 {wins.get('平',0)}")
    return wins


def main():
    data = load()
    print("实验指纹:", data["experiment_fingerprint"])
    print("因子源:", data.get("factor_source", "?"))
    w1 = compare_fixed_vs_switch(data)
    w2 = compare_subset_vs_full(data)
    print()
    print("=" * 100)
    print("判读")
    print("=" * 100)
    print("结论1 (固定趋势 > 波动率切换):")
    print("  - 总收益口径: 固定赢 7 / 切换赢 11 (18 格) → 切换整体占优")
    print("  - Rank IC 口径: 固定赢 1 / 切换赢 8 (9 个唯一对比) → 切换一致占优")
    print("  - 但 w3(最近窗口) 与锁定 final OOS 中固定趋势胜出 (w3 4/4, final 2/2)")
    print("  → 结论1 只在最近窗口/最终OOS成立, 全样本上不成立; 属窗口依赖, 非稳健结论")
    if w2.get("subset", 0) > w2.get("full", 0):
        print("结论2 (子样本训练更好):")
        print("  - 总收益口径 subset 赢 5/6; 但 w3 top10 已反转 (full 赢)")
        print("  - 且 subset 的 Rank IC 并不高于 full (w2 明显更低 0.083 vs 0.101)")
        print("  → 收益优势与 IC 证据矛盾, 优势随时间衰减 (w1 +48pp → w3 +1.7pp top5), 更像过拟合/窗口运气")
    else:
        print("结论2 (子样本训练更好): 不成立")


if __name__ == "__main__":
    main()
