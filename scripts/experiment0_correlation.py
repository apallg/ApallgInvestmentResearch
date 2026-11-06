"""实验 0: 策略模型分数相关性诊断

加载 4 个方向模型, 对近期区间预测分数, 算 Pearson 相关矩阵。
决定后续走「合并优化单一模型」还是「独立混合」。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DIRECTION_MODELS = {
    "trend_momentum": ["656749665990832865", "1fa5eecd2bb04b4ebc1b37f4a5e18884"],
    "volume_breakout": ["656749665990832865", "235f1dbec8144ebe97a4a570f4293be9"],
    "oversold_rebound": ["656749665990832865", "3272fb4fe4a84fe1b24e4df5319b4c28"],
    "value_defense": ["656749665990832865", "1fa5eecd2bb04b4ebc1b37f4a5e18884"],
}

NAMES = ["trend_momentum", "volume_breakout", "oversold_rebound", "value_defense"]


def _extract_cross_section(pred, d_str):
    """从 pred DataFrame 取某一天的横截面分数 {instrument: score}"""
    try:
        day = pred.loc[d_str]
    except Exception:
        # 尝试 (datetime, instrument) MultiIndex
        try:
            day = pred.xs(d_str, level="datetime")
        except Exception:
            return {}
    if hasattr(day, "iloc"):
        scores = day.iloc[:, 0] if day.ndim > 1 else day
        out = {}
        for c, v in scores.items():
            code = str(c).split(".")[0] if isinstance(c, str) else str(c)
            out[code] = float(v)
        return out
    return {}


def main():
    from core.adapters.qlib.models import QlibModelManager

    mgr = QlibModelManager()
    predict_start, predict_end = "2026-06-01", "2026-08-14"
    instruments = "a500"

    preds = {}
    for dir_name in NAMES:
        exp_id, run_id = DIRECTION_MODELS[dir_name]
        try:
            pred, _ = mgr.predict_dual(exp_id, run_id,
                                       predict_start=predict_start, predict_end=predict_end,
                                       instruments=instruments)
            preds[dir_name] = pred
            print(f"[加载] {dir_name}: {len(pred)} 条预测")
        except Exception as e:
            print(f"[失败] {dir_name}: {e}")

    if len(preds) < 2:
        print("模型加载不足, 无法算相关")
        return

    # 取所有预测日期, 挑最近 N 个交易日截面算相关
    try:
        all_dates = sorted(set(preds[NAMES[0]].index.get_level_values("datetime")))
    except Exception:
        all_dates = sorted(set(preds[NAMES[0]].index))
    recent_dates = all_dates[-5:]  # 最近 5 个交易日

    # 逐日算相关, 取平均
    corr_sum = np.zeros((len(NAMES), len(NAMES)))
    n_valid = 0
    for d in recent_dates:
        d_str = str(d)[:10]
        sections = {n: _extract_cross_section(preds[n], d_str) for n in NAMES if n in preds}
        # 对齐共同股票
        common = set.intersection(*[set(s.keys()) for s in sections.values()]) if sections else set()
        if len(common) < 30:
            continue
        mat = np.array([[sections[a][c] for c in sorted(common)] for a in NAMES if a in sections])
        corr = np.corrcoef(mat)
        corr_sum += corr
        n_valid += 1

    if n_valid == 0:
        print("无有效截面, 无法算相关")
        return

    corr = corr_sum / n_valid
    print(f"\n=== 模型分数相关矩阵 (平均 {n_valid} 个截面, 股票池={instruments}) ===")
    print("            " + "".join(f"{n[:10]:>12}" for n in NAMES))
    for i, n in enumerate(NAMES):
        print(f"{n:12s}" + "".join(f"{corr[i, j]:>12.3f}" for j in range(len(NAMES))))

    print("\n=== 判断 (按实验文档阈值) ===")
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            c = corr[i, j]
            verdict = "高度相关→合并" if c > 0.8 else ("部分独立→混合" if c > 0.5 else "独立→保留")
            print(f"  {NAMES[i]:16s} vs {NAMES[j]:16s} r={c:.3f}  → {verdict}")


if __name__ == "__main__":
    main()
