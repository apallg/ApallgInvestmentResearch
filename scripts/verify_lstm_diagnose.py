"""
诊断 LSTM vs LGB 预测相关性 0.039 是不是 bug。

核心问题: 全样本 spearman 0.039 好到可疑, 排查是"真实正交 alpha"还是"index 错位 / 数据 key 差异"。

从 MLflow 直接读两个 run 的 pred.pkl + label.pkl, 不做训练。

判读:
  - 若逐日截面相关性也接近 0 (mean ~0.04), 则正交性真实 (每天两模型都在预测不同东西)
  - 若逐日相关性高 (0.3+) 但全样本 0.039, 则存在 index 错位 bug
"""
import sys
import os
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

LGB_RUN = "mlruns/656749665990832865/f844682489904f61a28915c35361f6a8/artifacts"
LSTM_RUN = "mlruns/656749665990832865/dbfad7126dc9461abeac73816947bd26/artifacts"


def s(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    pl = s(load(os.path.join(LGB_RUN, "pred.pkl")))
    pn = s(load(os.path.join(LSTM_RUN, "pred.pkl")))
    label = s(load(os.path.join(LGB_RUN, "label.pkl")))

    print("=== index 结构 ===")
    print("LGB  index head:", list(pl.index[:3]))
    print("LSTM index head:", list(pn.index[:3]))
    print("LGB  index equals LSTM:", pl.index.equals(pn.index))
    print("LGB  index sorted == LSTM sorted:", pl.index.sort_values().equals(pn.index.sort_values()))
    print("label index equals LGB:", label.index.equals(pl.index))

    print("\n=== 预测分布 ===")
    for name, p in [("LGB", pl), ("LSTM", pn), ("label", label)]:
        print(f"{name:5s}: mean={p.mean():+.6f} std={p.std():.6f} min={p.min():+.6f} max={p.max():+.6f} "
              f"nan={p.isna().sum()} const={p.nunique()<=1}")

    print("\n=== 全样本相关性 ===")
    from scipy.stats import spearmanr, pearsonr
    mask = pl.notna() & pn.notna()
    print("spearman:", spearmanr(pl[mask], pn[mask]))
    print("pearson :", pearsonr(pl[mask], pn[mask]))

    print("\n=== 逐日截面相关性 (LSTM vs LGB) ===")
    df = pd.DataFrame({"lgb": pl, "lstm": pn}).dropna()
    daily_corr = df.groupby("datetime").apply(lambda g: g["lgb"].corr(g["lstm"]))
    daily_corr = daily_corr.dropna()
    print(f"n={len(daily_corr)} mean={daily_corr.mean():+.4f} std={daily_corr.std():.4f} "
          f"min={daily_corr.min():+.4f} max={daily_corr.max():+.4f}")
    print("分位:", {q: round(float(daily_corr.quantile(q)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.95)})

    print("\n=== 逐日 IC 对照 ===")
    df2 = pd.DataFrame({"lgb": pl, "lstm": pn, "l": label}).dropna()
    ic_lgb = df2.groupby("datetime").apply(lambda g: g["lgb"].corr(g["l"])).dropna()
    ic_lstm = df2.groupby("datetime").apply(lambda g: g["lstm"].corr(g["l"])).dropna()
    print(f"LGB  daily IC: mean={ic_lgb.mean():+.4f} std={ic_lgb.std():.4f}")
    print(f"LSTM daily IC: mean={ic_lstm.mean():+.4f} std={ic_lstm.std():.4f}")

    # 关键: 每天的 (相关性, LGB IC, LSTM IC) 三元组, 看高 IC 日的相关性是否也低
    print("\n=== 逐日相关性 vs 逐日 IC (判断正交是否真实) ===")
    merged = pd.DataFrame({"corr": daily_corr, "ic_lgb": ic_lgb, "ic_lstm": ic_lstm}).dropna()
    if len(merged):
        print("corr 与 ic_lgb 的相关:", float(np.corrcoef(merged["corr"], merged["ic_lgb"])[0, 1]))
        print("corr 与 ic_lstm 的相关:", float(np.corrcoef(merged["corr"], merged["ic_lstm"])[0, 1]))
        # 分相关性档位看 IC
        for lo, hi, tag in [(merged["corr"] <= 0.1, "corr<=0.1"),
                            ((merged["corr"] > 0.1) & (merged["corr"] <= 0.3), "0.1<corr<=0.3"),
                            (merged["corr"] > 0.3, "corr>0.3")]:
            sub = merged[lo]
            if len(sub):
                print(f"  {tag:14s}: n={len(sub):4d} ic_lgb={sub['ic_lgb'].mean():+.4f} "
                      f"ic_lstm={sub['ic_lstm'].mean():+.4f}")

    print("\n=== 预测差/比 (排除尺度差异) ===")
    print("LSTM-LGB 逐日均值:", float((pn - pl).groupby(pn.index.get_level_values('datetime')).mean().mean()))


if __name__ == "__main__":
    main()
