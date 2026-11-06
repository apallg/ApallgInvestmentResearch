"""识别 data/qlib_data/prediction_cache/lgb_lstm/*.pkl 中 w3 (2025-07-01~2026-06-30) 预测对应哪个专家模型。

方法: 对每个 w3 缓存文件, 按 10 交易日调仓日计算全截面 Spearman Rank IC,
与 expert_router_oos_h10_matured.json 中 w3 各专家的 rank_ic 匹配。
"""
import sys
from pathlib import Path
import glob
import pickle
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.domain.backtest.common import norm_code
from core.domain.factors.forward_returns import get_forward_cache

JSON = json.loads((ROOT / "data/experiments/expert_router_oos_h10_matured.json").read_text(encoding="utf-8"))
W3 = JSON["windows"]["w3"]
PERIOD = W3["period"]
EXPECTED = {}
for mode in ("full", "weighted", "subset"):
    for role in ("trend", "rebound"):
        key = f"{mode}/{role}"
        EXPECTED[key] = W3["modes"][mode][role]["metrics"]["top10"]["rank_ic"]
EXPECTED["core"] = W3["core_baseline"]["top10"]["rank_ic"]


def trading_days(start, end):
    from core.data import Database
    rows = Database.fetchall(
        "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date",
        [start, end])
    return [r["trade_date"] for r in rows]


def main():
    fc = get_forward_cache()
    days = trading_days(PERIOD["test"][0], PERIOD["test"][1])
    rebal = days[::10]
    print(f"测试期 {PERIOD['test']} 交易日 {len(days)}, 调仓日 {len(rebal)}")
    print("预期 rank_ic:", {k: round(v, 4) for k, v in EXPECTED.items()})
    print()

    files = sorted(glob.glob(str(ROOT / "data/qlib_data/prediction_cache/lgb_lstm/*.pkl")))
    for f in files:
        try:
            c = pickle.load(open(f, "rb"))
            p = c["prediction"]
        except Exception as e:
            print("ERR", Path(f).name[:8], e)
            continue
        d0 = str(p.index.get_level_values(0).min())[:10]
        d1 = str(p.index.get_level_values(0).max())[:10]
        if not (d0 <= PERIOD["test"][0] and d1 >= PERIOD["test"][1]):
            continue
        ics = []
        for d in rebal:
            d_str = str(d)[:10]
            try:
                day = p.xs(d_str, level="datetime")
            except Exception:
                continue
            scores = day.iloc[:, 0] if hasattr(day, "columns") else day
            scores = scores.dropna()
            num = {norm_code(k): float(v) for k, v in scores.items()}
            codes = [c0 for c0 in num if len(c0) == 6]
            if len(codes) < 100:
                continue
            fwd_map = fc.get_batch(codes, pd.Timestamp(d_str), 10)
            xs, ys = [], []
            for c0 in codes:
                r_ = fwd_map.get(c0)
                if r_ is None or not np.isfinite(r_):
                    continue
                xs.append(num[c0])
                ys.append(r_)
            if len(xs) >= 100:
                ic = pd.Series(xs).rank().corr(pd.Series(ys).rank())
                if ic is not None and np.isfinite(ic):
                    ics.append(float(ic))
        if not ics:
            print(Path(f).name[:8], d0, d1, "no IC")
            continue
        mean_ic = np.mean(ics)
        # 找最接近的预期
        best = min(EXPECTED, key=lambda k: abs(EXPECTED[k] - mean_ic))
        print(f"{Path(f).name[:8]}  {d0}~{d1}  n_ic={len(ics):2d}  mean_ic={mean_ic:.4f}  "
              f"最近匹配: {best} ({EXPECTED[best]:.4f}) 差={mean_ic-EXPECTED[best]:+.4f}")


if __name__ == "__main__":
    main()
