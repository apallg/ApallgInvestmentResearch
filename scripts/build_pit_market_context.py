import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments" / "pit_market_context_20260916.parquet"
META = ROOT / "data" / "experiments" / "pit_market_context_20260916.json"
START = "2021-12-01"
END = "2026-07-31"


def main():
    sys.path.insert(0, str(ROOT))
    from core.adapters.qlib.config import init_qlib
    init_qlib()
    from qlib.data import D

    instrument_file = ROOT / "data" / "qlib_data" / "instruments" / "a500.txt"
    instruments = []
    for line in instrument_file.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            code, first, last = fields[:3]
            if last >= START and first <= END:
                instruments.append((code, pd.Timestamp(first), pd.Timestamp(last)))
    codes = [x[0] for x in instruments]
    data = D.features(codes, ["$close"], start_time=START, end_time=END, freq="day")
    close = data["$close"].unstack("instrument").sort_index()
    close = close.replace([np.inf, -np.inf], np.nan)
    close = close.dropna(axis=1, how="all")
    ret = close.pct_change(fill_method=None)
    market_ret = ret.mean(axis=1, skipna=True)
    coverage = ret.notna().mean(axis=1)
    level = (1.0 + market_ret.fillna(0.0)).cumprod()
    abs_path = level.diff(15).abs()
    path = level.diff().abs().rolling(15, min_periods=15).sum()
    context = pd.DataFrame(index=close.index)
    context["pit_stock_count"] = ret.notna().sum(axis=1)
    context["pit_coverage"] = coverage
    context["market_level"] = level
    context["R15"] = level / level.shift(15) - 1.0
    context["ER15"] = (abs_path / path.replace(0, np.nan)).fillna(0.0).clip(0, 1)
    context["DD15"] = level / level.rolling(15, min_periods=15).max() - 1.0
    context["R3"] = level / level.shift(3) - 1.0
    context["vol15"] = ret.rolling(15, min_periods=15).std().mean(axis=1) * np.sqrt(252.0)
    ma20 = close.rolling(20, min_periods=20).mean()
    context["B20"] = (close > ma20).where(close.notna() & ma20.notna()).mean(axis=1)
    context["dB3"] = context["B20"] - context["B20"].shift(3)

    r15_stock = close / close.shift(15) - 1.0
    r3_stock = close / close.shift(3) - 1.0
    continuation = []
    for date in close.index:
        prior = r15_stock.loc[date - pd.tseries.offsets.BDay(3):date - pd.tseries.offsets.BDay(3)]
        if prior.empty:
            continuation.append(np.nan)
            continue
        asof = prior.iloc[0]
        follow = r3_stock.loc[date] if date in r3_stock.index else pd.Series(dtype=float)
        common = asof.dropna().index.intersection(follow.dropna().index)
        if len(common) < 100:
            continuation.append(np.nan)
            continue
        ranks = asof[common].rank(pct=True)
        strong = follow[common][ranks >= 0.8].mean()
        weak = follow[common][ranks <= 0.2].mean()
        continuation.append(float(strong - weak))
    context["continuation3"] = continuation
    context = context.reset_index().rename(columns={"datetime": "date"})
    context["date"] = pd.to_datetime(context["date"])
    context = context.replace([np.inf, -np.inf], np.nan)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(OUT, index=False)
    META.write_text(json.dumps({
        "schema_version": 1,
        "start": START,
        "end": END,
        "instrument_file": str(instrument_file.relative_to(ROOT)),
        "instrument_count": len(codes),
        "rows": len(context),
        "columns": list(context.columns),
        "definitions": {
            "market_ret": "equal_weight return over the PIT-valid instrument universe from a500.txt; this is not a verified historical index-membership series",
            "ER15": "abs(level_t-level_t-15)/sum(abs daily level changes), zero denominator -> 0",
            "continuation3": "at t-3 fixed prior15d-return top/bottom quintiles, 3d subsequent strong-minus-weak return",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(context.tail().to_string(index=False))
    print(f"rows={len(context)} instruments={len(codes)} output={OUT}")


if __name__ == "__main__":
    main()
