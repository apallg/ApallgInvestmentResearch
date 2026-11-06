import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "data/experiments/multiscale_group_research_20261103/current_isolated_input"
    generation = root / "data/qlib_data/generations/7ac678dfe6e144999a8cfb9954d7d7c7"
    calendar = pd.DatetimeIndex(pd.to_datetime((generation / "calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    frozen = pd.DatetimeIndex(pd.to_datetime((root / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    if not calendar[:len(frozen)].equals(frozen):
        raise RuntimeError("原始代际交易日历与冻结面板不一致")
    generation_info = json.loads((generation / "generation.json").read_text(encoding="utf-8"))
    if generation_info["generation_id"] != generation.name or generation_info["status"] != "ready":
        raise RuntimeError("原始代际不可用")
    universe = json.loads((source / "universe.json").read_text(encoding="utf-8"))
    selected = calendar[(calendar >= "2018-01-01") & (calendar <= "2019-01-31")]
    positions = calendar.get_indexer(selected)
    frames = []
    fields = ("open", "high", "low", "close", "volume")
    for code in sorted(universe):
        feature_dir = generation / "features" / code
        values = {}
        for field in fields:
            path = feature_dir / f"{field}.day.bin"
            if not path.exists():
                break
            data = np.fromfile(path, dtype="<f4")
            if not len(data):
                break
            offset = int(data[0])
            index = positions - offset + 1
            column = np.full(len(selected), np.nan, dtype=np.float32)
            valid = (index >= 1) & (index < len(data))
            column[valid] = data[index[valid]]
            values[field] = column
        if len(values) != len(fields):
            continue
        frame = pd.DataFrame(values, index=selected)
        frame = frame.loc[np.isfinite(frame["close"]) & (frame["close"] > 0) &
                          np.isfinite(frame["volume"]) & (frame["volume"] > 0)]
        if frame.empty:
            continue
        frame["instrument"] = code
        frames.append(frame.reset_index(names="datetime").set_index(["datetime", "instrument"]))
    panel = pd.concat(frames).sort_index()
    comparison = panel.loc[panel.index.get_level_values("datetime") >= "2019-01-01"]
    reference = pd.read_parquet(source / "new_2019.parquet", columns=list(fields),
                                filters=[("datetime", ">=", pd.Timestamp("2019-01-01")),
                                         ("datetime", "<=", pd.Timestamp("2019-01-31"))]).sort_index()
    matched = comparison.index.intersection(reference.index)
    if len(matched) < 50000:
        raise RuntimeError("原始代际与冻结行情的重叠股票日不足")
    left = comparison.loc[matched, list(fields)].to_numpy()
    right = reference.loc[matched, list(fields)].to_numpy()
    equal = np.isclose(left, right, rtol=0, atol=1e-6, equal_nan=True)
    if not equal.all():
        row, column = np.where(~equal)
        examples = [{"date": str(matched[number][0].date()), "instrument": matched[number][1],
                     "field": fields[field], "generation": float(left[number, field]),
                     "frozen": float(right[number, field])} for number, field in zip(row[:10], column[:10])]
        raise RuntimeError(f"原始代际行情与冻结面板不等: {int((~equal).sum())} cells; examples={examples}")
    preheat = panel.loc[panel.index.get_level_values("datetime") < "2019-01-01"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preheat_2018.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.equals(preheat):
            raise RuntimeError("预热快照已存在且内容不同")
    else:
        preheat.to_parquet(path, compression="zstd")
    signature = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {"status": "passed", "generation": generation.name, "rows_2018": len(preheat),
             "common_2019_rows": len(matched), "max_2019_absolute_difference": float(np.max(np.abs(left - right))),
             "preheat_sha256": signature}
    (output / "preheat_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
