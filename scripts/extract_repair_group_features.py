import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("generation", choices=["old", "new"])
parser.add_argument("--start-year", choices=["2019", "2021", "2023", "2025"])
parser.add_argument("--kernels", type=int, default=2)
args = parser.parse_args()
OUT = ROOT / "data/experiments/repair_group_research_20261025"
factors = json.loads((OUT / "factors.json").read_text(encoding="utf-8"))
provider = ROOT / ("data/qlib_data_prewrite_20261025" if args.generation == "old" else "data/qlib_data")
cache_dir = ROOT / ("output/fund_cache_prewrite_20261025" if args.generation == "old" else "data/fund_cache")
audit = json.loads((OUT / "input_audit.json").read_text(encoding="utf-8"))
if audit["affected_bin_factors"]:
    selected = list(factors)
else:
    selected = [key for key, value in factors.items() if value["uses_fund_cache"]] if args.generation == "old" else list(factors)

import qlib
from qlib.constant import REG_CN

qlib.init(provider_uri=str(provider), region=REG_CN, kernels=args.kernels, joblib_backend="threading", expression_cache=None, dataset_cache=None,
          exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm", "kwargs": {"uri": "file:" + str(OUT / "mlruns"), "default_exp_name": "research_only"}})
sys.excepthook = sys.__excepthook__
from qlib.data import D
from core.adapters.qlib import custom_handler
from core.data import fundamental
from scripts.research_feature_ops import rolling_last_rank, aligned_correlation

ids = sorted({int(value) for key in selected for value in re.findall(r"FundVal\(\$close,\s*(\d+)\)", factors[key]["expression"])})
columns = [fundamental.FIELD_ID_TO_NAME[value] for value in ids]
cached = pd.read_parquet(cache_dir / "fund_features.parquet", columns=["code", "trade_date"] + columns)
for column in columns:
    part = cached[["code", "trade_date", column]].dropna()
    fundamental._cache[column] = part.set_index(["code", "trade_date"])[column]
fundamental._cache_loaded = True
custom_handler._FUND_CACHE_LOADED = True
del cached, part
gc.collect()
for register in (custom_handler._register_ts_rank, custom_handler._register_sign_op, custom_handler._register_dayofweek, custom_handler._register_const, custom_handler._register_fundval):
    register()
from qlib.data.ops import Operators, Rolling, Corr as NativeCorr
alignment_events = {}


class Ts_Rank(Rolling):
    def __init__(self, feature, N):
        super().__init__(feature, N, "mean")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return rolling_last_rank(series, self.N)


Operators.register([Ts_Rank])


class Corr(NativeCorr):
    def _load_internal(self, instrument, start_index, end_index, *args):
        left = self.feature_left.load(instrument, start_index, end_index, *args)
        right = self.feature_right.load(instrument, start_index, end_index, *args)
        if not left.index.equals(right.index):
            alignment_events[f"{instrument}:{self}"] = {"instrument": instrument, "expression": str(self), "left_rows": len(left), "right_rows": len(right), "start_index": start_index, "end_index": end_index}
        return aligned_correlation(left, right, self.N)


Operators.register([Corr])

instruments = {}
for line in (ROOT / "data/qlib_data/instruments/all.txt").read_text(encoding="utf-8").splitlines():
    code, start, end = line.split("\t")[:3]
    if re.match(r"^(sh(?:60|68)|sz(?:00|30)|bj[489])", code.lower()):
        instruments.setdefault(code.lower(), []).append((start, end))
universe_path = OUT / "universe.json"
if universe_path.exists():
    if json.loads(universe_path.read_text(encoding="utf-8")) != {code: [list(pair) for pair in ranges] for code, ranges in instruments.items()}:
        raise RuntimeError("冻结股票池发生变化")
else:
    universe_path.write_text(json.dumps(instruments, ensure_ascii=False, indent=2), encoding="utf-8")
intervals = {code: [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in ranges] for code, ranges in instruments.items()}
fields = [factors[key]["expression"] for key in selected]
fields += ["Ref($open, -11)/Ref($open, -1)-1", "$open", "$close", "$high", "$low", "$volume", "$factor"]
names = selected + ["label", "open", "close", "high", "low", "volume", "factor"]
for first, last in (("2019-01-01", "2020-12-31"), ("2021-01-01", "2022-12-31"), ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-10-23")):
    if args.start_year and first[:4] != args.start_year:
        continue
    path = OUT / f"{args.generation}_{first[:4]}.parquet"
    if path.exists():
        raise RuntimeError(f"拒绝复用未验证的旧抽取结果: {path}")
    print(json.dumps({"stage": "extract", "generation": args.generation, "start": first, "end": last, "features": len(selected), "stocks": len(intervals)}), flush=True)
    frame = D.features(intervals, fields, start_time=first, end_time=last, freq="day")
    frame.columns = names
    frame = frame.reorder_levels(["datetime", "instrument"]).sort_index().astype(np.float32)
    if frame.index.has_duplicates:
        raise RuntimeError("抽取出现重复股票日")
    temporary = path.with_suffix(f".parquet.{os.getpid()}.tmp")
    frame.to_parquet(temporary, compression="zstd")
    temporary.replace(path)
    print(json.dumps({"stage": "extracted", "generation": args.generation, "start": first, "rows": len(frame), "finite_labels": int(frame["label"].notna().sum())}), flush=True)
    (OUT / f"{args.generation}_{first[:4]}_correlation_alignment.json").write_text(json.dumps(list(alignment_events.values()), indent=2), encoding="utf-8")
    del frame
    gc.collect()
if all((OUT / f"{args.generation}_{year}.parquet").exists() for year in (2019, 2021, 2023, 2025)):
    completion_path = OUT / f"{args.generation}_extraction_complete.json"
    temporary = completion_path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"generation": args.generation, "columns": names, "research_only": True, "strict_pit_verified": False}), encoding="utf-8")
    temporary.replace(completion_path)
