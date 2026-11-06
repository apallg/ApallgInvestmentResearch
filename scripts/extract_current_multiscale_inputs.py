import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "true")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.research_feature_ops import rolling_last_rank, aligned_correlation


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/experiments/multiscale_group_research_20261103/current_input")
    parser.add_argument("--years", nargs="+", choices=("2019", "2021", "2023", "2025"), default=["2019", "2021", "2023", "2025"])
    parser.add_argument("--kernels", type=int, default=2)
    parser.add_argument("--isolated-cache", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    reference = ROOT / "data/experiments/fundval_repair_research_20261026"
    source = ROOT / "data/qlib_data"
    cache = out / "fund_cache" if args.isolated_cache else ROOT / "data/fund_cache"
    factors = json.loads((reference / "factors.json").read_text(encoding="utf-8"))
    factors_path = out / "factors.json"
    if factors_path.exists() and json.loads(factors_path.read_text(encoding="utf-8")) != factors:
        raise RuntimeError("已冻结表达式清单发生变化")
    factors_path.write_text(json.dumps(factors, ensure_ascii=False, indent=2), encoding="utf-8")
    calendar_file = source / "calendars/day.txt"
    calendar_signature = digest(calendar_file)
    qlib_pointer_signature = digest(source / "current.json")

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(source), region=REG_CN, kernels=args.kernels, joblib_backend="threading",
              expression_cache=None, dataset_cache=None,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": (out / "mlruns").as_uri(), "default_exp_name": "research_only"}})
    from qlib.data import D
    from qlib.data.ops import Operators, Rolling, Corr as NativeCorr
    from core.adapters.qlib import custom_handler
    from core.adapters.qlib.generation import qlib_generation_read_lock
    from core.data import fundamental

    fundamental.CACHE_DIR = cache
    cache.mkdir(parents=True, exist_ok=True)
    if args.isolated_cache and not (cache / "current.json").exists():
        loaded = fundamental.preload_cache(force=True)
    else:
        loaded = fundamental.preload_cache()
    if not loaded:
        raise RuntimeError("当前基本面缓存加载失败")
    generation = json.loads((cache / "current.json").read_text(encoding="utf-8"))["generation"]
    cache_file = cache / "generations" / generation / "fund_features.parquet"
    snapshot = {"contract_version": 2, "generation": generation, "data_sha256": digest(cache_file),
                "qlib_calendar_sha256": calendar_signature, "qlib_pointer_sha256": qlib_pointer_signature,
                "source_provider": str(source), "isolated_fund_cache": args.isolated_cache,
                "research_only": True, "strict_pit_verified": False}
    identity_path = out / "input_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != snapshot:
        raise RuntimeError("输入代际发生变化；不可在同一目录续跑")
    identity_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    custom_handler._FUND_CACHE_LOADED = True
    for register in (custom_handler._register_ts_rank, custom_handler._register_sign_op,
                     custom_handler._register_dayofweek, custom_handler._register_const,
                     custom_handler._register_fundval):
        register()

    class Ts_Rank(Rolling):
        def __init__(self, feature, N):
            super().__init__(feature, N, "mean")

        def _load_internal(self, instrument, start_index, end_index, *arguments):
            values = self.feature.load(instrument, start_index, end_index, *arguments)
            return rolling_last_rank(values, self.N)

    class Corr(NativeCorr):
        def _load_internal(self, instrument, start_index, end_index, *arguments):
            left = self.feature_left.load(instrument, start_index, end_index, *arguments)
            right = self.feature_right.load(instrument, start_index, end_index, *arguments)
            return aligned_correlation(left, right, self.N)

    Operators.register([Ts_Rank, Corr])
    instruments = {}
    for line in (source / "instruments/all.txt").read_text(encoding="utf-8").splitlines():
        code, start, end = line.split("\t")[:3]
        if re.match(r"^(sh(?:60|68)|sz(?:00|30)|bj[489])", code.lower()):
            instruments.setdefault(code.lower(), []).append((start, end))
    intervals = {code: [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in ranges]
                 for code, ranges in instruments.items()}
    universe_path = out / "universe.json"
    serializable = {code: [list(pair) for pair in ranges] for code, ranges in instruments.items()}
    if universe_path.exists() and json.loads(universe_path.read_text(encoding="utf-8")) != serializable:
        raise RuntimeError("冻结股票池发生变化")
    universe_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    names = list(factors) + ["label", "open", "close", "high", "low", "volume", "factor"]
    fields = [factors[name]["expression"] for name in factors]
    fields += ["Ref($open, -11)/Ref($open, -1)-1", "$open", "$close", "$high", "$low", "$volume", "$factor"]
    final_date = pd.Timestamp((source / "calendars/day.txt").read_text(encoding="utf-8").splitlines()[-1])
    chunks = (("2019", "2019-01-01", "2020-12-31"),
              ("2021", "2021-01-01", "2022-12-31"),
              ("2023", "2023-01-01", "2024-12-31"),
              ("2025", "2025-01-01", str(final_date.date())))
    for year, start, end in chunks:
        if year not in args.years:
            continue
        path = out / f"new_{year}.parquet"
        if path.exists():
            print(json.dumps({"stage": "skipped_existing", "year": year, "path": str(path)}), flush=True)
            continue
        print(json.dumps({"stage": "extract", "year": year, "start": start, "end": end}), flush=True)
        with qlib_generation_read_lock(source):
            if digest(source / "current.json") != qlib_pointer_signature:
                raise RuntimeError("Qlib输入代际已变化")
            frame = D.features(intervals, fields, start_time=start, end_time=end, freq="day")
        frame.columns = names
        frame = frame.reorder_levels(["datetime", "instrument"]).sort_index().astype(np.float32)
        if frame.index.has_duplicates:
            raise RuntimeError("抽取股票日重复")
        temporary = path.with_suffix(".tmp")
        frame.to_parquet(temporary, compression="zstd")
        temporary.replace(path)
        print(json.dumps({"stage": "complete", "year": year, "rows": len(frame), "sha256": digest(path)}), flush=True)
        del frame
        gc.collect()
    if (json.loads((cache / "current.json").read_text(encoding="utf-8"))["generation"] != generation
            or digest(calendar_file) != calendar_signature or digest(source / "current.json") != qlib_pointer_signature):
        raise RuntimeError("抽取期间输入代际变化，所有结果不可用于模型比较")
    if all((out / f"new_{year}.parquet").exists() for year in ("2019", "2021", "2023", "2025")):
        (out / "extraction_complete.json").write_text(json.dumps({"research_only": True, "generation": generation,
                                                                   "calendar_sha256": calendar_signature}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
