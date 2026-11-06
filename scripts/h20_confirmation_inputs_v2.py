import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.shared import config
from core.adapters.qlib.generation import qlib_generation_read_lock
from scripts.h20_confirmation_v2 import OUTPUT, digest, save_json


def preprocess(panel, feature_keys):
    from qlib.data.dataset.processor import Fillna, ProcessInf
    from joblib import parallel_backend
    from core.adapters.qlib.custom_processors import CSWinsorize

    raw = panel[feature_keys].replace([np.inf, -np.inf], np.nan)
    wrapped = CSWinsorize()(pd.concat({"feature": raw}, axis=1))
    if np.isinf(wrapped.to_numpy()).any():
        with parallel_backend("threading"):
            wrapped = ProcessInf()(wrapped)
    result = Fillna()(wrapped)["feature"].astype(np.float32)
    if result.isna().any().any() or not np.isfinite(result.to_numpy()).all():
        raise RuntimeError("原40因子预处理后仍有缺失或非有限值")
    return result


def source_identity():
    provider = Path(config.QLIB_DATA_DIR)
    cache = ROOT / "data/fund_cache"
    cache_pointer = json.loads((cache / "current.json").read_text(encoding="utf-8"))
    data_file = cache / "generations" / cache_pointer["generation"] / "fund_features.parquet"
    files = {"qlib_pointer": Path(config.QLIB_DATA_ROOT) / "current.json",
             "qlib_manifest": provider / "generation.json",
             "calendar": provider / "calendars/day.txt",
             "instruments": provider / "instruments/all.txt",
             "fund_cache_pointer": cache / "current.json",
             "fund_cache_data": data_file,
             "factor_expressions": ROOT / "data/experiments/fundval_repair_research_20261026/factors.json",
             "verified_delist_adjudications": ROOT / "data/historical_repair/adjudication/verified_delist_date_adjudications.json",
             "input_code": Path(__file__),
             "preprocess_code": ROOT / "core/adapters/qlib/custom_processors.py"}
    hashes = {name: digest(path) for name, path in files.items()}
    return provider, cache, files, hashes


def setup_qlib(provider, cache):
    import qlib
    from qlib.constant import REG_CN
    from qlib.data.ops import Operators, Rolling, Corr as NativeCorr
    from core.adapters.qlib import custom_handler
    from core.data import fundamental
    from scripts.research_feature_ops import rolling_last_rank, aligned_correlation

    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=2, joblib_backend="threading",
              expression_cache=None, dataset_cache=None,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": (OUTPUT / "mlruns").as_uri(), "default_exp_name": "h20_research_only"}})
    fundamental.CACHE_DIR = cache
    if not fundamental.preload_cache():
        raise RuntimeError("当前 FundVal 缓存无法加载")
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
    from qlib.data import D
    return D


def extract(args):
    from core.data import Database

    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    as_of = pd.Timestamp(args.as_of)
    provider, cache, files, hashes = source_identity()
    calendar = pd.DatetimeIndex(pd.to_datetime(files["calendar"].read_text(encoding="utf-8").splitlines()))
    if not len(calendar) or calendar[-1] != as_of:
        raise RuntimeError("当前Qlib交易日历尚未到达指定截止日")
    factors = json.loads(files["factor_expressions"].read_text(encoding="utf-8"))
    if list(factors) != protocol["factor_keys"] or len(factors) != 40:
        raise RuntimeError("冻结因子顺序与实际表达式不符")
    latest = Database.fetchone("SELECT MAX(trade_date) AS d FROM daily_quotes")
    if pd.Timestamp(latest["d"]) < as_of:
        raise RuntimeError("数据库日行情未到指定截止日")
    destination = OUTPUT / "inputs" / ("samples" if args.limit_stocks else "full") / args.as_of
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get("source_sha256", {}).get(name) != value for name, value in hashes.items()) or manifest.get("status") not in ("engineering_sample", "qualified_for_research_training"):
            raise RuntimeError("已有输入签名冲突或不完整")
        print(json.dumps({"status": manifest["status"], "as_of": args.as_of}, ensure_ascii=False))
        return
    destination.mkdir(parents=True, exist_ok=True)
    state = destination / "extract_state.json"
    if state.exists():
        if json.loads(state.read_text(encoding="utf-8")) != hashes:
            raise RuntimeError("未完成输入目录与当前来源签名冲突")
    else:
        if any(destination.iterdir()):
            raise RuntimeError("未完成输入目录缺少签名状态")
        save_json(state, hashes)
    start = as_of - pd.DateOffset(months=55)
    intervals = {}
    for line in files["instruments"].read_text(encoding="utf-8").splitlines():
        code, first, last = line.split("\t")[:3]
        if re.match(r"^(sh(?:60|68)|sz(?:00|30))", code.lower()):
            intervals.setdefault(code.lower(), []).append((pd.Timestamp(first), pd.Timestamp(last)))
    if args.limit_stocks:
        intervals = dict(list(sorted(intervals.items()))[:args.limit_stocks])
    if not intervals:
        raise RuntimeError("沪深原研究股票池为空")
    D = setup_qlib(provider, cache)
    fields = [factors[key]["expression"] for key in protocol["factor_keys"]] + ["$open", "$close", "$volume"]
    names = protocol["factor_keys"] + ["open", "close", "volume"]
    parts = []
    for year in range(start.year, as_of.year + 1):
        low = max(start, pd.Timestamp(f"{year}-01-01"))
        high = min(as_of, pd.Timestamp(f"{year}-12-31"))
        if high < low:
            continue
        part = destination / f"raw_{year}.parquet"
        part_meta = part.with_suffix(".json")
        if part.exists() or part_meta.exists():
            if not part.exists() or not part_meta.exists() or json.loads(part_meta.read_text(encoding="utf-8")).get("sha256") != digest(part):
                raise RuntimeError(f"年度分片签名冲突: {year}")
            frame = pd.read_parquet(part)
        else:
            with qlib_generation_read_lock(config.QLIB_DATA_ROOT):
                if any(digest(files[name]) != hashes[name] for name in ("qlib_pointer", "calendar", "instruments", "fund_cache_pointer")):
                    raise RuntimeError("抽取中输入数据代发生变化")
                frame = D.features(intervals, fields, start_time=str(low.date()), end_time=str(high.date()), freq="day")
            frame.columns = names
            frame = frame.reorder_levels(["datetime", "instrument"]).sort_index().astype(np.float32)
            if frame.index.has_duplicates:
                raise RuntimeError("因子面板存在重复股票日")
            temporary = part.with_suffix(".tmp")
            frame.to_parquet(temporary, compression="zstd")
            temporary.replace(part)
            save_json(part_meta, {"sha256": digest(part), "rows": len(frame), "start": str(low.date()), "end": str(high.date())})
        parts.append(frame)
    panel = pd.concat(parts).sort_index()
    if panel.index.has_duplicates:
        raise RuntimeError("跨年因子面板存在重复股票日")
    panel = panel.loc[panel["close"].gt(0) & panel["volume"].gt(0)]
    if not (panel.index.get_level_values("datetime") == as_of).any():
        raise RuntimeError("因子面板缺少截止日")
    if any(digest(files[name]) != hashes[name] for name in ("qlib_pointer", "calendar", "instruments", "fund_cache_pointer", "fund_cache_data")):
        raise RuntimeError("抽取完成后输入数据代发生变化")
    if not np.isfinite(panel.loc[panel.index.get_level_values("datetime") == as_of, "open"]).any():
        raise RuntimeError("截止日开盘价全部无效")
    if not args.limit_stocks:
        denominator, sources, future_days = build_denominator(panel, calendar, as_of)
        expected = denominator.index
        if not expected.isin(panel.index).all() or panel.loc[expected].index.has_duplicates:
            raise RuntimeError("共同股票池重复")
        valid_today = (denominator.index.get_level_values("datetime") == as_of)
        today = panel.loc[denominator.index[valid_today]]
        if not (today["open"].gt(0) & today["close"].gt(0) & today["volume"].gt(0)).all():
            raise RuntimeError("当日预期可交易股票池与Qlib行情不一致")
        denom_path = destination / "denominator.parquet"
        denominator.to_parquet(denom_path, compression="zstd")
        save_json(destination / "denominator_sources.json", sources)
        hashes["denominator_sources"] = digest(destination / "denominator_sources.json")
        calendar = calendar.append(future_days)
    output = destination / "panel.parquet"
    panel.to_parquet(output, compression="zstd")
    calendar_path = destination / "calendar.txt"
    calendar_path.write_text("\n".join(calendar.astype(str)) + "\n", encoding="utf-8")
    missing_history = sources["historical_expected_missing_rows"] if not args.limit_stocks else 0
    save_json(manifest_path, {"status": "engineering_sample" if args.limit_stocks else
                              "blocked_historical_coverage" if missing_history else "qualified_for_research_training",
                              "as_of": args.as_of, "source_sha256": hashes,
                              "panel_sha256": digest(output), "calendar_sha256": digest(calendar_path),
                              "denominator_sha256": digest(denom_path) if not args.limit_stocks else None,
                              "neutralization": False,
                              "historical_expected_missing_rows": missing_history,
                              "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "rows": len(panel), "stocks": len(intervals), "first_date": str(panel.index.get_level_values("datetime").min().date()),
                              "last_date": args.as_of, "research_only": True, "production_qualified": False})
    print(json.dumps({"status": "panel_extracted", "rows": len(panel), "directory": str(destination)}, ensure_ascii=False), flush=True)


def build_denominator(panel, calendar, as_of):
    from core.data import Database
    from core.data.sources.historical_universe import fetch_tushare_open_calendar, fetch_tushare_suspensions

    old = ROOT / "data/historical_repair/denominator/d8f7e2602f5b"
    pieces = [pd.read_parquet(path, columns=["code", "trade_date", "full_day_suspended"])
              for path in sorted(old.glob("expected_keys_*.parquet"))]
    historical = pd.concat(pieces, ignore_index=True)
    historical["trade_date"] = pd.to_datetime(historical["trade_date"])
    last_old = historical["trade_date"].max()
    first_panel = panel.index.get_level_values("datetime").min()
    historical = historical.loc[(historical["trade_date"] >= first_panel)
                                & (historical["trade_date"] <= last_old)
                                & ~historical["full_day_suspended"].fillna(False).astype(bool)]
    old_index = pd.MultiIndex.from_arrays([historical["trade_date"],
        historical["code"].astype(str).str.zfill(6).map(lambda c: ("sh" if c.startswith("6") else "sz") + c)],
        names=["datetime", "instrument"])
    recent = calendar[calendar > last_old]
    rows = Database.fetchall("SELECT observed_at,ts_code,list_status,list_date,delist_date "
                             "FROM stock_identity_observations WHERE observed_at >= %s ORDER BY observed_at,ts_code",
                             ((last_old - pd.Timedelta(days=7)).date(),))
    observations = pd.DataFrame(rows)
    if observations.empty:
        raise RuntimeError("缺少近期身份原始观测")
    observed_times = sorted(observations["observed_at"].unique())
    future = fetch_tushare_open_calendar((as_of + pd.Timedelta(days=1)).date(),
                                          (as_of + pd.Timedelta(days=32)).date())
    future_days = pd.DatetimeIndex(future["cal_date"])
    future_days = future_days[future_days > as_of]
    if not len(future_days):
        raise RuntimeError("无法获得下一交易日官方日历")
    susp = fetch_tushare_suspensions((last_old + pd.Timedelta(days=1)).date(), as_of.date())
    susp_keys = set(zip(susp["trade_date"], susp["ts_code"]))
    adjudication_path = ROOT / "data/historical_repair/adjudication/verified_delist_date_adjudications.json"
    adjudications = json.loads(adjudication_path.read_text(encoding="utf-8"))["records"]
    verified_delist = {row["ts_code"]: pd.Timestamp(row["official_delist_date"])
                       for row in adjudications
                       if row.get("status") == "verified"
                       and row.get("classification") == "delisted_on_registry_date_nontrading"
                       and row.get("official_delist_date")}
    expected = []
    daily_audit = []
    for day in recent:
        later = calendar[calendar > day] if day < as_of else future_days
        next_day = later[0] if len(later) else as_of + pd.Timedelta(days=32)
        deadline_utc = (pd.Timestamp(next_day.date()).tz_localize("Asia/Shanghai")
                        + pd.Timedelta(hours=9, minutes=25)).tz_convert("UTC").tz_localize(None)
        allowed = [value for value in observed_times if value <= deadline_utc]
        if not allowed:
            raise RuntimeError(f"{day.date()} 没有当时可用的身份原始快照")
        selected = allowed[-1]
        source = observations.loc[observations["observed_at"] == selected]
        rows_for_day = []
        suspended = 0
        for row in source.itertuples(index=False):
            code = str(row.ts_code)
            if not code.endswith((".SH", ".SZ")) or not re.match(r"^(?:6\d{5}\.SH|(?:00|30)\d{4}\.SZ)$", code):
                continue
            if pd.isna(row.list_date) or pd.Timestamp(row.list_date) > day:
                continue
            if code in verified_delist and verified_delist[code] <= day:
                continue
            if row.list_status != "L" and (pd.isna(row.delist_date) or pd.Timestamp(row.delist_date) <= day):
                continue
            if (day, code) in susp_keys:
                suspended += 1
                continue
            instrument = ("sh" if code.endswith(".SH") else "sz") + code[:6]
            rows_for_day.append((day, instrument))
        expected.extend(rows_for_day)
        daily_audit.append({"date": str(day.date()), "identity_observed_at_utc": str(selected),
                            "expected": len(rows_for_day), "suspended": suspended})
    recent_index = pd.MultiIndex.from_tuples(expected, names=old_index.names)
    old_valid = old_index.intersection(panel.index)
    denominator_index = old_valid.append(recent_index).unique().sort_values()
    denominator = pd.DataFrame({"eligible": np.ones(len(denominator_index), dtype=np.uint8)}, index=denominator_index)
    if denominator.index.has_duplicates:
        raise RuntimeError("共同股票池存在重复证券日期")
    actual = panel.loc[panel.index.get_level_values("datetime").isin(recent)]
    tradable = actual.index[(actual["open"].gt(0) & actual["close"].gt(0) & actual["volume"].gt(0)).to_numpy()]
    missing = recent_index.difference(tradable)
    if len(missing):
        raise RuntimeError(f"近期预期股票池缺行情: {missing[:20].tolist()}，共{len(missing)}行")
    sources = {"historical_denominator_manifest_sha256": digest(old / "manifest.json"),
               "historical_denominator_pit_verified": False,
               "verified_delist_adjudications_sha256": digest(adjudication_path),
               "adjudicated_delists_in_recent_window": {code: str(value.date()) for code, value in verified_delist.items()
                                                        if last_old < value <= as_of},
               "identity_observations": daily_audit,
               "suspension_rows": susp.to_dict(orient="records"),
               "future_calendar": future.to_dict(orient="records"),
               "new_dates": len(recent), "recent_expected_rows": len(recent_index),
               "historical_expected_missing_rows": len(old_index) - len(old_valid),
               "research_only": True}
    return denominator, sources, future_days


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--limit-stocks", type=int)
    extract(parser.parse_args(argv))


if __name__ == "__main__":
    main()
