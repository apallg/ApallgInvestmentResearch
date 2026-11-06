import argparse
import hashlib
import json
import os
import pickle
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

from scripts.research_feature_ops import aligned_correlation, rolling_last_rank


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 2026, 3407), default=42)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    old = ROOT / "mlruns/656749665990832865/b0e89e44065d4a5f9028956e09028af9"
    factors = json.loads((source / "factors.json").read_text(encoding="utf-8"))
    with (old / "artifacts/factor_keys.json").open("rb") as handle:
        keys = json.loads(pickle.load(handle))["custom_factor_keys"]
    if list(factors) != list(keys):
        raise RuntimeError("旧工件因子顺序不一致")
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    mismatches = [key for key, detail in factors.items() if resolve_factor_expression(key)[0] != detail["expression"]]
    if mismatches:
        raise RuntimeError(f"当前因子解析式与冻结表达式不同: {mismatches}")
    old_predictions = pd.read_pickle(old / "artifacts/pred.pkl")
    instruments = sorted(old_predictions.index.get_level_values("instrument").unique())
    if len(instruments) != 2985:
        raise RuntimeError("旧模型股票池无法从预测工件恢复")
    actual = json.loads((output.parent / "old_actual_model_params.json").read_text(encoding="utf-8"))
    if actual["params_pkl_sha256"] != digest(old / "artifacts/params.pkl"):
        raise RuntimeError("旧模型参数工件已变化")
    identity = json.loads((source / "input_identity.json").read_text(encoding="utf-8"))
    provider = ROOT / "data/qlib_data"
    cache = source / "fund_cache"
    if digest(provider / "current.json") != identity["qlib_pointer_sha256"] or digest(provider / "calendars/day.txt") != identity["qlib_calendar_sha256"]:
        raise RuntimeError("Qlib代际或日历变化")
    cache_generation = json.loads((cache / "current.json").read_text(encoding="utf-8"))["generation"]
    if cache_generation != identity["generation"] or digest(cache / "generations" / cache_generation / "fund_features.parquet") != identity["data_sha256"]:
        raise RuntimeError("隔离FundVal缓存代际变化")
    import qlib
    from qlib.constant import REG_CN
    from qlib.data.ops import Operators, Rolling, Corr as NativeCorr
    from core.data import fundamental
    from core.adapters.qlib import custom_handler

    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=2, joblib_backend="threading",
              expression_cache=None, dataset_cache=None,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": (output / "mlruns").as_uri(), "default_exp_name": "research_only"}})
    fundamental.CACHE_DIR = cache
    if not fundamental.preload_cache():
        raise RuntimeError("隔离FundVal缓存无法加载")
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
    from core.adapters.qlib.models import PARAM_PROFILES
    from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion

    params = dict(actual["lightgbm_params"])
    params["num_threads"] = 4
    params.pop("verbose_eval", None)
    PARAM_PROFILES["old_reference_isolated"] = {"lightgbm": params}
    if args.preflight_only:
        report = {"status": "preflight_passed", "old_run_id": old.name,
                  "stocks": len(instruments), "factors": len(keys),
                  "current_fund_cache_generation": identity["generation"],
                  "same_date_fusion_training_started": False}
        (output / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return
    result = train_lgb_lstm_fusion(
        factor_keys=keys, instruments=instruments,
        train_period=("2019-01-01", "2025-12-31"),
        valid_period=("2026-01-01", "2026-06-30"),
        test_period=("2026-07-01", "2026-08-31"),
        label_horizon=10, label_norm="csrank", fusion_weight=0.5,
        param_profile="old_reference_isolated", lstm_epochs=80,
        label_mode="tradable_open_to_open", seed=args.seed,
        calibration_period=("2026-04-01", "2026-06-30"),
    )
    prediction = result["fused_pred"]
    if prediction.isna().any() or not np.isfinite(prediction).all():
        raise RuntimeError("安全重训融合预测存在缺失")
    current = pd.read_parquet(source / "new_2025.parquet", columns=["label"],
                              filters=[("datetime", ">=", pd.Timestamp("2026-07-01")),
                                       ("datetime", "<=", pd.Timestamp("2026-08-31"))])["label"]
    pair = pd.concat({"old_score": old_predictions, "new_score": prediction,
                      "raw_label": current}, axis=1).dropna()
    if pair.empty:
        raise RuntimeError("新旧模型没有共同股票日")
    pair.to_parquet(output / f"same_date_pair_{args.seed}.parquet", compression="zstd")
    daily = []
    for day, section in pair.groupby(level="datetime"):
        daily.append({"date": str(day.date()), "stocks": len(section),
                      "old_rank_ic": section["old_score"].corr(section["raw_label"], method="spearman"),
                      "new_rank_ic": section["new_score"].corr(section["raw_label"], method="spearman"),
                      "old_top5_gross": section.nlargest(5, "old_score")["raw_label"].mean(),
                      "new_top5_gross": section.nlargest(5, "new_score")["raw_label"].mean()})
    daily = pd.DataFrame(daily)
    daily["top5_difference"] = daily["new_top5_gross"] - daily["old_top5_gross"]
    daily.to_csv(output / f"same_date_daily_{args.seed}.csv", index=False)
    calendar = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    test_days = calendar[(calendar >= "2026-07-01") & (calendar <= "2026-08-31")]
    signal_dates = [day for day in test_days[::10] if calendar.get_loc(day) + 11 < len(calendar) and calendar[calendar.get_loc(day) + 11] <= test_days[-1]]
    nonoverlap = daily.loc[daily["date"].isin([str(day.date()) for day in signal_dates])]
    report = {"status": "diagnostic_same_date_safe_retrain", "old_run_id": old.name,
              "seed": args.seed, "old_seed_recovered": False, "same_structure": True,
              "same_input_generation": False, "same_training_boundary": False,
              "old_boundary_leakage": True, "current_fund_cache_generation": identity["generation"],
              "common_rows": len(pair), "common_days": pair.index.get_level_values("datetime").nunique(),
              "old_only_rows": int(len(old_predictions.index.difference(pair.index))),
              "new_only_rows": int(len(prediction.index.difference(pair.index))),
              "fused_metrics": result["fused"], "lgb_metrics": result["lgb"],
              "lstm_metrics": result["lstm"], "test_start": "2026-07-01", "test_end": "2026-08-31",
              "valid_purged_to": "2026-03-16", "calibration_purged_to": "2026-06-12",
              "old_common_rank_ic": float(daily["old_rank_ic"].mean()),
              "new_common_rank_ic": float(daily["new_rank_ic"].mean()),
              "daily_top5_gross_difference": float(daily["top5_difference"].mean()),
              "nonoverlap_10d_blocks": len(nonoverlap),
              "nonoverlap_top5_gross_difference": float(nonoverlap["top5_difference"].mean()) if len(nonoverlap) else None}
    (output / f"same_date_fusion_{args.seed}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / f"same_date_lgb_{args.seed}.txt").write_text(result["lgb_model"].model.model_to_string(), encoding="utf-8")
    with (output / f"same_date_lstm_{args.seed}.pkl").open("wb") as handle:
        pickle.dump({"state": result["lstm_state"], "scaler": result["lstm_scaler"],
                     "factor_keys": keys, "seq_len": 20, "fusion_weight": 0.5}, handle, protocol=5)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
