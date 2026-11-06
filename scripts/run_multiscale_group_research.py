import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiscale_research_features import build_multiscale_features, build_style_groups


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("MULTISCALE_SOURCE", ROOT / "data/experiments/fundval_repair_research_20261026"))
OLD = ROOT / "mlruns/656749665990832865/b0e89e44065d4a5f9028956e09028af9"
OUTPUT = Path(os.environ.get("MULTISCALE_OUTPUT", ROOT / "data/experiments/multiscale_group_research_20261103"))
SEEDS = (42, 2026, 3407)
COMPARISONS = (("L", "D"), ("W", "L"), ("M", "W"), ("M", "D"),
               ("U", "D"), ("G6", "U"), ("GI", "U"), ("GS", "U"), ("GI", "G6"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_contract():
    params = {}
    for path in (OLD / "params").iterdir():
        if path.is_file():
            params[path.name] = path.read_text(encoding="utf-8").strip()
    with (OLD / "artifacts/factor_keys.json").open("rb") as handle:
        factor_keys = json.loads(pickle.load(handle))["custom_factor_keys"]
    expected = {
        "model.class": "LGB+LSTM", "fusion_weight": "0.5", "label_horizon": "10",
        "label_mode": "tradable_open_to_open", "label_norm": "csrank",
        "dataset.kwargs.segments.train": "('2019-01-01', '2025-12-31')",
        "dataset.kwargs.segments.valid": "('2026-01-01', '2026-06-30')",
    }
    mismatches = {key: params.get(key) for key, value in expected.items() if params.get(key) != value}
    if mismatches:
        raise RuntimeError(f"指定旧工件配置不符: {mismatches}")
    factors = json.loads((SOURCE / "factors.json").read_text(encoding="utf-8"))
    if list(factors) != factor_keys:
        raise RuntimeError("冻结因子及顺序与指定旧工件不一致")
    return params, factor_keys, factors


def fold_windows(calendar):
    windows = {}
    for number in range(1, 9):
        test_start = pd.Timestamp("2023-01-01") + pd.DateOffset(months=6 * (number - 1))
        valid_start = test_start - pd.DateOffset(months=6)
        calibration_start = test_start - pd.DateOffset(months=3)
        train_start = valid_start - pd.DateOffset(years=3)
        train_end = calendar[calendar < valid_start][-12]
        valid_end = calendar[calendar < calibration_start][-12]
        calibration_end = calendar[calendar < test_start][-12]
        test_end = min(test_start + pd.DateOffset(months=6) - pd.Timedelta(days=1), pd.Timestamp("2026-09-15"))
        windows[f"W{number}"] = {"train": [str(train_start.date()), str(train_end.date())],
                                 "valid": [str(valid_start.date()), str(valid_end.date())],
                                 "calibrate": [str(calibration_start.date()), str(calibration_end.date())],
                                 "test": [str(test_start.date()), str(test_end.date())],
                                 "purged_label_end": {
                                     "train": str(calendar[calendar.get_loc(train_end) + 11].date()),
                                     "valid": str(calendar[calendar.get_loc(valid_end) + 11].date()),
                                     "calibrate": str(calendar[calendar.get_loc(calibration_end) + 11].date()),
                                 }}
    return windows


def preflight():
    from core.data.connection import Database

    params, factor_keys, factors = reference_contract()
    calendar_path = ROOT / "data/qlib_data/calendars/day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text(encoding="utf-8").splitlines()))
    source_paths = [SOURCE / f"new_{year}.parquet" for year in (2019, 2021, 2023, 2025)]
    context_path = SOURCE / "context/context_features.parquet"
    coverage = json.loads(context_path.with_name("coverage.json").read_text(encoding="utf-8"))
    identity = json.loads((SOURCE / "input_identity.json").read_text(encoding="utf-8"))
    if coverage["membership_interval_conflicts"] or coverage["feature_contract_version"] != 2:
        raise RuntimeError("历史行业或上下文契约不合格")
    if coverage["last_date"] < "2026-09-15":
        raise RuntimeError("历史上下文未覆盖全部固定折")
    if any(not path.exists() for path in source_paths):
        raise RuntimeError("冻结的四个行情特征分片不完整")
    old_predictions = pd.read_pickle(OLD / "artifacts/pred.pkl")
    old_labels = pd.read_pickle(OLD / "artifacts/label.pkl")
    if not old_predictions.index.equals(old_labels.index):
        raise RuntimeError("旧工件预测和标签索引不一致")
    if old_predictions.index.get_level_values("datetime").max() > pd.Timestamp("2026-08-31"):
        raise RuntimeError("旧工件预测覆盖与记录不符")
    source_end = str(pd.read_parquet(source_paths[-1], columns=["close"]).index.get_level_values("datetime").max().date())
    membership_rows = Database.fetchall("SELECT ts_code,index_code,in_date,out_date FROM stock_index_membership ORDER BY ts_code,in_date,index_code")
    membership_snapshot = json.dumps(membership_rows, ensure_ascii=False, default=str, separators=(",", ":"))
    membership_path = OUTPUT / "stock_index_membership_snapshot.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if membership_path.exists() and membership_path.read_text(encoding="utf-8") != membership_snapshot:
        raise RuntimeError("行业归属快照变化；不可在同一实验目录续跑")
    membership_path.write_text(membership_snapshot, encoding="utf-8")
    manifest = {
        "status": "preflight_passed", "research_only": True, "deployment_eligible": False,
        "strict_pit_verified": False, "execution_verified": False,
        "old_run_id": OLD.name, "old_experiment_id": OLD.parent.name,
        "old_artifact_prediction_period": ["2026-07-01", "2026-08-31"],
        "old_artifact_prediction_rows": len(old_predictions),
        "reference_params": params, "factor_keys": factor_keys,
        "factor_expressions": factors,
        "source_panel_end": source_end, "current_qlib_calendar_end": str(calendar.max().date()),
        "source_is_current_generation": source_end == str(calendar.max().date()),
        "source_warning": "冻结面板早于当前Qlib日历；历史再验证有效，不能标作当前代际训练" if source_end != str(calendar.max().date()) else None,
        "fund_cache_contract": identity,
        "context_coverage": {key: coverage[key] for key in ("rows", "last_date", "unknown_group_share", "industry_vintage", "industry_pit_verified", "missing_indicator_rate_before_value_fill")},
        "folds": fold_windows(calendar), "seeds": list(SEEDS),
        "comparisons": [f"{left}-{right}" for left, right in COMPARISONS],
        "source_files_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [*source_paths, context_path,
            context_path.with_name("coverage.json"), SOURCE / "factors.json", SOURCE / "input_identity.json",
            calendar_path, ROOT / "data/qlib_data/current.json",
            *sorted((ROOT / "data/historical_repair/denominator/d8f7e2602f5b").glob("expected_keys_*.parquet"))]},
        "membership_snapshot_sha256": sha256(membership_path),
        "old_artifacts_sha256": {path.name: sha256(path) for path in (OLD / "artifacts").iterdir() if path.is_file()},
        "old_params_sha256": {path.name: sha256(path) for path in (OLD / "params").iterdir() if path.is_file()},
        "code_files_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (ROOT / "scripts/multiscale_research_features.py",
                                                                                     ROOT / "scripts/train_multiscale_group_research.py",
                                                                                     ROOT / "scripts/analyze_multiscale_group_research.py")},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("冻结实验契约发生变化；禁止覆盖原结果")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(path), "source_panel_end": source_end,
                      "current_qlib_calendar_end": manifest["current_qlib_calendar_end"],
                      "source_is_current_generation": manifest["source_is_current_generation"],
                      "old_prediction_rows": len(old_predictions)}, ensure_ascii=False))


def feature_probe():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    path = SOURCE / "new_2025.parquet"
    panel = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume", "label"],
                            filters=[("datetime", ">=", pd.Timestamp("2026-07-01")), ("datetime", "<=", pd.Timestamp("2026-09-15"))])
    names = panel.index.get_level_values("instrument").unique()[:20]
    panel = panel.loc[panel.index.get_level_values("instrument").isin(names)]
    warmup = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"],
                             filters=[("datetime", ">=", pd.Timestamp("2025-01-01")), ("datetime", "<", pd.Timestamp("2026-07-01"))])
    warmup = warmup.loc[warmup.index.get_level_values("instrument").isin(names)]
    joined = pd.concat([warmup, panel]).sort_index()
    features = build_multiscale_features(joined[["open", "high", "low", "close", "volume"]], calendar)
    context = pd.read_parquet(SOURCE / "context/context_features.parquet",
                              columns=["context_log_circ_mv", "context_log_amount_mean60"],
                              filters=[("datetime", ">=", pd.Timestamp("2025-01-01")), ("datetime", "<=", pd.Timestamp("2026-09-15"))])
    context = context.reindex(joined.index)
    styles = build_style_groups(context, features["daily_vol_60"], calendar)
    result = {"rows": len(features), "stocks": len(names), "date_start": str(features.index.get_level_values("datetime").min().date()),
              "date_end": str(features.index.get_level_values("datetime").max().date()),
              "feature_count": len(features.columns),
              "non_missing_rates": features.notna().mean().to_dict(),
              "style_group_counts": {str(k): int(v) for k, v in styles.value_counts(dropna=False).items()},
              "source_files_sha256": manifest["source_files_sha256"]}
    (OUTPUT / "feature_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("non_missing_rates", "source_files_sha256")}, ensure_ascii=False))


def compare_old():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    old_predictions = pd.read_pickle(OLD / "artifacts/pred.pkl")
    old_labels = pd.read_pickle(OLD / "artifacts/label.pkl")
    source = pd.read_parquet(SOURCE / "new_2025.parquet", columns=["label", "close", "volume"],
                             filters=[("datetime", ">=", pd.Timestamp("2026-07-01")), ("datetime", "<=", pd.Timestamp("2026-08-31"))])
    matched = source.reindex(old_labels.index)
    comparable = pd.concat({"old": old_labels, "current_raw": matched["label"].replace([np.inf, -np.inf], np.nan)}, axis=1).dropna()
    daily_correlations = comparable.groupby(level="datetime").apply(lambda frame: frame["old"].corr(frame["current_raw"], method="spearman"))
    old_frame = pd.concat({"score": old_predictions, "label": matched["label"]}, axis=1).dropna()
    old_days = []
    for day, frame in old_frame.groupby(level="datetime"):
        old_days.append({"date": str(day.date()), "rank_ic": frame["score"].corr(frame["label"], method="spearman"),
                         "top5_gross": frame.nlargest(5, "score")["label"].mean(),
                         "top50_gross": frame.nlargest(50, "score")["label"].mean(),
                         "stocks": len(frame)})
    old_daily = pd.DataFrame(old_days)
    old_daily.to_csv(OUTPUT / "old_daily_metrics.csv", index=False)
    old_train_label_end = calendar[calendar.get_loc(pd.Timestamp("2025-12-31")) + 11]
    old_valid_label_end = calendar[calendar.get_loc(pd.Timestamp("2026-06-30")) + 11]
    report = {
        "old_run_id": OLD.name, "old_predictions": len(old_predictions), "common_finite_labels": len(comparable),
        "label_comparison": "daily Spearman of saved normalized label versus current raw return; value equality is inapplicable",
        "daily_rank_correlation_mean": float(daily_correlations.mean()),
        "daily_rank_correlation_min": float(daily_correlations.min()),
        "daily_rank_correlation_defined_dates": int(daily_correlations.notna().sum()),
        "saved_old_prediction_rank_ic_on_current_raw_label": float(old_daily["rank_ic"].mean()),
        "saved_old_prediction_top5_daily_gross_on_current_raw_label": float(old_daily["top5_gross"].mean()),
        "saved_old_prediction_evaluation_dates": len(old_daily),
        "old_train_last_label_end": str(old_train_label_end.date()),
        "old_valid_last_label_end": str(old_valid_label_end.date()),
        "old_train_valid_label_overlap": bool(old_train_label_end >= pd.Timestamp("2026-01-01")),
        "old_valid_test_label_overlap": bool(old_valid_label_end >= pd.Timestamp("2026-07-01")),
        "source_is_current_generation": manifest["source_is_current_generation"],
        "old_artifact_is_compatible_with_current_input": False,
        "reason": "旧工件缺少当前FundVal输入契约；历史预测可诊断，不能静默接入新缓存",
    }
    (OUTPUT / "old_artifact_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def execution_preflight():
    from core.data.connection import Database

    dates = Database.fetchone("SELECT COUNT(DISTINCT as_of) dates, MIN(as_of) first_date, MAX(as_of) last_date FROM security_reference_history")
    quotes = Database.fetchone("SELECT MIN(trade_date) first_date, MAX(trade_date) last_date FROM daily_quotes")
    report = {
        "required_period": ["2023-01-01", "2026-09-15"],
        "daily_quotes_period": [str(quotes["first_date"]), str(quotes["last_date"])],
        "security_reference_dates": int(dates["dates"]),
        "security_reference_period": [str(dates["first_date"]), str(dates["last_date"])],
        "historical_st_and_limit_rules_verified": False,
        "execution_verified": False,
        "account_result_allowed": False,
        "reason": "历史ST/证券状态只有稀疏快照，不能可靠判断所有执行日的涨跌停与可成交性",
        "recovery": "补齐并核验整个评估区间逐日证券状态及原始价格口径，随后用同一执行器重放所有方案",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "execution_readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=str))


def actual_old_params():
    with (OLD / "artifacts/params.pkl").open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "params"):
        raise RuntimeError("旧模型参数工件结构不符")
    report = {"old_run_id": OLD.name, "params_pkl_sha256": sha256(OLD / "artifacts/params.pkl"),
              "lightgbm_params": dict(model.params)}
    (OUTPUT / "old_actual_model_params.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "feature-probe", "compare-old", "execution-preflight", "actual-old-params"))
    args = parser.parse_args()
    {"preflight": preflight, "feature-probe": feature_probe, "compare-old": compare_old,
     "execution-preflight": execution_preflight, "actual-old-params": actual_old_params}[args.stage]()
