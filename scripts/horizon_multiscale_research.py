import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "true")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.run_multiscale_group_research import sha256

SOURCE = ROOT / "data/experiments/multiscale_group_research_20261103/current_isolated_input"
REFERENCE = ROOT / "data/experiments/multiscale_group_research_current_v2_20261103"
PRIOR = ROOT / "data/experiments/monthly_group_research_v4_20261103"
OUTPUT = Path(os.environ.get("HORIZON_RESEARCH_OUTPUT", ROOT / "data/experiments/horizon_multiscale_research_v2_4m"))
SEEDS = (42, 2026, 3407)
CLASSIFICATION_ARMS = ("U_D", "B3_D", "A2_D", "R4_D", "U_M", "B3_M", "A2_M", "R4_M")
SEQUENCE_ARMS = ("N0", "NL", "NS", "NF", "NT", "NR")


def calendar_index():
    return pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))


def fold_windows(calendar):
    windows = {}
    for number in range(1, 9):
        test_start = pd.Timestamp("2023-01-01") + pd.DateOffset(months=6 * (number - 1))
        calibrate_start = test_start - pd.DateOffset(months=4)
        valid_start = test_start - pd.DateOffset(months=7)
        train_start = test_start - pd.DateOffset(months=43)
        train_end = calendar[calendar < valid_start][-1]
        valid_end = calendar[calendar < calibrate_start][-1]
        calibrate_end = calendar[calendar < test_start][-1]
        test_end = min(test_start + pd.DateOffset(months=6) - pd.Timedelta(days=1), pd.Timestamp("2026-09-15"))
        windows[f"W{number}"] = {"train": [str(train_start.date()), str(train_end.date())],
                                 "valid": [str(valid_start.date()), str(valid_end.date())],
                                 "calibrate": [str(calibrate_start.date()), str(calibrate_end.date())],
                                 "test": [str(test_start.date()), str(test_end.date())]}
    return windows


def load_panel(start="2018-01-01", end="2026-09-15", columns=None):
    predicate = [("datetime", ">=", pd.Timestamp(start)), ("datetime", "<=", pd.Timestamp(end))]
    pieces = [pd.read_parquet(path, columns=columns, filters=predicate)
              for path in sorted(SOURCE.glob("new_*.parquet"))]
    panel = pd.concat(pieces).sort_index()
    if panel.index.has_duplicates:
        raise RuntimeError("冻结行情面板存在重复股票日")
    return panel


def make_calendar_labels(panel, calendar, horizon):
    dates = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    positions = calendar.get_indexer(dates)
    if (positions < 0).any():
        raise RuntimeError("行情日期无法映射统一交易日历")
    end_positions = positions + horizon + 1
    end_dates = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns]")
    valid_end = end_positions < len(calendar)
    end_dates.iloc[np.flatnonzero(valid_end)] = calendar.take(end_positions[valid_end]).to_numpy()
    opens = panel["open"].where(panel["open"] > 0)
    label = pd.Series(np.nan, index=panel.index, dtype=np.float32)
    for instrument, rows in opens.groupby(level="instrument", sort=False):
        stock = rows.droplevel("instrument").reindex(calendar)
        entry = stock.shift(-1)
        exit_ = stock.shift(-(horizon + 1))
        values = exit_.div(entry).sub(1).replace([np.inf, -np.inf], np.nan)
        keys = rows.index
        label.loc[keys] = values.reindex(keys.get_level_values("datetime")).to_numpy(dtype=np.float32)
    return label, end_dates


def rank_labels(label, calendar):
    dates = label.index.get_level_values("datetime")
    ranked = label.groupby(level="datetime").rank(pct=True).sub(0.5).mul(3.46)
    return ranked.astype(np.float32)


def common_universe_mask(panel):
    denominator = ROOT / "data/historical_repair/denominator/d8f7e2602f5b"
    years = range(int(panel.index.get_level_values("datetime").min().year),
                  int(panel.index.get_level_values("datetime").max().year) + 1)
    expected = pd.concat([pd.read_parquet(denominator / f"expected_keys_{year}.parquet",
                                          columns=["code", "trade_date", "full_day_suspended"])
                          for year in years if (denominator / f"expected_keys_{year}.parquet").exists()], ignore_index=True)
    expected = expected.loc[~expected["full_day_suspended"].astype(bool)]
    expected_index = pd.MultiIndex.from_arrays([pd.to_datetime(expected["trade_date"]), expected["code"].astype(str).str.zfill(6)])
    source_index = pd.MultiIndex.from_arrays([panel.index.get_level_values("datetime"), panel.index.get_level_values("instrument").str[2:]])
    return pd.Series(source_index.isin(expected_index) & panel["close"].gt(0).to_numpy() & panel["volume"].gt(0).to_numpy(), index=panel.index)


def segments(window, label, label_end, horizon):
    result = {}
    ordered = ("train", "valid", "calibrate", "test")
    for index, name in enumerate(ordered):
        start, end = map(pd.Timestamp, window[name])
        dates = label_end.index.get_level_values("datetime")
        mask = (dates >= start) & (dates <= end) & label.notna().to_numpy() & label_end.notna().to_numpy()
        if index < len(ordered) - 1:
            next_start = pd.Timestamp(window[ordered[index + 1]][0])
            mask &= label_end.to_numpy() < next_start.to_datetime64()
        result[name] = mask
    return result


def preflight():
    from core.data.connection import Database
    from core.adapters.qlib.models import DEFAULT_PARAMS, PARAM_PROFILES

    ref = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    source_identity = json.loads((SOURCE / "input_identity.json").read_text(encoding="utf-8"))
    coverage = json.loads((SOURCE / "context/coverage.json").read_text(encoding="utf-8"))
    calendar_path = ROOT / "data/qlib_data/calendars/day.txt"
    if ref.get("status") != "preflight_passed" or source_identity["qlib_pointer_sha256"] != ref["source_files_sha256"]["data\\qlib_data\\current.json"]:
        raise RuntimeError("上一轮冻结输入身份校验失败")
    if SOURCE.resolve() != (ROOT / "data/experiments/multiscale_group_research_20261103/current_isolated_input").resolve():
        raise RuntimeError("当前实验只允许读取已冻结隔离面板")
    if sha256(PRIOR / "preheat_2018.parquet") != json.loads((PRIOR / "preheat_audit.json").read_text(encoding="utf-8"))["preheat_sha256"]:
        raise RuntimeError("2018年月线预热快照哈希审计失败")
    panel_files = sorted(SOURCE.glob("new_*.parquet"))
    if not panel_files or not (SOURCE / "context/context_features.parquet").exists():
        raise RuntimeError("冻结行情或共同上下文文件不完整")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identities = Database.fetchall("SELECT ts_code,symbol,list_date,delist_date,exchange,market FROM stock_identity ORDER BY ts_code")
    identity_path = OUTPUT / "stock_identity_snapshot.json"
    identity_payload = json.dumps(identities, ensure_ascii=False, default=str, indent=2)
    if identity_path.exists() and identity_path.read_text(encoding="utf-8") != identity_payload:
        raise RuntimeError("股票身份/上市日期快照已变化；禁止覆盖")
    identity_path.write_text(identity_payload, encoding="utf-8")
    membership_source = PRIOR / "stock_index_membership_snapshot.json"
    membership_path = OUTPUT / membership_source.name
    if membership_path.exists() and sha256(membership_path) != sha256(membership_source):
        raise RuntimeError("历史行业快照已变化；禁止覆盖")
    if not membership_path.exists():
        membership_path.write_bytes(membership_source.read_bytes())
    history_summary = Database.fetchall("SELECT COUNT(*) AS row_count,MIN(as_of) AS first_date,MAX(as_of) AS last_date,COUNT(DISTINCT as_of) AS snapshot_days FROM security_reference_history")
    provenance = Database.fetchall("SELECT provenance_status,COUNT(*) AS row_count FROM security_reference_history GROUP BY provenance_status")
    manifest = {
        "status": "preflight_passed", "research_only": True, "production_model_unchanged": True,
        "strict_pit_verified": False, "execution_verified": False,
        "source_panel_end": coverage["last_date"], "context_industry_vintage": coverage["industry_vintage"],
        "industry_pit_verified": bool(coverage["industry_pit_verified"]),
        "history_denominator_verified": bool(coverage["history_denominator_verified"]),
        "factor_keys": ref["factor_keys"], "seeds": list(SEEDS),
        "folds": fold_windows(calendar_index()), "primary_folds": [f"W{i}" for i in range(1, 8)], "sensitivity_fold": "W8",
        "horizons": {"H10": 10, "H20": 20},
        "window_policy": {"train_months": 36, "early_stop_months": 3, "calibration_months": 4,
                           "purge": "each segment ends before next segment start by actual label_end"},
        "lightgbm_parameters": {**DEFAULT_PARAMS["lightgbm"], **PARAM_PROFILES["conservative"]["lightgbm"],
                                "num_threads": 4, "seed_policy": "each fixed experiment seed is assigned to seed/bagging/feature_fraction/data_random_seed"},
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyqlib", "lightgbm", "torch")},
        "classification_arms": list(CLASSIFICATION_ARMS), "sequence_arms": list(SEQUENCE_ARMS),
        "planned_predictions": 672, "comparison_family_size": 20,
        "data_qualification": {
            "historical_industry_publication_vintage": "unverified; R4 exploratory only",
            "board_effective_history": "security_reference_history is a sparse unverified snapshot; B3 is exploratory",
            "listing_date_history": "stock_identity immutable snapshot frozen for this experiment; source history not versioned",
            "trade_execution": "not qualified; historical ST/limit-state gaps remain",
        },
        "security_reference_history": {"summary": history_summary, "provenance": provenance},
        "stock_identity_snapshot_sha256": sha256(identity_path),
        "membership_snapshot_sha256": sha256(membership_path),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [*panel_files,
            SOURCE / "context/context_features.parquet", SOURCE / "context/coverage.json", SOURCE / "factors.json",
            SOURCE / "input_identity.json", calendar_path, PRIOR / "preheat_2018.parquet",
            PRIOR / "classification_snapshot.parquet",
            *sorted((ROOT / "data/historical_repair/denominator/d8f7e2602f5b").glob("expected_keys_*.parquet"))]},
        "code_sha256": {"scripts/horizon_multiscale_research.py": sha256(Path(__file__)),
                        "scripts/run_horizon_multiscale_research.py": sha256(ROOT / "scripts/run_horizon_multiscale_research.py"),
                        "scripts/horizon_multiscale_features.py": sha256(ROOT / "scripts/horizon_multiscale_features.py"),
                        "scripts/train_horizon_classification.py": sha256(ROOT / "scripts/train_horizon_classification.py"),
                        "scripts/train_horizon_sequence.py": sha256(ROOT / "scripts/train_horizon_sequence.py"),
                        "scripts/train_monthly_group_research.py": sha256(ROOT / "scripts/train_monthly_group_research.py"),
                        "scripts/monthly_group_features.py": sha256(ROOT / "scripts/monthly_group_features.py"),
                        "scripts/analyze_horizon_classification.py": sha256(ROOT / "scripts/analyze_horizon_classification.py"),
                        "scripts/analyze_horizon_sequence.py": sha256(ROOT / "scripts/analyze_horizon_sequence.py"),
                        "core/adapters/qlib/models.py": sha256(ROOT / "core/adapters/qlib/models.py"),
                        "core/adapters/qlib/rank_models.py": sha256(ROOT / "core/adapters/qlib/rank_models.py"),
                        "core/adapters/qlib/custom_processors.py": sha256(ROOT / "core/adapters/qlib/custom_processors.py"),
                        "scripts/horizon_sequence_features.py": sha256(ROOT / "scripts/horizon_sequence_features.py"),
                        "scripts/horizon_sequence_models.py": sha256(ROOT / "scripts/horizon_sequence_models.py"),
                        "tests/test_horizon_multiscale_features.py": sha256(ROOT / "tests/test_horizon_multiscale_features.py")},
    }
    target = OUTPUT / "manifest.json"
    if target.exists() and json.dumps(json.loads(target.read_text(encoding="utf-8")), sort_keys=True) != json.dumps(manifest, sort_keys=True, default=str):
        raise RuntimeError("实验冻结清单已变化；拒绝覆盖或续跑")
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "preflight_passed", "output": str(target), "panel_end": coverage["last_date"],
                      "industry_pit_verified": manifest["industry_pit_verified"], "planned_predictions": 672}, ensure_ascii=False))


def freeze_labels():
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "preflight_passed":
        raise RuntimeError("必须先完成只读预检")
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"冻结输入变化: {name}")
    for name, expected in manifest["code_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"冻结代码变化: {name}")
    if sha256(OUTPUT / "stock_identity_snapshot.json") != manifest["stock_identity_snapshot_sha256"]:
        raise RuntimeError("股票身份快照变化")
    if sha256(OUTPUT / "stock_index_membership_snapshot.json") != manifest["membership_snapshot_sha256"]:
        raise RuntimeError("历史行业快照变化")
    calendar = calendar_index()
    panel = load_panel("2018-01-01", manifest["source_panel_end"], ["open", "label", "close", "volume"])
    common_mask = common_universe_mask(panel)
    results = {}
    audits = {}
    for name, horizon in (("H10", 10), ("H20", 20)):
        path = OUTPUT / f"labels_{name}.parquet"
        label, label_end = make_calendar_labels(panel, calendar, horizon)
        label = label.where(common_mask)
        if name == "H10":
            old = panel["label"].replace([np.inf, -np.inf], np.nan)
            paired = pd.concat({"calendar": label, "previous": old}, axis=1).dropna()
            exact = np.isclose(paired["calendar"], paired["previous"], rtol=1e-6, atol=1e-7)
            audits["H10_previous_label_comparison"] = {
                "common_rows": int(len(paired)), "exact_close_rows": int(exact.sum()),
                "different_rows": int((~exact).sum()), "max_abs_difference": float((paired["calendar"] - paired["previous"]).abs().max()),
                "interpretation": "calendar endpoint labels preserve missing open dates; previous per-instrument labels can differ across quote gaps",
            }
            if int((~exact).sum()):
                raise RuntimeError("新冻结H10标签与已审计标签不一致；需解释并版本化后再继续")
        frame = pd.DataFrame({"label": label, "label_end": label_end})
        frame["label_rank"] = rank_labels(label, calendar)
        frame.to_parquet(path, compression="zstd")
        results[name] = {"path": str(path), "sha256": sha256(path), "finite_labels": int(label.notna().sum()),
                         "missing": int(label.isna().sum()), "max_label_end": str(label_end.max().date()) if label_end.notna().any() else None}
        fold_audit = {}
        for fold, window in manifest["folds"].items():
            end_series = label_end
            masks = segments(window, label, end_series, horizon)
            fold_audit[fold] = {}
            for segment, mask in masks.items():
                segment_rows = label.index[mask]
                fold_audit[fold][segment] = {"rows": int(mask.sum()), "first_date": str(segment_rows.get_level_values("datetime").min().date()) if len(segment_rows) else None,
                    "last_date": str(segment_rows.get_level_values("datetime").max().date()) if len(segment_rows) else None,
                    "max_label_end": str(label_end.loc[segment_rows].max().date()) if len(segment_rows) else None}
            for current, following in zip(("train", "valid", "calibrate"), ("valid", "calibrate", "test")):
                maximum = fold_audit[fold][current]["max_label_end"]
                if maximum and pd.Timestamp(maximum) >= pd.Timestamp(window[following][0]):
                    raise RuntimeError(f"{name}/{fold}/{current} 标签未在{following}前成熟")
            if fold_audit[fold]["train"]["max_label_end"] and fold_audit[fold]["valid"]["first_date"]:
                if pd.Timestamp(fold_audit[fold]["train"]["max_label_end"]) >= pd.Timestamp(window["valid"][0]):
                    raise RuntimeError(f"{name}/{fold} 训练标签未在验证前成熟")
        audits[name + "_folds"] = fold_audit
    audit = {"status": "labels_frozen", "label_definition": "Open(t+h+1)/Open(t+1)-1 on shared calendar; missing endpoints remain missing",
             "common_universe_rows": int(common_mask.sum()), "all_panel_rows": int(len(common_mask)),
             "common_universe_definition": "frozen historical expected stock-day denominator, excluding full-day suspensions, with positive close and volume",
             "labels": results, "audits": audits, "calendar_sha256": sha256(ROOT / "data/qlib_data/calendars/day.txt")}
    (OUTPUT / "label_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "labels": results, "H10_differences": audits["H10_previous_label_comparison"]["different_rows"]}, ensure_ascii=False))


def freeze_lightgbm_budget():
    ratios = {}
    for input_name in ("D", "M"):
        for group_name in ("B3", "A2", "R4"):
            samples = []
            for fold in ("W1", "W8"):
                for horizon in ("H10", "H20"):
                    path = OUTPUT / f"{fold}_{horizon}_42_training.json"
                    data = json.loads(path.read_text(encoding="utf-8"))
                    baseline = data[f"U_{input_name}"]["elapsed_seconds"]
                    expert = data[f"{group_name}_{input_name}"].get("experts", {})
                    expert_seconds = sum(value.get("elapsed_seconds", 0.0) for value in expert.values()
                                         if value.get("status") == "trained")
                    samples.append((baseline + expert_seconds) / max(baseline, 1e-9))
            ratios[f"{group_name}_{input_name}"] = {"measured_ratios": samples,
                "members": max(1, int(np.ceil(float(np.mean(samples)))))}
    payload = {"status": "frozen", "basis_folds": ["W1", "W8"], "basis_horizons": ["H10", "H20"],
               "seed": 42, "hardware": "same local host; LightGBM num_threads=4", "budgets": ratios,
               "manifest_sha256": sha256(OUTPUT / "manifest.json")}
    temporary = OUTPUT / "budget_lightgbm.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "budget_lightgbm.json")
    return payload


def run_command(script, args, folds, seeds, resume=False):
    command = [sys.executable, "-B", str(ROOT / "scripts" / script),
               "--horizons", *args.horizons, "--folds", *folds, "--seeds", *map(str, seeds)]
    if resume:
        command.append("--resume")
    import subprocess
    log_path = OUTPUT / f"{Path(script).stem}_{folds[0]}_{folds[-1]}_{seeds[0]}_{seeds[-1]}.log"
    attempt = 0
    while True:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"command": command, "attempt": attempt + 1}) + "\n")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            status = process.wait()
        if status == 0:
            return 0
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        transient = any(token in tail.lower() for token in ("winerror 32", "sharing violation", "temporarily unavailable", "permissionerror"))
        if not transient or attempt >= 2:
            return status
        attempt += 1
        time.sleep(10 * (2 ** attempt))


def run_all(args):
    preflight()
    freeze_labels()
    folds, seeds = ["W1", "W8"], [42]
    for script in ("train_horizon_classification.py", "train_horizon_sequence.py"):
        if run_command(script, args, folds, seeds, resume=True):
            raise SystemExit(1)
    budget = freeze_lightgbm_budget()
    print(json.dumps({"budget_status": budget["status"], "budget_file": str(OUTPUT / "budget_lightgbm.json")}, ensure_ascii=False))
    folds, seeds = [f"W{i}" for i in range(1, 9)], list(SEEDS)
    for script in ("train_horizon_classification.py", "train_horizon_sequence.py"):
        if run_command(script, args, folds, seeds, resume=True):
            raise SystemExit(1)
    for script in ("analyze_horizon_classification.py", "analyze_horizon_sequence.py"):
        status = __import__("subprocess").call([sys.executable, "-B", str(ROOT / "scripts" / script)])
        if status:
            raise SystemExit(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "freeze", "freeze-labels", "prerun", "train", "evaluate", "confirm", "combine", "shadow", "status", "run-all"))
    parser.add_argument("--horizons", nargs="+", choices=("H10", "H20"), default=("H10", "H20"))
    parser.add_argument("--track", choices=("classification", "sequence", "both"), default="classification")
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.output_dir:
        global OUTPUT
        OUTPUT = Path(args.output_dir).resolve()
        os.environ["HORIZON_RESEARCH_OUTPUT"] = str(OUTPUT)
    if args.folds is None:
        args.folds = ["W1", "W8"] if args.stage == "prerun" else [f"W{i}" for i in range(1, 9)]
    if set(args.folds) - {f"W{i}" for i in range(1, 9)} or set(args.seeds) - set(SEEDS):
        raise ValueError("折或种子超出本实验预注册范围")
    if args.stage == "status":
        predictions = sorted(OUTPUT.glob("W*_H*_*_*.parquet"))
        arms = ["_".join(path.stem.split("_")[3:]) for path in predictions]
        print(json.dumps({"output": str(OUTPUT), "prediction_files": len(predictions),
                          "expected_formal_predictions": 672,
                          "classification_predictions": sum(arm in CLASSIFICATION_ARMS for arm in arms),
                          "sequence_predictions": sum(arm in SEQUENCE_ARMS for arm in arms)}, ensure_ascii=False))
    elif args.stage == "run-all":
        run_all(args)
    elif args.stage == "preflight":
        preflight()
    elif args.stage in ("freeze", "freeze-labels"):
        freeze_labels()
    elif args.stage in ("prerun", "train"):
        scripts = []
        if args.track in ("classification", "both"):
            scripts.append("train_horizon_classification.py")
        if args.track in ("sequence", "both"):
            scripts.append("train_horizon_sequence.py")
        for script in scripts:
            command = [sys.executable, "-B", str(ROOT / "scripts" / script),
                       "--horizons", *args.horizons, "--folds", *args.folds, "--seeds", *map(str, args.seeds)]
            if args.resume:
                command.append("--resume")
            status = __import__("subprocess").call(command)
            if status:
                raise SystemExit(status)
    elif args.stage == "evaluate":
        scripts = []
        if args.track in ("classification", "both"):
            scripts.append("analyze_horizon_classification.py")
        if args.track in ("sequence", "both"):
            scripts.append("analyze_horizon_sequence.py")
        for script in scripts:
            status = __import__("subprocess").call([sys.executable, "-B", str(ROOT / "scripts" / script)])
            if status:
                raise SystemExit(status)
    else:
        raise RuntimeError(f"阶段 {args.stage} 尚未解锁：先完成冻结标签、共同样本因果审计与工程预跑；不会静默调用上一轮标签结果")


if __name__ == "__main__":
    main()
