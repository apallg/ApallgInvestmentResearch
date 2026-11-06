import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HORIZON_RESEARCH_OUTPUT", "data/experiments/horizon_multiscale_research_v2_4m")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.horizon_multiscale_research import OUTPUT as LEGACY_OUTPUT, SEEDS, SEQUENCE_ARMS, calendar_index
from scripts.run_multiscale_group_research import sha256
from scripts.train_horizon_classification import load_fold, calibrate
from scripts.train_horizon_sequence import (
    DEVICE,
    HorizonFusionNet,
    build_daily_positions,
    transform_batch,
)


LEGACY_OUTPUT = LEGACY_OUTPUT.resolve()
OUTPUT = ROOT / "data/experiments/model_complementarity_research_v1_20261105"
FOLDS = tuple(f"W{i}" for i in range(1, 9))
HORIZONS = ("H10", "H20")
ARMS = ("G", "N", "F50", "FW")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def file_inventory():
    roots = [
        ROOT / "data/experiments/horizon_multiscale_research_v2_4m",
        ROOT / "data/experiments/monthly_group_research_v4_20261103",
        ROOT / "data/experiments/horizon_multiscale_research_20261104",
        ROOT / "data/experiments/multiscale_group_research_current_v2_20261103",
    ]
    records = []
    for folder in roots:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                "suffix": path.suffix.lower(), "mtime_ns": path.stat().st_mtime_ns})
    records.sort(key=lambda row: row["path"].casefold())
    totals = {}
    for row in records:
        name = Path(row["path"]).parts[2]
        entry = totals.setdefault(name, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += row["bytes"]
    result = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "roots": totals, "files": records}
    write_json(OUTPUT / "inventory.json", result)
    print(json.dumps({"output": str(OUTPUT / "inventory.json"), "roots": totals}, ensure_ascii=False))


def expected_test_index(fold, horizon):
    manifest = json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    low, high = map(pd.Timestamp, manifest["folds"][fold]["test"])
    labels = pd.read_parquet(LEGACY_OUTPUT / f"labels_{horizon}.parquet",
                             filters=[("datetime", ">=", low), ("datetime", "<=", high)])
    labels = labels.loc[labels["label"].notna(), ["label", "label_rank"]]
    audit_path = LEGACY_OUTPUT / f"{fold}_{horizon}_input_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    expected = labels.index
    if expected.has_duplicates:
        raise RuntimeError(f"冻结标签重复: {fold}/{horizon}")
    if audit and int(audit.get("segments", {}).get("test", -1)) != len(expected):
        raise RuntimeError(f"冻结标签行数与训练窗口审计不符: {fold}/{horizon}")
    return expected, labels, audit


def read_prediction(path):
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != ["datetime", "instrument"]:
        raise RuntimeError(f"预测索引不符合契约: {path.name}")
    if frame.index.has_duplicates or not {"score", "label", "label_rank"}.issubset(frame.columns):
        raise RuntimeError(f"预测重复或列缺失: {path.name}")
    if frame[["score", "label", "label_rank"]].isna().any().any():
        raise RuntimeError(f"预测或标签存在缺失: {path.name}")
    if not np.isfinite(frame[["score", "label", "label_rank"]].to_numpy(dtype=float)).all():
        raise RuntimeError(f"预测或标签存在非有限值: {path.name}")
    return frame


def strict_audit():
    manifest = json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    completed, failures = 0, []
    for horizon in HORIZONS:
        for fold in FOLDS:
            expected, expected_labels, audit = expected_test_index(fold, horizon)
            fold_results = {"fold": fold, "horizon": horizon, "expected_rows": len(expected),
                            "segments": audit["segments"], "predictions": 0}
            for seed in SEEDS:
                for arm in manifest["classification_arms"]:
                    path = LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    if not path.exists():
                        failures.append({"path": str(path), "reason": "missing"})
                        continue
                    try:
                        frame = read_prediction(path)
                        if not frame.index.equals(expected):
                            failures.append({"path": str(path), "reason": "expected_index_mismatch",
                                             "rows": len(frame), "expected_rows": len(expected)})
                            continue
                        if not np.array_equal(frame["label"].to_numpy(), expected_labels["label"].to_numpy()):
                            failures.append({"path": str(path), "reason": "label_mismatch"})
                            continue
                        if not np.array_equal(frame["label_rank"].to_numpy(), expected_labels["label_rank"].to_numpy()):
                            failures.append({"path": str(path), "reason": "label_rank_mismatch"})
                            continue
                        fold_results["predictions"] += 1
                    except Exception as exc:
                        failures.append({"path": str(path), "reason": type(exc).__name__, "detail": str(exc)})
            for seed in SEEDS:
                for arm in SEQUENCE_ARMS:
                    path = LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    if not path.exists():
                        failures.append({"path": str(path), "reason": "missing"})
                        continue
                    try:
                        frame = read_prediction(path)
                        if not frame.index.equals(expected):
                            failures.append({"path": str(path), "reason": "expected_index_mismatch",
                                             "rows": len(frame), "expected_rows": len(expected)})
                            continue
                        if not np.array_equal(frame["label"].to_numpy(), expected_labels["label"].to_numpy()):
                            failures.append({"path": str(path), "reason": "label_mismatch"})
                            continue
                        if not np.array_equal(frame["label_rank"].to_numpy(), expected_labels["label_rank"].to_numpy()):
                            failures.append({"path": str(path), "reason": "label_rank_mismatch"})
                            continue
                        fold_results["predictions"] += 1
                    except Exception as exc:
                        failures.append({"path": str(path), "reason": type(exc).__name__, "detail": str(exc)})
            completed += fold_results["predictions"]
            del expected, expected_labels
            gc.collect()
    result = {"status": "passed" if not failures and completed == 672 else "failed",
              "verified_predictions": completed, "expected_predictions": 672,
              "manifest_sha256": sha256(LEGACY_OUTPUT / "manifest.json"),
              "failures": failures, "research_only": True}
    write_json(OUTPUT / "legacy_strict_audit.json", result)
    print(json.dumps({"status": result["status"], "verified_predictions": completed,
                      "failures": len(failures), "output": str(OUTPUT / "legacy_strict_audit.json")}, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"旧预测严格审计失败: {len(failures)} 项")


def frame_digest(frame):
    values = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def compact_core(folds=FOLDS, horizons=HORIZONS, seeds=SEEDS):
    audit_path = OUTPUT / "legacy_strict_audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("必须先通过旧预测严格审计")
    legacy_manifest = json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    arms = tuple(legacy_manifest["classification_arms"]) + tuple(legacy_manifest["sequence_arms"])
    output_dir = OUTPUT / "legacy_core"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for horizon in horizons:
        for fold in folds:
            for seed in seeds:
                target = output_dir / f"{fold}_{horizon}_{seed}.parquet"
                existing = {"path": str(target.relative_to(ROOT)), "status": "verified"}
                if target.exists():
                    wide = pd.read_parquet(target)
                    if wide.index.has_duplicates or not set(["label", "label_rank", *arms]).issubset(wide.columns):
                        raise RuntimeError(f"已有核心预测表格式不符: {target.name}")
                    existing["rows"] = len(wide)
                    existing["sha256"] = sha256(target)
                    entries.append(existing)
                    continue
                base = None
                source_hashes = {}
                for arm in arms:
                    source = LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    frame = read_prediction(source)
                    source_hashes[arm] = {"sha256": sha256(source), "semantic_sha256": frame_digest(frame[["score", "label", "label_rank"]])}
                    if base is None:
                        base = pd.DataFrame({"label": frame["label"], "label_rank": frame["label_rank"]}, index=frame.index)
                    elif not base.index.equals(frame.index) or not np.array_equal(base["label"].to_numpy(), frame["label"].to_numpy()) or not np.array_equal(base["label_rank"].to_numpy(), frame["label_rank"].to_numpy()):
                        raise RuntimeError(f"合并核心预测时样本或标签不一致: {fold}/{horizon}/{seed}/{arm}")
                    base[arm] = frame["score"].to_numpy(dtype=np.float32)
                    del frame
                temporary = target.with_suffix(".tmp")
                base.to_parquet(temporary, compression="zstd")
                temporary.replace(target)
                wide = pd.read_parquet(target)
                if not wide.index.equals(base.index) or wide.isna().any().any() or not np.isfinite(wide.to_numpy(dtype=float)).all():
                    raise RuntimeError(f"核心预测表读回审计失败: {target.name}")
                for arm in arms:
                    source_frame = read_prediction(LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet")
                    if not source_frame.index.equals(wide.index) or not np.array_equal(source_frame["score"].to_numpy(), wide[arm].to_numpy()) or not np.array_equal(source_frame["label"].to_numpy(), wide["label"].to_numpy()) or not np.array_equal(source_frame["label_rank"].to_numpy(), wide["label_rank"].to_numpy()):
                        raise RuntimeError(f"合并前后数值不一致: {target.name}/{arm}")
                    del source_frame
                existing.update({"rows": len(wide), "columns": list(wide.columns), "sha256": sha256(target),
                                 "source_artifacts": source_hashes})
                entries.append(existing)
                print(json.dumps({"fold": fold, "horizon": horizon, "seed": seed, "rows": len(wide),
                                  "bytes": target.stat().st_size, "output": str(target)}, ensure_ascii=False), flush=True)
                del base, wide
                gc.collect()
    result = {"status": "complete" if len(entries) == 48 else "partial", "count": len(entries), "expected": 48,
              "legacy_manifest_sha256": sha256(LEGACY_OUTPUT / "manifest.json"),
              "strict_audit_sha256": sha256(audit_path), "columns": list(arms),
              "entries": entries, "lossless_semantic_roundtrip": True, "source_files_deleted": False}
    write_json(OUTPUT / "legacy_core_manifest.json", result)
    if len(entries) != len(folds) * len(horizons) * len(seeds):
        raise RuntimeError("旧预测核心表未完成")


def prune_legacy_predictions(apply=False):
    audit_path = OUTPUT / "legacy_strict_audit.json"
    core_path = OUTPUT / "legacy_core_manifest.json"
    if json.loads(audit_path.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("旧预测严格审计未通过")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    if core.get("status") != "complete" or core.get("count") != 48 or not core.get("lossless_semantic_roundtrip"):
        raise RuntimeError("合并后的核心预测表未完整通过读回核验")
    manifest = json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    arms = tuple(manifest["classification_arms"]) + tuple(manifest["sequence_arms"])
    files = []
    for horizon in HORIZONS:
        for fold in FOLDS:
            for seed in SEEDS:
                core_file = OUTPUT / "legacy_core" / f"{fold}_{horizon}_{seed}.parquet"
                core_frame = pd.read_parquet(core_file)
                if not set(["label", "label_rank", *arms]).issubset(core_frame.columns):
                    raise RuntimeError(f"核心表缺少必要列: {core_file.name}")
                for arm in arms:
                    source = LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    if not source.exists():
                        raise RuntimeError(f"待清理源文件不存在: {source.name}")
                    expected = next((row for entry in core["entries"] if entry["path"].endswith(f"{fold}_{horizon}_{seed}.parquet")
                                      for key, row in entry.get("source_artifacts", {}).items() if key == arm), None)
                    if expected and sha256(source) != expected["sha256"]:
                        raise RuntimeError(f"源文件哈希与已验证核心表清单不符: {source.name}")
                    if not np.array_equal(pd.read_parquet(source, columns=["score"])["score"].to_numpy(), core_frame[arm].to_numpy()):
                        raise RuntimeError(f"核心表分数读回不一致，拒绝清理: {source.name}")
                    files.append({"path": str(source.resolve()), "bytes": source.stat().st_size, "sha256": sha256(source)})
                del core_frame
                gc.collect()
    prune_path = OUTPUT / "prune_manifest.json"
    if not apply:
        result = {"status": "ready_to_apply", "root": str(LEGACY_OUTPUT), "files": files,
                  "count": len(files), "bytes": sum(row["bytes"] for row in files),
                  "preserved": ["labels_H10.parquet", "labels_H20.parquet", "all model weights and metadata", "source panels", "frozen snapshots"],
                  "reason": "verified lossless compact core predictions replace duplicate per-arm Parquet files"}
        write_json(prune_path, result)
        print(json.dumps({"status": result["status"], "count": result["count"],
                          "freeable_gib": round(result["bytes"] / 1024 ** 3, 3),
                          "manifest": str(prune_path)}, ensure_ascii=False))
        return
    if not prune_path.exists():
        raise RuntimeError("缺少待应用清理清单")
    plan = json.loads(prune_path.read_text(encoding="utf-8"))
    if plan.get("status") != "ready_to_apply" or plan.get("count") != 672:
        raise RuntimeError("清理清单状态或文件数不符合预期")
    root = LEGACY_OUTPUT.resolve()
    for row in plan["files"]:
        target = Path(row["path"]).resolve()
        if target.parent != root or target.suffix != ".parquet" or target.name.startswith("labels_"):
            raise RuntimeError(f"清理目标越界: {target}")
        if not target.exists() or target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"]:
            raise RuntimeError(f"清理前哈希或大小变化: {target.name}")
    deleted = []
    for row in plan["files"]:
        target = Path(row["path"]).resolve()
        target.unlink()
        deleted.append({"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]})
    plan["status"] = "applied"
    plan["deleted"] = deleted
    plan["deleted_bytes"] = sum(row["bytes"] for row in deleted)
    write_json(prune_path, plan)
    print(json.dumps({"status": plan["status"], "deleted_files": len(deleted),
                      "deleted_gib": round(plan["deleted_bytes"] / 1024 ** 3, 3)}, ensure_ascii=False))


def component_signature(fold, horizon, seed, arm):
    return hashlib.sha256(json.dumps({"legacy_manifest": sha256(LEGACY_OUTPUT / "manifest.json"),
        "legacy_label_audit": sha256(LEGACY_OUTPUT / "label_audit.json"),
        "fold": fold, "horizon": horizon, "seed": seed, "arm": arm,
        "model": "saved N0 and sequence LGB; inference only"}, sort_keys=True).encode()).hexdigest()


def infer_n0(model_file, values, index, segments, calendar, batch_size=4096):
    stored = torch.load(model_file, map_location="cpu", weights_only=False)
    model = HorizonFusionNet(values.shape[1] + 1, "N0").to(DEVICE)
    model.load_state_dict(stored["model"])
    model.eval()
    mean, scale = stored["mean"], stored["scale"]
    predictions = {}
    for name, rows in segments.items():
        endpoints = index.take(rows)
        positions = build_daily_positions(index, calendar, endpoints)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                pos = positions[start:start + batch_size]
                daily = transform_batch(values, pos, mean, scale, values.shape[1])
                pred = model(torch.from_numpy(daily).to(DEVICE), None)
                outputs.append(pred.detach().cpu().numpy())
        predictions[name] = pd.Series(np.concatenate(outputs).astype(np.float32), index=endpoints)
        del positions
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return predictions


def recover_components(folds=FOLDS, horizons=HORIZONS, seeds=SEEDS, batch_size=4096):
    audit_path = OUTPUT / "legacy_strict_audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("必须先通过旧预测严格审计")
    manifest = json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if not (OUTPUT / "manifest.json").exists():
        write_json(OUTPUT / "manifest.json", {"status": "component_recovery_started", "legacy_manifest": str((LEGACY_OUTPUT / "manifest.json").relative_to(ROOT)),
            "legacy_manifest_sha256": sha256(LEGACY_OUTPUT / "manifest.json"), "legacy_label_sha256": sha256(LEGACY_OUTPUT / "label_audit.json"),
            "factor_keys": manifest["factor_keys"], "seeds": list(SEEDS), "folds": list(FOLDS), "horizons": list(HORIZONS),
            "research_only": True, "production_model_unchanged": True})
    calendar = calendar_index()
    for horizon in horizons:
        for fold in folds:
            inputs, groups, calibration_group, labels, masks, load_audit = load_fold(fold, int(horizon[1:]))
            index = inputs["D"].index
            base = inputs["D"][manifest["factor_keys"]].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
            cal_rows, test_rows = np.flatnonzero(masks["calibrate"]), np.flatnonzero(masks["test"])
            for seed in seeds:
                pending = []
                for arm in ARMS:
                    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    meta_path = path.with_suffix(".json")
                    signature = component_signature(fold, horizon, seed, arm)
                    if path.exists() and meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        if meta.get("signature") != signature:
                            raise RuntimeError(f"续跑签名不匹配: {path.name}")
                        prior = read_prediction(path)
                        if not prior.index.equals(index[test_rows]):
                            raise RuntimeError(f"续跑预测样本变化: {path.name}")
                        continue
                    pending.append(arm)
                if not pending:
                    continue
                lgb_file = LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_sequence_lgb.pkl"
                lgb_data = pickle.loads(lgb_file.read_bytes())
                expected_lgb_signature = hashlib.sha256(json.dumps({"manifest": sha256(LEGACY_OUTPUT / "manifest.json"),
                    "labels": sha256(LEGACY_OUTPUT / "label_audit.json"), "horizon": horizon,
                    "fold": fold, "seed": seed, "arm": "LGB40", "parameters": manifest["lightgbm_parameters"],
                    "epochs": 80, "batch": 4096}, sort_keys=True).encode()).hexdigest()
                if lgb_data.get("signature") != expected_lgb_signature:
                    raise RuntimeError(f"LGB缓存签名不匹配: {lgb_file.name}")
                lgb_cal, lgb_test = lgb_data["scores"]["calibrate"], lgb_data["scores"]["test"]
                if not lgb_cal.index.equals(index[cal_rows]) or not lgb_test.index.equals(index[test_rows]):
                    raise RuntimeError(f"LGB原始分数样本与固定窗口不一致: {lgb_file.name}")
                need_nn = any(arm in ("N", "F50", "FW") for arm in pending)
                nn_scores = None
                sequence_meta = None
                if need_nn:
                    final_file = LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_N0.final.pt"
                    sequence_meta = json.loads((LEGACY_OUTPUT / f"{fold}_{horizon}_{seed}_N0.json").read_text(encoding="utf-8"))
                    expected_sequence_signature = hashlib.sha256(json.dumps({"manifest": sha256(LEGACY_OUTPUT / "manifest.json"),
                        "labels": sha256(LEGACY_OUTPUT / "label_audit.json"), "horizon": horizon,
                        "fold": fold, "seed": seed, "arm": "N0", "parameters": manifest["lightgbm_parameters"],
                        "epochs": 80, "batch": 4096}, sort_keys=True).encode()).hexdigest()
                    if sequence_meta.get("signature") != expected_sequence_signature:
                        raise RuntimeError(f"N0权重签名不匹配: {final_file.name}")
                    nn_scores = infer_n0(final_file, base, index, {"calibrate": cal_rows, "test": test_rows}, calendar, batch_size)
                nn_cal = nn_scores.get("calibrate") if nn_scores else None
                nn_test = nn_scores.get("test") if nn_scores else None
                nn_cal_rank = nn_cal.groupby(level="datetime").rank(pct=True) if nn_cal is not None else None
                nn_test_rank = nn_test.groupby(level="datetime").rank(pct=True) if nn_test is not None else None
                lgb_cal_rank = lgb_cal.groupby(level="datetime").rank(pct=True)
                lgb_test_rank = lgb_test.groupby(level="datetime").rank(pct=True)
                for arm in pending:
                    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
                    meta_path = path.with_suffix(".json")
                    signature = component_signature(fold, horizon, seed, arm)
                    weight = 0.0 if arm == "G" else 1.0 if arm == "N" else 0.5
                    if arm == "G":
                        cal_score, test_score = lgb_cal, lgb_test
                    elif arm == "N":
                        cal_score, test_score = nn_cal, nn_test
                    else:
                        if arm == "FW":
                            from scipy.optimize import minimize_scalar
                            target = labels["label_rank"].reindex(nn_cal_rank.index)
                            cal_frame = pd.concat({"n": nn_cal_rank, "g": lgb_cal_rank, "y": target}, axis=1)
                            if cal_frame.isna().any().any() or not np.isfinite(cal_frame.to_numpy()).all():
                                raise RuntimeError(f"校准组合数据缺失: {fold}/{horizon}/{seed}")
                            daily_groups = [group for _, group in cal_frame.groupby(level="datetime", sort=True)]
                            def objective(value):
                                errors = [(value * group["n"].to_numpy() + (1 - value) * group["g"].to_numpy() - group["y"].to_numpy()) ** 2
                                          for group in daily_groups]
                                return float(np.mean([np.mean(error) for error in errors]) + 0.01 * (value - 0.5) ** 2)
                            fit = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded",
                                                  options={"xatol": 1e-8, "maxiter": 500})
                            if not fit.success or not np.isfinite(fit.fun):
                                raise RuntimeError(f"学习融合权重失败: {fold}/{horizon}/{seed}")
                            weight = float(fit.x)
                        cal_score = weight * nn_cal_rank + (1 - weight) * lgb_cal_rank
                        test_score = weight * nn_test_rank + (1 - weight) * lgb_test_rank
                    calibrated, maps = calibrate(cal_score, labels["label"], calibration_group,
                                                 test_score, calibration_group)
                    frame = pd.DataFrame({"score": calibrated,
                        "label": labels["label"].reindex(calibrated.index),
                        "label_rank": labels["label_rank"].reindex(calibrated.index)})
                    if not frame.index.equals(index[test_rows]) or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
                        raise RuntimeError(f"恢复预测不完整或非有限: {path.name}")
                    reconstruction_error = None
                    if arm == "F50":
                        old = pd.read_parquet(OUTPUT / "legacy_core" / f"{fold}_{horizon}_{seed}.parquet", columns=["N0"])
                        reconstruction_error = float(np.max(np.abs(frame["score"].to_numpy() - old["N0"].to_numpy())))
                        if reconstruction_error > 1e-6:
                            raise RuntimeError(f"F50重建误差超过1e-6: {fold}/{horizon}/{seed}={reconstruction_error}")
                    temp = path.with_suffix(".tmp")
                    frame.to_parquet(temp, compression="zstd")
                    temp.replace(path)
                    lgb_restore = sequence_meta.get("lgb_restore_max_abs_error") if sequence_meta else None
                    write_json(meta_path, {"signature": signature, "arm": arm, "weight": weight,
                        "score_rows": len(frame), "calibration_rows": len(cal_score),
                        "restore_max_abs_error": sequence_meta.get("restore_max_abs_error") if arm != "G" else lgb_restore,
                        "reconstruction_max_abs_error": reconstruction_error,
                        "legacy_manifest_sha256": sha256(LEGACY_OUTPUT / "manifest.json"), "calibration_group": "historical_industry_x_GS8"})
                    print(json.dumps({"fold": fold, "horizon": horizon, "seed": seed, "arm": arm,
                                      "weight": weight, "rows": len(frame), "output": str(path)}, ensure_ascii=False), flush=True)
                    del frame
                    gc.collect()
                del lgb_data, lgb_cal, lgb_test, nn_scores
                gc.collect()
            del inputs, groups, calibration_group, labels, masks, base
            gc.collect()
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    completed = sum(1 for horizon in HORIZONS for fold in FOLDS for seed in SEEDS for arm in ARMS
                    if (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet").exists()
                    and (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.json").exists())
    manifest["status"] = "component_predictions_recovered" if completed == 192 else "component_predictions_partial"
    manifest["arms"] = list(ARMS)
    manifest["prediction_count"] = completed
    manifest["planned_predictions"] = 192
    write_json(manifest_path, manifest)


def prediction_frame(fold, horizon, seed, arm):
    path = OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet"
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise RuntimeError(f"缺少方案预测或审计: {path.name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("signature") != component_signature(fold, horizon, seed, arm):
        raise RuntimeError(f"预测签名不匹配: {path.name}")
    frame = read_prediction(path)
    if len(frame) != int(meta.get("score_rows", -1)):
        raise RuntimeError(f"预测行数与审计元数据不一致: {path.name}")
    if meta.get("calibration_group") != "historical_industry_x_GS8":
        raise RuntimeError(f"校准分层不一致: {path.name}")
    return frame


def daily_metrics(frame, formed_dates):
    if frame.index.has_duplicates or frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise RuntimeError("评价输入不完整或非有限")
    top5, top50, pool, ics = {}, {}, {}, {}
    for date, part in frame.groupby(level="datetime", sort=True):
        if len(part) < 50:
            raise RuntimeError(f"{date.date()} 共同股票池不足 Top50")
        top5[date] = float(part.nlargest(5, "score")["label"].mean())
        top50[date] = float(part.nlargest(50, "score")["label"].mean())
        pool[date] = float(part["label"].mean())
        ics[date] = float(part["score"].corr(part["label_rank"], method="spearman"))
    dates = pd.DatetimeIndex(formed_dates)
    values5, values50 = pd.Series(top5, dtype=float).reindex(dates), pd.Series(top50, dtype=float).reindex(dates)
    values_pool = pd.Series(pool, dtype=float).reindex(dates)
    if values5.isna().any() or values50.isna().any() or values_pool.isna().any():
        raise RuntimeError("预定篮子日期缺失")
    return {"top5": values5.to_numpy(), "top50": values50.to_numpy(), "pool": values_pool.to_numpy(),
            "top5_excess": (values5 - values_pool).to_numpy(),
            "top50_excess": (values50 - values_pool).to_numpy(),
            "rank_ic": float(np.mean(list(ics.values()))), "daily_dates": len(ics)}


def complementarity_metrics(g, n):
    if not g.index.equals(n.index) or not np.array_equal(g["label_rank"].to_numpy(), n["label_rank"].to_numpy()):
        raise RuntimeError("G/N互补性诊断样本或标签不一致")
    g_rank = g["score"].groupby(level="datetime").rank(pct=True)
    n_rank = n["score"].groupby(level="datetime").rank(pct=True)
    label_rank = g["label_rank"]
    pred_corr, error_corr, overlaps = [], [], []
    for date, gi in g_rank.groupby(level="datetime", sort=True):
        ni = n_rank.xs(date, level="datetime")
        yi = label_rank.xs(date, level="datetime")
        gi = gi.droplevel("datetime")
        pred_corr.append(float(gi.corr(ni)))
        error_corr.append(float((gi - yi).corr(ni - yi)))
        g_top = set(gi.nlargest(5).index)
        n_top = set(ni.nlargest(5).index)
        overlaps.append(len(g_top & n_top) / 5.0)
    return {"daily_rank_prediction_correlation": float(np.nanmean(pred_corr)),
            "daily_rank_error_correlation": float(np.nanmean(error_corr)),
            "top5_overlap_fraction": float(np.mean(overlaps))}


def paired_block_intervals(fold_values, block_sizes, family_size=26, repetitions=10000, seed=20261105):
    rng, results = np.random.default_rng(seed), {}
    for block in block_sizes:
        samples, total = np.zeros(repetitions, dtype=np.float64), 0
        for values in fold_values:
            values = np.asarray(values, dtype=np.float64)
            if not len(values) or not np.isfinite(values).all():
                raise RuntimeError("bootstrap输入包含缺失篮子或非有限收益")
            blocks = int(np.ceil(len(values) / block))
            starts = rng.integers(0, len(values), size=(repetitions, blocks))
            offsets = np.arange(block)
            draws = values[(starts[:, :, None] + offsets[None, None, :]) % len(values)]
            samples += draws.reshape(repetitions, -1)[:, :len(values)].sum(axis=1)
            total += len(values)
        if total == 0:
            raise RuntimeError("bootstrap没有样本")
        samples /= total
        results[str(block)] = {"ordinary_95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "bonferroni_95_family26": [float(np.quantile(samples, 0.05 / family_size / 2)),
                                       float(np.quantile(samples, 1 - 0.05 / family_size / 2))]}
    return results


def analyze_components(repetitions=10000):
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("缺少本轮实验清单")
    contrasts = (("G", "N"), ("F50", "G"), ("F50", "N"), ("FW", "G"), ("FW", "N"), ("FW", "F50"))
    required = len(FOLDS) * len(HORIZONS) * len(SEEDS) * len(ARMS)
    actual = sum(1 for horizon in HORIZONS for fold in FOLDS for seed in SEEDS for arm in ARMS
                 if (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.parquet").exists()
                 and (OUTPUT / f"{fold}_{horizon}_{seed}_{arm}.json").exists())
    if actual != required:
        result = {"status": "pending", "actual_predictions": actual, "required_predictions": required,
                  "comparisons": [], "research_only": True}
        write_json(OUTPUT / "analysis.json", result)
        print(json.dumps({"status": "pending", "actual_predictions": actual, "required_predictions": required}, ensure_ascii=False))
        return
    manifest = json.loads((LEGACY_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    calendar, comparisons, complementarity = calendar_index(), [], {}
    for horizon in HORIZONS:
        length = int(horizon[1:])
        block_days = (20, 40, 60) if horizon == "H10" else (40, 60, 80)
        block_sizes = tuple(max(1, value // length) for value in block_days)
        fold_data = {}
        for fold in FOLDS:
            start, end = map(pd.Timestamp, manifest["folds"][fold]["test"])
            test_calendar = calendar[(calendar >= start) & (calendar <= end)]
            formed = test_calendar[np.arange(len(test_calendar)) % length == 0]
            expected_index, expected_labels, _ = expected_test_index(fold, horizon)
            seed_metrics, ensemble_metrics, complementarity_seed = {}, {}, {}
            for seed in SEEDS:
                seed_metrics[seed] = {}
                for arm in ARMS:
                    frame = prediction_frame(fold, horizon, seed, arm)
                    if not frame.index.equals(expected_index):
                        raise RuntimeError(f"{fold}/{horizon}/{seed}/{arm}: 股票日覆盖不等于冻结标签分母")
                    if not np.array_equal(frame["label"].to_numpy(), expected_labels["label"].to_numpy()) or not np.array_equal(frame["label_rank"].to_numpy(), expected_labels["label_rank"].to_numpy()):
                        raise RuntimeError(f"{fold}/{horizon}/{seed}/{arm}: 标签与冻结标签不一致")
                    seed_metrics[seed][arm] = daily_metrics(frame, formed)
                    del frame
                g_frame, n_frame = (prediction_frame(fold, horizon, seed, arm) for arm in ("G", "N"))
                complementarity_seed[seed] = complementarity_metrics(g_frame, n_frame)
                del g_frame, n_frame
            for arm in ARMS:
                frames = [prediction_frame(fold, horizon, seed, arm) for seed in SEEDS]
                if any(not frames[0].index.equals(frame.index) or not np.array_equal(frames[0]["label"].to_numpy(), frame["label"].to_numpy()) for frame in frames[1:]):
                    raise RuntimeError(f"{fold}/{horizon}/{arm}: 种子共同样本或标签不一致")
                mean_frame = pd.DataFrame({"score": np.mean([frame["score"].to_numpy(dtype=np.float64) for frame in frames], axis=0),
                    "label": frames[0]["label"].to_numpy(), "label_rank": frames[0]["label_rank"].to_numpy()}, index=frames[0].index)
                ensemble_metrics[arm] = daily_metrics(mean_frame, formed)
                del frames, mean_frame
            fold_data[fold] = {"ensemble": ensemble_metrics, "seeds": seed_metrics,
                               "complementarity": complementarity_seed}
            values = list(complementarity_seed.values())
            complementarity[f"{fold}_{horizon}"] = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
            del expected_index, expected_labels
            gc.collect()
        for candidate, baseline in contrasts:
            paired, folds_result = [], []
            seed_values = {str(seed): [] for seed in SEEDS}
            down_candidate = down_baseline = 0.0
            for fold in FOLDS[:7]:
                row = fold_data[fold]
                c, b = row["ensemble"][candidate], row["ensemble"][baseline]
                delta = c["top5"] - b["top5"]
                paired.append(delta)
                down_candidate += float(np.square(np.minimum(c["top5"], 0)).sum())
                down_baseline += float(np.square(np.minimum(b["top5"], 0)).sum())
                for seed in SEEDS:
                    seed_values[str(seed)].extend((row["seeds"][seed][candidate]["top5"] - row["seeds"][seed][baseline]["top5"]).tolist())
                folds_result.append({"fold": fold, "basket_count": len(delta), "top5_diff_bp": float(delta.mean() * 10000),
                    "top50_diff_bp": float((c["top50"] - b["top50"]).mean() * 10000),
                    "top5_excess_diff_bp": float((c["top5_excess"] - b["top5_excess"]).mean() * 10000),
                    "top50_excess_diff_bp": float((c["top50_excess"] - b["top50_excess"]).mean() * 10000),
                    "common_pool_mean_bp": float(c["pool"].mean() * 10000),
                    "rank_ic_diff": c["rank_ic"] - b["rank_ic"], "candidate_top5_bp": float(c["top5"].mean() * 10000),
                    "baseline_top5_bp": float(b["top5"].mean() * 10000)})
            intervals = paired_block_intervals(paired, block_sizes, repetitions=repetitions)
            mean_bp = float(np.concatenate(paired).mean() * 10000)
            seed_means = {seed: float(np.mean(values) * 10000) for seed, values in seed_values.items()}
            positive_folds = sum(row["top5_diff_bp"] > 0 for row in folds_result)
            positive_seeds = sum(value > 0 for value in seed_means.values())
            rank_diff = float(np.mean([row["rank_ic_diff"] for row in folds_result]))
            semivar_ratio = down_candidate / down_baseline if down_baseline > 0 else (0.0 if down_candidate == 0 else None)
            lower = intervals[str(block_sizes[0])]["bonferroni_95_family26"][0]
            screen = bool(mean_bp >= (10 if horizon == "H10" else 20) and positive_folds >= 5 and positive_seeds >= 2
                          and rank_diff >= -0.005 and semivar_ratio is not None and semivar_ratio <= 1.10)
            w8 = fold_data["W8"]["ensemble"]
            comparisons.append({"horizon": horizon, "comparison": f"{candidate}-{baseline}", "family_size": 26,
                "mean_top5_diff_bp": mean_bp, "folds_positive": positive_folds, "seed_mean_diff_bp": seed_means,
                "positive_seeds": positive_seeds, "mean_rank_ic_diff": rank_diff,
                "downside_semivariance_ratio": semivar_ratio, "screen_pass": screen,
                "stable_improvement": bool(screen and lower > 0), "folds": folds_result,
                "block_intervals": intervals,
                "W8_top5_diff_bp": float((w8[candidate]["top5"] - w8[baseline]["top5"]).mean() * 10000)})
    result = {"status": "evaluated", "primary_folds": list(FOLDS[:7]), "sensitivity_fold": "W8",
        "actual_predictions": actual, "required_predictions": required, "comparisons": comparisons,
        "complementarity": complementarity,
        "research_only": True, "strict_pit_verified": False, "execution_verified": False,
        "bootstrap_repetitions": repetitions, "family_size": 26,
        "caveats": ["历史窗口已被研究接触", "旧行业和身份历史发布版本未经认证", "未包含账户执行成本"]}
    write_json(OUTPUT / "analysis.json", result)
    print(json.dumps({"status": result["status"], "comparisons": len(comparisons),
        "stable_improvements": sum(row["stable_improvement"] for row in comparisons),
        "output": str(OUTPUT / "analysis.json")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "audit", "compact", "prune", "recover", "analyze", "status"))
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.stage == "inventory":
        file_inventory()
    elif args.stage == "audit":
        strict_audit()
    elif args.stage == "compact":
        compact_core(args.folds, args.horizons, args.seeds)
    elif args.stage == "prune":
        prune_legacy_predictions(args.apply)
    elif args.stage == "recover":
        recover_components(args.folds, args.horizons, args.seeds, args.batch_size)
    elif args.stage == "analyze":
        analyze_components()
    else:
        manifest = OUTPUT / "manifest.json"
        inventory = OUTPUT / "inventory.json"
        report = OUTPUT / "legacy_strict_audit.json"
        print(json.dumps({"output": str(OUTPUT), "manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None,
                          "inventory_exists": inventory.exists(), "audit": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None},
                         ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
