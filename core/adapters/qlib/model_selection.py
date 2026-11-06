import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

from core.shared import config
from core.shared.disabled_models import is_disabled
from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE, get_model_label_mode


def _read_value(path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return ast.literal_eval(text)
    except Exception:
        return None


def _artifact_metadata(artifacts):
    path = artifacts / "factor_keys.json"
    if not path.exists():
        return {}
    try:
        raw = path.read_bytes()
        start = raw.index(b"{")
        end = raw.rindex(b"}")
        return json.loads(raw[start:end + 1].decode("utf-8", errors="ignore"))
    except Exception:
        return {}


def _model_factors(run_dir, metadata):
    values = metadata.get("custom_factor_keys") or []
    if values:
        return list(dict.fromkeys(str(value) for value in values))
    value = _read_value(run_dir / "params" / "dataset.kwargs.handler.kwargs.factor_keys")
    return list(dict.fromkeys(str(item) for item in value)) if isinstance(value, (list, tuple)) else []


def _period_end(run_dir, metadata, name):
    value = _read_value(run_dir / "params" / f"dataset.kwargs.segments.{name}")
    if isinstance(value, (list, tuple)) and len(value) > 1:
        return str(value[1])[:10]
    direct = metadata.get(f"effective_{name}_end") or metadata.get(f"{name}_end")
    if direct:
        return str(direct)[:10]
    return None


def _period_start(run_dir, metadata, name):
    value = _read_value(run_dir / "params" / f"dataset.kwargs.segments.{name}")
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])[:10]
    direct = metadata.get(f"effective_{name}_start") or metadata.get(f"{name}_start")
    return str(direct)[:10] if direct else None


def _artifact_fingerprint(exp_id, run_id, artifacts, train_end, valid_end, test_end=None):
    digest = hashlib.sha256()
    digest.update("|".join([exp_id, run_id, train_end or "", valid_end or "", test_end or ""]).encode("utf-8"))
    for name in ("factor_keys.json", "params.pkl", "params_lgb.pkl", "params_xgb.pkl", "params_lin.pkl", "params_lstm.pkl"):
        path = artifacts / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def list_models_for_factor_set(factor_keys, cutoff, pool=None, allow_test_window=False):
    cutoff_date = dt.date.fromisoformat(str(cutoff)[:10])
    wanted_keys = list(dict.fromkeys(str(item) for item in factor_keys or []))
    wanted = set(wanted_keys)
    root = Path(str(config.QLIB_MLFLOW_URI).replace("file:", ""))
    candidates = []
    if not root.exists():
        return candidates
    from core.adapters.qlib.models import _read_run_metric
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ("0", ".trash"):
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            artifacts = run_dir / "artifacts"
            if not any((artifacts / name).exists() for name in ("params.pkl", "params_lgb.pkl")):
                continue
            metadata = _artifact_metadata(artifacts)
            label_mode = get_model_label_mode(run_dir)
            label_horizon = metadata.get("label_horizon")
            if label_horizon is None:
                label_horizon = _read_value(run_dir / "params" / "label_horizon")
            train_end = _period_end(run_dir, metadata, "train")
            valid_end = _period_end(run_dir, metadata, "valid") or train_end
            test_end = _period_end(run_dir, metadata, "test")
            train_start = _period_start(run_dir, metadata, "train")
            valid_start = _period_start(run_dir, metadata, "valid")
            test_start = _period_start(run_dir, metadata, "test")
            observed_end = max([value for value in (valid_end, test_end) if value], default=None)
            eligibility_end = valid_end if allow_test_window else observed_end
            reasons = []
            disabled = is_disabled(exp_dir.name, run_dir.name)
            if disabled:
                reasons.append("模型已停用")
            if label_mode != TRADABLE_LABEL_MODE:
                reasons.append(f"标签口径 {label_mode or '未知'} 不是可交易开盘到开盘")
            if not eligibility_end:
                reasons.append("缺少训练时点信息")
            try:
                if eligibility_end and dt.date.fromisoformat(eligibility_end) > cutoff_date:
                    reasons.append(f"模型数据截止 {eligibility_end} 晚于使用截止 {cutoff_date.isoformat()}")
            except ValueError:
                reasons.append("模型日期格式无效")
            instruments = _read_value(run_dir / "params" / "dataset.kwargs.handler.kwargs.instruments")
            if pool and isinstance(instruments, str) and not instruments.startswith("list_"):
                if instruments.lower() not in (str(pool).lower(), "all"):
                    reasons.append(f"股票池 {instruments} 与 {pool} 不一致")
            model_factors = _model_factors(run_dir, metadata)
            common = set(model_factors) & wanted
            warnings = []
            if wanted and not model_factors:
                reasons.append("模型缺少可核验的因子清单")
            elif wanted_keys and model_factors != wanted_keys:
                reasons.append("模型因子清单或顺序与Profile不一致")
            ic = _read_run_metric(run_dir, "IC")
            fingerprint = _artifact_fingerprint(
                exp_dir.name, run_dir.name, artifacts, train_end, valid_end, test_end)
            candidates.append({
                "model": (exp_dir.name, run_dir.name),
                "experiment_id": exp_dir.name,
                "run_id": run_dir.name,
                "factor_set": metadata.get("factor_set") or "",
                "label_horizon": label_horizon,
                "label_mode": label_mode,
                "instruments": instruments,
                "train_end": train_end,
                "train_start": train_start,
                "valid_end": valid_end,
                "valid_start": valid_start,
                "test_end": test_end,
                "test_start": test_start,
                "observed_end": observed_end,
                "factor_count": len(model_factors),
                "factor_keys": model_factors,
                "factor_overlap": len(common) / len(model_factors) if model_factors else 0.0,
                "baseline_coverage": len(common) / len(wanted) if wanted else 0.0,
                "disabled": disabled,
                "compatible": not reasons,
                "compatibility_reasons": reasons,
                "compatibility_warnings": warnings,
                "ic": float(ic) if ic is not None else None,
                "artifact_fingerprint": fingerprint,
            })
    candidates.sort(key=lambda item: (
        item["compatible"], item["baseline_coverage"], item["factor_overlap"], item["observed_end"] or "",
        item["ic"] if item["ic"] is not None else -999.0,
    ), reverse=True)
    return candidates


def select_model_for_factor_set(factor_keys, cutoff, pool=None):
    return next((item for item in list_models_for_factor_set(factor_keys, cutoff, pool)
                 if item["compatible"]), None)


def get_model_for_factor_set(experiment_id, run_id, factor_keys, cutoff, pool=None, allow_test_window=False):
    selected = next((item for item in list_models_for_factor_set(factor_keys, cutoff, pool, allow_test_window)
                     if item["experiment_id"] == str(experiment_id)
                     and item["run_id"] == str(run_id)), None)
    if selected is None:
        raise ValueError(f"Qlib 模型不存在: {experiment_id}/{run_id}")
    if not selected["compatible"]:
        raise ValueError("所选 Qlib 模型不可用: " + "；".join(selected["compatibility_reasons"]))
    return selected
