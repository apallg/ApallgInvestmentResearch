import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data import Database

RUN_ID = "5501ed88f12145f0a454d68d92dbad76"
EXPERIMENT_ID = "656749665990832865"
TRAIN = ("2023-01-01", "2025-12-31")
VALID = ("2026-01-01", "2026-03-31")
CALIBRATE = ("2026-04-01", "2026-06-30")
TEST = ("2026-07-01", "2026-09-15")
SEED = 42
GROUPS = {
    "金融地产": {"银行", "非银金融", "房地产"},
    "资源材料": {"煤炭", "石油石化", "有色金属", "钢铁", "基础化工", "建筑材料"},
    "基建公用": {"公用事业", "交通运输", "环保", "建筑装饰"},
    "工业制造": {"机械设备", "电力设备", "国防军工", "汽车", "轻工制造"},
    "消费医疗": {"食品饮料", "农林牧渔", "家用电器", "纺织服饰", "商贸零售", "社会服务", "美容护理", "医药生物"},
    "科技传媒": {"电子", "计算机", "通信", "传媒"},
}
GROUP_NAMES = list(GROUPS)
PANEL_PATH = ROOT / "data/experiments/stock_context_p1_w8_qfqfix_20261020/context_features.parquet"
RESULT_DIR = ROOT / "data/experiments/stock_group_split_p1_pilot_20261020_qfqfix"

coverage_path = PANEL_PATH.with_name("coverage.json")
if not coverage_path.exists():
    raise RuntimeError("P1面板缺少质量报告，禁止启动实验")
coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
rates = coverage.get("missing_indicator_rate_before_value_fill") or {}
if coverage.get("feature_contract_version") != 2 or not rates:
    raise RuntimeError("P1面板特征契约未通过，须重建面板")
if any(float(value) > 0.02 for value in rates.values()):
    raise RuntimeError(f"P1面板未达到98%有效覆盖门槛: {rates}")
if coverage.get("history_denominator_verified") is not True:
    raise RuntimeError("历史股票池与停牌分母尚未核验，禁止启动正式实验")
if coverage.get("stockyear_coverage_verified") is not True:
    raise RuntimeError("每个股票年度95%覆盖门槛尚未核验")
if coverage.get("industry_pit_verified") is not True:
    raise RuntimeError("历史行业分类缺少发布时间版本，禁止作为严格PIT实验")
from backend.service.historical_quality import get_historical_quality
if not get_historical_quality()["training_ready"]:
    raise RuntimeError("数据中心历史行情质量检查未通过，禁止启动正式实验")

run_dir = ROOT / "mlruns" / EXPERIMENT_ID / RUN_ID / "artifacts"
raw_keys = pickle.loads((run_dir / "factor_keys.json").read_bytes())
factor_keys = json.loads(raw_keys)["custom_factor_keys"]
factor_digest = hashlib.sha256(json.dumps(factor_keys, ensure_ascii=False).encode("utf-8")).hexdigest()
from core.adapters.qlib.models import QlibModelManager, PARAM_PROFILES
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

manager = QlibModelManager()
source_valid = (VALID[0], CALIBRATE[1])
task_config = manager._build_task_config(
    "lightgbm_rank", "WhiteBoxHandler", "all", TRAIN, source_valid, TEST,
    model_params=dict(PARAM_PROFILES["conservative"]["lightgbm"]),
    label_horizon=10, custom_factor_keys=factor_keys,
    enable_cross_sectional=False, enable_winsorize=True,
    label_norm="none", label_mode="tradable_open_to_open",
)
source = init_instance_by_config(task_config["dataset"])
prepared = source.prepare(["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
valid_dates = pd.to_datetime(prepared[1].index.get_level_values("datetime"))
calibration_rows = int(((valid_dates >= pd.Timestamp(CALIBRATE[0])) & (valid_dates <= pd.Timestamp(CALIBRATE[1]))).sum())
if calibration_rows == 0:
    raise RuntimeError(f"Qlib source valid segment does not cover calibration period {CALIBRATE}")
print(json.dumps({"source_valid_rows": len(prepared[1]), "source_valid_start": str(valid_dates.min().date()), "source_valid_end": str(valid_dates.max().date()), "calibration_rows": calibration_rows}, ensure_ascii=False), flush=True)
if not PANEL_PATH.exists():
    raise FileNotFoundError(f"P1上下文面板不存在: {PANEL_PATH}")
context_panel = pd.read_parquet(PANEL_PATH)
context_keys = [column for column in context_panel.columns if column != "group_id"]
extra_frames = []
group_frames = []
for frame in prepared:
    index = frame.index
    aligned = context_panel.reindex(index)
    missing_panel_share = float(aligned[context_keys].isna().all(axis=1).mean())
    if missing_panel_share > 0.01:
        raise RuntimeError(f"P1面板无法对齐Qlib样本，全部上下文缺失比例={missing_panel_share:.2%}")
    aligned[context_keys] = aligned[context_keys].fillna(0).astype(np.float32)
    extra_frames.append(aligned[context_keys])
    group_frames.append(aligned["group_id"].fillna(len(GROUP_NAMES)).astype("uint8").rename("group_id"))
extra_features = pd.concat(extra_frames).sort_index()
group_by_index = pd.concat(group_frames).sort_index()
del prepared, extra_frames, group_frames

def group_filter(group_id):
    def apply(index):
        return group_by_index.reindex(index).to_numpy() == group_id
    return apply

from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion

base_args = {
    "instruments": "all",
    "train_period": TRAIN,
    "valid_period": VALID,
    "test_period": TEST,
    "label_horizon": 10,
    "label_norm": "none",
    "fusion_weight": 0.5,
    "param_profile": "conservative",
    "label_mode": "tradable_open_to_open",
    "dataset": source,
    "seed": SEED,
    "calibration_period": CALIBRATE,
}
result_a = train_lgb_lstm_fusion(factor_keys=factor_keys, **base_args)
context_factor_keys = factor_keys + context_keys
result_b = train_lgb_lstm_fusion(
    factor_keys=context_factor_keys, extra_features=extra_features, **base_args
)

test_group = group_by_index.reindex(result_b["fused_pred"].index).astype("Int64")
calibration_group = group_by_index.reindex(result_b["calibration_pred"].index).astype("Int64")
expert_results = {}
expert_failures = {}
for group_id, group_name in enumerate(GROUP_NAMES):
    train_dates = pd.to_datetime(group_by_index.index.get_level_values("datetime")) <= pd.Timestamp(TRAIN[1])
    training_groups = group_by_index.loc[train_dates]
    selected = training_groups == group_id
    selected_train = int(selected.sum())
    selected_stocks = int(training_groups.index.get_level_values("instrument")[selected.to_numpy()].nunique())
    selected_days = int(training_groups.index.get_level_values("datetime")[selected.to_numpy()].nunique())
    if selected_train < 50000 or selected_stocks < 100 or selected_days < 500:
        expert_failures[group_name] = {
            "reason": "expert_minimum_sample_gate",
            "train_rows": selected_train,
            "stocks": selected_stocks,
            "days": selected_days,
        }
        continue
    try:
        expert_results[group_id] = train_lgb_lstm_fusion(
            factor_keys=context_factor_keys,
            extra_features=extra_features,
            train_sample_filter=group_filter(group_id),
            **base_args,
        )
    except Exception as exc:
        expert_failures[group_name] = {"reason": type(exc).__name__, "message": str(exc)[:500]}

def fit_calibration(prediction, label, group_ids):
    frame = pd.concat({"score": prediction, "label": label, "group": group_ids}, axis=1).dropna()
    dates = frame.index.get_level_values("datetime")
    frame["weight"] = 1.0 / frame.groupby(level="datetime")["score"].transform("size")
    weight = frame["weight"].to_numpy(dtype=float)
    score = frame["score"].to_numpy(dtype=float)
    target = frame["label"].to_numpy(dtype=float)
    mean_score = np.average(score, weights=weight)
    mean_target = np.average(target, weights=weight)
    variance = np.average((score - mean_score) ** 2, weights=weight)
    raw_global_slope = np.average((score - mean_score) * (target - mean_target), weights=weight) / variance if variance > 1e-12 else 0.0
    global_slope = raw_global_slope if raw_global_slope > 1e-12 else 1.0
    global_intercept = mean_target - global_slope * mean_score
    maps = {}
    for group_id in range(len(GROUP_NAMES) + 1):
        part = frame.loc[frame["group"] == group_id]
        n_dates = int(part.index.get_level_values("datetime").nunique())
        if n_dates == 0:
            maps[group_id] = (global_intercept, global_slope, 0)
            continue
        w = part["weight"].to_numpy(dtype=float)
        x = part["score"].to_numpy(dtype=float)
        y = part["label"].to_numpy(dtype=float)
        mx = np.average(x, weights=w)
        my = np.average(y, weights=w)
        var = np.average((x - mx) ** 2, weights=w)
        raw_slope = np.average((x - mx) * (y - my), weights=w) / var if var > 1e-12 else 0.0
        slope = raw_slope if raw_slope > 1e-12 else global_slope
        intercept = my - slope * mx
        shrink = n_dates / (n_dates + 40.0)
        maps[group_id] = (
            shrink * intercept + (1 - shrink) * global_intercept,
            shrink * slope + (1 - shrink) * global_slope,
            n_dates,
        )
    return maps

def apply_calibration(prediction, group_ids, maps):
    group_ids = group_ids.reindex(prediction.index)
    output = pd.Series(np.nan, index=prediction.index, dtype=float)
    for group_id, (intercept, slope, _) in maps.items():
        mask = group_ids == group_id
        output.loc[mask] = intercept + slope * prediction.loc[mask]
    return output

label_test = result_b["label"].astype(float)
label_valid = result_b["valid_label"].astype(float)
calibrations = {}
calibrations["A"] = fit_calibration(result_a["calibration_pred"], result_a["calibration_label"], calibration_group.reindex(result_a["calibration_pred"].index))
calibrations["B"] = fit_calibration(result_b["calibration_pred"], result_b["calibration_label"], calibration_group)
predictions = {
    "A": apply_calibration(result_a["fused_pred"], test_group.reindex(result_a["fused_pred"].index), calibrations["A"]),
    "B": apply_calibration(result_b["fused_pred"], test_group, calibrations["B"]),
}
combined_c_calibration = pd.Series(np.nan, index=result_b["calibration_pred"].index, dtype=float)
combined_c_test = pd.Series(np.nan, index=result_b["fused_pred"].index, dtype=float)
for group_id, result in expert_results.items():
    calibration_mask = calibration_group.reindex(result["calibration_pred"].index) == group_id
    test_mask = test_group.reindex(result["fused_pred"].index) == group_id
    combined_c_calibration.loc[result["calibration_pred"].index[calibration_mask.fillna(False)]] = result["calibration_pred"].loc[calibration_mask.fillna(False)]
    combined_c_test.loc[result["fused_pred"].index[test_mask.fillna(False)]] = result["fused_pred"].loc[test_mask.fillna(False)]
calibrations["C"] = fit_calibration(combined_c_calibration, result_b["calibration_label"].reindex(combined_c_calibration.index), calibration_group)
predictions["C"] = apply_calibration(combined_c_test, test_group.reindex(combined_c_test.index), calibrations["C"])
unknown_mask = test_group == len(GROUP_NAMES)
predictions["C"].loc[unknown_mask] = predictions["B"].reindex(predictions["C"].index).loc[unknown_mask]
fallback_mask = predictions["C"].isna()
predictions["C"].loc[fallback_mask] = predictions["B"].reindex(predictions["C"].index).loc[fallback_mask]

result_dir = RESULT_DIR
result_dir.mkdir(parents=True, exist_ok=True)
output_frames = []
for arm, prediction in predictions.items():
    output_frames.append(pd.concat({"prediction": prediction, "label": label_test.reindex(prediction.index), "group_id": test_group.reindex(prediction.index)}, axis=1).assign(arm=arm, seed=SEED, fold="W8"))
prediction_frame = pd.concat(output_frames).rename_axis(index={"datetime": "date"}).reset_index()
prediction_frame.to_parquet(result_dir / "predictions.parquet", index=False)

summary = {}
for arm, prediction in predictions.items():
    frame = pd.concat({"score": prediction, "label": label_test.reindex(prediction.index), "group": test_group.reindex(prediction.index)}, axis=1).dropna()
    daily_ic = frame.groupby(level="datetime").apply(lambda part: part["score"].corr(part["label"], method="spearman")).dropna()
    summary[arm] = {
        "n_rows": int(len(frame)),
        "n_dates": int(frame.index.get_level_values("datetime").nunique()),
        "rank_ic_mean": float(daily_ic.mean()) if len(daily_ic) else None,
        "unknown_share": float((frame["group"] == len(GROUP_NAMES)).mean()) if len(frame) else None,
        "group_counts": {str(k): int(v) for k, v in frame["group"].value_counts().sort_index().items()},
    }
summary["paired_top5_10d_gross_diagnostic"] = {}
for arm, prediction in predictions.items():
    common_dates = sorted(set(prediction.dropna().index.get_level_values("datetime")) & set(label_test.dropna().index.get_level_values("datetime")))
    signal_dates = common_dates[::10]
    values = []
    for date in signal_dates:
        score = prediction.xs(date, level="datetime").dropna().nlargest(5).index
        values.append(float(label_test.xs(date, level="datetime").reindex(score).mean()))
    summary["paired_top5_10d_gross_diagnostic"][arm] = {"signals": len(values), "mean_return": float(np.mean(values)) if values else None, "returns": values}

manifest = {
    "status": "p1_context_pilot_complete",
    "evidence_stage": "single_incomplete_historical_fold_diagnostic",
    "date_frozen": "2026-10-20",
    "reference_run_id": f"{EXPERIMENT_ID}/{RUN_ID}",
    "factor_count": len(factor_keys),
    "factor_keys_sha256": factor_digest,
    "taxonomy": "G1 six broad industries plus unknown; historical membership applied only after in_date",
    "context_panel_rows": len(context_panel),
    "arms": {"A": "40 original features, unified model", "B": "same plus 39 P1 context features, unified model", "C": "same features, six group experts with B fallback for unknown"},
    "calibration_method": "separate 2026Q2, strictly positive slopes; non-positive group slope falls back to pooled slope to preserve within-group order",
    "train": TRAIN,
    "early_stopping": VALID,
    "calibration": CALIBRATE,
    "test": TEST,
    "label": "tradable_open_to_open, horizon 10, label_norm none",
    "universe": "Qlib all instrument history",
    "seed": SEED,
    "lstm_epochs": 80,
    "expert_failures": expert_failures,
    "full_plan_decision_eligible": False,
    "context_features": context_keys,
    "limitations": ["one seed and W8 only", "W8 is a partial historical window", "industry source has no fully versioned publication vintage", "no weekly execution ledger in this pilot"],
}
(result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"result_dir": str(result_dir), "summary": summary, "manifest": manifest}, ensure_ascii=False))
