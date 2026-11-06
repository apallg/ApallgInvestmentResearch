"""补跑 10 日可交易标签的三个滚动 OOS 窗口。"""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adapters.qlib.config import init_qlib
from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion, persist_fusion_run
from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager
from backend.service.instrument_resolver import build_filtered_instruments


SETS_PATH = ROOT / "core" / "domain" / "factors" / "factor_pool_state.json"
OUT_PATH = ROOT / "data" / "experiments" / "rolling_oos_h10.json"

WINDOWS = {
    "w1": {
        "train": ("2019-07-01", "2023-12-31"),
        "valid": ("2024-01-01", "2024-06-30"),
        "test": ("2024-07-01", "2025-06-30"),
    },
    "w2": {
        "train": ("2020-07-01", "2024-12-31"),
        "valid": ("2025-01-01", "2025-06-30"),
        "test": ("2025-07-01", "2026-06-30"),
    },
    "w3": {
        "train": ("2021-07-01", "2025-06-30"),
        "valid": ("2025-07-01", "2025-12-31"),
        "test": ("2026-01-01", "2026-06-30"),
    },
}


def _load_groups():
    curated = json.loads(SETS_PATH.read_text(encoding="utf-8"))["curated"]
    core = list(curated["core_base"]["flat_list"])
    trend = list(dict.fromkeys(core + curated["trend_satellite"]["flat_list"]))
    rebound = list(dict.fromkeys(core + curated["rebound_satellite"]["flat_list"]))
    return {"core": core, "trend": trend, "rebound": rebound}


def _raw_top_metrics(prediction, label):
    import pandas as pd

    frame = pd.concat([
        prediction.rename("prediction"), label.rename("label"),
    ], axis=1).replace([float("inf"), float("-inf")], float("nan")).dropna()
    result = {}
    for top_n in (5, 10, 20):
        daily = frame.groupby(level="datetime", sort=False).apply(
            lambda group: group.nlargest(top_n, "prediction")["label"].mean()
        )
        universe = frame.groupby(level="datetime", sort=False)["label"].mean()
        excess = daily - universe
        result[f"top{top_n}_mean"] = float(daily.mean())
        result[f"top{top_n}_excess"] = float(excess.mean())
        result[f"top{top_n}_positive_ratio"] = float((excess > 0).mean())
    daily_ic = frame.groupby(level="datetime", sort=False).apply(
        lambda group: group["prediction"].corr(group["label"], method="spearman")
    ).dropna()
    result["rank_ic"] = float(daily_ic.mean()) if len(daily_ic) else None
    result["icir"] = (
        float(daily_ic.mean() / daily_ic.std())
        if len(daily_ic) > 1 and daily_ic.std() > 0 else None
    )
    result["n_dates"] = int(len(daily_ic))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--windows", nargs="+", choices=tuple(WINDOWS), default=tuple(WINDOWS))
    args = parser.parse_args()

    groups = _load_groups()
    prefixes = ["600", "601", "603", "605", "000", "001", "002", "003"]
    instruments = build_filtered_instruments("a500", prefixes)
    if not isinstance(instruments, list):
        raise RuntimeError("无法构建与既有模型一致的主板股票池")
    union = list(dict.fromkeys(key for keys in groups.values() for key in keys))

    init_qlib()
    manager = QlibModelManager()
    output = {
        "label_horizon": 10,
        "label_mode": "tradable_open_to_open",
        "instrument_count": len(instruments),
        "factor_counts": {key: len(value) for key, value in groups.items()},
        "epochs": args.epochs,
        "windows": {},
    }
    for window_id in args.windows:
        period = WINDOWS[window_id]
        print(f"window={window_id} train={period['train']} valid={period['valid']} test={period['test']}", flush=True)
        source_config = manager._build_task_config(
            "lightgbm_rank", "WhiteBoxHandler", instruments,
            period["train"], period["valid"], period["test"],
            model_params=dict(PARAM_PROFILES["conservative"]["lightgbm"]),
            label_horizon=10, custom_factor_keys=union,
            label_norm="csrank", enable_cross_sectional=False,
            enable_winsorize=True, label_mode="tradable_open_to_open",
        )
        from qlib.utils import init_instance_by_config
        source = init_instance_by_config(source_config["dataset"])
        window_result = {}
        for role, keys in groups.items():
            print(f"training window={window_id} role={role} factors={len(keys)}", flush=True)
            result = train_lgb_lstm_fusion(
                factor_keys=keys, instruments=instruments,
                train_period=period["train"], valid_period=period["valid"],
                test_period=period["test"], label_horizon=10,
                label_mode="tradable_open_to_open", label_norm="csrank",
                fusion_weight=0.5, param_profile="conservative",
                lstm_epochs=args.epochs, dataset=source,
            )
            from qlib.data.dataset.handler import DataHandlerLP
            raw_label = source.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
            if hasattr(raw_label, "columns"):
                raw_label = raw_label.iloc[:, 0]
            metrics = _raw_top_metrics(result["fused_pred"], raw_label)
            run_id = persist_fusion_run(
                result, keys, instruments, period["train"], period["valid"],
                period["test"], label_horizon=10,
                label_mode="tradable_open_to_open", label_norm="csrank",
                fusion_weight=0.5, param_profile="conservative",
                experiment_name=f"rolling_oos_h10_{window_id}_{role}",
            )
            metrics.update({"role": role, "run_id": run_id, "factor_count": len(keys)})
            window_result[role] = metrics
            del result
            gc.collect()
        output["windows"][window_id] = {"period": period, "models": window_result}
        OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        del source
        gc.collect()
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
