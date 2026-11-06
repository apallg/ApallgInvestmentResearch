"""在可交易开盘到开盘标签下训练核心/卫星 LGB+LSTM 对照组。"""
import argparse
import json
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adapters.qlib.config import init_qlib
from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion, persist_fusion_run
from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager
from backend.service.instrument_resolver import build_filtered_instruments


SETS_PATH = ROOT / "core" / "domain" / "factors" / "factor_pool_state.json"
OUT_DIR = ROOT / "data" / "experiments"


def _load_sets():
    curated = json.loads(SETS_PATH.read_text(encoding="utf-8"))["curated"]
    core = list(curated["core_base"]["flat_list"])
    return core, list(curated["trend_satellite"]["flat_list"]), list(curated["rebound_satellite"]["flat_list"])


def _raw_top_metrics(prediction, label):
    import pandas as pd

    p = prediction.rename("prediction")
    y = label.rename("label")
    frame = pd.concat([p, y], axis=1).replace([float("inf"), float("-inf")], float("nan")).dropna()
    result = {}
    for top_n in (5, 10, 20):
        daily = frame.groupby(level="datetime", sort=False).apply(
            lambda group: group.nlargest(top_n, "prediction")["label"].mean()
        )
        universe = frame.groupby(level="datetime", sort=False)["label"].mean()
        excess = daily - universe
        result[f"top{top_n}_mean"] = float(daily.mean())
        result[f"top{top_n}_excess"] = float(excess.mean())
        result[f"top{top_n}_excess_positive_ratio"] = float((excess > 0).mean())
    daily_ic = frame.groupby(level="datetime", sort=False).apply(
        lambda group: group["prediction"].corr(group["label"], method="spearman")
    ).dropna()
    result["rank_ic"] = float(daily_ic.mean()) if len(daily_ic) else None
    result["icir"] = float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) > 1 and daily_ic.std() > 0 else None
    result["n_dates"] = int(len(daily_ic))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--roles", nargs="+", choices=("core", "trend", "rebound"), default=("core", "trend", "rebound"))
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    core, trend, rebound = _load_sets()
    groups = {
        "core": core,
        "trend": list(dict.fromkeys(core + trend)),
        "rebound": list(dict.fromkeys(core + rebound)),
    }
    prefixes = ["600", "601", "603", "605", "000", "001", "002", "003"]
    instruments = build_filtered_instruments("a500", prefixes)
    if not isinstance(instruments, list):
        raise RuntimeError("无法构建与既有模型一致的主板股票池")
    union = list(dict.fromkeys(core + trend + rebound))
    train_period = ("2022-01-01", "2024-12-31")
    valid_period = ("2025-01-01", "2025-06-30")
    test_period = ("2025-07-01", "2026-06-30")
    label_mode = "tradable_open_to_open"

    init_qlib()
    manager = QlibModelManager()
    source_config = manager._build_task_config(
        "lightgbm_rank", "WhiteBoxHandler", instruments,
        train_period, valid_period, test_period,
        model_params=dict(PARAM_PROFILES["conservative"]["lightgbm"]),
        label_horizon=args.horizon, custom_factor_keys=union,
        label_norm="csrank", enable_cross_sectional=False,
        enable_winsorize=True, label_mode=label_mode,
    )
    from qlib.utils import init_instance_by_config
    source = init_instance_by_config(source_config["dataset"])

    output = {
        "horizon": args.horizon,
        "label_mode": label_mode,
        "instrument_count": len(instruments),
        "factor_counts": {"core": len(core), "trend": len(trend), "rebound": len(rebound)},
        "models": {},
    }
    for role in args.roles:
        keys = groups[role]
        name = "core_control" if role == "core" else f"core_plus_{role}"
        print(f"training {name} h={args.horizon} factors={len(keys)}", flush=True)
        result = train_lgb_lstm_fusion(
            factor_keys=keys, instruments=instruments,
            train_period=train_period, valid_period=valid_period, test_period=test_period,
            label_horizon=args.horizon, label_mode=label_mode,
            label_norm="csrank", fusion_weight=0.5,
            param_profile="conservative", lstm_epochs=args.epochs,
            dataset=source,
        )
        from qlib.data.dataset.handler import DataHandlerLP
        raw_label = result["label"]
        try:
            raw_label = source.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
            if hasattr(raw_label, "columns"):
                raw_label = raw_label.iloc[:, 0]
        except Exception:
            pass
        metrics = _raw_top_metrics(result["fused_pred"], raw_label)
        run_id = persist_fusion_run(
            result, keys, instruments, train_period, valid_period, test_period,
            label_horizon=args.horizon, label_mode=label_mode,
            label_norm="csrank", fusion_weight=0.5,
            param_profile="conservative",
            experiment_name=f"repaired_tradable_h{args.horizon}_{role}",
        )
        metrics.update({"run_id": run_id, "role": role, "factor_count": len(keys)})
        output["models"][name] = metrics
        (OUT_DIR / f"repaired_tradable_h{args.horizon}_{role}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        del result
        gc.collect()
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
