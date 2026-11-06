import ast
import gc
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_ID = "5501ed88f12145f0a454d68d92dbad76"
EXPERIMENT_ID = "656749665990832865"
RUN_DIR = ROOT / "mlruns" / EXPERIMENT_ID / RUN_ID
OUT_DIR = ROOT / "data" / "experiments" / "qlib_label_norm_ab_20261015"
NORMS = ("csrank", "none", "cszscore")
TOP_NS = (5, 10, 20, 50)
WINDOWS = {
    "w1": {"train": ("2019-01-01", "2023-12-31"), "valid": ("2024-01-01", "2024-06-30"), "test": ("2024-07-01", "2024-12-31")},
    "w2": {"train": ("2019-01-01", "2024-06-30"), "valid": ("2024-07-01", "2024-12-31"), "test": ("2025-01-01", "2025-06-30")},
    "w3": {"train": ("2019-01-01", "2024-12-31"), "valid": ("2025-01-01", "2025-06-30"), "test": ("2025-07-01", "2025-12-31")},
    "w4": {"train": ("2019-01-01", "2025-06-30"), "valid": ("2025-07-01", "2025-12-31"), "test": ("2026-01-01", "2026-06-30")},
    "w5": {"train": ("2019-01-01", "2025-12-31"), "valid": ("2026-01-01", "2026-06-30"), "test": ("2026-07-01", "2026-09-15")},
}


def load_contract():
    params = RUN_DIR / "params" / "dataset.kwargs.handler.kwargs.factor_keys"
    factor_keys = ast.literal_eval(params.read_text(encoding="utf-8").strip())
    instruments_path = ROOT / "data" / "qlib_data" / "instruments" / "a500.txt"
    prefixes = ("sz000", "sz001", "sz002", "sz003", "sh600", "sh601", "sh603", "sh605")
    instruments = [
        line.split("\t", 1)[0].strip().lower()
        for line in instruments_path.read_text(encoding="utf-8").splitlines()
        if line.split("\t", 1)[0].strip().lower().startswith(prefixes)
    ]
    if len(factor_keys) != 40 or len(instruments) != 2986:
        raise RuntimeError(f"实验输入与新模型不一致: factors={len(factor_keys)}, instruments={len(instruments)}")
    return factor_keys, instruments


def save_pickle(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        pickle.dump(value, f, protocol=5)
    temp.replace(path)


def raw_series(source, segment):
    import pandas as pd
    from qlib.data.dataset.handler import DataHandlerLP

    value = source.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_R)
    if isinstance(value, pd.DataFrame):
        if isinstance(value.columns, pd.MultiIndex):
            value = value.xs("label", axis=1, level=0).iloc[:, 0]
        else:
            value = value.iloc[:, 0]
    return value.astype(float)


def daily_metrics(pred, label):
    import numpy as np
    import pandas as pd

    frame = pd.concat([pred.rename("pred"), label.rename("label")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for date, group in frame.groupby(level="datetime", sort=True):
        row = {"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "coverage": int(len(group))}
        row["rank_ic"] = group["pred"].corr(group["label"], method="spearman")
        universe = float(group["label"].mean())
        for top_n in TOP_NS:
            selected = group.nlargest(top_n, "pred")
            top_mean = float(selected["label"].mean())
            row[f"top{top_n}_return"] = top_mean
            row[f"top{top_n}_excess"] = top_mean - universe
            row[f"top{top_n}_positive"] = top_mean > 0
        rows.append(row)
    return rows, frame


def summarize_daily(rows):
    import numpy as np

    result = {"n_dates": len(rows)}
    for metric in ["rank_ic"] + [f"top{n}_{suffix}" for n in TOP_NS for suffix in ("return", "excess", "positive")]:
        values = [float(r[metric]) for r in rows if r.get(metric) is not None and np.isfinite(r[metric])]
        result[f"{metric}_mean"] = float(np.mean(values)) if values else None
    result["coverage_min"] = min((r["coverage"] for r in rows), default=0)
    result["coverage_mean"] = float(np.mean([r["coverage"] for r in rows])) if rows else 0.0
    return result


def bootstrap_delta(rows_by_fold, metric, n_boot=3000, block_len=20, seed=20261015):
    import numpy as np

    rng = np.random.default_rng(seed)
    folds = []
    for rows in rows_by_fold:
        values = np.asarray([r[metric] for r in rows if r.get(metric) is not None and np.isfinite(r[metric])], dtype=float)
        if len(values):
            folds.append(values)
    if not folds:
        return {"mean": None, "ci95": [None, None], "positive_day_ratio": None}
    observed = float(np.average([v.mean() for v in folds], weights=[len(v) for v in folds]))
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        means = []
        weights = []
        for values in folds:
            sampled = []
            while len(sampled) < len(values):
                start = int(rng.integers(0, max(1, len(values) - block_len + 1)))
                sampled.extend(values[start:min(start + block_len, len(values))].tolist())
            means.append(float(np.mean(sampled[:len(values)])))
            weights.append(len(values))
        samples[b] = float(np.average(means, weights=weights))
    all_values = np.concatenate(folds)
    return {
        "mean": observed,
        "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "positive_day_ratio": float(np.mean(all_values > 0)),
    }


def run_backtest(pred, label, window_id, norm, period):
    import pandas as pd
    import core.adapters.qlib.backtest as qlib_backtest
    from core.adapters.qlib.backtest import QlibBacktestRunner

    qlib_backtest.DEFAULT_EXCHANGE_KWARGS = {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "open",
        "open_cost": 0.0004,
        "close_cost": 0.0009,
        "min_cost": 5,
    }
    runner = QlibBacktestRunner()
    result = runner.run_from_predictions(
        predictions=pred,
        label=label,
        topk=50,
        n_drop=5,
        start_time=period["test"][0],
        end_time=period["test"][1],
        freq="day",
        experiment_name=f"label_norm_ab_20261015_{window_id}_{norm}",
    )
    report = result.get("report_normal")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return {"error": "回测未生成 report_normal"}
    account = report["account"].astype(float) if "account" in report else None
    daily_return = report["return"].astype(float) if "return" in report else None
    benchmark = report["bench"].astype(float) if "bench" in report else None
    summary = {
        "rows": int(len(report)),
        "initial_account": float(account.iloc[0]) if account is not None else None,
        "final_account": float(account.iloc[-1]) if account is not None else None,
        "total_return": float(account.iloc[-1] / account.iloc[0] - 1) if account is not None and account.iloc[0] else None,
        "benchmark_return": float((1 + benchmark.iloc[1:]).prod() - 1) if benchmark is not None and len(benchmark) > 1 else None,
        "mean_daily_return": float(daily_return.iloc[1:].mean()) if daily_return is not None and len(daily_return) > 1 else None,
        "max_drawdown": float((account / account.cummax() - 1).min()) if account is not None else None,
        "total_cost": float(report["total_cost"].iloc[-1]) if "total_cost" in report else None,
        "total_turnover": float(report["total_turnover"].iloc[-1]) if "total_turnover" in report else None,
        "qlib_metrics": result.get("metrics", {}),
    }
    summary["excess_return"] = (
        summary["total_return"] - summary["benchmark_return"]
        if summary["total_return"] is not None and summary["benchmark_return"] is not None else None
    )
    return summary


def main():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

    from core.adapters.qlib.config import init_qlib
    from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion
    from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager
    from qlib.utils import init_instance_by_config

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    factor_keys, instruments = load_contract()
    instruments_sha256 = hashlib.sha256("\n".join(instruments).encode()).hexdigest()
    contract = {
        "source_model": f"{EXPERIMENT_ID}/{RUN_ID}",
        "factor_keys": factor_keys,
        "factor_count": len(factor_keys),
        "instruments_count": len(instruments),
        "instruments_sha256": instruments_sha256,
        "label_horizon": 10,
        "label_mode": "tradable_open_to_open",
        "param_profile": "default",
        "fusion_weight": 0.5,
        "lstm_seq_len": 20,
        "lstm_epochs": 80,
        "random_seed": 42,
        "topk": 50,
        "n_drop": 5,
        "freq": "day",
        "benchmark": "sh000300",
        "norms": list(NORMS),
        "windows": WINDOWS,
        "costs": {"buy": 0.0004, "sell": 0.0009, "minimum": 5, "limit_threshold": 0.095},
    }
    contract_path = OUT_DIR / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("输出目录中已有不同实验合同，拒绝覆盖")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    init_qlib()
    manager = QlibModelManager()
    from qlib.data.dataset.handler import DataHandlerLP

    for window_id, period in WINDOWS.items():
        for norm in NORMS:
            stem = f"{window_id}_{norm}"
            pred_path = OUT_DIR / f"{stem}_pred.pkl"
            label_path = OUT_DIR / f"{stem}_label.pkl"
            if pred_path.exists() and label_path.exists():
                print(f"skip existing {stem}", flush=True)
                continue
            print(f"start {stem}: norm={norm} train={period['train']} valid={period['valid']} test={period['test']}", flush=True)
            source_config = manager._build_task_config(
                "lightgbm_rank", "WhiteBoxHandler", instruments,
                period["train"], period["valid"], period["test"],
                model_params=dict(PARAM_PROFILES["default"]["lightgbm"]),
                label_horizon=10,
                custom_factor_keys=factor_keys,
                label_norm=norm,
                enable_cross_sectional=False,
                enable_winsorize=True,
                label_mode="tradable_open_to_open",
            )
            source = init_instance_by_config(source_config["dataset"])
            result = train_lgb_lstm_fusion(
                factor_keys=factor_keys,
                instruments=instruments,
                train_period=period["train"],
                valid_period=period["valid"],
                test_period=period["test"],
                label_horizon=10,
                label_mode="tradable_open_to_open",
                label_norm=norm,
                fusion_weight=0.5,
                param_profile="default",
                lstm_epochs=80,
                dataset=source,
            )
            predictions = result["fused_pred"].astype(float)
            labels = source.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
            if hasattr(labels, "columns"):
                if getattr(labels.columns, "nlevels", 1) > 1:
                    labels = labels.xs("label", axis=1, level=0)
                labels = labels.iloc[:, 0]
            labels = labels.astype(float)
            save_pickle(pred_path, predictions)
            save_pickle(label_path, labels)
            print(f"done {stem}: predictions={len(predictions)} labels={len(labels)}", flush=True)
            del result, source, source_config
            gc.collect()

    variants = {}
    for window_id, period in WINDOWS.items():
        variants[window_id] = {}
        series = {}
        for norm in NORMS:
            with (OUT_DIR / f"{window_id}_{norm}_pred.pkl").open("rb") as f:
                pred = pickle.load(f)
            with (OUT_DIR / f"{window_id}_{norm}_label.pkl").open("rb") as f:
                label = pickle.load(f)
            series[norm] = (pred, label)
        common = series[NORMS[0]][0].dropna().index
        for norm in NORMS[1:]:
            common = common.intersection(series[norm][0].dropna().index)
        for norm in NORMS:
            _, label = series[norm]
            common = common.intersection(label.dropna().index)
        aligned_label = series[NORMS[0]][1].reindex(common)
        for norm in NORMS:
            pred = series[norm][0].reindex(common)
            daily, _ = daily_metrics(pred, aligned_label)
            summary = summarize_daily(daily)
            summary["prediction_rows_common"] = int(len(common))
            summary["backtest"] = run_backtest(pred, aligned_label, window_id, norm, period)
            variants[window_id][norm] = summary
            (OUT_DIR / f"{window_id}_{norm}_daily.json").write_text(
                json.dumps(daily, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
            )
            print(f"backtest {window_id}/{norm}: {summary['backtest']}", flush=True)

    comparisons = {}
    for norm in NORMS[1:]:
        comparisons[norm] = {}
        for top_n in TOP_NS:
            fold_rows = []
            for window_id, period in WINDOWS.items():
                arm_rows = json.loads((OUT_DIR / f"{window_id}_{norm}_daily.json").read_text(encoding="utf-8"))
                base_rows = json.loads((OUT_DIR / f"{window_id}_csrank_daily.json").read_text(encoding="utf-8"))
                base_map = {row["date"]: row for row in base_rows}
                fold_rows.append([
                    {"delta": row[f"top{top_n}_excess"] - base_map[row["date"]][f"top{top_n}_excess"]}
                    for row in arm_rows if row["date"] in base_map
                ])
            comparisons[norm][f"top{top_n}_excess_delta_vs_csrank"] = bootstrap_delta(
                [[{"v": row["delta"]} for row in rows] for rows in fold_rows], "v"
            )
        for key in ("rank_ic",):
            fold_rows = []
            for window_id in WINDOWS:
                arm_rows = json.loads((OUT_DIR / f"{window_id}_{norm}_daily.json").read_text(encoding="utf-8"))
                base_rows = json.loads((OUT_DIR / f"{window_id}_csrank_daily.json").read_text(encoding="utf-8"))
                base_map = {row["date"]: row for row in base_rows}
                fold_rows.append([
                    {"v": row[key] - base_map[row["date"]][key]}
                    for row in arm_rows if row.get(key) is not None and base_map.get(row["date"], {}).get(key) is not None
                ])
            comparisons[norm]["rank_ic_delta_vs_csrank"] = bootstrap_delta(fold_rows, "v")
        per_window_backtest = {}
        for window_id in WINDOWS:
            arm = variants[window_id][norm]["backtest"]
            base = variants[window_id]["csrank"]["backtest"]
            per_window_backtest[window_id] = {
                "arm_excess_return": arm.get("excess_return"),
                "csrank_excess_return": base.get("excess_return"),
                "delta": (arm.get("excess_return") - base.get("excess_return"))
                if arm.get("excess_return") is not None and base.get("excess_return") is not None else None,
            }
        comparisons[norm]["backtest_by_window"] = per_window_backtest

    report = {
        "status": "completed",
        "contract": contract,
        "results": variants,
        "comparisons_vs_csrank": comparisons,
        "caveats": [
            "股票池冻结为当前 Qlib a500 主板 2986 只，历史结果是对该当前候选集合的条件比较，不代表无幸存者偏差的全市场历史回测。",
            "TopN 标签收益是重叠的 10 日前瞻收益；差值置信区间用 20 交易日移动块 bootstrap。",
            "成本后组合回测采用 Qlib 日频 TopkDropoutStrategy (Top50, n_drop=5)，成本按买入 0.04%、卖出 0.09%、最低 5 元，并使用开盘成交和 9.5% 涨跌停阈值；与生产方向回测的持仓/止损规则不同。",
            "各 OOS 窗口训练一个模型；当前 LSTM 初始化固定 seed=42，因此结论仍需多 seed 重复确认。",
        ],
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)


if __name__ == "__main__":
    main()
