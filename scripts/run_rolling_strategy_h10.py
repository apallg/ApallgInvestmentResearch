"""用滚动 OOS 的 10 日模型运行同口径策略回测。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.service.instrument_resolver import build_filtered_instruments
from core.domain.backtest.engine import DirectionBacktestRunner


def _experiment_id(run_id):
    matches = list((ROOT / "mlruns").glob(f"*/{run_id}"))
    if not matches:
        raise RuntimeError(f"找不到 MLflow run: {run_id}")
    return matches[0].parent.name


def _summary(result):
    summary = result.get("summary") or {}
    diagnostics = result.get("trade_diagnostics") or {}
    keys = (
        "total_return", "annual_return", "max_drawdown", "sharpe",
        "profit_factor", "win_rate", "n_trades", "annual_excess",
    )
    output = {key: summary.get(key) for key in keys if key in summary}
    output.update({
        "transaction_cost": diagnostics.get("transaction_cost"),
        "stop_loss_count": diagnostics.get("stop_loss_count"),
        "avg_stop_return": diagnostics.get("avg_stop_return"),
    })
    return output


def main():
    source = json.loads((ROOT / "data" / "experiments" / "rolling_oos_h10.json").read_text(encoding="utf-8"))
    prefixes = ["600", "601", "603", "605", "000", "001", "002", "003"]
    instruments = build_filtered_instruments("a500", prefixes)
    if not isinstance(instruments, list):
        raise RuntimeError("无法构建与训练一致的股票池")
    output = {"freq": "10d", "stop_source": "off", "pool_size": len(instruments), "windows": {}}
    for window_id, window in source["windows"].items():
        period = window["period"]
        runs = window["models"]
        runner = DirectionBacktestRunner()
        print(f"backtest {window_id} {period['test'][0]}~{period['test'][1]}", flush=True)
        result = runner.run(
            start=period["test"][0], end=period["test"][1], pool=instruments,
            top_n=5, freq="10d", use_timing=True, cash_yield=True,
            use_crf=True, use_vol_switch=True, vol_label_horizon=10,
            state_models={
                "low_vol": [_experiment_id(runs["trend"]["run_id"]), runs["trend"]["run_id"]],
                "high_vol": [_experiment_id(runs["rebound"]["run_id"]), runs["rebound"]["run_id"]],
            }, stop_source="off", sector_fusion=False,
        )
        output["windows"][window_id] = {
            "period": period["test"],
            "metrics": _summary(result),
            "routing": result.get("model_routing"),
        }
        (ROOT / "data" / "experiments" / "rolling_strategy_h10.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
        )
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
