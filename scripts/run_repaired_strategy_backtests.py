"""运行修正标签模型的显式高低波路由策略回测。"""
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.service.instrument_resolver import build_filtered_instruments
from core.domain.backtest.engine import DirectionBacktestRunner


RUNS = {
    5: {
        "core": ("577465034734809505", "901b77972ae34467abddf34c87686d6c"),
        "trend": ("427925476646487394", "74c8f4ca9ef14eb7b438d6605de6a57d"),
        "rebound": ("802317758672341946", "b8e13686bd2740fab0dbb8331e6e6d07"),
    },
    10: {
        "core": ("840417442523026424", "a7f7d79c9f70438f93063f1b11dfb222"),
        "trend": ("314512136643759036", "a317876802924b6ca0fadd947460a931"),
        "rebound": ("471409160243425125", "1882d6477fed4ec8950b8ba166c5d16d"),
    },
    20: {
        "core": ("236387241205159736", "ab16997e74304fc391e1da4b94719e3a"),
        "trend": ("307160882825920581", "d15af143ae654acda5e37978e8d82586"),
        "rebound": ("873751003747923661", "9fcd8c4fb32f446f9d9026f29b71151b"),
    },
}


def _summary(result):
    summary = result.get("summary") or {}
    diagnostics = result.get("trade_diagnostics") or {}
    keys = ("total_return", "annual_return", "max_drawdown", "sharpe", "profit_factor",
            "win_rate", "n_trades", "annual_excess")
    output = {key: summary.get(key) for key in keys if key in summary}
    output.update({
        "transaction_cost": diagnostics.get("transaction_cost"),
        "stop_loss_count": diagnostics.get("stop_loss_count"),
        "avg_stop_return": diagnostics.get("avg_stop_return"),
    })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, choices=(5, 10, 20), default=(5, 10, 20))
    args = parser.parse_args()
    prefixes = ["600", "601", "603", "605", "000", "001", "002", "003"]
    instruments = build_filtered_instruments("a500", prefixes)
    if not isinstance(instruments, list):
        raise RuntimeError("无法构建与训练一致的股票池")
    output = {"start": "2025-07-01", "end": "2026-06-30", "pool_size": len(instruments), "runs": {}}
    for horizon in args.horizons:
        model_runs = RUNS[horizon]
        state_models = {
            "low_vol": list(model_runs["trend"]),
            "high_vol": list(model_runs["rebound"]),
        }
        runner = DirectionBacktestRunner()
        freq = f"{horizon}d"
        print(f"backtest h={horizon} route stop=off", flush=True)
        routed = runner.run(
            start=output["start"], end=output["end"], pool=instruments,
            top_n=5, freq=freq, use_timing=True, cash_yield=True,
            use_crf=True, use_vol_switch=True, vol_label_horizon=horizon,
            state_models=state_models, stop_source="off", sector_fusion=False,
        )
        output["runs"][f"h{horizon}_route_off"] = {
            "metrics": _summary(routed),
            "routing": routed.get("model_routing"),
        }
        if horizon == 10:
            print("backtest h=10 route stop=fixed", flush=True)
            fixed = runner.run(
                start=output["start"], end=output["end"], pool=instruments,
                top_n=5, freq=freq, use_timing=True, cash_yield=True,
                use_crf=True, use_vol_switch=True, vol_label_horizon=horizon,
                state_models=state_models, stop_source="fixed", sector_fusion=False,
            )
            output["runs"]["h10_route_fixed"] = {
                "metrics": _summary(fixed),
                "routing": fixed.get("model_routing"),
            }
        Path(ROOT / "data" / "experiments" / "repaired_strategy_backtests.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
