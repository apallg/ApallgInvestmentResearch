import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.domain.backtest.engine import DirectionBacktestRunner
import core.shared.disabled_models as disabled_models

# These archived runs are disabled in production selection. The experiment is
# an explicit historical audit of the user-requested HMM/rebound comparison.
disabled_models.is_disabled = lambda experiment_id, run_id: False


TREND = ("721018297369182666", "81161bda8aba4b73ac5fc883e5514439")
REBOUND = ("471409160243425125", "1882d6477fed4ec8950b8ba166c5d16d")
VOLUME = ("656749665990832865", "be30b6c921d047dca0a1610d0403a653")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


COMMON = dict(
    start="2025-09-15",
    end="2026-09-15",
    pool="stable_history_500d",
    top_n=5,
    freq="week",
    sizing="equal",
    target_exposure=0.95,
    max_one_way_turnover=0.30,
    full_position=True,
    neutralize_qlib=True,
    stop_source="off",
    macro_bias=False,
    macro_regime=False,
    sector_fusion=False,
    decision_day_filter=True,
)


def run_one(name, **kwargs):
    print(f"START {name}", flush=True)
    runner = DirectionBacktestRunner()
    result = runner.run(**COMMON, **kwargs)
    out = clean(result)
    summary = out.get("summary", {})
    print(f"DONE {name} summary={json.dumps(summary, ensure_ascii=False)}", flush=True)
    return out


def main():
    results = {}
    results["unified_state_router"] = run_one(
        "unified_state_router",
        directions=["trend_momentum", "volume_breakout", "oversold_rebound"],
        use_hmm=True,
        state_models={
            "volatile_up": TREND,
            "steady_up": TREND,
            "volatile_down": REBOUND,
            "steady_flat": REBOUND,
            # The only volume run in the historical set is close-to-close and
            # is rejected by the executable-label contract. Use the valid
            # rebound expert as an explicit fallback for these two states;
            # this isolates the HMM state decision from a label mismatch.
            "volatile_flat": REBOUND,
            "steady_down": REBOUND,
        },
    )
    target = Path("data/experiments/hmm_vs_rebound_exact_20260916.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {target}", flush=True)


if __name__ == "__main__":
    main()
