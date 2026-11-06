import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.run_expert_router_oos as experiment

experiment.FINAL_OOS = dict(experiment.FINAL_OOS)
experiment.FINAL_OOS["test"] = ("2026-07-01", "2026-08-10")
source_output = Path("data/experiments/expert_router_oos_h10_parallel2.json")
matured_output = Path("data/experiments/expert_router_oos_h10_matured.json")
if not matured_output.exists():
    previous = json.loads(source_output.read_text(encoding="utf-8"))
    groups = {key: list(value) for key, value in previous["config"]["factor_groups"].items()}
    instruments = list(previous["config"]["instruments"])
    requested_modes = ["full", "weighted", "subset"]
    payload = {
        "schema_version": 2, "label_horizon": experiment.LABEL_HORIZON,
        "label_mode": experiment.LABEL_MODE, "rebalance_step": experiment.REBALANCE_STEP,
        "top_ns": experiment.TOP_NS, "router_top_n": experiment.ROUTER_TOP_N,
        "costs": {"commission": experiment.COMMISSION_RATE, "slippage": experiment.SLIPPAGE_RATE,
                  "stamp_duty": experiment.STAMP_DUTY}, "windows": experiment.WINDOWS,
        "final_oos": experiment.FINAL_OOS, "modes": requested_modes, "epochs": 80,
        "factor_groups": groups, "instruments": instruments,
        "factor_hash": experiment._stable_hash(groups),
        "instrument_hash": experiment._stable_hash(sorted(instruments)),
        "router_features": experiment.ROUTER_FEATURES,
        "implementation_hashes": experiment._implementation_hashes(),
    }
    previous["experiment_fingerprint"] = experiment._stable_hash(payload)
    previous["config"] = payload
    with matured_output.open("x", encoding="utf-8") as output_file:
        json.dump(previous, output_file, ensure_ascii=False, indent=2)
sys.argv = [
    "run_expert_router_oos.py",
    "--modes", "full", "weighted", "subset",
    "--windows", "w1", "w2", "w3",
    "--run-final-oos", "--final-mode", "weighted",
    "--epochs", "80",
    "--output", str(matured_output),
]
experiment.main()
