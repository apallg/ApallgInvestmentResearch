import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.service.score_ablation import run_score_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--rebalance-step", type=int, default=10)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--candidate-depth", type=int, default=500)
    parser.add_argument("--max-exit-delay", type=int, default=5)
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-non-pit-reference", action="store_true")
    args = parser.parse_args()
    report, path = run_score_ablation(
        prediction_file=args.prediction_file,
        start=args.start,
        end=args.end,
        model_id=args.model_id,
        top_n=args.top_n,
        horizon=args.horizon,
        rebalance_step=args.rebalance_step,
        max_dates=args.max_dates,
        candidate_depth=args.candidate_depth,
        max_exit_delay=args.max_exit_delay,
        output_dir=args.output_dir,
        allow_non_pit_reference=args.allow_non_pit_reference,
    )
    summary = {
        name: {
            "net_return": value["summary"]["net_return"],
            "max_drawdown": value["summary"]["max_drawdown"],
            "mean_turnover": value["summary"]["mean_one_way_turnover"],
            "mean_delta": value["paired_vs_baseline"]["mean_net_return_delta"],
            "verdict": report["evidence_assessment"][name]["verdict"],
        }
        for name, value in report["replay"]["variants"].items()
    }
    print(json.dumps({
        "report": str(path),
        "experiment_fingerprint": report["experiment_fingerprint"],
        "periods": len(report["decision_dates"]),
        "controls": report["controls"],
        "production_change_allowed": report["production_change_allowed"],
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
