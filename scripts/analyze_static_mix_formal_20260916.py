import json
import math
import random
from pathlib import Path


SOURCE = Path("data/experiments/formal_rolling_router_account_20260916.json")
TARGET = Path("data/experiments/static_mix_vs_router_diagnostic_20260916.json")


def max_drawdown(values):
    peak = values[0]
    result = 0.0
    for value in values:
        peak = max(peak, value)
        result = min(result, value / peak - 1.0)
    return result


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    accounts = source["accounts"]
    trend = accounts["fixed_trend"]["daily"]
    rebound = accounts["fixed_rebound"]["daily"]
    hmm = accounts["HMM_point_in_time"]["daily"]
    assert len(trend) == len(rebound) == len(hmm)
    assert [x["date"] for x in trend] == [x["date"] for x in rebound] == [x["date"] for x in hmm]

    mix_nav = [0.5 * x["nav"] + 0.5 * y["nav"] for x, y in zip(trend, rebound)]
    mix_returns = [mix_nav[i] / mix_nav[i - 1] - 1.0 for i in range(1, len(mix_nav))]
    hmm_returns = [hmm[i]["nav"] / hmm[i - 1]["nav"] - 1.0 for i in range(1, len(hmm))]
    delta = [x - y for x, y in zip(hmm_returns, mix_returns)]
    rebound_returns = [rebound[i]["nav"] / rebound[i - 1]["nav"] - 1.0 for i in range(1, len(rebound))]
    hmm_minus_rebound = [x - y for x, y in zip(hmm_returns, rebound_returns)]

    def block_ci(values, block, n_boot=5000, seed=20260916):
        rng = random.Random(seed + block)
        n = len(values)
        samples = []
        for _ in range(n_boot):
            sampled = []
            while len(sampled) < n:
                start = rng.randrange(0, n)
                sampled.extend(values[(start + offset) % n] for offset in range(block))
            samples.append(sum(sampled[:n]) / n)
        samples.sort()
        return {"mean": sum(values) / n, "lower": samples[int(n_boot * 0.025)], "upper": samples[int(n_boot * 0.975)], "n": n, "block": block}
    result = {
        "status": "diagnostic_static_mix_from_existing_account_artifact",
        "source": str(SOURCE),
        "warning": "source formal rolling artifact has known contract mismatches; this is not final OOS evidence",
        "n_periods": len(mix_nav),
        "static_50_50": {
            "final_nav": mix_nav[-1],
            "return": mix_nav[-1] - 1.0,
            "max_drawdown": max_drawdown(mix_nav),
            "mean_period_return": sum(mix_returns) / len(mix_returns),
        },
        "hmm_point_in_time": {
            "final_nav": hmm[-1]["nav"],
            "return": hmm[-1]["nav"] - 1.0,
            "max_drawdown": max_drawdown([x["nav"] for x in hmm]),
            "mean_period_return": sum(hmm_returns) / len(hmm_returns),
        },
        "hmm_minus_static_mix": {
            "final_nav_delta": hmm[-1]["nav"] - mix_nav[-1],
            "mean_period_return_delta": sum(delta) / len(delta),
            "positive_period_share": sum(x > 0 for x in delta) / len(delta),
            "negative_period_share": sum(x < 0 for x in delta) / len(delta),
            "best_period_delta": max(delta),
            "worst_period_delta": min(delta),
            "ci_4_period_block": block_ci(delta, 4),
            "ci_8_period_block": block_ci(delta, 8),
        },
        "hmm_minus_fixed_rebound": {
            "final_nav_delta": hmm[-1]["nav"] - rebound[-1]["nav"],
            "mean_period_return_delta": sum(hmm_minus_rebound) / len(hmm_minus_rebound),
            "positive_period_share": sum(x > 0 for x in hmm_minus_rebound) / len(hmm_minus_rebound),
            "ci_4_period_block": block_ci(hmm_minus_rebound, 4),
            "ci_8_period_block": block_ci(hmm_minus_rebound, 8),
        },
        "periods": [
            {
                "date": hmm[i]["date"],
                "static_mix_nav": mix_nav[i],
                "hmm_nav": hmm[i]["nav"],
                "hmm_minus_static_mix_return": None if i == 0 else delta[i - 1],
            }
            for i in range(len(hmm))
        ],
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "periods"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
