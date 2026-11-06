import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "experiments" / "rank_persistence_winrate_expansion_20260926"
VARIANTS = ("A_baseline", "B_stability_first", "C_strict_confirm")
DRAW_COUNT = 20000
SEED = 20260926


def summarize(segments):
    a = np.concatenate([segment[0] for segment in segments])
    b = np.concatenate([segment[1] for segment in segments])
    difference = b - a
    wins = int((difference > 0).sum())
    losses = int((difference < 0).sum())
    b_only = int(((b > 0) & (a <= 0)).sum())
    a_only = int(((a > 0) & (b <= 0)).sum())
    non_ties = wins + losses
    discordant = b_only + a_only
    result = {
        "weeks": len(a),
        "a_profitable_weeks": int((a > 0).sum()),
        "b_profitable_weeks": int((b > 0).sum()),
        "a_profit_win_rate": float((a > 0).mean()),
        "b_profit_win_rate": float((b > 0).mean()),
        "profit_win_rate_difference_pp": float(((b > 0).mean() - (a > 0).mean()) * 100),
        "b_only_profitable_weeks": b_only,
        "a_only_profitable_weeks": a_only,
        "b_beats_a_weeks": wins,
        "b_loses_to_a_weeks": losses,
        "ties": len(a) - non_ties,
        "b_beats_a_share_all_weeks": wins / len(a),
        "b_beats_a_share_non_ties": wins / non_ties if non_ties else None,
        "mean_net_return_difference_pp": float(difference.mean() * 100),
        "iid_sign_test_p_greater": (
            sum(math.comb(non_ties, k) for k in range(wins, non_ties + 1)) / 2 ** non_ties
            if non_ties else None
        ),
        "iid_paired_profit_test_p_greater": (
            sum(math.comb(discordant, k) for k in range(b_only, discordant + 1)) / 2 ** discordant
            if discordant else None
        ),
        "bootstrap": {},
    }
    for block_size in (4, 8, 13):
        rng = np.random.default_rng(SEED)
        sample_a = []
        sample_b = []
        for segment_a, segment_b in segments:
            length = len(segment_a)
            starts = rng.integers(0, length, size=(DRAW_COUNT, math.ceil(length / block_size)))
            indices = ((starts[:, :, None] + np.arange(block_size)) % length).reshape(DRAW_COUNT, -1)[:, :length]
            sample_a.append(segment_a[indices])
            sample_b.append(segment_b[indices])
        sampled_a = np.concatenate(sample_a, axis=1)
        sampled_b = np.concatenate(sample_b, axis=1)
        sampled_difference = sampled_b - sampled_a
        sampled_profit_difference = (sampled_b > 0).mean(axis=1) - (sampled_a > 0).mean(axis=1)
        sampled_wins = (sampled_difference > 0).sum(axis=1)
        sampled_non_ties = (sampled_difference != 0).sum(axis=1)
        active = sampled_non_ties > 0
        result["bootstrap"][str(block_size)] = {
            "block_weeks": block_size,
            "draws": DRAW_COUNT,
            "seed": SEED,
            "segment_boundaries_preserved": True,
            "profit_win_rate_difference_ci95_pp": (np.quantile(sampled_profit_difference, [0.025, 0.975]) * 100).tolist(),
            "non_tie_outperformance_share_ci95": (
                np.quantile(sampled_wins[active] / sampled_non_ties[active], [0.025, 0.975]).tolist()
                if active.any() else None
            ),
            "mean_net_return_difference_ci95_pp": (np.quantile(sampled_difference.mean(axis=1), [0.025, 0.975]) * 100).tolist(),
        }
    result["profit_win_rate_evidence_pass"] = all(
        row["profit_win_rate_difference_ci95_pp"][0] > 0
        for row in result["bootstrap"].values()
    )
    result["outperformance_share_evidence_pass"] = all(
        row["non_tie_outperformance_share_ci95"] is not None
        and row["non_tie_outperformance_share_ci95"][0] > 0.5
        for row in result["bootstrap"].values()
    )
    return result


def main():
    reports = {}
    manifest = {}
    weekly_rows = []
    seen_dates = set()
    baseline_contract = None
    baseline_factors = None
    for segment in ("earlier", "recent"):
        path = OUTPUT / segment / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        contract = {key: value for key, value in report["contracts"].items() if key not in ("start", "end", "decision_end")}
        if baseline_contract is not None and contract != baseline_contract:
            raise ValueError("跨批回放策略合同不一致")
        baseline_contract = contract
        prediction_path = Path(report["model"]["prediction_path"])
        prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if prediction_hash != report["model"]["prediction_sha256"]:
            raise ValueError("预测制品发生变化")
        factor_content = pickle.loads((prediction_path.parent / "factor_keys.json").read_bytes())
        factors = json.loads(factor_content)["custom_factor_keys"]
        if baseline_factors is not None and factors != baseline_factors:
            raise ValueError("跨批因子表达式键顺序不一致")
        baseline_factors = factors
        returns = {}
        for variant in VARIANTS:
            periods = report["variants"][variant]["periods"]
            returns[variant] = {str(row["date"])[:10]: float(row["period_return"]) for row in periods}
            if len(returns[variant]) != len(periods):
                raise ValueError("重复的收益日期")
        dates = sorted(returns[VARIANTS[0]])
        if any(set(returns[variant]) != set(dates) for variant in VARIANTS):
            raise ValueError("A/B/C配对日期不一致")
        if seen_dates.intersection(dates) or dates[-1] > "2026-09-24":
            raise ValueError("跨批日期重复或超过数据上限")
        seen_dates.update(dates)
        arrays = {variant: np.array([returns[variant][day] for day in dates]) for variant in VARIANTS}
        reports[segment] = {
            "dates": dates,
            "arrays": arrays,
            "model": report["model"],
            "summaries": {variant: report["variants"][variant]["summary"] for variant in VARIANTS},
            "comparisons": {variant: summarize([(arrays[VARIANTS[0]], arrays[variant])]) for variant in VARIANTS[1:]},
            "b_c_returns_equal": bool(np.array_equal(arrays[VARIANTS[1]], arrays[VARIANTS[2]])),
        }
        for day in dates:
            weekly_rows.append({"segment": segment, "period_date_label": day, **{variant: returns[variant][day] for variant in VARIANTS}})
        params = prediction_path.parent.parent / "params"
        manifest[segment] = {
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prediction_sha256": prediction_hash,
            "factor_count": len(factors),
            "model": report["model"],
            "contracts": report["contracts"],
            "nominal_train": (params / "dataset.kwargs.segments.train").read_text(),
            "nominal_valid": (params / "dataset.kwargs.segments.valid").read_text(),
            "param_profile": (params / "param_profile").read_text(),
            "data_warnings": report["data_warnings"],
        }
    aggregate = {
        variant: summarize([(row["arrays"][VARIANTS[0]], row["arrays"][variant]) for row in reports.values()])
        for variant in VARIANTS[1:]
    }
    for row in reports.values():
        del row["arrays"]
    code_paths = [
        "scripts/run_rank_persistence_ab.py", "scripts/analyze_rank_persistence_winrate_expansion.py",
        "core/domain/backtest/engine.py", "core/domain/backtest/common.py",
        "core/domain/backtest/metrics.py", "core/domain/portfolio/allocator.py",
        "core/domain/screening/exclusion.py", "docs/rank_persistence_winrate_expansion_protocol_20260926.md",
    ]
    result = {
        "status": "exploratory_archived_oos_winrate_expansion",
        "data_cutoff": "2026-09-24",
        "interpretation": "按档案测试时段扩展；已查看过历史，数据和执行审计不足，不能认定为独立确认或完整16批协议。",
        "profit_definition": "reported portfolio period_return > 0",
        "pair_unit": "portfolio weekly period, not stock slot",
        "iid_tests_role": "diagnostic only; adjacent weekly dependence is handled with segmented block bootstrap",
        "aggregate": aggregate,
        "segments": reports,
        "manifest": manifest,
        "code_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in code_paths},
    }
    (OUTPUT / "winrate_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(weekly_rows).to_csv(OUTPUT / "weekly_pairs.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"aggregate": aggregate, "segments": {key: row["comparisons"] for key, row in reports.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
