import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


root = Path(__file__).resolve().parent.parent
source = root / "data/experiments/expert_router_oos_h10_matured.json"
raw = source.read_bytes()
data = json.loads(raw)
rows = []
for window, result in [*data["windows"].items(), ("final", data["final_oos"])]:
    for mode, router in result["router"].items():
        for record in router["daily"]:
            rows.append({"window": window, "mode": mode, **record})

frame = pd.DataFrame(rows)
assert not frame.duplicated(["window", "date", "mode"]).any()
assert frame.groupby(["window", "date"])["future_vol_probability"].nunique().eq(1).all()
assert np.isfinite(frame[["future_vol_probability", "trend_net", "rebound_net"]]).all().all()
frame["mid"] = frame["future_vol_probability"].between(0.36, 0.59)
frame["binary"] = np.where(
    frame["future_vol_probability"] > 0.5, frame["rebound_net"], frame["trend_net"])
frame["cash_gain"] = np.where(frame["mid"], -frame["binary"], 0.0)
by_date = frame.groupby(["window", "date"], as_index=False).agg({
    "mid": "first", "trend_net": "mean", "rebound_net": "mean",
    "binary": "mean", "cash_gain": "mean"})

report = {
    "source": str(source.relative_to(root)),
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "experiment_fingerprint": data["experiment_fingerprint"],
    "thresholds": [0.36, 0.59],
    "label_horizon": data["config"]["label_horizon"],
    "rebalance_step": data["config"]["rebalance_step"],
    "label_mode": data["config"]["label_mode"],
    "costs": data["config"]["costs"],
    "record_count": len(frame),
    "decision_date_count": int(frame["date"].nunique()),
    "mid_record_count": int(frame["mid"].sum()),
    "mid_decision_date_count": int(frame.loc[frame["mid"], "date"].nunique()),
    "comparisons": {},
    "windows": {},
    "limitations": [
        "Thresholds were selected using these data; these are exploratory statistics.",
        "Date aggregation removes same-date pseudo-replication, not serial dependence or selection bias.",
        "Expert net returns retain their original holdings and costs; policy switching and cash liquidation are not replayed.",
        "This is a return substitution diagnostic, not an executable portfolio backtest.",
        "The final OOS lock selects the expert mode, not an independently validated mid-volatility cash rule.",
    ],
}
for name, sample in [
    ("pooled_records", frame), ("equal_mode_mean_by_date", by_date),
    *[(f"mode_{mode}", part) for mode, part in frame.groupby("mode")],
]:
    gain = sample["cash_gain"].to_numpy()
    middle = sample.loc[sample["mid"]]
    interval = stats.t.interval(
        0.95, len(gain) - 1, loc=gain.mean(), scale=stats.sem(gain))
    report["comparisons"][name] = {
        "n": len(sample), "mid_n": len(middle),
        "mean_gain_pp": float(gain.mean() * 100),
        "two_sided_t_p": float(stats.ttest_1samp(gain, 0.0).pvalue),
        "nominal_95pct_ci_pp": [float(v * 100) for v in interval],
        "mid_mean_return_pct": {
            key: float(middle[key].mean() * 100)
            for key in ("trend_net", "rebound_net", "binary")},
    }
for window, sample in by_date.groupby("window"):
    middle = sample.loc[sample["mid"]]
    report["windows"][window] = {
        "n": len(sample), "mid_n": len(middle),
        "mean_gain_pp": float(sample["cash_gain"].mean() * 100),
        "mid_mean_return_pct": {
            key: float(middle[key].mean() * 100) if len(middle) else None
            for key in ("trend_net", "rebound_net", "binary")},
    }
report["final_oos_lock"] = data["final_oos_lock"]
target = root / "data/experiments/mid_vol_cash_audit_20260908.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
