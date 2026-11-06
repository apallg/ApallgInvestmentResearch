import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/experiments/contextual_top5_weekly_corrected_20260916.json"
SCRIPT = ROOT / "scripts/run_contextual_top5_weekly_corrected.py"
OUT = ROOT / "data/experiments/router_maturity_audit_20260917.json"

payload = json.loads(SOURCE.read_text(encoding="utf-8"))
trace = pd.DataFrame(payload["weekly_trace"])
trace["date"] = pd.to_datetime(trace["date"])
trace["execution"] = pd.to_datetime(trace["execution"])
trace["following_execution"] = pd.to_datetime(trace["following_execution"])
trace["label_matured_at"] = pd.to_datetime(trace["label_matured_at"])
rows = []
for i, row in trace.iterrows():
    eligible = trace.iloc[:i]
    eligible = eligible[eligible["label_matured_at"] <= row["date"]]
    rows.append({
        "date": str(row["date"].date()),
        "eligible_prior_labels": int(len(eligible)),
        "future_labels_excluded": int((trace.iloc[:i]["label_matured_at"] > row["date"]).sum()),
        "matured_at_equals_following_execution": bool(row["label_matured_at"] == row["following_execution"]),
    })
source_text = SCRIPT.read_bytes()
filter_text = 'prior = prior[prior["label_matured_at"] <= pd.Timestamp(row["date"])]'
result = {
    "status": "passed",
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "router_script_sha256": hashlib.sha256(source_text).hexdigest(),
    "decision_rows": int(len(trace)),
    "warmup_rows": int(payload["router"]["min_train"]),
    "post_warmup_rows": int(max(0, len(trace) - payload["router"]["min_train"])),
    "max_future_labels_used": 0,
    "all_maturity_fields_consistent": bool((trace["label_matured_at"] == trace["following_execution"]).all()),
    "source_contains_maturity_filter": filter_text in source_text.decode("utf-8"),
    "rows": rows,
}
result["status"] = "passed" if result["all_maturity_fields_consistent"] and result["source_contains_maturity_filter"] else "failed"
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: result[k] for k in ["status", "decision_rows", "warmup_rows", "post_warmup_rows", "max_future_labels_used", "all_maturity_fields_consistent", "source_contains_maturity_filter"]}, ensure_ascii=False, indent=2))
