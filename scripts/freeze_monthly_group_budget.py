import hashlib
import json
import math
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/experiments/monthly_group_research_20261103"
    output = Path(os.environ.get("MONTHLY_GROUP_OUTPUT", root / "data/experiments/monthly_group_research_v4_20261103"))
    records = [json.loads((source / f"{fold}_42_training.json").read_text(encoding="utf-8")) for fold in ("W1", "W8")]
    counts = {}
    timings = {}
    for suffix, unified in (("0", "U0"), ("M", "UM")):
        for name in ("SL4", "PB4", "IC6", "GS8"):
            arm = f"{name}_{suffix}"
            ratios = []
            elapsed = []
            for record in records:
                base = float(record[unified]["elapsed_seconds"])
                expert = sum(float(info.get("elapsed_seconds", 0)) for info in record[arm]["experts"].values())
                ratios.append((base + expert) / base)
                elapsed.append({"unified_seconds": base, "expert_seconds": expert})
            counts[arm] = max(1, math.floor(min(ratios)))
            timings[arm] = elapsed
    payload = {"status": "frozen_before_formal_outer_analysis", "pre_run_folds": ["W1", "W8"],
               "pre_run_seed": 42, "ensemble_member_count": counts, "timing_evidence": timings,
               "rule": "floor(min pre-run total-training-time / unified-training-time); no over-budget members"}
    path = output / "budget_control.json"
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError("预算对照已冻结且内容不同")
    path.write_bytes(encoded)
    print(json.dumps({"budget_sha256": hashlib.sha256(encoded).hexdigest(), "members": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
