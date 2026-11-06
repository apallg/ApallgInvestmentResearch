import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "docs" / "business_audit_20261026"

with (AUDIT_DIR / "progress.json").open("r", encoding="utf-8-sig") as stream:
    progress = json.load(stream)
continuity = progress["current_continuity_counts"]

with (AUDIT_DIR / "issues.tsv").open("r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))

core_rows = [row for row in rows if not row["id"].startswith(("VR", "FX"))]
auxiliary_rows = [row for row in rows if row not in core_rows]
state_counts = Counter(row.get("continuity_state", "") for row in core_rows)
auxiliary_counts = Counter(row.get("continuity_state", "") for row in auxiliary_rows)
unresolved = [row for row in core_rows if row.get("continuity_state") in ("partial", "unverified")]
priority_counts = Counter(row["priority"] for row in unresolved)
package_counts = Counter(row["package"] for row in unresolved)

expected = {
    "rows": progress["governance_items"],
    "core": continuity["core"],
    "local_verified": continuity["local_verified"],
    "partial": continuity["partial"],
    "unverified": continuity["unverified"],
    "unresolved": continuity["unresolved"],
    "priority": dict(sorted(continuity["priority"].items())),
    "package": dict(sorted(continuity["packages"].items())),
    "auxiliary": dict(sorted(progress["auxiliary_continuity_counts"].items())),
}
actual = {
    "rows": len(rows),
    "core": len(core_rows),
    "local_verified": state_counts["local_verified"],
    "partial": state_counts["partial"],
    "unverified": state_counts["unverified"],
    "unresolved": len(unresolved),
    "priority": dict(sorted(priority_counts.items())),
    "package": dict(sorted(package_counts.items())),
    "auxiliary": dict(sorted(auxiliary_counts.items())),
}
if actual != expected:
    raise SystemExit(f"issues.tsv连续状态统计不符: expected={expected}; actual={actual}")

with (AUDIT_DIR / "issues.json").open("r", encoding="utf-8-sig") as stream:
    payload = json.load(stream)
json_states = {item["id"]: item.get("continuity_state") for item in payload["items"]}
tsv_states = {row["id"]: row.get("continuity_state") for row in rows}
if json_states != tsv_states:
    raise SystemExit("issues.json 与 issues.tsv 的 continuity_state 不一致")
json_workflows = {item["id"]: item.get("workflow_status", "") for item in payload["items"]}
tsv_workflows = {row["id"]: row.get("workflow_status", "") for row in rows}
if json_workflows != tsv_workflows:
    raise SystemExit("issues.json 与 issues.tsv 的 workflow_status 不一致")
if payload.get("stats", {}).get("continuity_state") != {
    "core_issues": continuity["core"],
    "local_verified": continuity["local_verified"],
    "partial": continuity["partial"],
    "unverified": continuity["unverified"],
    "unresolved": continuity["unresolved"],
    "priority_unresolved": continuity["priority"],
    "package_unresolved": continuity["packages"],
}:
    raise SystemExit("issues.json连续状态汇总与逐ID状态不一致")

print(json.dumps(actual, ensure_ascii=False, sort_keys=True))
