import ast
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/experiments/repair_group_research_20261025"
OUT.mkdir(parents=True, exist_ok=True)
old = ROOT / "data/qlib_data_prewrite_20261025"
new = ROOT / "data/qlib_data"
reference = ROOT / "mlruns/656749665990832865/5501ed88f12145f0a454d68d92dbad76"
keys = ast.literal_eval((reference / "params/dataset.kwargs.handler.kwargs.factor_keys").read_text(encoding="utf-8"))
from core.adapters.qlib.factor_resolver import resolve_factor_expression

factors = {}
for key in keys:
    expression, reason = resolve_factor_expression(key)
    if not expression:
        raise RuntimeError(f"因子解析失败 {key}: {reason}")
    factors[key] = {"expression": expression, "bin_fields": sorted(set(re.findall(r"\$([a-zA-Z_][a-zA-Z_0-9]*)", expression))), "uses_fund_cache": "FundVal(" in expression}
(OUT / "factors.json").write_text(json.dumps(factors, ensure_ascii=False, indent=2), encoding="utf-8")

old_calendar = (old / "calendars/day.txt").read_bytes()
new_calendar = (new / "calendars/day.txt").read_bytes()
if old_calendar != new_calendar:
    raise RuntimeError("旧新日历不同，必须按日期对齐后再比较")
calendar = old_calendar.decode("utf-8").splitlines()
window = np.asarray(["2019-01-01" <= date <= "2026-10-23" for date in calendar])
old_files = {p.relative_to(old / "features").as_posix(): p for p in (old / "features").glob("*/*.bin")}
new_files = {p.relative_to(new / "features").as_posix(): p for p in (new / "features").glob("*/*.bin")}
totals = defaultdict(lambda: defaultdict(int))
changes = []
fingerprints = {"old": hashlib.sha256(old_calendar), "new": hashlib.sha256(new_calendar)}
for position, name in enumerate(sorted(set(old_files) | set(new_files))):
    content = {}
    for generation, files in (("old", old_files), ("new", new_files)):
        content[generation] = files[name].read_bytes() if name in files else b""
        fingerprints[generation].update(name.encode("utf-8") + hashlib.sha256(content[generation]).digest())
    field = name.split("/")[1].split(".")[0]
    totals[field]["files"] += 1
    if content["old"] == content["new"]:
        continue
    row = {"instrument": name.split("/")[0], "field": field}
    aligned = {}
    for generation in ("old", "new"):
        values = np.frombuffer(content[generation], dtype="<f4")
        arr = np.full(len(calendar), np.nan, dtype=np.float32)
        if len(values):
            start = int(values[0])
            if start < 0 or start + len(values) - 1 > len(calendar):
                raise RuntimeError(f"非法bin范围: {generation}/{name}")
            arr[start:start + len(values) - 1] = values[1:]
        aligned[generation] = arr[window]
        row[f"{generation}_finite"] = int(np.isfinite(arr[window]).sum())
    before, after = aligned["old"], aligned["new"]
    both = np.isfinite(before) & np.isfinite(after)
    row["changed_values"] = int((both & (before != after)).sum())
    row["lost_values"] = int((np.isfinite(before) & ~np.isfinite(after)).sum())
    row["gained_values"] = int((~np.isfinite(before) & np.isfinite(after)).sum())
    for key, value in row.items():
        if isinstance(value, int):
            totals[field][key] += value
    totals[field]["changed_files"] += 1
    changes.append(row)
    if position and position % 10000 == 0:
        print(f"audited {position} bin files", flush=True)
with (OUT / "bin_changes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=["instrument", "field", "old_finite", "new_finite", "changed_values", "lost_values", "gained_values"])
    writer.writeheader()
    writer.writerows(changes)

from backend.service.historical_quality import get_historical_preflight
preflight = get_historical_preflight(end="2026-10-23")
(OUT / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
cache_fingerprints = {}
for generation, directory in (("old", ROOT / "output/fund_cache_prewrite_20261025"), ("new", ROOT / "data/fund_cache")):
    path = directory / "fund_features.parquet"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    cache_fingerprints[generation] = {"directory": str(directory), "sha256": digest.hexdigest(), "size": path.stat().st_size}
changed_fields = {field for field, counts in totals.items() if counts["changed_files"]}
summary = {
    "scope": "2026-10-25 Qlib and fundamental-cache generation change only; not the full October historical quote repair",
    "research_only": True,
    "strict_pit_verified": False,
    "calendar_identical": True,
    "calendar_end": calendar[-1],
    "provider_fingerprints": {key: value.hexdigest() for key, value in fingerprints.items()},
    "cache_fingerprints": cache_fingerprints,
    "bin_field_changes": dict(totals),
    "affected_bin_factors": [key for key, value in factors.items() if changed_fields.intersection(value["bin_fields"])],
    "external_cache_factors": [key for key, value in factors.items() if value["uses_fund_cache"]],
    "full_repair_before_snapshot_available": False,
    "formal_training_ready": preflight["quality"]["training_ready"],
}
(OUT / "input_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False), flush=True)
