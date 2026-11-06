import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/repair_group_research_20261025"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
(OUT / "workflow_pid.json").write_text(json.dumps({"pid": os.getpid(), "research_only": True}), encoding="utf-8")
try:
    for stage in ("run_repair_group_research", "analyze_repair_group_research"):
        with (OUT / f"{stage}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(ROOT / f"scripts/{stage}.py")], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"{stage} failed; see {OUT / (stage + '.log')}")
    (OUT / "status.json").write_text(json.dumps({"state": "complete", "phase": "analysis_complete", "research_only": True, "deployment_eligible": False, "report": str(OUT / "report.md")}, indent=2), encoding="utf-8")
except Exception:
    reason = traceback.format_exc()
    (OUT / "status.json").write_text(json.dumps({"state": "failed", "research_only": True, "reason": reason}, indent=2), encoding="utf-8")
    (OUT / "failure.txt").write_text(reason, encoding="utf-8")
    raise
