import argparse
import json
import msvcrt
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.horizon_architecture_research import OUTPUT, write_json


def status():
    path = OUTPUT / "workflow_status.json"
    print(path.read_text(encoding="utf-8") if path.exists() else json.dumps({"status": "not_started"}))


def run(wait_pid):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT / "workflow.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        stages = [
            ("base_audit", ["scripts/audit_horizon_architecture.py"]),
            ("base_evaluate", ["scripts/analyze_horizon_architecture.py", "evaluate"]),
            ("improvement_prerun", ["scripts/horizon_architecture_improvement.py", "prerun", "--resume"]),
            ("improvement_budget", ["scripts/horizon_architecture_improvement.py", "budget"]),
            ("improvement_train", ["scripts/horizon_architecture_improvement.py", "train", "--resume"]),
            ("improvement_evaluate", ["scripts/analyze_horizon_improvement.py"]),
        ]
        try:
            if wait_pid:
                write_json(OUTPUT / "workflow_status.json", {"status": "waiting_base_training", "pid": wait_pid})
                while psutil.pid_exists(wait_pid):
                    time.sleep(30)
            base = len(list(OUTPUT.glob("W*_H*_*.parquet")))
            if base != 432:
                error = OUTPUT / "base_stderr.log"
                tail = error.read_text(encoding="utf-8", errors="replace")[-4000:] if error.exists() else ""
                raise RuntimeError(f"基础矩阵只有{base}/432份，停止后续阶段。错误尾部: {tail}")
            for name, arguments in stages:
                disk = {drive: round(shutil.disk_usage(drive).free / 1024 ** 3, 2) for drive in ("C:/", "G:/")}
                if disk["C:/"] < 10 or disk["G:/"] < 10:
                    raise RuntimeError(f"磁盘保留空间不足: {disk}")
                write_json(OUTPUT / "workflow_status.json", {"status": "running", "stage": name, "free_gib": disk})
                with (OUTPUT / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (OUTPUT / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
                    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=stdout, stderr=stderr)
                if result.returncode:
                    detail = (OUTPUT / f"{name}.stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
                    raise RuntimeError(f"{name}退出码{result.returncode}: {detail}")
                if name in ("base_evaluate", "improvement_evaluate"):
                    result_path = OUTPUT / ("analysis.json" if name == "base_evaluate" else "improvement_analysis.json")
                    analysis = json.loads(result_path.read_text(encoding="utf-8"))
                    if analysis.get("status") != "evaluated" or analysis.get("completed") != analysis.get("required"):
                        raise RuntimeError(f"{name}矩阵仍不完整")
            write_json(OUTPUT / "workflow_status.json", {"status": "research_matrices_evaluated", "base_predictions": 432,
                                                        "improvement_predictions": 360, "production_model_unchanged": True})
        except Exception as error:
            write_json(OUTPUT / "workflow_status.json", {"status": "failed", "error": str(error),
                                                        "production_model_unchanged": True})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "status"))
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if args.stage == "status":
        status()
    else:
        run(args.wait_pid)
