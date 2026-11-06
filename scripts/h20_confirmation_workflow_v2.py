import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.shared import config
from scripts.h20_confirmation_v2 import OUTPUT, save_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of")
    args = parser.parse_args(argv)
    lock_path = OUTPUT / "workflow.lock"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError("H20 v2 已有一个训练或记录工作流实例")
        try:
            run(args.as_of)
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def run(as_of_arg):
    if shutil.disk_usage("C:/").free < 10 * 1024 ** 3 or shutil.disk_usage("G:/").free < 10 * 1024 ** 3:
        raise RuntimeError("C盘或G盘剩余空间低于10GiB，停止新增研究任务")
    local_now = datetime.now(ZoneInfo("Asia/Shanghai"))
    calendar = pd.DatetimeIndex(pd.to_datetime(
        (config.QLIB_DATA_DIR / "calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    as_of = pd.Timestamp(as_of_arg) if as_of_arg else calendar[-1]
    if as_of not in calendar or as_of > pd.Timestamp(local_now.date()):
        raise RuntimeError("指定截止日不在当前生效交易日历或来自未来")
    date = str(as_of.date())
    state_path = OUTPUT / "workflow_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"tasks": {}}
    inputs = OUTPUT / "inputs/full" / date
    protocol = OUTPUT / "protocol.json"
    if not protocol.exists():
        raise RuntimeError("先冻结 H20 v2 协议")

    def execute(stage, extra=()):
        key = f"{date}:{stage}"
        prior = state["tasks"].get(key, {})
        if prior.get("status") == "complete":
            artifact = {"prepare-inputs": inputs / "manifest.json",
                        "record": OUTPUT / "records" / date / "predictions.json",
                        "retrain": OUTPUT / "models" / date / "audit.json"}[stage]
            if not artifact.exists():
                raise RuntimeError(f"工作流已标完成但产物缺失: {artifact}")
            return
        if prior.get("status") == "blocked" or prior.get("attempts", 0) >= 3:
            raise RuntimeError(f"{stage} 已停止自动重试，须审计 {state_path} 与工作流日志")
        command = [sys.executable, str(ROOT / "scripts/h20_confirmation_v2.py"), stage, *extra]
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log_dir = OUTPUT / "workflow_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{date}_{stage}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (log_dir / f"{date}_{stage}.stderr.log").write_text(result.stderr, encoding="utf-8")
        retryable = any(word in result.stderr.lower() for word in
                        ("timeout", "timed out", "connection", "temporar", "i/o error"))
        attempts = prior.get("attempts", 0) + 1
        state["tasks"][key] = {"status": "complete" if result.returncode == 0 else
                                "failed_retryable" if retryable and attempts < 3 else "blocked",
                                "attempts": attempts,
                                "started_at_utc": started,
                                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                                "exit_code": result.returncode,
                                "last_error": result.stderr[-500:] if result.returncode else None}
        save_json(state_path, state)
        if result.returncode:
            if state["tasks"][key]["status"] == "blocked":
                raise RuntimeError(f"{stage} 失败，证据保留在 {log_dir}")
            raise RuntimeError(f"{stage} 暂时失败，等待后续调度重试")

    if not (inputs / "manifest.json").exists():
        execute("prepare-inputs", ("--as-of", date))
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "qualified_for_research_training":
        raise RuntimeError("输入未取得研究训练资格")
    common = ("--panel", str(inputs / "panel.parquet"), "--denominator", str(inputs / "denominator.parquet"),
              "--calendar", str(inputs / "calendar.txt"), "--provenance", str(inputs / "manifest.json"),
              "--as-of", date)
    models = sorted((OUTPUT / "models").glob("*/audit.json")) if (OUTPUT / "models").exists() else []
    next_days = pd.DatetimeIndex(pd.to_datetime((inputs / "calendar.txt").read_text(encoding="utf-8").splitlines()))
    future = next_days[next_days > as_of]
    deadline = (datetime.combine(future[0].date(), time(9, 25), tzinfo=ZoneInfo("Asia/Shanghai"))
                if len(future) else None)
    recording_window = (local_now >= datetime.combine(as_of.date(), time(15), tzinfo=ZoneInfo("Asia/Shanghai"))
                        and deadline is not None and local_now < deadline)
    eligible_models = [path for path in models if pd.Timestamp(path.parent.name) < as_of]
    if eligible_models and recording_window:
        execute("record", common)
    if not models:
        execute("retrain", common)
    elif calendar.get_indexer([as_of])[0] - calendar.get_indexer([pd.Timestamp(models[-1].parent.name)])[0] >= 63:
        execute("retrain", common)
    print(json.dumps({"status": "workflow_checked", "as_of": date,
                      "observation_started": (OUTPUT / "observation.json").exists()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
