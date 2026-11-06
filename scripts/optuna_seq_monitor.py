"""
接续 Optuna 搜索: 监控当前 running job, 完成后顺序跑剩余因子集, 每完成一个清理 Optuna 中间产物.
"""
import time, sys, requests, json, shutil
from pathlib import Path
from datetime import datetime

BASE = "http://localhost:8001"
CURRENT_JOB = sys.argv[1] if len(sys.argv) > 1 else None  # 正在跑的 job_id
REMAINING = sys.argv[2:] if len(sys.argv) > 2 else ["oversold_rebound", "volume_breakout", "value_defense"]

# 只清理 V2 的 mlruns（V1 的不用管，.env 已修正）
MLRUNS_DIR = Path("C:/Users/zhj/Desktop/AShareSignalAdvisor/Documents/Apallg投研V2/mlruns")

PAYLOAD = {
    "search_dual": True, "instruments": "csi300+csi500",
    "train_start": "2019-01-01", "train_end": "2025-12-31",
    "valid_start": "2026-01-01", "valid_end": "2026-06-30",
    "test_start": "2026-07-01", "test_end": "2026-06-30",
    "label_horizon": 5, "label_norm": "none", "n_trials": 50, "timeout": 0,
}

def ts():
    return datetime.now().strftime("%H:%M:%S")

def api_poll(job_id):
    try:
        r = requests.get(f"{BASE}/api/qlib/models/train/{job_id}", timeout=10)
        d = r.json()["data"]
        return d["status"], d.get("phase",""), d.get("message","")[:60]
    except:
        return "error", "", ""

def api_start(factor_set):
    p = dict(PAYLOAD)
    p["use_curated"] = factor_set
    r = requests.post(f"{BASE}/api/qlib/models/optuna-search", json=p, timeout=30)
    jid = r.json()["data"]["job_id"]
    print(f"[{ts()}] START {factor_set} (job={jid[:8]})")
    return jid

def cleanup():
    """只删除 Optuna 搜索产生的中间 trial 目录，保留正式训练好的模型（单 run 的 experiment）。"""
    d = MLRUNS_DIR
    if not d.exists():
        return
    try:
        deleted = 0
        kept = 0
        for exp_dir in d.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "0" or exp_dir.name == ".trash":
                continue
            runs = [rd for rd in exp_dir.iterdir() if rd.is_dir()]
            if len(runs) > 5:
                # Optuna 搜索产生的中间实验（trial 数 > 5）
                sz = sum(
                    f.stat().st_size for rd in runs
                    for f in rd.rglob("*") if f.is_file()
                ) / 1024 / 1024
                shutil.rmtree(exp_dir)
                deleted += 1
                print(f"[{ts()}] CLEANUP optuna trials: {exp_dir.name} ({len(runs)} runs, {sz:.0f}MB)")
            else:
                kept += 1
        if deleted:
            print(f"[{ts()}] CLEANUP done: {deleted} optuna exps deleted, {kept} trained models kept")
        else:
            print(f"[{ts()}] CLEANUP: no optuna trials to delete ({kept} models kept)")
    except Exception as e:
        print(f"[{ts()}] CLEANUP fail: {e}")

def wait_job(job_id, label, max_h=6):
    t0 = time.time()
    ok_since = t0
    while time.time() - t0 < max_h * 3600:
        s, p, m = api_poll(job_id)
        elapsed = int((time.time() - t0) / 60)
        print(f"[{ts()}] [{label}] {s}/{p} {m} ({elapsed}m)")
        if s == "completed":
            print(f"[{ts()}] DONE: {label}")
            return True
        if s == "failed":
            print(f"[{ts()}] FAIL: {label}: {m}")
            return False
        if s in ("running","started"):
            ok_since = time.time()
        elif s == "error" and time.time() - ok_since > 120:
            print(f"[{ts()}] WARN: {label} unresponsive >2min")
        time.sleep(60)
    print(f"[{ts()}] TIMEOUT: {label}")
    return False

# --- main ---
print(f"[{ts()}] ===== Optuna Resume: current={CURRENT_JOB}, remaining={REMAINING} =====")

if CURRENT_JOB:
    print(f"\n[{ts()}] --- trend_momentum (resume) ---")
    ok = wait_job(CURRENT_JOB, "trend_momentum")
    if not ok:
        print(f"[{ts()}] WARN: trend_momentum incomplete, continuing...")
    cleanup()

for fs in REMAINING:
    print(f"\n[{ts()}] --- {fs} ---")
    jid = api_start(fs)
    ok = wait_job(jid, fs)
    cleanup()

# summary
print(f"\n[{ts()}] ===== Summary =====")
r = requests.get(f"{BASE}/api/qlib/optuna-params", timeout=10)
for res in r.json().get("data", []):
    fs = res.get("factor_set","?")
    li, xi = res.get("lgb_best_ic"), res.get("xgb_best_ic")
    print(f"  {fs}: LGB_IC={li}, XGB_IC={xi}")
print(f"[{ts()}] Done.")
