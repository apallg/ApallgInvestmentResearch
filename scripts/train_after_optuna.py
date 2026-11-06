"""
Optuna搜索全部完成后，依次训练三个因子集的模型（双模型LGB+XGB融合）。
"""
import time, requests, sys
from datetime import datetime

BASE = "http://localhost:8001"
FACTOR_SETS = sys.argv[1:] if len(sys.argv) > 1 else [
    "trend_momentum", "oversold_rebound", "volume_breakout"
]

PAYLOAD = {
    "train_dual": True,
    "param_profile": "optuna",
    "instruments": "csi300+csi500",
    "train_start": "2019-01-01",
    "train_end": "2025-12-31",
    "valid_start": "2026-01-01",
    "valid_end": "2026-06-30",
    "test_start": "2026-07-01",
    "test_end": "2026-08-31",
    "label_horizon": 5,
    "auto_backtest": True,
    "topk": 50,
    "n_drop": 5,
}

def ts():
    return datetime.now().strftime("%H:%M:%S")

def start_train(factor_set):
    p = dict(PAYLOAD)
    p["use_curated"] = factor_set
    r = requests.post(f"{BASE}/api/qlib/models/train", json=p, timeout=30)
    jid = r.json()["data"]["job_id"]
    print(f"[{ts()}] TRAIN START {factor_set} (job={jid[:8]})")
    return jid

def poll_train(job_id, label, max_h=3):
    t0 = time.time()
    while time.time() - t0 < max_h * 3600:
        try:
            r = requests.get(f"{BASE}/api/qlib/models/train/{job_id}", timeout=10)
            d = r.json()["data"]
            s, msg = d.get("status",""), d.get("message","")[:80]
        except:
            s, msg = "error", ""
        elapsed = int((time.time() - t0) / 60)
        print(f"[{ts()}] [{label}] {s} {msg} ({elapsed}m)")
        if s == "completed":
            result = d.get("result", {})
            ic = result.get("ic_mean", "?")
            lgb_ic = result.get("lgb_ic_mean", "?")
            xgb_ic = result.get("xgb_ic_mean", "?")
            print(f"[{ts()}] TRAIN DONE {label}: IC={ic}, LGB_IC={lgb_ic}, XGB_IC={xgb_ic}")
            return True
        if s == "failed" or s == "error":
            print(f"[{ts()}] TRAIN FAIL {label}: {d.get('error', msg)}")
            return False
        time.sleep(30)
    print(f"[{ts()}] TRAIN TIMEOUT {label}")
    return False

print(f"[{ts()}] ===== Train After Optuna: {FACTOR_SETS} =====")

for fs in FACTOR_SETS:
    print(f"\n[{ts()}] --- Training {fs} ---")
    jid = start_train(fs)
    ok = poll_train(jid, fs)
    if not ok:
        print(f"[{ts()}] WARN: {fs} training failed, continuing...")

print(f"\n[{ts()}] ===== All training complete =====")
print(f"[{ts()}] Summary:")
for fs in FACTOR_SETS:
    r = requests.get(f"{BASE}/api/qlib/optuna-params", timeout=10)
    for res in r.json().get("data", []):
        if res.get("factor_set") == fs:
            print(f"  {fs}: LGB_IC={res.get('lgb_best_ic')}, XGB_IC={res.get('xgb_best_ic')}")
print(f"[{ts()}] Done.")
