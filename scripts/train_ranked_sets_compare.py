"""
训练四套因子集 (uniform/icir/split/gain) 对比, 配置对齐 uniform48+Optuna 那次。

四套都用相同的 Optuna 超参 (复制 trend_momentum.json), 唯一变量是因子集排序方式。
输出: 每套的 ic_mean / ric_mean / icir 对比表。

用法: python scripts/train_ranked_sets_compare.py
"""
import time
import os
import shutil
import json
import argparse

import requests

BASE = "http://localhost:8001"
SETS = ["trend_momentum", "trend_momentum_icir", "trend_momentum_split", "trend_momentum_gain"]

PAYLOAD = {
    "train_dual": True,
    "param_profile": "optuna",
    "instruments": "a500",
    "train_start": "2022-01-01",
    "train_end": "2024-12-31",
    "valid_start": "2025-01-01",
    "valid_end": "2025-06-30",
    "test_start": "2025-07-01",
    "test_end": "2026-06-30",
    "label_horizon": 10,
    "label_norm": "none",
    "early_stop_metric": "mse",
    "auto_backtest": False,
    "topk": 50,
    "n_drop": 5,
}


def _load_env():
    env = {}
    with open(".env", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _get_token(env):
    r = requests.post(f"{BASE}/api/auth/login", json={
        "username": env.get("LOGIN_USERNAME", "admin"),
        "password": env.get("LOGIN_PASSWORD", ""),
    }, timeout=10)
    r.raise_for_status()
    return r.json()["token"]


def _copy_optuna_params():
    src = "data/optuna_params/trend_momentum.json"
    if not os.path.exists(src):
        print(f"[OPTUNA] 无 {src}, 四套各自回退 conservative (超参不一致!)")
        return
    for name in SETS[1:]:
        dst = f"data/optuna_params/{name}.json"
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"[OPTUNA] 复制 {src} -> {dst}")


def _start_train(token, factor_set):
    p = dict(PAYLOAD)
    p["use_curated"] = factor_set
    r = requests.post(f"{BASE}/api/qlib/models/train", json=p,
                      headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code != 200:
        print(f"[TRAIN] 启动失败 {factor_set}: {r.status_code} {r.text[:200]}")
        return None
    return r.json()["data"]["job_id"]


def _poll(token, job_id, label, max_h=3):
    t0 = time.time()
    while time.time() - t0 < max_h * 3600:
        try:
            r = requests.get(f"{BASE}/api/qlib/models/train/{job_id}",
                             headers={"Authorization": f"Bearer {token}"}, timeout=10)
            d = r.json()["data"]
            s = d.get("status", "")
        except Exception:
            s = "error"
        if s == "completed":
            res = d.get("result", {})
            return {
                "ic_mean": res.get("ic_mean"),
                "ric_mean": res.get("ric_mean"),
                "icir": res.get("icir"),
                "lgb_ic": res.get("lgb_ic_mean"),
                "xgb_ic": res.get("xgb_ic_mean"),
            }
        if s in ("failed", "error"):
            print(f"[POLL] {label} 失败: {d.get('error', s)}")
            return None
        time.sleep(30)
    print(f"[POLL] {label} 超时")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS), help="逗号分隔的因子集名, 默认全部")
    args = ap.parse_args()

    env = _load_env()
    token = _get_token(env)
    _copy_optuna_params()

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]

    results = {}
    for fs in sets:
        print(f"\n=== 训练 {fs} ===", flush=True)
        jid = _start_train(token, fs)
        if not jid:
            continue
        print(f"  job={jid[:8]} 训练中...", flush=True)
        r = _poll(token, jid, fs)
        results[fs] = r
        if r:
            print(f"  完成: ic_mean={r['ic_mean']}, ric_mean={r['ric_mean']}, icir={r['icir']}", flush=True)

    print("\n" + "=" * 70)
    print(f"{'因子集':<24} {'ic_mean':>10} {'ric_mean':>10} {'icir':>8} {'lgb_ic':>8} {'xgb_ic':>8}")
    print("=" * 70)
    for fs in SETS:
        r = results.get(fs)
        if not r:
            print(f"{fs:<24} {'(失败)':>10}")
            continue
        def fmt(v):
            return f"{v*100:.3f}%" if isinstance(v, (int, float)) else "--"
        print(f"{fs:<24} {fmt(r['ic_mean']):>10} {fmt(r['ric_mean']):>10} "
              f"{r['icir'] if r['icir'] is not None else '--':>8} {fmt(r['lgb_ic']):>8} {fmt(r['xgb_ic']):>8}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/ranked_sets_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("\n已落盘 logs/ranked_sets_compare.json")


if __name__ == "__main__":
    main()
