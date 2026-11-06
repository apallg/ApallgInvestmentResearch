"""训练四套因子集对比 (v2): custom_factor_keys 直传, 绕过 use_curated 状态加载。

v1 的问题: use_curated 依赖后端进程的 factor pool 单例缓存, 脚本独立进程保存的
trend_momentum_icir/split/gain 后端看不到 → 回退默认 Alpha158。
v2: 脚本自己读 flat_list + Optuna 参数, 通过 custom_factor_keys/lgb_params/xgb_params 直传。

用法: python scripts/train_ranked_sets_v2.py [--sets a,b,c,d]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse

import requests

BASE = "http://localhost:8001"
SETS = ["trend_momentum", "trend_momentum_icir", "trend_momentum_split", "trend_momentum_gain"]

PAYLOAD = {
    "train_dual": True,
    "factor_set": "WhiteBoxHandler",
    "param_profile": "default",
    "instruments": "a500",
    "train_start": "2022-01-01", "train_end": "2024-12-31",
    "valid_start": "2025-01-01", "valid_end": "2025-06-30",
    "test_start": "2025-07-01", "test_end": "2026-06-30",
    "label_horizon": 10, "label_norm": "none", "early_stop_metric": "mse",
    "auto_backtest": False, "topk": 50, "n_drop": 5,
}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open("logs/train_v2.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def get_token():
    env = {}
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    r = requests.post(f"{BASE}/api/auth/login", json={
        "username": env.get("LOGIN_USERNAME", "admin"),
        "password": env.get("LOGIN_PASSWORD", "")}, timeout=10)
    return r.json()["token"]


def read_factor_lists(names):
    from core.domain.factors.registry import get_pool
    pool = get_pool()
    out = {}
    for name in names:
        flat = pool.get_curated_factors(name)
        out[name] = flat
        log(f"读 {name}: {len(flat)} 因子")
    return out


def read_optuna_params():
    with open("data/optuna_params/trend_momentum.json", encoding="utf-8") as f:
        d = json.load(f)
    return d["lgb_best_params"], d["xgb_best_params"]


def start(token, factor_keys, lgb_params, xgb_params, label):
    p = dict(PAYLOAD)
    p["custom_factor_keys"] = factor_keys
    p["lgb_params"] = lgb_params
    p["xgb_params"] = xgb_params
    r = requests.post(f"{BASE}/api/qlib/models/train", json=p,
                      headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code != 200:
        log(f"启动失败 {label}: {r.status_code} {r.text[:200]}")
        return None
    return r.json()["data"]["job_id"]


def wait_done(token, jid, label):
    while True:
        try:
            r = requests.get(f"{BASE}/api/qlib/models/train/{jid}",
                             headers={"Authorization": f"Bearer {token}"}, timeout=10)
            s = r.json()["data"].get("status")
        except Exception:
            s = "error"
        if s == "completed":
            res = r.json()["data"].get("result", {})
            log(f"{label} 完成: ic_mean={res.get('ic_mean')}, ric_mean={res.get('ric_mean')}, icir={res.get('icir')}")
            return res
        if s in ("failed", "error"):
            log(f"{label} 失败: {s}")
            return None
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    args = ap.parse_args()
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]

    token = get_token()
    factor_lists = read_factor_lists(sets)
    lgb_p, xgb_p = read_optuna_params()
    log(f"Optuna 参数: lgb={list(lgb_p)[:3]}... xgb={list(xgb_p)[:3]}...")

    results = {}
    for fs in sets:
        keys = factor_lists.get(fs)
        if not keys:
            log(f"{fs}: 无因子列表, 跳过")
            continue
        log(f"=== 训练 {fs} ({len(keys)} 因子) ===")
        jid = start(token, keys, lgb_p, xgb_p, fs)
        if not jid:
            continue
        res = wait_done(token, jid, fs)
        results[fs] = res

    with open("logs/ranked_sets_compare.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"结果已写 logs/ranked_sets_compare.json")


if __name__ == "__main__":
    main()
