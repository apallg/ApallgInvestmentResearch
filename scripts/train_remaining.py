"""续跑剩余三套训练: 等 icir(已启动), 依次启动 split/gain, 结果写文件。

进度写 logs/train_remaining.log, 结果写 logs/ranked_sets_compare.json。
"""
import time
import os
import json

import requests

BASE = "http://localhost:8001"
ICIR_JOB = "05df2a98"
TO_START = ["trend_momentum_split", "trend_momentum_gain"]

PAYLOAD = {
    "train_dual": True, "param_profile": "optuna", "instruments": "a500",
    "train_start": "2022-01-01", "train_end": "2024-12-31",
    "valid_start": "2025-01-01", "valid_end": "2025-06-30",
    "test_start": "2025-07-01", "test_end": "2026-06-30",
    "label_horizon": 10, "label_norm": "none", "early_stop_metric": "mse",
    "auto_backtest": False, "topk": 50, "n_drop": 5,
}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with open("logs/train_remaining.log", "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


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


def job_status(token, jid):
    r = requests.get(f"{BASE}/api/qlib/models/train/{jid}",
                     headers={"Authorization": f"Bearer {token}"}, timeout=10)
    return r.json()["data"].get("status")


def start(token, fs):
    p = dict(PAYLOAD)
    p["use_curated"] = fs
    r = requests.post(f"{BASE}/api/qlib/models/train", json=p,
                      headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code != 200:
        log(f"启动失败 {fs}: {r.status_code} {r.text[:200]}")
        return None
    return r.json()["data"]["job_id"]


def wait_done(token, jid, label):
    while True:
        s = job_status(token, jid)
        if s == "completed":
            log(f"{label} 完成")
            return True
        if s in ("failed", "error"):
            log(f"{label} 失败: {s}")
            return False
        time.sleep(30)


def fetch_results():
    from core.data import Database
    rows = Database.fetchall(
        "SELECT run_id, factor_set, ic_mean, ric_mean, icir, "
        "lgb_ic_mean, xgb_ic_mean, custom_factor_keys, created_at "
        "FROM training_results WHERE created_at >= '2026-08-16 19:19:00' "
        "ORDER BY created_at")
    return rows


def main():
    token = get_token()
    log("=== 续跑开始 ===")

    # 1. 等 icir (已启动)
    log(f"等待 icir job {ICIR_JOB}")
    wait_done(token, ICIR_JOB, "icir")

    # 2. 依次 split / gain
    for fs in TO_START:
        jid = start(token, fs)
        if jid:
            log(f"已启动 {fs} (job={jid[:8]})")
            wait_done(token, jid, fs)

    # 3. 汇总结果写 json
    log("=== 训练完成, 汇总结果 ===")
    rows = fetch_results()
    by_cfk = {}
    for r in rows:
        keys = r["custom_factor_keys"]
        if not keys:
            continue
        try:
            keys = json.loads(keys) if isinstance(keys, str) else keys
        except Exception:
            continue
        # 按因子集识别: 用 cfk 长度 + 首因子
        head = keys[0] if keys else ""
        n = len(keys)
        label = "?"
        if n == 48 and head in ("MOM", "IMOM", "IMom"):
            label = "uniform"
        elif head in ("a158_roc20", "a191_mom_26", "a191_mom_21", "a191_mom_19"):
            label = "icir/split/gain(待细判)"
        by_cfk.setdefault(label, r)

    out = []
    for r in rows:
        if r["factor_set"] != "WhiteBoxHandler":
            continue
        out.append({
            "created_at": str(r["created_at"]),
            "ic_mean": r["ic_mean"], "ric_mean": r["ric_mean"], "icir": r["icir"],
            "lgb_ic": r["lgb_ic_mean"], "xgb_ic": r["xgb_ic_mean"],
            "cfk_len": len(json.loads(r["custom_factor_keys"])) if r["custom_factor_keys"] else None,
        })
    with open("logs/ranked_sets_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"结果已写 logs/ranked_sets_compare.json ({len(out)} 条 WhiteBoxHandler)")


if __name__ == "__main__":
    main()
