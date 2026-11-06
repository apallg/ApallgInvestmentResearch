"""训练「去冗余后全量因子」(表达式去重 226) 一次模型, 对照 uniform48。

流程: step1 取 trend_momentum 6 大类候选 → 表达式去重 → 过滤无法解析因子
      → 直传 custom_factor_keys + Optuna 超参训练 (与 uniform48 同配置, 唯一变量=因子数)。

用法:
  python scripts/train_dedup_226.py --dry-run     # 只打印因子列表, 不训练
  python scripts/train_dedup_226.py               # 训练
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse

import requests

BASE = "http://localhost:8001"

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
    with open("logs/train_dedup_226.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def get_candidates():
    from core.domain.factors.selector import DIRECTION_PRESETS, group_by_category, _dedupe_by_expression
    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter

    preset = DIRECTION_PRESETS["trend_momentum"]
    pool = get_pool()
    active = pool.get_active_factors()
    adapter = PublicFactorAdapter()
    category_groups = group_by_category(list(active.keys()), adapter, pool_active=active)

    candidates = []
    for cat in preset["categories"]:
        keys = category_groups.get(cat, [])
        log(f"[候选] {cat}: {len(keys)} 个")
        candidates.extend(keys)
    candidates = list(dict.fromkeys(candidates))
    log(f"[候选] 合计 {len(candidates)} 个")
    candidates = _dedupe_by_expression(candidates)
    log(f"[候选] 表达式去重后 {len(candidates)} 个")
    return candidates


def filter_resolvable(keys):
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    valid = []
    skipped = []
    for k in keys:
        expr, reason = resolve_factor_expression(k)
        if expr:
            valid.append(k)
        else:
            skipped.append((k, reason))
    if skipped:
        log(f"[过滤] {len(keys)} → {len(valid)} (跳过 {len(skipped)} 无法解析)")
        for k, r in skipped[:10]:
            log(f"  跳过 {k}: {r}")
    return valid


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


def read_optuna_params():
    with open("data/optuna_params/trend_momentum.json", encoding="utf-8") as f:
        d = json.load(f)
    return d["lgb_best_params"], d["xgb_best_params"]


def start(token, factor_keys, lgb_p, xgb_p):
    p = dict(PAYLOAD)
    p["custom_factor_keys"] = factor_keys
    p["lgb_params"] = lgb_p
    p["xgb_params"] = xgb_p
    r = requests.post(f"{BASE}/api/qlib/models/train", json=p,
                      headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code != 200:
        log(f"启动失败: {r.status_code} {r.text[:300]}")
        return None
    return r.json()["data"]["job_id"]


def wait_done(token, jid):
    while True:
        try:
            r = requests.get(f"{BASE}/api/qlib/models/train/{jid}",
                             headers={"Authorization": f"Bearer {token}"}, timeout=10)
            d = r.json()["data"]
            s = d.get("status")
        except Exception:
            s = "error"
        if s == "completed":
            res = d.get("result", {})
            log(f"完成: ic_mean={res.get('ic_mean')}, ric_mean={res.get('ric_mean')}, "
                f"icir={res.get('icir')}, lgb_ic={res.get('lgb_ic_mean')}, xgb_ic={res.get('xgb_ic_mean')}")
            return res
        if s in ("failed", "error"):
            log(f"失败: {s} {d.get('error', '')[:200]}")
            return None
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keys = get_candidates()
    keys = filter_resolvable(keys)
    log(f"最终训练因子数: {len(keys)}")

    if args.dry_run:
        with open("logs/dedup_226_keys.json", "w", encoding="utf-8") as f:
            json.dump(keys, f, ensure_ascii=False, indent=2)
        log("[dry-run] 因子列表已写 logs/dedup_226_keys.json")
        return

    token = get_token()
    lgb_p, xgb_p = read_optuna_params()
    log(f"Optuna 参数: lgb colsample={lgb_p.get('colsample_bytree')}, "
        f"num_leaves={lgb_p.get('num_leaves')}, max_depth={lgb_p.get('max_depth')}")
    jid = start(token, keys, lgb_p, xgb_p)
    if not jid:
        return
    log(f"已启动 job={jid[:8]}, 训练中...")
    wait_done(token, jid)


if __name__ == "__main__":
    main()
