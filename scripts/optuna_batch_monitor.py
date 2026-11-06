#!/usr/bin/env python3
"""Optuna 批量搜索监控 — 依次跑 4 个精选集。
用法: python scripts/optuna_batch_monitor.py
"""

import subprocess, time, json, sys
from datetime import datetime

API = "http://localhost:8001/api/qlib/models"

FACTOR_SETS = [
    "trend_momentum",
    "oversold_rebound",
    "volume_breakout",
    "value_defense",
]

# 搜索参数 (与前端 Optuna 搜索配置一致)
SEARCH_PARAMS = {
    "search_dual": True,
    "instruments": "a500",
    "train_start": "2019-01-01",
    "train_end": "2025-12-31",
    "valid_start": "2026-01-01",
    "valid_end": "2026-06-30",
    "label_horizon": 5,
    "label_norm": "none",
    "n_trials": 50,
    "timeout": 0,
}

CURL = "curl"
POLL_INTERVAL = 60  # 秒


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def curl_get(url):
    r = subprocess.run(
        [CURL, "-s", url],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def curl_post(url, payload):
    r = subprocess.run(
        [CURL, "-s", "-X", "POST", url, "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def start_search(factor_set):
    """启动一个因子集的 Optuna 搜索，返回 job_id"""
    params = dict(SEARCH_PARAMS)
    params["use_curated"] = factor_set
    resp = curl_post(f"{API}/optuna-search", params)
    if not resp or "data" not in resp:
        log(f"  ❌ 启动失败: {resp}")
        return None
    job_id = resp["data"]["job_id"]
    log(f"  ✅ 已启动: job_id={job_id}")
    return job_id


def poll_job(job_id):
    """轮询任务状态，返回 (status, info_dict)"""
    resp = curl_get(f"{API}/train/{job_id}")
    if not resp or "data" not in resp:
        return None, {}
    data = resp["data"]
    return data.get("status"), data


def monitor_job(job_id, factor_set):
    """轮询一个任务直到完成或超时，返回结果"""
    max_wait = 12 * 3600  # 最多等 12 小时
    start = time.time()
    last_phase = ""
    trial_count = 0

    while time.time() - start < max_wait:
        status, info = poll_job(job_id)
        if status is None:
            log(f"  ⚠️ 轮询失败, 5s 后重试...")
            time.sleep(5)
            continue

        phase = info.get("phase", "")
        msg = info.get("message", "")
        lgb_ic = info.get("lgb_best_ic")

        # 进度日志：仅在 phase 变化或 IC 更新时打印
        if phase != last_phase:
            log(f"  📊 [{factor_set}] 阶段: {phase} | {msg}")
            last_phase = phase
        if lgb_ic is not None and lgb_ic != 0:
            log(f"  📊 [{factor_set}] LGB best IC: {lgb_ic:.4f}")

        if status == "completed":
            result = info.get("result", {})
            log(f"  ✅ [{factor_set}] 完成!")
            log(f"     LGB IC: {result.get('lgb_best_ic', 'N/A')}")
            log(f"     XGB IC: {result.get('xgb_best_ic', 'N/A')}")
            log(f"     IC Mean: {result.get('ic_mean', 'N/A')}")
            return result

        if status in ("failed", "cancelled"):
            log(f"  ❌ [{factor_set}] 失败: {info}")
            return None

        # 打印 LGB trial 进度（每20次打印一次，减少日志噪音）
        trial_count += 1
        if trial_count % 20 == 0:
            log(f"  ⏳ [{factor_set}] 轮询中... ({phase}: {(time.time()-start)/60:.0f}min)")

        time.sleep(POLL_INTERVAL)

    log(f"  ⚠️ [{factor_set}] 超时 (>{max_wait/3600:.0f}h)")
    return None


def main():
    log("=" * 50)
    log("Optuna 批量搜索启动")
    log(f"因子集: {FACTOR_SETS}")
    log(f"参数: {json.dumps(SEARCH_PARAMS, indent=2)}")
    log("=" * 50)

    # 检查后端
    health = curl_get("http://localhost:8001/api/system/health")
    if not health:
        log("❌ 后端未就绪，退出")
        sys.exit(1)
    log("✅ 后端已连接")

    results = {}

    for i, fs in enumerate(FACTOR_SETS, 1):
        log(f"\n{'='*50}")
        log(f"[{i}/{len(FACTOR_SETS)}] 开始搜索: {fs}")
        log(f"{'='*50}")

        job_id = start_search(fs)
        if not job_id:
            log(f"  ⚠️ 跳过 {fs}")
            continue

        result = monitor_job(job_id, fs)
        if result:
            results[fs] = result

    # 最终汇总
    log(f"\n{'='*50}")
    log("全部搜索完成!")
    log(f"{'='*50}")
    for fs, r in results.items():
        log(f"  {fs}:")
        log(f"    LGB IC: {r.get('lgb_best_ic', 'N/A')}")
        log(f"    XGB IC: {r.get('xgb_best_ic', 'N/A')}")
        log(f"    IC Mean: {r.get('ic_mean', 'N/A')}")

    # 列出已保存结果
    params_list = curl_get("http://localhost:8001/api/qlib/optuna-params")
    if params_list and "data" in params_list:
        log(f"\n📁 已保存的 Optuna 结果:")
        for p in params_list["data"]:
            log(f"  {p.get('factor_set'):20s} | LGB IC: {p.get('lgb_best_ic', 'N/A')} | XGB IC: {p.get('xgb_best_ic', 'N/A')} | {p.get('saved_at', '')}")

    log("\n✅ 全部完成!")


if __name__ == "__main__":
    main()
