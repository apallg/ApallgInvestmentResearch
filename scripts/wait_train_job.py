"""轮询已有训练 job 直到完成, 结果写 logs/wait_train_job.log。

用法: python scripts/wait_train_job.py <job_id> [label]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import requests

BASE = "http://localhost:8001"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open("logs/wait_train_job.log", "a", encoding="utf-8") as f:
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


def main():
    jid = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "job"
    token = get_token()
    log(f"开始轮询 {label} job={jid[:8]}")
    last_status = ""
    while True:
        try:
            r = requests.get(f"{BASE}/api/qlib/models/train/{jid}",
                             headers={"Authorization": f"Bearer {token}"}, timeout=10)
            d = r.json()["data"]
            s = d.get("status")
        except Exception as e:
            s = "error"
            log(f"查询异常: {e}")
        if s != last_status:
            log(f"状态: {s} {str(d.get('message', ''))[:100]}")
            last_status = s
        if s == "completed":
            res = d.get("result", {})
            out = {
                "job_id": jid, "label": label,
                "ic_mean": res.get("ic_mean"),
                "ric_mean": res.get("ric_mean"),
                "icir": res.get("icir"),
                "lgb_ic": res.get("lgb_ic_mean"),
                "xgb_ic": res.get("xgb_ic_mean"),
            }
            log(f"完成: ic_mean={res.get('ic_mean')}, ric_mean={res.get('ric_mean')}, "
                f"icir={res.get('icir')}, lgb_ic={res.get('lgb_ic_mean')}, xgb_ic={res.get('xgb_ic_mean')}")
            with open("logs/wait_train_job_result.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2, default=str)
            return
        if s in ("failed", "error"):
            log(f"失败: {s} {str(d.get('error', ''))[:300]}")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
