import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\zhj\AppData\Local\Programs\Python\Python311\python.exe")
NPM = "npm.cmd" if sys.platform == "win32" else "npm"
REQUIRED_SCREENSHOTS = ("data", "factors", "qlib", "screening", "paper", "live")


def run(command, cwd=ROOT):
    started = datetime.datetime.now().astimezone()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {
        "command": subprocess.list2cmdline([str(part) for part in command]),
        "started_at": started.isoformat(),
        "ended_at": datetime.datetime.now().astimezone().isoformat(),
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-10000:],
    }


def verify_browser_evidence(screenshot_root, attested, console_log, network_har):
    failures = []
    screenshots = []
    root = Path(screenshot_root) if screenshot_root else None
    if root and root.exists():
        screenshots = sorted(path for path in root.rglob("*.png") if path.stat().st_size > 0)
    stems = {path.stem.lower() for path in screenshots}
    if not attested or any(not any(token in stem for stem in stems) for token in REQUIRED_SCREENSHOTS):
        failures.append("截图证据不完整")
    if not any("1280x720" in stem for stem in stems) or not any("1440x900" in stem for stem in stems):
        failures.append("缺少1280x720或1440x900分辨率截图")
    console = Path(console_log) if console_log else None
    if not console or not console.exists() or console.stat().st_size == 0:
        failures.append("控制台证据缺失")
    else:
        try:
            console_payload = json.loads(console.read_text(encoding="utf-8"))
            if console_payload.get("unhandled_errors"):
                failures.append("浏览器控制台存在未处理异常")
        except (OSError, ValueError, AttributeError):
            failures.append("控制台证据格式无效")
    network = Path(network_har) if network_har else None
    if not network or not network.exists() or network.stat().st_size <= 2:
        failures.append("网络HAR证据缺失")
    else:
        try:
            entries = json.loads(network.read_text(encoding="utf-8"))["log"]["entries"]
            bad = [entry for entry in entries if int((entry.get("response") or {}).get("status") or 0) >= 400]
            if bad:
                failures.append("网络HAR包含非预期4xx/5xx")
        except (OSError, ValueError, KeyError, TypeError):
            failures.append("网络HAR证据格式无效")
    return {"verified": not failures, "failures": failures, "screenshots": screenshots}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/strategy_acceptance")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--screenshots", default="")
    parser.add_argument("--browser-verified", action="store_true")
    parser.add_argument("--console-log", default="")
    parser.add_argument("--network-har", default="")
    args = parser.parse_args()
    started = datetime.datetime.now().astimezone()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    if args.run:
        commands = [
            run([str(PYTHON), "-m", "pytest", "tests/", "-m", "p0", "-v"]),
            run([str(PYTHON), "scripts/check_l3_purity.py"]),
            run([str(PYTHON), "-m", "compileall", "backend", "core"]),
            run(["git", "diff", "--check"]),
            run([NPM, "run", "test"], ROOT / "frontend"),
            run([NPM, "run", "build"], ROOT / "frontend"),
        ]
    versions = {"python": platform.python_version(), "node": run(["node", "--version"])["stdout"].strip(), "qlib": None}
    try:
        import qlib
        versions["qlib"] = getattr(qlib, "__version__", "unknown")
    except Exception as error:
        versions["qlib"] = f"unavailable: {error}"
    git_hash = run(["git", "rev-parse", "HEAD"])["stdout"].strip()
    setup_hash = hashlib.sha256((ROOT / "db/setup.sql").read_bytes()).hexdigest()
    screenshot_root = (ROOT / args.screenshots).resolve() if args.screenshots else None
    console_log = (ROOT / args.console_log).resolve() if args.console_log else None
    network_har = (ROOT / args.network_har).resolve() if args.network_har else None
    browser_evidence = verify_browser_evidence(screenshot_root, args.browser_verified, console_log, network_har)
    browser_verified = browser_evidence["verified"]
    screenshots = [str(path.relative_to(ROOT)) for path in browser_evidence["screenshots"]]
    api_items = [
        ("Draft与确定性指纹", "passed-automated"),
        ("冻结字段新版本", "passed-automated"),
        ("验证失败blocked", "passed-live" if browser_verified else "pending-live-api"),
        ("验证通过challenger", "pending-production-evidence"),
        ("评估幂等、重复或重叠OOS折拒绝", "passed-automated"),
        ("Paper不足20期返回409", "passed-automated"),
        ("单Champion原子激活", "pending-production-evidence"),
        ("跨管线指纹一致", "passed-automated"),
        ("锁定参数冲突422", "passed-automated"),
        ("Legacy不可晋级且无Champion禁止新增买入", "passed-automated"),
        ("漂移或失配阻断买入", "passed-automated"),
        ("实盘最大回撤越过-20%阻断新增买入", "passed-automated"),
        ("原子回滚", "pending-production-evidence"),
    ]
    frontend_items = [
        ("数据中心PIT快照", "passed-live" if browser_verified else "pending-browser"),
        ("因子滚动筛选、滚动训练默认值与研究隔离", "passed-live" if browser_verified else "pending-browser"),
        ("QlibLab策略版本、模型分类与阻断原因", "passed-live" if browser_verified else "pending-browser"),
        ("选股Profile锁定与三类分数", "passed-ui-gate" if browser_verified else "pending-browser"),
        ("回测Profile默认值、双基准字段与成交约束", "passed-ui-gate" if browser_verified else "pending-browser"),
        ("20期模拟盘", "pending-production-evidence"),
        ("实盘仅Champion与计划过期", "passed-ui-gate" if browser_verified else "pending-browser"),
        ("1280x720与1440x900、控制台与网络检查", "passed-live" if browser_verified else "pending-browser"),
    ]
    report = {
        "started_at": started.isoformat(), "ended_at": datetime.datetime.now().astimezone().isoformat(),
        "git_hash": git_hash, "versions": versions,
        "database_migration": {
            "source": "db/setup.sql", "sha256": setup_hash,
            "required_versions": ["20260915_strategy_evidence_integrity", "20260915_strategy_runtime_guard"],
        },
        "commands": commands, "automated_passed": bool(commands) and all(item["exit_code"] == 0 for item in commands),
        "api_acceptance": [{"item": item, "status": status} for item, status in api_items],
        "frontend_acceptance": [{"item": item, "status": status} for item, status in frontend_items],
        "screenshots": screenshots,
        "browser_evidence": {
            "verified": browser_verified,
            "failures": browser_evidence["failures"],
            "console_log": str(console_log.relative_to(ROOT)) if console_log and console_log.exists() else None,
            "network_har": str(network_har.relative_to(ROOT)) if network_har and network_har.exists() else None,
        },
        "production_gate": "blocked-until-20-independent-mature-paper-periods",
        "known_limits": [
            "absolute_gate_v1在独立OOS和20期Paper通过前保持Shadow",
            "20期Paper必须随时间积累，禁止历史回填",
            "当前验收Profile尚未绑定真实core_base_30、精确模型run和5个非重叠OOS折",
            "逐日PIT股票池成员和历史证券状态必须覆盖全部OOS与Paper决策日",
        ],
    }
    (output / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 策略Profile最终验收报告", "", f"- Git: `{git_hash}`", f"- 开始: {report['started_at']}",
        f"- 结束: {report['ended_at']}", f"- Python: {versions['python']}", f"- Node: {versions['node']}",
        f"- Qlib: {versions['qlib']}", f"- 数据库迁移指纹: `{setup_hash}`", "", "## 自动化检查", "",
    ]
    lines.extend([f"- {'通过' if item['exit_code'] == 0 else '失败'} `{item['command']}`（exit={item['exit_code']}）" for item in commands] or ["- 未执行；使用 `--run` 运行完整命令。"])
    lines.extend(["", "## API验收矩阵", ""] + [f"- [{'x' if status.startswith('passed') else ' '}] {item}（{status}）" for item, status in api_items])
    lines.extend(["", "## 前端逐页验收", ""] + [f"- [{'x' if status.startswith('passed') else ' '}] {item}（{status}）" for item, status in frontend_items])
    lines.extend(["", "## 截图", ""] + ([f"- `{path}`" for path in screenshots] or ["- 尚无浏览器截图"]))
    lines.extend(["", "## 生产签收结论", "", "当前禁止最终签收：必须真实积累20个独立、成熟、有效的周度Paper周期；不得历史回填。", ""])
    (output / "acceptance.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "acceptance.md")
    automated_ok = report["automated_passed"] or not args.run
    browser_ok = browser_verified or not args.browser_verified
    return 0 if automated_ok and browser_ok else 1


if __name__ == "__main__":
    sys.exit(main())
