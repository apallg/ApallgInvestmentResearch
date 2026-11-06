"""
L3 纯度检查脚本
扫描 core/domain/ 下所有 .py 文件，报告 L3 纯度违规。

规则：
- 禁止: from core.data, datetime.now(), datetime.date.today(), 文件 I/O
- 允许: time.time() 用于性能计时, 同层 domain import
- 白名单: 已知存量违规文件，纯化一个移除一个（白名单只减不增）

用法:
  python scripts/check_l3_purity.py        # 检查所有文件
  python scripts/check_l3_purity.py --strict  # 白名单中的文件也检查
"""
import ast
import re
import sys
from pathlib import Path

DOMAIN_DIR = Path(__file__).resolve().parent.parent / "core" / "domain"

# ---- 白名单：正在纯化中的文件 ----
# 纯化完成一个文件就从这里移除一行。只减不增。
WHITELIST = set()

# ---- 违规检测模式 ----
# 分两级:
#   SEVERE: 顶层 eager import (模块加载时产生副作用) — 非白名单立即失败
#   STANDARD: 时间函数/文件 I/O — 白名单跟踪 (函数内惰性 import 属已知偏差, 不计入)
FORBIDDEN_PATTERNS = [
    # A: 数据访问 — 仅顶层 import (无缩进)
    (r"^from core\.data\b", "A: 顶层 import (from core.data)"),
    (r"^import core\.data\b", "A: 顶层 import (import core.data)"),
    # B: 时间函数 — 覆盖别名变体 (dt.date.today / _dt.date.today / datetime.date.today)
    (r"\bdatetime\.now\(\)", "B: 时间函数 (datetime.now())"),
    (r"\bdate\.today\(\)", "B: 时间函数 (date.today())"),
    # C: 文件 I/O
    (r"\bopen\([^)]*['\"][^)]*\)", "C: 文件 I/O (open)"),
    (r"\.read_text\(\)", "C: 文件 I/O (Path.read_text)"),
    (r"\.write_text\(.*\)", "C: 文件 I/O (Path.write_text)"),
    (r"\.read_bytes\(\)", "C: 文件 I/O (Path.read_bytes)"),
    (r"\.write_bytes\(.*\)", "C: 文件 I/O (Path.write_bytes)"),
    (r"\.to_parquet\(.*\)", "C: 文件 I/O (to_parquet)"),
    (r"\.to_csv\(.*\)", "C: 文件 I/O (to_csv)"),
    (r"\bshutil\.(rmtree|copy|move)\b", "C: 文件 I/O (shutil)"),
    # D: 跨层 — 仅顶层 import
    (r"^from backend\.", "D: 顶层 import (from backend)"),
]

# 例外模式（不计违规）
EXCEPTION_PATTERNS = [
    r"#\s*noqa:\s*L3",          # 显式豁免标记
    r"time\.time\(\).*#\s*perf", # 性能计时标记
    r"^\s*#",                   # 注释行
    r"^\s*\"\"\"",              # docstring 开始
    r"L3\s+禁止",               # 文档说明中的文字
    r"raise ValueError.*L3",    # 守卫 raise (含 date.today 字样的守卫)
]

def _is_exception(line: str) -> bool:
    """检查行是否匹配例外模式"""
    for pat in EXCEPTION_PATTERNS:
        if re.search(pat, line):
            return True
    return False


def check_file(filepath: Path) -> list:
    rel = filepath.relative_to(DOMAIN_DIR.parent.parent)
    violations = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return violations

    lines = content.split("\n")
    in_docstring = False
    for lineno, line in enumerate(lines, 1):
        # 跟踪三引号 docstring 状态 (docstring 内容不是代码, 不计违规)
        if '"""' in line or "'''" in line:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if _is_exception(line):
            continue
        for pattern, category in FORBIDDEN_PATTERNS:
            m = re.search(pattern, line)
            if m:
                violations.append(
                    f"  {rel}:{lineno}  {category}\n    → {line.strip()[:100]}"
                )
                break  # 每行只报第一个违规
    return violations


def check_lazy_imports(filepath: Path) -> list:
    """AST 扫描函数体内的惰性跨层 import (from core.data/core.adapters/backend)。
    属 CLAUDE.md 已知架构偏差 (L3 惰性 import 绕过了顶层检查), 单独报告不阻断。
    用 col_offset>0 判断语句缩进 (模块顶层 import 的 col_offset=0, 正则已覆盖)。"""
    rel = filepath.relative_to(DOMAIN_DIR.parent.parent)
    lazy = []
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except Exception:
        return lazy
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset <= 0:
            continue  # 模块顶层, 由正则检测
        mod = node.module if isinstance(node, ast.ImportFrom) else (node.names[0].name if node.names else "")
        if mod and (mod.startswith("core.data") or mod.startswith("core.adapters")
                    or mod.startswith("backend")):
            lazy.append(f"  {rel}:{node.lineno}  惰性跨层 import ({mod})")
    return lazy


def main():
    strict = "--strict" in sys.argv
    all_violations = []
    lazy_imports = []
    whitelisted_count = 0
    total_files = 0

    for filepath in sorted(DOMAIN_DIR.rglob("*.py")):
        if filepath.name == "__init__.py" and filepath.stat().st_size < 50:
            continue  # 跳过空 __init__.py
        total_files += 1
        rel = str(filepath.relative_to(DOMAIN_DIR.parent.parent)).replace("\\", "/")

        if rel in WHITELIST and not strict:
            violations = check_file(filepath)
            if violations:
                whitelisted_count += len(violations)
            continue

        violations = check_file(filepath)
        if violations:
            all_violations.extend(violations)

        # 惰性跨层 import (已知偏差, 仅统计报告)
        lazy_imports.extend(check_lazy_imports(filepath))

    # 惰性跨层 import 统计 (已知偏差, 不阻断)
    if lazy_imports:
        print(f"\n{'='*60}")
        print(f"[INFO] 惰性跨层 import (已知架构偏差): {len(lazy_imports)} 处")
        print(f"       (CLAUDE.md 已知偏差; 顶层 import 已由正则拦截, 此为函数内惰性引用)")
        for v in lazy_imports:
            print(v)
        print(f"{'='*60}")

    # 输出
    if all_violations:
        print(f"\n{'='*60}")
        print(f"[FAIL] Non-whitelist violations: {len(all_violations)}")
        print(f"{'='*60}")
        for v in all_violations:
            print(v)
        print(f"\n{'='*60}")
        print(f"[FAIL] Total: {len(all_violations)} violations (non-whitelist)")
        print(f"[INFO] Whitelist violations: {whitelisted_count} (suppressed)")
        print(f"[INFO] Whitelist files: {len(WHITELIST)}")
        print(f"[INFO] Files checked: {total_files}")
        print(f"{'='*60}")
        return 1
    else:
        print(f"\n{'='*60}")
        print(f"[PASS] Non-whitelist files: zero violations")
        print(f"[INFO] Whitelist violations: {whitelisted_count} (to be fixed)")
        print(f"[INFO] Whitelist files: {len(WHITELIST)}")
        print(f"[INFO] Files checked: {total_files}")
        print(f"{'='*60}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
