"""导出活跃因子池完整清单 (key + source + category + label + expr), 供人工组合因子集。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.domain.factors.registry import get_pool
from core.domain.factors.adapter import PublicFactorAdapter
from core.domain.factors.selector import group_by_category


def main():
    pool = get_pool()
    active = pool.get_active_factors()
    adapter = PublicFactorAdapter()
    groups = group_by_category(list(active.keys()), adapter, pool_active=active)

    lines = []
    lines.append(f"总活跃因子: {len(active)}")
    for cat, keys in groups.items():
        lines.append(f"\n===== {cat} ({len(keys)}) =====")
        for k in keys:
            info = adapter.all_factors.get(k, {})
            label = info.get("label", "")
            src = info.get("source", "")
            expr = (info.get("expr") or "").replace("\n", " ")
            if len(expr) > 100:
                expr = expr[:100] + "..."
            lines.append(f"  {k:<26} [{src:<10}] {label:<22} {expr}")

    out = "\n".join(lines)
    out_path = os.path.join("logs", "active_factors_dump.txt")
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\n\n已保存到 {out_path}")


if __name__ == "__main__":
    main()
