"""
验证「插入顺序选经典因子所以 icir 稳」这个结论。

实际机制假设: 精选排序用的 ICIR = abs(ic_mean)/ic_std, 丢弃了方向,
导致「稳定为负 IC」的 _n 扩展因子被误判为高质量, 而插入顺序不依赖 ICIR 所以免疫。

验证三个点:
  1. icir(排序)选中的因子里, 有多少 ic_mean 是负的 (被 abs 误导)
  2. uniform(插入顺序)选中的因子, ic_mean 符号分布是否更健康
  3. _n 扩展因子的 daily_ics 是否大量重复 (同一份数据多个 key)

用法: python scripts/verify_insert_order_hypothesis.py trend_momentum
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import defaultdict

PLACEHOLDERS = {"flow", "size", "tech", "value", "quality", "test_f1", "momentum"}


def _build_sets(direction):
    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter
    from core.domain.factors.selector import curate_direction

    pool_mgr = get_pool()
    active = pool_mgr.get_active_factors()
    active_keys = list(active.keys())
    adapter = PublicFactorAdapter()

    _, flat_a, _ = curate_direction(direction, active_keys, adapter, None,
                                    save=False, regime_aware=True,
                                    pool_active=active, method="uniform")
    _, flat_b, _ = curate_direction(direction, active_keys, adapter, None,
                                    save=False, regime_aware=True,
                                    pool_active=active, method="icir")
    fa = [k for k in flat_a if k not in PLACEHOLDERS]
    fb = [k for k in flat_b if k not in PLACEHOLDERS]
    return fa, fb


def _fetch(keys):
    from core.data import Database
    if not keys:
        return {}
    ph = ','.join(['%s'] * len(keys))
    rows = Database.fetchall(
        f"SELECT factor_key, icir, ic_mean, daily_ics FROM factor_validation_scores "
        f"WHERE factor_key IN ({ph}) AND daily_ics IS NOT NULL", keys)
    out = {}
    for r in rows:
        ics = [float(p[1]) for p in (json.loads(r["daily_ics"]) if isinstance(r["daily_ics"], str) else r["daily_ics"])]
        out[r["factor_key"]] = {
            "icir": r["icir"], "ic_mean": r["ic_mean"], "ics": ics,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", nargs="?", default="trend_momentum")
    args = ap.parse_args()

    fa, fb = _build_sets(args.direction)
    da, db = _fetch(fa), _fetch(fb)
    print(f"uniform(插入顺序) 可训练因子 {len(fa)} 个, 有校验数据 {len(da)} 个")
    print(f"icir(排序)     可训练因子 {len(fb)} 个, 有校验数据 {len(db)} 个")

    # 1. ic_mean 符号分布
    def sign_dist(d, tag):
        pos = sum(1 for v in d.values() if (v["ic_mean"] or 0) > 0)
        neg = sum(1 for v in d.values() if (v["ic_mean"] or 0) < 0)
        zero = len(d) - pos - neg
        print(f"\n[{tag}] ic_mean 符号: 正={pos} 负={neg} 零={zero}")
        return neg

    neg_a = sign_dist(da, "uniform(插入顺序)")
    neg_b = sign_dist(db, "icir(排序)")

    # 2. icir 排序里「高 icir 但负 ic_mean」的因子 (被 abs 误导)
    misled = sorted(
        [(k, v["icir"], v["ic_mean"]) for k, v in db.items()
         if (v["icir"] or 0) > 1.0 and (v["ic_mean"] or 0) < 0],
        key=lambda x: x[1], reverse=True)
    print(f"\n[icir排序] 高ICIR(>1)但负IC均值的因子 (被 abs 误导): {len(misled)} 个")
    for k, icir, icm in misled[:15]:
        print(f"    {k:<30} icir={icir:+.3f} ic_mean={icm:+.4f}")

    # 3. daily_ics 重复检测 (同一份数据被多个 key 共享)
    sig_to_keys = defaultdict(list)
    for key, v in {**da, **db}.items():
        sig = tuple(round(x, 6) for x in v["ics"])
        sig_to_keys[sig].append(key)
    dup_groups = {s: ks for s, ks in sig_to_keys.items() if len(ks) > 1}
    print(f"\n[daily_ics重复] 完全相同的 IC 序列被多个 key 共享: {len(dup_groups)} 组")
    for sig, ks in sorted(dup_groups.items(), key=lambda x: -len(x[1]))[:8]:
        print(f"    {len(ks)} 个 key 共享: {', '.join(ks[:6])}{'...' if len(ks)>6 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
