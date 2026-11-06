"""
验证去冗余是否误删好因子 — 复现 deduplicate_factors 的贪心聚类, 记录被删因子详情。

关键问题: 去冗余用"单日截面 |IC|"选每组的代表 (adapter.py:910), 单日 IC 噪声大,
          可能误删"单日 IC 低但真实 alpha 高"的因子。

输出: 被删因子列表 + 它们的单日 |IC| + 与保留因子的相关性 + 保留因子的 IC。
      判断标准: 被删因子中 |IC| 与保留因子接近 (差距小) 的越多, 说明单日 IC
      区分度差, 误删越接近"掷骰子"。

用法:
  python scripts/check_dedupe_quality.py trend_momentum
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt
import numpy as np


def main():
    direction = sys.argv[1] if len(sys.argv) > 1 else "trend_momentum"
    pool = sys.argv[2] if len(sys.argv) > 2 else "csi300"
    anchor_str = sys.argv[3] if len(sys.argv) > 3 else "2026-07-14"
    anchor = dt.date.fromisoformat(anchor_str)

    from core.domain.factors.selector import DIRECTION_PRESETS, group_by_category
    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter
    from core.domain.factors.validation import OverfittingChecker
    from core.domain.screening._screener import get_screener
    from scipy.stats import spearmanr

    # 取候选 (与 refine 脚本 step1 一致)
    preset = DIRECTION_PRESETS[direction]
    pool_mgr = get_pool()
    active = pool_mgr.get_active_factors()
    adapter = PublicFactorAdapter()
    category_groups = group_by_category(list(active.keys()), adapter, pool_active=active)
    candidates = []
    for cat in preset["categories"]:
        candidates.extend(category_groups.get(cat, []))
    candidates = list(dict.fromkeys(candidates))

    # 快照
    screener = get_screener()
    snapshot = screener.precompute_snapshot(pool=pool, eval_date=anchor, skip_limit_up=True)
    checker = OverfittingChecker()
    snap_map = {anchor.isoformat(): snapshot}

    # 复现 deduplicate_factors 的 vecs + ic_abs
    vecs = {}
    for key in candidates:
        v = adapter.to_variant(key)
        if not v:
            continue
        cs = checker._variant_scores(v, anchor, anchor.isoformat(), pool, None, snap_map, screener, allow_composite=False)
        if cs is None:
            continue
        codes, scores = cs
        if len(codes) < 20:
            continue
        vecs[key] = {c: float(s) for c, s in zip(codes, scores)}

    all_codes = sorted({c for m in vecs.values() for c in m})
    fwd = checker._batch_forward_returns(all_codes, 20, anchor)
    fwd_map = {c: r for c, r in zip(all_codes, fwd)}
    ic_abs = {}
    for key, m in vecs.items():
        xs, ys = [], []
        for c, val in m.items():
            r = fwd_map.get(c)
            if r is not None:
                xs.append(val)
                ys.append(r)
        if len(xs) >= 10:
            corr, _ = spearmanr(xs, ys)
            ic_abs[key] = abs(float(corr)) if corr is not None and not np.isnan(corr) else 0.0
        else:
            ic_abs[key] = 0.0

    # 贪心聚类 + 记录被删详情
    order = sorted(vecs.keys(), key=lambda k: ic_abs.get(k, 0.0), reverse=True)
    kept = []
    dropped = []  # (dropped_key, kept_key, corr, ic_dropped, ic_kept)
    for key in order:
        m = vecs[key]
        redundant = False
        for kk in kept:
            common = [c for c in m if c in vecs[kk]]
            if len(common) < 20:
                continue
            a = np.array([m[c] for c in common])
            b = np.array([vecs[kk][c] for c in common])
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                continue
            r = abs(float(np.corrcoef(a, b)[0, 1]))
            if not np.isnan(r) and r > 0.8:
                dropped.append((key, kk, round(r, 3), round(ic_abs.get(key, 0), 4), round(ic_abs.get(kk, 0), 4)))
                redundant = True
                break
        if not redundant:
            kept.append(key)

    unresolved = [k for k in candidates if k not in vecs]
    print(f"方向 {direction}: 候选 {len(candidates)}, 可求值 {len(vecs)}, 保留 {len(kept)}, 被删 {len(dropped)}, 不可求值 {len(unresolved)}")
    print(f"\n被删因子的单日 |IC| 分布:")
    ic_dropped = sorted([d[3] for d in dropped], reverse=True)
    for thr in [0.10, 0.08, 0.05, 0.03, 0.02, 0.01]:
        n = sum(1 for x in ic_dropped if x >= thr)
        print(f"  |IC| >= {thr:.2f}: {n} 个")
    print(f"\n被删因子里 |IC| 最高的 20 个 (可能误删, 它们单日 IC 不低却被删):")
    for d in sorted(dropped, key=lambda x: -x[3])[:20]:
        gap = d[4] - d[3]
        print(f"  {d[0]:<38} 被 {d[1]:<32} 代表 |r|={d[2]:<5} 被删IC={d[3]:<7} 保留IC={d[4]:<7} 差距={gap:.4f}")
    # 关键指标: 被删因子中, 有多少 IC 差距极小 (单日 IC 无法可靠区分)
    small_gap = [d for d in dropped if (d[4] - d[3]) < 0.01]
    print(f"\n结论: {len(dropped)} 个被删因子中, {len(small_gap)} 个与保留因子 IC 差距 < 0.01 (单日 IC 区分度差, 误删≈掷骰子)")


if __name__ == "__main__":
    main()
