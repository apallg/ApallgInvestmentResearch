"""
重算 factor_validation_scores 的 icir 字段为带符号 (ic_mean/ic_std, 不用 abs)。

背景: 原 recalc-all-icir 用 abs(ic_mean)/ic_std, 丢弃 IC 方向, 导致「稳定为负 IC」的
_n 扩展因子被误判为高质量。此脚本把 icir 更新为带符号值, 使负 IC 因子沉底。

用法: python scripts/recalc_icir_signed.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("recalc_icir")


def main():
    from core.data import Database
    from core.domain.factors.registry import get_pool

    if not Database.is_available():
        logger.error("数据库不可用")
        return 1

    pool_mgr = get_pool()
    active_keys = list(pool_mgr.get_active_factors().keys())
    logger.info(f"活跃因子 {len(active_keys)} 个")

    BATCH = 200
    factor_ics = {}
    for i in range(0, len(active_keys), BATCH):
        batch = active_keys[i:i + BATCH]
        ph = ','.join(['%s'] * len(batch))
        try:
            rows = Database.fetchall(
                f"SELECT factor_key, daily_ics FROM factor_validation_scores "
                f"WHERE daily_ics IS NOT NULL AND factor_key IN ({ph})", batch)
        except Exception:
            continue
        for r in rows:
            key = r["factor_key"]
            daily_ics = r["daily_ics"]
            if not daily_ics:
                continue
            try:
                pairs = json.loads(daily_ics) if isinstance(daily_ics, str) else daily_ics
            except Exception:
                continue
            ics = [float(p[1]) for p in pairs if len(p) >= 2 and p[1] is not None]
            factor_ics.setdefault(key, []).extend(ics)

    updated = 0
    neg = 0
    skipped = 0
    for key in active_keys:
        all_ics = factor_ics.get(key, [])
        if len(all_ics) < 3:
            skipped += 1
            continue
        ic_mean = sum(all_ics) / len(all_ics)
        ic_std = (sum((x - ic_mean) ** 2 for x in all_ics) / len(all_ics)) ** 0.5
        icir = ic_mean / ic_std if ic_std > 0 else 0  # 带符号, 不用 abs
        if icir < 0:
            neg += 1
        try:
            Database.execute(
                "UPDATE factor_validation_scores SET icir=%s, ic_mean=%s, ic_std=%s, "
                "n_ic_dates=%s WHERE factor_key=%s",
                (round(icir, 6), round(ic_mean, 6), round(ic_std, 6), len(all_ics), key))
            updated += 1
        except Exception:
            pass

    logger.info(f"已更新 {updated} 个因子 (带符号 icir), 其中负 icir {neg} 个, 跳过 {skipped} 个数据不足")
    return 0


if __name__ == "__main__":
    sys.exit(main())
