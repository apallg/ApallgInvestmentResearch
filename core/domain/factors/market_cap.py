"""
历史 point-in-time 市值读取 — 供因子截面市值中性化使用。
数据来源: daily_basic 表 (由 scripts/collect_daily_basic.py 从 TuShare 回填, total_mv 单位万元)。
"""
import logging
import datetime as dt

logger = logging.getLogger(__name__)

__all__ = ["get_market_cap_map"]


def get_market_cap_map(codes, eval_date):
    """返回 {code: total_mv}, 取每只 code 在 eval_date 当日或之前最近一条 total_mv。
    表空 / 无数据 / MySQL 不可用时返回 {} (触发下游退化为仅行业中性化)。"""
    if len(codes) == 0:
        return {}
    try:
        from core.data import Database
        if not Database.is_available():
            return {}
        end = eval_date if hasattr(eval_date, "isoformat") else dt.date.fromisoformat(str(eval_date))
        start = end - dt.timedelta(days=15)
        codes = list(codes)
        ph = ",".join(["%s"] * len(codes))
        rows = Database.fetchall(
            f"SELECT code, total_mv FROM daily_basic "
            f"WHERE code IN ({ph}) AND trade_date BETWEEN %s AND %s "
            f"ORDER BY code, trade_date",
            codes + [start.isoformat(), end.isoformat()],
        )
        result = {}
        for r in rows:
            mv = r.get("total_mv")
            if mv is None:
                continue
            mv = float(mv)
            if mv > 0:
                result[r["code"]] = mv  # 升序遍历, 最近日期覆盖前值
        return result
    except Exception as e:
        logger.debug(f"[市值] get_market_cap_map 失败: {e}")
        return {}
