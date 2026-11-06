"""
capital_flow_daily 四档资金列迁移 + 历史回补脚本。

为 capital_flow_daily 添加散户/中户/大户/机构四档净流入列, 并回补历史数据。

工作流:
  1. ALTER TABLE 加 4 列 (幂等)
  2. 取最近 N 个交易日 (默认 20)
  3. 逐日调用 integrity 的 moneyflow 回补

用法:
  python scripts/migrate_capital_flow_tiers.py            # 加列 + 回补 20 日
  python scripts/migrate_capital_flow_tiers.py --days 60  # 回补 60 日
  python scripts/migrate_capital_flow_tiers.py --no-backfill  # 仅加列
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging

from core.data.connection import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_capital_flow_tiers")

_COLUMNS = [
    ("sm_net_flow", "DECIMAL(16,2) DEFAULT NULL", "散户净流入(小单<5万, 万元)"),
    ("md_net_flow", "DECIMAL(16,2) DEFAULT NULL", "中户净流入(中单5-20万, 万元)"),
    ("lg_net_flow", "DECIMAL(16,2) DEFAULT NULL", "大户净流入(大单20-100万, 万元)"),
    ("elg_net_flow", "DECIMAL(16,2) DEFAULT NULL", "机构净流入(超大单>100万, 万元)"),
]


def add_columns():
    for name, ddl, comment in _COLUMNS:
        sql = f"ALTER TABLE capital_flow_daily ADD COLUMN {name} {ddl} COMMENT '{comment}'"
        try:
            Database.execute(sql)
            logger.info(f"{name} 列添加成功")
        except Exception as e:
            if "Duplicate column" in str(e) or "already exists" in str(e):
                logger.info(f"{name} 列已存在, 跳过")
            else:
                logger.warning(f"ALTER TABLE {name} 失败: {e}")


def get_trade_dates(days):
    rows = Database.fetchall(
        "SELECT DISTINCT trade_date FROM capital_flow_daily ORDER BY trade_date DESC LIMIT %s",
        (days * 6000,))
    if not rows:
        return []
    dates = [str(r["trade_date"]) for r in rows]
    return sorted(set(dates))[-days:]


def backfill(dates):
    from core.data.integrity import _repair_capital_flow
    total = 0
    for d in dates:
        try:
            n = _repair_capital_flow(d)
            total += n
            logger.info(f"[回补] {d}: {n} 只")
        except Exception as e:
            logger.warning(f"[回补] {d} 失败: {e}")
    return total


def main():
    parser = argparse.ArgumentParser(description="capital_flow_daily 四档资金列迁移")
    parser.add_argument("--days", type=int, default=20, help="回补最近 N 个交易日")
    parser.add_argument("--no-backfill", action="store_true", help="仅加列, 不回补")
    args = parser.parse_args()

    if not Database.is_available():
        logger.error("数据库不可用, 请检查 .env 配置")
        return 1

    add_columns()

    if args.no_backfill:
        logger.info("仅加列完成, 跳过回补")
        return 0

    dates = get_trade_dates(args.days)
    logger.info(f"待回补交易日: {len(dates)} 个")
    total = backfill(dates)
    logger.info(f"回补完成: 共 {total} 只次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
