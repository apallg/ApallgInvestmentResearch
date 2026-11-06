"""
adj_factor 迁移脚本 — 为 daily_quotes 添加后复权因子列并回填历史数据。

用途: 将 daily_quotes 从"裸前复权价"升级为"adj_factor 可用的 price"，为 point-in-time 复权重建打基础。

工作流:
  1. ALTER TABLE 加 adj_factor 列 (幂等)
  2. 获取所有有行情数据的股票列表
  3. 逐只从 Tushare 拉 adj_factor, MERGE 写入 daily_quotes
  4. 统计迁移结果

用法:
  python scripts/migrate_adj_factor.py           # 全量迁移 (全部股票, 2012起)
  python scripts/migrate_adj_factor.py --dryrun  # 仅检查, 不写库
  python scripts/migrate_adj_factor.py --limit 50  # 仅迁移前50只 (测试用)
"""
import sys
import os
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import time
import numpy as np
import pandas as pd
from core.data.connection import Database
from core.data.repos.quotes import DailyQuotesRepo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_adj_factor")

ALTER_SQL = (
    "ALTER TABLE daily_quotes "
    "ADD COLUMN adj_factor DECIMAL(12,6) DEFAULT NULL "
    "COMMENT '后复权因子 (原始价×adj_factor=后复权价, Tushare pro.adj_factor)'"
)


def add_column():
    try:
        Database._execute_with_retry(lambda cur: cur.execute(ALTER_SQL) or True, "SELECT 1", commit=True)
        logger.info("adj_factor 列添加成功")
    except Exception as e:
        if "Duplicate column" in str(e) or "already exists" in str(e):
            logger.info("adj_factor 列已存在, 跳过")
        else:
            logger.warning(f"ALTER TABLE 失败: {e}")


def get_stock_list():
    rows = Database.fetchall(
        "SELECT DISTINCT code FROM daily_quotes WHERE code NOT LIKE 'sh%' AND code NOT LIKE 'sz%' "
        "ORDER BY code")
    return [r["code"] for r in rows] if rows else []


def fetch_and_merge(code, dryrun=False):
    from core.data.sources.tushare import get_adj_factor
    from core.data.sources.tushare import RateLimiter, _rate_limit as orig_rate_limit
    from importlib.metadata import version
    from uuid import uuid4

    df = get_adj_factor(code, start_date="20120101")
    if df is None or df.empty:
        return 0
    if dryrun:
        return len(df)
    factor_date = "date" if "date" in df.columns else "trade_date"
    df.attrs["provenance"] = {
        "batch_id": str(uuid4()),
        "source": "tushare",
        "provider_version": version("tushare"),
        "adjustment_mode": "factor_only",
        "anchor_date": str(pd.to_datetime(df[factor_date].max()).date()),
        "input_payload_sha256": hashlib.sha256(
            df.reindex(sorted(df.columns), axis=1).to_json(
                orient="split", date_format="iso", double_precision=15, default_handler=str
            ).encode("utf-8")
        ).hexdigest(),
    }
    DailyQuotesRepo.merge_adj_factors(code, df)
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="daily_quotes adj_factor 列迁移")
    parser.add_argument("--dryrun", action="store_true", help="仅检查, 不写库")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 只股票 (0=全部)")
    args = parser.parse_args()

    if not Database.is_available():
        logger.error("数据库不可用, 请检查 .env 配置")
        return 1

    add_column()

    codes = get_stock_list()
    logger.info(f"共 {len(codes)} 只股票有待迁移 adj_factor")

    if args.limit > 0:
        codes = codes[:args.limit]
        logger.info(f"限制模式: 仅处理前 {len(codes)} 只")

    total_rows = 0
    success = 0
    fail = 0
    t0 = time.monotonic()

    for i, code in enumerate(codes):
        try:
            n = fetch_and_merge(code, dryrun=args.dryrun)
            total_rows += n
            success += 1
            if (i + 1) % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed * 60
                logger.info(f"进度 {i+1}/{len(codes)} ({(i+1)/len(codes)*100:.0f}%), "
                            f"{total_rows} 行, 速率 {rate:.0f} 只/分")
            # Tushare 频率控制: 每分钟最多 ~180 次
            if (i + 1) % 50 == 0:
                time.sleep(1)
        except Exception as e:
            fail += 1
            if fail <= 5:
                logger.warning(f"{code} 迁移失败: {e}")

    elapsed = time.monotonic() - t0
    dry_tag = " [dryrun, 未写入]" if args.dryrun else ""
    logger.info(f"迁移完成{dry_tag}: 成功 {success}/{len(codes)}, 失败 {fail}, "
                f"写入 {total_rows} 行, 耗时 {elapsed/60:.1f} 分")

    # 统计覆盖率
    if not args.dryrun:
        total = Database.fetchone("SELECT COUNT(*) as cnt FROM daily_quotes")["cnt"]
        filled = Database.fetchone(
            "SELECT COUNT(*) as cnt FROM daily_quotes WHERE adj_factor IS NOT NULL")["cnt"]
        pct = filled / total * 100 if total > 0 else 0
        logger.info(f"adj_factor 覆盖率: {filled}/{total} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
