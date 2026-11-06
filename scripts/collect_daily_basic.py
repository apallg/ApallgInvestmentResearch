"""
TuShare daily_basic 历史市值采集 — 用于因子截面市值中性化

用法:
  python -m scripts.collect_daily_basic                # 采集 daily_quotes 覆盖的全部缺失交易日
  python -m scripts.collect_daily_basic --days 90      # 只采集最近 90 个交易日
  python -m scripts.collect_daily_basic --start 20240101 --end 20241231
"""
import sys
import os
import time
import logging
import argparse
import datetime
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts
from core.shared.config import TUSHARE_TOKEN, MYSQL_ENABLED


def _safe_float(v):
    """NaN/None → None, 正常的 → float"""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect_daily_basic")

if not TUSHARE_TOKEN:
    logger.error("请在 .env 文件中设置 TUSHARE_TOKEN=你的token")
    sys.exit(1)
if not MYSQL_ENABLED:
    logger.error("MySQL 未启用")
    sys.exit(1)

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def _target_dates(start=None, end=None, days=None):
    """从 daily_quotes 取交易日, 排除已在 daily_basic 中的日期"""
    from core.data import Database
    sql = "SELECT DISTINCT trade_date FROM daily_quotes"
    conds, params = [], []
    if start:
        conds.append("trade_date >= %s")
        params.append(datetime.datetime.strptime(start, "%Y%m%d").date().isoformat())
    if end:
        conds.append("trade_date <= %s")
        params.append(datetime.datetime.strptime(end, "%Y%m%d").date().isoformat())
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY trade_date"
    all_dates = [r["trade_date"] for r in Database.fetchall(sql, params)]
    if days:
        all_dates = all_dates[-days:]
    done = {r["trade_date"] for r in Database.fetchall("SELECT DISTINCT trade_date FROM daily_basic WHERE pe IS NOT NULL")}
    all_need = [d for d in all_dates if d not in done]
    return all_need


def collect(start=None, end=None, days=None):
    from core.data import Database
    dates = _target_dates(start, end, days)
    if not dates:
        logger.info("[市值] 无缺失交易日, 已是最新")
        return
    logger.info(f"[市值] 待采集 {len(dates)} 个交易日 ({dates[0]}~{dates[-1]})")
    saved = 0
    for i, d in enumerate(dates):
        ymd = d.strftime("%Y%m%d")
        try:
            df = pro.daily_basic(trade_date=ymd, fields="ts_code,total_mv,circ_mv,float_share,pe,pb,turnover_rate")
            if df is None or df.empty:
                continue
            values = []
            for _, row in df.iterrows():
                code = str(row["ts_code"]).split(".")[0]
                pe = _safe_float(row.get("pe"))
                pb = _safe_float(row.get("pb"))
                float_share = _safe_float(row.get("float_share"))
                tr = _safe_float(row.get("turnover_rate"))
                values.append((code, d.isoformat(), row.get("total_mv"), row.get("circ_mv"), float_share, pe, pb, tr))
            for j in range(0, len(values), 1000):
                chunk = values[j:j + 1000]
                ph = ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s)"] * len(chunk))
                flat = [x for v in chunk for x in v]
                Database.execute(
                    f"INSERT INTO daily_basic (code, trade_date, total_mv, circ_mv, float_share, pe, pb, turnover_rate) "
                    f"VALUES {ph} "
                    f"ON DUPLICATE KEY UPDATE "
                    f"total_mv=VALUES(total_mv), circ_mv=VALUES(circ_mv), float_share=VALUES(float_share), "
                    f"pe=VALUES(pe), pb=VALUES(pb), turnover_rate=VALUES(turnover_rate)",
                    flat,
                )
            saved += len(values)
            if i % 5 == 0:
                logger.info(f"[市值] {ymd}: {len(values)} 只 → 累计 {saved} ({i+1}/{len(dates)})")
        except Exception as e:
            logger.warning(f"[市值] {ymd} 失败: {str(e)[:80]}")
        time.sleep(0.35)
    logger.info(f"[市值] 完成: {len(dates)} 天, 共写入 {saved} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TuShare daily_basic 历史市值采集")
    parser.add_argument("--start", default=None, help="起始日 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日 YYYYMMDD")
    parser.add_argument("--days", type=int, default=None, help="只采集最近 N 个交易日")
    args = parser.parse_args()
    t0 = time.time()
    collect(args.start, args.end, args.days)
    logger.info(f"全部完成, 总耗时 {time.time() - t0:.0f}s")
