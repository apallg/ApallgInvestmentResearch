"""
北向资金 + 资金流向 历史回补脚本 (独立, 不修改现有代码)
用法:
  py -m scripts.backfill_flow                      # 回补最近1年
  py -m scripts.backfill_flow --start 20240101     # 从指定日期开始
"""
import argparse
import logging
import sys
from datetime import date, timedelta

sys.path.insert(0, ".")

logger = logging.getLogger("backfill_flow")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_trading_dates(start, end):
    """获取 start~end 间的交易日列表 (TuShare → weekday 回退)"""
    try:
        from core.data.sources.tushare import get_pro
        pro = get_pro()
        if pro:
            df = pro.trade_cal(exchange="SSE", start_date=start.strftime("%Y%m%d"),
                               end_date=end.strftime("%Y%m%d"), is_open="1")
            if df is not None and not df.empty:
                dates = sorted(df["cal_date"].tolist())
                return [date.fromisoformat(d) for d in dates if d >= start.strftime("%Y%m%d")]
    except Exception as e:
        logger.warning(f"TuShare交易日历失败: {e}")

    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def backfill_capital_flow(start, end):
    """资金流向回补 (TuShare moneyflow, 支持历史日期)"""
    from core.data.sources.tushare import get_pro
    from core.data import Database
    pro = get_pro()
    if not pro:
        logger.error("TuShare不可用")
        return
    if not Database.is_available():
        logger.error("MySQL不可用")
        return

    dates = get_trading_dates(start, end)
    logger.info(f"资金流向回补: {len(dates)} 个交易日")

    total = 0
    for d in dates:
        ds = d.strftime("%Y%m%d")
        try:
            df = pro.moneyflow(trade_date=ds, limit=6000)
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                try:
                    code = r["ts_code"].split(".")[0]
                    net_amt = float(r.get("net_mf_amount", 0) or 0)
                    turnover_val = float(r.get("turnover", 0) or 0) or 1.0
                    main_pct = round(max(-100.0, min(100.0,
                        net_amt / turnover_val * 100 if abs(turnover_val) > 1 else 0)), 4)
                    Database.execute(
                        "INSERT INTO capital_flow_daily "
                        "(code, trade_date, main_net_flow, main_net_pct, turnover) "
                        "VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                        "main_net_flow=VALUES(main_net_flow), "
                        "main_net_pct=VALUES(main_net_pct), "
                        "turnover=VALUES(turnover)",
                        (code, d, net_amt, main_pct, turnover_val),
                    )
                    total += 1
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"资金流向 {ds} 失败: {e}")
            continue

        if dates.index(d) % 20 == 0:
            logger.info(f"  资金流向 {ds}: 累计{total}条")

    logger.info(f"资金流向回补完成: {total} 条")


def backfill_north_flow(start, end):
    """北向资金回补 (TuShare moneyflow_hsgt, 支持历史日期)"""
    from core.data.sources.tushare import get_pro
    from core.data import Database
    pro = get_pro()
    if not pro:
        logger.error("TuShare不可用")
        return
    if not Database.is_available():
        logger.error("MySQL不可用")
        return

    # 确保 north_flow 表存在
    Database.execute("""
        CREATE TABLE IF NOT EXISTS `north_flow` (
          id INT AUTO_INCREMENT PRIMARY KEY,
          code VARCHAR(10) NOT NULL,
          trade_date DATE NOT NULL,
          net_flow DECIMAL(16,2) DEFAULT 0,
          buy_amount DECIMAL(16,2) DEFAULT 0,
          sell_amount DECIMAL(16,2) DEFAULT 0,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_code_date (code, trade_date),
          INDEX idx_date (trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    dates = get_trading_dates(start, end)
    logger.info(f"北向资金回补: {len(dates)} 个交易日")

    total = 0
    batch = 0
    for d in dates:
        if batch >= 180:
            import time as _t; _t.sleep(65)
            batch = 0
        ds = d.strftime("%Y%m%d")
        try:
            df = pro.moneyflow_hsgt(trade_date=ds)
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                try:
                    code = r["ts_code"].split(".")[0]
                    Database.execute(
                        "INSERT INTO north_flow "
                        "(code, trade_date, net_flow, buy_amount, sell_amount) "
                        "VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                        "net_flow=VALUES(net_flow), "
                        "buy_amount=VALUES(buy_amount), "
                        "sell_amount=VALUES(sell_amount)",
                        (code, d,
                         float(r.get("net_amount", 0) or 0),
                         float(r.get("buy_amount", 0) or 0),
                         float(r.get("sell_amount", 0) or 0)),
                    )
                    total += 1
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"北向资金 {ds} 失败: {e}")
            continue
        batch += 1

        if dates.index(d) % 20 == 0:
            logger.info(f"  北向资金 {ds}: 累计{total}条")

    logger.info(f"北向资金回补完成: {total} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="北向资金+资金流向历史回补")
    parser.add_argument("--start", default=(date.today() - timedelta(days=365)).strftime("%Y%m%d"),
                        help="起始日期 YYYYMMDD, 默认1年前")
    parser.add_argument("--end", default=date.today().strftime("%Y%m%d"),
                        help="结束日期 YYYYMMDD, 默认今天")
    parser.add_argument("--skip-capital", action="store_true", help="跳过资金流向")
    parser.add_argument("--skip-north", action="store_true", help="跳过北向资金")
    args = parser.parse_args()

    start = date.fromisoformat(args.start[:4]+"-"+args.start[4:6]+"-"+args.start[6:8])
    end = date.fromisoformat(args.end[:4]+"-"+args.end[4:6]+"-"+args.end[6:8])

    if not args.skip_capital:
        backfill_capital_flow(start, end)
    if not args.skip_north:
        backfill_north_flow(start, end)
