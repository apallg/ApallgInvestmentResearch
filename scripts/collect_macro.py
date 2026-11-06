"""
Tushare 宏观数据采集 — shibor/lpr/cn_m/cn_cpi/cn_ppi/cn_pmi
用法:
  python -m scripts.collect_macro                    # 全部采集(增量)
  python -m scripts.collect_macro --tables shibor    # 仅Shibor
  python -m scripts.collect_macro --backfill 201901  # 历史回补
积分需求: shibor/lpr=120分, cn_*=2000分
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from time import sleep
import pandas as pd

sys.path.insert(0, ".")

from core.shared.config import TUSHARE_TOKEN, MYSQL_ENABLED
from core.data.sources.tushare import get_pro

logger = logging.getLogger("collect_macro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if not TUSHARE_TOKEN:
    logger.error("请在 .env 文件中设置 TUSHARE_TOKEN")
    sys.exit(1)


def _execute(sql, params=None):
    from core.data import execute_sql
    execute_sql(sql, params)


def _exists(table, date_col, date_val):
    from core.data import Database
    rows = Database.fetchall(f"SELECT 1 FROM {table} WHERE {date_col}=%s LIMIT 1", (date_val,))
    return bool(rows)


def _upsert(table, row, unique_col="trade_date"):
    columns = list(row.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    updates = ", ".join(f"{c}=VALUES({c})" for c in columns if c != unique_col)
    sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updates}"
    _execute(sql, tuple(row.values()))


# ============================================================
# Shibor (日频, 120积分)
# ============================================================
def collect_shibor(start_date=None, end_date=None):
    pro = get_pro()
    if pro is None:
        logger.error("Tushare 不可用")
        return 0

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    saved = 0
    offset = 0
    while True:
        try:
            df = pro.shibor(start_date=start_date, end_date=end_date, limit=5000, offset=offset)
        except Exception as e:
            logger.warning(f"shibor 第{offset}批失败: {e}")
            break
        if df is None or df.empty:
            break
        for _, r in df.iterrows():
            row = {"trade_date": r["date"], "on_rate": _f(r, "on"), "w1_rate": _f(r, "1w"),
                   "w2_rate": _f(r, "2w"), "m1_rate": _f(r, "1m"), "m3_rate": _f(r, "3m"),
                   "m6_rate": _f(r, "6m"), "m9_rate": _f(r, "9m"), "y1_rate": _f(r, "1y")}
            _upsert("macro_shibor", row)
            saved += 1
        if len(df) < 5000:
            break
        offset += 5000
        sleep(0.3)

    logger.info(f"[shibor] {saved} 条 ({start_date}~{end_date})")
    return saved


# ============================================================
# LPR (日频, 120积分)
# ============================================================
def collect_lpr(start_date=None, end_date=None):
    pro = get_pro()
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    saved = 0
    observed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if pro is not None:
        try:
            sleep(0.35)
            df = pro.shibor_lpr(start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                for _, r in df.iterrows():
                    row = {
                        "trade_date": r["date"], "lpr_1y": _f(r, "1y"), "lpr_5y": _f(r, "5y"),
                        "source_name": "tushare", "source_url": None, "source_observed_at": observed_at,
                    }
                    _upsert("macro_lpr", row)
                    saved += 1
        except Exception as e:
            logger.warning(f"Tushare lpr 失败: {e}")
    try:
        from core.data.sources.chinamoney import fetch_lpr_announcements
        for record in fetch_lpr_announcements():
            ymd = record["trade_date"].replace("-", "")
            if start_date <= ymd <= end_date:
                _upsert("macro_lpr", {
                    "trade_date": record["trade_date"], "lpr_1y": record["lpr_1y"],
                    "lpr_5y": record["lpr_5y"], "source_name": "chinamoney",
                    "source_url": record["source_url"], "source_observed_at": observed_at,
                })
                saved += 1
    except Exception as e:
        logger.warning(f"中国货币网 LPR 回退源失败: {e}")
    logger.info(f"[lpr] {saved} 条 ({start_date}~{end_date})")
    return saved


# ============================================================
# cn_m 货币供应量 (月频, 2000积分)
# ============================================================
def collect_money_supply(start_m=None, end_m=None):
    pro = get_pro()
    if pro is None:
        return 0

    if end_m is None:
        end_m = datetime.now().strftime("%Y%m")
    if start_m is None:
        start_m = (datetime.now() - timedelta(days=365*5)).strftime("%Y%m")

    saved = 0
    try:
        df = pro.cn_m(start_m=start_m, end_m=end_m, fields="month,m0,m0_yoy,m1,m1_yoy,m2,m2_yoy")
    except Exception as e:
        logger.warning(f"cn_m 失败: {e}")
        return 0
    if df is None or df.empty:
        return 0

    for _, r in df.iterrows():
        month_str = str(r["month"])
        if len(month_str) == 6:
            trade_date = f"{month_str[:4]}-{month_str[4:6]}-01"
            # 推到月末
            import calendar
            y, m = int(month_str[:4]), int(month_str[4:6])
            last_day = calendar.monthrange(y, m)[1]
            trade_date = f"{y:04d}-{m:02d}-{last_day:02d}"
        else:
            trade_date = str(r["month"])

        row = {"trade_date": trade_date, "m1": _f(r, "m1"), "m1_yoy": _f(r, "m1_yoy"),
               "m2": _f(r, "m2"), "m2_yoy": _f(r, "m2_yoy")}
        _upsert("macro_money_supply", row)
        saved += 1

    logger.info(f"[money_supply] {saved} 条 ({start_m}~{end_m})")
    return saved


# ============================================================
# cn_cpi (月频, 2000积分)
# ============================================================
def collect_cpi(start_m=None, end_m=None):
    pro = get_pro()
    if pro is None:
        return 0

    if end_m is None:
        end_m = datetime.now().strftime("%Y%m")
    if start_m is None:
        start_m = (datetime.now() - timedelta(days=365*5)).strftime("%Y%m")

    saved = 0
    try:
        df = pro.cn_cpi(start_m=start_m, end_m=end_m, fields="month,nt_yoy,nt_mom,cnt_yoy")
    except Exception as e:
        logger.warning(f"cn_cpi 失败: {e}")
        return 0
    if df is None or df.empty:
        return 0

    for _, r in df.iterrows():
        month_str = str(r["month"])
        trade_date = f"{month_str[:4]}-{month_str[4:6]}-01" if len(month_str) >= 6 else str(r["month"])
        row = {"trade_date": trade_date, "cpi_yoy": _f(r, "nt_yoy"), "cpi_mom": _f(r, "nt_mom"),
               "core_cpi_yoy": _f(r, "cnt_yoy"), "food_yoy": None}
        _upsert("macro_cpi", row)
        saved += 1

    logger.info(f"[cpi] {saved} 条 ({start_m}~{end_m})")
    return saved


# ============================================================
# cn_ppi (月频, 2000积分)
# ============================================================
def collect_ppi(start_m=None, end_m=None):
    pro = get_pro()
    if pro is None:
        return 0

    if end_m is None:
        end_m = datetime.now().strftime("%Y%m")
    if start_m is None:
        start_m = (datetime.now() - timedelta(days=365*5)).strftime("%Y%m")

    saved = 0
    try:
        df = pro.cn_ppi(start_m=start_m, end_m=end_m, fields="month,ppi_yoy,ppi_mom,ppi_mp_yoy")
    except Exception as e:
        logger.warning(f"cn_ppi 失败: {e}")
        return 0
    if df is None or df.empty:
        return 0

    for _, r in df.iterrows():
        month_str = str(r["month"])
        trade_date = f"{month_str[:4]}-{month_str[4:6]}-01" if len(month_str) >= 6 else str(r["month"])
        row = {"trade_date": trade_date, "ppi_yoy": _f(r, "ppi_yoy"),
               "ppi_mom": _f(r, "ppi_mom"), "ppi_mp_yoy": _f(r, "ppi_mp_yoy")}
        _upsert("macro_ppi", row)
        saved += 1

    logger.info(f"[ppi] {saved} 条 ({start_m}~{end_m})")
    return saved


# ============================================================
# cn_pmi (月频, 2000积分)
# ============================================================
def collect_pmi(start_m=None, end_m=None):
    pro = get_pro()
    if pro is None:
        return 0

    if end_m is None:
        end_m = datetime.now().strftime("%Y%m")
    if start_m is None:
        start_m = (datetime.now() - timedelta(days=365*5)).strftime("%Y%m")

    saved = 0
    try:
        df = pro.cn_pmi(start_m=start_m, end_m=end_m, fields="month,pmi010000,pmi010500,pmi010800,pmi011200,pmi020100")
    except Exception as e:
        logger.warning(f"cn_pmi 失败: {e}")
        return 0
    if df is None or df.empty:
        return 0

    for _, r in df.iterrows():
        month_str = str(r["month"])
        trade_date = f"{month_str[:4]}-{month_str[4:6]}-01" if len(month_str) >= 6 else str(r["month"])
        row = {"trade_date": trade_date, "pmi": _f(r, "pmi010000"),
               "pmi_new_order": _f(r, "pmi010500"),
               "pmi_employment": _f(r, "pmi010800"),
               "pmi_input_price": _f(r, "pmi011200"),
               "pmi_non_mfg": _f(r, "pmi020100")}
        _upsert("macro_pmi", row)
        saved += 1

    logger.info(f"[pmi] {saved} 条 ({start_m}~{end_m})")
    return saved


def _f(row, key):
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return float(v)


COLLECTORS = {
    "shibor": collect_shibor,
    "lpr": collect_lpr,
    "money": collect_money_supply,
    "cpi": collect_cpi,
    "ppi": collect_ppi,
    "pmi": collect_pmi,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tushare 宏观数据采集")
    parser.add_argument("--tables", default="all", help="逗号分隔, all=全部, 可选: shibor,lpr,money,cpi,ppi,pmi")
    parser.add_argument("--backfill", help="历史回补起始月 (YYYYMM)")
    args = parser.parse_args()

    if args.tables == "all":
        targets = list(COLLECTORS.keys())
    else:
        targets = [t.strip() for t in args.tables.split(",") if t.strip() in COLLECTORS]

    start = args.backfill
    for name in targets:
        func = COLLECTORS[name]
        try:
            if start and name not in ("shibor", "lpr"):
                func(start_m=start)
            elif start:
                func(start_date=start)
            else:
                func()
        except Exception as e:
            logger.error(f"[{name}] 采集失败: {e}")
