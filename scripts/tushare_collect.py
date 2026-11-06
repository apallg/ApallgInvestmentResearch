"""
Tushare 数据采集脚本 — 一次性拉取全市场基本面/资金流向/北向资金, 写入 MySQL

用法:
  python -m scripts.tushare_collect                    # 全量采集(最近4个季度财报+最近60天资金流)
  python -m scripts.tushare_collect --tables fund      # 只采集基本面
  python -m scripts.tushare_collect --tables flow      # 只采集资金流向
  python -m scripts.tushare_collect --tables north     # 只采集北向资金
"""
import sys
import os
import time
import logging
import argparse
import datetime
import threading
import hashlib
from decimal import Decimal, InvalidOperation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts
import pandas as pd
from core.shared.config import TUSHARE_TOKEN, MYSQL_ENABLED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tushare_collect")

if not TUSHARE_TOKEN:
    logger.error("请在 .env 文件中设置 TUSHARE_TOKEN=你的token")
    sys.exit(1)

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def ts_code_to_symbol(ts_code):
    """600519.SH → 600519"""
    return ts_code.split(".")[0] if "." in str(ts_code) else str(ts_code)


def _same_numeric_values(left, right):
    if left is None or pd.isna(left):
        return right is None or pd.isna(right)
    if right is None or pd.isna(right):
        return False
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return str(left) == str(right)


def collect_fundamentals():
    if not MYSQL_ENABLED:
        logger.error("MySQL 未启用")
        return {"saved": 0, "tushare": 0, "akshare": 0}
    from backend.service.fundamental_collection import fetch_and_save_fundamentals
    result = fetch_and_save_fundamentals()
    logger.info("[基本面] 采集完成: %s", result)
    return result


def _is_rate_limit_error(err):
    """检测 Tushare 频率超限错误"""
    msg = str(err).lower()
    return "频率超限" in msg or "200次" in msg or "rate limit" in msg


def _rate_limited_sleep(last_call_time, min_interval=0.35):
    """确保两次 Tushare API 调用间隔 >= min_interval 秒，防止频率超限"""
    elapsed = time.time() - last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)


def collect_forecast():
    """采集分析师一致预期: forecast → forecast_data"""
    from core.data import Database
    if not MYSQL_ENABLED:
        logger.error("MySQL 未启用")
        return

    Database.execute("""
        CREATE TABLE IF NOT EXISTS `forecast_data` (
          id INT AUTO_INCREMENT PRIMARY KEY,
          code VARCHAR(10) NOT NULL,
          end_date DATE NOT NULL,
          ann_date DATE DEFAULT NULL,
          net_profit_min DECIMAL(20,4) DEFAULT NULL,
          net_profit_max DECIMAL(20,4) DEFAULT NULL,
          p_change_min DECIMAL(10,4) DEFAULT NULL,
          p_change_max DECIMAL(10,4) DEFAULT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_code_enddate (code, end_date),
          INDEX idx_code (code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    logger.info("[一致预期] 获取股票列表...")
    try:
        stock_df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")
        all_codes = stock_df["ts_code"].tolist()
    except Exception as e:
        logger.error(f"[一致预期] 获取股票列表失败: {e}")
        return

    saved = 0
    last_api_time = time.time()
    MAX_RETRIES = 3
    MIN_INTERVAL = 0.15

    for i, ts_code in enumerate(all_codes):
        for retry in range(MAX_RETRIES):
            try:
                _rate_limited_sleep(last_api_time, MIN_INTERVAL)
                df = pro.forecast(ts_code=ts_code,
                                  fields="ts_code,ann_date,end_date,net_profit_min,net_profit_max,p_change_min,p_change_max",
                                  limit=50)
                last_api_time = time.time()
                if df is None or df.empty:
                    break
                batch_saved = 0
                for _, row in df.iterrows():
                    try:
                        code = ts_code_to_symbol(row["ts_code"])
                        Database.execute(
                            "INSERT INTO forecast_data (code, end_date, ann_date, net_profit_min, net_profit_max, p_change_min, p_change_max) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                            "ON DUPLICATE KEY UPDATE "
                            "ann_date=VALUES(ann_date), net_profit_min=VALUES(net_profit_min), net_profit_max=VALUES(net_profit_max), "
                            "p_change_min=VALUES(p_change_min), p_change_max=VALUES(p_change_max)",
                            (code, row.get("end_date"), row.get("ann_date"),
                             row.get("net_profit_min"), row.get("net_profit_max"),
                             row.get("p_change_min"), row.get("p_change_max")),
                        )
                        batch_saved += 1
                    except Exception:
                        pass
                saved += batch_saved
                break
            except Exception as e:
                last_api_time = time.time()
                err_str = str(e)[:80]
                if _is_rate_limit_error(e) and retry < MAX_RETRIES - 1:
                    wait = 65 + retry * 10
                    logger.warning(f"[一致预期] 频率超限，等待 {wait}s 后重试 ({ts_code}, 重试{retry+1})")
                    time.sleep(wait)
                    continue
                break

        if (i + 1) % 200 == 0:
            logger.info(f"[一致预期] 进度 {i+1}/{len(all_codes)}: 累计 {saved} 条")

    logger.info(f"[一致预期] 完成: 共写入 {saved} 条")


def collect_income():
    """采集利润表关键字段: income → income_data"""
    from core.data import Database
    if not MYSQL_ENABLED:
        logger.error("MySQL 未启用")
        return

    Database.execute("""
        CREATE TABLE IF NOT EXISTS `income_data` (
          id INT AUTO_INCREMENT PRIMARY KEY,
          code VARCHAR(10) NOT NULL,
          end_date DATE NOT NULL,
          ann_date DATE DEFAULT NULL,
          operate_profit DECIMAL(20,4) DEFAULT NULL,
          total_revenue DECIMAL(20,4) DEFAULT NULL,
          oper_cost DECIMAL(20,4) DEFAULT NULL,
          sell_exp DECIMAL(20,4) DEFAULT NULL,
          admin_exp DECIMAL(20,4) DEFAULT NULL,
          fin_exp DECIMAL(20,4) DEFAULT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_code_enddate (code, end_date),
          INDEX idx_code (code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    Database.execute("""
        CREATE TABLE IF NOT EXISTS `income_data_source_variants` (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          variant_key CHAR(64) NOT NULL,
          code VARCHAR(10) NOT NULL,
          end_date DATE NOT NULL,
          ann_date DATE DEFAULT NULL,
          report_type VARCHAR(8) DEFAULT NULL,
          comp_type VARCHAR(8) DEFAULT NULL,
          payload JSON NOT NULL,
          collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_variant_key (variant_key),
          INDEX idx_code_end_date (code, end_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    from core.data.income_data_migration import ensure_income_data_version_fields
    ensure_income_data_version_fields()

    logger.info("[利润表] 获取股票列表...")
    try:
        stock_df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
        all_codes = stock_df["ts_code"].tolist()
        industries = {
            row["ts_code"]: "" if pd.isna(row.get("industry")) else str(row.get("industry"))
            for _, row in stock_df.iterrows()
        }
    except Exception as e:
        logger.error(f"[利润表] 获取股票列表失败: {e}")
        return

    saved = 0
    write_errors = 0
    write_error_samples = []
    fetch_errors = 0
    fetch_error_samples = []
    fetch_error_codes = []
    last_api_time = time.time()
    MAX_RETRIES = 3
    MIN_INTERVAL = 0.32

    for i, ts_code in enumerate(all_codes):
        for retry in range(MAX_RETRIES):
            try:
                _rate_limited_sleep(last_api_time, MIN_INTERVAL)
                df = pro.income(ts_code=ts_code,
                                fields="ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,update_flag,operate_profit,total_revenue,oper_cost,sell_exp,admin_exp,fin_exp",
                                limit=50)
                last_api_time = time.time()
                if df is None or df.empty:
                    break
                batch_saved = 0
                by_end_date = {}
                industry = industries.get(ts_code, "")
                if "银行" in industry:
                    expected_comp_type = "2"
                elif "保险" in industry:
                    expected_comp_type = "3"
                elif "证券" in industry:
                    expected_comp_type = "4"
                elif any(kind in industry for kind in ("多元金融", "信托", "租赁")):
                    expected_comp_type = "7"
                elif industry:
                    expected_comp_type = "1"
                else:
                    expected_comp_type = None
                available_comp_types = set()
                for _, row in df.iterrows():
                    try:
                        code = ts_code_to_symbol(row["ts_code"])
                        raw_payload = row.to_json(date_format="iso", date_unit="s", force_ascii=False)
                        variant_key = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
                        Database.execute(
                            "INSERT IGNORE INTO income_data_source_variants "
                            "(variant_key,code,end_date,ann_date,report_type,comp_type,payload) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (variant_key, code, row.get("end_date"), row.get("ann_date"),
                             row.get("report_type"), row.get("comp_type"), raw_payload),
                        )
                        if str(row.get("report_type")) == "1":
                            if not pd.isna(row.get("comp_type")):
                                comp_type = str(row.get("comp_type"))
                                available_comp_types.add(comp_type)
                                if comp_type == expected_comp_type:
                                    by_end_date.setdefault(row.get("end_date"), []).append(row)
                    except Exception as e:
                        write_errors += 1
                        if len(write_error_samples) < 5:
                            write_error_samples.append(
                                f"{ts_code}/{row.get('end_date')}: {type(e).__name__}: {str(e)[:160]}"
                            )

                ambiguous = []
                if expected_comp_type is None or expected_comp_type not in available_comp_types:
                    logger.error(
                        f"[利润表] 无法用公司行业确定合并报表 comp_type: {ts_code}, "
                        f"industry={industry!r}, expected={expected_comp_type}, available={sorted(available_comp_types)}; "
                        "已保留供应商原始版本并跳过规范表写入"
                    )
                for end_date, rows in by_end_date.items():
                    if ambiguous:
                        continue
                    available_f_ann_dates = [str(row.get("f_ann_date")) for row in rows
                                             if not pd.isna(row.get("f_ann_date"))]
                    if not available_f_ann_dates:
                        ambiguous.append(str(end_date))
                        continue
                    latest_f_ann_date = max(available_f_ann_dates)
                    rows = [row for row in rows
                            if not pd.isna(row.get("f_ann_date"))
                            and str(row.get("f_ann_date")) == latest_f_ann_date]
                    unique_rows = {}
                    for row in rows:
                        values_key = tuple(
                            None if pd.isna(row.get(field)) else str(row.get(field))
                            for field in ("ann_date", "operate_profit", "total_revenue", "oper_cost", "sell_exp", "admin_exp", "fin_exp")
                        )
                        unique_rows.setdefault(values_key, row)
                    if len(unique_rows) != 1:
                        ambiguous.append(str(end_date))
                        continue
                    row = max(unique_rows.values(), key=lambda value: str(value.get("update_flag") or ""))
                    try:
                        code = ts_code_to_symbol(row["ts_code"])
                        existing = Database.fetchone(
                            "SELECT f_ann_date,operate_profit,total_revenue,oper_cost,sell_exp,admin_exp,fin_exp "
                            "FROM income_data WHERE code=%s AND end_date=%s",
                            (code, row.get("end_date")),
                        )
                        if existing and existing.get("f_ann_date") is None:
                            logger.warning(
                                "[利润表] %s/%s 旧规范行缺少版本证据；已保留源版本，不自动覆盖，请进入候选核验",
                                code, row.get("end_date"),
                            )
                            continue
                        if existing:
                            incoming_f_ann_date = str(row.get("f_ann_date")).replace("-", "")[:8]
                            existing_f_ann_date = str(existing["f_ann_date"]).replace("-", "")[:8]
                            if incoming_f_ann_date <= existing_f_ann_date:
                                fields = ("operate_profit", "total_revenue", "oper_cost", "sell_exp", "admin_exp", "fin_exp")
                                if incoming_f_ann_date == existing_f_ann_date and any(
                                    not _same_numeric_values(existing.get(field), row.get(field)) for field in fields
                                ):
                                    logger.error(
                                        "[利润表] 同一实际公告日存在数据库/来源冲突，已保留源版本且不覆盖: %s/%s/%s",
                                        code, row.get("end_date"), incoming_f_ann_date,
                                    )
                                continue
                        values = (
                            code, row.get("end_date"), row.get("ann_date"), row.get("f_ann_date"), row.get("update_flag"),
                            row.get("operate_profit"), row.get("total_revenue"),
                            row.get("oper_cost"), row.get("sell_exp"), row.get("admin_exp"), row.get("fin_exp"),
                        )
                        values = tuple(None if pd.isna(value) else value for value in values)
                        Database.execute(
                            "INSERT INTO income_data (code, end_date, ann_date, f_ann_date, update_flag, operate_profit, total_revenue, oper_cost, sell_exp, admin_exp, fin_exp) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                            "ON DUPLICATE KEY UPDATE "
                            "operate_profit=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(operate_profit),operate_profit),"
                            "total_revenue=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(total_revenue),total_revenue),"
                            "oper_cost=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(oper_cost),oper_cost),"
                            "sell_exp=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(sell_exp),sell_exp),"
                            "admin_exp=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(admin_exp),admin_exp),"
                            "fin_exp=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(fin_exp),fin_exp),"
                            "update_flag=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(update_flag),update_flag),"
                            "f_ann_date=IF(income_data.f_ann_date IS NOT NULL AND VALUES(f_ann_date)>income_data.f_ann_date,VALUES(f_ann_date),f_ann_date)",
                            values,
                        )
                        batch_saved += 1
                    except Exception as e:
                        write_errors += 1
                        if len(write_error_samples) < 5:
                            write_error_samples.append(
                                f"{ts_code}/{row.get('end_date')}: {type(e).__name__}: {str(e)[:160]}"
                            )
                if ambiguous:
                    logger.error(f"[利润表] 同期合并口径冲突，已保留原库且不覆盖 {ts_code}: {ambiguous}")
                saved += batch_saved
                break
            except Exception as e:
                last_api_time = time.time()
                err_str = str(e)[:80]
                if _is_rate_limit_error(e) and retry < MAX_RETRIES - 1:
                    wait = 65 + retry * 10
                    logger.warning(f"[利润表] 频率超限，等待 {wait}s 后重试 ({ts_code}, 重试{retry+1})")
                    time.sleep(wait)
                    continue
                if type(e).__name__ in ("ReadTimeout", "ConnectTimeout", "ConnectionError") and retry < MAX_RETRIES - 1:
                    wait = 3 * (retry + 1)
                    logger.warning(f"[利润表] 网络请求失败，等待 {wait}s 后重试 ({ts_code}, 重试{retry+1})")
                    time.sleep(wait)
                    continue
                fetch_errors += 1
                fetch_error_codes.append(ts_code)
                if len(fetch_error_samples) < 5:
                    fetch_error_samples.append(f"{ts_code}: {type(e).__name__}: {str(e)[:160]}")
                break

        if (i + 1) % 200 == 0:
            logger.info(f"[利润表] 进度 {i+1}/{len(all_codes)}: 累计 {saved} 条")

    logger.info(f"[利润表] 完成: 共写入 {saved} 条, 写入失败 {write_errors} 条, 数据源失败 {fetch_errors} 只")
    if write_error_samples:
        logger.warning(f"[利润表] 写入失败样例: {write_error_samples}")
    if fetch_error_samples:
        logger.warning(f"[利润表] 数据源失败样例: {fetch_error_samples}")
    if fetch_error_codes:
        logger.warning(f"[利润表] 数据源失败证券列表: {fetch_error_codes}")


def collect_moneyflow(days=60):
    """采集资金流向: moneyflow → capital_flow_daily
    只传 trade_date + limit=6000 即可返回全市场,每天1次调用。"""
    from core.data import Database
    if not MYSQL_ENABLED:
        logger.error("MySQL 未启用")
        return

    try:
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=days*3)).strftime("%Y%m%d")
        cal_df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
        trade_dates = cal_df[cal_df["is_open"] == 1]["cal_date"].tolist()[-days:]
    except Exception:
        trade_dates = [(datetime.date.today() - datetime.timedelta(days=i)).strftime("%Y%m%d")
                       for i in range(days * 2)][::-1]

    logger.info(f"[资金流向] 开始采集 {len(trade_dates)} 天 ({trade_dates[0]}~{trade_dates[-1]}, 每天1次调用)...")
    saved = 0
    for i, d in enumerate(trade_dates):
        try:
            df = pro.moneyflow(trade_date=d, limit=6000)
            if df is None or df.empty:
                continue
            day_saved = 0
            for _, row in df.iterrows():
                try:
                    code = row["ts_code"].split(".")[0]
                    Database.execute(
                        "INSERT INTO capital_flow_daily "
                        "(code, trade_date, main_net_flow, main_net_pct, turnover) "
                        "VALUES (%s,%s,%s,%s,%s) "
                        "ON DUPLICATE KEY UPDATE "
                        "main_net_flow=VALUES(main_net_flow), "
                        "main_net_pct=VALUES(main_net_pct), "
                        "turnover=VALUES(turnover)",
                        (code, d,
                         row.get("net_mf_amount", 0) or 0,
                         row.get("net_mf_vol", 0) or 0,
                         row.get("turnover", 0)),
                    )
                    day_saved += 1
                except Exception:
                    pass
            saved += day_saved
            if i % 5 == 0:
                logger.info(f"[资金流向] {d}: {day_saved} 只 → 累计 {saved} ({i+1}/{len(trade_dates)})")
        except Exception as e:
            err = str(e)[:80]
            if i % 5 == 0:
                logger.warning(f"[资金流向] {d} 失败: {err}")
        time.sleep(0.35)

    logger.info(f"[资金流向] 完成: {len(trade_dates)} 天, 共写入 {saved} 条")


def collect_north_flow(days=90):
    """采集北向资金: moneyflow_hsgt → north_flow

    moneyflow_hsgt 是市场级汇总,只返回总体净流入/买入/卖出,不需要 ts_code。
    存为 code='MARKET' 代表全市场北向资金。
    """
    from core.data import Database
    if not MYSQL_ENABLED:
        logger.error("MySQL 未启用")
        return

    try:
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=days*3)).strftime("%Y%m%d")
        cal_df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
        trade_dates = cal_df[cal_df["is_open"] == 1]["cal_date"].tolist()[-days:]
    except Exception:
        trade_dates = [(datetime.date.today() - datetime.timedelta(days=i)).strftime("%Y%m%d")
                       for i in range(days * 2)][::-1]

    logger.info(f"[北向资金] 开始采集 {len(trade_dates)} 天 ({trade_dates[0]}~{trade_dates[-1]})...")
    saved = 0
    for i, d in enumerate(trade_dates):
        try:
            # moneyflow_hsgt 返回市场汇总: buy/sell/net/net_rate
            df = pro.moneyflow_hsgt(trade_date=d)
            if df is None or df.empty:
                continue
            row = df.iloc[0]
            net = float(row.get("net", 0) or 0)
            buy_amt = float(row.get("buy", 0) or 0)
            sell_amt = float(row.get("sell", 0) or 0)
            # 市场汇总存入 code='MARKET'; 个股北向通过 hk_hold 明细获取
            Database.execute(
                "INSERT INTO north_flow (code, trade_date, net_flow) "
                "VALUES (%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE net_flow=VALUES(net_flow)",
                ("MARKET", d, net),
            )
            saved += 1
            if i % 5 == 0:
                logger.info(f"[北向资金] {d}: 净流入{net/1e8:.1f}亿 ({i+1}/{len(trade_dates)})")
        except Exception as e:
            err = str(e)[:80]
            if i % 5 == 0:
                logger.warning(f"[北向资金] {d} 失败: {err}")
        time.sleep(0.3)

    logger.info(f"[北向资金] 完成: 共写入 {saved} 条")


def collect_analyst_forecast():
    from backend.service.analyst_forecast_collection import collect_latest_analyst_forecast_reports
    return collect_latest_analyst_forecast_reports()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tushare 数据采集")
    parser.add_argument("--tables", default="all",
                       help="采集哪些表: all / fund / flow / north")
    args = parser.parse_args()

    t0 = time.time()
    if args.tables in ("all", "fund"):
        collect_fundamentals()
    if args.tables in ("all", "flow"):
        collect_moneyflow()
    if args.tables in ("all", "north"):
        collect_north_flow()
    logger.info(f"全部完成, 总耗时 {time.time()-t0:.0f}s")
