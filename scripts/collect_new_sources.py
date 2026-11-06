"""
新增数据源采集：融资融券、龙虎榜、十大股东。
优先 QMT/xtdata，降级 Tushare。
"""
import logging
import calendar
import math
from time import sleep
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _get_pro():
    from core.data.sources.tushare import get_pro
    return get_pro()


def _integer_or_none(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


# ═══════════════════════════════════════════════════════════════
# QMT 辅助
# ═══════════════════════════════════════════════════════════════

_qmt_available = None


def _qmt_xtdata():
    global _qmt_available
    if _qmt_available is False:
        return None
    try:
        from xtquant import xtdata
        _qmt_available = True
        return xtdata
    except ImportError:
        _qmt_available = False
        return None


# ═══════════════════════════════════════════════════════════════
# 1. 融资融券
# ═══════════════════════════════════════════════════════════════

def collect_margin_daily(trade_date=None):
    """
    采集融资融券日频数据。
    QMT 优先 (get_market_data_ex 含 margin 字段)，降级 Tushare pro.margin_detail。
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y%m%d")

    xtdata = _qmt_xtdata()
    if xtdata:
        try:
            n = _collect_margin_via_qmt(xtdata, trade_date)
            if n > 0:
                return n
        except Exception as e:
            logger.warning(f"QMT 融资融券采集失败，降级 Tushare: {e}")

    return _collect_margin_via_tushare(trade_date)


def _collect_margin_via_qmt(xtdata, trade_date):
    """QMT 采集融资融券 — 通过财务数据字段获取"""
    from core.data import Database

    field_list = [
        "marginBalance", "marginBuy", "marginRepay",
        "shortVolume", "shortBalance", "shortSell", "shortRepay",
        "totalBalance",
    ]
    try:
        raw = xtdata.get_market_data_ex(
            stock_list=[],
            period="1d",
            start_time=trade_date,
            end_time=trade_date,
            field_list=field_list,
        )
    except Exception as e:
        logger.debug(f"QMT margin 字段不可用: {e}")
        return 0

    if raw is None:
        return 0

    saved = 0
    rows = []
    if isinstance(raw, dict):
        for symbol, df in raw.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            code = symbol.split(".")[0] if "." in symbol else symbol
            for _, r in df.iterrows():
                rows.append((trade_date, code,
                             _integer_or_none(r.get("marginBalance")),
                             _integer_or_none(r.get("marginBuy")),
                             _integer_or_none(r.get("marginRepay")),
                             _integer_or_none(r.get("shortVolume")),
                             _integer_or_none(r.get("shortBalance")),
                             _integer_or_none(r.get("shortSell")),
                             _integer_or_none(r.get("shortRepay")),
                             _integer_or_none(r.get("totalBalance")),
                             "QMT"))

    if rows:
        Database.executemany(
            "INSERT INTO margin_daily (trade_date, stock_code, margin_balance, margin_buy, "
            "margin_repay, short_volume, short_balance, short_sell, short_repay, total_balance, source) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE margin_balance=VALUES(margin_balance), "
            "margin_buy=VALUES(margin_buy), margin_repay=VALUES(margin_repay), "
            "short_volume=VALUES(short_volume), short_balance=VALUES(short_balance), "
            "short_sell=VALUES(short_sell), short_repay=VALUES(short_repay), "
            "total_balance=VALUES(total_balance), source=VALUES(source)",
            rows)
        saved = len(rows)

    logger.info(f"[融资融券] QMT: {saved} 条 ({trade_date})")
    return saved


def _collect_margin_via_tushare(trade_date):
    """Tushare 采集融资融券"""
    from core.data import Database
    from core.data.sources.tushare import _rate_limit

    pro = _get_pro()
    if not pro:
        logger.warning("[融资融券] Tushare 不可用")
        return 0

    _rate_limit()
    sleep(0.35)
    try:
        df = pro.margin_detail(trade_date=trade_date)
    except Exception as e:
        logger.warning(f"[融资融券] Tushare 失败: {e}")
        return 0

    if df is None or df.empty:
        return 0

    saved = 0
    rows = []
    for _, r in df.iterrows():
        code = str(r.get("ts_code", "")).split(".")[0] if "." in str(r.get("ts_code", "")) else str(r.get("ts_code", ""))
        if not code:
            continue
        rows.append((trade_date, code,
                     _integer_or_none(r.get("rzye")),
                     _integer_or_none(r.get("rzmre")),
                     _integer_or_none(r.get("rzche")),
                     _integer_or_none(r.get("rqyl")),
                     _integer_or_none(r.get("rqye")),
                     _integer_or_none(r.get("rqmcl")),
                     _integer_or_none(r.get("rqchl")),
                     _integer_or_none(r.get("rzrqye")),
                     "TUSHARE"))
        saved += 1

    if rows:
        Database.executemany(
            "INSERT INTO margin_daily (trade_date, stock_code, margin_balance, margin_buy, "
            "margin_repay, short_volume, short_balance, short_sell, short_repay, total_balance, source) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE margin_balance=VALUES(margin_balance), "
            "margin_buy=VALUES(margin_buy), margin_repay=VALUES(margin_repay), "
            "short_volume=VALUES(short_volume), short_balance=VALUES(short_balance), "
            "short_sell=VALUES(short_sell), short_repay=VALUES(short_repay), "
            "total_balance=VALUES(total_balance), source=VALUES(source)",
            rows)

    logger.info(f"[融资融券] Tushare: {saved} 条 ({trade_date})")
    return saved


# ═══════════════════════════════════════════════════════════════
# 2. 龙虎榜
# ═══════════════════════════════════════════════════════════════

def collect_lhb_daily(trade_date=None):
    """
    采集龙虎榜数据 (Tushare pro.top_list)。
    QMT 无直接龙虎榜 API。
    """
    from core.data import Database
    from core.data.sources.tushare import _rate_limit

    if trade_date is None:
        trade_date = datetime.now().strftime("%Y%m%d")

    pro = _get_pro()
    if not pro:
        logger.warning("[龙虎榜] Tushare 不可用")
        return 0

    _rate_limit()
    sleep(0.35)
    try:
        df = pro.top_list(trade_date=trade_date)
    except Exception as e:
        logger.warning(f"[龙虎榜] Tushare 失败: {e}")
        return 0

    if df is None or df.empty:
        return 0

    saved = 0
    rows = []
    for _, r in df.iterrows():
        code = str(r.get("ts_code", "")).split(".")[0] if "." in str(r.get("ts_code", "")) else str(r.get("ts_code", ""))
        if not code:
            continue

        def _f(key):
            v = r.get(key)
            return float(v) if v is not None and str(v) != "nan" else None

        rows.append((trade_date, code, str(r.get("name", "")),
                     _f("close"), _f("pct_change"),
                     int(r.get("l_buy", 0) or 0) - int(r.get("l_sell", 0) or 0),
                     int(r.get("l_buy", 0) or 0),
                     int(r.get("l_sell", 0) or 0),
                     int(r.get("amount", 0) or 0),
                     None, None, None,
                     _f("turnover_rate"), _f("float_mv"),
                     None, None, None, None,
                     str(r.get("reason", "")), None))

    if rows:
        Database.executemany(
            "INSERT INTO lhb_daily (trade_date, stock_code, stock_name, close, pct_chg, "
            "lhb_net_buy, lhb_buy, lhb_sell, lhb_turnover, market_turnover, net_pct, "
            "turnover_pct, turnover_rate, float_mv, post_1d, post_2d, post_5d, post_10d, "
            "reason, interpretation) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE lhb_net_buy=VALUES(lhb_net_buy), "
            "lhb_buy=VALUES(lhb_buy), lhb_sell=VALUES(lhb_sell), "
            "lhb_turnover=VALUES(lhb_turnover), close=VALUES(close), "
            "pct_chg=VALUES(pct_chg), reason=VALUES(reason)",
            rows)
        saved = len(rows)

    logger.info(f"[龙虎榜] Tushare: {saved} 条 ({trade_date})")
    return saved


# ═══════════════════════════════════════════════════════════════
# 3. 十大股东（季度）
# ═══════════════════════════════════════════════════════════════

def collect_shareholder_quarterly(lookback_quarters=3):
    """
    采集股东户数季度数据 (Tushare pro.stk_holdernumber)。
    覆盖最近 N 个季度。
    """
    from core.data import Database
    from core.data.sources.tushare import _rate_limit, get_pro as _ts_pro

    pro = _get_pro()
    if not pro:
        logger.warning("[股东户数] Tushare 不可用")
        return 0

    today = datetime.now()
    quarter_end_month = ((today.month - 1) // 3 + 1) * 3
    quarter_end_day = calendar.monthrange(today.year, quarter_end_month)[1]
    if today.date() < datetime(today.year, quarter_end_month, quarter_end_day).date():
        quarter_end_month -= 3
        if quarter_end_month <= 0:
            quarter_end_month += 12
            base_year = today.year - 1
        else:
            base_year = today.year
    else:
        base_year = today.year
    saved = 0
    for i in range(lookback_quarters):
        month = quarter_end_month - i * 3
        year = base_year
        while month <= 0:
            month += 12
            year -= 1
        end_d = calendar.monthrange(year, month)[1]
        end_date = f"{year}{month:02d}{end_d}"

        _rate_limit()
        sleep(0.35)
        try:
            df = pro.stk_holdernumber(end_date=end_date)
        except Exception as e:
            logger.warning(f"[股东户数] Tushare {end_date} 失败: {e}")
            continue

        if df is None or df.empty:
            continue

        rows = []
        for _, r in df.iterrows():
            code = str(r.get("ts_code", "")).split(".")[0] if "." in str(r.get("ts_code", "")) else str(r.get("ts_code", ""))
            if not code:
                continue

            def _f(key):
                v = r.get(key)
                return float(v) if v is not None and str(v) != "nan" else None

            def _i(key):
                v = r.get(key)
                try:
                    f = float(v)
                    return None if math.isnan(f) else int(f)
                except (TypeError, ValueError):
                    return None

            rows.append((code, end_date,
                         _i("holder_num"),
                         str(r.get("ann_date", "")),
                         _f("top10_hold_ratio"),
                         _f("top10_float_hold_ratio"),
                         _f("fund_num"),
                         _f("fund_hold_ratio"),
                         _f("holder_num_change")))

        if rows:
            Database.executemany(
                "INSERT INTO shareholder_quarterly (stock_code, report_date, total_holders, "
                "announce_date, top10_hold_pct, top10_float_pct, fund_holders, "
                "fund_hold_pct, holders_change) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE total_holders=VALUES(total_holders), "
                "announce_date=VALUES(announce_date), "
                "top10_hold_pct=VALUES(top10_hold_pct), "
                "top10_float_pct=VALUES(top10_float_pct), "
                "fund_holders=VALUES(fund_holders), "
                "fund_hold_pct=VALUES(fund_hold_pct), "
                "holders_change=VALUES(holders_change)",
                rows)
            saved += len(rows)

    logger.info(f"[股东户数] Tushare: {saved} 条 ({lookback_quarters} 季)")
    return saved


# ═══════════════════════════════════════════════════════════════
# 增量入口（供 jobs.py 调用）
# ═══════════════════════════════════════════════════════════════

def collect_margin_incremental(trade_date=None):
    """融资融券增量采集 — 仅当日"""
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y%m%d")
    saved = collect_margin_daily(trade_date)
    if saved:
        return saved
    current = datetime.strptime(trade_date, "%Y%m%d")
    for offset in range(1, 6):
        fallback_date = (current - timedelta(days=offset)).strftime("%Y%m%d")
        saved = collect_margin_daily(fallback_date)
        if saved:
            logger.info(f"[融资融券] {trade_date} 无数据，回退到 {fallback_date}: {saved} 条")
            return saved
    return 0


def collect_lhb_incremental(trade_date=None):
    """龙虎榜增量采集"""
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y%m%d")
    return collect_lhb_daily(trade_date)


def collect_shareholder_incremental():
    """股东户数增量 — 最近 2 季"""
    return collect_shareholder_quarterly(lookback_quarters=2)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if "--incremental" in sys.argv:
        td = sys.argv[sys.argv.index("--date") + 1] if "--date" in sys.argv else None
        n1 = collect_margin_incremental(td)
        n2 = collect_lhb_incremental(td)
        n3 = collect_shareholder_incremental()
        print(f"完成: margin={n1}, lhb={n2}, shareholder={n3}")
    else:
        td = datetime.now().strftime("%Y%m%d")
        n1 = collect_margin_daily(td)
        n2 = collect_lhb_daily(td)
        n3 = collect_shareholder_quarterly()
        print(f"完成: margin={n1}, lhb={n2}, shareholder={n3}")
