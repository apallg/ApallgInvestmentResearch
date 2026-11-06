"""
A股交易时间工具：交易日历 / 交易时段判断 / 收盘前窗口
"""
import datetime
import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

MORNING_START = datetime.time(9, 30)
MORNING_END = datetime.time(11, 30)
AFTERNOON_START = datetime.time(13, 0)
AFTERNOON_END = datetime.time(15, 0)
A_SHARE_TIMEZONE = datetime.timezone(datetime.timedelta(hours=8))


def normalize_a_share_datetime(value):
    if not isinstance(value, datetime.datetime):
        raise TypeError("时间必须是datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=A_SHARE_TIMEZONE)
    return value.astimezone(A_SHARE_TIMEZONE)


def market_date(at=None):
    if at is None:
        at = datetime.datetime.now(datetime.timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=A_SHARE_TIMEZONE)
    return at.astimezone(A_SHARE_TIMEZONE).date()

_trading_dates_cache: Optional[Set[datetime.date]] = None
_cache_updated: Optional[datetime.date] = None
_cache_lock = __import__('threading').Lock()


def _build_trading_dates() -> Set[datetime.date]:
    """从 Tushare 获取交易日历，失败时回退 akshare → weekday"""
    today = market_date()
    start = "20050101"  # 覆盖全部历史回测区间, 防止老区间 is_trading_day 全 False
    end = (today + datetime.timedelta(days=400)).strftime("%Y%m%d")

    # ── 第一选择: Tushare ──
    try:
        from core.shared.config import TUSHARE_TOKEN
        if TUSHARE_TOKEN:
            import tushare as ts
            ts.set_token(TUSHARE_TOKEN)
            pro = ts.pro_api()
            df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
            if df is not None and len(df) > 0:
                dates = set()
                for v in df["cal_date"]:
                    try:
                        dates.add(datetime.date.fromisoformat(str(v)[:10]))
                    except Exception:
                        pass
                if len(dates) > 100:
                    logger.info(f"交易日历已加载: {len(dates)} 个交易日 (Tushare)")
                    return dates
    except Exception as e:
        logger.warning(f"Tushare 交易日历获取失败: {e}")

    # ── 第二选择: akshare (新浪) ──
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        if df is not None and len(df) > 0:
            col = df.columns[0]
            dates = set()
            for v in df[col]:
                if hasattr(v, "date"):
                    dates.add(v.date())
                else:
                    dates.add(datetime.datetime.strptime(str(v)[:10], "%Y-%m-%d").date())
            logger.info(f"交易日历已加载: {len(dates)} 个交易日 (akshare)")
            return dates
    except Exception as e:
        logger.warning(f"akshare 交易日历获取失败: {e}")

    # ── 最终回退: weekday ──
    logger.warning("交易日历使用 weekday 回退 (不含节假日)")
    dates = set()
    for i in range(-400, 400):
        d = today + datetime.timedelta(days=i)
        if d.weekday() < 5:
            dates.add(d)
    return dates


TRADING_DATES_KEY = "trading:dates"


def _refresh_cache():
    global _trading_dates_cache, _cache_updated
    today = market_date()
    with _cache_lock:
        if _trading_dates_cache is not None and _cache_updated == today:
            return
    from core.shared.infrastructure import redis_client as rds
    cached = rds._get(TRADING_DATES_KEY)
    if cached is not None:
        with _cache_lock:
            _trading_dates_cache = cached if isinstance(cached, set) else set(
                datetime.date.fromisoformat(d) for d in cached
            )
            _cache_updated = today
        return
    new_cache = _build_trading_dates()
    with _cache_lock:
        _trading_dates_cache = new_cache
        _cache_updated = today
    rds._set(TRADING_DATES_KEY, _trading_dates_cache, expire=86400)


def get_trading_days_between(start: datetime.date, end: datetime.date):
    """返回 start (不含) 到 end (不含) 之间的 A 股交易日列表, 按日期升序"""
    _refresh_cache()
    result = []
    current = start + datetime.timedelta(days=1)
    while current < end:
        if current in _trading_dates_cache:
            result.append(current)
        current += datetime.timedelta(days=1)
    return result


def is_trading_day(dt: datetime.datetime = None) -> bool:
    """判断是否为 A 股交易日"""
    if dt is None:
        dt = datetime.datetime.now(A_SHARE_TIMEZONE)
    _refresh_cache()
    if hasattr(dt, "date"):
        return dt.date() in _trading_dates_cache
    return dt in _trading_dates_cache


def get_latest_trading_day(ref_date: datetime.date = None) -> datetime.date:
    """返回 ref_date 当天或之前最近的 A 股交易日

    从 ref_date 向前回退 (最多 14 天), 找到第一个交易日。
    默认 ref_date = 今天。
    """
    if ref_date is None:
        ref_date = market_date()
    _refresh_cache()
    for offset in range(14):
        d = ref_date - datetime.timedelta(days=offset)
        if d in _trading_dates_cache:
            return d
    # 最终回退: 跳过周末
    d = ref_date
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d


def is_trading_time(dt: datetime.datetime = None) -> bool:
    """判断当前是否在 A 股交易时段内 (9:30-11:30 或 13:00-15:00)"""
    if dt is None:
        dt = datetime.datetime.now(A_SHARE_TIMEZONE)
    if not is_trading_day(dt):
        return False
    t = dt.time()
    return (MORNING_START <= t <= MORNING_END) or (AFTERNOON_START <= t <= AFTERNOON_END)


def is_force_close_window(dt: datetime.datetime = None, force_close_time: str = "14:54") -> bool:
    """判断是否在收盘前强制平仓窗口内 (默认 14:54-15:00)"""
    if dt is None:
        dt = datetime.datetime.now(A_SHARE_TIMEZONE)
    if not is_trading_day(dt):
        return False
    t = dt.time()
    h, m = map(int, force_close_time.split(":"))
    fc = datetime.time(h, m)
    return t >= fc and t <= AFTERNOON_END


def get_trading_session(dt: datetime.datetime = None) -> str:
    """返回当前时段: morning / afternoon / lunch_break / closed / non_trading"""
    if dt is None:
        dt = datetime.datetime.now(A_SHARE_TIMEZONE)
    if not is_trading_day(dt):
        return "non_trading"
    t = dt.time()
    if MORNING_START <= t <= MORNING_END:
        return "morning"
    if AFTERNOON_START <= t <= AFTERNOON_END:
        return "afternoon"
    if MORNING_END < t < AFTERNOON_START:
        return "lunch_break"
    return "closed"


def time_until_close(dt: datetime.datetime = None) -> int:
    """距离收盘(15:00)还有多少秒，已收盘返回负值"""
    if dt is None:
        dt = datetime.datetime.now(A_SHARE_TIMEZONE)
    if not is_trading_day(dt):
        return -1
    close_dt = datetime.datetime.combine(dt.date(), AFTERNOON_END, tzinfo=dt.tzinfo)
    return int((close_dt - dt).total_seconds())


def next_trading_time(dt: datetime.datetime = None) -> Optional[datetime.datetime]:
    """返回下一个交易时段的起始时间"""
    if dt is None:
        dt = datetime.datetime.now(A_SHARE_TIMEZONE)
    current_date = dt.date()
    session = get_trading_session(dt)

    if session == "non_trading":
        for offset in range(7):
            check = current_date + datetime.timedelta(days=offset)
            if is_trading_day(datetime.datetime.combine(check, datetime.time(12, 0))):
                return datetime.datetime.combine(check, MORNING_START)
    elif session == "morning":
        return datetime.datetime.combine(current_date, AFTERNOON_START)
    elif session in ("lunch_break", "closed"):
        for offset in range(1, 7):
            check = current_date + datetime.timedelta(days=offset)
            if is_trading_day(datetime.datetime.combine(check, datetime.time(12, 0))):
                return datetime.datetime.combine(check, MORNING_START)

    return None
