from dataclasses import dataclass
from datetime import datetime


REALTIME_SOURCE_NAMES = {
    "tencent": "腾讯财经 (实时)",
    "sina": "新浪财经 (实时)",
}

DAILY_QUOTE_CONTRACT = {
    "price_basis": "qfq",
    "volume_unit": "share",
    "amount_unit": "yuan",
    "adj_factor": "tushare_hfq_factor",
}

REALTIME_QUOTE_CONTRACT = {
    "price_basis": "raw",
    "volume_unit": "share",
    "amount_unit": "yuan",
}

SOURCE_CAPABILITIES = {
    "tencent": {"quote_realtime", "index_realtime", "ranking"},
    "sina": {"quote_realtime", "index_realtime", "ranking"},
    "ths": {"quote_realtime", "intraday_bar", "sector_realtime"},
    "tushare": {
        "stock_basic", "trade_calendar", "daily", "weekly", "monthly",
        "financial", "macro",
    },
}


@dataclass(frozen=True)
class QuoteSnapshot:
    code: str
    price: float
    source: str
    observed_at: str
    received_at: str
    session_date: str
    payload: dict


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def source_supports(source, capability):
    return capability in SOURCE_CAPABILITIES.get(source, set())
