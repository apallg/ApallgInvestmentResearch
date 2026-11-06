import pandas as pd
from datetime import datetime as PyDateTime
from types import SimpleNamespace

from core.data.fetcher import DataFetcher
from core.shared.infrastructure.cache import CacheManager


class FakeCache:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.reads = []
        self.writes = []

    def get_redis(self, key):
        self.reads.append(("redis", key))
        return self.values.get(key)

    def get(self, key):
        self.reads.append(("disk", key))
        return self.values.get(key)

    def get_cached(self, key):
        return self.values.get(key)

    def set(self, key, value, expire=3600):
        self.writes.append((key, value, expire))
        self.values[key] = value

    def persist(self, key, value, expire=3600):
        self.writes.append((key, value, expire))


def _daily_frame(start_date, end_date):
    dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="B")
    return pd.DataFrame({"date": dates, "close": range(1, len(dates) + 1)})


def _index_daily_frame(dates):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": [10.0] * len(dates),
        "high": [11.0] * len(dates),
        "low": [9.0] * len(dates),
        "close": [10.5] * len(dates),
        "volume": [100] * len(dates),
        "amount": [1000] * len(dates),
    })


def _freeze_index_fetcher_clock(monkeypatch):
    import core.data.fetcher as fetcher_module

    monkeypatch.setattr(
        fetcher_module, "datetime",
        SimpleNamespace(now=lambda: PyDateTime(2026, 10, 30, 12, 0)),
    )
    monkeypatch.setattr(
        "core.shared.trading_time.get_latest_trading_day",
        lambda _: PyDateTime(2026, 10, 29).date(),
    )


def test_index_daily_cached_rows_exclude_current_incomplete_trading_day(monkeypatch):
    _freeze_index_fetcher_clock(monkeypatch)
    cached = _index_daily_frame(["2026-10-29", "2026-10-30"])
    cache = FakeCache({CacheManager.make_key("idx_daily", "sh000300", "30"): cached})
    fetcher = DataFetcher(cache=cache)

    result = fetcher.get_index_daily("沪深300", days=30)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-29"]


def test_index_daily_tencent_rows_exclude_current_incomplete_trading_day(monkeypatch):
    _freeze_index_fetcher_clock(monkeypatch)
    fetcher = DataFetcher(cache=FakeCache())
    monkeypatch.setattr(
        fetcher, "_tencent_daily_raw",
        lambda *args: _index_daily_frame(["2026-10-29", "2026-10-30"]),
    )

    result = fetcher.get_index_daily("沪深300", days=30)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-29"]


def test_short_range_cache_cannot_satisfy_later_longer_request(monkeypatch):
    symbol = "600519"
    end_date = "20261030"
    short_start = "20261001"
    long_start = "20260101"
    old_short = _daily_frame(short_start, end_date)
    old_key = CacheManager.make_key("daily", "raw", symbol)
    cache = FakeCache({old_key: old_short})
    fetcher = DataFetcher(cache=cache)
    fetcher._get_db_latest_date = lambda: None
    source_calls = []

    def fetch_source(code, start_date, end_date, adjust):
        source_calls.append((code, start_date, end_date, adjust))
        return _daily_frame(start_date, end_date)

    monkeypatch.setattr("core.domain.screening.exclusion.is_excluded", lambda symbol: (False, None))
    monkeypatch.setattr("core.shared.config.MYSQL_ENABLED", False)
    monkeypatch.setattr(fetcher, "_tushare_daily", fetch_source)
    monkeypatch.setattr(fetcher, "_tencent_daily", lambda *args: None)

    short = fetcher.get_stock_daily(symbol, short_start, end_date, adjust="raw")
    longer = fetcher.get_stock_daily(symbol, long_start, end_date, adjust="raw")

    assert len(short) == len(old_short)
    assert len(longer) > len(short)
    assert source_calls == [
        (symbol, "2026-10-01", "2026-10-30", "raw"),
        (symbol, "2026-01-01", "2026-10-30", "raw"),
    ]
    assert all(key.startswith("daily_v2:") for key, _, _ in cache.writes)


def test_daily_cache_with_rows_before_request_start_is_ignored(monkeypatch):
    symbol = "600519"
    start_date = "20261001"
    end_date = "20261030"
    key = CacheManager.make_key("daily_v2", symbol, start_date, end_date, "raw")
    outside = _daily_frame("20260901", end_date)
    cache = FakeCache({key: outside})
    fetcher = DataFetcher(cache=cache)
    fetcher._get_db_latest_date = lambda: None
    fresh = _daily_frame(start_date, end_date)

    monkeypatch.setattr("core.domain.screening.exclusion.is_excluded", lambda value: (False, None))
    monkeypatch.setattr("core.shared.config.MYSQL_ENABLED", False)
    monkeypatch.setattr(fetcher, "_tushare_daily", lambda *args: fresh)
    monkeypatch.setattr(fetcher, "_tencent_daily", lambda *args: None)

    result = fetcher.get_stock_daily(symbol, start_date, end_date, adjust="raw")

    assert result["date"].min() == pd.Timestamp("2026-10-01")
    assert result["date"].max() == pd.Timestamp("2026-10-30")
    assert fetcher.get_sources()["stock_daily"] == "TuShare(兜底)"


def test_tushare_daily_response_outside_request_range_falls_back_to_tencent(monkeypatch):
    symbol = "600519"
    start_date = "20261001"
    end_date = "20261030"
    cache = FakeCache()
    fetcher = DataFetcher(cache=cache)
    fetcher._get_db_latest_date = lambda: None
    invalid_source = _daily_frame("20260930", end_date)
    fallback_source = _daily_frame(start_date, end_date)

    monkeypatch.setattr("core.domain.screening.exclusion.is_excluded", lambda value: (False, None))
    monkeypatch.setattr("core.shared.config.MYSQL_ENABLED", False)
    monkeypatch.setattr(fetcher, "_tushare_daily", lambda *args: invalid_source)
    monkeypatch.setattr(fetcher, "_tencent_daily", lambda *args: fallback_source)

    result = fetcher.get_stock_daily(symbol, start_date, end_date, adjust="raw")

    assert result["date"].min() == pd.Timestamp("2026-10-01")
    assert result["date"].max() == pd.Timestamp("2026-10-30")
    assert fetcher.get_sources()["stock_daily"] == "腾讯IFZQ HTTP(兜底)"


def test_database_daily_response_outside_request_range_is_not_cached(monkeypatch):
    symbol = "600519"
    start_date = "20261001"
    end_date = "20261030"
    key = CacheManager.make_key("daily_v2", symbol, start_date, end_date, "raw")
    cache = FakeCache()
    fetcher = DataFetcher(cache=cache)
    fetcher._get_db_latest_date = lambda: None
    invalid_database_rows = _daily_frame("20260930", end_date)
    fresh_source_rows = _daily_frame(start_date, end_date)

    monkeypatch.setattr("core.domain.screening.exclusion.is_excluded", lambda value: (False, None))
    monkeypatch.setattr("core.shared.config.MYSQL_ENABLED", True)
    monkeypatch.setattr("core.data.repos.quotes.DailyQuotesRepo.get_range", lambda *args: invalid_database_rows)
    monkeypatch.setattr("core.data.repos.quotes.DailyQuotesRepo.convert_from_qfq", lambda *args: invalid_database_rows)
    monkeypatch.setattr(fetcher, "_tushare_daily", lambda *args: fresh_source_rows)
    monkeypatch.setattr(fetcher, "_tencent_daily", lambda *args: None)

    result = fetcher.get_stock_daily(symbol, start_date, end_date, adjust="raw")

    assert result["date"].min() == pd.Timestamp("2026-10-01")
    assert result["date"].max() == pd.Timestamp("2026-10-30")
    assert cache.values[key]["date"].min() == pd.Timestamp("2026-10-01")
    assert fetcher.get_sources()["stock_daily"] == "TuShare(兜底)"


def test_qfq_cache_is_bypassed_without_a_proven_anchor_version(monkeypatch):
    symbol = "600519"
    start_date = "20260101"
    end_date = "20261030"
    key = CacheManager.make_key("daily_v2", symbol, start_date, end_date, "qfq")
    stale = _daily_frame(start_date, end_date)
    stale["close"] = -1
    cache = FakeCache({key: stale})
    fetcher = DataFetcher(cache=cache)
    fetcher._get_db_latest_date = lambda: None
    fresh = _daily_frame(start_date, end_date)

    monkeypatch.setattr("core.domain.screening.exclusion.is_excluded", lambda symbol: (False, None))
    monkeypatch.setattr("core.shared.config.MYSQL_ENABLED", False)
    monkeypatch.setattr(fetcher, "_tushare_daily", lambda *args: fresh)
    monkeypatch.setattr(fetcher, "_tencent_daily", lambda *args: None)

    result = fetcher.get_stock_daily(symbol, start_date, end_date, adjust="qfq")

    assert result["close"].equals(fresh["close"])
    assert cache.reads == []
    assert cache.writes == []
