import datetime
import heapq
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.mark.unit
@pytest.mark.p0
def test_tushare_rate_limit_waits_until_token_is_acquired():
    from core.data.sources import tushare

    with patch.object(tushare._tushare_limiter, "acquire", side_effect=[0.25, 0]) as acquire, \
         patch.object(tushare.time, "sleep") as sleep:
        tushare._rate_limit()

    sleep.assert_called_once_with(0.25)
    assert acquire.call_count == 2


@pytest.mark.unit
@pytest.mark.p0
def test_tushare_interleaved_waiters_keep_partial_tokens(monkeypatch):
    from core.data.sources import tushare

    clock = [0.0]
    monkeypatch.setattr(tushare.time, "monotonic", lambda: clock[0])
    limiter = tushare.RateLimiter(max_calls=80, per_seconds=60)
    for _ in range(80):
        assert limiter.acquire() == 0
    waiting = [(0.0, "a"), (0.375, "b")]
    completed = []
    for _ in range(20):
        if not waiting:
            break
        clock[0], worker = heapq.heappop(waiting)
        delay = limiter.acquire()
        if delay == 0:
            completed.append(worker)
        else:
            heapq.heappush(waiting, (clock[0] + delay, worker))

    assert sorted(completed) == ["a", "b"]
    assert clock[0] == pytest.approx(1.5)


@pytest.mark.unit
@pytest.mark.p0
def test_tushare_stock_fetch_retries_transient_failures():
    from scripts.fast_collect import fetch_one_tushare

    price = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-24"]),
        "open": [10.0], "high": [10.5], "low": [9.8], "close": [10.2],
        "volume": [1000], "amount": [10000], "pct_chg": [2.0], "turnover": [1.0],
    })
    factor = pd.DataFrame({"date": pd.to_datetime(["2026-08-24"]), "adj_factor": [1.2]})

    with patch("core.data.sources.tushare.get_daily_kline", side_effect=[None, price]) as get_price, \
         patch("core.data.sources.tushare.get_adj_factor", side_effect=[None, factor]) as get_factor, \
         patch("scripts.fast_collect.time.sleep"):
        result = fetch_one_tushare("600519", "20260820", "20260824")

    assert result is not None
    assert get_price.call_count == 2
    assert get_factor.call_count == 2
    assert result[1]["adj_factor"].iloc[0] == pytest.approx(1.2)


@pytest.mark.unit
@pytest.mark.p0
def test_tushare_snapshot_rebases_qfq_and_writes_all_dates():
    from scripts import fast_collect

    prices = {
        "20260821": pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"], "trade_date": ["20260821", "20260821"],
            "open": [10.0, 8.0], "high": [10.0, 8.0], "low": [10.0, 8.0], "close": [10.0, 8.0],
            "vol": [10.0, 8.0], "amount": [20.0, 16.0], "pct_chg": [0.0, 0.0],
        }),
        "20260824": pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"], "trade_date": ["20260824", "20260824"],
            "open": [10.0, 8.0], "high": [10.0, 8.0], "low": [10.0, 8.0], "close": [10.0, 8.0],
            "vol": [12.0, 9.0], "amount": [24.0, 18.0], "pct_chg": [0.0, 0.0],
        }),
    }
    factors = {
        "20260821": pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260821"], "adj_factor": [1.0]}),
        "20260824": pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260824"], "adj_factor": [2.0]}),
    }
    pro = MagicMock()
    pro.daily.side_effect = lambda trade_date: prices[trade_date]
    pro.adj_factor.side_effect = lambda trade_date: factors[trade_date]
    written = []

    with patch("core.data.sources.tushare.get_pro", return_value=pro), \
         patch("core.data.sources.tushare._rate_limit"), \
         patch("core.data.DailyQuotesRepo.save_batch", side_effect=lambda code, frame: written.append((code, frame))):
        saved = fast_collect.collect_tushare_snapshots(
            ["000001", "000002"],
            [datetime.date(2026, 8, 21), datetime.date(2026, 8, 24)],
        )

    assert saved == 1
    assert written[0][0] == "000001"
    assert written[0][1]["close"].tolist() == [5.0, 10.0]
    assert written[0][1]["volume"].tolist() == [1000, 1200]


@pytest.mark.unit
@pytest.mark.p0
def test_baostock_incremental_only_fetches_stale_codes():
    from scripts import fast_collect

    cursor = MagicMock()
    cursor.fetchall.return_value = [
        {"code": "000001", "last_date": datetime.date(2026, 8, 24), "cnt_recent": 4},
        {"code": "000002", "last_date": datetime.date(2026, 8, 21), "cnt_recent": 3},
    ]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-24"]),
        "open": [10.0], "high": [10.5], "low": [9.8], "close": [10.2],
        "volume": [1000], "amount": [10000], "turnover": [1.0],
        "pct_chg": [2.0], "amplitude": [7.0],
    })

    with patch("core.data.Database.get_connection", return_value=connection), \
         patch("core.data.Database.return_connection"), \
         patch("core.shared.trading_time.is_trading_day", return_value=True), \
         patch("baostock.login"), patch("baostock.logout"), \
         patch.object(fast_collect, "_baostock_with_fallback", return_value=("000002", frame)) as fetch, \
         patch.object(fast_collect, "save_batch") as save, \
         patch.object(fast_collect.time, "sleep"):
        saved = fast_collect.collect_incremental(
            ["000001", "000002"],
            source="baostock",
            delay=0,
            latest_trading_day=datetime.date(2026, 8, 24),
        )

    assert saved == 1
    fetch.assert_called_once()
    assert fetch.call_args.args[1] == "000002"
    save.assert_called_once_with("000002", frame)


@pytest.mark.unit
@pytest.mark.p0
def test_daily_collection_cron_includes_monday():
    from core.shared.infrastructure.jobs import DataCollectionScheduler

    manager = MagicMock()
    manager.get_meta.return_value = None
    manager._started = True
    scheduler = DataCollectionScheduler()

    with patch("core.shared.infrastructure.scheduler.get_unified_scheduler", return_value=manager), \
         patch("core.shared.infrastructure.jobs.is_trading_day", return_value=False), \
         patch("core.shared.infrastructure.jobs.config.COLLECT_ENABLED", True), \
         patch("core.shared.infrastructure.jobs.config.MYSQL_ENABLED", True):
        scheduler.start()

    assert manager.register.call_args.args[2] == "30 18 * * mon-fri"


@pytest.mark.unit
@pytest.mark.p0
def test_daily_repair_uses_baostock_and_requested_trade_date():
    from core.data import integrity

    with patch("scripts.fast_collect.get_all_codes", return_value=["000001"]), \
         patch("scripts.fast_collect.collect_incremental", return_value=1) as collect, \
         patch("core.shared.config.COLLECT_DATA_SOURCE", "baostock"):
        saved = integrity._repair_daily_quotes("2026-08-24")

    assert saved == 1
    assert collect.call_args.kwargs["source"] == "baostock"
    assert collect.call_args.kwargs["latest_trading_day"] == "2026-08-24"
