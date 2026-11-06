import pandas as pd
import pytest


@pytest.mark.unit
@pytest.mark.p0
def test_get_daily_kline_applies_qfq_adjustment(monkeypatch):
    from core.data.sources import tushare

    class FakePro:
        def daily(self, **kwargs):
            assert "adj" not in kwargs
            return pd.DataFrame([
                {
                    "trade_date": "20260102", "open": 10.0, "high": 12.0,
                    "low": 9.0, "close": 10.0, "vol": 100.0, "amount": 1000.0,
                },
                {
                    "trade_date": "20260105", "open": 10.0, "high": 12.0,
                    "low": 9.0, "close": 10.0, "vol": 100.0, "amount": 1000.0,
                },
            ])

        def adj_factor(self, **kwargs):
            return pd.DataFrame([
                {"trade_date": "20260102", "adj_factor": 1.0},
                {"trade_date": "20260105", "adj_factor": 2.0},
            ])

    monkeypatch.setattr(tushare, "get_pro", lambda: FakePro())
    monkeypatch.setattr(tushare, "_rate_limit", lambda: None)

    result = tushare.get_daily_kline("600519", "20260102", "20260105", adj="qfq")

    assert result["close"].tolist() == [5.0, 10.0]
    assert result["open"].tolist() == [5.0, 10.0]
    assert result["volume"].tolist() == [10000, 10000]
    assert result["amount"].tolist() == [1000000, 1000000]


@pytest.mark.unit
@pytest.mark.p0
def test_baostock_fallback_uses_raw_prices_and_tushare_factor(monkeypatch):
    from scripts import fast_collect

    class Result:
        error_code = "0"

        def __init__(self):
            self.rows = iter([
                ["2026-01-02", "10", "11", "9", "10", "100", "1000", "1", "0"],
                ["2026-01-05", "10", "11", "9", "10", "100", "1000", "1", "0"],
            ])

        def next(self):
            try:
                self.current = next(self.rows)
                return True
            except StopIteration:
                return False

        def get_row_data(self):
            return self.current

    class BaoStock:
        def query_history_k_data_plus(self, *args, **kwargs):
            assert kwargs["adjustflag"] == "3"
            return Result()

    monkeypatch.setattr(
        "core.data.sources.tushare.get_adj_factor",
        lambda *args, **kwargs: pd.DataFrame({
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "adj_factor": [1.0, 2.0],
        }),
    )
    _, result = fast_collect._baostock_with_fallback(
        BaoStock(), "600519", "2026-01-02", "2026-01-05", retries=1,
    )

    assert result["close"].tolist() == [5.0, 10.0]
    assert result["adj_factor"].tolist() == [1.0, 2.0]


@pytest.mark.unit
@pytest.mark.p0
def test_daily_quotes_save_rebases_history_in_same_transaction(monkeypatch):
    from core.data import DailyQuotesRepo, Database

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05"]),
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.0],
        "volume": [100], "amount": [1000], "amplitude": [0.0],
        "pct_chg": [0.0], "turnover": [0.0], "adj_factor": [2.0],
    })
    frame.attrs["provenance"] = {
        "batch_id": "test-batch", "source": "test", "provider_version": "test-v1",
        "adjustment_mode": "qfq", "anchor_date": "2026-01-05",
        "input_payload_sha256": "a" * 64,
    }
    calls = []

    class Cursor:
        def __init__(self):
            self.fetchall_count = 0

        def execute(self, sql, params):
            calls.append(("execute", sql, params))

        def fetchone(self):
            return {"trade_date": "2026-01-01", "adj_factor": 1.0}

        def fetchall(self):
            self.fetchall_count += 1
            if self.fetchall_count == 1:
                return []
            return [{
                "trade_date": "2026-01-05", "open": 5.0, "high": 5.5, "low": 4.5,
                "close": 5.0, "volume": 100, "amount": 1000.0, "amplitude": 0.0,
                "pct_chg": 0.0, "turnover": 0.0, "adj_factor": 2.0,
            }]

        def executemany(self, sql, rows):
            calls.append(("executemany", sql, rows))

    transaction_calls = []
    monkeypatch.setattr(
        Database,
        "transaction",
        lambda operation: (transaction_calls.append("begin") or operation(Cursor())),
    )

    DailyQuotesRepo.save_batch("600519", frame)

    rebase = next(call for call in calls if call[0] == "execute" and call[1].startswith("UPDATE daily_quotes"))
    assert rebase[2] == (0.5, 0.5, 0.5, 0.5, "600519")
    assert len(transaction_calls) == 1


@pytest.mark.unit
@pytest.mark.p0
def test_historical_quote_collection_preserves_existing_prices(monkeypatch):
    from core.data import DailyQuotesRepo, Database

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-05"]),
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.0],
        "volume": [100], "amount": [1000], "amplitude": [0.0],
        "pct_chg": [0.0], "turnover": [0.0], "adj_factor": [2.0],
    })
    frame.attrs["provenance"] = {
        "batch_id": "test-historical", "source": "test", "provider_version": "test-v1",
        "adjustment_mode": "qfq", "anchor_date": "2026-01-05", "anchor_factor": 1.0,
        "input_payload_sha256": "b" * 64,
    }
    statements = []
    queries = []

    class Cursor:
        def __init__(self):
            self.fetchall_count = 0

        def execute(self, sql, params):
            queries.append(sql)

        def fetchone(self):
            return {"trade_date": pd.Timestamp("2026-01-05").date(), "adj_factor": 1.0}

        def fetchall(self):
            self.fetchall_count += 1
            if self.fetchall_count == 1:
                return []
            return [{
                "trade_date": "2025-01-05", "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.0, "volume": 100, "amount": 1000.0, "amplitude": 0.0,
                "pct_chg": 0.0, "turnover": 0.0, "adj_factor": 1.0,
            }]

        def executemany(self, sql, rows):
            statements.append(sql)

    monkeypatch.setattr(Database, "transaction", lambda operation: operation(Cursor()))

    DailyQuotesRepo.save_batch("600519", frame)

    assert "close=COALESCE(close,VALUES(close))" in statements[0]
    assert not any(sql.startswith("UPDATE daily_quotes") for sql in queries)


@pytest.mark.unit
@pytest.mark.p0
def test_database_transaction_commits_after_operation(monkeypatch):
    from core.data import Database

    events = []

    class CursorContext:
        def __enter__(self):
            events.append("cursor_enter")
            return object()

        def __exit__(self, *args):
            events.append("cursor_exit")

    class Connection:
        def begin(self):
            events.append("begin")

        def cursor(self):
            return CursorContext()

        def commit(self):
            events.append("commit")

    connection = Connection()
    monkeypatch.setattr(Database, "_borrow", lambda: connection)
    monkeypatch.setattr(Database, "_recycle", lambda value: events.append("recycle"))

    result = Database.transaction(lambda cursor: events.append("operation") or 7)

    assert result == 7
    assert events == [
        "begin", "cursor_enter", "operation", "cursor_exit", "commit", "recycle",
    ]


@pytest.mark.unit
@pytest.mark.p0
def test_get_daily_kline_can_include_adjustment_factor_for_rebuild(monkeypatch):
    from core.data.sources import tushare

    class FakePro:
        def daily(self, **kwargs):
            return pd.DataFrame([
                {"trade_date": "20260102", "open": 10, "high": 10, "low": 10,
                 "close": 10, "vol": 10, "amount": 20},
                {"trade_date": "20260105", "open": 10, "high": 10, "low": 10,
                 "close": 10, "vol": 10, "amount": 20},
            ])

        def adj_factor(self, **kwargs):
            return pd.DataFrame([
                {"trade_date": "20260102", "adj_factor": 1.0},
                {"trade_date": "20260105", "adj_factor": 2.0},
            ])

    monkeypatch.setattr(tushare, "get_pro", lambda: FakePro())
    monkeypatch.setattr(tushare, "_rate_limit", lambda: None)

    result = tushare.get_daily_kline(
        "000001", "20260102", "20260105", adj="qfq", include_adj_factor=True,
    )

    assert result["adj_factor"].tolist() == [1.0, 2.0]
