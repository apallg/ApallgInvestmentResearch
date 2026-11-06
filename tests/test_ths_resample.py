import asyncio
import pytest
from unittest.mock import patch

from core.data.ths_collector import THSCollector, _parse_intraday_raw, _resample_to_ohlc
from core.data.repos.intraday import IntradayDataRepo
from core.shared.minute_data import has_complete_minute_sequence, has_verified_minute_semantics


@pytest.mark.p0
def test_minute_sequence_detects_gaps_and_allows_lunch_break():
    from datetime import datetime, timedelta, timezone

    assert has_complete_minute_sequence([
        datetime.fromisoformat(f"2026-10-30 09:{minute:02d}:00")
        for minute in (31, 32, 33)
    ])
    assert has_complete_minute_sequence([
        datetime.fromisoformat(value)
        for value in (
            "2026-10-30 11:29:00", "2026-10-30 11:30:00",
            "2026-10-30 13:00:00", "2026-10-30 13:01:00",
        )
    ])
    assert not has_complete_minute_sequence([
        datetime.fromisoformat(f"2026-10-30 09:{minute:02d}:00")
        for minute in (31, 33)
    ])
    assert not has_complete_minute_sequence([
        datetime.fromisoformat(value)
        for value in ("2026-10-30 11:30:00", "2026-10-30 12:00:00", "2026-10-30 13:01:00")
    ])
    assert not has_complete_minute_sequence([
        datetime.fromisoformat("2026-10-30 09:31:00"),
        datetime.fromisoformat("2026-10-30T09:32:00+08:00"),
    ])
    shanghai = timezone(timedelta(hours=8))
    assert has_complete_minute_sequence([
        datetime(2026, 10, 30, 9, minute, tzinfo=shanghai)
        for minute in (31, 32, 33)
    ])
    assert not has_complete_minute_sequence([
        datetime(2026, 10, 30, 9, minute, tzinfo=timezone.utc)
        for minute in (31, 32, 33)
    ])


@pytest.mark.p0
def test_intraday_api_reports_minute_gaps_as_partial(monkeypatch):
    import asyncio
    from datetime import datetime, timedelta, timezone
    from backend.api.ths import get_intraday

    now = datetime(2026, 10, 30, 9, 34, 30, tzinfo=timezone(timedelta(hours=8)))
    rows = [
        {"bar_time": "2026-10-30 09:31:00", "price": 10, "quantity_status": "unknown"},
        {"bar_time": "2026-10-30 09:33:00", "price": 10, "quantity_status": "unknown"},
    ]

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    monkeypatch.setattr("backend.api.ths.datetime", FixedDateTime)
    monkeypatch.setattr("backend.service.market_data.MarketDataService.get_live_intraday",
                        lambda code: (rows, "test-source"))

    payload = asyncio.run(get_intraday("000001"))

    assert payload["meta"]["sequence_status"] == "partial"
    assert payload["meta"]["data_envelope"]["status"] == "partial"


def test_live_intraday_rejects_response_for_different_trade_date():
    collector = THSCollector.__new__(THSCollector)
    collector.session = object()
    response = type("Response", (), {
        "text": 'quotebridge({"hs_000001":{"date":"20000101","data":"0930,10,100,10"}})'
    })()
    with patch("core.data.ths_collector._update_referer"), \
         patch.object(collector, "_request", return_value=response):
        assert collector.fetch_intraday("000001") is None


@pytest.mark.p0
def test_60min_source_parser_does_not_promote_unverified_quantity_fields():
    collector = THSCollector.__new__(THSCollector)
    collector.session = object()
    response = type("Response", (), {
        "text": 'quotebridge({"data":"202610300900,10,11,9,10.5,123,456,0.5"})'
    })()
    with patch("core.data.ths_collector._update_referer"), \
         patch.object(collector, "_request", return_value=response):
        rows = collector.fetch_60min_kline("000001")

    assert rows == [{
        "code": "000001",
        "bar_time": "2026-10-30 09:00:00",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": None,
        "amount": None,
        "quantity_status": "unknown",
        "pct_chg": 0.5,
    }]


@pytest.mark.p0
def test_60min_persistence_redacts_unverified_quantity_fields():
    row = {
        "bar_time": "2026-10-30 09:00:00", "open": 10, "high": 11,
        "low": 9, "close": 10.5, "volume": 123, "amount": 456, "pct_chg": 0.5,
    }
    with patch("core.data.ths_collector.Database.executemany") as save:
        THSCollector._save_60min(THSCollector.__new__(THSCollector), "000001", [row])

    assert save.call_args.args[1][0][6:8] == (None, None)


@pytest.mark.p0
def test_60min_api_redacts_legacy_quantities_and_reports_unverified_contract():
    from backend.api.ths import get_60min_kline

    legacy_row = {
        "bar_time": "2026-10-30 10:00:00", "open": 10, "high": 11,
        "low": 9, "close": 10.5, "volume": 123, "amount": 456, "pct_chg": 0.5,
    }
    with patch("backend.api.ths.Database.fetchall", return_value=[legacy_row]):
        result = asyncio.run(get_60min_kline("000001"))

    assert result["data"][0]["volume"] is None
    assert result["data"][0]["amount"] is None
    assert result["data"][0]["quantity_status"] == "unknown"
    assert result["meta"]["data_envelope"]["status"] == "partial"
    assert "source_quantity_semantics" in result["meta"]["data_envelope"]["missing_reason"]


def test_index_daily_endpoint_selects_the_latest_250_rows_in_chronological_order():
    from backend.api.ths import get_index_daily

    rows = [{"trade_date": "2025-01-02", "close": 3200}]
    with patch("backend.api.ths.Database.fetchall", return_value=rows) as fetch:
        result = asyncio.run(get_index_daily("000001"))

    query = fetch.call_args.args[0]
    assert "ORDER BY trade_date DESC LIMIT 250" in query
    assert "AS recent_index_rows ORDER BY trade_date ASC" in query
    assert result["data"] == rows


def test_historical_intraday_rejects_response_for_different_trade_date():
    collector = THSCollector.__new__(THSCollector)
    collector.session = object()
    response = type("Response", (), {
        "text": 'quotebridge({"hs_000001":{"date":"20261030","data":"0930,10,100,10"}})'
    })()
    with patch("core.data.ths_collector._update_referer"), \
         patch.object(collector, "_request", return_value=response):
        assert collector.fetch_intraday_history("000001", "2026-10-29") is None


def test_historical_intraday_labels_rows_only_with_matching_source_date():
    collector = THSCollector.__new__(THSCollector)
    collector.session = object()
    response = type("Response", (), {
        "text": 'quotebridge({"hs_000001":{"date":"20261029","data":"0930,10,100,10"}})'
    })()
    with patch("core.data.ths_collector._update_referer"), \
         patch.object(collector, "_request", return_value=response):
        rows = collector.fetch_intraday_history("000001", "2026-10-29")
    assert rows[0]["bar_time"] == "2026-10-29 09:30:00"


@pytest.mark.p0
def test_five_minute_bars_do_not_mix_hours_or_sessions():
    rows = []
    for hour, price, volume in (("09", 10, 1), ("10", 20, 2), ("13", 30, 3)):
        for minute in range(31, 36):
            rows.append({
                "bar_time": f"2026-10-26 {hour}:{minute:02d}:00",
                "price": price,
                "volume": volume,
                "amount": price * volume,
            })
    result = _resample_to_ohlc(rows[-1], rows)
    assert result == {"open": 30, "high": 30, "low": 30, "close": 30,
                      "volume": None, "amount": None, "quantity_status": "unknown"}


@pytest.mark.p0
def test_five_minute_aggregation_sums_only_fully_verified_quantities():
    rows = [
        {
            "bar_time": f"2026-10-26 09:{minute:02d}:00",
            "price": 10 + offset,
            "volume": 100 + offset,
            "amount": 1000.25 + offset,
            "quantity_status": "verified",
        }
        for offset, minute in enumerate(range(31, 36))
    ]

    result = _resample_to_ohlc(rows[-1], rows)

    assert result == {
        "open": 10,
        "high": 14,
        "low": 10,
        "close": 14,
        "volume": 510,
        "amount": 5011.25,
        "quantity_status": "verified",
    }


def test_five_minute_storage_records_source_batch_and_keeps_semantics_unknown():
    rows = [
        {
            "bar_time": f"2026-10-30 09:{minute:02d}:00",
            "price": 10 + offset,
            "volume": None,
            "amount": None,
            "raw_fields": [f"09{minute:02d}", str(10 + offset), "unknown"],
            "quantity_status": "unknown",
        }
        for offset, minute in enumerate(range(31, 36))
    ]
    observed_at = "2026-10-30T09:36:00+08:00"

    with patch("core.data.ths_collector.Database.executemany") as write:
        count = THSCollector._save_5min_rows(
            THSCollector.__new__(THSCollector), "000001", rows, observed_at=observed_at
        )

    sql, params = write.call_args.args
    assert count == 1
    assert "source_id" in sql and "source_version" in sql
    assert "semantic_certification_id" in sql and "payload_fingerprint" in sql
    assert "resample_contract_version" in sql
    assert params[0][8] == "unknown"
    assert params[0][9:14] == (
        "ths_web_time", "v2", None, observed_at, params[0][13]
    )
    assert len(params[0][13]) == 64
    assert params[0][14] == "a-share-session-v2"


def test_existing_minute_table_gets_unknown_by_default_provenance_columns():
    existing_columns = [
        {"Field": name}
        for name in (
            "id", "code", "bar_time", "open", "high", "low", "close",
            "volume", "amount", "created_at",
        )
    ]
    with patch("core.data.ths_collector._tables_ready", False), \
         patch("core.data.ths_collector.Database.execute") as execute, \
         patch("core.data.ths_collector.Database.fetchall", return_value=existing_columns):
        from core.data.ths_collector import ensure_tables

        ensure_tables()

    alters = [call.args[0] for call in execute.call_args_list if "ALTER TABLE" in call.args[0]]
    assert "ALTER TABLE ths_minute_bars_v2 ADD COLUMN quantity_status VARCHAR(24) NOT NULL DEFAULT 'unknown'" in alters
    assert any("semantic_certification_id VARCHAR(128) NULL" in sql for sql in alters)
    assert any("payload_fingerprint CHAR(64) NULL" in sql for sql in alters)


@pytest.mark.p0
def test_five_minute_aggregation_does_not_promote_mixed_or_invalid_quantities():
    rows = [
        {
            "bar_time": f"2026-10-26 09:{minute:02d}:00",
            "price": 10,
            "volume": 100,
            "amount": 1000,
            "quantity_status": "verified",
        }
        for minute in range(31, 36)
    ]
    rows[-1]["quantity_status"] = "unknown"

    mixed = _resample_to_ohlc(rows[-1], rows)
    assert mixed["volume"] is None
    assert mixed["amount"] is None
    assert mixed["quantity_status"] == "unknown"

    rows[-1]["quantity_status"] = "verified"
    rows[-1]["volume"] = 1.5
    assert _resample_to_ohlc(rows[-1], rows) is None


@pytest.mark.p0
def test_five_minute_bar_ending_on_hour_uses_previous_hour_minutes():
    rows = [
        {"bar_time": f"2026-10-26 09:{minute:02d}:00", "price": minute,
         "volume": 1, "amount": minute}
        for minute in range(56, 60)
    ]
    rows.append({"bar_time": "2026-10-26 10:00:00", "price": 60, "volume": 1, "amount": 60})
    result = _resample_to_ohlc(rows[-1], rows)
    assert result["open"] == 56
    assert result["close"] == 60
    assert result["volume"] is None
    assert result["quantity_status"] == "unknown"


@pytest.mark.p0
def test_five_minute_bar_rejects_incomplete_window():
    rows = [
        {"bar_time": f"2026-10-26 09:{minute:02d}:00", "price": minute,
         "volume": 1, "amount": minute}
        for minute in (31, 32, 34, 35)
    ]

    assert _resample_to_ohlc(rows[-1], rows) is None


@pytest.mark.p0
def test_five_minute_bar_rejects_lunch_boundary_and_emits_first_afternoon_bar():
    lunch_boundary = {"bar_time": "2026-10-26 13:00:00", "price": 50,
                      "volume": 1, "amount": 50}
    assert _resample_to_ohlc(lunch_boundary, [lunch_boundary]) is None

    rows = [
        {"bar_time": f"2026-10-26 13:{minute:02d}:00", "price": minute,
         "volume": 1, "amount": minute}
        for minute in range(1, 6)
    ]
    result = _resample_to_ohlc(rows[-1], rows)

    assert result["open"] == 1
    assert result["close"] == 5


@pytest.mark.p0
def test_five_minute_bar_excludes_same_clock_time_from_previous_date():
    rows = [{"bar_time": "2026-10-23 13:35:00", "price": 1, "volume": 100, "amount": 100}]
    rows.extend(
        {"bar_time": f"2026-10-26 13:{minute:02d}:00", "price": 10, "volume": 2, "amount": 20}
        for minute in range(31, 36)
    )
    result = _resample_to_ohlc(rows[-1], rows)
    assert result["open"] == result["close"] == 10
    assert result["volume"] is None
    assert result["quantity_status"] == "unknown"


@pytest.mark.p0
def test_five_minute_aggregation_accepts_shanghai_aware_times_and_rejects_other_offsets():
    rows = [
        {
            "bar_time": f"2026-10-26T09:{minute:02d}:00+08:00",
            "price": 10 + offset,
            "volume": None,
            "amount": None,
            "quantity_status": "unknown",
        }
        for offset, minute in enumerate(range(31, 36))
    ]
    accepted = _resample_to_ohlc(rows[-1], rows)
    assert accepted["open"] == 10
    assert accepted["close"] == 14

    utc_rows = [
        {**row, "bar_time": row["bar_time"].replace("+08:00", "+00:00")}
        for row in rows
    ]
    assert _resample_to_ohlc(utc_rows[-1], utc_rows) is None


@pytest.mark.p0
def test_five_minute_aggregation_fails_closed_on_unparseable_minute_time():
    rows = [
        {
            "bar_time": f"2026-10-26 09:{minute:02d}:00",
            "price": 10,
            "volume": None,
            "amount": None,
            "quantity_status": "unknown",
        }
        for minute in range(31, 36)
    ]
    rows.append({"bar_time": "not-a-timestamp", "price": 10})

    assert _resample_to_ohlc(rows[4], rows) is None


@pytest.mark.p0
def test_unverified_source_quantity_fields_are_preserved_but_not_exposed_as_volume_or_amount():
    row = _parse_intraday_raw("093500,10,100,9.9,1000", "000001", "2026-10-27")[0]
    assert row["raw_fields"] == ["093500", "10", "100", "9.9", "1000"]
    assert row["volume"] is None
    assert row["amount"] is None
    assert row["quantity_status"] == "unknown"


@pytest.mark.p0
def test_minute_semantics_gate_requires_registered_source_time_units_basis_and_quantity_contract(monkeypatch):
    import core.shared.minute_data as minute_data

    envelope = {
        "schema": "data-envelope-v1",
        "source": "test-provider",
        "semantic_certification_id": "test-provider-minute-v1",
        "observed_at": "2026-10-30T10:00:00+08:00",
        "available_at": "2026-10-30T10:00:01+08:00",
        "as_of": "2026-10-30",
        "version": "provider-contract-v1",
        "unit": {"price": "CNY/share", "volume": "share", "amount": "CNY"},
        "price_basis": "raw",
        "coverage": {"requested": 5, "accepted": 5, "missing": 0},
        "status": "verified",
        "missing_reason": [],
        "source_quantity_semantics": "incremental-per-minute",
    }

    assert has_verified_minute_semantics(envelope) is False
    monkeypatch.setattr(minute_data, "_CERTIFIED_MINUTE_CONTRACTS", frozenset({
        ("test-provider", "provider-contract-v1", "test-provider-minute-v1"),
    }))
    assert has_verified_minute_semantics(envelope) is True
    assert has_verified_minute_semantics({**envelope, "price_basis": "unknown"}) is False
    assert has_verified_minute_semantics({**envelope, "source_quantity_semantics": "unknown"}) is False
    assert has_verified_minute_semantics({**envelope, "source_quantity_semantics": "assumed-correct"}) is False
    assert has_verified_minute_semantics({**envelope, "observed_at": "not-a-timestamp"}) is False
    assert has_verified_minute_semantics({**envelope, "available_at": "not-a-timestamp"}) is False
    assert has_verified_minute_semantics({**envelope, "as_of": "not-a-date"}) is False
    assert has_verified_minute_semantics({**envelope, "available_at": "2026-10-30T09:59:59+08:00"}) is False
    assert has_verified_minute_semantics({**envelope, "available_at": "2026-10-30T01:59:59Z"}) is False
    assert has_verified_minute_semantics({**envelope, "unit": {"volume": "share"}}) is False
    assert has_verified_minute_semantics({**envelope, "unit": {**envelope["unit"], "volume": "lot"}}) is False
    assert has_verified_minute_semantics({**envelope, "coverage": {"requested": 5, "accepted": 4, "missing": 1}}) is False
    assert has_verified_minute_semantics({**envelope, "coverage": {"requested": 5, "accepted": 5, "missing": 1}}) is False
    assert has_verified_minute_semantics({**envelope, "coverage": {"requested": True, "accepted": True, "missing": 0}}) is False
    assert has_verified_minute_semantics({key: value for key, value in envelope.items() if key != "coverage"}) is False
    assert has_verified_minute_semantics({**envelope, "semantic_certification_id": "unregistered"}) is False


@pytest.mark.p0
@pytest.mark.parametrize("price", ["NaN", "Infinity", "-Infinity", "0", "-1"])
def test_intraday_parser_rejects_non_finite_or_non_positive_prices(price):
    assert _parse_intraday_raw(f"093500,{price},100,9.9,1000", "000001", "2026-10-27") == []


@pytest.mark.p0
@pytest.mark.parametrize("price", [float("nan"), float("inf"), 0, -1])
def test_five_minute_aggregation_rejects_invalid_price_values(price):
    rows = [
        {"bar_time": f"2026-10-26 09:{minute:02d}:00", "price": 10,
         "volume": None, "amount": None}
        for minute in range(31, 36)
    ]
    rows[-1]["price"] = price

    assert _resample_to_ohlc(rows[-1], rows) is None


@pytest.mark.p0
def test_live_intraday_consumers_reject_unknown_quantity_semantics():
    import datetime
    from backend.service.rebalance import _try_ths_intraday_live

    rows = [{
        "bar_time": "2026-10-27 09:59:00",
        "price": 10,
        "volume": None,
        "amount": None,
        "quantity_status": "unknown",
    }]
    now = datetime.datetime(2026, 10, 27, 10, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    with patch("core.shared.trading_time.market_date", return_value=now.date()), \
            patch("core.shared.trading_time.is_trading_time", return_value=True):
        with patch("backend.service.market_data.MarketDataService.get_live_intraday", return_value=(rows, "ths")):
            with patch("core.data.ths_collector.THSCollector._save_intraday") as save_raw:
                assert _try_ths_intraday_live("000001", "2026-10-27", now=now) is None
                save_raw.assert_called_once_with("000001", rows)


@pytest.mark.p0
@pytest.mark.parametrize("bar_time", [
    "2026-10-29 09:59:00",
    "2026-10-30 09:40:00",
])
def test_live_intraday_rejects_cross_date_and_stale_verified_bars(monkeypatch, bar_time):
    import datetime
    from unittest.mock import Mock
    from backend.service.rebalance import _try_ths_intraday_live

    now = datetime.datetime(2026, 10, 30, 10, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    rows = [{
        "bar_time": bar_time,
        "price": 10,
        "volume": 100,
        "amount": 1000,
        "quantity_status": "verified",
    }]
    monkeypatch.setattr("core.shared.trading_time.market_date", lambda at: at.date())
    monkeypatch.setattr("core.shared.trading_time.is_trading_time", lambda at: True)
    monkeypatch.setattr("backend.service.market_data.MarketDataService.get_live_intraday",
                        lambda code: (rows, "ths"))
    save = Mock()
    monkeypatch.setattr("core.data.ths_collector.THSCollector._save_intraday", save)
    monkeypatch.setattr("core.data.ths_collector._resample_to_ohlc", lambda *args: {
        "open": 10, "high": 10, "low": 10, "close": 10,
        "volume": 100, "amount": 1000, "quantity_status": "verified",
    })

    assert _try_ths_intraday_live("000001", "2026-10-30", now=now) is None
    save.assert_not_called()


@pytest.mark.p0
def test_live_intraday_accepts_only_fresh_current_session_verified_bars(monkeypatch):
    import datetime
    from backend.service.rebalance import _try_ths_intraday_live

    now = datetime.datetime(2026, 10, 30, 10, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    rows = [{
        "bar_time": "2026-10-30 09:59:00",
        "price": 10,
        "volume": 100,
        "amount": 1000,
        "quantity_status": "verified",
    }]
    monkeypatch.setattr("core.shared.trading_time.market_date", lambda at: at.date())
    monkeypatch.setattr("core.shared.trading_time.is_trading_time", lambda at: True)
    monkeypatch.setattr("backend.service.market_data.MarketDataService.get_live_intraday",
                        lambda code: (rows, "ths"))
    monkeypatch.setattr("core.data.ths_collector.THSCollector._save_intraday", lambda *args: None)
    monkeypatch.setattr("core.data.ths_collector._resample_to_ohlc", lambda *args: {
        "open": 10, "high": 10, "low": 10, "close": 10,
        "volume": 100, "amount": 1000, "quantity_status": "verified",
    })

    result = _try_ths_intraday_live("000001", "2026-10-30", now=now)
    assert result is not None
    assert str(result.index[-1]) == "2026-10-30 09:59:00"


@pytest.mark.p0
def test_live_intraday_uses_verified_quantity_from_real_resampler(monkeypatch):
    import datetime
    from backend.service.rebalance import _try_ths_intraday_live

    now = datetime.datetime(2026, 10, 30, 10, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    rows = [
        {
            "bar_time": f"2026-10-30 09:{minute:02d}:00",
            "price": 10 + offset,
            "volume": 100 + offset,
            "amount": 1000 + offset,
            "quantity_status": "verified",
        }
        for offset, minute in enumerate(range(51, 56))
    ]
    monkeypatch.setattr("core.shared.trading_time.market_date", lambda at: at.date())
    monkeypatch.setattr("core.shared.trading_time.is_trading_time", lambda at: True)
    monkeypatch.setattr(
        "backend.service.market_data.MarketDataService.get_live_intraday",
        lambda code: (rows, "verified-source"),
    )
    saved = []
    monkeypatch.setattr(
        "core.data.ths_collector.THSCollector._save_intraday",
        lambda self, code, values: saved.append((code, values)),
    )

    result = _try_ths_intraday_live("000001", "2026-10-30", now=now)

    assert result is not None
    assert str(result.index[-1]) == "2026-10-30 09:55:00"
    assert result.iloc[-1]["volume"] == 510
    assert result.iloc[-1]["amount"] == 5010
    assert saved == [("000001", rows)]


@pytest.mark.p0
def test_history_repo_rejects_verified_rows_with_missing_amount():
    row = {
        "bar_time": "2026-10-30 09:35:00",
        "open": 10,
        "high": 10.1,
        "low": 9.9,
        "close": 10,
        "volume": 100,
        "amount": None,
        "quantity_status": "verified",
    }
    with patch("core.data.repos.intraday.Database.fetchall", return_value=[row]) as fetch:
        assert IntradayDataRepo.get_history_bars("000001", "2026-10-30", "2026-10-30") == []

    sql = fetch.call_args.args[0]
    assert "volume IS NOT NULL" in sql
    assert "amount IS NOT NULL" in sql


@pytest.mark.parametrize("field,value", [
    ("volume", 100.5),
    ("open", 0),
    ("close", -1),
    ("high", 9.8),
    ("low", 10.2),
    ("amount", float("inf")),
])
def test_history_repo_rejects_verified_rows_with_invalid_price_or_quantity(field, value):
    row = {
        "bar_time": "2026-10-30 09:35:00",
        "open": 10,
        "high": 10.1,
        "low": 9.9,
        "close": 10,
        "volume": 100,
        "amount": 1000,
        "quantity_status": "verified",
    }
    row[field] = value

    with patch("core.data.repos.intraday.Database.fetchall", return_value=[row]):
        assert IntradayDataRepo.get_history_bars("000001", "2026-10-30", "2026-10-30") == []


def test_today_repo_rejects_entire_snapshot_when_one_verified_row_is_invalid():
    rows = [
        {
            "bar_time": "2026-10-30 09:35:00",
            "open": 10,
            "high": 10.1,
            "low": 9.9,
            "close": 10,
            "volume": 100,
            "amount": 1000,
            "quantity_status": "verified",
        },
        {
            "bar_time": "2026-10-30 09:40:00",
            "open": 10,
            "high": 10.1,
            "low": 9.9,
            "close": 10,
            "volume": 100.5,
            "amount": 1000,
            "quantity_status": "verified",
        },
    ]

    with patch("core.data.repos.intraday.Database.fetchall", return_value=rows):
        assert IntradayDataRepo.get_today_bars("000001", "2026-10-30") is None


@pytest.mark.p0
def test_intraday_bars_api_returns_price_when_quantity_semantics_are_unknown():
    from datetime import datetime, timedelta
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router
    from core.shared.trading_time import A_SHARE_TIMEZONE

    now = datetime.now(A_SHARE_TIMEZONE) - timedelta(seconds=10)
    app = FastAPI()
    app.include_router(router)
    rows = [{
        "bar_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "price": 10.25,
        "volume": 123,
        "amount": 1260.75,
        "avg_price": 10.2,
        "quantity_status": "unknown",
    }]
    with patch("core.data.ths_collector.THSCollector.fetch_intraday", return_value=rows):
        response = TestClient(app).get("/intraday-bars/000001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["bars"] == [{
        "time": now.strftime("%H:%M"),
        "price": 10.25,
        "volume": None,
        "amount": None,
        "avg_price": None,
        "quantity_status": "unknown",
    }]
    assert payload["data"]["quantity_status"] == "unknown"
    assert payload["data"]["semantic_status"] == "unverified"
    assert payload["data"]["price_basis"] == "unknown"
    assert payload["data"]["data_envelope"]["status"] == "partial"
    assert payload["data"]["volume_unit"] == "unknown"
    assert payload["data"]["amount_unit"] == "unknown"


@pytest.mark.p0
def test_intraday_bars_api_hides_quantities_without_source_semantics_contract():
    from datetime import datetime, timedelta
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router
    from core.shared.trading_time import A_SHARE_TIMEZONE

    now = datetime.now(A_SHARE_TIMEZONE) - timedelta(seconds=10)
    app = FastAPI()
    app.include_router(router)
    rows = [{
        "bar_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "price": 10.25,
        "volume": 123,
        "amount": 1260.75,
        "avg_price": 10.2,
        "quantity_status": "verified",
    }]
    with patch("core.data.ths_collector.THSCollector.fetch_intraday", return_value=rows):
        payload = TestClient(app).get("/intraday-bars/000001").json()["data"]

    assert payload["bars"][0]["price"] == 10.25
    assert payload["bars"][0]["volume"] is None
    assert payload["bars"][0]["amount"] is None
    assert payload["bars"][0]["quantity_status"] == "unknown"
    assert payload["semantic_status"] == "unverified"


@pytest.mark.p0
def test_intraday_bars_api_marks_missing_minute_as_partial():
    from datetime import datetime, timedelta, timezone
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    now = datetime(2026, 10, 30, 9, 34, 10, tzinfo=timezone(timedelta(hours=8)))
    rows = [
        {"bar_time": "2026-10-30 09:31:00", "price": 10.0, "quantity_status": "unknown"},
        {"bar_time": "2026-10-30 09:33:00", "price": 10.1, "quantity_status": "unknown"},
    ]
    app = FastAPI()
    app.include_router(router)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    with patch("backend.api.rebalance.datetime", FixedDateTime), \
         patch("core.data.ths_collector.THSCollector.fetch_intraday", return_value=rows):
        payload = TestClient(app).get("/intraday-bars/000001").json()["data"]

    assert payload["sequence_status"] == "partial"
    assert payload["data_envelope"]["status"] == "partial"


@pytest.mark.p0
def test_intraday_bars_api_skips_invalid_prices_and_nulls_invalid_average_price():
    from datetime import datetime, timedelta
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router
    from core.shared.trading_time import A_SHARE_TIMEZONE

    now = datetime.now(A_SHARE_TIMEZONE) - timedelta(seconds=10)
    bar_time = now.strftime("%Y-%m-%d %H:%M:%S")
    invalid_bar_time = (now - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S")
    app = FastAPI()
    app.include_router(router)
    rows = [
        {"bar_time": invalid_bar_time, "price": float("nan"), "avg_price": 10},
        {"bar_time": bar_time, "price": 10.25, "avg_price": float("inf"),
         "quantity_status": "unknown"},
    ]
    with patch("core.data.ths_collector.THSCollector.fetch_intraday", return_value=rows):
        response = TestClient(app).get("/intraday-bars/000001")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["bars"] == [{
        "time": now.strftime("%H:%M"),
        "price": 10.25,
        "volume": None,
        "amount": None,
        "avg_price": None,
        "quantity_status": "unknown",
    }]


@pytest.mark.p0
@pytest.mark.parametrize("volume,amount", [
    (None, 100),
    (float("inf"), 100),
    (-1, 100),
    (1.5, 100),
    (0, 100),
    (100, float("nan")),
])
def test_intraday_bars_api_downgrades_invalid_verified_quantities(volume, amount):
    from datetime import datetime
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    now = datetime.now().replace(second=0, microsecond=0)
    app = FastAPI()
    app.include_router(router)
    rows = [{
        "bar_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "price": 10.25,
        "volume": volume,
        "amount": amount,
        "avg_price": 10.2,
        "quantity_status": "verified",
    }]
    with patch("core.data.ths_collector.THSCollector.fetch_intraday", return_value=rows):
        response = TestClient(app).get("/intraday-bars/000001")

    assert response.status_code == 200
    bar = response.json()["data"]["bars"][0]
    assert bar["quantity_status"] == "unknown"
    assert bar["volume"] is None
    assert bar["amount"] is None


@pytest.mark.p0
def test_new_minute_bars_are_isolated_from_unverified_history():
    rows = [
        {"bar_time": f"2026-10-27 09:{minute:02d}:00", "price": 10, "volume": 2, "amount": 20}
        for minute in range(31, 36)
    ]
    with patch("core.data.ths_collector.Database.executemany") as write:
        THSCollector._save_5min_rows(THSCollector.__new__(THSCollector), "000001", rows)
    sql = write.call_args.args[0]
    assert "ths_minute_bars_v2" in sql
    assert "quantity_status" in sql
    for column in ("open", "high", "low", "close", "volume", "amount", "quantity_status"):
        assert (
            f"{column}=IF(quantity_status='verified' AND VALUES(quantity_status)<>"
            f"'verified',{column},VALUES({column}))"
        ) in sql
    assert write.call_args.args[1][0][6:9] == (None, None, "unknown")

    with patch("core.data.repos.intraday.Database.fetchall", return_value=[]) as read:
        IntradayDataRepo.get_history_bars("000001", "2026-10-27", "2026-10-27")
    assert "ths_minute_bars_v2" in read.call_args.args[0]
    assert "quantity_status = 'verified'" in read.call_args.args[0]
    with patch("core.data.repos.intraday.Database.fetchall", return_value=[]) as read:
        IntradayDataRepo.get_today_bars("000001", "2026-10-27")
    assert "quantity_status = 'verified'" in read.call_args.args[0]
    with patch("core.data.repos.intraday.Database.fetchall", return_value=[]) as read:
        assert IntradayDataRepo.get_latest_time("000001", "2026-10-27") is None
    assert "quantity_status = 'verified'" in read.call_args.args[0]


@pytest.mark.p0
def test_live_intraday_rejects_fractional_verified_volume(monkeypatch):
    import datetime
    from unittest.mock import Mock
    from backend.service.rebalance import _try_ths_intraday_live

    now = datetime.datetime(2026, 10, 30, 10, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    rows = [{
        "bar_time": "2026-10-30 09:59:00",
        "price": 10,
        "volume": 100.5,
        "amount": 1000,
        "quantity_status": "verified",
    }]
    monkeypatch.setattr("core.shared.trading_time.market_date", lambda at: at.date())
    monkeypatch.setattr("core.shared.trading_time.is_trading_time", lambda at: True)
    monkeypatch.setattr("backend.service.market_data.MarketDataService.get_live_intraday",
                        lambda code: (rows, "ths"))
    save = Mock()
    monkeypatch.setattr("core.data.ths_collector.THSCollector._save_intraday", save)

    assert _try_ths_intraday_live("000001", "2026-10-30", now=now) is None
    save.assert_not_called()
@pytest.mark.parametrize("field,value", [("close", -1), ("volume", 100.5)])
def test_latest_time_rejects_freshness_claim_when_any_verified_bar_is_invalid(field, value):
    rows = [
        {
            "bar_time": "2026-10-30 09:35:00",
            "open": 10,
            "high": 10.1,
            "low": 9.9,
            "close": 10,
            "volume": 100,
            "amount": 1000,
            "quantity_status": "verified",
        },
        {
            "bar_time": "2026-10-30 09:40:00",
            "open": 10,
            "high": 10.1,
            "low": 9.9,
            "close": 10,
            "volume": 100,
            "amount": 1000,
            "quantity_status": "verified",
        },
    ]
    rows[1][field] = value

    with patch("core.data.repos.intraday.Database.fetchall", return_value=rows):
        assert IntradayDataRepo.get_latest_time("000001", "2026-10-30") is None


def test_latest_time_uses_only_a_fully_validated_snapshot():
    rows = [
        {
            "bar_time": "2026-10-30 09:35:00",
            "open": 10,
            "high": 10.1,
            "low": 9.9,
            "close": 10,
            "volume": 100,
            "amount": 1000,
            "quantity_status": "verified",
        },
        {
            "bar_time": "2026-10-30 09:40:00",
            "open": 10,
            "high": 10.1,
            "low": 9.9,
            "close": 10,
            "volume": 100,
            "amount": 1000,
            "quantity_status": "verified",
        },
    ]

    with patch("core.data.repos.intraday.Database.fetchall", return_value=rows):
        assert IntradayDataRepo.get_latest_time("000001", "2026-10-30") == "2026-10-30 09:40:00"
