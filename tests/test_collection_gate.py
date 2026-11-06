import datetime

from core.data.collection_gate import GateCheckService, RULES


def test_daily_data_before_latest_trading_day_is_stale():
    service = GateCheckService()
    rule = next(r for r in RULES if r.table == "daily_quotes")

    assert service._is_date_fresh(
        rule,
        datetime.date(2026, 8, 17),
        datetime.date(2026, 8, 18),
    ) is False


def test_daily_data_on_latest_trading_day_is_fresh():
    service = GateCheckService()
    rule = next(r for r in RULES if r.table == "daily_quotes")

    assert service._is_date_fresh(
        rule,
        datetime.date(2026, 8, 18),
        datetime.date(2026, 8, 18),
    ) is True


def test_margin_freshness_uses_trading_day_count(monkeypatch):
    service = GateCheckService()
    rule = next(r for r in RULES if r.table == "margin_daily")
    monkeypatch.setattr(
        "core.shared.trading_time.get_trading_days_between",
        lambda start, end: [datetime.date(2026, 10, day) for day in (21, 22, 23)],
    )

    assert service._is_date_fresh(
        rule,
        datetime.date(2026, 10, 20),
        datetime.date(2026, 10, 23),
    ) is False


def test_margin_freshness_fails_closed_without_calendar(monkeypatch):
    service = GateCheckService()
    rule = next(r for r in RULES if r.table == "margin_daily")

    def fail_calendar(start, end):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr("core.shared.trading_time.get_trading_days_between", fail_calendar)
    assert service._is_date_fresh(
        rule,
        datetime.date(2026, 10, 20),
        datetime.date(2026, 10, 23),
    ) is False


def test_datetime_date_column_is_counted_by_calendar_date(monkeypatch):
    from core.data import Database

    rule = next(r for r in RULES if r.table == "stock_sectors")
    queries = []

    monkeypatch.setattr(Database, "is_available", classmethod(lambda cls: True))

    def fake_fetchone(sql, params=()):
        queries.append(sql)
        if "MAX" in sql:
            return {"latest": datetime.datetime(2026, 8, 14, 2, 11, 12)}
        if "SUM(CASE WHEN DATE(`updated_at`) = %s" in sql:
            return {"total": 3129, "latest_rows": 3129}
        if "IS NULL" in sql:
            return {"cnt": 0}
        return {"cnt": 3129, "total": 3129}

    monkeypatch.setattr(Database, "fetchone", classmethod(lambda cls, sql, params=(): fake_fetchone(sql, params)))

    result = GateCheckService()._check_single(rule, datetime.date(2026, 8, 14))

    assert result.row_count == 3129
    assert result.null_field_ratio == {"sector_type": 0.0}
    assert all("DATE(`updated_at`) = %s" in query for query in queries[1:])
