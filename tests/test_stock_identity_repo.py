import datetime

from core.data.repos.stock_identity import StockIdentityRepo


def test_get_listing_dates_normalizes_exchange_code_and_preserves_request_key(monkeypatch):
    captured = {}

    def fetchall(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return [{"ts_code": "600000.SH", "symbol": "600000", "list_date": datetime.date(1999, 11, 10)}]

    monkeypatch.setattr("core.data.repos.stock_identity.Database.fetchall", fetchall)

    result = StockIdentityRepo.get_listing_dates(["sh600000", "600000.SH", "600001"])

    assert captured["params"] == ["600000", "600001"]
    assert result == {
        "sh600000": datetime.date(1999, 11, 10),
        "600000.SH": datetime.date(1999, 11, 10),
    }


def test_get_listing_dates_parses_database_date_strings(monkeypatch):
    monkeypatch.setattr(
        "core.data.repos.stock_identity.Database.fetchall",
        lambda sql, params: [{"ts_code": "000001.SZ", "symbol": "000001", "list_date": "1991-04-03"}],
    )

    assert StockIdentityRepo.get_listing_dates(["000001"]) == {
        "000001": datetime.date(1991, 4, 3),
    }
