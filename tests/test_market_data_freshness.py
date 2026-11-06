from datetime import datetime

import pandas as pd

from core.domain.market_data.freshness import assess_batch, assess_quote


def _quote(observed_at, code="600519"):
    return {
        "price": 100.0,
        "source": "tencent",
        "observed_at": observed_at,
        "received_at": observed_at,
        "session_date": "2026-08-18",
        "code": code,
    }


def test_assess_quote_rejects_old_snapshot():
    now = datetime(2026, 8, 18, 10, 0, 5)
    status = assess_quote("600519", _quote("2026-08-18T10:00:00"), now, 5)
    assert status["usable"] is True

    old = assess_quote("600519", _quote("2026-08-18T09:59:59"), now, 5)
    assert old["usable"] is False
    assert old["freshness"] == "stale"
    assert old["missing_reason"] == "stale_quote"


def test_assess_quote_normalizes_timezone_offsets_before_age_check():
    now = datetime(2026, 8, 18, 10, 0, 5)
    status = assess_quote("600519", _quote("2026-08-18T02:00:04Z"), now, 5)

    assert status["usable"] is True
    assert status["age_seconds"] == 1


def test_assess_quote_reports_missing_and_invalid_data_reasons():
    now = datetime(2026, 8, 18, 10, 0, 5)
    missing = assess_quote("600519", None, now, 5)
    invalid_price = assess_quote("600519", {**_quote("2026-08-18T10:00:04"), "price": 0}, now, 5)
    missing_time = assess_quote("600519", {**_quote(None), "observed_at": None}, now, 5)
    future = assess_quote("600519", _quote("2026-08-18T10:00:20"), now, 5)

    assert missing["missing_reason"] == "missing_quote"
    assert invalid_price["missing_reason"] == "invalid_price"
    assert missing_time["missing_reason"] == "missing_or_invalid_observed_at"
    assert future["missing_reason"] == "future_observed_at"


def test_assess_quote_reports_invalid_session_reason():
    now = datetime(2026, 8, 18, 10, 0, 5)
    quote = {**_quote("2026-08-18T10:00:04"), "session_date": "2026-08-17"}

    status = assess_quote("600519", quote, now, 5, expected_session_date="2026-08-18")

    assert status["usable"] is False
    assert status["missing_reason"] == "wrong_session_date"


def test_assess_quote_rejects_previous_session():
    now = datetime(2026, 8, 18, 10, 0, 5)
    status = assess_quote(
        "600519",
        {**_quote("2026-08-17T10:00:00"), "session_date": "2026-08-17"},
        now,
        300,
        expected_session_date="2026-08-18",
    )
    assert status["usable"] is False
    assert status["freshness"] == "stale"
    assert status["missing_reason"] == "wrong_session_date"


def test_assess_batch_strict_mode_rejects_partial_quotes():
    now = datetime(2026, 8, 18, 10, 0, 5)
    quotes = {"600519": _quote("2026-08-18T10:00:04")}
    result = assess_batch(quotes, ["600519", "000001"], now, 5, require_complete=True)
    assert result["usable"] is False
    assert result["partial"] is True
    assert result["missing_codes"] == ["000001"]
    assert result["coverage"] == {"requested": 2, "accepted": 1, "missing": 1}
    assert result["statuses"]["000001"]["missing_reason"] == "missing_quote"


def test_realtime_router_never_uses_tushare_or_unknown_source(monkeypatch):
    from core.data.sources import realtime
    from core.shared import config

    calls = []
    observed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    class FakeSource:
        def get_quotes(self, symbols):
            calls.append(tuple(symbols))
            return {"600519": {"price": 100.0, "date": observed_at[:10], "time": observed_at[11:]}}

    monkeypatch.setattr(config, "REALTIME_QUOTE_CHAIN", ("tushare", "tencent"))
    monkeypatch.setattr(realtime, "get_source", lambda name: FakeSource())
    result = realtime.RealtimeEngine().get_quotes(["600519"])

    assert result["600519"]["source"] == "tencent"
    assert len(calls) == 1


def test_realtime_router_preserves_exchange_qualified_index_code(monkeypatch):
    from core.data.sources import realtime

    calls = []
    observed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    class FakeSource:
        def get_quotes(self, symbols):
            calls.append(tuple(symbols))
            return {"000001": {"price": 3823.62, "date": observed_at[:10], "time": observed_at[11:]}}

    monkeypatch.setattr(realtime, "get_source", lambda name: FakeSource())

    result = realtime.RealtimeEngine().get_quotes(["sh000001"], source_chain=("tencent",))

    assert calls == [("sh000001",)]
    assert result["000001"]["price"] == 3823.62


def test_tencent_quote_normalizes_amount_from_ten_thousand_yuan():
    from core.data.sources.tencent import TencentSource
    from core.data.sources.realtime import RealtimeEngine

    fields = [""] * 40
    fields[1] = "上证指数"
    fields[2] = "000001"
    fields[3] = "3823.62"
    fields[4] = "3888.37"
    fields[5] = "3878.41"
    fields[6] = "452350675"
    fields[30] = "20260928161401"
    fields[33] = "3878.41"
    fields[34] = "3806.67"
    fields[37] = "80454370"

    result = TencentSource()._parse_response("~".join(fields))["000001"]

    assert result["amount"] == 804543700000
    assert result["date"] == "20260928"
    assert result["time"] == "16:14:01"
    assert RealtimeEngine._parse_quote_time(result) == datetime(2026, 9, 28, 16, 14, 1)


def test_daily_quotes_conversion_preserves_qfq_contract():
    from core.data.repos.quotes import DailyQuotesRepo

    frame = pd.DataFrame({
        "trade_date": ["2026-08-17", "2026-08-18"],
        "open": [5.0, 20.0],
        "high": [6.0, 22.0],
        "low": [4.0, 19.0],
        "close": [5.0, 20.0],
        "adj_factor": [2.0, 4.0],
        "volume": [100000, 200000],
        "amount": [1000000, 2000000],
    })

    raw = DailyQuotesRepo.convert_from_qfq(frame, "")
    hfq = DailyQuotesRepo.convert_from_qfq(frame, "hfq")

    assert raw["close"].tolist() == [10.0, 20.0]
    assert hfq["close"].tolist() == [20.0, 80.0]
    assert raw["volume"].tolist() == frame["volume"].tolist()
    assert raw["amount"].tolist() == frame["amount"].tolist()


def test_tushare_daily_normalizes_volume_and_amount_units(monkeypatch):
    from core.data.sources import tushare

    class FakePro:
        def daily(self, **kwargs):
            assert "adj" not in kwargs
            return pd.DataFrame([{
                "trade_date": "20260818",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "vol": 1234,
                "amount": 5678,
            }])

        def adj_factor(self, **kwargs):
            return pd.DataFrame([{
                "trade_date": "20260818",
                "adj_factor": 2.0,
            }])

    monkeypatch.setattr(tushare, "get_pro", lambda: FakePro())
    monkeypatch.setattr(tushare, "_rate_limit", lambda: None)
    result = tushare.get_daily_kline("600519", "20260818", "20260818", adj="qfq")

    assert result.iloc[0]["volume"] == 123400
    assert result.iloc[0]["amount"] == 5678000


def test_tencent_kline_respects_requested_adjustment(monkeypatch):
    from core.data.sources import tencent_kline

    class Response:
        def json(self):
            return {
                "code": 0,
                "data": {
                    "sh600519": {
                        "qfqday": [["20260818", "10", "11", "12", "9", "100", "1000"]],
                        "hfqday": [["20260818", "20", "22", "24", "18", "100", "2000"]],
                        "day": [["20260818", "30", "33", "36", "27", "100", "3000"]],
                    }
                },
            }

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(tencent_kline, "get_shared_session", lambda: Session())
    fetch = tencent_kline.fetch_tencent_kline_raw("600519", adj="qfq")[1][0]
    assert fetch["close"] == 11
    assert tencent_kline.fetch_tencent_kline_raw("600519", adj="hfq")[1][0]["close"] == 22
    assert tencent_kline.fetch_tencent_kline_raw("600519", adj="")[1][0]["close"] == 33
