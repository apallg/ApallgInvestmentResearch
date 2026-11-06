import datetime

import pandas as pd
import pytest

from backend.service.market_data import MarketDataService


@pytest.mark.unit
@pytest.mark.p0
def test_default_screening_decision_date_uses_latest_complete_daily_quote(monkeypatch):
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_latest_complete_date",
        lambda minimum_rows: datetime.date(2026, 9, 23),
    )

    decision_date = MarketDataService.get_screening_decision_date()

    assert decision_date == datetime.date(2026, 9, 23)


@pytest.mark.unit
@pytest.mark.p0
def test_live_index_quotes_preserve_exchange_prefix(monkeypatch):
    requested = []

    def get_quotes(_self, codes, source_chain=None):
        requested.extend(codes)
        observed_at = datetime.datetime.now().isoformat(timespec="seconds")
        return {
            "000001": {
                "name": "上证指数",
                "price": 3823.62,
                "source": "tencent",
                "observed_at": observed_at,
                "received_at": observed_at,
                "session_date": datetime.date.today().isoformat(),
            }
        }

    monkeypatch.setattr("backend.service.market_data.RealtimeEngine.get_quotes", get_quotes)

    result = MarketDataService.get_live_quotes(["sh000001"], "live_display", "index")

    assert requested == ["sh000001"]
    assert result["data"]["000001"]["name"] == "上证指数"
    assert result["meta"]["coverage"] == {"requested": 1, "accepted": 1, "missing": 0}
    assert result["meta"]["data_envelope"]["schema"] == "data-envelope-v2"
    assert result["meta"]["data_envelope"]["status"] == "available"


@pytest.mark.unit
def test_live_quotes_empty_request_returns_complete_unavailable_metadata():
    result = MarketDataService.get_live_quotes([])

    assert result["data"] == {}
    assert result["meta"]["freshness"] == "unavailable"
    assert result["meta"]["coverage"] == {"requested": 0, "accepted": 0, "missing": 0}
    assert result["meta"]["data_envelope"]["status"] == "unavailable"


@pytest.mark.unit
@pytest.mark.parametrize("use_case", ["execuation", "broker", "", [], {}])
def test_unknown_live_quote_use_case_fails_closed_before_provider_access(monkeypatch, use_case):
    monkeypatch.setattr(
        "backend.service.market_data.RealtimeEngine.get_quotes",
        lambda *_args, **_kwargs: pytest.fail("未知用途不得读取实时行情"),
    )

    result = MarketDataService.get_live_quotes(["000001"], use_case)

    assert result["data"] == {}
    assert result["meta"]["use_case_eligible"] is False
    assert result["meta"]["freshness"] == "unavailable"
    assert result["meta"]["coverage"] == {"requested": 1, "accepted": 0, "missing": 1}
    assert result["meta"]["quote_statuses"]["000001"]["missing_reason"] == "unsupported_use_case"


@pytest.mark.unit
@pytest.mark.parametrize("use_case", ["execution", "risk"])
@pytest.mark.parametrize("missing_field", ["source", "received_at"])
def test_live_use_cases_fail_closed_when_quote_provenance_is_incomplete(monkeypatch, use_case, missing_field):
    observed_at = datetime.datetime.now().isoformat(timespec="seconds")
    quote = {
        "name": "测试",
        "price": 10.0,
        "source": "tencent",
        "observed_at": observed_at,
        "received_at": observed_at,
        "session_date": datetime.date.today().isoformat(),
        "price_basis": "raw",
    }
    quote.pop(missing_field)
    monkeypatch.setattr(
        "backend.service.market_data.RealtimeEngine.get_quotes",
        lambda *_args, **_kwargs: {"000001": quote},
    )

    result = MarketDataService.get_live_quotes(["000001"], use_case)

    assert result["data"] == {}
    assert result["meta"]["use_case_eligible"] is False
    assert result["meta"]["freshness"] == "unavailable"
    assert result["meta"]["coverage"] == {"requested": 1, "accepted": 0, "missing": 1}
    assert result["meta"]["source_coverage"]["accepted"] == 1
    assert result["meta"]["data_envelope"]["status"] == "unavailable"
    assert result["meta"]["data_envelope"]["use_case_blocked"] == use_case


@pytest.mark.unit
@pytest.mark.parametrize("use_case", ["execution", "risk"])
def test_live_use_cases_accept_fresh_complete_provenance(monkeypatch, use_case):
    observed_at = datetime.datetime.now().isoformat(timespec="seconds")
    monkeypatch.setattr(
        "backend.service.market_data.RealtimeEngine.get_quotes",
        lambda *_args, **_kwargs: {"000001": {
            "name": "测试",
            "price": 10.0,
            "source": "tencent",
            "observed_at": observed_at,
            "received_at": observed_at,
            "session_date": datetime.date.today().isoformat(),
            "price_basis": "raw",
        }},
    )

    result = MarketDataService.get_live_quotes(["000001"], use_case)

    assert "000001" in result["data"]
    assert result["meta"]["use_case_eligible"] is True
    assert result["meta"]["data_envelope"]["status"] == "available"


@pytest.mark.unit
def test_live_use_gate_checks_each_quote_not_only_aggregate_source(monkeypatch):
    observed_at = datetime.datetime.now().isoformat(timespec="seconds")
    quote = {
        "price": 10.0,
        "source": "tencent",
        "observed_at": observed_at,
        "received_at": observed_at,
        "session_date": datetime.date.today().isoformat(),
        "price_basis": "raw",
    }
    incomplete_quote = {**quote}
    incomplete_quote.pop("source")
    monkeypatch.setattr(
        "backend.service.market_data.RealtimeEngine.get_quotes",
        lambda *_args, **_kwargs: {"000001": quote, "000002": incomplete_quote},
    )

    result = MarketDataService.get_live_quotes(["000001", "000002"], "execution")

    assert result["data"] == {}
    assert result["meta"]["use_case_eligible"] is False
    assert result["meta"]["row_contract_failures"] == {"000002": ["source_missing"]}
    assert any(item == "000002:source_missing"
               for item in result["meta"]["data_envelope"]["validation_errors"])


@pytest.mark.unit
def test_live_use_gate_rejects_received_time_before_observation(monkeypatch):
    observed_at = datetime.datetime.now()
    quote = {
        "price": 10.0,
        "source": "tencent",
        "observed_at": observed_at.isoformat(timespec="seconds"),
        "received_at": (observed_at - datetime.timedelta(seconds=1)).isoformat(timespec="seconds"),
        "session_date": datetime.date.today().isoformat(),
        "price_basis": "raw",
    }
    monkeypatch.setattr(
        "backend.service.market_data.RealtimeEngine.get_quotes",
        lambda *_args, **_kwargs: {"000001": quote},
    )

    result = MarketDataService.get_live_quotes(["000001"], "execution")

    assert result["data"] == {}
    assert result["meta"]["use_case_eligible"] is False
    assert "available_before_observed" in result["meta"]["data_envelope"]["validation_errors"]


@pytest.mark.unit
@pytest.mark.p0
def test_live_sector_snapshot_with_missing_quotes_is_partial(monkeypatch):
    from datetime import date
    from core.data.repos import stock_industry_history

    monkeypatch.setattr("core.shared.trading_time.market_date", lambda: date(2026, 10, 30))
    monkeypatch.setattr(
        stock_industry_history,
        "get_latest_available_industry_snapshot",
        lambda as_of: {
            "payload_sha256": "snapshot-hash",
            "available_after_date": "2026-10-20",
            "source_name": "industry-source",
            "snapshot_id": "snapshot-1",
            "classification_standard": "test",
            "observed_at": "2026-10-29T15:00:00+08:00",
        },
    )
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_latest_complete_date",
        lambda *args, **kwargs: date(2026, 10, 30),
    )
    monkeypatch.setattr(
        stock_industry_history,
        "get_industry_snapshot_market_stats",
        lambda *_: [{
            "sector_name": "行业A", "member_count": 2, "quote_count": 1,
            "pct_chg": 1.5,
        }],
    )

    result = MarketDataService.get_live_sectors()

    assert result["meta"]["data_envelope"]["status"] == "partial"
    assert result["meta"]["data_envelope"]["missing_reason"] == "daily_quote_coverage_incomplete"
    assert result["data"][0]["rank_eligible"] is False
    assert result["data"][0]["pct_chg"] is None


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_readiness_rejects_calendar_behind_decision_date(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.service.market_data.config.QLIB_DATA_ROOT", str(tmp_path / "qlib"))
    class Bridge:
        def __init__(self, data_dir):
            pass

        def get_status(self):
            return {"synced": True, "last_date": "2026-09-21"}

    monkeypatch.setattr("core.adapters.qlib.bridge.QlibDataBridge", Bridge)
    monkeypatch.setattr("backend.service.market_data.config.COLLECT_MIN_DAILY_ROWS", 1)
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_tradable_codes_for_date",
        lambda *args, **kwargs: ["000001"],
    )
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_valid_row_count_for_date",
        lambda *args, **kwargs: 1,
    )

    readiness = MarketDataService.get_qlib_readiness(datetime.date(2026, 9, 23))

    assert readiness == {
        "ready": False,
        "decision_date": "2026-09-23",
        "latest_date": "2026-09-21",
        "daily_rows": 1,
        "minimum_rows": 1,
        "expected": 1,
        "available": 0,
        "coverage_rate": 0.0,
        "missing_codes": ["000001"],
    }


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_readiness_accepts_calendar_covering_historical_date(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.service.market_data.config.QLIB_DATA_ROOT", str(tmp_path / "qlib"))
    class Bridge:
        def __init__(self, data_dir):
            pass

        def get_status(self):
            return {"synced": True, "last_date": "2026-09-23"}

        def get_date_coverage(self, decision_date, expected_codes):
            return {
                "expected": len(expected_codes), "available": len(expected_codes),
                "coverage_rate": 1.0, "missing_codes": [],
            }

    monkeypatch.setattr("core.adapters.qlib.bridge.QlibDataBridge", Bridge)
    monkeypatch.setattr("backend.service.market_data.config.COLLECT_MIN_DAILY_ROWS", 1)
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_tradable_codes_for_date",
        lambda *args, **kwargs: ["000001"],
    )
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_valid_row_count_for_date",
        lambda *args, **kwargs: 1,
    )

    readiness = MarketDataService.get_qlib_readiness(datetime.date(2026, 9, 21))

    assert readiness["ready"] is True


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_readiness_rejects_partial_daily_quote_batch(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.service.market_data.config.QLIB_DATA_ROOT", str(tmp_path / "qlib"))
    class Bridge:
        def __init__(self, data_dir):
            pass

        def get_status(self):
            return {"synced": True, "last_date": "2026-09-23"}

        def get_date_coverage(self, decision_date, expected_codes):
            return {
                "expected": len(expected_codes), "available": len(expected_codes),
                "coverage_rate": 1.0, "missing_codes": [],
            }

    MarketDataService._qlib_readiness_cache.clear()
    monkeypatch.setattr("core.adapters.qlib.bridge.QlibDataBridge", Bridge)
    monkeypatch.setattr("backend.service.market_data.config.COLLECT_MIN_DAILY_ROWS", 4800)
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_tradable_codes_for_date",
        lambda *args, **kwargs: ["000001", "000002"],
    )
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_valid_row_count_for_date",
        lambda *args, **kwargs: 2,
    )

    readiness = MarketDataService.get_qlib_readiness(datetime.date(2026, 9, 23))

    assert readiness["ready"] is False
    assert readiness["daily_rows"] == 2


@pytest.mark.unit
def test_conditional_market_env_is_built_from_repository_rows(monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=21)
    rows = [
        {"trade_date": date, "breadth": 0.6, "mean_ret": 1.0}
        for date in dates
    ]
    monkeypatch.setattr(
        "core.data.repos.quotes.DailyQuotesRepo.get_market_daily_aggregates",
        lambda start_date, end_date: rows,
    )

    env = MarketDataService.get_conditional_market_env(end_date=dates[-1])

    assert env[dates[-1].strftime("%Y-%m-%d")][0] == 0.6
    assert env[dates[-1].strftime("%Y-%m-%d")][1] < 0


@pytest.mark.unit
@pytest.mark.parametrize("end", ["2025-02-01", "2026-02-01"])
def test_market_aggregate_chunks_cover_range_without_gaps(monkeypatch, end):
    from core.data.repos.quotes import DailyQuotesRepo

    requested = []

    def fetchall(sql, params):
        dates = pd.date_range(params[0], params[1])
        requested.extend(dates)
        return [{"trade_date": date} for date in dates]

    monkeypatch.setattr("core.data.repos.quotes.Database.fetchall", fetchall)
    rows = DailyQuotesRepo.get_market_daily_aggregates("2025-01-01", end)

    assert requested == list(pd.date_range("2025-01-01", end))
    assert len(rows) == len(requested)


@pytest.mark.unit
@pytest.mark.p0
def test_backtest_environment_covers_start_with_complete_warmup(monkeypatch):
    from backend.api.backtest import DirectionBacktestRequest
    from backend.service.backtest import BacktestService

    class Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return kwargs

    def aggregates(start, end):
        return [
            {"trade_date": date, "breadth": 0.4, "mean_ret": -1.0}
            for date in pd.bdate_range(start, end)
        ]

    monkeypatch.setattr("core.domain.backtest.engine.DirectionBacktestRunner", Runner)
    monkeypatch.setattr("core.data.repos.quotes.DailyQuotesRepo.get_market_daily_aggregates", aggregates)
    params = BacktestService.run_direction(DirectionBacktestRequest(
        start="2025-09-01", end="2026-08-31", use_vol_switch=True,
        use_conditional_switch=True,
    ))
    env = params["conditional_market_env"]

    assert env["2025-09-01"] == pytest.approx((0.4, 1 - 0.99 ** 20))
    assert env["2026-08-31"] == pytest.approx((0.4, 1 - 0.99 ** 20))
