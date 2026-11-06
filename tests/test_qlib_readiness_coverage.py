import datetime

import pytest

from backend.service.market_data import MarketDataService


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_readiness_fails_when_tradable_stable_pool_instrument_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.service.market_data.config.QLIB_DATA_ROOT", str(tmp_path / "qlib"))
    MarketDataService._qlib_readiness_cache.clear()
    class Bridge:
        def __init__(self, data_dir):
            pass

        def get_status(self):
            return {"synced": True, "last_date": "2026-09-23"}

        def get_date_coverage(self, decision_date, expected_codes):
            assert expected_codes == ["000001", "000002"]
            return {
                "expected": 2,
                "available": 1,
                "coverage_rate": 0.5,
                "missing_codes": ["000002"],
            }

    monkeypatch.setattr("core.adapters.qlib.bridge.QlibDataBridge", Bridge)
    monkeypatch.setattr("backend.service.market_data.config.COLLECT_MIN_DAILY_ROWS", 1)
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
    assert readiness["coverage_rate"] == 0.5
    assert readiness["missing_codes"] == ["000002"]


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_readiness_reuses_short_lived_deep_check(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.service.market_data.config.QLIB_DATA_ROOT", str(tmp_path / "qlib"))
    calls = []

    class Bridge:
        def __init__(self, data_dir):
            pass

        def get_status(self):
            return {"synced": True, "last_date": "2026-09-23"}

        def get_date_coverage(self, decision_date, expected_codes):
            calls.append(decision_date)
            return {
                "expected": 1, "available": 1, "coverage_rate": 1.0,
                "missing_codes": [],
            }

    MarketDataService._qlib_readiness_cache.clear()
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

    first = MarketDataService.get_qlib_readiness(datetime.date(2026, 9, 23))
    second = MarketDataService.get_qlib_readiness(datetime.date(2026, 9, 23))

    assert first == second
    assert calls == ["2026-09-23"]
