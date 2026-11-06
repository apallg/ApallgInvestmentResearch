import datetime

import pytest
from fastapi import HTTPException

from backend.service.instrument_resolver import (
    _stable_history_500d_instruments,
    build_filtered_instruments,
    resolve_instruments,
)
from core.shared import config
from backend.service.training import TrainingService
from core.data.repos.instruments import InstrumentRepo


def test_stable_history_query_uses_500th_valid_unique_observation(monkeypatch):
    calls = []

    def fetchall(sql, params=None):
        calls.append((sql, params))
        return [{
            "code": "600000",
            "eligible_start": datetime.date(2022, 1, 4),
            "eligible_end": datetime.date(2026, 10, 30),
        }]

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)

    result = InstrumentRepo.get_stable_history_500d_ranges(["600000", "000001"])

    assert result == {
        "600000": (datetime.date(2022, 1, 4), datetime.date(2026, 10, 30)),
    }
    sql, params = calls[0]
    assert "ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date)" in sql
    assert "history_day = 500" in sql
    assert "close > 0" in sql
    assert "HAVING COUNT(*) >= 500" in sql
    assert params == ("000001", "600000")


def test_stable_history_resolver_returns_point_in_time_ranges(monkeypatch):
    monkeypatch.setattr(
        "core.data.repos.instruments.InstrumentRepo.get_stable_history_500d_ranges",
        lambda codes=None: {
            "600000": (datetime.date(2022, 1, 4), datetime.date(2026, 10, 30)),
            "000001": (datetime.date(2023, 1, 3), datetime.date(2026, 10, 30)),
        },
    )

    result = _stable_history_500d_instruments(["60"])

    assert result == {
        "sh600000": ["2022-01-04", "2026-10-30"],
    }


def test_a500_named_pool_resolves_to_intervals_instead_of_static_alias(monkeypatch):
    expected = {"sh600000": ["2022-01-04", "2026-10-30"]}
    monkeypatch.setattr(
        "backend.service.instrument_resolver._stable_history_500d_instruments",
        lambda code_prefixes=None: expected,
    )

    assert resolve_instruments("a500") == expected
    assert resolve_instruments("stable_history_500d") == expected


def test_all_pool_prefix_filter_uses_historical_quote_ranges(monkeypatch):
    monkeypatch.setattr(config, "MYSQL_ENABLED", True)
    monkeypatch.setattr(
        "core.data.repos.instruments.InstrumentRepo.get_quote_instrument_ranges",
        lambda: {
            "600000": (datetime.date(2000, 1, 1), datetime.date(2026, 10, 30)),
            "000001": (datetime.date(1991, 4, 3), datetime.date(2026, 10, 30)),
        },
    )

    result = build_filtered_instruments("all", ["60"])

    assert result == {
        "sh600000": ["2000-01-01", "2026-10-30"],
    }


@pytest.mark.parametrize("pool", ["csi300", "csi500", "csi300+csi500", "small_cap"])
def test_training_rejects_current_only_universes(pool):
    with pytest.raises(HTTPException) as exc:
        TrainingService._require_historical_training_ready(pool)

    assert exc.value.status_code == 409
    assert "unsupported_historical_pools" in exc.value.detail


def test_training_rejects_composite_pool_that_would_erase_500d_intervals():
    with pytest.raises(HTTPException) as exc:
        TrainingService._require_historical_training_ready("a500+all")

    assert exc.value.status_code == 409
    assert exc.value.detail["history_pools"] == ["a500"]


def test_training_continues_to_historical_quality_preflight_for_supported_pool(monkeypatch):
    monkeypatch.setattr(
        "backend.service.historical_quality.get_historical_preflight",
        lambda: {"status": "ready", "blockers": []},
    )

    TrainingService._require_historical_training_ready("stable_history_500d")
