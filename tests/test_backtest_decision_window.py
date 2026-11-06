import pandas as pd
import pytest

from core.domain.backtest.engine import _backtest_decision_dates
from core.shared import get_freq_meta
from scripts.run_rank_persistence_forward import (
    terminal_execution_session,
    weekly_decision_dates,
)


def test_extended_price_window_does_not_add_a_later_weekly_decision():
    index_df = pd.DataFrame({
        "trade_date": ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09"],
    })

    dates = _backtest_decision_dates(
        "2026-09-28", "2026-10-09", "2026-09-30", "W-THU",
        get_freq_meta("week"), index_df,
    )

    assert [value.isoformat() for value in dates] == ["2026-09-30"]


def test_decision_end_must_be_inside_market_data_window():
    with pytest.raises(ValueError, match="decision_end"):
        _backtest_decision_dates(
            "2026-09-28", "2026-10-08", "2026-10-09", "W-THU",
            get_freq_meta("week"), None,
        )


def test_default_decision_window_preserves_project_weekly_anchor_calendar():
    calendar_path = "data/qlib_data/calendars/day.txt"
    calendar_dates = [
        value for value in pd.read_csv(calendar_path, header=None)[0].astype(str).tolist()
        if "2020-01-02" <= value <= "2026-09-24"
    ]
    index_df = pd.DataFrame({"trade_date": calendar_dates})
    actual = _backtest_decision_dates(
        "2020-01-02", "2026-09-24", None, "W-THU",
        get_freq_meta("week"), index_df,
    )
    expected = weekly_decision_dates(
        [pd.Timestamp(value).date() for value in calendar_dates],
        [pd.Timestamp(value).date() for value in calendar_dates],
        pd.Timestamp("2026-09-24").date(),
    )

    assert actual == expected
    assert len(actual) == 347


def test_terminal_return_waits_when_execution_session_is_not_in_calendar():
    assert terminal_execution_session(
        pd.Timestamp("2026-09-24").date(),
        [pd.Timestamp("2026-09-24").date()],
    ) is None


def test_terminal_execution_session_uses_first_post_holiday_session():
    assert terminal_execution_session(
        pd.Timestamp("2026-09-30").date(),
        [
            pd.Timestamp(value).date()
            for value in ("2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09")
        ],
    ) == pd.Timestamp("2026-10-08").date()
