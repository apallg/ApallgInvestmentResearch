import numpy as np
import pandas as pd

from scripts.multiscale_research_features import build_multiscale_features, build_style_groups


def test_week_and_month_only_become_visible_after_completion():
    calendar = pd.bdate_range("2025-01-01", "2026-04-30")
    dates = calendar[calendar >= "2025-01-02"]
    index = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
    values = np.linspace(10, 20, len(index))
    quotes = pd.DataFrame({"open": values, "high": values + 1, "low": values - 1,
                           "close": values, "volume": np.full(len(index), 100.0)}, index=index)
    first = build_multiscale_features(quotes, calendar)
    changed = quotes.copy()
    changed.loc[(pd.Timestamp("2026-04-03"), "sh600000"), "close"] = 1000.0
    changed.loc[(pd.Timestamp("2026-04-30"), "sh600000"), "close"] = 1000.0
    second = build_multiscale_features(changed, calendar)
    pd.testing.assert_series_equal(first.loc[(pd.Timestamp("2026-04-01"), "sh600000"), first.columns.str.startswith("week_")],
                                   second.loc[(pd.Timestamp("2026-04-01"), "sh600000"), second.columns.str.startswith("week_")])
    pd.testing.assert_series_equal(first.loc[(pd.Timestamp("2026-04-15"), "sh600000"), first.columns.str.startswith("month_")],
                                   second.loc[(pd.Timestamp("2026-04-15"), "sh600000"), second.columns.str.startswith("month_")])
    assert first.loc[(pd.Timestamp("2026-04-06"), "sh600000"), "week_return_1"] != second.loc[(pd.Timestamp("2026-04-06"), "sh600000"), "week_return_1"]
    assert first.loc[(pd.Timestamp("2026-04-30"), "sh600000"), "month_return_1"] != second.loc[(pd.Timestamp("2026-04-30"), "sh600000"), "month_return_1"]


def test_style_group_uses_previous_month_snapshot():
    calendar = pd.bdate_range("2025-01-01", "2025-03-31")
    instruments = ["sh600000", "sz000001"]
    index = pd.MultiIndex.from_product([calendar, instruments], names=["datetime", "instrument"])
    context = pd.DataFrame({"context_log_circ_mv": 1.0, "context_log_amount_mean60": 1.0}, index=index)
    context.loc[(slice(None), "sz000001"), :] = 2.0
    volatility = pd.Series(np.where(index.get_level_values("instrument") == "sz000001", 2.0, 1.0), index=index)
    first = build_style_groups(context, volatility, calendar)
    altered = context.copy()
    altered.loc[(pd.Timestamp("2025-02-28"), "sh600000"), "context_log_circ_mv"] = 10.0
    second = build_style_groups(altered, volatility, calendar)
    assert pd.isna(first.loc[(pd.Timestamp("2025-01-10"), "sh600000")])
    assert first.loc[(pd.Timestamp("2025-02-10"), "sh600000")] == second.loc[(pd.Timestamp("2025-02-10"), "sh600000")]
    assert first.loc[(pd.Timestamp("2025-03-10"), "sh600000")] != second.loc[(pd.Timestamp("2025-03-10"), "sh600000")]


def test_holiday_year_boundary_and_missing_last_stock_bar():
    calendar = pd.bdate_range("2024-01-01", "2026-01-09").difference(pd.DatetimeIndex(["2025-12-25", "2026-01-01"]))
    dates = calendar.difference(pd.DatetimeIndex(["2025-12-31"]))
    index = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
    values = np.arange(len(index), dtype=float) + 10
    quotes = pd.DataFrame({"open": values, "high": values + 1, "low": values - 1,
                           "close": values, "volume": 100.0}, index=index)
    first = build_multiscale_features(quotes, calendar)
    changed = quotes.copy()
    changed.loc[(pd.Timestamp("2026-01-02"), "sh600000"), "close"] = 9999.0
    second = build_multiscale_features(changed, calendar)
    assert first.loc[(pd.Timestamp("2025-12-30"), "sh600000"), "month_return_1"] == second.loc[(pd.Timestamp("2025-12-30"), "sh600000"), "month_return_1"]
    assert first.loc[(pd.Timestamp("2026-01-02"), "sh600000"), "month_return_1"] == second.loc[(pd.Timestamp("2026-01-02"), "sh600000"), "month_return_1"]
    assert first.loc[(pd.Timestamp("2026-01-05"), "sh600000"), "week_return_1"] != second.loc[(pd.Timestamp("2026-01-05"), "sh600000"), "week_return_1"]


def test_full_week_and_month_suspension_do_not_shorten_returns():
    calendar = pd.bdate_range("2024-01-01", "2026-04-30")
    suspended = ((calendar >= "2026-02-01") & (calendar <= "2026-02-28")) | ((calendar >= "2026-04-06") & (calendar <= "2026-04-10"))
    dates = calendar[~suspended]
    index = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
    close = np.arange(len(index), dtype=float) + 10
    quotes = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": 100.0}, index=index)
    features = build_multiscale_features(quotes, calendar)
    assert np.isnan(features.loc[(pd.Timestamp("2026-03-02"), "sh600000"), "month_return_1"])
    assert np.isnan(features.loc[(pd.Timestamp("2026-04-20"), "sh600000"), "week_return_1"])
