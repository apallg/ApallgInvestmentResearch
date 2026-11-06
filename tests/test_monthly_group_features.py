import numpy as np
import pandas as pd

from scripts.monthly_group_features import build_classifications, build_monthly_features
from scripts.multiscale_research_features import build_multiscale_features


def test_month_features_match_frozen_implementation_and_hide_future_month():
    calendar = pd.bdate_range("2024-01-01", "2026-04-30")
    dates = calendar.difference(pd.bdate_range("2026-02-01", "2026-02-28"))
    index = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
    prices = np.arange(len(index), dtype=float) + 10
    quotes = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1,
                           "close": prices, "volume": 100.0}, index=index)
    month = build_monthly_features(quotes, calendar)
    prior = build_multiscale_features(quotes, calendar)
    pd.testing.assert_frame_equal(month, prior[month.columns])
    altered = quotes.copy()
    altered.loc[(pd.Timestamp("2026-04-30"), "sh600000"), "close"] = 1000.0
    changed = build_monthly_features(altered, calendar)
    pd.testing.assert_series_equal(month.loc[(pd.Timestamp("2026-04-15"), "sh600000")],
                                   changed.loc[(pd.Timestamp("2026-04-15"), "sh600000")])
    assert np.isnan(month.loc[(pd.Timestamp("2026-03-02"), "sh600000"), "month_return_1"])


def test_classification_uses_previous_month_snapshot():
    calendar = pd.bdate_range("2025-01-01", "2025-03-31")
    index = pd.MultiIndex.from_product([calendar, ["sh600000", "sz000001"]], names=["datetime", "instrument"])
    code = index.get_level_values("instrument")
    values = np.where(code == "sh600000", 1.0, 2.0)
    context = pd.DataFrame({"context_log_circ_mv": values, "context_log_amount_mean60": values,
                            "context_missing_size": 0, "context_missing_liquidity": 0,
                            "context_missing_market_relation": 0, "context_beta120": values,
                            "context_corr120": values, "context_residvol120": values}, index=index)
    quotes = pd.DataFrame({"close": np.where(code == "sh600000", 10.0, 20.0)}, index=index)
    industry = pd.Series(0, index=index)
    first, _ = build_classifications(context, pd.DataFrame(index=index), quotes, industry, calendar)
    altered = context.copy()
    altered.loc[(pd.Timestamp("2025-02-28"), "sh600000"), "context_log_circ_mv"] = 10.0
    second, _ = build_classifications(altered, pd.DataFrame(index=index), quotes, industry, calendar)
    assert (first.loc[(pd.Timestamp("2025-02-10"), "sh600000")] == second.loc[(pd.Timestamp("2025-02-10"), "sh600000")]).all()
    assert first.loc[(pd.Timestamp("2025-03-10"), "sh600000"), "SL4"] != second.loc[(pd.Timestamp("2025-03-10"), "sh600000"), "SL4"]
    assert (first.loc[(pd.Timestamp("2025-01-10"), "sh600000")] == -1).all()


def test_month_end_holiday_and_year_boundary_publish_after_close():
    calendar = pd.bdate_range("2023-01-02", "2025-01-31").difference(pd.DatetimeIndex(["2024-12-31", "2025-01-01"]))
    index = pd.MultiIndex.from_product([calendar, ["sh600000"]], names=["datetime", "instrument"])
    quotes = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0}, index=index)
    changed = quotes.copy()
    changed.loc[(pd.Timestamp("2024-12-30"), "sh600000"), "close"] = 20.0
    before = build_monthly_features(quotes, calendar)
    after = build_monthly_features(changed, calendar)
    assert before.loc[(pd.Timestamp("2024-12-27"), "sh600000"), "month_return_1"] == after.loc[(pd.Timestamp("2024-12-27"), "sh600000"), "month_return_1"]
    assert after.loc[(pd.Timestamp("2024-12-30"), "sh600000"), "month_return_1"] == 1.0
    assert after.loc[(pd.Timestamp("2025-01-02"), "sh600000"), "month_return_1"] == 1.0
