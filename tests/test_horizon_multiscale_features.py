import numpy as np
import pandas as pd

from scripts.horizon_multiscale_features import build_monthly_bars, monthly_history_for_date
from scripts.horizon_multiscale_research import make_calendar_labels
from scripts.horizon_sequence_features import build_daily_long_features, build_monthly_sequence_table, daily_window_positions
from scripts.horizon_sequence_models import HorizonFusionNet
from scripts.horizon_multiscale_research import fold_windows
import scripts.horizon_multiscale_features as feature_module


def test_monthly_history_is_calendar_padded_and_only_publishes_completed_months():
    calendar = pd.bdate_range("2024-01-01", "2024-04-30")
    dates = calendar[(calendar <= pd.Timestamp("2024-04-12")) & (calendar.to_period("M") != pd.Period("2024-02", freq="M"))]
    index = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
    quotes = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
                           "volume": 100.0}, index=index)
    bars = build_monthly_bars(quotes, calendar)
    mid_month = monthly_history_for_date(bars, "sh600000", "2024-04-12", calendar)
    month_end = monthly_history_for_date(bars, "sh600000", "2024-04-30", calendar)
    assert mid_month.index[-1] == pd.Period("2024-03", freq="M")
    assert month_end.index[-1] == pd.Period("2024-04", freq="M")
    assert len(mid_month) == len(month_end) == 12
    assert mid_month.loc[pd.Period("2024-02", freq="M"), "valid_day_ratio"] == 0.0
    assert mid_month.loc[pd.Period("2024-02", freq="M"), "missing_close_gap"] == 1.0


def test_horizon_labels_use_shared_calendar_dates_without_skipping_missing_quote_days():
    calendar = pd.bdate_range("2024-01-01", periods=30)
    dates = calendar[:20]
    index = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
    opens = pd.Series(np.arange(20, dtype=float) + 10, index=index)
    opens.loc[(calendar[5], "sh600000")] = np.nan
    panel = pd.DataFrame({"open": opens})
    label, end = make_calendar_labels(panel, calendar, 10)
    decision_date = calendar[1]
    assert end.loc[(decision_date, "sh600000")] == calendar[12]
    assert np.isnan(label.loc[(calendar[4], "sh600000")])


def test_monthly_routes_do_not_use_future_prices(monkeypatch):
    calendar = pd.bdate_range("2024-01-01", periods=130)
    instruments = [f"sh{600000 + number}" for number in range(22)]
    index = pd.MultiIndex.from_product([calendar, instruments], names=["datetime", "instrument"])
    days = np.tile(np.arange(len(calendar), dtype=float), len(instruments))
    stocks = np.repeat(np.arange(len(instruments), dtype=float), len(calendar))
    close = 20 + stocks * 0.03 + days * (0.01 + stocks * 0.0002)
    quotes = pd.DataFrame({"close": close}, index=index)
    context = pd.DataFrame({"context_board_main": 1.0, "context_board_star": 0.0,
        "context_board_chinext": 0.0, "context_board_beijing": 0.0}, index=index)
    monkeypatch.setattr(feature_module, "historical_industry", lambda rows: (pd.Series(0, index=rows, dtype=np.int16), ["I0"]))
    listing = {instrument: pd.Timestamp("2010-01-01") for instrument in instruments}
    first, _, _ = feature_module.build_groups(quotes, context, calendar, listing)
    altered = quotes.copy()
    future = quotes.index.get_level_values("datetime") > pd.Timestamp("2024-05-31")
    altered.loc[future, "close"] *= 4.0
    second, _, _ = feature_module.build_groups(altered, context, calendar, listing)
    june = (first.index.get_level_values("datetime").to_period("M") == pd.Period("2024-06", freq="M"))
    assert (first.loc[june, "R4"].to_numpy() == second.loc[june, "R4"].to_numpy()).all()
    assert (first.loc[june, "R4"] >= 0).all()


def test_sequence_features_keep_calendar_holes_and_future_independence():
    calendar = pd.bdate_range("2024-01-01", periods=300)
    dates = calendar.delete([15, 110])
    index = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
    values = np.arange(len(dates), dtype=float) + 10
    quotes = pd.DataFrame({"open": values, "high": values + 1, "low": values - 1,
                           "close": values, "volume": values * 100}, index=index)
    endpoints = index[[30, 150, 250]]
    positions = daily_window_positions(index, calendar, endpoints, length=20)
    assert positions.shape == (3, 20)
    assert (positions[0] == -1).any()
    daily = build_daily_long_features(quotes, calendar)
    changed = quotes.copy()
    future = changed.index.get_level_values("datetime") > endpoints[1][0]
    changed.loc[future, ["open", "high", "low", "close", "volume"]] *= 5
    changed_daily = build_daily_long_features(changed, calendar)
    assert np.allclose(daily.loc[:endpoints[1][0]].to_numpy(),
                       changed_daily.loc[:endpoints[1][0]].to_numpy(), equal_nan=True)
    monthly = build_monthly_sequence_table(quotes, calendar)
    changed_monthly = build_monthly_sequence_table(changed, calendar)
    endpoint_month = endpoints[1][0].to_period("M")
    assert np.allclose(monthly.loc[(slice(None), slice(None)), :].loc[
        monthly.index.get_level_values("month") < endpoint_month].to_numpy(),
        changed_monthly.loc[changed_monthly.index.get_level_values("month") < endpoint_month].to_numpy(),
        equal_nan=True)


def test_sequence_model_arms_accept_the_preregistered_shapes():
    import torch

    daily = torch.zeros((3, 20, 41))
    monthly = torch.zeros((3, 12, 12))
    for arm in ("N0", "NL", "NS"):
        assert HorizonFusionNet(41, arm)(daily).shape == (3,)
    for arm in ("NF", "NT", "NR"):
        assert HorizonFusionNet(41, arm)(daily, monthly).shape == (3,)


def test_four_month_calibration_window_has_three_month_early_stop_and_three_year_train():
    calendar = pd.bdate_range("2018-01-01", "2027-12-31")
    windows = fold_windows(calendar)
    first = windows["W1"]
    assert first["train"][0] == "2019-06-01"
    assert first["valid"][0] == "2022-06-01"
    assert first["calibrate"][0] == "2022-09-01"
    assert first["test"][0] == "2023-01-01"


def test_monthly_sequence_uses_last_completed_natural_month_only():
    from scripts.horizon_sequence_features import build_monthly_sequence_table
    from scripts.train_horizon_sequence import build_monthly_lookup, monthly_batch

    calendar = pd.bdate_range("2024-01-01", "2024-08-30")
    index = pd.MultiIndex.from_product([calendar, ["sh600000"]], names=["datetime", "instrument"])
    level = np.arange(len(calendar), dtype=np.float32)
    quotes = pd.DataFrame({"open": 10 + level, "high": 12 + level, "low": 9 + level,
                           "close": 11 + level, "volume": 100 + level}, index=index)
    table = build_monthly_sequence_table(quotes, calendar)
    lookup = build_monthly_lookup(table, calendar)
    endpoints = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-06-14"), "sh600000"),
                                           (pd.Timestamp("2024-06-28"), "sh600000")],
                                          names=["datetime", "instrument"])
    panel = pd.DataFrame(index=endpoints)
    rows = monthly_batch(panel.index, np.arange(len(endpoints)), lookup, calendar)
    assert rows.shape == (2, 12, 12)
    assert not np.array_equal(rows[0, -1, :6], rows[1, -1, :6])

