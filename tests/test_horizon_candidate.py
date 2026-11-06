import numpy as np
import pandas as pd
import pytest

from core.adapters.qlib.horizon_candidate import blend_rank, fit_rank_weight, fit_residual_weight, mature_update_dates, percentile_rank


def test_weight_uses_matching_percentile_label_scale():
    index = pd.MultiIndex.from_product([pd.to_datetime(["2023-01-03", "2023-01-04"]), ["a", "b", "c"]], names=["datetime", "instrument"])
    g = pd.Series([1, 2, 3, 3, 2, 1], index=index)
    n = pd.Series([3, 2, 1, 1, 2, 3], index=index)
    label = percentile_rank(n)
    weight = fit_rank_weight(percentile_rank(g), percentile_rank(n), label)
    assert weight > 0.9
    assert np.allclose(blend_rank(percentile_rank(g), percentile_rank(n), 1), label)


def test_weight_rejects_changed_denominator():
    index = pd.MultiIndex.from_product([pd.to_datetime(["2023-01-03"]), ["a", "b"]], names=["datetime", "instrument"])
    score = pd.Series([0.1, 0.2], index=index)
    with pytest.raises(ValueError):
        fit_rank_weight(score, score.iloc[:1], score)


def test_residual_weight_can_turn_off_unhelpful_member():
    index = pd.MultiIndex.from_product([pd.to_datetime(["2023-01-03"]), ["a", "b"]], names=["datetime", "instrument"])
    base = pd.Series([0.25, 0.75], index=index)
    residual = pd.Series([0.5, -0.5], index=index)
    assert fit_residual_weight(base, residual, base) == 0


def test_mature_dates_use_actual_label_end():
    calendar = pd.date_range("2023-01-01", periods=5, freq="B")
    ends = pd.Series([calendar[2], calendar[3], calendar[4], pd.NaT, pd.NaT], index=calendar)
    assert list(mature_update_dates(calendar, ends, calendar[4])) == [calendar[0], calendar[1]]
