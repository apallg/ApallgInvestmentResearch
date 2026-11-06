from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from scripts.h20_confirmation_forward_v2 import record_window
from scripts.h20_confirmation_inputs_v2 import preprocess
from core.domain.screening.score_pipeline import build_score_result


def test_record_window_allows_close_and_overnight_before_next_open():
    calendar = pd.DatetimeIndex(["2026-11-06", "2026-11-09"])
    date = calendar[0]
    zone = ZoneInfo("Asia/Shanghai")
    for moment in (datetime(2026, 11, 6, 15, 0, tzinfo=zone),
                   datetime(2026, 11, 9, 9, 24, tzinfo=zone)):
        assert record_window(date, calendar, moment).hour == 9
    for moment in (datetime(2026, 11, 6, 14, 59, tzinfo=zone),
                   datetime(2026, 11, 9, 9, 25, tzinfo=zone)):
        with pytest.raises(RuntimeError):
            record_window(date, calendar, moment)


def test_preprocess_uses_full_day_cross_section_before_subset():
    codes = [f"sh{600000 + value}" for value in range(20)]
    index = pd.MultiIndex.from_arrays([[pd.Timestamp("2026-11-05")] * 20, codes],
                                       names=["datetime", "instrument"])
    panel = pd.DataFrame({"factor": [float(value) for value in range(19)] + [1000.0]}, index=index)
    processed = preprocess(panel.copy(), ["factor"])
    assert processed.iloc[-1, 0] < 1000
    assert np.isfinite(processed.to_numpy()).all()
    assert processed.iloc[-1, 0] != preprocess(panel.iloc[-5:].copy(), ["factor"]).iloc[-1, 0]


def test_three_seed_average_then_one_project_percentile_mapping():
    codes = ["600001", "600002", "600003"]
    seeds = [np.array([0.9, 0.5, 0.1]), np.array([0.1, 0.5, 0.9]), np.array([0.9, 0.5, 0.1])]
    average = np.mean(seeds, axis=0)
    result = build_score_result(dict(zip(codes, average)), neutralize_requested=False,
                                neutralize_status="disabled", model_input_semantics="rank_blend")
    assert result["audit"]["neutralize_status"] == "disabled"
    np.testing.assert_allclose([result["scores"][code] for code in codes], [1 / 3, -1 / 3, -1])
