import numpy as np
import pandas as pd

from core.adapters.qlib.horizon_candidate_runtime import _sequence_scores


class MaskCount:
    def __call__(self, batch):
        return batch[:, :, -1].sum(dim=1)


def test_sequence_keeps_missing_trading_day_slot():
    calendar = pd.date_range("2023-01-02", periods=25, freq="B")
    index = pd.MultiIndex.from_product([calendar, ["a", "b"]], names=["datetime", "instrument"])
    frame = pd.DataFrame({"x": np.arange(len(index), dtype=np.float32)}, index=index)
    frame = frame.drop(index=(calendar[-7], "b"))
    expected = pd.MultiIndex.from_product([[calendar[-1]], ["a", "b"]], names=index.names)
    scores = _sequence_scores(MaskCount(), np.array([0], dtype=np.float32),
                              np.array([1], dtype=np.float32), frame, expected, calendar)
    assert scores.loc[(calendar[-1], "a")] == 20
    assert scores.loc[(calendar[-1], "b")] == 19
