import numpy as np
import pandas as pd

from scripts.horizon_architecture_improvement import make_inner_masks


def test_inner_oof_purges_labels_at_both_boundaries():
    calendar = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    index = pd.MultiIndex.from_product([calendar, ["a", "b"]], names=["datetime", "instrument"])
    labels = pd.DataFrame({"label": np.ones(len(index)),
                           "label_end": index.get_level_values("datetime") + pd.offsets.BDay(20)}, index=index)
    outer_train = np.ones(len(index), dtype=bool)
    cohort_start = pd.Timestamp("2021-07-01")
    cohort_end = pd.Timestamp("2022-01-01")
    masks = make_inner_masks(index, labels, outer_train, cohort_start, cohort_end)
    assert masks["train"].sum() > 0 and masks["valid"].sum() > 0 and masks["test"].sum() > 0
    assert labels.loc[masks["train"], "label_end"].max() < cohort_start - pd.DateOffset(months=3)
    assert labels.loc[masks["valid"], "label_end"].max() < cohort_start
    assert index[masks["train"]].intersection(index[masks["test"]]).empty
