from unittest.mock import patch

import pytest


@pytest.mark.unit
@pytest.mark.p1
def test_rolling_train_uses_and_records_each_anchor_snapshot():
    from core.adapters.qlib.rolling_train import rolling_train_dual

    calls = []

    class Manager:
        def train(self, **kwargs):
            calls.append(kwargs)
            return {"ic_mean": 0.1, "ric_mean": 0.2, "icir": 0.3}

    snapshots = {
        "2020-01-01": {
            "factor_set": "WhiteBoxHandler",
            "custom_factor_keys": ["factor_a"],
            "selection_cutoff": "2019-11-30",
            "selection_source": "matured_rolling_test",
            "effective_date": "2019-11-15",
            "fingerprint": "fingerprint-a",
            "factor_specs": {"factor_a": {"expr": "$close"}},
        },
        "2020-04-01": {
            "factor_set": "WhiteBoxHandler",
            "custom_factor_keys": ["factor_b"],
            "selection_cutoff": "2020-02-29",
            "selection_source": "explicit_snapshot",
            "effective_date": "2020-02-01",
            "fingerprint": "fingerprint-b",
            "factor_specs": {"factor_b": {"expr": "$open"}},
        },
    }

    with patch("core.adapters.qlib.models.QlibModelManager", Manager):
        result = rolling_train_dual(
            pool="all",
            start_anchor="2020-01-01",
            end_anchor="2020-04-01",
            step_months=3,
            train_years=1,
            valid_months=1,
            train_dual=False,
            factor_snapshots_by_anchor=snapshots,
        )

    assert [call["custom_factor_keys"] for call in calls] == [["factor_a"], ["factor_b"]]
    assert [call["factor_selection_snapshot"] for call in calls] == list(snapshots.values())
    assert [round_entry["custom_factor_keys"] for round_entry in result["rounds"]] == [
        ["factor_a"], ["factor_b"],
    ]
    assert [round_entry["factor_selection_fingerprint"] for round_entry in result["rounds"]] == [
        "fingerprint-a", "fingerprint-b",
    ]
