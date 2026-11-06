import numpy as np
import pandas as pd
import pytest

from core.domain.factors.evidence import regroup_evidence_by_state
from core.domain.regime.volatility_history import build_volatility_history


pytestmark = pytest.mark.unit


def _index_frame(size=700, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.012, size)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=size, freq="B"),
        "close": 3500 * np.cumprod(1 + returns),
    })


def test_volatility_history_outputs_every_mature_trading_day():
    frame = _index_frame()
    rows = build_volatility_history(frame)

    assert len(rows) == len(frame) - 271
    assert rows[0]["trade_date"] == frame.iloc[271]["date"].date().isoformat()
    assert rows[-1]["trade_date"] == frame.iloc[-1]["date"].date().isoformat()
    assert {row["vol_state"] for row in rows} == {"high_vol", "low_vol"}
    assert all(row["method"] == "rule_based_pit_v1" for row in rows)


def test_volatility_history_is_point_in_time_stable():
    frame = _index_frame()
    cutoff = frame.iloc[550]["date"].date().isoformat()
    original = {
        row["trade_date"]: row for row in build_volatility_history(frame)
        if row["trade_date"] <= cutoff
    }
    changed = frame.copy()
    changed.loc[changed.index > 550, "close"] *= np.linspace(1.0, 3.0, len(changed) - 551)
    rebuilt = {
        row["trade_date"]: row for row in build_volatility_history(changed)
        if row["trade_date"] <= cutoff
    }

    assert rebuilt == original


def test_regroup_evidence_uses_existing_oriented_daily_ic():
    rows = [{
        "run_id": "old", "factor_key": "f", "family": None,
        "fold_id": "2024-01-01", "anchor_date": "2024-01-01", "state": "all",
        "direction_sign": -1, "raw_ic": -0.02, "oriented_ic": 0.02,
        "coverage": 1.0,
        "metrics": {
            "daily_ics": [
                ["2024-01-02", 0.01], ["2024-01-03", 0.03],
                ["2024-01-04", -0.01], ["2024-01-05", 0.05],
            ],
            "l1_pass": True, "dedup_pass": False,
        },
    }]
    state_map = {
        "2024-01-02": "high_vol", "2024-01-03": "high_vol",
        "2024-01-04": "low_vol", "2024-01-05": "low_vol",
    }

    result = regroup_evidence_by_state(rows, state_map, "new", {"f": "动量"})
    states = {row["state"]: row for row in result}

    assert set(states) == {"all", "high_vol", "low_vol"}
    assert states["all"]["run_id"] == "new"
    assert states["all"]["family"] == "动量"
    assert states["high_vol"]["oriented_ic"] == pytest.approx(0.02)
    assert states["high_vol"]["raw_ic"] == pytest.approx(-0.02)
    assert states["low_vol"]["oriented_ic"] == pytest.approx(0.02)
    assert states["low_vol"]["sample_count"] == 2
