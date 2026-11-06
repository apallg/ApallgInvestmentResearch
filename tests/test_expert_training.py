import json

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.api.qlib.models import TrainRequest
from backend.service.expert_training import (
    build_state_subset_profile,
    load_expert_factor_keys,
)
from backend.service.model_presenter import make_display_name
from backend.service.training import TrainingService


def _market(role):
    dates = pd.bdate_range("2023-01-02", periods=80)
    rising = np.linspace(0, 1, len(dates))
    if role == "trend":
        trend, oversold, breadth, vol = rising, rising, rising, rising[::-1]
    else:
        trend, oversold, breadth, vol = rising, rising, rising[::-1], rising
    return pd.DataFrame({
        "date": dates,
        "trend_strength": trend,
        "oversold_degree": oversold,
        "market_breadth": breadth,
        "realized_vol_percentile": vol,
    })


@pytest.mark.unit
@pytest.mark.parametrize("role", ["trend", "rebound"])
def test_state_subset_selects_high_affinity_half(role):
    market = _market(role)
    selected, profile = build_state_subset_profile(
        market, ("2023-01-02", "2023-04-21"), role)
    assert len(selected) == 40
    assert selected == market["date"].dt.strftime("%Y-%m-%d").tolist()[40:]
    assert profile["training_mode"] == "state_subset"
    assert profile["expert_role"] == role
    assert profile["state_subset_ratio"] == 0.5


@pytest.mark.unit
def test_expert_factor_keys_are_core_plus_role_satellite(tmp_path):
    path = tmp_path / "factor_pool_state.json"
    path.write_text(json.dumps({"curated": {
        "core_base": {"flat_list": ["core_a", "shared"]},
        "trend_satellite": {"flat_list": ["trend_a", "shared"]},
        "rebound_satellite": {"flat_list": ["rebound_a"]},
    }}), encoding="utf-8")
    assert load_expert_factor_keys("trend", path) == ["core_a", "shared", "trend_a"]
    assert load_expert_factor_keys("rebound", path) == ["core_a", "shared", "rebound_a"]


@pytest.mark.unit
def test_expert_factor_keys_default_uses_factor_pool(monkeypatch):
    class Pool:
        state = {"curated": {
            "core_base": {"flat_list": ["core_a"]},
            "trend_satellite": {"flat_list": ["trend_a"]},
        }}

    monkeypatch.setattr("core.domain.factors.registry.get_pool", lambda: Pool())
    assert load_expert_factor_keys("trend") == ["core_a", "trend_a"]


@pytest.mark.unit
def test_train_request_validates_expert_role():
    req = TrainRequest(training_mode="state_subset", expert_role="trend", train_lgb_lstm=True)
    assert req.expert_role == "trend"
    with pytest.raises(ValidationError):
        TrainRequest(training_mode="state_subset", expert_role="invalid", train_lgb_lstm=True)


@pytest.mark.unit
def test_state_subset_requires_lgb_lstm():
    req = TrainRequest(training_mode="state_subset", expert_role="trend", train_lgb_lstm=False)
    with pytest.raises(HTTPException) as exc:
        TrainingService.submit_train(req, object())
    assert exc.value.status_code == 400


@pytest.mark.unit
def test_state_subset_display_name_includes_role():
    name = make_display_name({
        "model_class": "LGB+LSTM",
        "training_mode": "state_subset",
        "expert_role": "trend",
        "factor_label": "自选40因子",
        "instruments": "list_3137_stocks",
        "train_period": "('2021-01-01', '2025-12-31')",
        "label_period": 10,
    })
    assert "趋势子样本" in name


@pytest.mark.unit
def test_trained_model_list_exposes_state_subset_metadata(tmp_path, monkeypatch):
    run = tmp_path / "1" / "run123"
    artifacts = run / "artifacts"
    params = run / "params"
    artifacts.mkdir(parents=True)
    params.mkdir()
    (artifacts / "params.pkl").write_bytes(b"model")
    (artifacts / "params_lstm.pkl").write_bytes(b"lstm")
    values = {
        "model.class": "LGB+LSTM",
        "training_mode": "state_subset",
        "expert_role": "rebound",
        "label_horizon": "10",
        "state_subset_n_dates": "300",
        "state_subset_total_dates": "600",
        "state_subset_ratio": "0.5",
    }
    for key, value in values.items():
        (params / key).write_text(value, encoding="utf-8")
    monkeypatch.setattr("backend.service.training.config.QLIB_MLFLOW_URI", f"file:{tmp_path}")
    models = TrainingService.list_trained_models()["data"]
    model = next(item for item in models if item.get("run_id") == "run123")
    assert model["is_state_subset"] is True
    assert model["training_mode"] == "state_subset"
    assert model["expert_role"] == "rebound"
    assert model["state_subset_n_dates"] == 300
    assert model["label_period"] == 10
    assert "反弹子样本" in model["display_name"]
