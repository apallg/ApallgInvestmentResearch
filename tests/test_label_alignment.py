import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backend.api.qlib.models import (
    EnsembleTrainRequest,
    OptunaSearchRequest,
    RollingTrainRequest,
    TrainRequest,
)
from backend.service.training import TrainingService
from core.adapters.qlib.models import QlibModelManager, _build_label_expression, _purge_training_segments
from core.domain.backtest.engine import DirectionBacktestRunner
from core.domain.regime.vol_predictor import _future_vol_labels
from core.adapters.qlib.model_contract import get_model_label_mode


pytestmark = pytest.mark.unit


def test_future_vol_label_uses_only_forward_returns():
    close = pd.Series(np.arange(100.0, 112.0))
    frame = pd.DataFrame({"close": close})
    horizon = 3
    labels = _future_vol_labels(frame, horizon)
    rets = close.pct_change(fill_method=None)
    expected = rets.iloc[1:4].std() * np.sqrt(250)
    assert labels.iloc[0] == pytest.approx(expected)
    assert labels.iloc[-3:].isna().all()


def test_future_vol_label_changes_with_future_not_past():
    close = pd.Series(np.arange(100.0, 112.0))
    original = _future_vol_labels(pd.DataFrame({"close": close}), 3)
    changed_past = close.copy()
    changed_past.iloc[0] = 10000
    assert _future_vol_labels(pd.DataFrame({"close": changed_past}), 3).iloc[3] == pytest.approx(original.iloc[3])
    changed_future = close.copy()
    changed_future.iloc[4] = 10000
    assert _future_vol_labels(pd.DataFrame({"close": changed_future}), 3).iloc[3] != pytest.approx(original.iloc[3])


def test_tradable_open_to_open_label_expression():
    assert _build_label_expression(5) == "Ref($open, -6) / Ref($open, -1) - 1"
    assert _build_label_expression(5, "close_to_close") == "Ref($close, -5) / $close - 1"
    assert _build_label_expression(5, "tradable_open_to_open") == "Ref($open, -6) / Ref($open, -1) - 1"
    assert _build_label_expression(20, "tradable_open_to_open") == "Ref($open, -21) / Ref($open, -1) - 1"


def test_label_expression_rejects_invalid_mode():
    with pytest.raises(ValueError):
        _build_label_expression(5, "future_close")


def test_training_segments_purge_open_label_overlap():
    calendar = pd.date_range("2024-01-01", periods=40, freq="B")
    train, valid, test = _purge_training_segments(
        ("2024-01-01", "2024-01-26"),
        ("2024-01-29", "2024-02-09"),
        ("2024-02-12", "2024-02-23"),
        purge_gap=3,
        calendar=calendar,
    )
    assert train[1] == "2024-01-23"
    assert valid[1] == "2024-02-06"
    assert test == ("2024-02-12", "2024-02-23")


def test_adapter_training_entrypoints_reject_legacy_label():
    manager = QlibModelManager.__new__(QlibModelManager)
    periods = {
        "train_period": ("2022-01-01", "2024-12-31"),
        "valid_period": ("2025-01-01", "2025-06-30"),
        "test_period": ("2025-07-01", "2025-12-31"),
    }
    with pytest.raises(ValueError, match="tradable_open_to_open"):
        manager.train("lightgbm", "Alpha158", "csi300", **periods, label_mode="close_to_close")
    with pytest.raises(ValueError, match="tradable_open_to_open"):
        manager.train_dual_model("Alpha158", "csi300", **periods, label_mode="close_to_close")
    with pytest.raises(ValueError, match="tradable_open_to_open"):
        manager.train_with_optuna("Alpha158", "csi300", **periods, label_mode="close_to_close")


def test_direction_model_cache_rejects_legacy_label(monkeypatch, tmp_path):
    cached = {"trend_momentum": ["exp", "run"]}
    cleared = []
    runner = DirectionBacktestRunner.__new__(DirectionBacktestRunner)
    monkeypatch.setattr("core.data.SettingsRepo.get", staticmethod(lambda key: cached))
    monkeypatch.setattr("core.data.SettingsRepo.set", staticmethod(lambda key, value: cleared.append((key, value))))
    monkeypatch.setattr(runner, "_validate_cached_run", lambda *_args: True)
    monkeypatch.setattr("core.shared.disabled_models.is_disabled", lambda *_args: False)
    monkeypatch.setattr("core.adapters.qlib.model_contract.get_model_label_mode", lambda *_args: "close_to_close")
    monkeypatch.setattr("core.domain.backtest.engine.MODEL_CACHE_FILE", str(tmp_path / "direction_models.json"))
    monkeypatch.setattr(runner, "_find_best_model_for_direction", lambda *_args, **_kwargs: None)

    assert runner._load_models_cached(["trend_momentum"]) == {}
    assert cleared == [("direction_models", {})]


def test_predict_dual_maps_stable_history_pool_to_qlib_a500(monkeypatch, tmp_path):
    artifacts = tmp_path / "exp" / "run" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "params_lstm.pkl").write_bytes(b"placeholder")
    monkeypatch.setattr("core.shared.config.QLIB_MLFLOW_URI", f"file:{tmp_path}")
    monkeypatch.setattr("core.shared.disabled_models.is_disabled", lambda *_args: False)

    captured = {}

    def fake_predict(*_args, **kwargs):
        captured["instruments"] = kwargs["instruments"]
        return "prediction", "dataset"

    monkeypatch.setattr("core.adapters.qlib.lstm_fusion.predict_lgb_lstm_fusion", fake_predict)
    manager = QlibModelManager.__new__(QlibModelManager)

    assert manager.predict_dual(
        "exp", "run", predict_start="2026-01-01", predict_end="2026-01-02",
        instruments="stable_history_500d",
    ) == ("prediction", "dataset")
    assert captured["instruments"] == "a500"


@pytest.mark.parametrize(("expression", "expected"), [
    ("['Ref($open, -11) / Ref($open, -1) - 1']", "tradable_open_to_open"),
    ("['Ref($close, -10) / $close - 1']", "close_to_close"),
])
def test_model_label_mode_is_inferred_for_legacy_runs(tmp_path, expression, expected):
    params = tmp_path / "params"
    params.mkdir()
    (params / "dataset.kwargs.handler.kwargs.label").write_text(expression, encoding="utf-8")

    assert get_model_label_mode(tmp_path) == expected


def test_model_label_mode_rejects_conflicting_metadata(tmp_path):
    artifacts = tmp_path / "artifacts"
    params = tmp_path / "params"
    artifacts.mkdir()
    params.mkdir()
    (artifacts / "factor_keys.json").write_text(
        '{"label_mode":"tradable_open_to_open"}', encoding="utf-8")
    (params / "dataset.kwargs.handler.kwargs.label").write_text(
        "['Ref($close, -10) / $close - 1']", encoding="utf-8")

    assert get_model_label_mode(tmp_path) is None


@pytest.mark.parametrize("request_cls", [
    TrainRequest,
    OptunaSearchRequest,
    RollingTrainRequest,
    EnsembleTrainRequest,
])
def test_qlib_training_requests_use_tradable_label(request_cls):
    assert request_cls().label_mode == "tradable_open_to_open"
    with pytest.raises(ValidationError):
        request_cls(label_mode="close_to_close")


class _ImmediateTasks:
    def add_task(self, callback, *args, **kwargs):
        callback(*args, **kwargs)


class _NoopRedis:
    def __init__(self, *args, **kwargs):
        pass

    def setex(self, *args, **kwargs):
        pass


def test_optuna_service_forwards_tradable_label(monkeypatch):
    monkeypatch.setattr(TrainingService, "_require_historical_training_ready", lambda pool=None: None)
    calls = []

    class Manager:
        def train_with_optuna(self, **kwargs):
            calls.append(kwargs)
            return {"best_params": {}, "best_ic": 0.0, "best_trial": 0}

    monkeypatch.setattr("core.adapters.qlib.models.QlibModelManager", Manager)
    monkeypatch.setattr("backend.service.training.resolve_instruments", lambda value: value)
    monkeypatch.setattr("backend.service.training.save_optuna_result", lambda *args: None)
    monkeypatch.setattr("redis.Redis", _NoopRedis)

    TrainingService.submit_optuna(OptunaSearchRequest(n_trials=1), _ImmediateTasks())

    assert calls[0]["label_mode"] == "tradable_open_to_open"


def test_optuna_dual_service_forwards_tradable_label_to_both_searches(monkeypatch):
    monkeypatch.setattr(TrainingService, "_require_historical_training_ready", lambda pool=None: None)
    calls = []

    class Manager:
        def train_with_optuna(self, **kwargs):
            calls.append(kwargs)
            return {"best_params": {}, "best_ic": 0.0, "best_trial": 0}

    monkeypatch.setattr("core.adapters.qlib.models.QlibModelManager", Manager)
    monkeypatch.setattr("backend.service.training.resolve_instruments", lambda value: value)
    monkeypatch.setattr("backend.service.training.save_optuna_result", lambda *args: None)
    monkeypatch.setattr("redis.Redis", _NoopRedis)

    TrainingService.submit_optuna(
        OptunaSearchRequest(n_trials=1, search_dual=True), _ImmediateTasks())

    assert len(calls) == 2
    assert [call["model_name"] for call in calls] == ["lightgbm", "xgboost"]
    assert all(call["label_mode"] == "tradable_open_to_open" for call in calls)


def test_rolling_service_forwards_tradable_label(monkeypatch):
    monkeypatch.setattr(TrainingService, "_require_historical_training_ready", lambda pool=None: None)
    calls = []

    def rolling_train_dual(**kwargs):
        calls.append(kwargs)
        return {"rounds": [], "oos": {}, "n_planned": 0, "n_trained": 0}

    monkeypatch.setattr("core.adapters.qlib.rolling_train.rolling_train_dual", rolling_train_dual)
    monkeypatch.setattr("backend.service.training.resolve_instruments", lambda value: value)
    monkeypatch.setattr("redis.Redis", _NoopRedis)

    TrainingService.submit_rolling_train(RollingTrainRequest(), _ImmediateTasks())

    assert calls[0]["label_mode"] == "tradable_open_to_open"


def test_ensemble_service_forwards_tradable_label(monkeypatch):
    monkeypatch.setattr(TrainingService, "_require_historical_training_ready", lambda pool=None: None)
    calls = []
    prediction = pd.Series([0.1])

    class Manager:
        def train(self, **kwargs):
            calls.append(kwargs)
            return {
                "predictions": prediction,
                "label": prediction,
                "ic_mean": 0.0,
                "icir": 0.0,
            }

    monkeypatch.setattr("core.adapters.qlib.models.QlibModelManager", Manager)
    monkeypatch.setattr(
        "core.adapters.qlib.ensemble.QlibEnsembleAdapter.merge_predictions",
        lambda *args, **kwargs: prediction,
    )
    monkeypatch.setattr("redis.Redis", _NoopRedis)

    TrainingService.submit_ensemble_train(
        EnsembleTrainRequest(model_names=["lightgbm"], auto_backtest=False),
        _ImmediateTasks(),
    )

    assert calls[0]["label_mode"] == "tradable_open_to_open"
