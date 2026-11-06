from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException


class _NoopRedis:
    def __init__(self, *args, **kwargs):
        pass

    def setex(self, *args, **kwargs):
        pass


def _request():
    return SimpleNamespace(
        training_mode="full",
        train_lgb_lstm=False,
        expert_role=None,
        instruments="csi300",
        use_factor_pool=True,
        train_end="2025-01-31",
    )


def test_training_rejects_missing_historical_factor_pool_snapshot():
    from backend.service.training import TrainingService

    background_tasks = Mock()
    with patch.object(TrainingService, "_require_historical_training_ready"), \
         patch("core.domain.factors.registry.get_pool") as get_pool:
        get_pool.return_value.get_active_factors.return_value = {}
        with pytest.raises(HTTPException) as error:
            TrainingService.submit_train(_request(), background_tasks)

    assert error.value.status_code == 409
    background_tasks.add_task.assert_not_called()


def test_training_rejects_factor_pool_snapshot_read_failure():
    from backend.service.training import TrainingService

    background_tasks = Mock()
    with patch.object(TrainingService, "_require_historical_training_ready"), \
         patch("core.domain.factors.registry.get_pool") as get_pool:
        get_pool.side_effect = RuntimeError("pool state unavailable")
        with pytest.raises(HTTPException) as error:
            TrainingService.submit_train(_request(), background_tasks)

    assert error.value.status_code == 503
    background_tasks.add_task.assert_not_called()


def _rolling_request():
    from backend.api.qlib.models import RollingTrainRequest

    return RollingTrainRequest(
        pool="csi300",
        start_anchor="2020-01-01",
        end_anchor="2020-01-01",
        train_years=1,
        valid_months=1,
        use_factor_pool=True,
    )


def test_rolling_training_freezes_historical_snapshot_before_queueing():
    from backend.service.training import TrainingService

    class ImmediateTasks:
        def add_task(self, callback, *args, **kwargs):
            callback(*args, **kwargs)

    captured = []
    historical = {
        "as_of": "2019-11-30",
        "categories": {
            "white_box": {
                "effective_date": "2019-10-15",
                "source_anchor": "2019-10-01",
                "provenance": "matured_rolling_test",
                "keys": ["factor_a"],
                "factor_specs": {"factor_a": {"expr": "$close"}},
            },
        },
        "factors": {"factor_a": {"expr": "$close"}},
    }

    with patch.object(TrainingService, "_require_historical_training_ready"), \
         patch("core.adapters.qlib.generation.assert_no_active_historical_release"), \
         patch("core.adapters.qlib.rolling_train.make_rolling_rounds", return_value=[(
             "2020-01-01", ("2018-12-01", "2019-11-30"), ("2019-12-01", "2019-12-31"),
             ("2020-01-01", "2020-03-31"),
         )]), \
         patch("core.data.FactorPoolStateRepo.today_iso", return_value="2020-01-31"), \
         patch("core.domain.factors.registry.get_pool") as get_pool, \
         patch("core.adapters.qlib.rolling_train.rolling_train_dual", side_effect=lambda **kwargs: captured.append(kwargs) or {
             "rounds": [], "oos": {}, "n_planned": 1, "n_trained": 0,
         }), \
         patch("backend.service.training.resolve_instruments", return_value="csi300"), \
         patch("backend.service.training.run_with_qlib_read_access", side_effect=lambda callback: callback()), \
         patch("redis.Redis", _NoopRedis):
        get_pool.return_value.get_active_factor_snapshot.return_value = historical
        TrainingService.submit_rolling_train(_rolling_request(), ImmediateTasks())

    snapshot = captured[0]["factor_snapshots_by_anchor"]["2020-01-01"]
    assert snapshot["selection_cutoff"] == "2019-11-30"
    assert snapshot["effective_date"] == "2019-10-15"
    assert snapshot["custom_factor_keys"] == ["factor_a"]
    assert snapshot["factor_specs"]["factor_a"]["expr"] == "$close"


def test_rolling_training_rejects_missing_historical_snapshot_before_queueing():
    from backend.service.training import TrainingService

    background_tasks = Mock()
    with patch.object(TrainingService, "_require_historical_training_ready"), \
         patch("core.adapters.qlib.generation.assert_no_active_historical_release"), \
         patch("core.adapters.qlib.rolling_train.make_rolling_rounds", return_value=[(
             "2020-01-01", ("2018-12-01", "2019-11-30"), ("2019-12-01", "2019-12-31"),
             ("2020-01-01", "2020-03-31"),
         )]), \
         patch("core.data.FactorPoolStateRepo.today_iso", return_value="2020-01-31"), \
         patch("core.domain.factors.registry.get_pool") as get_pool:
        get_pool.return_value.get_active_factor_snapshot.return_value = None
        with pytest.raises(HTTPException) as error:
            TrainingService.submit_rolling_train(_rolling_request(), background_tasks)

    assert error.value.status_code == 409
    background_tasks.add_task.assert_not_called()
