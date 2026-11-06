import numpy as np
import pandas as pd
import pytest

from core.adapters.qlib.lstm_fusion import (
    _build_inference_sequences,
    _build_sequences,
    _prediction_cache_key,
    _rebuild_sequence_scaler,
    _validate_lstm_bundle,
)
from core.domain.regime.model_routing import (
    normalize_model_ref,
    normalize_state_models,
    resolve_state_key,
    validate_state_models,
)


pytestmark = pytest.mark.unit


def test_validate_complete_volatility_mapping():
    mapping = {
        "high_vol": ["exp", "rebound"],
        "low_vol": "exp/trend",
    }

    assert validate_state_models(mapping, True) == {
        "high_vol": ("exp", "rebound"),
        "low_vol": ("exp", "trend"),
    }


def test_validate_rejects_partial_mapping():
    with pytest.raises(ValueError, match="low_vol"):
        validate_state_models({"high_vol": ["exp", "run"]}, True)


def test_validate_rejects_path_traversal_model_id():
    with pytest.raises(ValueError, match="无效模型标识"):
        validate_state_models({
            "high_vol": ["..", "rebound"],
            "low_vol": ["exp", "trend"],
        }, True)


def test_empty_mapping_keeps_automatic_routing():
    assert normalize_state_models({}) == {}
    assert validate_state_models({}, False) == {}


def test_invalid_empty_value_is_not_silently_treated_as_auto():
    with pytest.raises(ValueError, match="无效模型映射"):
        normalize_state_models({"high_vol": ""})


def test_normalize_fixed_model_ref():
    assert normalize_model_ref("exp/run") == ("exp", "run")
    assert normalize_model_ref(None) is None


def test_normalize_fixed_model_ref_rejects_path_traversal():
    with pytest.raises(ValueError, match="无效模型标识"):
        normalize_model_ref(["..", "run"])


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("oversold_rebound", "high_vol"), ("trend_momentum", "low_vol")],
)
def test_resolve_volatility_state(direction, expected):
    assert resolve_state_key("steady_up", direction, True) == expected
    assert resolve_state_key("steady_up", direction, False) == "steady_up"


def test_inference_sequences_keep_lookback_and_prediction_dates():
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    index = pd.MultiIndex.from_product(
        [dates, ["sh600000", "sz000001"]], names=["datetime", "instrument"])
    features = pd.DataFrame(
        np.arange(len(index) * 2, dtype=np.float32).reshape(len(index), 2),
        index=index,
        columns=["f1", "f2"],
    )

    sequences, sequence_index = _build_inference_sequences(
        features, "2026-01-21", seq_len=20)

    assert sequences.shape == (10, 20, 2)
    assert min(item[0] for item in sequence_index) == pd.Timestamp("2026-01-21")
    assert max(item[0] for item in sequence_index) == pd.Timestamp("2026-01-25")


def test_rebuilt_scaler_matches_training_sequence_weighting():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    index = pd.MultiIndex.from_product(
        [dates, ["sh600000", "sz000001"]], names=["datetime", "instrument"])
    features = pd.DataFrame(
        np.arange(len(index) * 2, dtype=np.float32).reshape(len(index), 2),
        index=index,
        columns=["f1", "f2"],
    )
    labels = pd.Series(0.0, index=index)
    segments = pd.Series("train", index=index)
    sequences, _, _, _ = _build_sequences(features, labels, segments, seq_len=20, train_step=5)

    expected = sequences.reshape(-1, 2)
    mean, scale = _rebuild_sequence_scaler(features, dates[-1], seq_len=20, train_step=5)

    np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(scale, expected.std(axis=0), rtol=1e-6)


def test_bundle_validation_preserves_training_contract():
    state = {"lstm.weight": np.array([1.0], dtype=np.float32)}
    bundle = {
        "state_dict": state,
        "scaler": {"mean": [1.0, 2.0], "scale": [3.0, 4.0]},
        "factor_keys": ["f1", "f2"],
        "seq_len": 20,
        "fusion_weight": 0.5,
    }

    mean, scale, loaded_state = _validate_lstm_bundle(
        bundle, ["f1", "f2"], 20, 0.5)

    np.testing.assert_array_equal(mean, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(scale, np.array([3.0, 4.0], dtype=np.float32))
    assert loaded_state is state


def test_bundle_validation_rejects_factor_order_change():
    bundle = {
        "state_dict": {},
        "scaler": {"mean": [1.0, 2.0], "scale": [3.0, 4.0]},
        "factor_keys": ["f1", "f2"],
        "seq_len": 20,
        "fusion_weight": 0.5,
    }

    with pytest.raises(RuntimeError, match="因子顺序"):
        _validate_lstm_bundle(bundle, ["f2", "f1"], 20, 0.5)


def test_prediction_cache_key_isolated_by_universe_and_data_revision():
    base = dict(
        experiment_id="exp",
        run_id="run",
        predict_start="2026-08-24",
        predict_end="2026-08-24",
        factor_keys=["f1"],
        seq_len=20,
        fusion_weight=0.5,
        artifact_signature="artifact-v1",
        data_signature="data-v1",
    )

    key_a500 = _prediction_cache_key(instruments="a500", **base)
    key_all = _prediction_cache_key(instruments="all", **base)
    key_new_data = _prediction_cache_key(
        instruments="a500", **{**base, "data_signature": "data-v2"})

    assert key_a500 != key_all
    assert key_a500 != key_new_data
