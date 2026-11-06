import pandas as pd
import pytest
import pickle

from backend.service.score_ablation import (
    _execution_outcomes,
    _price_indexes,
    load_history_pool,
    load_prediction_cache,
    pool_members_on,
    run_score_ablation,
)
from core.domain.screening.score_replay import replay_score_variants


@pytest.mark.p1
@pytest.mark.unit
def test_history_pool_uses_point_in_time_intervals_and_explicit_semantics(tmp_path):
    path = tmp_path / "a500.txt"
    path.write_text(
        "sh600000\t2025-01-01\t2025-06-30\n"
        "sz000001\t2025-07-01\t2025-12-31\n"
        "sh688001\t2025-01-01\t2025-12-31\n",
        encoding="utf-8",
    )

    intervals, metadata = load_history_pool(path)

    assert pool_members_on(intervals, "2025-03-01") == {"600000"}
    assert pool_members_on(intervals, "2025-08-01") == {"000001"}
    assert metadata["semantic_name"] == "project_a500_history_pool"
    assert metadata["not_an_index_constituent_claim"] is True


@pytest.mark.p1
@pytest.mark.unit
def test_execution_outcomes_blocks_t1_limit_up_and_delays_limit_down_exit():
    calendar = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    frames = {
        "600000": pd.DataFrame({
            "trade_date": calendar,
            "open": [10, 11, 11, 11],
            "close": [10, 11, 11, 11],
            "volume": [100, 100, 100, 100],
            "adj_factor": [1, 1, 1, 1],
        }),
        "000001": pd.DataFrame({
            "trade_date": calendar,
            "open": [10, 10, 9, 9.5],
            "close": [10, 10, 9, 9.5],
            "volume": [100, 100, 100, 100],
            "adj_factor": [1, 1, 1, 1],
        }),
    }
    raw = {"2025-01-02": {"600000": 2.0, "000001": 1.0}}

    returns, eligible, diagnostics = _execution_outcomes(
        raw, [pd.Timestamp("2025-01-02")], list(calendar), frames,
        _price_indexes(frames), {}, horizon=1, max_exit_delay=1,
    )

    assert eligible["2025-01-02"] == {"000001"}
    assert returns["2025-01-02"]["000001"] == pytest.approx(-0.05)
    assert diagnostics["2025-01-02"]["entry_limit_up"] == 1
    assert diagnostics["2025-01-02"]["delayed_exit"] == 1


@pytest.mark.p1
@pytest.mark.unit
def test_replay_reports_missing_selected_labels_and_drawdown():
    raw = {
        "2025-01-01": {"a": 2, "b": 1},
        "2025-01-02": {"a": 2, "b": 1},
    }
    variants = {"baseline": {"scores_by_date": raw, "config": {}}}
    returns = {"2025-01-01": {"a": -0.1}, "2025-01-02": {}}

    result = replay_score_variants(raw, variants, 1, returns)
    summary = result["variants"]["baseline"]["summary"]

    assert summary["missing_selected_returns"] == 1
    assert summary["max_drawdown"] < 0


@pytest.mark.p1
@pytest.mark.unit
def test_replay_rejects_t1_untradable_selection_without_refilling_from_future():
    raw = {"2025-01-01": {"limit_up": 2, "second": 1}}
    variants = {"baseline": {"scores_by_date": raw, "config": {}}}
    returns = {"2025-01-01": {"second": 0.2}}

    result = replay_score_variants(
        raw, variants, 1, returns,
        entry_eligible_by_date={"2025-01-01": {"second"}},
    )

    period = result["variants"]["baseline"]["periods"][0]
    assert period["selected"] == ["limit_up"]
    assert period["executed"] == []
    assert period["entry_rejected"] == ["limit_up"]
    assert period["gross_return"] == 0
    assert period["cost"] == 0


@pytest.mark.p1
@pytest.mark.unit
def test_paired_statistics_use_affected_periods_for_sparse_feature():
    raw = {
        "2025-01-01": {"a": 2, "b": 1},
        "2025-01-02": {"a": 2, "b": 1},
        "2025-01-03": {"a": 2, "b": 1},
    }
    variants = {
        "baseline": {"scores_by_date": raw, "config": {}},
        "sparse": {"scores_by_date": {
            "2025-01-01": {"a": 2, "b": 1},
            "2025-01-02": {"a": 2, "b": 1},
            "2025-01-03": {"a": 1, "b": 2},
        }, "config": {}},
    }
    returns = {
        "2025-01-01": {"a": 0, "b": 0},
        "2025-01-02": {"a": 0, "b": 0},
        "2025-01-03": {"a": 0, "b": 0.1},
    }

    paired = replay_score_variants(raw, variants, 1, returns)["variants"]["sparse"]["paired_vs_baseline"]

    assert paired["affected_period_count"] == 1
    assert paired["coverage_rate"] == pytest.approx(1 / 3)
    assert paired["conditional_positive_delta_rate"] == 1.0


@pytest.mark.p1
@pytest.mark.unit
def test_ablation_rejects_candidate_depth_smaller_than_topn():
    with pytest.raises(ValueError, match="candidate_depth"):
        run_score_ablation("missing.pkl", top_n=50, candidate_depth=49)


@pytest.mark.p1
@pytest.mark.unit
def test_legacy_prediction_cache_never_claims_raw_or_resolved_provenance(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-01"), "sh600000")],
        names=["datetime", "instrument"],
    )
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump({"key": "opaque", "prediction": pd.Series([0.9], index=index)}, handle)

    _, metadata = load_prediction_cache(path, trusted=True)

    assert metadata["score_semantics"] == "unknown_legacy_cache"
    assert metadata["provenance_status"] == "unresolved_cache_provenance"
    assert metadata["raw_model_score_available"] is False


@pytest.mark.p1
@pytest.mark.unit
def test_prediction_cache_metadata_resolves_and_checks_expected_model(tmp_path, monkeypatch):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-01"), "sh600000")],
        names=["datetime", "instrument"],
    )
    path = tmp_path / "v2.pkl"
    artifacts = tmp_path / "mlruns" / "exp" / "run" / "artifacts"
    artifacts.mkdir(parents=True)
    for name, content in (("params.pkl", b"lgb"), ("params_lstm.pkl", b"lstm"),
                          ("lstm_bundle.pkl", b"bundle")):
        (artifacts / name).write_bytes(content)
    from backend.service import score_ablation
    monkeypatch.setattr(score_ablation.config, "QLIB_MLFLOW_URI", f"file:{tmp_path / 'mlruns'}")
    signature = score_ablation._model_artifact_signature("exp/run")
    payload = {
        "key": "opaque",
        "prediction": pd.Series([0.9], index=index),
        "metadata": {
            "model_id": "exp/run",
            "artifact_signature": signature,
            "score_semantics": "daily_cross_sectional_rank_fusion",
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    _, metadata = load_prediction_cache(path, expected_model_id="exp/run", trusted=True)

    assert metadata["provenance_status"] == "verified_embedded_metadata"
    assert metadata["score_semantics"] == "daily_cross_sectional_rank_fusion"
    with pytest.raises(ValueError, match="model_id"):
        load_prediction_cache(path, expected_model_id="other/run", trusted=True)


@pytest.mark.p1
@pytest.mark.unit
def test_formal_ablation_blocks_without_historical_reference_snapshots(tmp_path, monkeypatch):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-01"), "sh600000")],
        names=["datetime", "instrument"],
    )
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump({"prediction": pd.Series([0.9], index=index)}, handle)
    from backend.service import score_ablation
    monkeypatch.setattr(score_ablation.Database, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="PIT"):
        monkeypatch.setattr(
            score_ablation.config, "QLIB_PROVIDER_URI", f"file:{tmp_path}")
        internal = tmp_path / "prediction_cache" / path.name
        internal.parent.mkdir()
        path.replace(internal)
        run_score_ablation(internal)
