import hashlib
import json
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
import pandas as pd

from scripts.capture_rank_persistence_shadow import (
    contract_fingerprint_for,
    security_reference_fingerprint,
    score_fingerprint,
    snapshot_metadata_fingerprint,
    validate_prior_snapshot,
)
from scripts import capture_rank_persistence_shadow as shadow_capture
from scripts.run_rank_persistence_ab import align_period_returns_to_decisions
from scripts.run_rank_persistence_forward import (
    dataframe_fingerprint,
    history_from_snapshots,
    canonical_report_hash,
    prediction_from_snapshots,
    price_cache_fingerprint,
    validate_existing_evaluation,
    weekly_decision_dates,
    pre_registered_primary_window,
    current_weekly_anchor,
)
from scripts.analyze_rank_persistence_ab import summarize_delta


def snapshot(contract_files=None):
    if contract_files is None:
        contract_files = {"core/model.py": "source-a"}
    ranked = [("600001", 0.5), ("600002", 0.4)]
    references = {
        "600001": {"name": "甲", "industry": "行业甲", "is_st": False, "update_date": "2026-09-28"},
        "600002": {"name": "乙", "industry": "行业乙", "is_st": False, "update_date": "2026-09-28"},
    }
    reference_rows = [{"code": code, **value} for code, value in sorted(references.items())]
    result = {
        "schema_version": 3,
        "prediction_date": "2026-09-28",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "captured_after_close_or_holiday": True,
        "earliest_execution_date": "2026-09-29",
        "market_data_date": "2026-09-28",
        "expected_latest_trading_date": "2026-09-28",
        "market_quote_rows": 5210,
        "source": "live_model",
        "model": {
            "experiment_id": "656749665990832865",
            "run_id": "7e5e67ba9b714c76b0808a2626d0f7d1",
            "label_mode": "tradable_open_to_open",
        },
        "scores_sha256": score_fingerprint(ranked),
        "score_count": len(ranked),
        "ranked_scores": [
            {"rank": rank, "code": code, "score": score}
            for rank, (code, score) in enumerate(ranked, 1)
        ],
        "model_fingerprint": "model-a",
        "pool_fingerprint": "pool-a",
        "pool": "stable_history_500d",
        "neutralize_qlib": False,
        "prediction_artifact_sha256": "artifact-hash",
        "security_reference_source": "BaoStock.query_stock_industry",
        "security_reference_observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "security_references": references,
        "security_references_sha256": security_reference_fingerprint(reference_rows),
        "contract_files_sha256": contract_files,
        "contract_fingerprint": contract_fingerprint_for(contract_files),
    }
    result["snapshot_metadata_sha256"] = snapshot_metadata_fingerprint(result)
    return result


@pytest.mark.unit
def test_matching_prior_snapshot_is_accepted():
    prior = snapshot()
    assert validate_prior_snapshot(
        prior, "2026-09-28", "model-a", "pool-a",
        prior["contract_fingerprint"],
    )


@pytest.mark.unit
def test_legacy_prior_snapshot_is_not_used_for_confirmation():
    prior = snapshot()
    for schema_version in (1, 2):
        legacy = dict(prior)
        legacy["schema_version"] = schema_version
        if schema_version == 1:
            legacy.pop("contract_fingerprint")
            legacy.pop("contract_files_sha256")
        assert not validate_prior_snapshot(
            legacy, "2026-09-28", "model-a", "pool-a", "contract-a"
        )


@pytest.mark.unit
def test_changed_code_contract_fails_closed():
    with pytest.raises(RuntimeError, match="代码合同指纹不一致"):
        validate_prior_snapshot(
            snapshot({"core/model.py": "source-old"}), "2026-09-28", "model-a",
            "pool-a", contract_fingerprint_for({"core/model.py": "source-new"}),
        )


@pytest.mark.unit
def test_changed_model_or_pool_fails_closed():
    with pytest.raises(RuntimeError, match="模型或股票池合同不一致"):
        validate_prior_snapshot(
            snapshot(), "2026-09-28", "model-changed", "pool-a", "contract-a"
        )


@pytest.mark.unit
def test_missing_or_tampered_security_references_fail_closed():
    prior = snapshot()
    prior["security_references"].pop("600002")
    with pytest.raises(RuntimeError, match="证券参考信息覆盖或哈希校验失败"):
        validate_prior_snapshot(
            prior, "2026-09-28", "model-a", "pool-a",
            prior["contract_fingerprint"],
        )

    prior = snapshot()
    prior["security_references"]["600001"]["is_st"] = True
    with pytest.raises(RuntimeError, match="证券参考信息覆盖或哈希校验失败"):
        validate_prior_snapshot(
            prior, "2026-09-28", "model-a", "pool-a",
            prior["contract_fingerprint"],
        )


@pytest.mark.unit
def test_tampered_scores_or_contract_manifest_fail_closed():
    prior = snapshot()
    prior["ranked_scores"][0]["score"] = 0.6
    with pytest.raises(RuntimeError, match="哈希或顺序校验失败"):
        validate_prior_snapshot(
            prior, "2026-09-28", "model-a", "pool-a",
            prior["contract_fingerprint"],
        )

    prior = snapshot()
    prior["contract_files_sha256"]["core/model.py"] = "tampered"
    with pytest.raises(RuntimeError, match="代码合同清单哈希校验失败"):
        validate_prior_snapshot(
            prior, "2026-09-28", "model-a", "pool-a",
            prior["contract_fingerprint"],
        )


@pytest.mark.unit
def test_snapshot_rejects_wrong_source_and_decision_timing():
    prior = snapshot()
    prior["source"] = "qlib_cache"
    with pytest.raises(RuntimeError, match="元数据哈希校验失败"):
        validate_prior_snapshot(
            prior, "2026-09-28", "model-a", "pool-a",
            prior["contract_fingerprint"],
        )


@pytest.mark.unit
def test_database_prediction_source_fails_closed():
    prior = snapshot()
    prior["source"] = "database"
    prior["snapshot_metadata_sha256"] = snapshot_metadata_fingerprint(prior)
    with pytest.raises(RuntimeError, match="决策时点或数据来源合同不匹配"):
        validate_prior_snapshot(
            prior, "2026-09-28", "model-a", "pool-a",
            prior["contract_fingerprint"],
        )

    prior = snapshot()
    prior["captured_after_close_or_holiday"] = False
    prior["snapshot_metadata_sha256"] = snapshot_metadata_fingerprint(prior)
    with pytest.raises(RuntimeError, match="决策时点或数据来源合同不匹配"):
        validate_prior_snapshot(
            prior, "2026-09-28", "model-a", "pool-a",
            prior["contract_fingerprint"],
        )


@pytest.mark.unit
def test_shadow_capture_reads_calendar_pool_and_predictions_under_shared_qlib_lease(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib"
    (qlib_dir / "calendars").mkdir(parents=True)
    (qlib_dir / "instruments").mkdir()
    (qlib_dir / "calendars" / "day.txt").write_text("2026-10-30\n", encoding="utf-8")
    pool_path = qlib_dir / "instruments" / "a500.txt"
    pool_path.write_text("sh600001\t2020-01-01\t2026-10-30\n", encoding="utf-8")
    run_root = tmp_path / "mlruns" / shadow_capture.MODEL[0] / shadow_capture.MODEL[1] / "artifacts"
    run_root.mkdir(parents=True)
    for name in ("params.pkl", "params_lstm.pkl", "lstm_bundle.pkl", "factor_keys.json", "pred.pkl"):
        (run_root / name).write_bytes(b"artifact")
    monkeypatch.setattr(shadow_capture.config, "QLIB_DATA_DIR", qlib_dir)
    monkeypatch.setattr(shadow_capture.config, "QLIB_MLFLOW_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(shadow_capture, "CONTRACT_FILES", [])
    monkeypatch.setattr(shadow_capture, "get_model_label_mode", lambda run_dir: "tradable_open_to_open")
    lease_events = []
    monkeypatch.setattr(
        "backend.service.qlib_access.run_with_qlib_read_access",
        lambda callback, *args, **kwargs: (
            lease_events.append("entered"), callback(*args, **kwargs), lease_events.append("exited")
        )[1],
    )

    class CapturingScreener:
        def _load_qlib_predictions(self, *args, **kwargs):
            assert lease_events == ["entered"]
            return {
                "scores": {"600001": 0.7},
                "audit": {"context": {"source": "live_model", "prediction_date": "2026-10-30"}},
            }

    monkeypatch.setattr(shadow_capture, "BuySideScreener", CapturingScreener)
    result = shadow_capture._load_qlib_inputs(date(2026, 10, 30))

    assert lease_events == ["entered", "exited"]
    assert result["calendar"] == ["2026-10-30"]
    assert result["pool_path"] == pool_path
    assert result["scores"] == {"600001": 0.7}


@pytest.mark.unit
def test_period_return_is_attributed_to_prior_decision():
    aligned = align_period_returns_to_decisions(
        [
            {"date": "2025-07-03"},
            {"date": "2025-07-10"},
            {"date": "2025-07-17"},
        ],
        [
            {"date": "2025-07-10", "period_return": 0.02},
            {"date": "2025-07-17", "period_return": -0.01},
        ],
    )
    assert aligned == {
        "2025-07-03": {"period_return": 0.02, "outcome_end_date": "2025-07-10"},
        "2025-07-10": {"period_return": -0.01, "outcome_end_date": "2025-07-17"},
    }


@pytest.mark.unit
def test_period_return_alignment_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="数量不匹配"):
        align_period_returns_to_decisions([], [{"date": "2025-07-10"}])


@pytest.mark.unit
def test_circular_block_bootstrap_is_deterministic_and_counts_pairs():
    values = [0.01, -0.02, 0.0, 0.03, -0.01, 0.02]
    first = summarize_delta(values, block_size=2, draws=50, seed=17)
    second = summarize_delta(values, block_size=2, draws=50, seed=17)
    assert first == second
    assert (first["n_periods"], first["wins"], first["losses"], first["ties"]) == (6, 3, 2, 1)


@pytest.mark.unit
def test_circular_block_bootstrap_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="有限数组"):
        summarize_delta([0.01, float("nan"), 0.0, 0.02], block_size=2)


@pytest.mark.unit
def test_forward_prediction_and_history_use_only_captured_daily_rows():
    snapshots = []
    for day, code, score in (
        ("2026-09-28", "600001", 0.8),
        ("2026-09-28", "600002", 0.7),
        ("2026-09-29", "600001", 0.9),
        ("2026-09-29", "600002", 0.6),
        ("2026-09-30", "600001", 0.85),
        ("2026-09-30", "600002", 0.5),
    ):
        snapshot_row = next((row for row in snapshots if row["prediction_date"] == day), None)
        if snapshot_row is None:
            snapshot_row = {"prediction_date": day, "ranked_scores": []}
            snapshots.append(snapshot_row)
        snapshot_row["ranked_scores"].append({
            "rank": len(snapshot_row["ranked_scores"]) + 1,
            "code": code,
            "score": score,
        })

    prediction, ranks, stable, counts = history_from_snapshots(snapshots)
    assert len(prediction) == 6
    assert ranks["2026-09-30"]["600001"] == 1
    assert counts["2026-09-30"]["600001"] == 3
    assert "600001" in stable["2026-09-30"]
    assert "600002" in stable["2026-09-30"]


@pytest.mark.unit
def test_weekly_decision_dates_snap_to_prior_session_on_holiday():
    sessions = [
        date.fromisoformat(value)
        for value in ("2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09")
    ]
    captured = set(sessions)
    decisions = weekly_decision_dates(
        sessions, captured, date.fromisoformat("2026-10-01"),
        current_anchor=date.fromisoformat("2026-09-30"),
    )
    assert decisions == [date.fromisoformat("2026-09-30")]


@pytest.mark.unit
def test_weekly_decision_dates_match_engine_after_long_holiday():
    sessions = [
        date.fromisoformat(value)
        for value in (
            "2022-01-24", "2022-01-25", "2022-01-26", "2022-01-27", "2022-01-28",
            "2022-02-07", "2022-02-08", "2022-02-09", "2022-02-10", "2022-02-11",
        )
    ]
    decisions = weekly_decision_dates(sessions, sessions, date.fromisoformat("2022-02-11"))
    assert decisions == [
        date.fromisoformat("2022-01-27"),
        date.fromisoformat("2022-01-28"),
        date.fromisoformat("2022-02-10"),
    ]


@pytest.mark.unit
def test_current_weekly_anchor_finds_previous_session_across_long_holiday(monkeypatch):
    sessions = [
        date.fromisoformat(value)
        for value in ("2022-01-27", "2022-01-28", "2022-02-07", "2022-02-08")
    ]
    monkeypatch.setattr(
        "scripts.run_rank_persistence_forward.get_trading_days_between",
        lambda start, end: sessions,
    )
    monkeypatch.setattr(
        "scripts.run_rank_persistence_forward.is_trading_day",
        lambda _: False,
    )
    assert current_weekly_anchor(
        date.fromisoformat("2022-01-28"), sessions,
        today=date.fromisoformat("2022-02-03"),
    ) == date.fromisoformat("2022-01-28")


@pytest.mark.unit
def test_current_weekly_anchor_runs_on_holiday_after_last_session(monkeypatch):
    sessions = [
        date.fromisoformat(value)
        for value in ("2026-09-28", "2026-09-29", "2026-09-30")
    ]
    monkeypatch.setattr(
        "scripts.run_rank_persistence_forward.get_trading_days_between",
        lambda start, end: sessions,
    )
    monkeypatch.setattr(
        "scripts.run_rank_persistence_forward.is_trading_day",
        lambda _: False,
    )
    last_session = date.fromisoformat("2026-09-30")
    assert current_weekly_anchor(
        last_session, sessions, today=last_session,
    ) is None
    assert current_weekly_anchor(
        last_session, sessions, today=date.fromisoformat("2026-10-01"),
    ) == last_session


@pytest.mark.unit
def test_primary_window_is_fixed_to_first_52_matured_periods():
    eligible = [date(2026, 1, 1) + pd.Timedelta(days=7 * index) for index in range(60)]
    first_window = pre_registered_primary_window(eligible[:53])
    extended_window = pre_registered_primary_window(eligible)
    assert first_window == extended_window
    assert len(first_window[2]) == 53
    assert first_window[0] == eligible[0]
    assert first_window[1] == eligible[52]


@pytest.mark.unit
def test_primary_window_is_not_available_before_52_returns_mature():
    eligible = [date(2026, 1, 1) + pd.Timedelta(days=7 * index) for index in range(52)]
    assert pre_registered_primary_window(eligible) is None


@pytest.mark.unit
def test_forward_market_data_fingerprints_are_content_and_order_stable():
    first = pd.DataFrame({
        "code": ["600001", "600001"],
        "trade_date": ["2026-09-28", "2026-09-29"],
        "close": [10.0, 11.0],
        "adj_factor": [1.0, None],
    })
    reordered = first.iloc[::-1].reset_index(drop=True)
    assert dataframe_fingerprint(first) == dataframe_fingerprint(reordered)
    first_hash = price_cache_fingerprint({"600001": first})
    reordered_hash = price_cache_fingerprint({"600001": reordered})
    assert first_hash == reordered_hash
    assert first_hash["missing_adj_factor_rows"] == 1


@pytest.mark.unit
def test_forward_market_data_fingerprint_rejects_empty_data():
    with pytest.raises(RuntimeError, match="数据表为空"):
        dataframe_fingerprint(pd.DataFrame())


@pytest.mark.unit
def test_existing_forward_evaluation_requires_both_intact_outputs(tmp_path):
    report_path = tmp_path / "report.json"
    audit_path = tmp_path / "audit.csv"
    audit = b"\xef\xbb\xbfrow\nvalue\n"
    report = {
        "evaluation_id": "evaluation-a",
        "audit_sha256": hashlib.sha256(audit).hexdigest(),
    }
    report["report_sha256"] = canonical_report_hash(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    audit_path.write_bytes(audit)
    assert validate_existing_evaluation(report_path, audit_path, "evaluation-a")

    report["changed"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="评估报告内容哈希校验失败"):
        validate_existing_evaluation(report_path, audit_path, "evaluation-a")
    report.pop("changed")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    audit_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="逐股审计表哈希校验失败"):
        validate_existing_evaluation(report_path, audit_path, "evaluation-a")

    audit_path.unlink()
    with pytest.raises(RuntimeError, match="产物不完整"):
        validate_existing_evaluation(report_path, audit_path, "evaluation-a")
