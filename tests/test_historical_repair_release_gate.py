from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def bypass_mysql_advisory_lock_for_unit_tests():
    from contextlib import nullcontext

    from backend.service import historical_repair

    with patch.object(
        historical_repair,
        "_historical_release_advisory_lock",
        side_effect=lambda *_args, **_kwargs: nullcontext(),
    ):
        yield


@pytest.mark.unit
@pytest.mark.p0
def test_release_gate_accumulates_field_source_and_baseline_blockers():
    from backend.service.historical_repair import _historical_release_gate

    status, blockers = _historical_release_gate(
        [["source status conflict"], ["database fingerprint changed"], ["field amount mismatch"]],
        "ready",
        "ready",
    )

    assert status == "blocked"
    assert blockers == ["source status conflict", "database fingerprint changed", "field amount mismatch"]


@pytest.mark.unit
@pytest.mark.p0
def test_release_gate_rejects_persisted_rule_ids_without_blocker_messages():
    from backend.service.historical_repair import _historical_release_gate

    status, blockers = _historical_release_gate(
        [[]],
        "ready",
        "ready",
        [["DA04-SOURCE-STATUS-001"]],
    )

    assert status == "blocked"
    assert blockers == [
        "DA04-SOURCE-STATUS-001: 持久化阻塞规则索引存在，但阻塞说明缺失；需重新执行G2预检",
    ]


@pytest.mark.unit
@pytest.mark.p0
def test_release_gate_accepts_index_matching_persisted_blocker():
    from backend.service.historical_repair import _historical_release_gate

    status, blockers = _historical_release_gate(
        [["DA04-SOURCE-STATUS-001: source status conflict"]],
        "ready",
        "ready",
        [["DA04-SOURCE-STATUS-001"]],
    )

    assert status == "blocked"
    assert blockers == ["DA04-SOURCE-STATUS-001: source status conflict"]


@pytest.mark.unit
@pytest.mark.p0
@pytest.mark.parametrize(
    ("blockers", "blocker_rule_ids", "assessment_status", "qlib_status", "expected"),
    [
        (["DA04-SOURCE-STATUS-001: source status conflict"], ["DA04-SOURCE-STATUS-001"], "ready", "ready", "source status conflict"),
        ([], [], "blocked", "ready", "字段/交易状态评估未通过"),
        ([], [], "ready", "blocked", "Qlib候选校验未通过"),
        ([], ["DA04-SOURCE-STATUS-001"], "ready", "ready", "阻塞说明缺失"),
    ],
)
def test_batch_publish_reuses_shared_gate_before_database_or_release_writes(
    blockers, blocker_rule_ids, assessment_status, qlib_status, expected,
):
    from contextlib import nullcontext
    from backend.service import historical_release_batch, historical_repair

    release_id = "release00001"
    manifest = {
        "release_id": release_id,
        "status": "ready",
        "publishable": True,
        "release_change_contract": "g2_effective_change_v1",
        "blockers": blockers,
        "blocker_rule_ids": blocker_rule_ids,
        "assessment": {"status": assessment_status},
        "qlib": {"status": qlib_status},
    }
    with patch.object(historical_repair, "_release_publish_lock", nullcontext()), \
         patch.object(historical_repair, "_historical_release_advisory_lock", side_effect=lambda *_args, **_kwargs: nullcontext()), \
         patch.object(historical_repair, "get_historical_release", return_value=manifest), \
         patch.object(historical_release_batch.HistoricalRepairRepo, "get_shard_inputs") as get_inputs, \
         patch.object(historical_release_batch.repair, "_release_root") as release_root:
        with pytest.raises(RuntimeError, match=expected):
            historical_release_batch.publish_historical_release_batch([release_id])

    get_inputs.assert_not_called()
    release_root.assert_not_called()
    assert manifest["status"] == "ready"
    assert manifest["publishable"] is True


@pytest.mark.unit
@pytest.mark.p0
def test_field_repair_date_axes_reject_duplicate_or_invalid_dates_before_set_merge():
    import pandas as pd
    from core.domain.data_quality.historical_release import invalid_or_duplicate_date_axes

    invalid = invalid_or_duplicate_date_axes({
        "G0": ["2020-01-02", "2020-01-03"],
        "candidate": ["2020-01-02", "2020-01-02"],
        "comparison": ["2020-01-02", "2020-01-03"],
        "TuShare": ["2020-01-02", "not-a-date"],
        "BaoStock": pd.Series(["2020-01-02", "2020-01-03"]),
    })

    assert invalid == ["candidate", "TuShare"]


@pytest.mark.unit
@pytest.mark.p0
def test_publish_rejects_component_status_conflict_even_when_manifest_says_ready(tmp_path):
    from backend.service import historical_repair

    manifest = {
        "release_id": "release-test",
        "release_change_contract": "g2_effective_change_v1",
        "status": "ready",
        "publishable": True,
        "database_repair_required": True,
        "qlib_repair_required": False,
        "blockers": [],
        "assessment": {"status": "ready"},
        "qlib": {"status": "blocked", "files": {}},
    }
    manifest_path = tmp_path / "manifest.json"

    with patch.object(historical_repair, "_release_manifest_path", return_value=manifest_path), \
         patch.object(historical_repair, "load_json", return_value=manifest), \
         patch.object(historical_repair, "save_json_atomic") as save:
        with pytest.raises(RuntimeError, match="未通过预检"):
            historical_repair._publish_historical_release("release-test")

    assert manifest["publishable"] is False
    assert "DA04-QLIB-CANDIDATE-001: Qlib候选校验未通过" in manifest["blockers"]
    save.assert_called_once()


@pytest.mark.unit
@pytest.mark.p0
def test_publish_rejects_persisted_blocker_rule_index_without_messages(tmp_path):
    from backend.service import historical_repair

    manifest = {
        "release_id": "release-test",
        "release_change_contract": "g2_effective_change_v1",
        "status": "ready",
        "publishable": True,
        "blockers": [],
        "blocker_rule_ids": ["DA04-SOURCE-STATUS-001"],
        "assessment": {"status": "ready"},
        "qlib": {"status": "ready"},
    }
    manifest_path = tmp_path / "manifest.json"

    with patch.object(historical_repair, "_release_manifest_path", return_value=manifest_path), \
         patch.object(historical_repair, "load_json", return_value=manifest), \
         patch.object(historical_repair, "save_json_atomic") as save:
        with pytest.raises(RuntimeError, match="未通过预检"):
            historical_repair._publish_historical_release("release-test")

    assert manifest["status"] == "blocked"
    assert manifest["publishable"] is False
    assert manifest["blockers"] == [
        "DA04-SOURCE-STATUS-001: 持久化阻塞规则索引存在，但阻塞说明缺失；需重新执行G2预检",
    ]
    save.assert_called_once()


@pytest.mark.unit
@pytest.mark.p0
def test_publish_rejects_database_baseline_drift_before_any_write(tmp_path):
    from backend.service import historical_repair
    from core.data.repos.historical_repair import HistoricalRepairRepo

    release_dir = tmp_path / "release"
    shadow_dir = tmp_path / "shadow"
    raw_dir = shadow_dir / "raw" / "000001"
    raw_dir.mkdir(parents=True)
    release_dir.mkdir()
    candidate_path = release_dir / "candidate.parquet"
    preimage_path = release_dir / "preimage.parquet"
    source_candidate_path = raw_dir / "2020_qfq_candidate.parquet"
    comparison_path = raw_dir / "2020_comparison.parquet"
    for path in (candidate_path, preimage_path, source_candidate_path, comparison_path):
        path.write_bytes(b"fixture")
    release_path = release_dir / "manifest.json"
    shadow_path = shadow_dir / "manifest.json"
    manifest = {
        "release_id": "release-test",
        "release_change_contract": "g2_effective_change_v1",
        "status": "ready",
        "publishable": True,
        "database_repair_required": True,
        "qlib_repair_required": False,
        "blockers": [],
        "assessment": {"status": "ready"},
        "qlib": {"status": "ready", "files": {}},
        "candidate_path": str(candidate_path),
        "preimage_path": str(preimage_path),
        "candidate_sha256": "same-hash",
        "preimage_sha256": "same-hash",
        "batch_id": "batch-test",
        "code": "000001",
        "year": 2020,
        "source_candidate_path": str(source_candidate_path),
        "source_candidate_sha256": "same-hash",
        "comparison_sha256": "same-hash",
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "input_fingerprint": "prepared-baseline",
        "qlib_generation_id": None,
    }
    shadow = {"work_items": [{"code": "000001", "year": 2020}]}

    def load_json(path):
        if path == release_path:
            return manifest
        if path == shadow_path:
            return shadow
        return None

    with patch.object(historical_repair, "_release_manifest_path", return_value=release_path), \
         patch.object(historical_repair, "_manifest_path", return_value=shadow_path), \
         patch.object(historical_repair, "load_json", side_effect=load_json), \
         patch.object(historical_repair, "_file_sha256", return_value="same-hash"), \
         patch.object(historical_repair, "_shadow_files_valid", return_value=True), \
         patch.object(HistoricalRepairRepo, "get_shard_inputs", return_value={"fingerprint": "new-baseline"}), \
         patch.object(HistoricalRepairRepo, "publish_missing_values") as publish, \
         patch.object(historical_repair, "save_json_atomic"):
        with pytest.raises(RuntimeError, match="正式库基线在预检后变化"):
            historical_repair._publish_historical_release("release-test")

    assert manifest["status"] == "blocked"
    assert manifest["publishable"] is False
    assert any("分片指纹" in blocker for blocker in manifest["blockers"])
    publish.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_file_verifier_rejects_external_change(tmp_path):
    from backend.service import historical_repair
    from core.shared import config

    root = tmp_path / "qlib"
    parent = root / "generations" / "parent"
    candidate = root / "generations" / "candidate"
    source = parent / "features" / "sz000001" / "close.day.bin"
    target = candidate / "features" / "sz000001" / "close.day.bin"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"expected-parent")
    target.write_bytes(b"expected-candidate")
    before_sha256 = historical_repair._file_sha256(source)
    manifest = {
        "qlib_generation_parent_id": "parent",
        "qlib_generation_stage_dir": str(root / "generations" / ".staging-candidate"),
        "qlib": {"files": {"close": {
            "generation_relative_path": "features/sz000001/close.day.bin",
            "before_sha256": before_sha256,
            "after_sha256": historical_repair._file_sha256(target),
        }}},
    }
    source.write_bytes(b"external-parent")
    with patch.object(config, "QLIB_DATA_ROOT", str(root)):
        with pytest.raises(RuntimeError, match="已变化，拒绝回滚"):
            historical_repair._verify_qlib_generation_release_files(manifest, "parent", "before")

    target.write_bytes(b"external-candidate")
    with patch.object(config, "QLIB_DATA_ROOT", str(root)):
        with pytest.raises(RuntimeError, match="已变化，拒绝回滚"):
            historical_repair._verify_qlib_generation_release_files(manifest, "candidate", "after")

    manifest["qlib"]["files"]["close"]["generation_relative_path"] = "../../outside.bin"
    with patch.object(config, "QLIB_DATA_ROOT", str(root)):
        with pytest.raises(RuntimeError, match="超出受管代目录"):
            historical_repair._verify_qlib_generation_release_files(manifest, "candidate", "after")


@pytest.mark.unit
@pytest.mark.p0
def test_interrupted_publish_without_after_fingerprint_is_marked_for_manual_recovery(tmp_path):
    from contextlib import nullcontext
    from backend.service import historical_repair
    from core.data.repos.historical_repair import HistoricalRepairRepo

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "release_id": "release-uncertain",
        "status": "publishing",
        "code": "000001",
        "date_range": {"start": "2020-01-01", "end": "2020-01-31"},
        "input_fingerprint": "before",
        "preimage_fingerprint": "before",
        "db_after_fingerprint": None,
    }
    with patch.object(historical_repair, "_release_manifest_path", return_value=manifest_path), \
         patch.object(historical_repair, "load_json", return_value=manifest), \
         patch.object(historical_repair, "_historical_release_advisory_lock", return_value=nullcontext()), \
         patch.object(HistoricalRepairRepo, "get_shard_inputs", return_value={"fingerprint": "unknown-commit-state"}), \
         patch.object(historical_repair, "save_json_atomic") as save:
        with pytest.raises(RuntimeError, match="缺少已提交后指纹"):
            historical_repair.rollback_historical_release("release-uncertain")

    assert manifest["status"] == "recovery_required"
    assert "拒绝推测并覆盖" in manifest["rollback_error"]
    save.assert_called_once_with(manifest, manifest_path)


@pytest.mark.unit
@pytest.mark.p0
def test_manual_recovery_requires_live_database_and_qlib_fingerprints(tmp_path):
    from backend.service import historical_repair

    manifest_path = tmp_path / "manifest.json"
    manifest = {"release_id": "release-manual", "status": "recovery_required", "db_after_fingerprint": None}
    snapshot = {
        "database_fingerprint": "a" * 64,
        "preimage_fingerprint": "b" * 64,
        "qlib_state_fingerprint": "c" * 64,
        "qlib_state": {"mode": "generation", "current_generation_id": "generation-current"},
    }
    with patch.object(historical_repair, "_release_manifest_path", return_value=manifest_path), \
         patch.object(historical_repair, "load_json", return_value=manifest), \
         patch.object(historical_repair, "_historical_release_recovery_snapshot", return_value=snapshot), \
         patch.object(historical_repair, "save_json_atomic") as save, \
         patch.object(historical_repair, "_rollback_historical_release") as rollback:
        with pytest.raises(RuntimeError, match="正式库已变化"):
            historical_repair.authorize_historical_release_recovery_rollback(
                "release-manual", "d" * 64, "c" * 64, "admin", "核验后恢复到发布前快照", "CASE-123",
            )

    save.assert_not_called()
    rollback.assert_not_called()
    assert "manual_recovery_decisions" not in manifest


@pytest.mark.unit
@pytest.mark.p0
def test_manual_recovery_rejects_changed_qlib_state(tmp_path):
    from backend.service import historical_repair

    manifest_path = tmp_path / "manifest.json"
    manifest = {"release_id": "release-manual", "status": "recovery_required", "db_after_fingerprint": None}
    snapshot = {
        "database_fingerprint": "a" * 64,
        "preimage_fingerprint": "b" * 64,
        "qlib_state_fingerprint": "c" * 64,
        "qlib_state": {"mode": "generation", "current_generation_id": "generation-current"},
    }
    with patch.object(historical_repair, "_release_manifest_path", return_value=manifest_path), \
         patch.object(historical_repair, "load_json", return_value=manifest), \
         patch.object(historical_repair, "_historical_release_recovery_snapshot", return_value=snapshot), \
         patch.object(historical_repair, "save_json_atomic") as save, \
         patch.object(historical_repair, "_rollback_historical_release") as rollback:
        with pytest.raises(RuntimeError, match="Qlib状态已变化"):
            historical_repair.authorize_historical_release_recovery_rollback(
                "release-manual", "a" * 64, "d" * 64, "admin", "核验后恢复到发布前快照", "CASE-123",
            )

    save.assert_not_called()
    rollback.assert_not_called()
    assert "manual_recovery_decisions" not in manifest


@pytest.mark.unit
@pytest.mark.p0
def test_manual_recovery_audits_operator_decision_before_resuming_rollback(tmp_path):
    from backend.service import historical_repair

    manifest_path = tmp_path / "manifest.json"
    manifest = {"release_id": "release-manual", "status": "recovery_required", "db_after_fingerprint": None}
    snapshot = {
        "database_fingerprint": "a" * 64,
        "preimage_fingerprint": "b" * 64,
        "qlib_state_fingerprint": "c" * 64,
        "qlib_state": {"mode": "generation", "current_generation_id": "generation-current"},
    }
    with patch.object(historical_repair, "_release_manifest_path", return_value=manifest_path), \
         patch.object(historical_repair, "load_json", return_value=manifest), \
         patch.object(historical_repair, "_historical_release_recovery_snapshot", return_value=snapshot), \
         patch.object(historical_repair, "save_json_atomic") as save, \
         patch.object(historical_repair, "_rollback_historical_release", return_value={"status": "rolled_back"}) as rollback:
        result = historical_repair.authorize_historical_release_recovery_rollback(
            "release-manual", "a" * 64, "c" * 64, "admin-user", "核验后恢复到发布前快照", "CASE-123",
        )

    assert result["status"] == "rolled_back"
    assert manifest["db_after_fingerprint"] == "a" * 64
    decision = manifest["manual_recovery_decisions"][0]
    assert decision["action"] == "restore_preimage"
    assert decision["actor"] == "admin-user"
    assert decision["evidence_ref"] == "CASE-123"
    assert decision["database_fingerprint"] == "a" * 64
    assert decision["qlib_state_fingerprint"] == "c" * 64
    save.assert_called_once_with(manifest, manifest_path)
    rollback.assert_called_once_with("release-manual")


@pytest.mark.unit
@pytest.mark.p0
def test_manual_recovery_api_requires_admin_before_service_call():
    import asyncio

    from fastapi import HTTPException

    from backend.api import auth, data_manager

    with patch.object(auth, "require_admin_actor", side_effect=HTTPException(403, "需要管理员权限")), \
         patch("backend.service.historical_repair.authorize_historical_release_recovery_rollback") as recover:
        with pytest.raises(HTTPException) as error:
            asyncio.run(data_manager.history_release_recovery_restore(
                "release-manual", {"reason": "核验后恢复到发布前快照"}, None,
            ))

    assert error.value.status_code == 403
    recover.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
@pytest.mark.parametrize(
    ("handler_name", "service_path"),
    [
        ("history_release_publish", "backend.service.historical_repair.publish_historical_release"),
        ("history_release_rollback", "backend.service.historical_repair.rollback_historical_release"),
    ],
)
def test_historical_release_mutations_require_admin_before_service_call(handler_name, service_path):
    import asyncio

    from fastapi import HTTPException

    from backend.api import auth, data_manager

    handler = getattr(data_manager, handler_name)
    with patch.object(auth, "require_admin_actor", side_effect=HTTPException(403, "需要管理员权限")), \
         patch(service_path) as mutate:
        with pytest.raises(HTTPException) as error:
            asyncio.run(handler("release-manual", None))

    assert error.value.status_code == 403
    mutate.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_rollback_can_resume_after_database_restore_failure(tmp_path):
    import pandas as pd

    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo

    release_path = tmp_path / "manifest.json"
    candidate_path = tmp_path / "candidate.parquet"
    preimage_path = tmp_path / "preimage.parquet"
    candidate_path.write_bytes(b"candidate")
    preimage_path.write_bytes(b"preimage")
    manifest = {
        "release_id": "release-retry",
        "status": "published",
        "batch_id": "batch-retry",
        "code": "000001",
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "rollback_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "candidate_path": str(candidate_path),
        "candidate_sha256": "candidate-hash",
        "preimage_path": str(preimage_path),
        "db_after_fingerprint": "after",
        "preimage_fingerprint": "before",
        "qlib_generation_published_id": "generation-new",
        "qlib_generation_parent_id": "generation-old",
        "qlib": {"files": {}},
    }
    batch = {"source_versions_at_collection": {"tushare": "isolated", "baostock": "isolated"}}
    current_generation = {"id": "generation-new"}

    def rollback_generation(target_id, **kwargs):
        assert target_id == "generation-old"
        assert kwargs["expected_current_generation_id"] == "generation-new"
        current_generation["id"] = "generation-old"

    def load_json(path):
        return manifest if path == release_path else batch

    with patch.object(historical_repair, "_release_manifest_path", return_value=release_path), \
         patch.object(historical_repair, "_manifest_path", return_value=tmp_path / "batch" / "manifest.json"), \
         patch.object(historical_repair, "load_json", side_effect=load_json), \
         patch.object(historical_repair, "load_parquet", return_value=pd.DataFrame([{"trade_date": "2020-01-01"}])), \
         patch.object(historical_repair, "_file_sha256", return_value="candidate-hash"), \
         patch.object(historical_repair, "save_json_atomic"), \
         patch.object(HistoricalRepairRepo, "get_shard_inputs", side_effect=[
             {"fingerprint": "after"}, {"fingerprint": "after"}, {"fingerprint": "after"},
             {"fingerprint": "after"}, {"fingerprint": "before"},
         ]), \
         patch.object(HistoricalRepairRepo, "restore_shard", side_effect=[RuntimeError("isolated DB failure"), {
             "restored_fingerprint": "before", "rows": 1, "audit_rows": 1,
         }]) as restore, \
         patch.object(generation, "has_qlib_generation_pointer", return_value=True), \
         patch.object(generation, "resolve_qlib_data_dir", side_effect=lambda _root: tmp_path / current_generation["id"]), \
         patch.object(generation, "rollback_qlib_generation", side_effect=rollback_generation) as rollback:
        with pytest.raises(RuntimeError, match="可重试恢复"):
            historical_repair.rollback_historical_release("release-retry")

        assert manifest["status"] == "rollback_required"
        assert manifest["rollback_state"] == {"qlib": "complete", "database": "pending"}

        result = historical_repair.rollback_historical_release("release-retry")

    assert result["status"] == "rolled_back"
    assert result["rollback_state"] == {"qlib": "complete", "database": "complete"}
    assert rollback.call_count == 1
    assert restore.call_count == 2


@pytest.mark.unit
@pytest.mark.p0
def test_publish_compensates_when_qlib_pointer_changes_before_publish_raises(tmp_path):
    from pathlib import Path

    import pandas as pd

    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.adapters.qlib import historical_patch
    from core.data.repos.historical_repair import HistoricalRepairRepo

    release_dir = tmp_path / "release"
    shadow_dir = tmp_path / "shadow"
    raw_dir = shadow_dir / "raw" / "000001"
    release_dir.mkdir()
    raw_dir.mkdir(parents=True)
    candidate_path = release_dir / "candidate.parquet"
    preimage_path = release_dir / "preimage.parquet"
    source_candidate_path = raw_dir / "2020_qfq_candidate.parquet"
    comparison_path = raw_dir / "2020_comparison.parquet"
    for file_path in (candidate_path, preimage_path, source_candidate_path, comparison_path):
        file_path.write_bytes(b"fixture")
    release_path = release_dir / "manifest.json"
    shadow_path = shadow_dir / "manifest.json"
    manifest = {
        "release_id": "release-pointer-fault",
        "release_change_contract": "g2_effective_change_v1",
        "status": "ready",
        "publishable": True,
        "database_repair_required": True,
        "qlib_repair_required": True,
        "blockers": [],
        "assessment": {"status": "ready"},
        "qlib": {"status": "ready", "files": {}},
        "candidate_path": str(candidate_path),
        "preimage_path": str(preimage_path),
        "candidate_sha256": "same-hash",
        "preimage_sha256": "same-hash",
        "batch_id": "batch-test",
        "code": "000001",
        "year": 2020,
        "source_candidate_path": str(source_candidate_path),
        "source_candidate_sha256": "same-hash",
        "comparison_sha256": "same-hash",
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "rollback_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "input_fingerprint": "baseline",
        "preimage_fingerprint": "baseline",
        "qlib_generation_id": "generation-old",
        "qlib_dir": str(tmp_path / "qlib" / "generations" / "generation-old"),
        "price_basis": {"anchor_date": "2020-12-31"},
    }
    shadow = {
        "work_items": [{"code": "000001", "year": 2020}],
        "source_versions_at_collection": {"tushare": "isolated", "baostock": "isolated"},
    }
    current_generation = {"id": "generation-old"}
    stage = {
        "generation_id": None,
        "stage_dir": None,
        "parent_generation_id": "generation-old",
        "data_root": tmp_path / "qlib",
    }

    def stage_generation(data_root=None, generation_id=None, expected_parent_generation_id=None, **kwargs):
        return {
            **stage,
            "generation_id": generation_id,
            "stage_dir": Path(data_root) / "generations" / f".staging-{generation_id}",
            "parent_generation_id": expected_parent_generation_id,
            "lifecycle_lease": kwargs.get("lifecycle_lease", False),
            "owner": kwargs.get("owner"),
        }

    def load_json(path):
        if path == release_path:
            return manifest
        if path == shadow_path:
            return shadow
        return None

    def publish_generation(stage_arg, *_args, **_kwargs):
        current_generation["id"] = stage_arg["generation_id"]
        raise RuntimeError("simulated exception after pointer swap")

    def rollback_generation(target_id, **kwargs):
        assert target_id == "generation-old"
        assert kwargs["expected_current_generation_id"] == current_generation["id"]
        current_generation["id"] = "generation-old"

    with patch.object(historical_repair, "_release_manifest_path", return_value=release_path), \
         patch.object(historical_repair, "_manifest_path", return_value=shadow_path), \
         patch.object(historical_repair, "load_json", side_effect=load_json), \
         patch.object(historical_repair, "load_parquet", return_value=pd.DataFrame([{"date": "2020-12-31"}])), \
         patch.object(historical_repair, "_file_sha256", return_value="same-hash"), \
         patch.object(historical_repair, "_shadow_files_valid", return_value=True), \
         patch.object(historical_repair, "save_json_atomic"), \
         patch.object(HistoricalRepairRepo, "get_shard_inputs", return_value={"fingerprint": "baseline"}), \
         patch.object(HistoricalRepairRepo, "publish_missing_values", return_value={
             "after_fingerprint": "after", "input_rows": 1, "audit_rows": 1,
         }), \
         patch.object(HistoricalRepairRepo, "restore_shard", return_value={
             "restored_fingerprint": "baseline", "rows": 1, "audit_rows": 1,
         }) as restore, \
         patch.object(generation, "stage_qlib_generation", side_effect=stage_generation), \
         patch.object(historical_patch, "promote_stock_patch"), \
         patch.object(generation, "has_qlib_generation_pointer", return_value=True), \
         patch.object(generation, "resolve_qlib_data_dir", side_effect=lambda _root: tmp_path / "qlib" / "generations" / current_generation["id"]), \
         patch.object(generation, "publish_qlib_generation", side_effect=publish_generation), \
         patch.object(generation, "rollback_qlib_generation", side_effect=rollback_generation) as rollback:
        with pytest.raises(RuntimeError, match="rolled_back"):
            historical_repair._publish_historical_release("release-pointer-fault")

    assert current_generation["id"] == "generation-old"
    assert manifest["status"] == "rolled_back"
    assert manifest.get("qlib_generation_published_id") == manifest["qlib_generation_stage_id"], (
        manifest.get("error"), manifest.get("rollback_errors"), manifest,
    )
    assert manifest["db_rollback_fingerprint"] == "baseline"
    rollback.assert_called_once()
    restore.assert_called_once()


@pytest.mark.unit
@pytest.mark.p0
def test_publish_cleans_renamed_generation_when_pointer_write_fails(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from unittest.mock import patch

    import pandas as pd

    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.adapters.qlib import historical_patch
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    root = tmp_path / "qlib"
    source = tmp_path / "qlib-source"
    for relative in ("calendars", "instruments", "features"):
        (source / relative).mkdir(parents=True, exist_ok=True)
    (source / "calendars" / "day.txt").write_text("2020-01-01\n", encoding="utf-8")
    (source / "instruments" / "all.txt").write_text("SH000001\t2020-01-01\t2020-12-31\n", encoding="utf-8")
    parent_id = "a" * 32
    parent_stage = generation.stage_qlib_generation(
        source_dir=source, data_root=root, generation_id=parent_id,
    )
    generation.publish_qlib_generation(parent_stage, expected_generation_id=None)

    release_dir = tmp_path / "release"
    shadow_dir = tmp_path / "shadow"
    raw_dir = shadow_dir / "raw" / "000001"
    release_dir.mkdir()
    raw_dir.mkdir(parents=True)
    candidate_path = release_dir / "candidate.parquet"
    preimage_path = release_dir / "preimage.parquet"
    source_candidate_path = raw_dir / "2020_qfq_candidate.parquet"
    comparison_path = raw_dir / "2020_comparison.parquet"
    for file_path in (candidate_path, preimage_path, source_candidate_path, comparison_path):
        file_path.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    release_path = release_dir / "manifest.json"
    shadow_path = shadow_dir / "manifest.json"
    manifest = {
        "release_id": "release-pointer",
        "release_change_contract": "g2_effective_change_v1",
        "status": "ready",
        "publishable": True,
        "database_repair_required": True,
        "qlib_repair_required": True,
        "blockers": [],
        "assessment": {"status": "ready"},
        "qlib": {"status": "ready", "files": {}},
        "candidate_path": str(candidate_path),
        "preimage_path": str(preimage_path),
        "candidate_sha256": digest,
        "preimage_sha256": digest,
        "batch_id": "batch-pointer",
        "code": "000001",
        "year": 2020,
        "source_candidate_path": str(source_candidate_path),
        "source_candidate_sha256": digest,
        "comparison_sha256": digest,
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "rollback_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "input_fingerprint": "baseline",
        "preimage_fingerprint": "baseline",
        "qlib_generation_id": parent_id,
        "qlib_dir": str(root / "generations" / parent_id),
        "price_basis": {"anchor_date": "2020-12-31"},
    }
    shadow = {
        "work_items": [{"code": "000001", "year": 2020}],
        "source_versions_at_collection": {"tushare": "isolated", "baostock": "isolated"},
    }
    real_load_json = historical_repair.load_json
    real_save_json_atomic = generation.save_json_atomic

    def load_json(path):
        if Path(path) == release_path:
            return manifest
        if Path(path) == shadow_path:
            return shadow
        return real_load_json(path)

    def fail_pointer_write(value, path):
        if Path(path).resolve() == (root / "current.json").resolve():
            raise OSError("simulated pointer persistence failure")
        return real_save_json_atomic(value, path)

    monkeypatch.setattr(config, "QLIB_DATA_ROOT", str(root))
    with patch.object(historical_repair, "_release_manifest_path", return_value=release_path), \
         patch.object(historical_repair, "_manifest_path", return_value=shadow_path), \
         patch.object(historical_repair, "load_json", side_effect=load_json), \
         patch.object(historical_repair, "load_parquet", return_value=pd.DataFrame([{"date": "2020-12-31"}])), \
         patch.object(historical_repair, "_shadow_files_valid", return_value=True), \
         patch.object(historical_repair, "save_json_atomic"), \
         patch.object(HistoricalRepairRepo, "get_shard_inputs", return_value={"fingerprint": "baseline"}), \
         patch.object(HistoricalRepairRepo, "publish_missing_values", return_value={
             "after_fingerprint": "after", "input_rows": 1, "audit_rows": 1,
         }), \
         patch.object(HistoricalRepairRepo, "restore_shard", return_value={
             "restored_fingerprint": "baseline", "rows": 1, "audit_rows": 1,
         }) as restore, \
         patch.object(historical_patch, "promote_stock_patch"), \
         patch.object(generation, "save_json_atomic", side_effect=fail_pointer_write):
        with pytest.raises(RuntimeError, match="rolled_back"):
            historical_repair._publish_historical_release("release-pointer")

    assert generation.resolve_qlib_data_dir(root).name == parent_id
    assert manifest["status"] == "rolled_back"
    assert manifest["qlib_generation_orphan_cleanup"] == "complete"
    candidate_id = manifest["qlib_generation_stage_id"]
    assert not (root / "generations" / candidate_id).exists()
    assert not (root / "generations" / f".staging-{candidate_id}").exists()
    restore.assert_called_once()


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_publish_and_rollback_use_real_temporary_files(tmp_path):
    from core.adapters.qlib.generation import (
        has_qlib_generation_pointer,
        publish_qlib_generation,
        resolve_qlib_data_dir,
        rollback_qlib_generation,
        stage_qlib_generation,
    )

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sh000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    first_stage = stage_qlib_generation(data_root=root)
    first = publish_qlib_generation(first_stage, expected_generation_id=None)
    second_stage = stage_qlib_generation(data_root=root)
    second = publish_qlib_generation(second_stage, expected_generation_id=first["generation_id"])

    with pytest.raises(RuntimeError, match="当前代已变化"):
        rollback_qlib_generation(
            first["generation_id"], data_root=root,
            expected_current_generation_id=first["generation_id"],
        )
    rollback_qlib_generation(
        first["generation_id"], data_root=root,
        expected_current_generation_id=second["generation_id"],
    )

    assert has_qlib_generation_pointer(root)
    assert resolve_qlib_data_dir(root).name == first["generation_id"]


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_publish_preserves_noncooperating_pointer_change(tmp_path):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    first_stage = generation.stage_qlib_generation(data_root=root)
    first = generation.publish_qlib_generation(first_stage, expected_generation_id=None)
    candidate = generation.stage_qlib_generation(data_root=root)
    external_stage = generation.stage_qlib_generation(data_root=root)
    external = generation.publish_qlib_generation(
        external_stage, expected_generation_id=first["generation_id"],
    )
    generation.save_json_atomic(first["pointer"], root / "current.json")

    original_pointer = generation._pointer
    pointer_reads = 0

    def external_pointer_change(data_root=None):
        nonlocal pointer_reads
        pointer_reads += 1
        if pointer_reads == 2:
            generation.save_json_atomic(external["pointer"], root / "current.json")
        return original_pointer(data_root)

    with patch.object(generation, "_pointer", side_effect=external_pointer_change):
        with pytest.raises(RuntimeError, match="外部修改"):
            generation.publish_qlib_generation(
                candidate, expected_generation_id=first["generation_id"],
            )

    assert original_pointer(root) == external["pointer"]
    assert (root / "generations" / candidate["generation_id"]).is_dir()
    assert generation.resolve_qlib_data_dir(root).name == external["generation_id"]


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_publish_reports_pointer_replaced_after_commit(tmp_path):
    from pathlib import Path
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    first_stage = generation.stage_qlib_generation(data_root=root)
    first = generation.publish_qlib_generation(first_stage, expected_generation_id=None)
    candidate = generation.stage_qlib_generation(data_root=root)
    external_stage = generation.stage_qlib_generation(data_root=root)
    external = generation.publish_qlib_generation(
        external_stage, expected_generation_id=first["generation_id"],
    )
    generation.save_json_atomic(first["pointer"], root / "current.json")

    real_save = generation.save_json_atomic

    def save_and_replace_pointer(value, path):
        result = real_save(value, path)
        if Path(path) == root / "current.json" and value.get("generation_id") == candidate["generation_id"]:
            real_save(external["pointer"], path)
        return result

    with patch.object(generation, "save_json_atomic", side_effect=save_and_replace_pointer):
        with pytest.raises(RuntimeError, match="外部修改"):
            generation.publish_qlib_generation(
                candidate, expected_generation_id=first["generation_id"],
            )

    assert generation._pointer(root) == external["pointer"]
    assert (root / "generations" / candidate["generation_id"]).is_dir()


@pytest.mark.unit
@pytest.mark.p0
def test_discard_stage_waits_for_shared_generation_mutation_lock(tmp_path):
    import threading
    from pathlib import Path

    from core.adapters.qlib.generation import (
        discard_qlib_generation_stage,
        qlib_generation_mutation_lock,
        stage_qlib_generation,
    )

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    stage = stage_qlib_generation(data_root=root)
    started = threading.Event()
    finished = threading.Event()

    def discard():
        started.set()
        discard_qlib_generation_stage(stage)
        finished.set()

    with qlib_generation_mutation_lock(root):
        worker = threading.Thread(target=discard)
        worker.start()
        assert started.wait(2)
        assert not finished.wait(0.1)

    worker.join(2)
    assert not worker.is_alive()
    assert finished.is_set()
    assert not Path(stage["stage_dir"]).exists()


@pytest.mark.unit
@pytest.mark.p0
def test_discard_stage_refuses_an_active_generation_directory(tmp_path):
    from pathlib import Path

    from core.adapters.qlib.generation import (
        discard_qlib_generation_stage,
        publish_qlib_generation,
        stage_qlib_generation,
    )

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    stage = stage_qlib_generation(data_root=root)
    published = publish_qlib_generation(stage, expected_generation_id=None)
    final_dir = Path(published["data_dir"])
    Path(stage["stage_dir"]).mkdir()

    with pytest.raises(RuntimeError, match="活动代，拒绝清理"):
        discard_qlib_generation_stage(stage)

    assert final_dir.is_dir()


@pytest.mark.unit
@pytest.mark.p0
def test_benchmark_sync_discards_stage_when_generation_publish_loses_parent_race(tmp_path):
    from pathlib import Path

    from backend.service import collection
    from core.adapters.qlib import generation
    from core.shared import config

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    initial = generation.publish_qlib_generation(
        generation.stage_qlib_generation(data_root=root), expected_generation_id=None,
    )

    class BackgroundTasks:
        def add_task(self, task):
            self.task = task

    class Bridge:
        def __init__(self, data_dir):
            self.calendar_dir = Path(data_dir) / "calendars"

    jobs = []
    with patch.object(config, "QLIB_DATA_ROOT", str(root)), \
         patch("core.adapters.qlib.bridge.QlibDataBridge", Bridge), \
         patch.object(collection, "sync_benchmark_indices", return_value={"status": "ready"}), \
         patch.object(collection, "_set_job", side_effect=lambda job_id, data: jobs.append(data)), \
         patch.object(generation, "publish_qlib_generation", side_effect=RuntimeError("Qlib当前代已变化")):
        background = BackgroundTasks()
        response = collection.CollectionService.sync_benchmarks(background)
        background.task()

    assert response["data"]["status"] == "started"
    assert jobs[-1]["status"] == "error"
    assert list((root / "generations").glob(".staging-*")) == []
    assert generation.resolve_qlib_data_dir(root).name == initial["generation_id"]
    assert collection._sync_lock.acquire(blocking=False)
    collection._sync_lock.release()


@pytest.mark.unit
@pytest.mark.p0
def test_legacy_qlib_sync_refuses_direct_write_after_generation_pointer_appears(tmp_path):
    from core.adapters.qlib.bridge import QlibDataBridge

    root = tmp_path / "qlib"
    root.mkdir()
    (root / "current.json").write_text("{}", encoding="utf-8")
    bridge = QlibDataBridge(root)

    with patch.object(bridge, "_sync_generation", return_value=None), \
         patch.object(bridge, "_sync_incremental_in_place") as direct_sync, \
         patch("core.shared.config.QLIB_DATA_ROOT", root), \
         patch("core.shared.config.QLIB_DATA_DIR", root):
        with pytest.raises(RuntimeError, match="拒绝直接修改旧数据目录"):
            bridge.sync_incremental()

    direct_sync.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_legacy_fundamental_export_requires_shared_qlib_lock(tmp_path):
    from scripts import dump_fundamental_to_qlib
    from core.adapters.qlib import generation
    from core.shared import config

    root = tmp_path / "qlib"
    root.mkdir()
    with patch.object(config, "QLIB_DATA_ROOT", root), \
         patch.object(config, "QLIB_DATA_DIR", root), \
         patch.object(generation, "has_qlib_generation_pointer", return_value=False), \
         patch.object(generation, "qlib_generation_mutation_lock", side_effect=RuntimeError("mutation lock reached")), \
         patch.object(dump_fundamental_to_qlib, "_dump_fundamental_in_place") as direct_export:
        with pytest.raises(RuntimeError, match="mutation lock reached"):
            dump_fundamental_to_qlib.dump_fundamental()

    direct_export.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_fundamental_export_rejects_non_active_generation_path(tmp_path):
    from scripts import dump_fundamental_to_qlib
    from core.shared import config

    root = tmp_path / "qlib"
    root.mkdir()
    active = root / "generations" / "active"
    old = root / "generations" / "old"
    active.mkdir(parents=True)
    old.mkdir()

    with patch.object(config, "QLIB_DATA_ROOT", root), \
         patch.object(config, "QLIB_DATA_DIR", active), \
         patch.object(dump_fundamental_to_qlib, "_dump_fundamental_in_place") as direct_export:
        with pytest.raises(RuntimeError, match="非当前活动代路径"):
            dump_fundamental_to_qlib.dump_fundamental(data_dir=old)

    direct_export.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_fundamental_export_publishes_verified_generation_stage(tmp_path):
    from scripts import dump_fundamental_to_qlib
    from core.adapters.qlib import generation
    from core.shared import config

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    initial = generation.publish_qlib_generation(
        generation.stage_qlib_generation(data_root=root), expected_generation_id=None,
    )
    active_dir = root / "generations" / initial["generation_id"]

    with patch.object(config, "QLIB_DATA_ROOT", root), \
         patch.object(config, "QLIB_DATA_DIR", active_dir), \
         patch.object(dump_fundamental_to_qlib, "_dump_fundamental_in_place", return_value={"source_snapshot_sha256": "snapshot"}):
        result = dump_fundamental_to_qlib.dump_fundamental()

    assert result["generation_id"] != initial["generation_id"]
    assert result["restart_required"] is True
    assert generation.resolve_qlib_data_dir(root).name == result["generation_id"]


@pytest.mark.unit
@pytest.mark.p0
def test_rollback_resumes_after_partial_unmanaged_qlib_file_restore(tmp_path):
    from contextlib import nullcontext
    import hashlib
    import pandas as pd

    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo

    release_dir = tmp_path / "release"
    backup_dir = release_dir / "preimage" / "qlib"
    backup_dir.mkdir(parents=True)
    release_path = release_dir / "manifest.json"
    candidate_path = release_dir / "candidate.parquet"
    preimage_path = release_dir / "preimage.parquet"
    candidate_path.write_bytes(b"candidate")
    preimage_path.write_bytes(b"preimage")
    records = {}
    for field, contents, after_contents in (
        ("close", b"before-close", b"after-close"),
        ("volume", b"before-volume", b"after-volume"),
    ):
        source = tmp_path / f"{field}.day.bin"
        backup = backup_dir / f"{field}.day.bin"
        source.write_bytes(after_contents if field == "close" else contents)
        backup.write_bytes(contents)
        records[field] = {
            "source": str(source),
            "before_sha256": hashlib.sha256(contents).hexdigest(),
            "after_sha256": hashlib.sha256(after_contents).hexdigest(),
            "backup_name": backup.name,
        }
    manifest = {
        "release_id": "release-partial-files",
        "status": "rollback_required",
        "batch_id": "batch-partial",
        "code": "000001",
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "rollback_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "candidate_path": str(candidate_path),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "preimage_path": str(preimage_path),
        "db_after_fingerprint": "after-db",
        "preimage_fingerprint": "before-db",
        "qlib": {"files": records},
    }
    batch = {"source_versions_at_collection": {"tushare": "isolated", "baostock": "isolated"}}

    def load_json(path):
        return manifest if path == release_path else batch

    with patch.object(historical_repair, "_release_manifest_path", return_value=release_path), \
         patch.object(historical_repair, "_manifest_path", return_value=tmp_path / "batch" / "manifest.json"), \
         patch.object(historical_repair, "load_json", side_effect=load_json), \
         patch.object(historical_repair, "load_parquet", return_value=pd.DataFrame([{"trade_date": "2020-01-01"}])), \
         patch.object(historical_repair, "save_json_atomic"), \
         patch.object(HistoricalRepairRepo, "get_shard_inputs", return_value={"fingerprint": "before-db"}), \
         patch.object(generation, "has_qlib_generation_pointer", return_value=False), \
         patch.object(generation, "qlib_generation_mutation_lock", side_effect=lambda *_args, **_kwargs: nullcontext()), \
         patch.object(HistoricalRepairRepo, "restore_shard") as restore:
        result = historical_repair.rollback_historical_release("release-partial-files")

    assert result["status"] == "rolled_back"
    assert result["rollback_state"] == {"qlib": "complete", "database": "complete"}
    assert (tmp_path / "close.day.bin").read_bytes() == b"before-close"
    assert (tmp_path / "volume.day.bin").read_bytes() == b"before-volume"
    restore.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_rollback_refuses_to_overwrite_qlib_file_changed_after_release(tmp_path):
    import hashlib
    import pandas as pd

    from backend.service import historical_repair
    from core.data.repos.historical_repair import HistoricalRepairRepo

    release_dir = tmp_path / "release"
    release_dir.mkdir()
    release_path = release_dir / "manifest.json"
    candidate_path = release_dir / "candidate.parquet"
    preimage_path = release_dir / "preimage.parquet"
    source = tmp_path / "close.day.bin"
    candidate_path.write_bytes(b"candidate")
    preimage_path.write_bytes(b"preimage")
    source.write_bytes(b"changed-after-release")
    manifest = {
        "release_id": "release-qlib-conflict",
        "status": "published",
        "batch_id": "batch-conflict",
        "code": "000001",
        "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "rollback_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "candidate_path": str(candidate_path),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "preimage_path": str(preimage_path),
        "db_after_fingerprint": "after-db",
        "preimage_fingerprint": "before-db",
        "qlib": {"files": {"close": {
            "source": str(source),
            "before_sha256": hashlib.sha256(b"before-release").hexdigest(),
            "after_sha256": hashlib.sha256(b"after-release").hexdigest(),
            "backup_name": "close.day.bin",
        }}},
    }
    batch = {"source_versions_at_collection": {"tushare": "isolated", "baostock": "isolated"}}

    def load_json(path):
        return manifest if path == release_path else batch

    with patch.object(historical_repair, "_release_manifest_path", return_value=release_path), \
         patch.object(historical_repair, "_manifest_path", return_value=tmp_path / "batch" / "manifest.json"), \
         patch.object(historical_repair, "load_json", side_effect=load_json), \
         patch.object(historical_repair, "load_parquet", return_value=pd.DataFrame([{"trade_date": "2020-01-01"}])), \
         patch.object(historical_repair, "save_json_atomic") as save, \
         patch.object(HistoricalRepairRepo, "get_shard_inputs", return_value={"fingerprint": "after-db"}), \
         patch.object(HistoricalRepairRepo, "restore_shard") as restore:
        with pytest.raises(RuntimeError, match="在发布后已变化"):
            historical_repair.rollback_historical_release("release-qlib-conflict")

    assert source.read_bytes() == b"changed-after-release"
    assert manifest["status"] == "published"
    save.assert_not_called()
    restore.assert_not_called()


@pytest.mark.unit
@pytest.mark.p0
def test_publish_and_rollback_entries_serialize_release_mutations():
    import threading

    from backend.service import historical_repair

    first_entered = threading.Event()
    second_entered = threading.Event()
    second_calling = threading.Event()
    release_first = threading.Event()
    calls = []

    def rollback_inner(_release_id):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        return "done"

    def run_second():
        second_calling.set()
        return historical_repair.rollback_historical_release("release-concurrent")

    with patch.object(historical_repair, "_rollback_historical_release", side_effect=rollback_inner):
        first = threading.Thread(
            target=historical_repair.rollback_historical_release,
            args=("release-concurrent",),
            name="first-rollback",
        )
        second = threading.Thread(target=run_second, name="second-rollback")
        first.start()
        assert first_entered.wait(timeout=2)
        second.start()
        assert second_calling.wait(timeout=2)
        try:
            assert not second_entered.wait(timeout=0.1)
        finally:
            release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert calls == ["first-rollback", "second-rollback"]
    assert second_entered.is_set()
