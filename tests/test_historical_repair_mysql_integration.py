import os
import uuid
import hashlib
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pymysql
import pytest


def _process_try_release_lock(database_name, release_id, output):
    from backend.service.historical_repair import _historical_release_advisory_lock
    from core.shared import config
    import core.data.connection as connection_module

    config.MYSQL_DATABASE = database_name
    try:
        with _historical_release_advisory_lock(release_id, timeout=0):
            output.put("acquired")
    except RuntimeError:
        output.put("blocked")
    finally:
        if connection_module._pool is not None:
            connection_module._pool.shutdown()
            connection_module._pool = None


def _process_publish_historical_release(database_name, qlib_root, release_id, release_manifest_path,
                                        batch_manifest_path, entered_publish, output, pause_before_publish=False):
    import time

    from backend.service import historical_repair
    from core.shared import config
    import core.data.connection as connection_module

    config.MYSQL_DATABASE = database_name
    config.QLIB_DATA_ROOT = str(qlib_root)
    publish_inner = historical_repair._publish_historical_release

    def publish_with_signal(current_release_id):
        entered_publish.set()
        if pause_before_publish:
            time.sleep(0.5)
        return publish_inner(current_release_id)

    try:
        with patch.object(historical_repair, "_release_manifest_path", return_value=Path(release_manifest_path)), \
             patch.object(historical_repair, "_manifest_path", return_value=Path(batch_manifest_path)), \
             patch.object(historical_repair, "_publish_historical_release", side_effect=publish_with_signal):
            result = historical_repair.publish_historical_release(release_id)
        output.put(("published", result.get("status")))
    except Exception as error:
        output.put(("error", str(error)))
    finally:
        if connection_module._pool is not None:
            connection_module._pool.shutdown()
            connection_module._pool = None


def _process_hold_qlib_generation_lock(data_root, started, acquired, release):
    from core.adapters.qlib.generation import qlib_generation_mutation_lock

    started.set()
    with qlib_generation_mutation_lock(data_root):
        acquired.set()
        release.wait(10)


def _process_publish_qlib_generation(stage, started, completed, output):
    from core.adapters.qlib.generation import publish_qlib_generation

    started.set()
    try:
        result = publish_qlib_generation(
            stage,
            metadata={"operation": "isolated_concurrent_test"},
            expected_generation_id=stage["parent_generation_id"],
        )
        output.put(("published", result["generation_id"]))
    except Exception as error:
        output.put(("error", str(error)))
    finally:
        completed.set()


def _process_discard_qlib_generation_stage(stage, started, completed, output):
    from core.adapters.qlib.generation import discard_qlib_generation_stage

    started.set()
    try:
        discard_qlib_generation_stage(stage)
        output.put(("discarded", stage["generation_id"]))
    except Exception as error:
        output.put(("error", str(error)))
    finally:
        completed.set()


def _process_crash_during_historical_release_publish(database_name, qlib_root, release_id,
                                                    release_manifest_path, batch_manifest_path, output):
    import os

    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.shared import config
    import core.data.connection as connection_module

    config.MYSQL_DATABASE = database_name
    config.QLIB_DATA_ROOT = str(qlib_root)
    historical_repair._release_manifest_path = lambda current_release_id: Path(release_manifest_path)
    historical_repair._manifest_path = lambda batch_id: Path(batch_manifest_path)
    save_json_atomic = generation.save_json_atomic

    def stop_before_pointer_persist(pointer, path):
        if Path(path).resolve() == (Path(qlib_root) / "current.json").resolve():
            os._exit(86)
        return save_json_atomic(pointer, path)

    generation.save_json_atomic = stop_before_pointer_persist
    try:
        historical_repair.publish_historical_release(release_id)
        output.put(("unexpected", "publish returned without reaching crash point"))
    except BaseException as error:
        output.put(("error", f"{type(error).__name__}: {error}"))
    finally:
        if connection_module._pool is not None:
            connection_module._pool.shutdown()
            connection_module._pool = None


def _process_recover_historical_release(database_name, qlib_root, release_id,
                                        release_manifest_path, batch_manifest_path, output):
    from backend.service import historical_repair
    from core.shared import config
    import core.data.connection as connection_module

    config.MYSQL_DATABASE = database_name
    config.QLIB_DATA_ROOT = str(qlib_root)
    historical_repair._release_manifest_path = lambda current_release_id: Path(release_manifest_path)
    historical_repair._manifest_path = lambda batch_id: Path(batch_manifest_path)
    try:
        result = historical_repair.rollback_historical_release(release_id)
        output.put(("recovered", result.get("status")))
    except Exception as error:
        output.put(("error", f"{type(error).__name__}: {error}"))
    finally:
        if connection_module._pool is not None:
            connection_module._pool.shutdown()
            connection_module._pool = None


@pytest.fixture
def isolated_mysql_schema():
    if os.environ.get("HISTORICAL_REPAIR_TEST_MYSQL") != "1":
        pytest.skip("set HISTORICAL_REPAIR_TEST_MYSQL=1 to run isolated MySQL integration")

    from core.shared import config
    import core.data.connection as connection_module

    original_pool = connection_module._pool
    database_name = "codex_da04_" + uuid.uuid4().hex[:12]
    original_database = config.MYSQL_DATABASE
    admin = pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
    )
    if original_pool is not None:
        if original_pool.stats()["borrowed"] != 0:
            admin.close()
            raise RuntimeError("Refusing to detach an application connection pool with borrowed connections")
        connection_module._pool = None
    pool = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE `{database_name}`")
            cursor.execute("""CREATE TABLE daily_quotes (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                code VARCHAR(10) NOT NULL,
                trade_date DATE NOT NULL,
                open DECIMAL(18,4), high DECIMAL(18,4), low DECIMAL(18,4), close DECIMAL(18,4),
                volume BIGINT, amount DECIMAL(24,4), amplitude DECIMAL(18,6), pct_chg DECIMAL(18,6),
                turnover DECIMAL(18,6), adj_factor DECIMAL(18,8),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_code_date (code, trade_date)
            ) ENGINE=InnoDB""")
            cursor.execute("""CREATE TABLE stock_identity (
                ts_code VARCHAR(16), symbol VARCHAR(10), exchange VARCHAR(8), list_date DATE, delist_date DATE,
                KEY ix_symbol (symbol)
            ) ENGINE=InnoDB""")
            cursor.execute("""CREATE TABLE daily_quote_ingest_audit (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                code VARCHAR(10) NOT NULL, trade_date DATE NOT NULL, task_id VARCHAR(64), batch_id VARCHAR(64) NOT NULL,
                source VARCHAR(32) NOT NULL, provider_version VARCHAR(64) NOT NULL, adjustment_mode VARCHAR(16) NOT NULL,
                anchor_date DATE NOT NULL, captured_at DATETIME(6) NOT NULL, input_payload_sha256 CHAR(64),
                action VARCHAR(16) NOT NULL, old_payload_sha256 CHAR(64), new_payload_sha256 CHAR(64)
            ) ENGINE=InnoDB""")
            cursor.execute("INSERT INTO stock_identity VALUES ('000001.SZ','000001','sz','1991-04-03',NULL)")
            cursor.execute("""INSERT INTO daily_quotes
                (code,trade_date,open,high,low,close,volume,amount,amplitude,pct_chg,turnover,adj_factor)
                VALUES ('000001','2020-01-02',10,11,9,10.5,1000,NULL,1,0.5,0.2,1.25)""")
        config.MYSQL_DATABASE = database_name
        pool = connection_module._get_pool()
        yield admin, database_name
    finally:
        if pool is not None:
            pool.shutdown()
        config.MYSQL_DATABASE = original_database
        connection_module._pool = original_pool
        with admin.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


@pytest.mark.integration
@pytest.mark.p0
def test_qlib_generation_pointer_lock_serializes_cross_process_publishers(tmp_path):
    from core.adapters.qlib.generation import (
        discard_qlib_generation_stage,
        has_qlib_generation_pointer,
        publish_qlib_generation,
        resolve_qlib_data_dir,
        stage_qlib_generation,
    )

    context = multiprocessing.get_context("spawn")
    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    first_stage = stage_qlib_generation(data_root=root)
    initial = publish_qlib_generation(first_stage, expected_generation_id=None)
    first_candidate = stage_qlib_generation(data_root=root)
    second_candidate = stage_qlib_generation(data_root=root)

    lock_started = context.Event()
    lock_acquired = context.Event()
    release_lock = context.Event()
    lock_process = context.Process(
        target=_process_hold_qlib_generation_lock,
        args=(str(root), lock_started, lock_acquired, release_lock),
    )
    output = context.Queue()
    first_started = context.Event()
    second_started = context.Event()
    first_completed = context.Event()
    second_completed = context.Event()
    first_process = context.Process(
        target=_process_publish_qlib_generation,
        args=(first_candidate, first_started, first_completed, output),
    )
    second_process = context.Process(
        target=_process_publish_qlib_generation,
        args=(second_candidate, second_started, second_completed, output),
    )

    lock_process.start()
    assert lock_started.wait(5)
    assert lock_acquired.wait(5)
    first_process.start()
    second_process.start()
    try:
        assert first_started.wait(5)
        assert second_started.wait(5)
        assert not first_completed.wait(0.3)
        assert not second_completed.wait(0.3)
        assert resolve_qlib_data_dir(root).name == initial["generation_id"]
    finally:
        release_lock.set()

    results = [output.get(timeout=10), output.get(timeout=10)]
    lock_process.join(10)
    first_process.join(10)
    second_process.join(10)
    assert lock_process.exitcode == 0
    assert first_process.exitcode == 0
    assert second_process.exitcode == 0
    assert sorted(result[0] for result in results) == ["error", "published"]
    winner_id = next(result[1] for result in results if result[0] == "published")
    loser_error = next(result[1] for result in results if result[0] == "error")
    loser = second_candidate if second_candidate["generation_id"] != winner_id else first_candidate
    assert "当前代已变化" in loser_error
    assert has_qlib_generation_pointer(root)
    assert resolve_qlib_data_dir(root).name == winner_id
    assert (root / "generations" / winner_id).is_dir()
    assert not (root / "generations" / (second_candidate["generation_id"] if winner_id == first_candidate["generation_id"] else first_candidate["generation_id"])).exists()
    discard_qlib_generation_stage(loser)


@pytest.mark.integration
@pytest.mark.p0
def test_cross_process_stage_discard_and_publish_are_serialized(tmp_path):
    from core.adapters.qlib.generation import (
        has_qlib_generation_pointer,
        publish_qlib_generation,
        resolve_qlib_data_dir,
        stage_qlib_generation,
    )

    context = multiprocessing.get_context("spawn")
    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    initial = publish_qlib_generation(
        stage_qlib_generation(data_root=root), expected_generation_id=None,
    )
    candidate = stage_qlib_generation(data_root=root)

    lock_started = context.Event()
    lock_acquired = context.Event()
    release_lock = context.Event()
    lock_process = context.Process(
        target=_process_hold_qlib_generation_lock,
        args=(str(root), lock_started, lock_acquired, release_lock),
    )
    publisher_started = context.Event()
    publisher_completed = context.Event()
    discard_started = context.Event()
    discard_completed = context.Event()
    output = context.Queue()
    publisher = context.Process(
        target=_process_publish_qlib_generation,
        args=(candidate, publisher_started, publisher_completed, output),
    )
    cleaner = context.Process(
        target=_process_discard_qlib_generation_stage,
        args=(candidate, discard_started, discard_completed, output),
    )

    lock_process.start()
    assert lock_started.wait(5)
    assert lock_acquired.wait(5)
    publisher.start()
    cleaner.start()
    try:
        assert publisher_started.wait(5)
        assert discard_started.wait(5)
        assert not publisher_completed.wait(0.3)
        assert not discard_completed.wait(0.3)
        assert resolve_qlib_data_dir(root).name == initial["generation_id"]
    finally:
        release_lock.set()

    results = [output.get(timeout=10), output.get(timeout=10)]
    lock_process.join(10)
    publisher.join(10)
    cleaner.join(10)
    assert lock_process.exitcode == 0
    assert publisher.exitcode == 0
    assert cleaner.exitcode == 0
    assert sorted(result[0] for result in results) in (["discarded", "error"], ["discarded", "published"])
    assert has_qlib_generation_pointer(root)
    active_id = resolve_qlib_data_dir(root).name
    assert active_id in (initial["generation_id"], candidate["generation_id"])
    assert (root / "generations" / active_id / "calendars" / "day.txt").is_file()
    assert not Path(candidate["stage_dir"]).exists()
    if active_id == initial["generation_id"]:
        error = next(result[1] for result in results if result[0] == "error")
        assert "候选目录" in error


def _service_release_fixture(tmp_path, database_name, qlib_fields=("close",)):
    from core.adapters.qlib.generation import publish_qlib_generation, stage_qlib_generation
    from core.adapters.qlib.historical_patch import prepare_stock_patch
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.data.sources.historical_quotes import get_source_versions
    from core.shared.persistence import save_json_atomic, save_parquet

    release_id = uuid.uuid4().hex[:12]
    batch_id = uuid.uuid4().hex[:12]
    release_dir = tmp_path / "releases" / release_id
    batch_dir = tmp_path / "batches" / batch_id
    raw_dir = batch_dir / "raw" / "000001"
    raw_dir.mkdir(parents=True)
    release_dir.mkdir(parents=True)

    inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    candidate = pd.DataFrame([{
        "date": pd.Timestamp("2020-01-02"), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
        "volume": 1000, "amount": 10000.0, "amplitude": 1.0, "pct_chg": 0.5,
        "turnover": 0.2, "adj_factor": 1.25,
    }])
    candidate_path = release_dir / "candidate.parquet"
    source_candidate_path = raw_dir / "2020_qfq_candidate.parquet"
    comparison_path = raw_dir / "2020_comparison.parquet"
    preimage_path = release_dir / "preimage.parquet"
    save_parquet(candidate, candidate_path)
    save_parquet(candidate, source_candidate_path)
    comparison_path.write_bytes(b"isolated-comparison")
    save_parquet(pd.DataFrame(inputs["rows"]), preimage_path)

    qlib_root = tmp_path / "qlib"
    qlib_code = qlib_root / "features" / "sz000001"
    (qlib_root / "calendars").mkdir(parents=True)
    (qlib_root / "instruments").mkdir()
    qlib_code.mkdir(parents=True)
    (qlib_root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (qlib_root / "instruments" / "all.txt").write_text("sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    for field in qlib_fields:
        np.asarray([0.0, np.nan], dtype="<f").tofile(qlib_code / f"{field}.day.bin")
    initial_stage = stage_qlib_generation(data_root=qlib_root)
    initial_generation = publish_qlib_generation(initial_stage, metadata={"purpose": "isolated-test"})
    current_qlib_dir = Path(initial_generation["data_dir"])
    patch_stage = tmp_path / "qlib-patch"
    qlib_patch = prepare_stock_patch(
        current_qlib_dir, "000001", candidate[["date", *qlib_fields]], patch_stage, fields=qlib_fields,
    )
    assert qlib_patch["status"] == "ready"
    qlib_files = qlib_patch["files"]

    file_hash = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    shadow_item = {
        "code": "000001", "year": 2020,
        "files": {
            "qfq_candidate": file_hash(source_candidate_path),
            "comparison": file_hash(comparison_path),
        },
    }
    batch_manifest_path = batch_dir / "manifest.json"
    save_json_atomic({
        "batch_id": batch_id,
        "pipeline_contract_version": "isolated-test",
        "source_versions_at_collection": get_source_versions(),
        "work_items": [shadow_item],
    }, batch_manifest_path)

    release_manifest_path = release_dir / "manifest.json"
    release_manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "release_change_contract": "g2_effective_change_v1",
        "status": "ready",
        "publishable": True,
        "database_repair_required": True,
        "qlib_repair_required": True,
        "blockers": [],
        "assessment": {"status": "ready", "matched_source_rows": 1, "candidate_rows": 1},
        "qlib": {"status": "ready", "files": qlib_files},
        "candidate_path": str(candidate_path),
        "candidate_sha256": file_hash(candidate_path),
        "source_candidate_path": str(source_candidate_path),
        "source_candidate_sha256": file_hash(source_candidate_path),
        "comparison_sha256": file_hash(comparison_path),
        "batch_id": batch_id,
        "code": "000001",
        "year": 2020,
        "date_range": {"start": "2020-01-01", "end": "2020-01-04"},
        "rollback_range": {"start": "2020-01-01", "end": "2020-01-04"},
        "input_fingerprint": inputs["fingerprint"],
        "preimage_fingerprint": inputs["fingerprint"],
        "preimage_path": str(preimage_path),
        "preimage_sha256": file_hash(preimage_path),
        "qlib_generation_id": initial_generation["generation_id"],
        "qlib_dir": str(current_qlib_dir),
        "price_basis": {"anchor_date": "2020-01-02", "anchor_factor": 1.25},
        "repair_fields": ["amount"],
        "full_stock_reanchor": None,
    }
    save_json_atomic(release_manifest, release_manifest_path)
    return {
        "batch_id": batch_id,
        "batch_manifest_path": batch_manifest_path,
        "initial_generation_id": initial_generation["generation_id"],
        "qlib_root": qlib_root,
        "release_id": release_id,
        "release_manifest_path": release_manifest_path,
        "release_manifest": release_manifest,
    }


@pytest.mark.integration
@pytest.mark.p0
def test_historical_repair_publish_rollback_and_mid_transaction_failure(isolated_mysql_schema):
    from core.data.repos.historical_repair import HistoricalRepairRepo

    admin, database_name = isolated_mysql_schema
    initial = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    frame = pd.DataFrame([{
        "date": "2020-01-02", "open": 10, "high": 11, "low": 9, "close": 10.5,
        "volume": 1000, "amount": 10000, "amplitude": 1, "pct_chg": 0.5,
        "turnover": 0.2, "adj_factor": 1.25,
    }])
    provenance = {
        "batch_id": "isolated", "source": "tushare+baostock", "provider_version": "isolated-mysql-test",
        "adjustment_mode": "qfq", "anchor_date": "2020-01-02", "input_payload_sha256": "a" * 64,
        "task_id": "da04-test",
    }

    published = HistoricalRepairRepo.publish_missing_values(
        "000001", "2020-01-01", "2020-01-04", frame, initial["fingerprint"], provenance,
        repair_fields=("amount",),
    )
    rolled_back = HistoricalRepairRepo.restore_shard(
        "000001", "2020-01-01", "2020-01-04", initial["rows"], published["after_fingerprint"], provenance,
    )
    restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")

    assert published["audit_rows"] == 1
    assert rolled_back["rows"] == 1
    assert restored["fingerprint"] == initial["fingerprint"]

    with admin.cursor() as cursor:
        cursor.execute(f"USE `{database_name}`")
        cursor.execute("""INSERT INTO daily_quotes
            (code,trade_date,open,high,low,close,volume,amount,amplitude,pct_chg,turnover,adj_factor)
            VALUES ('000001','2020-01-03',10,11,9,10.5,1000,NULL,1,0.5,0.2,1.25)""")
        cursor.execute("""CREATE TRIGGER fail_test_audit BEFORE INSERT ON daily_quote_ingest_audit
            FOR EACH ROW BEGIN
                IF NEW.trade_date='2020-01-03' THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected audit failure';
                END IF;
            END""")

    before_failure = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    failed_frame = frame.copy()
    failed_frame["date"] = ["2020-01-03"]
    with pytest.raises(pymysql.err.OperationalError, match="injected audit failure"):
        HistoricalRepairRepo.publish_missing_values(
            "000001", "2020-01-01", "2020-01-04", failed_frame,
            before_failure["fingerprint"], provenance, repair_fields=("amount",),
        )

    with admin.cursor() as cursor:
        cursor.execute("SELECT amount FROM daily_quotes WHERE code='000001' AND trade_date='2020-01-03'")
        assert cursor.fetchone()[0] is None
        cursor.execute("SELECT COUNT(*) FROM daily_quote_ingest_audit WHERE trade_date='2020-01-03'")
        assert cursor.fetchone()[0] == 0
        cursor.execute("DROP TRIGGER fail_test_audit")

    before_rollback = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    republished = HistoricalRepairRepo.publish_missing_values(
        "000001", "2020-01-01", "2020-01-04", frame, before_rollback["fingerprint"], provenance,
        repair_fields=("amount",),
    )
    with admin.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM daily_quote_ingest_audit WHERE action='rollback'")
        rollback_audits_before_failure = cursor.fetchone()[0]
        cursor.execute("""CREATE TRIGGER fail_test_rollback_audit BEFORE INSERT ON daily_quote_ingest_audit
            FOR EACH ROW BEGIN
                IF NEW.action='rollback' AND NEW.trade_date='2020-01-02' THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected rollback audit failure';
                END IF;
            END""")

    with pytest.raises(pymysql.err.OperationalError, match="injected rollback audit failure"):
        HistoricalRepairRepo.restore_shard(
            "000001", "2020-01-01", "2020-01-04", republished["before_rows"],
            republished["after_fingerprint"], provenance,
        )
    with admin.cursor() as cursor:
        cursor.execute("SELECT amount FROM daily_quotes WHERE code='000001' AND trade_date='2020-01-02'")
        assert float(cursor.fetchone()[0]) == 10000
        cursor.execute("SELECT COUNT(*) FROM daily_quote_ingest_audit WHERE action='rollback'")
        assert cursor.fetchone()[0] == rollback_audits_before_failure
        cursor.execute("DROP TRIGGER fail_test_rollback_audit")

    resumed_rollback = HistoricalRepairRepo.restore_shard(
        "000001", "2020-01-01", "2020-01-04", republished["before_rows"],
        republished["after_fingerprint"], provenance,
    )
    resumed = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    assert resumed_rollback["restored_fingerprint"] == before_rollback["fingerprint"]
    assert resumed["fingerprint"] == before_rollback["fingerprint"]


@pytest.mark.integration
@pytest.mark.p0
def test_full_stock_reanchor_rollback_recomputes_restored_anchor_fingerprint(isolated_mysql_schema):
    from core.data.repos.historical_repair import HistoricalRepairRepo

    admin, database_name = isolated_mysql_schema
    initial = HistoricalRepairRepo.get_full_stock_inputs("000001")
    backup = HistoricalRepairRepo.get_full_stock_backup("000001")
    frame = pd.DataFrame([{
        "date": "2020-01-02", "open": 12.5, "high": 13.75, "low": 11.25, "close": 13.125,
        "volume": 1000, "amount": 10000, "amplitude": 1, "pct_chg": 0.5,
        "turnover": 0.2, "adj_factor": 1.0,
    }])
    provenance = {
        "batch_id": "full-stock-reanchor-rollback", "source": "tushare+baostock",
        "provider_version": "isolated-mysql-test", "adjustment_mode": "qfq",
        "anchor_date": "2020-01-02", "input_payload_sha256": "f" * 64,
        "task_id": "da05-full-stock-reanchor",
    }
    published = HistoricalRepairRepo.publish_full_reanchor(
        "000001", frame, initial["fingerprint"], 1.25, 1.0, provenance,
    )
    assert published["after_fingerprint"] != initial["fingerprint"]

    rolled_back = HistoricalRepairRepo.restore_shard(
        "000001", "2020-01-02", "2020-01-02", backup,
        published["after_fingerprint"], provenance,
    )
    restored = HistoricalRepairRepo.get_full_stock_inputs("000001")

    assert restored["fingerprint"] == initial["fingerprint"]
    assert rolled_back["restored_fingerprint"] == initial["fingerprint"]
    assert rolled_back["rows"] == len(backup)


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_advisory_lock_is_shared_between_mysql_sessions(isolated_mysql_schema):
    import threading

    from backend.service.historical_repair import _historical_release_advisory_lock

    attempted = threading.Event()
    outcomes = []

    def acquire_from_second_session():
        attempted.set()
        try:
            with _historical_release_advisory_lock("same-release", timeout=0):
                outcomes.append("unexpectedly_acquired")
        except RuntimeError as error:
            outcomes.append(str(error))

    with _historical_release_advisory_lock("same-release"):
        thread = threading.Thread(target=acquire_from_second_session)
        thread.start()
        assert attempted.wait(timeout=2)
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert outcomes == ["无法获取历史发布互斥锁，拒绝并发写入"]
    with _historical_release_advisory_lock("same-release", timeout=0):
        assert True
    with pytest.raises(ValueError, match="injected operation failure"):
        with _historical_release_advisory_lock("failed-operation"):
            raise ValueError("injected operation failure")
    with _historical_release_advisory_lock("failed-operation", timeout=0):
        assert True


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_advisory_lock_blocks_a_separate_process(isolated_mysql_schema):
    from backend.service.historical_repair import _historical_release_advisory_lock

    _admin, database_name = isolated_mysql_schema
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_process_try_release_lock, args=(database_name, "cross-process", output))
    with _historical_release_advisory_lock("cross-process"):
        process.start()
        process.join(timeout=10)
        assert not process.is_alive()
    assert process.exitcode == 0
    assert output.get(timeout=2) == "blocked"
    with _historical_release_advisory_lock("cross-process", timeout=0):
        assert True
    output.close()


@pytest.mark.integration
@pytest.mark.p0
def test_two_backend_processes_cannot_publish_same_release_concurrently(isolated_mysql_schema, tmp_path):
    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    first_entered = context.Event()
    second_entered = context.Event()
    args = (
        database_name, fixture["qlib_root"], fixture["release_id"],
        fixture["release_manifest_path"], fixture["batch_manifest_path"],
    )
    first = context.Process(
        target=_process_publish_historical_release,
        args=(*args, first_entered, output, True),
    )
    second = context.Process(
        target=_process_publish_historical_release,
        args=(*args, second_entered, output, False),
    )
    first.start()
    assert first_entered.wait(timeout=10)
    second.start()
    assert not second_entered.wait(timeout=0.15)
    first.join(timeout=20)
    second.join(timeout=20)

    assert not first.is_alive()
    assert not second.is_alive()
    results = [output.get(timeout=2), output.get(timeout=2)]
    assert ("published", "published") in results
    assert any(kind == "error" and "DA04-RELEASE-STATE-001" in detail
               for kind, detail in results), results
    assert second_entered.is_set()
    output.close()


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_service_publishes_and_rolls_back_real_mysql_and_qlib(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        published = historical_repair.publish_historical_release(fixture["release_id"])
        published_generation = generation.resolve_qlib_data_dir(fixture["qlib_root"]).name
        published_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        assert published["status"] == "published"
        assert published_generation == published["qlib_generation_published_id"]
        assert published_inputs["rows"][0]["amount"] == 10000
        assert np.isclose(np.fromfile(Path(published["qlib_dir"]) / "features" / "sz000001" / "close.day.bin", dtype="<f")[1], 10.5)

        rolled_back = historical_repair.rollback_historical_release(fixture["release_id"])
        restored_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        restored_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["rollback_state"] == {"qlib": "complete", "database": "complete"}
    assert rolled_back["db_rollback_fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert restored_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert restored_qlib.name == fixture["initial_generation_id"]
    assert np.isnan(np.fromfile(restored_qlib / "features" / "sz000001" / "close.day.bin", dtype="<f")[1])


@pytest.mark.integration
@pytest.mark.p0
def test_rollback_manifest_write_failure_has_no_side_effect_and_can_retry(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    save_json_atomic = historical_repair.save_json_atomic
    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        published = historical_repair.publish_historical_release(fixture["release_id"])
        save_attempts = {"count": 0}

        def fail_rollback_transition_once(manifest, path):
            if manifest.get("status") == "rolling_back" and save_attempts["count"] == 0:
                save_attempts["count"] += 1
                raise OSError("injected rollback journal write failure")
            return save_json_atomic(manifest, path)

        with patch.object(historical_repair, "save_json_atomic", side_effect=fail_rollback_transition_once):
            with pytest.raises(OSError, match="journal write failure"):
                historical_repair.rollback_historical_release(fixture["release_id"])

        current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        persisted = historical_repair.load_json(fixture["release_manifest_path"])
        assert persisted["status"] == "published"
        assert current_inputs["fingerprint"] == published["db_after_fingerprint"]
        assert current_qlib.name == published["qlib_generation_published_id"]

        retried = historical_repair.rollback_historical_release(fixture["release_id"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")

    assert retried["status"] == "rolled_back"
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]


@pytest.mark.integration
@pytest.mark.p0
def test_final_rolled_back_manifest_write_failure_does_not_repeat_database_rollback(
    isolated_mysql_schema, tmp_path,
):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.connection import Database
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    save_json_atomic = historical_repair.save_json_atomic
    failed = {"value": False}

    def fail_final_rollback_manifest_once(manifest, path):
        if manifest.get("status") == "rolled_back" and not failed["value"]:
            failed["value"] = True
            raise OSError("injected final rollback manifest write failure")
        return save_json_atomic(manifest, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        historical_repair.publish_historical_release(fixture["release_id"])
        with patch.object(historical_repair, "save_json_atomic", side_effect=fail_final_rollback_manifest_once):
            with pytest.raises(RuntimeError, match="可重试恢复"):
                historical_repair.rollback_historical_release(fixture["release_id"])

        interrupted = historical_repair.load_json(fixture["release_manifest_path"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        qlib_parent = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        with Database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS count FROM daily_quote_ingest_audit WHERE batch_id=%s AND action='rollback'",
                    (fixture["release_id"],),
                )
                audit_count_before_retry = cursor.fetchone()["count"]

        assert failed["value"] is True
        assert interrupted["status"] == "rollback_required"
        assert interrupted["rollback_state"] == {"qlib": "complete", "database": "complete"}
        assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
        assert qlib_parent.name == fixture["initial_generation_id"]
        assert audit_count_before_retry == 1

        retried = historical_repair.rollback_historical_release(fixture["release_id"])
        with Database.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS count FROM daily_quote_ingest_audit WHERE batch_id=%s AND action='rollback'",
                    (fixture["release_id"],),
                )
                audit_count_after_retry = cursor.fetchone()["count"]

    assert retried["status"] == "rolled_back"
    assert audit_count_after_retry == audit_count_before_retry


@pytest.mark.integration
@pytest.mark.p0
@pytest.mark.parametrize("failed_component", ["qlib", "database"])
def test_recovery_retries_after_transition_and_error_journal_writes_both_fail(
    isolated_mysql_schema, tmp_path, failed_component,
):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    save_json_atomic = historical_repair.save_json_atomic
    failures = {"transition": False, "error": False}

    def fail_transition_and_error_journals_once(manifest, path):
        if (
            not failures["transition"]
            and manifest.get("status") == "rolling_back"
            and (manifest.get("rollback_state") or {}).get(failed_component) == "complete"
        ):
            failures["transition"] = True
            raise OSError(f"injected {failed_component} transition journal failure")
        if failures["transition"] and not failures["error"] and manifest.get("status") == "rollback_required":
            failures["error"] = True
            raise OSError("injected rollback error journal failure")
        return save_json_atomic(manifest, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        published = historical_repair.publish_historical_release(fixture["release_id"])
        with patch.object(historical_repair, "save_json_atomic", side_effect=fail_transition_and_error_journals_once):
            with pytest.raises(OSError, match="journal failure"):
                historical_repair.rollback_historical_release(fixture["release_id"])

        persisted = historical_repair.load_json(fixture["release_manifest_path"])
        current = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        active_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        assert failures == {"transition": True, "error": True}
        assert persisted["status"] == "rolling_back"
        assert active_qlib.name == fixture["initial_generation_id"]
        if failed_component == "qlib":
            assert persisted["rollback_state"] == {"qlib": "pending", "database": "pending"}
            assert current["fingerprint"] == published["db_after_fingerprint"]
        else:
            assert persisted["rollback_state"] == {"qlib": "complete", "database": "pending"}
            assert current["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]

        recovered = historical_repair.rollback_historical_release(fixture["release_id"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")

    assert recovered["status"] == "rolled_back"
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]


@pytest.mark.integration
@pytest.mark.p0
def test_external_database_write_during_rollback_is_not_overwritten(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.connection import Database
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    restore_shard = HistoricalRepairRepo.restore_shard

    def external_write_then_restore(*args, **kwargs):
        Database.transaction(lambda cursor: cursor.execute(
            "UPDATE daily_quotes SET amount=12345 WHERE code='000001' AND trade_date='2020-01-02'",
        ))
        return restore_shard(*args, **kwargs)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        historical_repair.publish_historical_release(fixture["release_id"])
        with patch.object(HistoricalRepairRepo, "restore_shard", side_effect=external_write_then_restore):
            with pytest.raises(RuntimeError, match="可重试恢复"):
                historical_repair.rollback_historical_release(fixture["release_id"])
        current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        blocked = historical_repair.load_json(fixture["release_manifest_path"])

    assert blocked["status"] == "rollback_required"
    assert blocked["rollback_state"] == {"qlib": "complete", "database": "pending"}
    assert current_inputs["rows"][0]["amount"] == 12345
    assert current_qlib.name == fixture["initial_generation_id"]


@pytest.mark.integration
@pytest.mark.p0
def test_external_qlib_generation_change_blocks_rollback_before_database_write(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        published = historical_repair.publish_historical_release(fixture["release_id"])
        external_stage = generation.stage_qlib_generation(data_root=fixture["qlib_root"])
        external = generation.publish_qlib_generation(
            external_stage,
            metadata={"operation": "external_test_write"},
            expected_generation_id=published["qlib_generation_published_id"],
        )

        with pytest.raises(RuntimeError, match="Qlib当前代已变化"):
            historical_repair.rollback_historical_release(fixture["release_id"])

        current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        unchanged_manifest = historical_repair.load_json(fixture["release_manifest_path"])

    assert current_inputs["fingerprint"] == published["db_after_fingerprint"]
    assert current_qlib.name == external["generation_id"]
    assert unchanged_manifest["status"] == "published"


@pytest.mark.integration
@pytest.mark.p0
@pytest.mark.parametrize("changed_generation", ["candidate", "parent"])
def test_qlib_generation_file_change_blocks_rollback_without_overwrite(
    isolated_mysql_schema, tmp_path, changed_generation,
):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        published = historical_repair.publish_historical_release(fixture["release_id"])
        generation_id = (
            published["qlib_generation_published_id"] if changed_generation == "candidate"
            else fixture["initial_generation_id"]
        )
        target = fixture["qlib_root"] / "generations" / generation_id / "features" / "sz000001" / "close.day.bin"
        target.write_bytes(target.read_bytes() + b"external-change")

        with pytest.raises(RuntimeError, match="已变化，拒绝回滚"):
            historical_repair.rollback_historical_release(fixture["release_id"])

        current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        unchanged_manifest = historical_repair.load_json(fixture["release_manifest_path"])

    assert current_inputs["fingerprint"] == published["db_after_fingerprint"]
    assert current_qlib.name == published["qlib_generation_published_id"]
    assert unchanged_manifest["status"] == "published"


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_service_compensates_after_real_qlib_pointer_swap_failure(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    publish_generation = generation.publish_qlib_generation

    def publish_then_fail(*args, **kwargs):
        publish_generation(*args, **kwargs)
        raise RuntimeError("injected failure after Qlib pointer swap")

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(generation, "publish_qlib_generation", side_effect=publish_then_fail):
        with pytest.raises(RuntimeError, match="rolled_back"):
            historical_repair.publish_historical_release(fixture["release_id"])
        restored_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        restored_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        failed_release = historical_repair.load_json(fixture["release_manifest_path"])

    assert failed_release["status"] == "rolled_back"
    assert failed_release["qlib_generation_published_id"] != fixture["initial_generation_id"]
    assert failed_release["db_rollback_fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert restored_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert restored_qlib.name == fixture["initial_generation_id"]


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_stays_published_when_post_commit_intent_cleanup_fails(
    isolated_mysql_schema, tmp_path,
):
    import json
    from pathlib import Path

    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    intent_paths = []
    unlink = Path.unlink

    def fail_intent_cleanup(path, *args, **kwargs):
        if path.parent == fixture["qlib_root"] / "generations" and path.name.startswith(".stage-copy-intent-"):
            intent_paths.append(path)
            raise OSError("simulated post-commit intent cleanup failure")
        return unlink(path, *args, **kwargs)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(Path, "unlink", fail_intent_cleanup):
        published = historical_repair.publish_historical_release(fixture["release_id"])

    current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
    release = historical_repair.load_json(fixture["release_manifest_path"])
    intent_path = intent_paths[0]
    current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    pointer = json.loads((fixture["qlib_root"] / "current.json").read_text(encoding="utf-8"))

    assert published["status"] == "published"
    assert release["status"] == "published"
    assert current_qlib.name == release["qlib_generation_published_id"]
    assert pointer["generation_id"] == current_qlib.name
    assert current_inputs["fingerprint"] == release["db_after_fingerprint"]
    assert intent_path.is_file()

    recovered_stage = generation.stage_qlib_generation(
        data_root=fixture["qlib_root"], generation_id="f" * 32,
    )

    assert current_qlib.is_dir()
    assert not intent_path.exists()
    assert generation.resolve_qlib_data_dir(fixture["qlib_root"]).name == current_qlib.name
    generation.discard_qlib_generation_stage(recovered_stage)


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_compensates_pointer_persistence_failure_after_generation_rename(
    isolated_mysql_schema, tmp_path,
):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    save_json_atomic = generation.save_json_atomic

    def fail_pointer_write(pointer, path):
        if Path(path).resolve() == (fixture["qlib_root"] / "current.json").resolve():
            raise OSError("simulated current pointer persistence failure")
        return save_json_atomic(pointer, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(generation, "save_json_atomic", side_effect=fail_pointer_write):
        with pytest.raises(RuntimeError, match="rolled_back"):
            historical_repair.publish_historical_release(fixture["release_id"])
        failed_release = historical_repair.load_json(fixture["release_manifest_path"])
        restored_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    candidate_id = failed_release["qlib_generation_stage_id"]
    assert failed_release["status"] == "rolled_back"
    assert failed_release["qlib_generation_orphan_cleanup"] == "complete"
    assert restored_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert not (fixture["qlib_root"] / "generations" / candidate_id).exists()
    assert not (fixture["qlib_root"] / "generations" / f".staging-{candidate_id}").exists()


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_service_removes_unpublished_stage_after_partial_patch_failure(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.adapters.qlib import historical_patch
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    promote_stock_patch = historical_patch.promote_stock_patch

    def promote_then_fail(files):
        promote_stock_patch(files)
        raise RuntimeError("injected failure after staged Qlib file replacement")

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(historical_patch, "promote_stock_patch", side_effect=promote_then_fail):
        with pytest.raises(RuntimeError, match="rolled_back"):
            historical_repair.publish_historical_release(fixture["release_id"])
        restored_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        restored_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        leftover_stages = list((fixture["qlib_root"] / "generations").glob(".staging-*"))

    assert restored_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert restored_qlib.name == fixture["initial_generation_id"]
    assert leftover_stages == []


@pytest.mark.integration
@pytest.mark.p0
@pytest.mark.parametrize("replacement_index", [1, 2])
def test_recovery_after_each_multi_file_qlib_replacement_point(
    isolated_mysql_schema, tmp_path, replacement_index,
):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.adapters.qlib import historical_patch
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name, qlib_fields=("close", "volume"))
    promote_stock_patch = historical_patch.promote_stock_patch

    def fail_at_replacement_point(files):
        for index, (field, record) in enumerate(files.items(), 1):
            promote_stock_patch({field: record})
            if index == replacement_index:
                raise SystemExit(f"simulated stop after Qlib file replacement {index}")

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(historical_patch, "promote_stock_patch", side_effect=fail_at_replacement_point):
        with pytest.raises(SystemExit, match=f"replacement {replacement_index}"):
            historical_repair.publish_historical_release(fixture["release_id"])

        interrupted = historical_repair.load_json(fixture["release_manifest_path"])
        staged_id = interrupted["qlib_generation_stage_id"]
        staged_dir = fixture["qlib_root"] / "generations" / f".staging-{staged_id}"
        assert interrupted["status"] == "database_published"
        assert staged_dir.is_dir()
        assert generation.resolve_qlib_data_dir(fixture["qlib_root"]).name == fixture["initial_generation_id"]

        recovered = historical_repair.rollback_historical_release(fixture["release_id"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert recovered["status"] == "rolled_back"
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert not staged_dir.exists()


@pytest.mark.integration
@pytest.mark.p0
def test_recovery_after_crash_during_qlib_generation_staging(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    def copy_partial_then_exit(source, destination, *args, **kwargs):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "partial-copy-marker").write_text("partial", encoding="utf-8")
        raise SystemExit("simulated process stop during Qlib generation staging")

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(generation.shutil, "copytree", side_effect=copy_partial_then_exit):
        with pytest.raises(SystemExit, match="during Qlib generation staging"):
            historical_repair.publish_historical_release(fixture["release_id"])

        interrupted = historical_repair.load_json(fixture["release_manifest_path"])
        stage_id = interrupted["qlib_generation_stage_id"]
        staged_dir = fixture["qlib_root"] / "generations" / f".staging-{stage_id}"
        copying_dir = fixture["qlib_root"] / "generations" / f".copying-{stage_id}"
        copy_intent = fixture["qlib_root"] / "generations" / f".stage-copy-intent-{stage_id}.json"
        current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        assert interrupted["status"] == "staging"
        assert interrupted["qlib_generation_stage_state"] == "copying"
        assert Path(interrupted["qlib_generation_stage_dir"]) == staged_dir.resolve()
        assert not staged_dir.exists()
        assert (copying_dir / "partial-copy-marker").read_text(encoding="utf-8") == "partial"
        assert copy_intent.is_file()
        assert current_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
        assert current_qlib.name == fixture["initial_generation_id"]
        recovery_snapshot = historical_repair.inspect_historical_release_recovery(fixture["release_id"])
        assert recovery_snapshot["database_fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
        assert recovery_snapshot["qlib_state"]["current_generation_id"] == fixture["initial_generation_id"]

        recovered = historical_repair.rollback_historical_release(fixture["release_id"])
        final_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        final_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert recovered["status"] == "rolled_back"
    assert final_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert final_qlib.name == fixture["initial_generation_id"]
    assert not staged_dir.exists()
    assert not copying_dir.exists()
    assert not copy_intent.exists()


@pytest.mark.integration
@pytest.mark.p0
def test_staging_recovery_requires_admin_when_database_changed_externally(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)

    def copy_partial_then_exit(source, destination, *args, **kwargs):
        Path(destination).mkdir(parents=True)
        raise SystemExit("simulated process stop during Qlib generation staging")

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(generation.shutil, "copytree", side_effect=copy_partial_then_exit):
        with pytest.raises(SystemExit, match="during Qlib generation staging"):
            historical_repair.publish_historical_release(fixture["release_id"])

        with admin.cursor() as cursor:
            cursor.execute("UPDATE daily_quotes SET close = 10.75 WHERE code = '000001' AND trade_date = '2020-01-02'")

        with pytest.raises(RuntimeError, match="缺少已提交后指纹"):
            historical_repair.rollback_historical_release(fixture["release_id"])
        recovery = historical_repair.inspect_historical_release_recovery(fixture["release_id"])
        assert recovery["status"] == "recovery_required"
        assert recovery["manual_restore_required"] is True

        recovered = historical_repair.authorize_historical_release_recovery_rollback(
            fixture["release_id"], recovery["database_fingerprint"], recovery["qlib_state_fingerprint"],
            "isolated-test-admin", "核对隔离库外部变化后授权恢复发布前快照", "test-evidence:DA04-staging-external-db-change",
        )
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        final_manifest = historical_repair.load_json(fixture["release_manifest_path"])

    assert recovered["status"] == "rolled_back"
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert final_manifest["manual_recovery_decisions"][0]["evidence_ref"] == "test-evidence:DA04-staging-external-db-change"
    assert list((fixture["qlib_root"] / "generations").glob(".staging-*")) == []


@pytest.mark.integration
@pytest.mark.p0
def test_recovery_after_stage_copy_before_ready_manifest_persist(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    save_json_atomic = historical_repair.save_json_atomic

    def fail_ready_stage_manifest_once(manifest, path):
        if manifest.get("status") == "staging" and manifest.get("qlib_generation_stage_state") == "ready":
            raise SystemExit("simulated process stop before ready-stage manifest persistence")
        return save_json_atomic(manifest, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(historical_repair, "save_json_atomic", side_effect=fail_ready_stage_manifest_once):
        with pytest.raises(SystemExit, match="before ready-stage manifest"):
            historical_repair.publish_historical_release(fixture["release_id"])

    interrupted = historical_repair.load_json(fixture["release_manifest_path"])
    staged_dir = fixture["qlib_root"] / "generations" / f".staging-{interrupted['qlib_generation_stage_id']}"
    assert interrupted["status"] == "staging"
    assert interrupted["qlib_generation_stage_state"] == "copying"
    assert (staged_dir / "calendars" / "day.txt").is_file()
    assert (staged_dir / "instruments" / "all.txt").is_file()
    assert (staged_dir / "features").is_dir()

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        recovered = historical_repair.rollback_historical_release(fixture["release_id"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert recovered["status"] == "rolled_back"
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert not staged_dir.exists()


@pytest.mark.integration
@pytest.mark.p0
def test_qlib_generation_change_before_stage_copy_is_not_reported_as_rolled_back(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    stage_qlib_generation = generation.stage_qlib_generation
    publish_qlib_generation = generation.publish_qlib_generation
    concurrent_generation = {}

    def switch_parent_then_stage(*args, **kwargs):
        other_stage = stage_qlib_generation(data_root=fixture["qlib_root"])
        concurrent_generation["id"] = publish_qlib_generation(
            other_stage, expected_generation_id=fixture["initial_generation_id"],
        )["generation_id"]
        return stage_qlib_generation(*args, **kwargs)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(generation, "stage_qlib_generation", side_effect=switch_parent_then_stage):
        with pytest.raises(RuntimeError, match="rollback_required"):
            historical_repair.publish_historical_release(fixture["release_id"])

        failed = historical_repair.load_json(fixture["release_manifest_path"])
        current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert failed["status"] == "rollback_required"
    assert "暂存开始前已变化" in failed["rollback_errors"][0]
    assert failed["qlib_generation_stage_id"] != concurrent_generation["id"]
    assert current_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == concurrent_generation["id"]
    assert current_qlib.name != fixture["initial_generation_id"]
    assert not (fixture["qlib_root"] / "generations" / f".staging-{failed['qlib_generation_stage_id']}").exists()


@pytest.mark.integration
@pytest.mark.p0
def test_recovery_removes_unpublished_generation_renamed_before_pointer_swap(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    save_json_atomic = generation.save_json_atomic

    def fail_pointer_write(pointer, path):
        if Path(path).resolve() == (fixture["qlib_root"] / "current.json").resolve():
            raise SystemExit("simulated process stop after candidate generation rename")
        return save_json_atomic(pointer, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(generation, "save_json_atomic", side_effect=fail_pointer_write):
        with pytest.raises(SystemExit, match="after candidate generation rename"):
            historical_repair.publish_historical_release(fixture["release_id"])

    interrupted = historical_repair.load_json(fixture["release_manifest_path"])
    candidate_dir = fixture["qlib_root"] / "generations" / interrupted["qlib_generation_stage_id"]
    stage_dir = fixture["qlib_root"] / "generations" / f".staging-{interrupted['qlib_generation_stage_id']}"
    assert interrupted["status"] == "database_published"
    assert interrupted.get("qlib_generation_published_id") is None
    assert candidate_dir.is_dir()
    assert not stage_dir.exists()
    assert generation.resolve_qlib_data_dir(fixture["qlib_root"]).name == fixture["initial_generation_id"]

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        recovered = historical_repair.rollback_historical_release(fixture["release_id"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert recovered["status"] == "rolled_back"
    assert recovered["qlib_generation_orphan_cleanup"] == "complete"
    assert recovered["qlib_generation_stage_cleanup"] == "complete"
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert not candidate_dir.exists()


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_recovery_after_os_process_crash(isolated_mysql_schema, tmp_path):
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared.persistence import load_json

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    crash_process = context.Process(
        target=_process_crash_during_historical_release_publish,
        args=(
            database_name,
            str(fixture["qlib_root"]),
            fixture["release_id"],
            str(fixture["release_manifest_path"]),
            str(fixture["batch_manifest_path"]),
            output,
        ),
    )
    crash_process.start()
    crash_process.join(30)
    if crash_process.is_alive():
        crash_process.terminate()
        crash_process.join(5)
        pytest.fail("publish process did not reach the injected OS crash point")
    assert crash_process.exitcode == 86

    interrupted = load_json(fixture["release_manifest_path"])
    candidate_dir = fixture["qlib_root"] / "generations" / interrupted["qlib_generation_stage_id"]
    active_marker = fixture["qlib_root"].parent / "historical_repair" / "active_releases" / f"{fixture['release_id']}.json"
    assert interrupted["status"] == "database_published"
    assert candidate_dir.is_dir()
    assert active_marker.is_file()
    assert generation.resolve_qlib_data_dir(fixture["qlib_root"]).name == fixture["initial_generation_id"]

    recovery_process = context.Process(
        target=_process_recover_historical_release,
        args=(
            database_name,
            str(fixture["qlib_root"]),
            fixture["release_id"],
            str(fixture["release_manifest_path"]),
            str(fixture["batch_manifest_path"]),
            output,
        ),
    )
    recovery_process.start()
    recovery_process.join(30)
    if recovery_process.is_alive():
        recovery_process.terminate()
        recovery_process.join(5)
        pytest.fail("recovery process did not finish after publisher process exit")
    assert recovery_process.exitcode == 0
    assert output.get(timeout=5) == ("recovered", "rolled_back")

    recovered = load_json(fixture["release_manifest_path"])
    restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    assert recovered["qlib_generation_orphan_cleanup"] == "complete"
    assert recovered["qlib_generation_stage_cleanup"] == "complete"
    assert not active_marker.exists()
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert generation.resolve_qlib_data_dir(fixture["qlib_root"]).name == fixture["initial_generation_id"]
    assert not candidate_dir.exists()


@pytest.mark.integration
@pytest.mark.p0
def test_recovery_preserves_unpublished_generation_when_candidate_file_changed(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    save_json_atomic = generation.save_json_atomic

    def fail_pointer_write(pointer, path):
        if Path(path).resolve() == (fixture["qlib_root"] / "current.json").resolve():
            raise SystemExit("simulated process stop after candidate generation rename")
        return save_json_atomic(pointer, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(generation, "save_json_atomic", side_effect=fail_pointer_write):
        with pytest.raises(SystemExit, match="after candidate generation rename"):
            historical_repair.publish_historical_release(fixture["release_id"])

    interrupted = historical_repair.load_json(fixture["release_manifest_path"])
    candidate_dir = fixture["qlib_root"] / "generations" / interrupted["qlib_generation_stage_id"]
    changed_file = candidate_dir / "features" / "sz000001" / "close.day.bin"
    changed_file.write_bytes(changed_file.read_bytes() + b"external-change")

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        with pytest.raises(RuntimeError, match="可重试恢复"):
            historical_repair.rollback_historical_release(fixture["release_id"])
        failed = historical_repair.load_json(fixture["release_manifest_path"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert failed["status"] == "rollback_required"
    assert failed["rollback_state"] == {"qlib": "complete", "database": "complete"}
    assert "generation" in failed["rollback_error"]
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert changed_file.read_bytes().endswith(b"external-change")
    assert candidate_dir.is_dir()


@pytest.mark.integration
@pytest.mark.p0
def test_rollback_final_state_recheck_detects_database_write_after_restore(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        published = historical_repair.publish_historical_release(fixture["release_id"])
    restore_shard = HistoricalRepairRepo.restore_shard

    def restore_then_external_write(*args, **kwargs):
        result = restore_shard(*args, **kwargs)
        with admin.cursor() as cursor:
            cursor.execute("UPDATE daily_quotes SET close = 10.85 WHERE code = '000001' AND trade_date = '2020-01-02'")
        return result

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(HistoricalRepairRepo, "restore_shard", side_effect=restore_then_external_write):
        with pytest.raises(RuntimeError, match="可重试恢复"):
            historical_repair.rollback_historical_release(fixture["release_id"])
        failed = historical_repair.load_json(fixture["release_manifest_path"])
        current = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert published["status"] == "published"
    assert failed["status"] == "rollback_required"
    assert failed["rollback_state"] == {"qlib": "complete", "database": "complete"}
    assert "终态核验" in failed["rollback_error"]
    assert current["fingerprint"] != fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]


@pytest.mark.integration
@pytest.mark.p0
def test_rollback_final_state_recheck_detects_qlib_write_after_parent_restore(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        historical_repair.publish_historical_release(fixture["release_id"])

    save_json_atomic = historical_repair.save_json_atomic
    injected = {"value": False}
    changed_file = (
        fixture["qlib_root"] / "generations" / fixture["initial_generation_id"]
        / "features" / "sz000001" / "close.day.bin"
    )

    def persist_then_change_parent(manifest, path):
        result = save_json_atomic(manifest, path)
        if (
            not injected["value"]
            and manifest.get("status") == "rolling_back"
            and manifest.get("rollback_state") == {"qlib": "complete", "database": "pending"}
        ):
            changed_file.write_bytes(changed_file.read_bytes() + b"external-change-after-parent-restore")
            injected["value"] = True
        return result

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(historical_repair, "save_json_atomic", side_effect=persist_then_change_parent):
        with pytest.raises(RuntimeError, match="可重试恢复"):
            historical_repair.rollback_historical_release(fixture["release_id"])
        failed = historical_repair.load_json(fixture["release_manifest_path"])
        restored = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert injected["value"] is True
    assert failed["status"] == "rollback_required"
    assert failed["rollback_state"] == {"qlib": "complete", "database": "complete"}
    assert "generation" in failed["rollback_error"]
    assert restored["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert changed_file.read_bytes().endswith(b"external-change-after-parent-restore")


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_recovery_after_database_persisted_before_qlib_swap(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.adapters.qlib import historical_patch
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(historical_patch, "promote_stock_patch", side_effect=SystemExit("simulated process stop")):
        with pytest.raises(SystemExit, match="simulated process stop"):
            historical_repair.publish_historical_release(fixture["release_id"])

    interrupted = historical_repair.load_json(fixture["release_manifest_path"])
    current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    assert interrupted["status"] == "database_published"
    assert current_inputs["fingerprint"] == interrupted["db_after_fingerprint"]

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        recovered = historical_repair.rollback_historical_release(fixture["release_id"])
        final_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert recovered["status"] == "rolled_back"
    assert final_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert list((fixture["qlib_root"] / "generations").glob(".staging-*")) == []


@pytest.mark.integration
@pytest.mark.p0
def test_historical_release_recovery_after_qlib_swap_before_final_manifest_write(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, database_name)
    save_json_atomic = historical_repair.save_json_atomic
    save_count = {"value": 0}

    def fail_before_persisting_final_state(manifest, path):
        save_count["value"] += 1
        if save_count["value"] == 5:
            raise SystemExit("simulated process stop after Qlib swap")
        return save_json_atomic(manifest, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(historical_repair, "save_json_atomic", side_effect=fail_before_persisting_final_state):
        with pytest.raises(SystemExit, match="simulated process stop after Qlib swap"):
            historical_repair.publish_historical_release(fixture["release_id"])

    interrupted = historical_repair.load_json(fixture["release_manifest_path"])
    active_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
    assert interrupted["status"] == "database_published"
    assert interrupted.get("qlib_generation_published_id") is None
    assert active_qlib.name == interrupted["qlib_generation_stage_id"]

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        recovered = historical_repair.rollback_historical_release(fixture["release_id"])
        final_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])

    assert recovered["status"] == "rolled_back"
    assert recovered["qlib_generation_published_id"] == interrupted["qlib_generation_stage_id"]
    assert final_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]


@pytest.mark.integration
@pytest.mark.p0
def test_manual_recovery_audit_restores_ambiguous_commit_in_isolated_mysql_and_qlib(isolated_mysql_schema, tmp_path):
    from backend.service import historical_repair
    from core.adapters.qlib import generation
    from core.data.repos.historical_repair import HistoricalRepairRepo
    from core.shared import config

    _admin, _database_name = isolated_mysql_schema
    fixture = _service_release_fixture(tmp_path, _database_name)
    save_json_atomic = historical_repair.save_json_atomic
    save_count = {"value": 0}

    def fail_before_after_fingerprint_persist(manifest, path):
        save_count["value"] += 1
        if save_count["value"] == 4:
            raise SystemExit("simulated process stop after DB commit before after-fingerprint manifest")
        return save_json_atomic(manifest, path)

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])), \
         patch.object(historical_repair, "save_json_atomic", side_effect=fail_before_after_fingerprint_persist):
        with pytest.raises(SystemExit, match="after DB commit"):
            historical_repair.publish_historical_release(fixture["release_id"])

    interrupted = historical_repair.load_json(fixture["release_manifest_path"])
    assert interrupted["status"] == "publishing"
    assert interrupted.get("db_after_fingerprint") is None
    current_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
    assert current_inputs["fingerprint"] != fixture["release_manifest"]["preimage_fingerprint"]

    with patch.object(historical_repair, "_release_manifest_path", return_value=fixture["release_manifest_path"]), \
         patch.object(historical_repair, "_manifest_path", return_value=fixture["batch_manifest_path"]), \
         patch.object(config, "QLIB_DATA_ROOT", str(fixture["qlib_root"])):
        with pytest.raises(RuntimeError, match="缺少已提交后指纹"):
            historical_repair.rollback_historical_release(fixture["release_id"])
        recovery = historical_repair.inspect_historical_release_recovery(fixture["release_id"])
        assert recovery["manual_restore_required"] is True
        rolled_back = historical_repair.authorize_historical_release_recovery_rollback(
            fixture["release_id"], recovery["database_fingerprint"], recovery["qlib_state_fingerprint"],
            "isolated-test-admin", "核对隔离库行和Qlib父代后恢复", "test-evidence:DA04-ambiguous-commit",
        )
        restored_inputs = HistoricalRepairRepo.get_shard_inputs("000001", "2020-01-01", "2020-01-04")
        current_qlib = generation.resolve_qlib_data_dir(fixture["qlib_root"])
        final_manifest = historical_repair.load_json(fixture["release_manifest_path"])

    assert rolled_back["status"] == "rolled_back"
    assert restored_inputs["fingerprint"] == fixture["release_manifest"]["preimage_fingerprint"]
    assert current_qlib.name == fixture["initial_generation_id"]
    assert final_manifest["manual_recovery_decisions"][0]["actor"] == "isolated-test-admin"
    assert final_manifest["manual_recovery_decisions"][0]["evidence_ref"] == "test-evidence:DA04-ambiguous-commit"
    assert list((fixture["qlib_root"] / "generations").glob(".staging-*")) == []
