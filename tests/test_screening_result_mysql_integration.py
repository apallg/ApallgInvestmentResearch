import os
import uuid
import asyncio
import multiprocessing
from unittest.mock import Mock

import pymysql
import pytest


def _read_screening_task_in_new_process(database_name, job_id, output):
    from core.shared import config
    import core.data.connection as connection_module
    from core.data.repos.screening_result import ScreeningResultRepo

    config.MYSQL_DATABASE = database_name
    try:
        task = ScreeningResultRepo.get(job_id)
        if task and task.get("status") in ("pending", "running"):
            task = ScreeningResultRepo.interrupt_if_stale(job_id)
        output.put((task.get("status") if task else None, task.get("result_json") if task else None))
    except Exception as error:
        output.put(("error", str(error)))
    finally:
        if connection_module._pool is not None:
            connection_module._pool.shutdown()
            connection_module._pool = None


@pytest.fixture
def isolated_screening_mysql():
    if os.environ.get("SCREENING_RESULT_TEST_MYSQL") != "1":
        pytest.skip("set SCREENING_RESULT_TEST_MYSQL=1 to run isolated MySQL integration")

    from core.shared import config
    import core.data.connection as connection_module
    from core.data.repos.screening_result import ScreeningResultRepo

    original_pool = connection_module._pool
    original_database = config.MYSQL_DATABASE
    database_name = "codex_sc02_" + uuid.uuid4().hex[:12]
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
    test_pool = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        config.MYSQL_DATABASE = database_name
        ScreeningResultRepo._ready = False
        test_pool = connection_module._get_pool()
        yield admin, database_name
    finally:
        if test_pool is not None:
            test_pool.shutdown()
        config.MYSQL_DATABASE = original_database
        connection_module._pool = original_pool
        ScreeningResultRepo._ready = False
        with admin.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_screening_task_persistence_lifecycle_restart_stale_and_rollback(isolated_screening_mysql, monkeypatch):
    from core.data.connection import Database
    from core.data.repos.screening_result import ScreeningResultRepo
    from backend.api import screening
    from backend.service.rebalance import RebalanceService

    _admin, database_name = isolated_screening_mysql
    ScreeningResultRepo.create_pending("job-complete", "stable_history_500d")
    ScreeningResultRepo.mark_running("job-complete", "worker-a")
    ScreeningResultRepo.heartbeat("job-complete", "worker-a", {"stage": "predict", "current": 3, "total": 5})
    result = {
        "job_id": "job-complete", "result_id": "job-complete", "status": "done",
        "candidates": [{"code": "000001"}], "recommended": [],
        "provenance": {"snapshot_fingerprint": "f" * 64},
    }
    ScreeningResultRepo.complete("job-complete", "worker-a", result)
    recovered_after_restart = ScreeningResultRepo.get("job-complete")
    assert recovered_after_restart["status"] == "done"
    assert recovered_after_restart["result_json"] == result
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    child = context.Process(
        target=_read_screening_task_in_new_process,
        args=(database_name, "job-complete", output),
    )
    child.start()
    restarted_status, restarted_result = output.get(timeout=15)
    child.join(timeout=15)
    assert child.exitcode == 0
    assert restarted_status == "done"
    assert restarted_result == result
    monkeypatch.setattr(screening, "_results", {})
    assert asyncio.run(screening.get_result("job-complete"))["data"] == result
    save = Mock(return_value={"save_id": 8})
    monkeypatch.setattr(RebalanceService, "save_screening", save)
    saved = asyncio.run(screening.save_screening_result("job-complete", "snapshot", ""))
    assert saved["data"] == {"save_id": 8}
    assert save.call_args.kwargs["screening_result"] == result
    with pytest.raises(RuntimeError, match="未处于可完成状态"):
        ScreeningResultRepo.complete("job-complete", "worker-a", result)
    assert ScreeningResultRepo.get("job-complete")["result_json"] == result

    ScreeningResultRepo.create_pending("job-stale", "stable_history_500d")
    ScreeningResultRepo.mark_running("job-stale", "worker-old")
    Database.execute(
        "UPDATE screening_task_results SET updated_at=DATE_SUB(NOW(3),INTERVAL 2 MINUTE) "
        "WHERE job_id=%s", ("job-stale",),
    )
    output = context.Queue()
    child = context.Process(
        target=_read_screening_task_in_new_process,
        args=(database_name, "job-stale", output),
    )
    child.start()
    stale_status, _ = output.get(timeout=15)
    child.join(timeout=15)
    assert child.exitcode == 0
    assert stale_status == "interrupted"

    ScreeningResultRepo.create_pending("job-rollback", "stable_history_500d")
    ScreeningResultRepo.mark_running("job-rollback", "worker-b")
    original_transaction = Database.transaction

    def rollback_after_update(operation):
        def fail_after_write(cursor):
            operation(cursor)
            raise RuntimeError("injected finalization failure")
        return original_transaction(fail_after_write)

    monkeypatch.setattr(Database, "transaction", rollback_after_update)
    with pytest.raises(RuntimeError, match="injected finalization failure"):
        ScreeningResultRepo.complete("job-rollback", "worker-b", {
            **result, "job_id": "job-rollback", "result_id": "job-rollback",
        })
    monkeypatch.setattr(Database, "transaction", original_transaction)
    failed_commit = ScreeningResultRepo.get("job-rollback")
    assert failed_commit["status"] == "running"
    assert failed_commit["result_json"] is None
