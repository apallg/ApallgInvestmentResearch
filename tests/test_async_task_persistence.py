from unittest.mock import Mock, patch

import pytest

from core.shared.infrastructure.async_task import AsyncTaskManager


def test_task_is_not_started_when_status_table_cannot_be_ensured():
    manager = AsyncTaskManager()

    with patch("core.shared.infrastructure.async_task._ensure_table", return_value=False), \
         patch("core.shared.infrastructure.thread_pool.thread_pool.light.submit") as submit:
        with pytest.raises(RuntimeError, match="状态表不可用"):
            manager.submit_with_id("task-table-error", lambda: None)

    submit.assert_not_called()
    assert manager._tasks == {}
    assert manager._submitting == set()


def test_task_is_not_started_when_initial_state_cannot_be_persisted():
    manager = AsyncTaskManager()

    with patch("core.shared.infrastructure.async_task._ensure_table", return_value=True), \
         patch("core.shared.infrastructure.async_task._db_save", return_value=False), \
         patch("core.shared.infrastructure.thread_pool.thread_pool.light.submit") as submit:
        with pytest.raises(RuntimeError, match="初始状态持久化失败"):
            manager.submit_with_id("task-save-error", lambda: None)

    submit.assert_not_called()
    assert manager._tasks == {}
    assert manager._cancel_events == {}
    assert manager._submitting == set()
    assert manager._history == []


def test_thread_pool_submission_failure_is_persisted_as_failed():
    manager = AsyncTaskManager()
    save = Mock(return_value=True)

    with patch("core.shared.infrastructure.async_task._ensure_table", return_value=True), \
         patch("core.shared.infrastructure.async_task._db_save", save), \
         patch("core.shared.infrastructure.thread_pool.thread_pool.light.submit",
               side_effect=RuntimeError("executor stopped")):
        with pytest.raises(RuntimeError, match="executor stopped"):
            manager.submit_with_id("task-submit-error", lambda: None)

    task = manager.get("task-submit-error")
    assert task.status == "failed"
    assert "executor stopped" in task.error
    assert manager.is_active("task-submit-error") is False
    assert save.call_count == 2
