import threading
from unittest.mock import Mock, patch

import pymysql

from core.data.connection import Database, _ConnectionPool


class FakeConnection:
    def __init__(self, execute_error=None, commit_error=None):
        self.execute_error = execute_error
        self.commit_error = commit_error
        self.rollback_calls = 0
        self.commit_calls = 0

    def begin(self):
        return None

    def cursor(self):
        connection = self

        class Cursor:
            lastrowid = 7

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params):
                if connection.execute_error:
                    raise connection.execute_error

        return Cursor()

    def commit(self):
        self.commit_calls += 1
        if self.commit_error:
            raise self.commit_error

    def rollback(self):
        self.rollback_calls += 1


def test_recycle_none_releases_slot_and_decrements_created_count():
    pool = object.__new__(_ConnectionPool)
    pool._borrowed = 1
    pool._created = 1
    pool._lock = threading.Lock()
    pool._sem = threading.Semaphore(0)

    pool.recycle(None)

    assert pool._borrowed == 0
    assert pool._created == 0
    assert pool._sem.acquire(blocking=False)


def test_business_sql_error_does_not_retry_and_reuses_healthy_connection():
    connection = FakeConnection(execute_error=pymysql.err.OperationalError(1644, "business error"))
    recycled = []

    with patch.object(Database, "_borrow", return_value=connection), patch.object(
        Database, "_recycle", side_effect=recycled.append
    ), patch("core.data.connection._is_connection_alive", return_value=True):
        try:
            Database.execute("CALL business_failure()")
        except pymysql.err.OperationalError as exc:
            assert exc.args[0] == 1644
        else:
            raise AssertionError("business SQL error should propagate")

    assert recycled == [connection]
    assert connection.rollback_calls == 0


def test_transaction_business_error_rolls_back_without_retry():
    connection = FakeConnection()
    recycled = []
    operation_calls = []
    connection.rollback = Mock()

    def fail_operation(cursor):
        operation_calls.append(1)
        raise pymysql.err.OperationalError(1644, "business error")

    with patch.object(Database, "_borrow", return_value=connection), patch.object(
        Database, "_recycle", side_effect=recycled.append
    ), patch("core.data.connection._is_connection_alive", return_value=True):
        try:
            Database.transaction(fail_operation)
        except pymysql.err.OperationalError as exc:
            assert exc.args[0] == 1644
        else:
            raise AssertionError("business SQL error should propagate")

    assert operation_calls == [1]
    assert connection.rollback.call_count == 1
    assert recycled == [connection]


def test_ambiguous_commit_failure_does_not_replay_transaction():
    connection = FakeConnection(commit_error=pymysql.err.OperationalError(2013, "lost connection"))
    recycled = []
    operation_calls = []
    destroyed = []

    with patch.object(Database, "_borrow", return_value=connection), patch.object(
        Database, "_recycle", side_effect=recycled.append
    ), patch("core.data.connection._is_connection_alive", return_value=True), patch(
        "core.data.connection._destroy_connection", side_effect=destroyed.append
    ):
        try:
            Database.transaction(lambda cursor: operation_calls.append(1))
        except pymysql.err.OperationalError as exc:
            assert exc.args[0] == 2013
        else:
            raise AssertionError("ambiguous commit error should propagate")

    assert operation_calls == [1]
    assert connection.commit_calls == 1
    assert recycled == [None]
    assert destroyed == [connection]
