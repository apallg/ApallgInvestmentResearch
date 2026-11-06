import pytest

from core.shared.infrastructure.smoke_test import SmokeTestRunner


def test_qlib_smoke_checks_run_under_shared_read_lease(monkeypatch):
    runner = SmokeTestRunner()
    active = []

    def run_with_lease(check):
        active.append(True)
        try:
            return check()
        finally:
            active.pop()

    monkeypatch.setattr("backend.service.qlib_access.run_with_qlib_read_access", run_with_lease)
    monkeypatch.setattr(runner, "_check_qlib_model", lambda: "model" if active else pytest.fail("Qlib model check ran without read lease"))
    monkeypatch.setattr(runner, "_check_snapshot", lambda: "snapshot" if active else pytest.fail("snapshot check ran without read lease"))

    checks = {name: fn for name, fn, _ in runner.checks}

    assert checks["Qlib模型"]() == "model"
    assert checks["选股快照"]() == "snapshot"

