"""
统一调度器单元测试 — 注册、生命周期、任务控制、状态查询
"""
import pytest
import time
from unittest.mock import patch, MagicMock

from core.shared.infrastructure.scheduler import UnifiedSchedulerManager


@pytest.fixture
def mgr():
    m = UnifiedSchedulerManager()
    yield m
    try:
        m.stop()
    except Exception:
        pass


class TestRegister:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_register_adds_meta(self, mgr):
        ran = []
        mgr.register("test-job", lambda: ran.append(1), "0 9 * * 1-5")
        meta = mgr.get_meta("test-job")
        assert meta is not None
        assert meta["name"] == "test-job"
        assert meta["running"] is False
        assert meta["run_count"] == 0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_duplicate_register_raises(self, mgr):
        mgr.register("dup", lambda: None, "0 0 * * *")
        with pytest.raises(ValueError, match="已注册"):
            mgr.register("dup", lambda: None, "0 0 * * *")

    @pytest.mark.unit
    @pytest.mark.p1
    def test_register_with_extra(self, mgr):
        mgr.register("extra-job", lambda: None, "0 0 * * *", extra={"key": "val"})
        meta = mgr.get_meta("extra-job")
        assert meta["key"] == "val"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_register_trading_day_only(self, mgr):
        mgr.register("td-job", lambda: None, "0 0 * * *", trading_day_only=True)
        meta = mgr.get_meta("td-job")
        assert meta is not None


class TestExecuteJob:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_successful_execution(self, mgr):
        ran = []
        mgr.register("ok", lambda: ran.append(True), "0 9 * * 1-5")
        mgr._execute_job("ok")
        assert len(ran) == 1
        meta = mgr.get_meta("ok")
        assert meta["run_count"] == 1
        assert meta["last_error"] is None
        assert meta["running"] is False

    @pytest.mark.unit
    @pytest.mark.p0
    def test_error_captured_in_meta(self, mgr):
        def boom():
            raise ValueError("测试错误")

        mgr.register("fail", boom, "0 9 * * 1-5")
        mgr._execute_job("fail")
        meta = mgr.get_meta("fail")
        assert meta["run_count"] == 1
        assert "测试错误" in (meta["last_error"] or "")
        assert meta["running"] is False

    @pytest.mark.unit
    @pytest.mark.p1
    def test_trading_day_skip(self, mgr):
        ran = []
        mgr.register("td", lambda: ran.append(True), "0 9 * * 1-5", trading_day_only=True)
        with patch("core.shared.trading_time.is_trading_day", return_value=False):
            mgr._execute_job("td")
        assert len(ran) == 0
        meta = mgr.get_meta("td")
        assert meta["run_count"] == 0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_trading_day_import_error_fallback(self, mgr):
        """is_trading_day 导入失败时不阻塞执行"""
        ran = []
        mgr.register("td2", lambda: ran.append(True), "0 9 * * 1-5", trading_day_only=True)
        with patch("core.shared.trading_time.is_trading_day", side_effect=ImportError):
            mgr._execute_job("td2")
        assert len(ran) == 1

    @pytest.mark.unit
    @pytest.mark.p1
    def test_run_count_increments(self, mgr):
        mgr.register("counter", lambda: None, "0 0 * * *")
        for _ in range(3):
            mgr._execute_job("counter")
        assert mgr.get_meta("counter")["run_count"] == 3


class TestLifecycle:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_start_stop(self, mgr):
        mgr.start()
        assert mgr._started is True
        mgr.stop()
        assert mgr._started is False

    @pytest.mark.unit
    @pytest.mark.p1
    def test_double_start_noop(self, mgr):
        mgr.start()
        mgr.start()
        assert mgr._started is True
        mgr.stop()


class TestTrigger:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_trigger_unknown_raises(self, mgr):
        with pytest.raises(ValueError, match="未知任务"):
            mgr.trigger("nonexistent")

    @pytest.mark.unit
    @pytest.mark.p1
    def test_trigger_skips_when_running(self, mgr):
        mgr.register("long-job", lambda: time.sleep(1), "0 0 * * *")
        mgr._job_meta["long-job"]["running"] = True
        mgr.trigger("long-job")  # should not raise
        assert mgr._job_meta["long-job"]["run_count"] == 0


class TestGetMeta:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_unknown_job_returns_none(self, mgr):
        assert mgr.get_meta("nonexistent") is None

    @pytest.mark.unit
    @pytest.mark.p1
    def test_meta_last_run_formatted(self, mgr):
        mgr.register("fmt-job", lambda: None, "0 0 * * *")
        mgr._execute_job("fmt-job")
        meta = mgr.get_meta("fmt-job")
        assert meta["last_run"] is not None
        assert "T" in meta["last_run"]

    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_all_meta(self, mgr):
        mgr.register("a", lambda: None, "0 0 * * *")
        mgr.register("b", lambda: None, "0 0 * * *")
        all_meta = mgr.get_all_meta()
        assert len(all_meta) == 2
        names = {m["name"] for m in all_meta}
        assert names == {"a", "b"}


class TestStatus:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_status_aggregated(self, mgr):
        mgr.register("s1", lambda: None, "0 0 * * *")
        s = mgr.status()
        assert s["job_count"] == 1
        assert len(s["jobs"]) == 1
        assert s["started"] is False

    @pytest.mark.unit
    @pytest.mark.p1
    def test_status_prefix_filter(self, mgr):
        mgr.register("risk-scan-morning", lambda: None, "0 9 * * 1-5")
        mgr.register("collect-daily", lambda: None, "30 18 * * 1-5")
        s = mgr.status(prefix="risk")
        assert len(s["jobs"]) == 1
        assert s["jobs"][0]["name"] == "risk-scan-morning"


class TestPauseResume:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_pause_resume_unknown_does_not_crash(self, mgr):
        mgr.pause("unknown")
        mgr.resume("unknown")
