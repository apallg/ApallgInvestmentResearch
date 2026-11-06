"""
DataCollectionScheduler 优化后的边界/异常路径测试
覆盖: _should_collect, Qlib后台同步, batch_size变更
"""
import datetime
import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ──────────────────────────────────────────────
# _should_collect 边界测试
# ──────────────────────────────────────────────

class TestShouldCollect:
    """DataCollectionScheduler._should_collect 的边界用例"""

    @pytest.fixture
    def scheduler(self):
        from core.shared.infrastructure.jobs import DataCollectionScheduler
        s = DataCollectionScheduler()
        return s

    # ── 正常路径 ──

    def test_table_not_exist_returns_true(self, scheduler):
        """表不存在 → SQL报错 → catch → 返回 True (需要采集)"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.side_effect = Exception("Table doesn't exist")
            assert scheduler._should_collect("nonexistent_table", 7) is True

    def test_table_empty_returns_true(self, scheduler):
        """表为空 (MAX返回NULL) → row['last_dt'] 为 None → 返回 True"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": None}
            assert scheduler._should_collect("empty_table", 7) is True

    def test_row_is_none_returns_true(self, scheduler):
        """fetchone 返回 None → 返回 True"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = None
            assert scheduler._should_collect("some_table", 7) is True

    def test_recent_collection_skips(self, scheduler):
        """1天前采集过, min_days=7 → 跳过(False)"""
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": yesterday}
            assert scheduler._should_collect("fundamental_data", 7) is False

    def test_old_collection_proceeds(self, scheduler):
        """10天前采集过, min_days=7 → 需要采集(True)"""
        old_date = datetime.date.today() - datetime.timedelta(days=10)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": old_date}
            assert scheduler._should_collect("fundamental_data", 7) is True

    def test_exactly_min_days_proceeds(self, scheduler):
        """恰好 min_days 天前 (delta==min_days, 不是 < min_days) → 需要采集"""
        exact_date = datetime.date.today() - datetime.timedelta(days=7)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": exact_date}
            assert scheduler._should_collect("fundamental_data", 7) is True

    def test_exactly_min_days_minus_one_skips(self, scheduler):
        """min_days-1 天前 → < min_days → 跳过"""
        date_6d_ago = datetime.date.today() - datetime.timedelta(days=6)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": date_6d_ago}
            assert scheduler._should_collect("fundamental_data", 7) is False

    # ── 日期类型兼容 ──

    def test_datetime_object_converts_to_date(self, scheduler):
        """datetime.datetime 对象 → .date() 转换 → 正确比较"""
        yesterday_dt = datetime.datetime.now() - datetime.timedelta(days=1)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": yesterday_dt}
            assert scheduler._should_collect("fundamental_data", 7) is False

    def test_date_object_direct_use(self, scheduler):
        """datetime.date 对象 → 直接使用"""
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": yesterday}
            assert scheduler._should_collect("fundamental_data", 7) is False

    # ── 异常路径 ──

    def test_db_connection_error_returns_true(self, scheduler):
        """DB 连接异常 → fail-open (返回 True, 宁多采不少采)"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.side_effect = OSError("Connection refused")
            assert scheduler._should_collect("fundamental_data", 7) is True

    def test_import_error_returns_true(self, scheduler):
        """Database import 失败 → 返回 True"""
        with patch("core.data.Database", side_effect=ImportError("No module")):
            assert scheduler._should_collect("fundamental_data", 7) is True

    def test_unexpected_date_type_returns_true(self, scheduler):
        """意外类型 → delta计算失败 → except → 返回 True"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": object()}
            assert scheduler._should_collect("fundamental_data", 7) is True

    # ── date_col 参数 ──

    def test_custom_date_column(self, scheduler):
        """自定义 date_col → SQL 使用指定列名"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": datetime.date.today()}
            scheduler._should_collect("macro_money_supply", 30, date_col="trade_date")
            call_sql = mock_db.fetchone.call_args[0][0]
            assert "MAX(trade_date)" in call_sql

    def test_default_date_column_is_created_at(self, scheduler):
        """默认 date_col → SQL 使用 MAX(created_at)"""
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": None}
            scheduler._should_collect("fundamental_data", 7)
            call_sql = mock_db.fetchone.call_args[0][0]
            assert "MAX(created_at)" in call_sql

    # ── 30天阈值(宏) ──

    def test_macro_monthly_29_days_skips(self, scheduler):
        """29天前采集的宏观月频 → < 30天 → 跳过"""
        date_29d = datetime.date.today() - datetime.timedelta(days=29)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": date_29d}
            assert scheduler._should_collect("macro_cpi", 30) is False

    def test_macro_monthly_31_days_proceeds(self, scheduler):
        """31天前采集的宏观月频 → >= 30天 → 需要采集"""
        date_31d = datetime.date.today() - datetime.timedelta(days=31)
        with patch("core.data.Database") as mock_db:
            mock_db.fetchone.return_value = {"last_dt": date_31d}
            assert scheduler._should_collect("macro_cpi", 30) is True


# ──────────────────────────────────────────────
# Qlib 后台同步测试
# ──────────────────────────────────────────────

def _setup_config_mock(mock_config, **overrides):
    """设置 config mock 的默认值"""
    defaults = {
        "COLLECT_SCHEDULE_TIME": "15:30", "COLLECT_STOCK_POOL": "all",
        "COLLECT_DATA_SOURCE": "tencent", "COLLECT_THREAD_COUNT": 16,
        "COLLECT_DELAY_SECONDS": 0, "COLLECT_LOOKBACK_DAYS": 3,
        "COLLECT_NORTH_FLOW_ENABLED": False, "COLLECT_CAPITAL_FLOW_ENABLED": False,
        "COLLECT_FUNDAMENTAL_ENABLED": False, "COLLECT_FORECAST_ENABLED": False,
        "COLLECT_INCOME_ENABLED": False, "COLLECT_AKSHARE_CONSENSUS_ENABLED": False,
        "COLLECT_DAILY_BASIC_ENABLED": False, "MACRO_COLLECT_ENABLED": False,
        "COLLECT_AUTO_SYNC_QLIB": False, "COLLECT_CONCEPT_SECTORS_ENABLED": False,
        "COLLECT_ETF_FLOW_ENABLED": False, "COLLECT_MIN_DAILY_ROWS": 1,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(mock_config, k, v)


def _submit_immediately(fn, *args, **kwargs):
    future = Future()
    try:
        future.set_result(fn(*args, **kwargs))
    except Exception as exc:
        future.set_exception(exc)
    return future


class TestQlibBackgroundSync:
    """Qlib 同步改为后台执行后的边界用例"""

    def test_saved_zero_still_syncs_when_qlib_calendar_is_behind(self):
        """其他任务已写日K时，本轮 saved=0 仍须补齐落后的 Qlib"""
        with patch("core.shared.infrastructure.jobs.DataCollectionScheduler._should_collect", return_value=False), \
             patch("core.shared.infrastructure.jobs.thread_pool") as mock_pool, \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.data.Database"), \
             patch("core.adapters.qlib.bridge.QlibDataBridge") as mock_bridge:
            _setup_config_mock(mock_config, COLLECT_AUTO_SYNC_QLIB=True)
            mock_bridge.return_value.get_status.return_value = {
                "synced": True, "last_date": "2026-09-21",
            }

            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes") as mock_codes, \
                 patch("scripts.fast_collect.collect_incremental") as mock_collect:
                mock_codes.return_value = ["000001"]
                mock_collect.return_value = 0  # saved=0
                s._do_collection(source="test")
                submitted = [call.args[0].__name__ for call in mock_pool.default.submit.call_args_list]
                assert "_qlib_sync" in submitted

    def test_saved_zero_skips_sync_when_qlib_calendar_is_current(self):
        with patch("core.shared.infrastructure.jobs.DataCollectionScheduler._should_collect", return_value=False), \
             patch("core.shared.infrastructure.jobs.thread_pool") as mock_pool, \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.data.Database"), \
             patch("core.adapters.qlib.bridge.QlibDataBridge") as mock_bridge:
            _setup_config_mock(mock_config, COLLECT_AUTO_SYNC_QLIB=True)
            mock_bridge.return_value.get_status.return_value = {
                "synced": True, "last_date": "2999-12-31",
            }
            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes") as mock_codes, \
                 patch("scripts.fast_collect.collect_incremental") as mock_collect:
                mock_codes.return_value = ["000001"]
                mock_collect.return_value = 0  # saved=0
                s._do_collection(source="test")
                submitted = [call.args[0].__name__ for call in mock_pool.default.submit.call_args_list]
                assert "_qlib_sync" not in submitted

    def test_saved_positive_submits_background_sync(self):
        """K线增量 saved>0 → 提交后台 Qlib 同步"""
        with patch("core.shared.infrastructure.jobs.DataCollectionScheduler._should_collect", return_value=False), \
             patch("core.shared.infrastructure.jobs.thread_pool") as mock_pool, \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.data.Database"):
            _setup_config_mock(mock_config, COLLECT_AUTO_SYNC_QLIB=True)

            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes") as mock_codes, \
                 patch("scripts.fast_collect.collect_incremental") as mock_collect:
                mock_codes.return_value = ["000001"]
                mock_collect.return_value = 5  # saved>0
                s._do_collection(source="test")
                submitted = [call.args[0].__name__ for call in mock_pool.default.submit.call_args_list]
                assert "_qlib_sync" in submitted

    def test_partial_daily_batch_never_advances_qlib_calendar(self):
        with patch("core.shared.infrastructure.jobs.DataCollectionScheduler._should_collect", return_value=False), \
             patch("core.shared.infrastructure.jobs.thread_pool") as mock_pool, \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.data.integrity.check_and_repair", return_value={"overall": True, "checks": []}), \
             patch("core.data.Database") as mock_db:
            _setup_config_mock(
                mock_config,
                COLLECT_AUTO_SYNC_QLIB=True,
                COLLECT_MIN_DAILY_ROWS=4800,
            )
            mock_db.fetchone.return_value = {"cnt": 5}

            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes", return_value=["000001"]), \
                 patch("scripts.fast_collect.collect_incremental", return_value=5):
                s._do_collection(source="test")

            submitted = [call.args[0].__name__ for call in mock_pool.default.submit.call_args_list]
            assert "_qlib_sync" not in submitted

    def test_qlib_sync_exception_in_background(self):
        """后台 Qlib 同步内部异常 → 返回 None → 不影响主流程"""
        from concurrent.futures import Future
        with patch("core.shared.infrastructure.jobs.DataCollectionScheduler._should_collect", return_value=False), \
             patch("core.shared.infrastructure.jobs.thread_pool") as mock_pool, \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.data.Database"):
            _setup_config_mock(mock_config, COLLECT_AUTO_SYNC_QLIB=True)

            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes") as mock_codes, \
                 patch("scripts.fast_collect.collect_incremental") as mock_collect:
                mock_codes.return_value = ["000001"]
                mock_collect.return_value = 5

                ok_future = Future()
                ok_future.set_result(None)
                mock_pool.default.submit.return_value = ok_future

                s._do_collection(source="test")
                assert s.last_result["status"] != "error"

    def test_busy_prevents_concurrent_collection(self):
        """_busy=True → 跳过重复调用"""
        from core.shared.infrastructure.jobs import DataCollectionScheduler
        s = DataCollectionScheduler()
        s._busy = True
        s._do_collection(source="test")
        assert s.last_collection is None


# ──────────────────────────────────────────────
# 集成: _do_collection 中各项开关组合
# ──────────────────────────────────────────────

class TestCollectionFrequencyGating:
    """验证降频采集逻辑: _should_collect 被正确调用"""

    def test_fundamentals_gated_by_should_collect(self):
        """基本面采集受 _should_collect('fundamental_data', 7) 控制"""
        with patch("core.shared.infrastructure.jobs.thread_pool") as mock_pool, \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.domain.screening._scoring_legacy.fetch_and_save_fundamentals", return_value={"saved": 0, "tushare": 0, "akshare": 0}), \
             patch("core.data.collection_gate.get_gate_service", side_effect=RuntimeError("test gate unavailable")), \
             patch("core.data.integrity.check_and_repair", return_value={"all_pass": True, "repaired": []}), \
             patch("core.data.Database"):
            _setup_config_mock(mock_config, COLLECT_FUNDAMENTAL_ENABLED=True)
            mock_pool.default.submit.side_effect = _submit_immediately

            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes") as mock_codes, \
                 patch("scripts.fast_collect.collect_incremental") as mock_collect, \
                 patch.object(s, "_should_collect", return_value=True) as mock_should:
                mock_codes.return_value = ["000001"]
                mock_collect.return_value = 1

                s._do_collection(source="test")

                calls = [c[0] for c in mock_should.call_args_list]
                assert any("fundamental_data" in str(c) for c in calls)

    def test_all_gated_tables_skipped_on_recent_collection(self):
        """所有被降频的表都在 _should_collect 返回 False 时跳过"""
        with patch("core.shared.infrastructure.jobs.thread_pool"), \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.data.Database"):
            _setup_config_mock(mock_config,
                                    COLLECT_FUNDAMENTAL_ENABLED=True,
                                    COLLECT_FORECAST_ENABLED=True,
                                    COLLECT_INCOME_ENABLED=True,
                                    COLLECT_AKSHARE_CONSENSUS_ENABLED=True,
                                    MACRO_COLLECT_ENABLED=True)

            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes") as mock_codes, \
                 patch("scripts.fast_collect.collect_incremental") as mock_collect, \
                 patch.object(s, "_should_collect", return_value=False), \
                 patch("core.domain.screening._scoring_legacy.fetch_and_save_fundamentals") as mock_fund, \
                 patch("scripts.tushare_collect.collect_forecast") as mock_forecast, \
                 patch("scripts.tushare_collect.collect_income") as mock_income, \
                 patch("scripts.collect_macro.collect_money_supply") as mock_m2, \
                 patch("scripts.collect_macro.collect_cpi") as mock_cpi, \
                 patch("scripts.collect_macro.collect_ppi") as mock_ppi, \
                 patch("scripts.collect_macro.collect_pmi") as mock_pmi:
                mock_codes.return_value = ["000001"]
                mock_collect.return_value = 1

                s._do_collection(source="test")

                mock_fund.assert_not_called()
                mock_forecast.assert_not_called()
                mock_income.assert_not_called()
                mock_m2.assert_not_called()
                mock_cpi.assert_not_called()
                mock_ppi.assert_not_called()
                mock_pmi.assert_not_called()

    def test_shibor_runs_daily_and_lpr_is_frequency_gated(self):
        """Shibor 每天采集，LPR 受 35 天频率门控"""
        with patch("core.shared.infrastructure.jobs.thread_pool") as mock_pool, \
             patch("core.shared.infrastructure.jobs.config") as mock_config, \
             patch("core.data.fundamental.preload_cache", return_value=True), \
             patch("core.domain.regime.history.ensure_today"), \
             patch("core.data.collection_gate.get_gate_service", side_effect=RuntimeError("test gate unavailable")), \
             patch("core.data.integrity.check_and_repair", return_value={"all_pass": True, "repaired": []}), \
             patch("core.data.Database"):
            _setup_config_mock(mock_config, MACRO_COLLECT_ENABLED=True)
            mock_pool.default.submit.side_effect = _submit_immediately

            from core.shared.infrastructure.jobs import DataCollectionScheduler
            s = DataCollectionScheduler()
            s._busy = False

            with patch("scripts.fast_collect.get_all_codes") as mock_codes, \
                 patch("scripts.fast_collect.collect_incremental") as mock_collect, \
                 patch.object(s, "_should_collect", return_value=False), \
                 patch("scripts.collect_macro.collect_shibor") as mock_shibor, \
                 patch("scripts.collect_macro.collect_lpr") as mock_lpr:
                mock_codes.return_value = ["000001"]
                mock_collect.return_value = 1

                s._do_collection(source="test")

                mock_shibor.assert_called_once()
                mock_lpr.assert_not_called()


# ──────────────────────────────────────────────
# 已删除的 65s 冷却验证
# ──────────────────────────────────────────────

class TestNoHardCodedCooldown:
    """验证 time.sleep(65) 已从 _do_collection 中移除"""

    def test_no_65_second_sleep_in_do_collection(self):
        """_do_collection 源码中不含 time.sleep(65)"""
        import inspect
        from core.shared.infrastructure.jobs import DataCollectionScheduler
        source = inspect.getsource(DataCollectionScheduler._do_collection)
        assert "time.sleep(65)" not in source
        assert "65s" not in source
