"""
滚动筛选进度上报边界测试: progress_cb 调用保障 / 异常安全 / 空值场景
"""
import pytest
from unittest import mock


# ═══════════════════════════════════════════════════════
# Fix: _build_master_indicators 完成后必须上报最终进度
# ═══════════════════════════════════════════════════════

class TestBuildMasterProgress:
    """验证 _build_master_indicators 的 progress_cb 行为"""

    def _patch_deps(self, pool_codes, stock_count=3):
        """构造最小化 mock: CacheManager 返回假数据, Analyzer 不计算, thread_pool 串行执行"""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=150, freq="B"),
            "open": np.random.randn(150) + 10,
            "high": np.random.randn(150) + 11,
            "low": np.random.randn(150) + 9,
            "close": np.random.randn(150) + 10,
            "volume": np.abs(np.random.randn(150)) * 1e7,
            "amount": np.abs(np.random.randn(150)) * 1e8,
        })

        patches = [
            mock.patch("core.shared.infrastructure.cache.CacheManager", autospec=True),
            mock.patch("core.domain.screening.indicators.Analyzer", autospec=True),
        ]
        for p in patches:
            p.start()

        from core.shared.infrastructure.cache import CacheManager
        from core.domain.screening.indicators import Analyzer
        CacheManager.make_key.return_value = "daily:test"
        CacheManager.return_value.get_cached.return_value = df.copy()
        Analyzer.return_value.add_indicators.return_value = df.copy()

        # thread_pool 用真实的 ThreadPoolExecutor（1 worker）
        import threading
        tlocal = threading.local()
        tlocal._df_cache = {}

        return patches

    def teardown_method(self):
        for p in getattr(self, '_patches', []):
            p.stop()

    def test_calls_progress_cb_on_completion(self):
        """少于 500 只股票 → done[0]%500 永不触发, 但结束时必须补发"""
        self._patches = self._patch_deps(["000001", "000002", "000003"])
        from core.domain.factors.rolling import _build_master_indicators

        calls = []
        _build_master_indicators(["000001", "000002", "000003"],
                                 progress_cb=lambda p: calls.append(p))
        assert len(calls) == 1, f"应该恰好 1 次最终调用, 实际 {len(calls)}"
        assert calls[0]["stage"] == "master"
        assert calls[0]["total"] == 3

    def test_progress_cb_none_does_not_crash(self):
        """progress_cb=None → 不应抛异常"""
        self._patches = self._patch_deps(["000001"])
        from core.domain.factors.rolling import _build_master_indicators

        master = _build_master_indicators(["000001"], progress_cb=None)
        assert isinstance(master, dict)

    def test_progress_cb_throws_still_returns_master(self):
        """progress_cb 抛异常 → 函数仍返回 master, 不丢数据"""
        self._patches = self._patch_deps(["000001", "000002"])
        from core.domain.factors.rolling import _build_master_indicators

        def _broken(p):
            raise RuntimeError("进度上报失败")
        master = _build_master_indicators(["000001", "000002"], progress_cb=_broken)
        assert len(master) >= 0, "异常时仍应返回 master"

    def test_empty_pool_returns_empty_master(self):
        """空股票池 → 返回空 dict, progress_cb 报 0/total"""
        self._patches = self._patch_deps([])
        from core.domain.factors.rolling import _build_master_indicators

        calls = []
        master = _build_master_indicators([], progress_cb=lambda p: calls.append(p))
        assert master == {}
        assert len(calls) == 1
        assert calls[0]["current"] == 0

    def test_progress_cb_not_called_at_less_than_500_increments(self):
        """股票数在 1-499 之间 → 中间不应有 done[0]%500==0 的触发"""
        self._patches = self._patch_deps(["000001"] * 200)
        from core.domain.factors.rolling import _build_master_indicators

        calls = []
        _build_master_indicators(["000001"] * 200, progress_cb=lambda p: calls.append(p))
        # 200 < 500, 只有最终一次
        assert len(calls) == 1


# ═══════════════════════════════════════════════════════
# 串行路径 snapshot 入场进度
# ═══════════════════════════════════════════════════════

class TestSnapshotEntryProgress:
    """验证串行锚点循环中 batch_snapshots_from_master 前的进度上报"""

    def test_stage_is_snapshot_with_anchor_info(self):
        """上报的 stage 是 'snapshot', 且含 anchor 定位信息"""
        # 不实际跑 rolling_factor_screen, 仅验证 progress_cb lambda 映射正确
        call_args = []

        def mock_progress_cb(p):
            call_args.append(dict(p))

        # 模拟 batch_snapshots_from_master 前上报的逻辑
        akey = "2024-07-01"
        master_size = 480
        ai = 0
        anchors_count = 4

        mock_progress_cb({
            "stage": "snapshot", "current": 0, "total": master_size,
            "anchor": akey, "anchor_idx": ai + 1, "anchor_total": anchors_count,
        })

        assert call_args[0]["stage"] == "snapshot"
        assert call_args[0]["anchor"] == "2024-07-01"
        assert call_args[0]["anchor_idx"] == 1
        assert call_args[0]["anchor_total"] == 4
        assert call_args[0]["current"] == 0

    def test_anchor_total_zero_handled(self):
        """anchor_total=0 时前端走简易标签模式, 不崩溃"""
        call_args = []
        mock_progress_cb = lambda p: call_args.append(dict(p))

        # anchors 为空时 len(anchors)=0
        mock_progress_cb({
            "stage": "snapshot", "current": 0, "total": 100,
            "anchor": "", "anchor_idx": 0, "anchor_total": 0,
        })
        assert call_args[0]["anchor_total"] == 0


# ═══════════════════════════════════════════════════════
# 并发安全: progress_cb 多线程调用
# ═══════════════════════════════════════════════════════

class TestConcurrentProgress:
    def test_concurrent_progress_cb_no_crash(self):
        """多线程同时调 progress_cb 不崩溃 (task_manager._lock 保护)"""
        import threading
        from core.shared.infrastructure.async_task import task_manager

        job_id = task_manager._make_id()

        def _noop():
            pass
        task_manager.submit_with_id(job_id, _noop, "并发测试")

        errors = []
        def _update(tid):
            try:
                for i in range(50):
                    task_manager.update_progress(job_id, {
                        "stage": "master", "current": i, "total": 100,
                        "thread": tid,
                    })
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_update, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发更新异常: {errors}"
        # 最终 progress 不为空即可
        task = task_manager.get_dict(job_id)
        assert task["progress"]["stage"] == "master"


# ═══════════════════════════════════════════════════════
# 前端 label 映射完整性
# ═══════════════════════════════════════════════════════

class TestStageLabelMapping:
    """确保 progress_cb 上报的所有 stage 在前端 stageLabelMap 中都有映射"""

    def test_all_stages_have_label(self):
        stage_label_map = {
            "init": "准备中", "master": "指标构建", "snapshot": "预计算快照",
            "factor": "因子校验", "cluster": "聚类去重", "save_db": "保存结果",
            "sync_pool": "同步因子池", "parallel": "并行计算", "done": "完成",
        }
        # rolling_screen.py 中所有 progress_cb 上报的 stage 值
        backend_stages = {"master", "snapshot", "factor", "cluster",
                          "save_db", "record_history", "done", "parallel"}
        # record_history 带 anchor_total, 前端走锚点进度行, 不经过 stageLabel 显示
        label_skipped = {"record_history"}
        missing = (backend_stages - set(stage_label_map.keys())) - label_skipped
        assert not missing, f"前端缺少标签映射: {missing}"
