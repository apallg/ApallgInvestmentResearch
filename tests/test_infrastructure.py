"""
基础设施测试 — 线程池职责分离 + pandas 补丁验证
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestErrorPool:
    """error_executor 独立于 light_executor"""

    def test_error_pool_exists(self):
        """error 池已注册"""
        from core.shared.infrastructure.thread_pool import thread_pool, PoolKind
        assert hasattr(thread_pool, 'error')
        assert PoolKind.ERROR == "error"

    def test_error_pool_independent(self):
        """error 池和 light 池是独立实例"""
        from core.shared.infrastructure.thread_pool import thread_pool
        assert thread_pool.error is not thread_pool.light
        assert thread_pool.error is not thread_pool.default

    def test_error_pool_single_worker(self):
        """error 池只有 1 个 worker"""
        from core.shared.infrastructure.thread_pool import thread_pool
        snap = thread_pool.error.snapshot()
        assert snap.max_workers == 1

    def test_error_pool_in_stats(self):
        """error 池出现在 stats 中"""
        from core.shared.infrastructure.thread_pool import thread_pool
        stats = thread_pool.stats()
        assert "error" in stats["pools"]


class TestPandasRollingPatch:
    """pandas 2.x 兼容补丁: rolling(window=0) → expanding()"""

    def test_patch_is_applied(self):
        """补丁已应用 (由 qlib_integration/config.py 导入)"""
        # qlib_integration 在导入时打补丁
        import core.adapters.qlib.config  # noqa: F401
        assert hasattr(pd.DataFrame, '_hermes_patched_roll'), \
            "pandas rolling 补丁未应用"

    def test_window_zero_uses_expanding(self):
        """window=0 时调用 expanding()"""
        import core.adapters.qlib.config  # noqa: F401
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        # rolling(window=0) 应返回 expanding 对象, 不抛出异常
        result = df["a"].rolling(0).mean()
        expected = df["a"].expanding().mean()
        pd.testing.assert_series_equal(result, expected)

    def test_window_nonzero_uses_rolling(self):
        """window>0 时正常使用 rolling()"""
        import core.adapters.qlib.config  # noqa: F401
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df["a"].rolling(2).mean()
        expected = pd.Series([np.nan, 1.5, 2.5, 3.5, 4.5], name="a")
        pd.testing.assert_series_equal(result, expected)

    def test_dataframe_and_series_both_patched(self):
        """DataFrame 和 Series 的 rolling 都被补丁覆盖"""
        import core.adapters.qlib.config  # noqa: F401
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert df.rolling(0).mean() is not None
        assert df["a"].rolling(0).mean() is not None


class TestAlphaHandlerPatch:
    """Alpha158.get_feature_config 补丁有正确的作用域"""

    def test_custom_handler_imports(self):
        """custom_handler 模块可导入"""
        try:
            import core.adapters.qlib.custom_handler  # noqa: F401
        except ImportError as e:
            pytest.skip(f"qlib 不可用: {e}")
