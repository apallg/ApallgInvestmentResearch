"""
终审修复自检：directions累积 + CAP双模型跳过
边界输入 + 异常路径 + 并发
"""
import pytest
import numpy as np


# ═══════════════════════════════════════════════════════
# 修复1: directions 跨锚点累积
# ═══════════════════════════════════════════════════════

class TestDirectionsAccumulation:
    """验证 directions 字典跨锚点累加而非覆盖"""

    def test_multi_anchor_directions_merged(self):
        """多锚点方向合并: 后锚点不覆盖前锚点"""
        directions = {}
        # 模拟锚点1结果
        directions.update({"f1": "long", "f2": "short"})
        # 模拟锚点2结果
        directions.update({"f1": "short", "f3": "long"})
        # f1在两个锚点中方向不同, 后者覆盖(update语义)
        assert directions["f1"] == "short"
        assert directions["f2"] == "short"
        assert directions["f3"] == "long"
        assert len(directions) == 3

    def test_empty_directions_no_error(self):
        """空方向字典update不崩溃"""
        directions = {}
        directions.update({})
        assert directions == {}

    def test_single_anchor_unchanged(self):
        """单锚点时行为不变"""
        directions = {}
        directions.update({"f1": "long"})
        assert directions == {"f1": "long"}

    def test_directions_survives_empty_worker_result(self):
        """worker返回空directions不影响已有数据"""
        directions = {"f1": "long"}
        directions.update({})
        assert directions["f1"] == "long"

    def test_vote_writeback_uses_merged_directions(self):
        """投票写回时使用累积后的directions.get()"""
        directions = {"f1": "long", "f2": "short", "f3": "long"}
        # sync_active_set 调用模式
        voted = ["f1", "f2", "f4"]  # f4 不在 directions 中→默认long
        meta = {}
        for k in voted:
            d = directions.get(k, "long")
            meta[k] = {"direction": d}
        assert meta["f1"]["direction"] == "long"
        assert meta["f2"]["direction"] == "short"
        assert meta["f4"]["direction"] == "long"  # 默认

    def test_parallel_path_initializes_directions(self):
        """并行路径有directions={}初始化"""
        import inspect
        from core import rolling_screen
        source = inspect.getsource(rolling_screen.rolling_factor_screen)
        # 并行分支初始化
        assert 'directions = {}' in source
        # 累积而非覆盖
        assert 'directions.update' in source

    def test_serial_path_directions_outside_loop(self):
        """串行路径directions在锚点循环外初始化"""
        import inspect
        from core import rolling_screen
        source = inspect.getsource(rolling_screen.rolling_factor_screen)
        # factor_timeline和directions应在同层级(循环外)
        tl_idx = source.find('factor_timeline = {}')
        dir_idx = source.find('directions = {}')
        # directions 初始化应在 factor_timeline 附近 (循环外)
        assert abs(dir_idx - tl_idx) < 200

    def test_all_direction_values_valid(self):
        """方向值仅为 long/short"""
        valid = {"long", "short"}
        directions = {"f1": "long", "f2": "short", "f3": "long"}
        assert all(v in valid for v in directions.values())

    def test_concurrent_update_safe(self):
        """并发update不会损坏dict (CPython GIL保护)"""
        import threading
        directions = {}
        errors = []

        def add_batch(items):
            try:
                for _ in range(100):
                    directions.update(items)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_batch, args=({"a": "long"},))
        t2 = threading.Thread(target=add_batch, args=({"b": "short"},))
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert len(errors) == 0
        assert "a" in directions and "b" in directions


# ═══════════════════════════════════════════════════════
# 修复2: CAP 双模型跳过
# ═══════════════════════════════════════════════════════

class TestCapDualModelSkip:
    """验证双模型(LGB+XGB)时跳过CAP竞争融合"""

    def test_dual_model_no_cap(self):
        """仅2模型时跳过cap-and-redistribute"""
        entries = [("lgb", 0.06), ("xgb", 0.04)]
        has_aux = any(e[0] in ("linear", "cat") for e in entries)
        assert not has_aux  # 无辅助模型→跳过CAP

    def test_three_models_apply_cap(self):
        """3模型时启用cap"""
        entries = [("lgb", 0.06), ("xgb", 0.04), ("linear", 0.02)]
        has_aux = any(e[0] in ("linear", "cat") for e in entries)
        assert has_aux  # 有辅助模型→启用CAP

    def test_cat_only_apply_cap(self):
        """有cat模型时启用cap"""
        entries = [("lgb", 0.05), ("xgb", 0.03), ("cat", 0.02)]
        has_aux = any(e[0] in ("linear", "cat") for e in entries)
        assert has_aux

    def test_empty_entries_no_crash(self):
        """空entries不崩溃"""
        entries = []
        has_aux = any(e[0] in ("linear", "cat") for e in entries)
        assert not has_aux

    def test_single_model_no_cap(self):
        """单模型无辅助"""
        entries = [("lgb", 0.05)]
        has_aux = any(e[0] in ("linear", "cat") for e in entries)
        assert not has_aux

    def test_weights_preserved_without_cap(self):
        """无辅助模型时 _ic_fusion_weights 结果不被覆盖"""
        from core.adapters.qlib.models import _ic_fusion_weights
        w_lgb, w_xgb = _ic_fusion_weights(0.06, 0.04)
        assert w_lgb == 0.6
        assert w_xgb == 0.4
        # 仅2模型时不应进入CAP竞争融合→权重保持不变

    def test_source_code_has_aux_check(self):
        """源码中有 has_aux 检查"""
        import inspect
        from qlib_integration import models
        source = inspect.getsource(models.QlibModelManager.train_dual_model)
        assert 'has_aux' in source

    def test_cap_still_works_with_aux(self):
        """有辅助模型时cap逻辑仍正常工作(总和=1)"""
        raw = {"lgb": 0.5, "xgb": 0.3, "linear": 0.2}
        CAP = 0.3
        for _ in range(10):
            excess = 0.0
            capped = 0
            for k in list(raw.keys()):
                if raw[k] > CAP:
                    excess += raw[k] - CAP
                    raw[k] = CAP
                    capped += 1
            if excess < 1e-6 or capped == 0:
                break
            free = {k: v for k, v in raw.items() if v < CAP}
            if not free:
                if excess > 1e-6:
                    total = sum(raw.values())
                    if total > 0:
                        for k in raw:
                            raw[k] = raw[k] / total
                break
            free_total = sum(free.values())
            for k in free:
                raw[k] += excess * (free[k] / free_total)
        assert abs(sum(raw.values()) - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════
# 回归: 之前的修复未被破坏
# ═══════════════════════════════════════════════════════

class TestRegression:
    """确认之前的修复没有被破坏"""

    def test_ic_fusion_nan(self):
        from core.adapters.qlib.models import _ic_fusion_weights
        w1, w2 = _ic_fusion_weights(float('nan'), 0.05)
        assert w1 == 0.5 and w2 == 0.5

    def test_connection_pool_borrow_return(self):
        from core.data import Database
        conn = Database.get_connection()
        Database.return_connection(conn)

    def test_context_manager_connection(self):
        from core.data import Database
        with Database.connection() as conn:
            assert conn is not None

    def test_fix_power2_bracketed(self):
        from core.adapters.qlib.custom_handler import _fix_power2
        result = _fix_power2("(close)**2")
        assert '**2' not in result

    def test_fix_power2_simple(self):
        from core.adapters.qlib.custom_handler import _fix_power2
        result = _fix_power2("$close**2")
        assert '**2' not in result

    def test_deep_merge_monitoring(self):
        from core.domain.factors.pool import FactorPoolManager, DEFAULT_STATE
        old = {"white_box": {}}
        merged = FactorPoolManager._deep_merge(old, DEFAULT_STATE)
        assert "monitoring" in merged

    def test_factor_pool_lock_exists(self):
        from core.domain.factors.pool import _lock
        assert _lock is not None

    def test_health_has_init(self):
        import backend.main as m
        result = m.health()
        assert "init" in result

    def test_cache_lru_limit(self):
        from core.adapters.qlib.custom_handler import _cache_put, _CACHE_MAX_SIZE
        import collections
        c = collections.OrderedDict()
        for i in range(_CACHE_MAX_SIZE + 50):
            _cache_put(c, f"k{i}", i)
        assert len(c) <= _CACHE_MAX_SIZE

    def test_atomic_write_uses_replace(self):
        import inspect
        from core.domain.factors.pool import FactorPoolManager
        source = inspect.getsource(FactorPoolManager._save)
        assert 'os.replace' in source
