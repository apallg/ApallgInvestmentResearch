"""
因子注册中心 — 提供 get_pool() / get_screener() 的统一入口。
本模块不依赖任何 core 业务模块 (回退构造时的延迟 import 除外)。
"""
import threading
from typing import Callable, Optional, Any

_lock = threading.RLock()

_pool: Any = None
_pool_factory: Optional[Callable[[], Any]] = None
_pool_init_error: Optional[str] = None

_screener: Any = None
_screener_factory: Optional[Callable[[], Any]] = None
_screener_init_error: Optional[str] = None


# ── FactorPoolManager ──

def register_pool_factory(factory: Callable[[], Any]):
    global _pool_factory
    with _lock:
        _pool_factory = factory


def get_pool():
    global _pool, _pool_factory, _pool_init_error
    if _pool is not None:
        return _pool
    with _lock:
        if _pool is not None:
            return _pool
        if _pool_init_error:
            raise RuntimeError(f"FactorPoolManager 初始化失败: {_pool_init_error}")
        try:
            if _pool_factory is not None:
                _pool = _pool_factory()
            else:
                from core.domain.factors.pool import FactorPoolManager
                _pool = FactorPoolManager()
        except Exception as e:
            _pool_init_error = str(e)
            raise
        return _pool


def reload_pool():
    global _pool, _pool_factory, _pool_init_error
    with _lock:
        _pool_init_error = None
        try:
            if _pool_factory is not None:
                _pool = _pool_factory()
            else:
                from core.domain.factors.pool import FactorPoolManager
                _pool = FactorPoolManager()
        except Exception as e:
            _pool_init_error = str(e)
            raise
        return _pool


# ── BuySideScreener ──

def register_screener_factory(factory: Callable[[], Any]):
    global _screener_factory
    with _lock:
        _screener_factory = factory


def get_screener():
    global _screener, _screener_factory, _screener_init_error
    if _screener is not None:
        return _screener
    with _lock:
        if _screener is not None:
            return _screener
        if _screener_init_error:
            raise RuntimeError(f"BuySideScreener 初始化失败: {_screener_init_error}")
        try:
            if _screener_factory is not None:
                _screener = _screener_factory()
            else:
                from core.domain.screening._screener import BuySideScreener
                _screener = BuySideScreener()
        except Exception as e:
            _screener_init_error = str(e)
            raise
        return _screener
