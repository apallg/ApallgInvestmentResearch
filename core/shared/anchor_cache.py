"""
锚点缓存工具 — 供 config_tuner 和 rolling_screen 共享, 断开二者互引。
"""
import gc
from typing import Dict

_mem_cache: Dict[str, dict] = {}


def clear_mem_cache():
    """清空内存缓存 (替代 config_tuner._clear_mem_cache)"""
    _mem_cache.clear()
    gc.collect()


def get_mem_cache() -> Dict[str, dict]:
    return _mem_cache
