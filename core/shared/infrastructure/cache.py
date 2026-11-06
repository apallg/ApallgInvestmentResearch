"""
缓存管理 - Redis + diskcache 双层缓存，MySQL 最终回退
"""
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
import diskcache as dc
from core.shared import config

from core.shared.infrastructure import redis_client as rds

logger = logging.getLogger(__name__)


class CacheManager:
    """双层缓存：Redis (共享) → diskcache (本地) → MySQL (回退)"""

    _persist_lock = threading.Lock()
    _per_key_locks: dict = {}
    _per_key_locks_lock = threading.Lock()
    _save_count: int = 0  # 类级别, 跨实例计数 (避免多实例各计各的永不触发清理)

    def __init__(self, cache_dir: Optional[Path] = None, use_redis: bool = True):
        self._cache = dc.Cache(str(cache_dir or config.CACHE_DIR))
        self._redis_enabled = use_redis and rds.redis_enabled()

    @classmethod
    def _get_key_lock(cls, key: str) -> threading.Lock:
        with cls._per_key_locks_lock:
            if key not in cls._per_key_locks:
                # 定期清理闲置锁，防止无限增长（每 1000 个新 key 清一次）
                if len(cls._per_key_locks) > 10000:
                    cls._per_key_locks = {k: v for k, v in cls._per_key_locks.items()
                                          if v.locked() or k == key}
                cls._per_key_locks[key] = threading.Lock()
            return cls._per_key_locks[key]

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def set(self, key: str, value: Any, expire: int = 3600):
        self._cache.set(key, value, expire=expire)

    def get_or_set(self, key: str, func: Callable, expire: int = 3600) -> Any:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self._get_key_lock(key):
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            result = func()
            self._cache.set(key, result, expire=expire)
            return result

    # ═══════════════════════════════════════
    # Redis 缓存方法
    # ═══════════════════════════════════════

    def get_redis(self, key: str) -> Any:
        if not self._redis_enabled:
            return None
        return rds._get(key)

    def set_redis(self, key: str, value: Any, expire: int = 3600):
        if not self._redis_enabled:
            return False
        return rds._set(key, value, expire)

    def delete_redis(self, key: str):
        if not self._redis_enabled:
            return
        rds._delete(key)

    def get_or_set_redis(self, key: str, func: Callable, expire: int = 3600) -> Any:
        if self._redis_enabled:
            cached = rds._get(key)
            if cached is not None:
                return cached
        with self._get_key_lock(key):
            if self._redis_enabled:
                cached = rds._get(key)
                if cached is not None:
                    return cached
            result = func()
            if self._redis_enabled:
                rds._set(key, result, expire)
            return result

    def get_cached(self, key):
        """只读缓存链：Redis → diskcache，不触发上游获取"""
        result = self.get_redis(key)
        if result is None or isinstance(result, bool) or (hasattr(result, "empty") and result.empty):
            result = self.get(key)
        return result

    def persist(self, key, data, expire=3600):
        """持久化数据到 Redis + MySQL，带批量清理（线程安全）"""
        try:
            self.set_redis(key, data, expire)
        except Exception:
            pass
        if not config.MYSQL_ENABLED:
            return
        try:
            from core.data import DataCacheRepo
            with self._persist_lock:
                expires_at = datetime.now() + timedelta(seconds=expire)
                DataCacheRepo.save(key, data, expires_at)
                self._save_count += 1
                if self._save_count >= 30:
                    DataCacheRepo.cleanup_old(3)
                    self._save_count = 0
        except Exception as e:
            logger.debug(f"MySQL持久化失败(非关键): {e}")

    def get_with_fallback(self, key, func, expire=3600):
        """获取数据：Redis → diskcache → func → MySQL 四级回退 (per-key 锁防同 key 并发重复触发上游)"""
        with self._get_key_lock(key):
            # 0. 查 Redis（跨进程共享缓存）
            if self._redis_enabled:
                result = rds._get(key)
                if result is not None:
                    return result

            # 1. 查本地缓存
            result = self.get(key)
            if result is not None:
                return result

            # 2. 调用 func() 获取新数据，写 Redis + (可选)diskcache + MySQL
            try:
                result = func()
                if not self._redis_enabled:
                    self.set(key, result, expire=expire)
                self.persist(key, result, expire)
                return result
            except Exception as e:
                logger.debug(f"主数据源获取失败({str(key)[:40]}): {e}")

            # 3. func() 失败 → 回退到 MySQL
            if config.MYSQL_ENABLED:
                try:
                    from core.data import DataCacheRepo
                    stale = DataCacheRepo.get(key)
                    if stale is not None:
                        logger.info(f"MySQL回退命中: {str(key)[:40]}")
                        return stale
                except Exception as e:
                    logger.debug(f"MySQL回退失败({str(key)[:40]}): {e}")
            raise Exception("所有数据源不可用")

    def remove(self, key: str):
        self._cache.pop(key, None)
        if self._redis_enabled:
            rds._delete(key)

    def close(self):
        self._cache.close()

    @staticmethod
    def make_key(prefix: str, *args) -> str:
        return f"{prefix}:" + ":".join(str(a) for a in args)

    @staticmethod
    def make_redis_key(prefix: str, *args) -> str:
        return rds.make_redis_key(prefix, *args)
