"""
Redis 连接管理 — 连接池、序列化、自动降级、自动启动
"""
import logging
import json
import pickle
import threading
import os
from core.shared import config

logger = logging.getLogger(__name__)

_redis_instance = None
_redis_checked = False
_redis_available = False
_redis_fail_count = 0
_lock = threading.Lock()

REDIS_HOME = os.getenv("REDIS_HOME", "")
REDIS_SERVER_DIR = REDIS_HOME if REDIS_HOME else r"C:\Users\zhj\Redis-8.8.0-Windows-x64-msys2-with-Service\Redis-8.8.0-Windows-x64-msys2-with-Service"
REDIS_SERVER_PATH = os.path.join(REDIS_SERVER_DIR, "redis-server.exe")
REDIS_SERVER_CONF = os.path.join(REDIS_SERVER_DIR, "redis.conf")
REDIS_SERVER_CONF_ARG = "redis.conf"  # MSYS2 构建的 redis-server 用 cwd 解析相对路径，不能传 Windows 绝对路径


def _check_redis():
    global _redis_checked, _redis_available, _redis_fail_count
    if _redis_checked:
        return _redis_available
    with _lock:
        if _redis_checked:
            return _redis_available
        if not config.REDIS_ENABLED:
            logger.info("Redis 未启用 (REDIS_ENABLED=false)")
            _redis_checked = True
            _redis_available = False
            return False
        try:
            import redis
            pool = redis.ConnectionPool(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD or None,
                max_connections=config.REDIS_MAX_CONNECTIONS,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=False,
            )
            r = redis.Redis(connection_pool=pool)
            r.ping()
            r.close()
            _redis_available = True
            _redis_fail_count = 0
            logger.info(f"Redis 连接成功 {config.REDIS_HOST}:{config.REDIS_PORT} DB{config.REDIS_DB}")
        except Exception as e:
            _redis_available = False
            _redis_fail_count += 1
            level = logging.WARNING if _redis_fail_count == 1 else logging.DEBUG
            logger.log(level, f"Redis 不可用 ({config.REDIS_HOST}:{config.REDIS_PORT}): {e}，将使用磁盘缓存")
        _redis_checked = True
        return _redis_available


def get_redis():
    global _redis_instance
    if not redis_enabled():
        return None
    if _redis_instance is not None:
        return _redis_instance
    with _lock:
        if _redis_instance is not None:
            return _redis_instance
        try:
            import redis
            pool = redis.ConnectionPool(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD or None,
                max_connections=config.REDIS_MAX_CONNECTIONS,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=False,
            )
            _redis_instance = redis.Redis(connection_pool=pool)
            return _redis_instance
        except Exception as e:
            logger.error(f"Redis 连接创建失败: {e}")
            return None


def redis_enabled():
    return _check_redis()


def redis_recheck():
    global _redis_checked, _redis_available, _redis_instance
    with _lock:
        _redis_checked = False
        _redis_available = False
        _redis_instance = None
    return _check_redis()


def ensure_redis_running():
    if redis_enabled():
        return True
    if not config.REDIS_ENABLED:
        logger.info("Redis 未启用 (REDIS_ENABLED=false)，跳过自动启动")
        return False
    if not os.path.exists(REDIS_SERVER_PATH):
        logger.warning(f"Redis 服务端不存在: {REDIS_SERVER_PATH}")
        return False
    try:
        import subprocess
        subprocess.Popen(
            [REDIS_SERVER_PATH, REDIS_SERVER_CONF_ARG],
            cwd=REDIS_SERVER_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        logger.info(f"Redis 已自动启动: {REDIS_SERVER_PATH}")
        import time
        time.sleep(2)  # 等 Redis 就绪
        return redis_recheck()
    except Exception as e:
        logger.warning(f"Redis 自动启动失败: {e}")
        return False


def redis_serialize(value):
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        return b'\x01' + pickle.dumps(value, protocol=5)
    if isinstance(value, (dict, list)):
        try:
            return b'\x02' + json.dumps(value, ensure_ascii=False).encode('utf-8')
        except (TypeError, ValueError):
            return b'\x01' + pickle.dumps(value, protocol=5)
    if isinstance(value, (str, int, float, bool, type(None))):
        return b'\x02' + json.dumps(value, ensure_ascii=False).encode('utf-8')
    return b'\x01' + pickle.dumps(value, protocol=5)


class _SafeUnpickler(pickle.Unpickler):
    """受限反序列化: 只允许 pandas/numpy/标准库常见内部缓存类型, 阻断任意 __reduce__ 代码执行"""

    _ALLOWED_MODULES = {
        "builtins", "collections", "collections.abc", "copyreg", "datetime",
        "decimal", "fractions", "itertools", "operator", "re", "codecs",
        "_codecs", "encodings", "encodings.utf_8", "dateutil", "dateutil.parser",
    }

    def find_class(self, module, name):
        if module.startswith("pandas.") or module.startswith("numpy.") \
                or module in self._ALLOWED_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"拒绝反序列化不可信模块: {module}.{name}")


def _safe_unpickle(payload):
    return _SafeUnpickler(payload).load()


def redis_deserialize(data):
    if not isinstance(data, bytes) or len(data) < 1:
        return None
    prefix = data[0:1]
    payload = data[1:]
    if prefix == b'\x02':
        try:
            return json.loads(payload.decode('utf-8'))
        except Exception:
            pass
    elif prefix == b'\x01':
        try:
            return _safe_unpickle(payload)
        except Exception:
            pass
    return None


def make_redis_key(prefix, *args):
    parts = [prefix] + [str(a) for a in args]
    return ":".join(parts)


def _get(key):
    """从 Redis 获取值，失败返回 None（不抛异常）"""
    try:
        r = get_redis()
        if r is None:
            return None
        data = r.get(key)
        if data is None:
            return None
        return redis_deserialize(data)
    except Exception as e:
        logger.debug(f"Redis GET {key} 失败: {e}")
        return None


def _set(key, value, expire=3600):
    """向 Redis 写入值，失败返回 False（不抛异常）"""
    try:
        r = get_redis()
        if r is None:
            return False
        data = redis_serialize(value)
        r.setex(key, expire, data)
        return True
    except Exception as e:
        logger.debug(f"Redis SET {key} 失败: {e}")
        return False


def _delete(key):
    """从 Redis 删除 key，失败不抛异常"""
    try:
        r = get_redis()
        if r is None:
            return
        r.delete(key)
    except Exception as e:
        logger.debug(f"Redis DEL {key} 失败: {e}")


def _exists(key):
    """检查 key 是否存在"""
    try:
        r = get_redis()
        if r is None:
            return False
        return r.exists(key) > 0
    except Exception:
        return False


def get_redis_stats():
    stats = {"enabled": redis_enabled(), "available": False, "error": ""}
    if not stats["enabled"]:
        stats["error"] = "未启用"
        return stats
    try:
        r = get_redis()
        if r is None:
            stats["error"] = "连接失败"
            return stats
        info = r.info("memory")
        stats["available"] = True
        stats["used_memory_human"] = info.get("used_memory_human", "0")
        stats["peak_memory_human"] = info.get("used_memory_peak_human", "0")
        stats["total_keys"] = r.dbsize()

        snapshot_keys = [k.decode() if isinstance(k, bytes) else k for k in r.keys("snapshot:*")]
        daily_keys = [k.decode() if isinstance(k, bytes) else k for k in r.keys("daily:*")]
        qlib_keys = [k.decode() if isinstance(k, bytes) else k for k in r.keys("qlib:*")]
        spot_keys = [k.decode() if isinstance(k, bytes) else k for k in r.keys("spot:*")]

        stats["snapshot_count"] = len(snapshot_keys)
        stats["daily_cache_count"] = len(daily_keys)
        stats["qlib_cache_count"] = len(qlib_keys)
        stats["spot_cache_count"] = len(spot_keys)

        dates = []
        for k in snapshot_keys:
            parts = k.split(":")
            if len(parts) >= 3:
                dates.append(parts[-1])
        if dates:
            dates.sort()
            stats["snapshot_date_start"] = dates[0]
            stats["snapshot_date_end"] = dates[-1]

        daily_dates = set()
        for k in daily_keys:
            parts = k.split(":")
            if len(parts) == 5:  # daily:CODE:START:END:ADJ
                daily_dates.add(parts[2])
                daily_dates.add(parts[3])
        if daily_dates:
            dlist = sorted(daily_dates)
            stats["daily_date_start"] = dlist[0]
            stats["daily_date_end"] = dlist[-1]

        codes = set()
        for k in daily_keys:
            parts = k.split(":")
            if len(parts) >= 2:
                codes.add(parts[1])  # daily:CODE 或 daily:CODE:start:end:adj
        stats["daily_stock_count"] = len(codes)

        return stats
    except Exception as e:
        stats["error"] = str(e)
        return stats
