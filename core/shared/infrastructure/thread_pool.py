"""
线程池兼容层 — 委托给 core.infrastructure.executors

三个真实执行器:
  - io_executor:    通用线程池 (_physical*2) — HTTP/DB/Redis/磁盘 IO
  - light_executor:  轻量线程池 (4) — 调度器/启动任务, 独立隔离, 不被 default 阻塞
  - cpu_executor:   进程池 (_physical-2) — 通过 run_cpu() 使用, 用于纯计算任务

default/light/error 三个池均由 ThreadPoolManager 统一管理。
default 是 IO 线程池 (非进程池) — CPU 密集型任务请用 run_cpu()。
"""

import threading
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List
from contextlib import contextmanager

from .executors import io_executor, light_executor, error_executor, run_batch

logger = logging.getLogger(__name__)


class PoolKind:
    DEFAULT = "default"
    LIGHT = "light"
    ERROR = "error"


@dataclass
class PoolSnapshot:
    kind: str
    max_workers: int
    active_count: int
    queued_count: int
    queue_limit: int
    total_submitted: int
    total_completed: int
    total_failed: int
    saturation_pct: float
    avg_wait_ms: float
    avg_exec_ms: float
    health: str


class ManagedThreadPool:

    def __init__(self, kind: str, max_workers: int, queue_size: int = 0,
                 task_timeout: float = 600, monitor=None, executor=None):
        self.kind = kind
        self.max_workers = max_workers
        self._queue_size = queue_size
        self._task_timeout = task_timeout
        self._executor = executor  # None → 走 io_executor
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._lock = threading.Lock()

    @property
    def raw_executor(self):
        return self._executor or io_executor

    def _get_executor(self):
        return self._executor or io_executor

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        with self._lock:
            self._submitted += 1

        def _wrap():
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._completed += 1
                return result
            except Exception:
                with self._lock:
                    self._failed += 1
                raise

        return self._get_executor().submit(_wrap)

    def map(self, fn: Callable, iterable, timeout: Optional[float] = None,
            max_workers: Optional[int] = None):
        items = list(iterable)
        limit = max_workers if max_workers is not None else self.max_workers * 4
        from .executors import run_batch
        global_limit = getattr(run_batch, "_force_limit", None)
        if global_limit and global_limit < limit:
            limit = global_limit
        for result in run_batch(fn, items, max_concurrent=limit, timeout=timeout,
                                executor=self._executor):
            yield result

    def map_all(self, fn: Callable, iterable, timeout: Optional[float] = None,
                max_workers: Optional[int] = None) -> List:
        return list(self.map(fn, iterable, timeout, max_workers=max_workers))

    @contextmanager
    def borrow(self):
        yield self

    def snapshot(self) -> PoolSnapshot:
        with self._lock:
            in_flight = self._submitted - self._completed - self._failed
            executing = min(in_flight, self.max_workers)
            return PoolSnapshot(
                kind=self.kind,
                max_workers=self.max_workers,
                active_count=executing,
                queued_count=max(0, in_flight - executing),
                queue_limit=-1,  # -1 = 无界队列 (ThreadPoolExecutor 内部队列无限)
                total_submitted=self._submitted,
                total_completed=self._completed,
                total_failed=self._failed,
                saturation_pct=round(executing / self.max_workers * 100, 1) if self.max_workers > 0 else 0,
                avg_wait_ms=-1,  # -1 = 未采集 (需在 _wrap 内打点)
                avg_exec_ms=-1,  # -1 = 未采集
                health="ok" if executing < self.max_workers * 0.7 else "busy",
            )

    def shutdown(self, wait: bool = True):
        logger.debug(f"[ThreadPool:{self.kind}] shutdown 委托给全局 executors.shutdown")


class ThreadPoolManager:

    def __init__(self):
        self._default = ManagedThreadPool(PoolKind.DEFAULT, io_executor._max_workers, queue_size=8)
        self._light = ManagedThreadPool(PoolKind.LIGHT, 4, executor=light_executor)
        self._error = ManagedThreadPool(PoolKind.ERROR, 1, executor=error_executor)
        self._shutdown = False

    @property
    def default(self) -> ManagedThreadPool:
        return self._default

    @property
    def light(self) -> ManagedThreadPool:
        return self._light

    @property
    def error(self) -> ManagedThreadPool:
        return self._error

    def submit(self, fn: Callable, *args, kind: str = None, **kwargs) -> Future:
        if kind == PoolKind.LIGHT:
            pool = self._light
        elif kind == PoolKind.ERROR:
            pool = self._error
        else:
            pool = self._default
        return pool.submit(fn, *args, **kwargs)

    def stats(self) -> Dict:
        ds = self._default.snapshot()
        ls = self._light.snapshot()
        es = self._error.snapshot()
        return {
            "pools": {
                "default": ds.__dict__,
                "light": ls.__dict__,
                "error": es.__dict__,
            },
            "total_active": ds.active_count + ls.active_count + es.active_count,
            "total_workers": ds.max_workers + ls.max_workers + es.max_workers,
            "total_completed": ds.total_completed + ls.total_completed + es.total_completed,
            "total_failed": ds.total_failed + ls.total_failed + es.total_failed,
            "health": "ok" if ds.health == "ok" and ls.health == "ok" and es.health == "ok" else "busy",
        }

    def shutdown(self, timeout: Optional[float] = None):
        if self._shutdown:
            return
        self._shutdown = True
        from .executors import shutdown as _shutdown
        _shutdown(wait=True, cancel_futures=True)
        logger.info("[ThreadPoolManager] shutdown 完成 (cancel_futures=True)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


thread_pool = ThreadPoolManager()
