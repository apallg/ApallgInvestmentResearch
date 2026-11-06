"""
统一执行器 — 替代旧 thread_pool.py 的四池架构

- io_executor:  ThreadPoolExecutor — HTTP/DB/Redis/磁盘 IO (懒初始化, _physical*2)
- light_executor: ThreadPoolExecutor — 调度器/启动任务 (4 线程, 独立隔离)
- error_executor: ThreadPoolExecutor — 错误日志写入 (1 线程, 不阻塞启动/调度)
- cpu_executor:  ProcessPoolExecutor — 因子计算/QLib训练/回测
  Windows 安全: 子进程自动设置 BLAS 单线程

使用:
    from core.shared.infrastructure.executors import io_executor, light_executor, run_batch, run_cpu

    future = io_executor.submit(fetch_data, url)
    result = run_cpu(heavy_compute, arg1, arg2)
    for result in run_batch(score_one, items):
        process(result)
"""

import os
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Callable, Optional, Any, Iterator, List

logger = logging.getLogger(__name__)

_cpu = os.cpu_count() or 8
_physical = _cpu // 2 if _cpu >= 16 else _cpu

# ── 线程池注册中心 (懒初始化, data-driven 替代 N 份拷贝) ──
_POOLS: dict = {}
_pools_lock = threading.Lock()


def _make_thread_pool(name: str, max_workers: int) -> ThreadPoolExecutor:
    key = name
    if key not in _POOLS:
        with _pools_lock:
            if key not in _POOLS:
                _POOLS[key] = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix=f"qt-{name}"
                )
    return _POOLS[key]


def _get_io_executor():
    return _make_thread_pool("io", _physical * 2)


def _get_light_executor():
    return _make_thread_pool("light", 4)


def _get_error_executor():
    return _make_thread_pool("error", 1)


def __getattr__(name: str):
    _registry = {"io_executor": _get_io_executor,
                 "light_executor": _get_light_executor,
                 "error_executor": _get_error_executor}
    if name in _registry:
        return _registry[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_cpu_executor: Optional[ProcessPoolExecutor] = None
_cpu_lock = threading.Lock()


def _process_initializer():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _get_cpu_executor() -> ProcessPoolExecutor:
    global _cpu_executor
    if _cpu_executor is None:
        with _cpu_lock:
            if _cpu_executor is None:
                _cpu_executor = ProcessPoolExecutor(
                    max_workers=max(1, _physical - 2),
                    initializer=_process_initializer,
                )
    return _cpu_executor


def run_cpu(fn: Callable, *args, **kwargs) -> Future:
    """提交 CPU 密集型任务到进程池。

    仅适用于输入输出小 (<1MB) 的纯计算任务。
    不要传 DataFrame 或带副作用的对象 (如 DB 连接、HTTP session)。
    """
    return _get_cpu_executor().submit(fn, *args, **kwargs)


def run_batch(
    fn: Callable,
    items: list,
    max_concurrent: Optional[int] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    timeout: Optional[float] = None,
) -> Iterator[Any]:
    """流式批量执行，信号量限流，滑动窗口提交。

    窗口大小 = max_concurrent，完成一个才提交下一个。内存 O(window) 而非 O(N)。
    """
    if not items:
        return

    from concurrent.futures import wait, FIRST_COMPLETED

    ex = executor or _get_io_executor()
    default_limit = getattr(run_batch, "_force_limit", None) or (_physical * 8)
    limit = max_concurrent or default_limit
    sem = threading.Semaphore(limit)
    lock = threading.Lock()
    total = len(items)
    completed = [0]
    idx = [0]
    results: List[Any] = [None] * total
    futures_map: dict = {}

    def _wrap(item):
        try:
            return fn(item)
        finally:
            sem.release()
            with lock:
                completed[0] += 1

    def _submit_next():
        while idx[0] < total:
            sem.acquire()
            with lock:
                if idx[0] >= total:
                    sem.release()
                    break
                i = idx[0]
                idx[0] += 1
            futures_map[ex.submit(_wrap, items[i])] = i

    _submit_next()

    while completed[0] < total:
        done, _ = wait(futures_map.keys(), timeout=timeout, return_when=FIRST_COMPLETED)
        for fut in done:
            i = futures_map.pop(fut)
            try:
                results[i] = fut.result(timeout=0)
            except Exception as e:
                logger.warning(f"[run_batch] 任务 [{i}/{total}] 失败: {type(e).__name__}: {e}")
                results[i] = e
        _submit_next()

    yield from results


def shutdown(wait: bool = True, cancel_futures: bool = True):
    """关闭所有执行器。cancel_futures=True 防止 hung 任务阻塞进程退出。"""
    for name, ref in _POOLS.items():
        try:
            ref.shutdown(wait=wait, cancel_futures=cancel_futures)
        except Exception:
            pass
    if _cpu_executor is not None:
        try:
            _cpu_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        except Exception:
            pass
    logger.info(f"[Executors] shutdown 完成 ({len(_POOLS)} pools + cpu)")
