"""统一调度管理器 — APScheduler BackgroundScheduler 包装

替代原来 5 个独立 daemon 线程的 while-sleep 轮询。
"""

import threading
import time
import logging
import datetime
from typing import Optional, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)

_mgr = None


def get_unified_scheduler():
    global _mgr
    if _mgr is None:
        _mgr = UnifiedSchedulerManager()
    return _mgr


class UnifiedSchedulerManager:

    def __init__(self):
        self._scheduler = BackgroundScheduler(
            timezone="Asia/Shanghai",
            executors={'default': {'type': 'threadpool', 'max_workers': 8}}
        )
        self._lock = threading.Lock()
        self._job_meta = {}   # name -> dict
        self._started = False

    # ── 注册 ─────────────────────────────────────────────

    def register(
        self,
        name: str,
        func: Callable[[], None],
        cron: str,
        *,
        trading_day_only: bool = False,
        max_instances: int = 1,
        extra: Optional[dict] = None,
    ):
        """注册一个定时任务。

        Args:
            name: 任务名（唯一标识，trigger/status 用此名）
            func: 任务函数
            cron: 5 字段 cron 表达式（分 时 日 月 周）
            trading_day_only: True 时非交易日自动跳过
            max_instances: APScheduler 最大并发实例数（≥1），1 等价于 _busy 保护
            extra: 额外元数据，会合并到 status 输出
        """
        if name in self._job_meta:
            raise ValueError(f"任务 {name} 已注册")

        with self._lock:
            self._job_meta[name] = {
                "func": func,
                "running": False,
                "last_run": None,
                "last_error": None,
                "run_count": 0,
                "trading_day_only": trading_day_only,
                "cron": cron,
                "extra": extra or {},
            }

        def _wrapper():
            self._execute_job(name)

        self._scheduler.add_job(
            _wrapper,
            CronTrigger.from_crontab(cron, timezone="Asia/Shanghai"),
            name=name,
            id=name,
            max_instances=max_instances,
            replace_existing=True,
            jitter=30,
        )
        logger.info(f"[unified] 注册: {name} cron={cron} trading_day_only={trading_day_only}")

    def register_interval(
        self,
        name: str,
        func: Callable[[], None],
        seconds: int,
        *,
        trading_time_only: bool = False,
        max_instances: int = 1,
        extra: Optional[dict] = None,
    ):
        """注册一个间隔执行任务。

        Args:
            name: 任务名
            func: 任务函数
            seconds: 执行间隔（秒）
            trading_time_only: True 时非交易时段自动跳过
            max_instances: 最大并发实例数
            extra: 额外元数据
        """
        if name in self._job_meta:
            raise ValueError(f"任务 {name} 已注册")

        from apscheduler.triggers.interval import IntervalTrigger

        with self._lock:
            self._job_meta[name] = {
                "func": func,
                "running": False,
                "last_run": None,
                "last_error": None,
                "run_count": 0,
                "trading_day_only": False,
                "trading_time_only": trading_time_only,
                "cron": f"*/{seconds}s",
                "extra": extra or {},
            }

        def _wrapper():
            if trading_time_only:
                from datetime import datetime
                now = datetime.now()
                if now.weekday() >= 5:
                    return
                t = now.hour * 100 + now.minute
                if not (930 <= t <= 1505):
                    return
            self._execute_job(name)

        self._scheduler.add_job(
            _wrapper,
            IntervalTrigger(seconds=seconds, timezone="Asia/Shanghai"),
            name=name,
            id=name,
            max_instances=max_instances,
            replace_existing=True,
        )
        logger.info(f"[unified] 注册: {name} interval={seconds}s trading_time_only={trading_time_only}")

    # ── 生命周期 ─────────────────────────────────────────

    def start(self):
        if self._started:
            return
        self._scheduler.start()
        self._started = True
        logger.info("[unified] 统一调度器已启动")

    def stop(self):
        self._scheduler.shutdown(wait=False)
        self._started = False
        for meta in self._job_meta.values():
            meta["running"] = False
        logger.info("[unified] 统一调度器已停止")

    # ── 内部执行 ─────────────────────────────────────────

    def _execute_job(self, name: str):
        meta = self._job_meta[name]

        if meta["trading_day_only"]:
            try:
                from core.shared.trading_time import is_trading_day
                if not is_trading_day():
                    logger.info(f"[{name}] 非交易日，跳过")
                    return
            except ImportError:
                pass

        meta["running"] = True
        meta["last_error"] = None
        try:
            meta["func"]()
        except Exception as e:
            logger.exception(f"[{name}] 执行失败: {e}")
            meta["last_error"] = str(e)
        finally:
            meta["running"] = False
            meta["last_run"] = time.time()
            meta["run_count"] += 1

    # ── 单个作业控制 ─────────────────────────────────────

    def pause(self, name: str):
        try:
            self._scheduler.pause_job(name)
            logger.info(f"[unified] 暂停: {name}")
        except Exception as e:
            logger.warning(f"[unified] 暂停 {name} 失败: {e}")

    def resume(self, name: str):
        try:
            self._scheduler.resume_job(name)
            logger.info(f"[unified] 恢复: {name}")
        except Exception as e:
            logger.warning(f"[unified] 恢复 {name} 失败: {e}")

    # ── 手动触发 ─────────────────────────────────────────

    def trigger(self, name: str):
        meta = self._job_meta.get(name)
        if meta is None:
            raise ValueError(f"未知任务: {name}")

        if meta["running"]:
            logger.warning(f"[{name}] 上一次执行尚未完成，跳过本次手动触发")
            return

        thread_pool.light.submit(self._execute_job, name)
        logger.info(f"[{name}] 手动触发")

    # ── 状态查询 ─────────────────────────────────────────

    def get_meta(self, name: str) -> Optional[dict]:
        """获取单个任务的元数据。返回 None 表示未注册。"""
        meta = self._job_meta.get(name)
        if meta is None:
            return None
        job = self._scheduler.get_job(name)
        next_run = getattr(job, "next_run_time", None) if job else None
        return {
            "name": name,
            "running": meta["running"],
            "scheduled": job is not None and next_run is not None,
            "next_run": next_run.isoformat() if next_run else None,
            "last_run": datetime.datetime.fromtimestamp(meta["last_run"]).isoformat() if meta["last_run"] else None,
            "last_error": meta["last_error"],
            "run_count": meta["run_count"],
            **meta["extra"],
        }

    def get_all_meta(self) -> list[dict]:
        return [self.get_meta(name) for name in self._job_meta]

    def status(self, prefix: Optional[str] = None):
        """聚合状态查询。prefix 非空时只返回匹配该前缀的任务状态。"""
        all_meta = self.get_all_meta()
        if prefix:
            all_meta = [m for m in all_meta if m and m["name"].startswith(prefix)]
        return {
            "jobs": all_meta,
            "started": self._started,
            "job_count": len(self._job_meta),
        }
