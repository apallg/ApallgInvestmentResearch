"""
统一异步任务框架 — 内存缓存 + MySQL 持久化

使用:
    from core.shared.infrastructure.async_task import task_manager
    job_id = task_manager.submit(lambda: do_work(), "训练Qlib模型")

然后在 API 端点中:
    task = task_manager.get(job_id)
    return {"data": {"job_id": job_id, "status": task.status, "result": task.result}}
"""

import threading
import time
import uuid
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_HISTORY = 500
TABLE_NAME = "async_tasks"
STALL_SECONDS = 600


class TaskCancelled(Exception):
    pass


@dataclass
class TaskInfo:
    job_id: str
    status: str = "pending"
    description: str = ""
    progress: Dict = field(default_factory=dict)
    result: Any = None
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "description": self.description,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _ensure_table():
    try:
        from core.data import Database
        Database.execute(f"""
            CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                `job_id` VARCHAR(32) PRIMARY KEY,
                `status` VARCHAR(20) NOT NULL DEFAULT 'pending',
                `description` VARCHAR(200) DEFAULT '',
                `progress` JSON,
                `result` JSON,
                `error` TEXT,
                `created_at` DATETIME(3) NOT NULL,
                `updated_at` DATETIME(3) NOT NULL,
                INDEX idx_status (`status`),
                INDEX idx_created (`created_at`)
            )
        """)
        return True
    except Exception as e:
        logger.warning(f"async_tasks 建表失败: {e}")
        return False


def _db_save(task: TaskInfo):
    try:
        from core.data import Database
        created = _ts_to_dt(task.created_at) if task.created_at else "1970-01-01"
        updated = _ts_to_dt(task.updated_at) if task.updated_at else "1970-01-01"
        Database.execute(
            f"INSERT INTO `{TABLE_NAME}` (job_id, status, description, progress, result, error, created_at, updated_at) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
            f"ON DUPLICATE KEY UPDATE status=VALUES(status), progress=VALUES(progress), "
            f"result=VALUES(result), error=VALUES(error), updated_at=VALUES(updated_at)",
            (task.job_id, task.status, task.description,
             json.dumps(task.progress, ensure_ascii=False, default=str) if task.progress else None,
             json.dumps(task.result, ensure_ascii=False, default=str) if task.result is not None else None,
             task.error or "", created, updated),
        )
        return True
    except Exception as e:
        logger.debug(f"async_task DB save 失败: {e}")
        return False


def _ts_to_dt(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class AsyncTaskManager:

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._cancel_events: Dict[str, threading.Event] = {}
        self._futures = {}
        self._submitting = set()
        self._history: List[str] = []
        self._table_ensured = False

    def _init_table(self):
        if self._table_ensured:
            return True
        self._table_ensured = _ensure_table() is True
        return self._table_ensured

    @staticmethod
    def _make_id() -> str:
        return uuid.uuid4().hex[:12]

    def submit(self, fn: Callable, description: str = "",
               progress_cb: Optional[Callable] = None, pool: str = "light") -> str:
        return self.submit_with_id(self._make_id(), fn, description, progress_cb, pool=pool)

    def submit_with_id(self, job_id: str, fn: Callable, description: str = "",
                       progress_cb: Optional[Callable] = None, pool: str = "light") -> str:
        if not self._init_table():
            raise RuntimeError("异步任务状态表不可用，任务未启动")
        with self._lock:
            future = self._futures.get(job_id)
            if job_id in self._submitting or (future is not None and not future.done()):
                return job_id
            self._submitting.add(job_id)
        task = TaskInfo(
            job_id=job_id, status="pending", description=description,
            progress={"stage": "pending", "message": description or "排队中"},
            created_at=time.time(), updated_at=time.time(),
        )
        with self._lock:
            self._tasks[job_id] = task
            self._cancel_events[job_id] = threading.Event()
            self._history.append(job_id)
            if len(self._history) > MAX_HISTORY:
                expired = self._history[:-MAX_HISTORY]
                self._history = self._history[-MAX_HISTORY:]
                for jid in expired:
                    self._tasks.pop(jid, None)
                    self._futures.pop(jid, None)

        if not _db_save(task):
            with self._lock:
                self._submitting.discard(job_id)
                self._tasks.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
                self._futures.pop(job_id, None)
                self._history = [item for item in self._history if item != job_id]
            raise RuntimeError("异步任务初始状态持久化失败，任务未启动")

        def _wrapper():
            with self._lock:
                if job_id in self._tasks:
                    if self._cancel_events.get(job_id, threading.Event()).is_set():
                        self._tasks[job_id].status = "cancelled"
                        self._tasks[job_id].progress = {
                            **self._tasks[job_id].progress,
                            "stage": "cancelled",
                            "message": "任务已取消",
                            "heartbeat_at": self._heartbeat_iso(),
                        }
                        self._tasks[job_id].updated_at = time.time()
                        _db_save(self._tasks[job_id])
                        return
                    self._tasks[job_id].status = "running"
                    self._tasks[job_id].updated_at = time.time()
                    _db_save(self._tasks[job_id])
            try:
                result = fn() if progress_cb is None else (fn(progress_cb) if callable(progress_cb) else fn())
                with self._lock:
                    if job_id in self._tasks:
                        cancelled = self._cancel_events.get(job_id, threading.Event()).is_set()
                        custom_status = result.get("_task_status") if isinstance(result, dict) else None
                        self._tasks[job_id].status = "cancelled" if cancelled else (custom_status or "completed")
                        self._tasks[job_id].result = result
                        if cancelled:
                            self._tasks[job_id].progress = {
                                **self._tasks[job_id].progress,
                                "stage": "cancelled",
                                "message": "任务已取消",
                            }
                        self._tasks[job_id].updated_at = time.time()
                        _db_save(self._tasks[job_id])
            except TaskCancelled:
                with self._lock:
                    if job_id in self._tasks:
                        self._tasks[job_id].status = "cancelled"
                        self._tasks[job_id].progress = {
                            **self._tasks[job_id].progress,
                            "stage": "cancelled",
                            "message": "任务已取消",
                        }
                        self._tasks[job_id].updated_at = time.time()
                        _db_save(self._tasks[job_id])
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"[AsyncTask] {job_id} 失败:\n{tb}")
                with self._lock:
                    if job_id in self._tasks:
                        self._tasks[job_id].status = "failed"
                        self._tasks[job_id].error = tb
                        self._tasks[job_id].updated_at = time.time()
                        _db_save(self._tasks[job_id])

        from core.shared.infrastructure.thread_pool import thread_pool
        target_pool = getattr(thread_pool, pool, thread_pool.default)
        try:
            future = target_pool.submit(_wrapper)
            with self._lock:
                self._futures[job_id] = future
                self._submitting.discard(job_id)
        except Exception as error:
            with self._lock:
                self._submitting.discard(job_id)
                task.status = "failed"
                task.error = f"任务提交到线程池失败: {type(error).__name__}: {error}"
                task.updated_at = time.time()
                _db_save(task)
            raise
        return job_id

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            future = self._futures.get(job_id)
            return job_id in self._submitting or bool(future is not None and not future.done())

    def get(self, job_id: str) -> Optional[TaskInfo]:
        task = self._tasks.get(job_id)
        if task is not None:
            return task
        return self._load_from_db(job_id)

    def get_dict(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            local_task = job_id in self._tasks
            task = self._tasks.get(job_id)
        if task is None:
            task = self._load_from_db(job_id)
        if task is None:
            return None
        result = task.to_dict()
        if not local_task and task.status in ("pending", "running", "cancel_requested"):
            result["status"] = "interrupted"
            result["progress"] = {
                **(result.get("progress") or {}),
                "stage": "interrupted",
                "stalled": False,
                "message": "后端进程已重启，原任务执行线程不存在，请重新提交",
            }
            return result
        if task.status in ("pending", "running") and task.updated_at:
            stalled = time.time() - task.updated_at > STALL_SECONDS
            if stalled:
                result["status"] = "stalled"
                result["progress"] = {
                    **(result.get("progress") or {}),
                    "stalled": True,
                    "message": "超过 10 分钟未收到心跳，任务可能停滞",
                }
        return result

    def list_recent(self, limit: int = 50) -> List[Dict]:
        if not self._history:
            return self._list_from_db(limit)
        jids = self._history[-limit:]
        result = [self._tasks[jid].to_dict() for jid in jids if jid in self._tasks]
        if len(result) < limit:
            result = self._list_from_db(limit)
        return result[:limit]

    def update_progress(self, job_id: str, progress: Dict):
        with self._lock:
            if job_id in self._tasks:
                self._tasks[job_id].progress = {
                    **(progress or {}),
                    "heartbeat_at": self._heartbeat_iso(),
                    "stalled": False,
                }
                self._tasks[job_id].updated_at = time.time()
                _db_save(self._tasks[job_id])

    def heartbeat(self, job_id: str, progress: Optional[Dict] = None):
        self.update_progress(job_id, progress or self._progress(job_id))

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(job_id)
            if task is None:
                task = self._load_from_db(job_id)
                if task is None:
                    return False
                self._tasks[job_id] = task
            if task.status in ("completed", "failed", "cancelled", "awaiting_promotion"):
                return False
            self._cancel_events.setdefault(job_id, threading.Event()).set()
            task.status = "cancel_requested"
            task.progress = {
                **(task.progress or {}),
                "stage": "cancel_requested",
                "message": "已请求取消，等待当前步骤结束",
                "heartbeat_at": self._heartbeat_iso(),
            }
            task.updated_at = time.time()
            _db_save(task)
            return True

    def mark_cancelled(self, job_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(job_id) or self._load_from_db(job_id)
            if task is None:
                return False
            self._tasks[job_id] = task
            self._cancel_events.setdefault(job_id, threading.Event()).set()
            task.status = "cancelled"
            task.progress = {
                **(task.progress or {}),
                "stage": "cancelled",
                "message": "任务已取消",
                "heartbeat_at": self._heartbeat_iso(),
            }
            task.updated_at = time.time()
            _db_save(task)
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        if event is not None and event.is_set():
            return True
        task = self._tasks.get(job_id)
        return bool(task and task.status == "cancel_requested")

    def check_cancelled(self, job_id: str):
        if self.is_cancel_requested(job_id):
            raise TaskCancelled(job_id)

    @staticmethod
    def _heartbeat_iso() -> str:
        import datetime
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _progress(self, job_id: str) -> Dict:
        task = self._tasks.get(job_id)
        return dict(task.progress) if task else {}

    def _load_from_db(self, job_id: str) -> Optional[TaskInfo]:
        try:
            from core.data import Database
            row = Database.fetchone(
                f"SELECT * FROM `{TABLE_NAME}` WHERE job_id=%s", (job_id,))
            if row is None:
                return None
            return self._row_to_task(row)
        except Exception:
            return None

    def _list_from_db(self, limit: int = 50) -> List[Dict]:
        try:
            from core.data import Database
            rows = Database.fetchall(
                f"SELECT * FROM `{TABLE_NAME}` ORDER BY created_at DESC LIMIT %s", (limit,))
            return [self._row_to_task(r).to_dict() for r in rows]
        except Exception:
            return []

    @staticmethod
    def _row_to_task(row) -> TaskInfo:
        def to_float(v):
            if v is None:
                return 0.0
            if hasattr(v, 'strftime'):
                return v.timestamp()
            return float(v) if isinstance(v, (int, float)) else 0.0

        return TaskInfo(
            job_id=row["job_id"],
            status=row.get("status", "pending"),
            description=row.get("description", ""),
            progress=json.loads(row["progress"]) if isinstance(row.get("progress"), str) else (row.get("progress") or {}),
            result=json.loads(row["result"]) if isinstance(row.get("result"), str) else row.get("result"),
            error=row.get("error", ""),
            created_at=to_float(row.get("created_at")),
            updated_at=to_float(row.get("updated_at")),
        )

    def cleanup(self, max_age_seconds: int = 86400):
        now = time.time()
        with self._lock:
            expired = [jid for jid, t in list(self._tasks.items())
                      if now - t.updated_at > max_age_seconds]
            for jid in expired:
                del self._tasks[jid]
                self._cancel_events.pop(jid, None)
            self._history = [jid for jid in self._history if jid in self._tasks]
        try:
            from core.data import Database
            import datetime
            cutoff = datetime.datetime.now() - datetime.timedelta(seconds=max_age_seconds)
            Database.execute(
                f"DELETE FROM `{TABLE_NAME}` WHERE updated_at < %s", (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
        except Exception:
            pass


task_manager = AsyncTaskManager()
