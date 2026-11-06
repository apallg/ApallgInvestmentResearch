"""
全局错误日志存储 — 按日期+指纹去重写入 MySQL error_logs 表
"""

import hashlib
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict

from core.shared import config

logger = logging.getLogger(__name__)

__all__ = ["log_error", "query_errors", "query_date_summary",
           "cleanup", "auto_cleanup"]

TABLE = "error_logs"
MAX_MSG_LEN = 200


def log_error(
    *,
    level: str = "error",
    source: str,
    method: str = "",
    path: str = "",
    exc_type: str,
    exc_msg: str = "",
    traceback: str = "",
) -> bool:
    if not config.ERROR_LOG_ENABLED:
        return False

    fp = hashlib.md5(f"{exc_type}:{exc_msg[:MAX_MSG_LEN]}".encode("utf-8")).hexdigest()

    if level == "critical" and config.ERROR_LOG_CRITICAL_NOTIFY:
        _notify_critical(exc_type, exc_msg, source, path)

    try:
        from core.data import Database
        if not Database.is_available():
            return False

        today = date.today()
        now = datetime.now()

        Database.execute(
            f"INSERT INTO `{TABLE}` "
            f"(error_date, created_at, level, source, method, path, "
            f" exc_type, exc_msg, traceback, count, fingerprint, last_at) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s,%s) "
            f"ON DUPLICATE KEY UPDATE "
            f"count=count+1, last_at=VALUES(last_at), traceback=VALUES(traceback), "
            f"level=VALUES(level)",
            (today, now, level, source, method, path,
             exc_type, exc_msg, traceback, fp, now)
        )

        return True
    except Exception:
        logger.exception("写入 error_logs 失败")
        return False


def _notify_critical(exc_type: str, exc_msg: str, source: str, path: str):
    try:
        from core.shared.infrastructure.notify import get_wecom, _safe_truncate_bytes
        wecom = get_wecom()
        if not wecom.enabled:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (f"## 🚨 系统严重错误\n"
               f"> 类型: **{exc_type}**\n"
               f"> 来源: {source}\n"
               f"> 路径: {path}\n"
               f"> 详情: {exc_msg[:200]}\n"
               f"> 时间: {now}\n")
        wecom.send_markdown(_safe_truncate_bytes(msg, 4096))
    except Exception:
        pass


def query_errors(
    *,
    date_str: str = None,
    level: str = None,
    source: str = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict]:
    if not config.ERROR_LOG_ENABLED:
        return []

    try:
        from core.data import Database
        if not Database.is_available():
            return []

        wheres = []
        params = []

        if date_str:
            wheres.append("error_date = %s")
            params.append(date_str)
        if level:
            wheres.append("level = %s")
            params.append(level)
        if source:
            wheres.append("source = %s")
            params.append(source)

        where_clause = " WHERE " + " AND ".join(wheres) if wheres else ""
        sql = (
            f"SELECT id, error_date, created_at, level, source, method, path, "
            f"exc_type, exc_msg, count, fingerprint, last_at "
            f"FROM `{TABLE}`{where_clause} "
            f"ORDER BY last_at DESC LIMIT %s OFFSET %s"
        )
        params.extend([limit, offset])

        rows = Database.fetchall(sql, tuple(params))
        results = []
        for r in rows:
            d = dict(r)
            for k in ("error_date", "created_at", "last_at"):
                if d.get(k):
                    d[k] = str(d[k])
            results.append(d)
        return results
    except Exception:
        logger.exception("查询 error_logs 失败")
        return []


def query_date_summary(days: int = 7) -> List[Dict]:
    if not config.ERROR_LOG_ENABLED:
        return []

    try:
        from core.data import Database
        if not Database.is_available():
            return []

        rows = Database.fetchall(
            f"SELECT error_date, level, COUNT(*) AS types, SUM(count) AS total "
            f"FROM `{TABLE}` WHERE error_date >= %s "
            f"GROUP BY error_date, level ORDER BY error_date DESC, FIELD(level,'critical','error','warning')",
            (str(date.today() - timedelta(days=days)),)
        )
        results = []
        for r in rows:
            d = dict(r)
            for k in ("error_date",):
                if d.get(k):
                    d[k] = str(d[k])
            if "total" in d and d["total"] is not None:
                d["total"] = float(d["total"])
            results.append(d)
        return results
    except Exception:
        logger.exception("查询 error_logs 摘要失败")
        return []


def cleanup(before_date: str) -> int:
    if not config.ERROR_LOG_ENABLED:
        return 0

    try:
        from core.data import Database
        if not Database.is_available():
            return 0

        Database.execute(
            f"DELETE FROM `{TABLE}` WHERE error_date < %s", (before_date,)
        )
        logger.info(f"error_logs 清理完成 (before {before_date})")
        return -1
    except Exception:
        logger.exception("清理 error_logs 失败")
        return 0


def auto_cleanup():
    before = date.today() - timedelta(days=config.ERROR_LOG_RETENTION_DAYS)
    cleanup(before.isoformat())
