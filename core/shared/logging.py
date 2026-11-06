"""统一日志基础设施: 格式 / 级别 / 命名层级 / 文件轮转 / 第三方库静音

用法:
    from core.shared.logging import setup_logging, get_logger
    setup_logging()               # 进程入口调用一次 (backend/main.py)
    logger = get_logger(__name__) # 业务代码拿 apallg.* 层级 logger
"""
import logging
import multiprocessing
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

from core.shared import config

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_QUIET_LOGGERS = {
    "apscheduler": logging.WARNING,
    "qlib": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
}

_configured = False


def get_logger(name):
    """返回 apallg 命名层级下的 logger。"""
    if name.startswith("apallg."):
        return logging.getLogger(name)
    return logging.getLogger(f"apallg.{name}")


def _is_main_process():
    """判断当前进程是否为主进程。

    Windows 下 multiprocessing 使用 spawn，子进程会重新导入 backend.main；
    如果每个子进程都打开同一个 RotatingFileHandler，日志轮转时会因文件被多进程
    占用而抛出 WinError 32。文件日志只应由主进程持有。
    """
    try:
        return multiprocessing.current_process().name == "MainProcess"
    except Exception:
        return True


def setup_logging(level=None, log_dir=None, log_file="apallg.log", enable_file=None):
    """统一日志配置, 幂等可重复调用。

    enable_file:
      - True  : 始终添加文件 handler；
      - False : 仅控制台；
      - None  : 自动判断，只有主进程添加文件 handler（推荐）。
    """
    global _configured
    if _configured:
        return logging.getLogger("apallg")

    root = logging.getLogger()
    root.setLevel(_resolve_level(level if level is not None else config.LOG_LEVEL))

    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if enable_file is None:
        enable_file = _is_main_process()

    log_dir = log_dir if log_dir is not None else config.LOG_DIR
    if enable_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, log_file),
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)

    for name, lvl in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)

    _configured = True
    return logging.getLogger("apallg")


def _resolve_level(level):
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)
