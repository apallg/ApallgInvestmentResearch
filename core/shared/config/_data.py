import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent.parent  # core/shared/config/_data.py → project root
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

CACHE_EXPIRE_DATA = 3600
CACHE_EXPIRE_LLM = 300
CACHE_EXPIRE_NAME = 86400
CACHE_EXPIRE_FINANCIAL = 86400

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "qilin_stock")
MYSQL_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "8"))
MYSQL_ENABLED = bool(MYSQL_PASSWORD and MYSQL_PASSWORD != "your_mysql_password_here")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

REALTIME_QUOTE_CHAIN = tuple(
    item.strip().lower()
    for item in os.getenv("REALTIME_QUOTE_CHAIN", "tencent,sina").split(",")
    if item.strip()
)
REALTIME_INDEX_CHAIN = tuple(
    item.strip().lower()
    for item in os.getenv("REALTIME_INDEX_CHAIN", "tencent,sina").split(",")
    if item.strip()
)
REALTIME_INTRADAY_CHAIN = tuple(
    item.strip().lower()
    for item in os.getenv("REALTIME_INTRADAY_CHAIN", "ths").split(",")
    if item.strip()
)
LOW_FREQUENCY_SOURCE = os.getenv("LOW_FREQUENCY_SOURCE", "tushare").strip().lower()
LIVE_QUOTE_MAX_AGE = int(os.getenv("LIVE_QUOTE_MAX_AGE", "5"))
LIVE_DISPLAY_MAX_AGE = int(os.getenv("LIVE_DISPLAY_MAX_AGE", "15"))

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "15"))
REDIS_SOCKET_TIMEOUT = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
REDIS_SOCKET_CONNECT_TIMEOUT = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "3"))

CACHE_EXPIRE_SECTOR_LIST = 900
CACHE_EXPIRE_SECTOR_STOCKS = 300
CACHE_EXPIRE_SECTOR_PERF = 30

__all__ = [
    "ROOT_DIR", "DATA_DIR", "CACHE_DIR",
    "CACHE_EXPIRE_DATA", "CACHE_EXPIRE_LLM", "CACHE_EXPIRE_NAME", "CACHE_EXPIRE_FINANCIAL",
    "MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE",
    "MYSQL_POOL_SIZE", "MYSQL_ENABLED",
    "TUSHARE_TOKEN",
    "REALTIME_QUOTE_CHAIN", "REALTIME_INDEX_CHAIN", "REALTIME_INTRADAY_CHAIN",
    "LOW_FREQUENCY_SOURCE", "LIVE_QUOTE_MAX_AGE", "LIVE_DISPLAY_MAX_AGE",
    "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_PASSWORD", "REDIS_ENABLED",
    "REDIS_MAX_CONNECTIONS", "REDIS_SOCKET_TIMEOUT", "REDIS_SOCKET_CONNECT_TIMEOUT",
    "CACHE_EXPIRE_SECTOR_LIST", "CACHE_EXPIRE_SECTOR_STOCKS", "CACHE_EXPIRE_SECTOR_PERF",
]
