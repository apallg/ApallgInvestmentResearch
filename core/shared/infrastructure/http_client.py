"""
全局共享 HTTP Session — 连接池复用，防止 Windows TCP 端口耗尽。

从 core/data_fetcher.py 提取，供 core/tencent_kline.py 等模块使用，
避免循环导入。
"""

import threading
import requests


_http_session = None
_http_session_lock = threading.Lock()


def get_shared_session():
    global _http_session
    if _http_session is None:
        with _http_session_lock:
            if _http_session is None:
                from requests.adapters import HTTPAdapter
                _http_session = requests.Session()
                adapter = HTTPAdapter(pool_connections=20, pool_maxsize=40, max_retries=2)
                _http_session.mount("http://", adapter)
                _http_session.mount("https://", adapter)
    return _http_session
