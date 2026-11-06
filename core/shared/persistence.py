"""通用文件持久化 + 时间获取 — pickle/JSON 保存/加载, 当前时间。

供 domain 层的模型/状态/配置持久化使用。文件 I/O 与时间获取集中于此,
domain 层 (L3) 只调本模块, 不直接 open/读写文件或调 datetime。
"""
import os
import json
import pickle
import datetime
import tempfile


def _ensure_dir(path):
    """确保路径的父目录存在 (空目录名安全, 不抛异常)。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_pickle(obj, path):
    """保存对象到 pickle 文件 (自动建目录)。"""
    _ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """从 pickle 文件加载对象, 文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data, path):
    """保存 dict 到 JSON 文件 (自动建目录)。"""
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_json_atomic(data, path):
    _ensure_dir(path)
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_json(path):
    """从 JSON 文件加载 dict, 文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text, path):
    """保存文本到文件 (自动建目录)。"""
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_text(path):
    """从文本文件加载内容, 文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_bytes(path):
    """从文件加载字节, 文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def remove_dir(path):
    """删除目录 (忽略不存在/错误)。"""
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def save_parquet(df, path):
    """保存 DataFrame 到 parquet 文件 (自动建目录)。"""
    _ensure_dir(path)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """从 parquet 文件加载 DataFrame, 文件不存在返回 None。"""
    import pandas as pd
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


# ── 当前时间 (L3 通过本模块获取, 不直接调 datetime) ──

def today_date():
    """返回今天日期 (date 对象)。实时场景入口使用。"""
    return datetime.date.today()


def today_iso():
    """返回今天日期 ISO 字符串。"""
    return datetime.date.today().isoformat()


def now_iso():
    """返回当前时间戳 ISO 字符串。"""
    return datetime.datetime.now().isoformat()


def now_strftime(fmt="%Y%m%d_%H%M%S"):
    """返回当前时间戳格式化字符串。"""
    return datetime.datetime.now().strftime(fmt)
