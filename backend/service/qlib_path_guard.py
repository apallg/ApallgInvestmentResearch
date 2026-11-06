"""Qlib 路径安全校验 — 防止模型标识路径穿越。"""
from pathlib import Path

from fastapi import HTTPException


def assert_run_path(mlruns_dir, experiment_id, run_id):
    """校验实验/运行标识为简单段并返回解析后的 run 目录, 防路径穿越 (../..)"""
    for part in (experiment_id, run_id):
        if not part or part in (".", "..") or any(ch in part for ch in "/\\:"):
            raise HTTPException(400, f"非法模型标识: {part!r}")
    return (Path(mlruns_dir) / experiment_id / run_id).resolve()
