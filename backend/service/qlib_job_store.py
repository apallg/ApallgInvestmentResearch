"""Qlib 任务状态存储 — 从 common.py 迁出的模块级状态。

保持与旧实现完全一致的内存字典 + 锁，避免迁移期行为变化。
"""
import threading

train_jobs = {}
train_lock = threading.Lock()
sync_lock = threading.Lock()
collect_job_id = None
