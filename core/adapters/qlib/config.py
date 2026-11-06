"""
qlib 初始化封装 — 懒加载，线程安全，环境变量预设 + DatasetH pickle 缓存

包含 pandas 2.x 兼容补丁: rolling(window=0) → expanding()
"""
import os
import hashlib
import json
import pickle
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── pandas 2.x 兼容补丁 ── 必须最先执行
import pandas as pd
if not hasattr(pd.DataFrame, '_hermes_patched_roll'):
    _orig_df_roll = pd.DataFrame.rolling
    _orig_s_roll = pd.Series.rolling
    def _patched_df_roll(obj, window, *args, **kwargs):
        if window == 0:
            return obj.expanding(*args, **kwargs)
        return _orig_df_roll(obj, window, *args, **kwargs)
    def _patched_s_roll(obj, window, *args, **kwargs):
        if window == 0:
            return obj.expanding(*args, **kwargs)
        return _orig_s_roll(obj, window, *args, **kwargs)
    pd.DataFrame.rolling = _patched_df_roll
    pd.Series.rolling = _patched_s_roll
    pd.DataFrame._hermes_patched_roll = True
# ──────────────────────────────────────

_initialized = False
_lock = threading.Lock()
_cache_dir = None


def _get_cache_dir():
    global _cache_dir
    if _cache_dir is None:
        from core.shared import config as app_config
        provider = app_config.QLIB_PROVIDER_URI
        if provider.startswith("file:"):
            provider = provider[5:]
        _cache_dir = Path(provider) / "dataset_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


def _make_dataset_cache_key(dataset_config):
    from core.adapters.qlib.fundval import current_feature_contract
    handler = dataset_config.get('kwargs', {}).get('handler', {})
    keys = handler.get('kwargs', {}).get('factor_keys', []) if isinstance(handler, dict) else []
    contract = current_feature_contract(keys) if keys else None
    raw = json.dumps({'dataset': dataset_config, 'feature_contract': contract}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def build_dataset_cached(dataset_config, max_age_days=7):
    """构建 DatasetH/TSDatasetH，优先走 pickle 磁盘缓存

    首次构建耗时 200-400s，后续命中缓存 <10s。
    缓存 key 由 factor_set + instruments + date_range + handler_kwargs 决定。
    """
    from qlib.utils import init_instance_by_config

    # 每次构建前先做一次轻量清理, 避免回测频繁执行时缓存堆积到 70GB+
    cleanup_dataset_cache(max_age_days=2, max_files=3)

    cache_key = _make_dataset_cache_key(dataset_config)
    cache_path = _get_cache_dir() / f"{cache_key}.pkl"

    if cache_path.exists():
        import time
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_days * 86400:
            try:
                with open(cache_path, "rb") as f:
                    dataset = pickle.load(f)
                logger.info(f"[Qlib Cache] 命中: {cache_key} (age={age/3600:.1f}h)")
                return dataset
            except Exception as e:
                logger.warning(f"[Qlib Cache] 反序列化失败, 重建: {e}")
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass

    logger.info(f"[Qlib Cache] 未命中, 构建 DatasetH (预计 2-5 分钟)...")
    dataset = init_instance_by_config(dataset_config)

    try:
        if hasattr(dataset, 'handler') and hasattr(dataset.handler, '_dump_all'):
            dataset.handler._dump_all = True
        if hasattr(dataset, '_dump_all'):
            dataset._dump_all = True
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f, protocol=5)
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"[Qlib Cache] 已保存: {cache_key} ({size_mb:.0f}MB)")
    except Exception as e:
        logger.warning(f"[Qlib Cache] pickle 保存失败 (非致命): {e}")

    return dataset


def cleanup_dataset_cache(max_age_days=7, max_files=5):
    """清理 dataset_cache 中过期或超量的 pickle 缓存

    先按天数清过期文件，再按文件数清多余的旧文件（保留最新 max_files 个）。
    """
    import time
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return
    threshold = time.time() - max_age_days * 86400
    deleted = 0
    freed = 0

    for f in cache_dir.glob("*.pkl"):
        try:
            if f.stat().st_mtime < threshold:
                size = f.stat().st_size
                f.unlink()
                deleted += 1
                freed += size
        except Exception:
            pass

    remaining = sorted(cache_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in remaining[max_files:]:
        try:
            size = f.stat().st_size
            f.unlink()
            deleted += 1
            freed += size
        except Exception:
            pass

    if deleted:
        logger.info(f"[Qlib Cache] 清理了 {deleted} 个缓存, 释放 {freed/(1024*1024):.0f}MB, 剩余 {min(len(remaining), max_files)} 个")


def _patch_qlib_git_encoding():
    """修复 Qlib Recorder._log_uncommitted_code 在 Windows 中文环境下的编码问题"""
    try:
        from qlib.workflow import recorder as _recorder_mod
        _orig = getattr(_recorder_mod.MLflowRecorder, '_log_uncommitted_code', None)
        if _orig is None:
            return

        def _patched(self):
            try:
                import subprocess
                status = subprocess.check_output(
                    ['git', 'status', '--porcelain'],
                    stderr=subprocess.STDOUT, encoding='utf-8', errors='replace'
                )
                diff = subprocess.check_output(
                    ['git', 'diff', '--cached'],
                    stderr=subprocess.STDOUT, encoding='utf-8', errors='replace'
                )
                if len(status) > 0:
                    self.client.log_text(self.id, status, 'code_status.txt')
                if len(diff) > 0:
                    self.client.log_text(self.id, diff, 'code_diff.txt')
            except Exception:
                try:
                    self.client.log_text(
                        self.id, '[encoding patch] skipped due to git output encoding issue',
                        'code_status.txt'
                    )
                except Exception:
                    pass

        _recorder_mod.MLflowRecorder._log_uncommitted_code = _patched
        logger.info("[Qlib Patch] git 编码修复已激活")
    except Exception as e:
        logger.warning(f"[Qlib Patch] 编码修复注入失败: {e}")

def reload_calendar():
    """清除 Qlib 日历缓存，强制下次 D.calendar() 重读磁盘文件"""
    try:
        from qlib.data.cache import H
        H["c"].clear()
        logger.info("[Qlib] 日历缓存已清除，下次查询将重读 day.txt")
    except Exception as e:
        logger.warning(f"[Qlib] 清除日历缓存失败: {e}")


def init_qlib():
    global _initialized
    from core.adapters.qlib.generation import assert_no_active_historical_release
    from core.shared import config as app_config

    assert_no_active_historical_release(app_config.QLIB_DATA_ROOT)
    if _initialized:
        return
    with _lock:
        if _initialized:
            return

        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        if os.name == "nt":
            os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

        import qlib
        from qlib.constant import REG_CN
        qlib.init(
            provider_uri=app_config.QLIB_PROVIDER_URI,
            region=REG_CN,
            joblib_backend="threading",
            expression_cache=None,
            dataset_cache=None,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": app_config.QLIB_MLFLOW_URI,
                    "default_exp_name": app_config.QLIB_EXPERIMENT_NAME,
                },
            },
        )
        # 强制重新注册自定义算子 (Qlib init 后 Operator 表被清空)
        from core.adapters.qlib.custom_handler import _register_ts_rank, _register_sign_op
        import core.adapters.qlib.custom_handler as _ch
        _ch._ts_rank_registered = False
        _ch._sign_op_registered = False
        _register_ts_rank()
        _register_sign_op()
        _ch._fundval_registered = False
        _ch._register_fundval()
        _patch_qlib_git_encoding()
        _initialized = True
