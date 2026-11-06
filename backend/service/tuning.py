import logging
import os
import uuid

from backend.service.factor_panel import FactorPanelService
from core.domain.tuning._tuner import get_tuner
from core.shared import config


logger = logging.getLogger(__name__)


class TuningService:
    @staticmethod
    def _prepare_nm_panel(tuner, pool, eval_weeks, code_prefixes, progress, job_id):
        if eval_weeks < 8 or not getattr(config, "FACTOR_PANEL_ENABLED", True):
            return None, None, None, None
        panel_service = FactorPanelService()
        store = arrays = None
        cache_id = str(job_id or uuid.uuid4().hex[:12])
        cache_dir = os.path.join("data", "cache", "tuning", f"nm_{cache_id}")
        try:
            opt_weeks = max(6, eval_weeks // 8)
            dates = tuner._get_eval_dates(weeks=opt_weeks, step=5)
            if progress is not None:
                progress.update({
                    "stage": "shared_panel", "current": 0, "total": len(dates),
                    "message": "构建N-M共享映射面板...",
                    "backing_mode": "memmap",
                })
            manifest_path, manifest, _codes = panel_service.build_shared_memmap(
                pool,
                dates,
                cache_dir,
                code_prefixes=code_prefixes,
                include_tech=True,
            )
            store, arrays, _manifest = panel_service.open_shared_memmap(
                manifest_path, dates,
            )
            if progress is not None:
                progress.update({
                    "stage": "shared_panel", "current": len(dates), "total": len(dates),
                    "message": f"N-M共享面板就绪: {len(store)}个日期",
                    "backing_mode": "memmap",
                    "mapped_mb": round(manifest["mapped_bytes"] / (1024 * 1024), 1),
                })
            return panel_service, store, arrays, cache_dir
        except Exception:
            logger.warning("N-M共享面板构建失败，回退兼容路径", exc_info=True)
            if arrays is not None:
                panel_service.close_shared_memmap(arrays)
            panel_service.cleanup_shared_memmap(cache_dir)
            return panel_service, None, None, None

    @staticmethod
    def _cleanup_nm_panel(panel_service, store, arrays, cache_dir):
        if hasattr(store, "clear"):
            store.clear()
        if panel_service is not None and arrays is not None:
            panel_service.close_shared_memmap(arrays)
        if panel_service is not None and cache_dir:
            panel_service.cleanup_shared_memmap(cache_dir)

    def auto_tune(self, weights, skips, tech_presets, l0_thresholds,
                  pool="all", top_n=30, eval_weeks=12, top_k=10,
                  date_start=None, date_end=None, code_prefixes=None,
                  progress=None, job_id=None):
        tuner = get_tuner()
        return tuner.auto_tune(
            weights=weights,
            skips=skips,
            tech_presets=tech_presets,
            l0_thresholds=l0_thresholds,
            pool=pool,
            top_n=top_n,
            eval_weeks=eval_weeks,
            top_k=top_k,
            date_start=date_start,
            date_end=date_end,
            code_prefixes=code_prefixes,
            progress=progress,
        )

    def phase2_weight_optimization(self, active_signals, pool="all", top_n=30,
                                   eval_weeks=12, date_start=None, date_end=None,
                                   top_k=10, code_prefixes=None, progress=None,
                                   regime=None, job_id=None):
        tuner = get_tuner()
        panel_service, store, arrays, cache_dir = self._prepare_nm_panel(
            tuner, pool, eval_weeks, code_prefixes, progress, job_id,
        )
        try:
            return tuner.phase2_weight_optimization(
                active_signals=active_signals,
                pool=pool,
                top_n=top_n,
                eval_weeks=eval_weeks,
                date_start=date_start,
                date_end=date_end,
                top_k=top_k,
                code_prefixes=code_prefixes,
                progress=progress,
                regime=regime,
                nm_snapshot_cache=store,
            )
        finally:
            self._cleanup_nm_panel(panel_service, store, arrays, cache_dir)
