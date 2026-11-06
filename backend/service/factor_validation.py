import datetime as dt
import logging

from backend.service.factor_panel import FactorPanelService
from core.domain.factors.adapter import PublicFactorAdapter
from core.domain.factors.validation import OverfittingChecker
from core.shared import config


logger = logging.getLogger(__name__)


class FactorValidationService:
    def portfolio_sort(self, factor_keys, pool="a500", eval_weeks=12, forward_days=20,
                       progress_cb=None):
        from core.domain.factors.portfolio_sort import run_batch_sort
        from core.domain.factors.registry import get_pool
        from core.shared.persistence import today_date
        from core.shared.trading_time import is_trading_day

        all_factors = get_pool().get_all_factors()
        factors = [
            {"key": key, "label": all_factors.get(key, {}).get("label", key)}
            for key in factor_keys
        ]
        end = today_date()
        start = end - dt.timedelta(weeks=eval_weeks)
        dates = []
        cursor = start
        while cursor <= end:
            if is_trading_day(cursor):
                dates.append(cursor)
            cursor += dt.timedelta(days=5)

        snapshots = None
        if getattr(config, "FACTOR_PANEL_ENABLED", True) and dates:
            try:
                snapshots, _codes = FactorPanelService().build_snapshot_store(
                    pool, dates, include_tech=True,
                )
            except Exception:
                logger.warning("分组回测共享面板构建失败，回退兼容快照", exc_info=True)
        try:
            return run_batch_sort(
                factors,
                pool=pool,
                start=start,
                end=end,
                eval_weeks=eval_weeks,
                forward_days=forward_days,
                progress_cb=progress_cb,
                snapshots=snapshots,
            )
        finally:
            if hasattr(snapshots, "clear"):
                snapshots.clear()

    def validate_public_factors(self, source="alpha158", eval_weeks=12, pool="all",
                                top_k=10, dedupe=True, anchor_date=None):
        adapter = PublicFactorAdapter()
        snapshots = None
        if getattr(config, "FACTOR_PANEL_ENABLED", True):
            try:
                anchor = dt.date.fromisoformat(anchor_date[:10]) if anchor_date else None
                checker = OverfittingChecker()
                if anchor is not None:
                    dates = set(checker._dates_between(
                        anchor - dt.timedelta(days=300), anchor + dt.timedelta(days=100), step=5,
                    ))
                    dates |= set(checker._get_eval_dates(
                        weeks=eval_weeks, step=5, anchor_date=anchor,
                    ))
                    eval_dates = sorted(dates)
                else:
                    eval_dates = checker._get_eval_dates(weeks=eval_weeks, step=5)[:10]
                snapshots, _codes = FactorPanelService().build_snapshot_store(
                    pool, eval_dates, include_tech=True,
                )
            except Exception:
                logger.warning("公开因子共享面板构建失败，回退兼容快照", exc_info=True)
                snapshots = None
        try:
            return adapter.batch_validate(
                source=source,
                eval_weeks=eval_weeks,
                pool=pool,
                top_k=top_k,
                dedupe=dedupe,
                anchor_date=anchor_date,
                snapshots=snapshots,
            )
        finally:
            if hasattr(snapshots, "clear"):
                snapshots.clear()
