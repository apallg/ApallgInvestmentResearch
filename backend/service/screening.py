"""
选股服务 — 编排快照获取→打分→排序→辩论→存储
"""
import logging
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ScreeningService:
    """选股用例编排器"""

    def __init__(self):
        from core.domain.screening._screener import get_screener
        from core.data import ScreeningSaveRepo
        self._screener = get_screener()
        self._repo = ScreeningSaveRepo

    def run_screening(
        self,
        pool: str = "all",
        top_n: int = 50,
        eval_date: Optional[date] = None,
        regime: str = None,
        source_weights: Optional[Dict] = None,
        save_name: Optional[str] = None,
        job_id: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        snapshot = self._screener.precompute_snapshot(pool=pool, eval_date=eval_date)
        result = self._screener.screen(
            pool=pool, top_n=top_n, snapshot=snapshot,
            regime=regime, source_weights=source_weights,
            eval_date=eval_date, **kwargs,
        )
        if save_name and job_id:
            self._repo.save(
                save_name=save_name, job_id=job_id, pool=pool,
                eval_date=str(eval_date or date.today()),
                candidates=result.get("candidates", []) if isinstance(result, dict) else result,
            )
        return result
