import datetime as dt
import traceback
import uuid

from core.data.fetcher import DataFetcher
from core.data.repos.regime_history import regime_history_repo
from core.domain.regime.volatility_history import build_volatility_history
from core.shared.infrastructure.async_task import TaskCancelled, task_manager
from core.shared.persistence import today_date


class RegimeHistoryService:
    def create_rebuild(self, request):
        start = dt.date.fromisoformat(str(request.get("start") or "2020-01-01")[:10])
        end = dt.date.fromisoformat(str(request.get("end") or today_date())[:10])
        if start >= end or end > today_date():
            raise ValueError("start/end 必须递增且不能晚于今天")
        job_id = f"volhist{uuid.uuid4().hex[:5]}"
        task_manager.submit_with_id(
            job_id,
            lambda: self._run(job_id, start, end),
            f"逐交易日重建波动状态 {start}~{end}",
            pool="default",
        )
        return {"job_id": job_id, "status": "pending", "start": str(start), "end": str(end)}

    def _run(self, job_id, start, end):
        try:
            regime_history_repo.ensure_schema()
            task_manager.update_progress(job_id, {
                "stage": "load_index", "current": 0, "total": 1,
                "message": "正在加载沪深300完整历史日线",
            })
            fetcher = DataFetcher()
            days = max(1200, (today_date() - start).days + 500)
            frame = fetcher.get_index_daily("沪深300", days=days)
            if frame is None or frame.empty:
                raise RuntimeError("沪深300历史日线为空")
            date_col = "trade_date" if "trade_date" in frame.columns else "date"
            earliest = dt.date.fromisoformat(str(frame[date_col].min())[:10])
            required_lookback = start - dt.timedelta(days=450)
            if earliest > required_lookback:
                raise RuntimeError(f"指数历史覆盖不足: 最早{earliest}, 至少需要{required_lookback}")
            task_manager.update_progress(job_id, {
                "stage": "compute", "current": 0, "total": len(frame),
                "source": fetcher.get_sources().get("index_daily"),
                "message": "正在按时点计算20日波动率与252日分位数",
            })
            rows = build_volatility_history(frame, start=start, end=end)
            if not rows:
                raise RuntimeError("没有可写入的历史波动状态")
            high_count = sum(row["vol_state"] == "high_vol" for row in rows)
            low_count = sum(row["vol_state"] == "low_vol" for row in rows)
            if not high_count or not low_count:
                raise RuntimeError("历史状态未同时包含high_vol和low_vol")
            written = 0
            for index in range(0, len(rows), 500):
                if task_manager.is_cancel_requested(job_id):
                    raise TaskCancelled(job_id)
                written += regime_history_repo.upsert(rows[index:index + 500])
                task_manager.update_progress(job_id, {
                    "stage": "persist", "current": written, "total": len(rows),
                    "high_vol_count": high_count, "low_vol_count": low_count,
                    "checkpoint": "market_regime_history",
                    "message": "正在逐批写入历史波动状态",
                })
            result = {
                "written": written,
                "start": str(start), "end": str(end),
                "source": fetcher.get_sources().get("index_daily"),
                "method": "rule_based_pit_v1",
                "high_vol_count": high_count,
                "low_vol_count": low_count,
                "database_summary": regime_history_repo.summary(start, end),
            }
            task_manager.update_progress(job_id, {
                "stage": "completed", "current": written, "total": written,
                "message": "逐交易日历史波动状态重建完成",
            })
            return result
        except TaskCancelled:
            raise
        except Exception:
            raise RuntimeError(traceback.format_exc())


regime_history_service = RegimeHistoryService()
