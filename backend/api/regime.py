"""市场状态识别 API"""
import logging
import anyio
from fastapi import APIRouter, Body, HTTPException, Query
from core.domain.regime.classifier import get_current_regime
from core.domain.regime.weights import (get_regime_weights, get_regime_tech_preset,
                                  REGIME_LABELS, REGIME_RATIONALE,
                                  get_regime_direction)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/history/rebuild")
def rebuild_regime_history(payload: dict = Body(default={})):
    try:
        from backend.service.regime_history import regime_history_service
        return {"data": regime_history_service.create_rebuild(payload)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/history/rebuild/{job_id}")
def rebuild_regime_history_status(job_id: str):
    from core.shared.infrastructure.async_task import task_manager
    task = task_manager.get_dict(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="历史波动状态任务不存在")
    return {"data": task}


@router.get("/current")
async def current_regime(apply_macro: bool = Query(True), smooth: bool = Query(True)):
    """当前市场状态 + 推荐权重"""
    def _do():
        rr = get_current_regime(apply_macro=apply_macro, smooth=smooth)
        weights = get_regime_weights(rr.regime)
        tech_preset = get_regime_tech_preset(rr.regime)
        direction = get_regime_direction(rr.regime)
        dir_count = 0
        try:
            from core.domain.factors.registry import get_pool
            curated = get_pool().get_curated_set(direction)
            if curated:
                dir_count = curated.get("stats", {}).get("total", len(curated.get("flat_list", [])))
        except Exception:
            pass
        return {
            "regime": rr.regime.value,
            "label": REGIME_LABELS.get(rr.regime, rr.regime.value),
            "confidence": rr.confidence,
            "sub_scores": rr.sub_scores,
            "details": rr.details,
            "as_of": rr.as_of_date,
            "suggested_weights": weights,
            "suggested_tech_preset": tech_preset,
            "rationale": REGIME_RATIONALE.get(rr.regime, ""),
            "direction": direction,
            "direction_factors": dir_count,
        }
    try:
        result = await anyio.to_thread.run_sync(_do)
        return {"data": result}
    except Exception as e:
        logger.error(f"获取当前状态失败: {e}")
        return {"data": {"error": str(e)}}
