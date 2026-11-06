"""风险告警 API"""
from typing import Optional, Dict
from functools import partial

import anyio
from fastapi import APIRouter, HTTPException, Body, Header

router = APIRouter()


@router.get("/")
async def list_alerts(limit: int = 50, level: str = ""):
    from core.data import RiskAlertRepo
    alerts = await anyio.to_thread.run_sync(RiskAlertRepo.get_recent, limit)
    if level:
        alerts = [a for a in alerts if a["risk_level"] == level]
    return {"data": alerts}


@router.get("/stats")
async def alert_stats(limit: int = 200):
    from core.data import RiskAlertRepo
    alerts = await anyio.to_thread.run_sync(RiskAlertRepo.get_recent, limit)
    stats = {"high": 0, "mid": 0, "low": 0}
    for a in alerts:
        level = a.get("risk_level", "")
        if level == "高风险":
            stats["high"] += 1
        elif level == "中风险":
            stats["mid"] += 1
        else:
            stats["low"] += 1
    return {"data": stats}


@router.get("/channels")
async def notification_channels():
    from core.shared.infrastructure.notify import get_channel_status
    result = await anyio.to_thread.run_sync(get_channel_status)
    return {"data": result}


@router.post("/test")
async def test_notification():
    from core.shared.infrastructure.notify import send_test_notification
    result = await anyio.to_thread.run_sync(send_test_notification)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"data": result}


@router.post("/scan/{code}")
async def scan_single(code: str, threshold: int = Body(0), include: Optional[Dict[str, bool]] = Body(None)):
    """扫描单只持仓，include 可选: {"technical":true,"financial":true,"patterns":true,"realtime":true}"""
    def _run():
        from core.data import HoldingsRepo
        from core.domain.portfolio.manager import PortfolioScanner
        holding = HoldingsRepo.get_by_code(code)
        if not holding:
            return None, f"未找到持仓: {code}"
        scanner = PortfolioScanner()
        result = scanner.scan_holding(holding, include=include)
        if result and result.get("risk_score") is not None and result["risk_score"] >= threshold:
            notify_threshold = holding.get("risk_threshold", 7)
            if notify_threshold is None:
                notify_threshold = 7
            notify = result["risk_score"] >= notify_threshold
            alert_id = scanner.persist_alert(result, notify=notify)
            result["alert_id"] = alert_id
            result["notification_status"] = (
                "pending" if notify and alert_id is not None
                else "unavailable" if notify
                else "not_required"
            )
        return result, None
    try:
        result, error = await anyio.to_thread.run_sync(_run)
        if error:
            raise HTTPException(404, error)
        if result is None:
            return {"data": {
                "risk_level": "未知",
                "risk_score": None,
                "risk_status": "unavailable",
                "message": "扫描未产生结果，可能数据不足",
            }}
        return {"data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"扫描 {code} 失败: {e}")

@router.post("/scan")
async def trigger_scan(threshold: int = Body(5), include: Optional[Dict[str, bool]] = Body(None)):
    """批量扫描全部持仓，include 可选: {"technical":true,"financial":false,...}"""
    from core.domain.portfolio.manager import PortfolioScanner
    try:
        scanner = PortfolioScanner()
        results = await anyio.to_thread.run_sync(partial(scanner.scan_all, threshold=threshold, include=include))
        unscored_count = sum(1 for item in results if item.get("risk_status") == "unknown")
        failed_count = sum(1 for item in results if item.get("risk_status") == "failed")
        skipped_count = sum(1 for item in results if item.get("risk_status") == "disabled")
        risk_count = sum(1 for item in results if item.get("alert_triggered") is True)
        scan_id = results[0].get("scan_id") if results else None
        return {"data": {
            "scan_id": scan_id,
            "total_count": len(results),
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "count": risk_count,
            "unscored_count": unscored_count,
            "results": results,
        }}
    except Exception as e:
        raise HTTPException(500, f"扫描失败: {e}")


@router.get("/outbox")
async def notification_outbox(limit: int = 100):
    from core.data import RiskAlertRepo
    rows = await anyio.to_thread.run_sync(RiskAlertRepo.get_outbox_recent, limit)
    return {"data": rows}


@router.post("/outbox/{outbox_id}/resolve")
async def resolve_notification_outbox(
    outbox_id: int,
    request: dict = Body(...),
    authorization: str | None = Header(None),
):
    from backend.api.auth import require_admin_actor
    from core.data import RiskAlertRepo

    actor = require_admin_actor(authorization)
    resolution = request.get("resolution")
    if resolution not in ("confirmed_sent", "confirmed_not_sent_requeue"):
        raise HTTPException(422, "resolution必须是confirmed_sent或confirmed_not_sent_requeue")
    try:
        result = await anyio.to_thread.run_sync(
            RiskAlertRepo.resolve_unknown_outbox,
            outbox_id,
            resolution,
            actor,
            request.get("reason"),
            request.get("evidence_ref"),
        )
        return {"data": result}
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.get("/outbox/{outbox_id}/resolutions")
async def notification_outbox_resolutions(outbox_id: int):
    from core.data import RiskAlertRepo
    rows = await anyio.to_thread.run_sync(RiskAlertRepo.get_outbox_resolutions, outbox_id)
    return {"data": rows}


@router.get("/scan-runs")
async def risk_scan_runs(limit: int = 50):
    from core.data import RiskAlertRepo
    rows = await anyio.to_thread.run_sync(RiskAlertRepo.get_recent_scans, limit)
    return {"data": rows}
