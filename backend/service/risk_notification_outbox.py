import logging

from core.data import RiskAlertRepo

logger = logging.getLogger(__name__)


def dispatch_risk_notification_outbox(limit=20):
    rows = RiskAlertRepo.claim_outbox(limit=limit)
    result = {"claimed": len(rows), "sent": 0, "retry": 0, "failed": 0, "delivery_unknown": 0, "state_update_failed": 0}
    if not rows:
        return result

    from core.shared.infrastructure.notify import get_wecom, send_risk_alert_result
    for row in rows:
        try:
            payload = row["payload"]
            if payload.get("message_type") == "markdown":
                outcome = get_wecom().send_markdown_result(payload.get("content", ""))
            else:
                outcome = send_risk_alert_result(**payload)
        except Exception as exc:
            outcome = {"status": "unknown", "error": f"{type(exc).__name__}: {exc}"}
            logger.exception("风控通知发送异常，outbox_id=%s", row["id"])

        try:
            status = RiskAlertRepo.finish_outbox(row, outcome)
            if status in result:
                result[status] += 1
            else:
                result["failed"] += 1
        except Exception:
            result["state_update_failed"] += 1
            logger.exception("风控通知状态落库失败，outbox_id=%s", row["id"])
    return result
