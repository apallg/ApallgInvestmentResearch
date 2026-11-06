"""通知模块 - 企业微信机器人(主) + Coze Workflow(备用)"""
import json
import logging
import requests
from datetime import datetime
from typing import Optional

from core.shared import config

logger = logging.getLogger(__name__)

WECOM_MARKDOWN_MAX_BYTES = 4096


def _safe_truncate_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    return truncated.decode("utf-8", errors="ignore").rstrip("…") + "…"


class WeComNotifier:
    """企业微信群机器人 Webhook 推送"""

    def __init__(self, key: Optional[str] = None):
        self.key = key or config.WECOM_BOT_KEY
        if self.key:
            if self.key.startswith("http"):
                self._url = self.key
            else:
                self._url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={self.key}"
        else:
            self._url = ""

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def send_markdown(self, content: str) -> bool:
        return self.send_markdown_result(content)["status"] == "sent"

    def send_markdown_result(self, content: str) -> dict:
        if not self.enabled:
            return {"status": "unavailable", "channel": "wecom", "error": "channel_not_configured"}
        try:
            resp = requests.post(
                self._url,
                json={"msgtype": "markdown", "markdown": {"content": content}},
                timeout=10,
            )
            result = resp.json()
            if resp.ok and result.get("errcode") == 0:
                logger.info("企业微信通知发送成功")
                return {"status": "sent", "channel": "wecom", "provider_response": result, "error": None}
            logger.error(f"企业微信返回错误: {result}")
            return {"status": "failed", "channel": "wecom", "provider_response": result, "error": str(result)[:1000]}
        except Exception as e:
            logger.error(f"企业微信通知发送失败: {e}")
            return {"status": "unknown", "channel": "wecom", "error": f"{type(e).__name__}: {e}"[:1000]}

    def send_alert_result(self, stock_name: str, stock_code: str,
                          risk_level: str, risk_score: int,
                          risk_detail: str, suggestion: str,
                          shares: int = 0, cost_price: float = 0) -> dict:
        if not self.enabled:
            return {"status": "unavailable", "error": "channel_not_configured"}

        level_emoji = {"高风险": "🚨", "中风险": "⚠️", "低风险": "ℹ️"}
        emoji = level_emoji.get(risk_level, "🔔")
        color = {"高风险": "warning", "中风险": "comment", "低风险": "info"}.get(risk_level, "comment")
        cost_str = f"¥{cost_price:,.2f}" if cost_price else "未知"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        detail = risk_detail or ""
        suggest = suggestion or "暂无"

        base_len = len(f"## {emoji} A股风险预警\n"
                        f"> 股票: <font color=\"{color}\">{stock_name}({stock_code})</font>\n"
                        f"> 风险等级: **{risk_level}** ({risk_score}/10)\n"
                        f"> 持仓: {shares}股 | 成本: {cost_str}\n"
                        f"> 风险详情: \n"
                        f"> 建议: \n"
                        f"> 时间: {now}\n".encode("utf-8"))

        remaining = WECOM_MARKDOWN_MAX_BYTES - base_len - 100
        detail_budget = int(remaining * 0.75)
        suggest_budget = remaining - detail_budget

        detail = _safe_truncate_bytes(detail, max(detail_budget, 200))
        suggest = _safe_truncate_bytes(suggest, max(suggest_budget, 80))

        content = (
            f"## {emoji} A股风险预警\n"
            f"> 股票: <font color=\"{color}\">{stock_name}({stock_code})</font>\n"
            f"> 风险等级: **{risk_level}** ({risk_score}/10)\n"
            f"> 持仓: {shares}股 | 成本: {cost_str}\n"
            f"> 风险详情: {detail}\n"
            f"> 建议: {suggest}\n"
            f"> 时间: {now}\n"
        )
        content = _safe_truncate_bytes(content, WECOM_MARKDOWN_MAX_BYTES)
        return self.send_markdown_result(content)

    def send_alert(self, stock_name: str, stock_code: str,
                   risk_level: str, risk_score: int,
                   risk_detail: str, suggestion: str,
                   shares: int = 0, cost_price: float = 0) -> bool:
        return self.send_alert_result(
            stock_name, stock_code, risk_level, risk_score, risk_detail,
            suggestion, shares, cost_price,
        )["status"] == "sent"

    def send_test(self) -> bool:
        return self.send_markdown(
            "## ✅ Apallg投研 测试通知\n"
            f"> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "> 如收到此消息，说明企业微信通知通道已正常工作\n"
        )


class CozeNotifier:
    """Coze Workflow API 推送通知（备用）"""

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None,
                 workflow_id: Optional[str] = None, bot_id: Optional[str] = None):
        self.token = token or config.COZE_API_TOKEN
        self.base_url = base_url or config.COZE_BASE_URL
        self.workflow_id = workflow_id or config.COZE_WORKFLOW_ID
        self.bot_id = bot_id or config.COZE_BOT_ID

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.workflow_id)

    def send_alert_result(self, stock_name: str, stock_code: str,
                          risk_level: str, risk_score: int,
                          risk_detail: str, suggestion: str,
                          shares: int = 0, cost_price: float = 0) -> dict:
        if not self.enabled:
            logger.debug("Coze 未配置，跳过通知")
            return {"status": "unavailable", "channel": "coze", "error": "channel_not_configured"}

        level_emoji = {"高风险": "🚨", "中风险": "⚠️", "低风险": "ℹ️"}
        emoji = level_emoji.get(risk_level, "🔔")
        cost_str = f"¥{cost_price:,.2f}" if cost_price else "未知"

        payload = {
            "workflow_id": self.workflow_id,
            "parameters": {
                "title": f"{emoji}【A股风险预警】",
                "stock_name": stock_name,
                "stock_code": stock_code,
                "shares": str(shares),
                "cost_price": cost_str,
                "risk_level": risk_level,
                "risk_score": str(risk_score),
                "risk_detail": risk_detail,
                "suggestion": suggestion,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "bot_id": self.bot_id or "",
        }

        try:
            resp = requests.post(
                f"{self.base_url}/v1/workflow/run",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=15,
            )
            result = resp.json()
            if resp.ok and result.get("code") == 0:
                logger.info(f"Coze 告警发送成功: {stock_name}({stock_code})")
                return {"status": "sent", "channel": "coze", "provider_response": result, "error": None}
            logger.error(f"Coze 返回错误: {result}")
            return {"status": "failed", "channel": "coze", "provider_response": result, "error": str(result)[:1000]}
        except Exception as e:
            logger.error(f"Coze 通知发送失败: {e}")
            return {"status": "unknown", "channel": "coze", "error": f"{type(e).__name__}: {e}"[:1000]}

    def send_alert(self, stock_name: str, stock_code: str,
                   risk_level: str, risk_score: int,
                   risk_detail: str, suggestion: str,
                   shares: int = 0, cost_price: float = 0) -> bool:
        return self.send_alert_result(
            stock_name, stock_code, risk_level, risk_score, risk_detail,
            suggestion, shares, cost_price,
        )["status"] == "sent"

    def send_test(self) -> bool:
        return self.send_alert(
            stock_name="测试股票", stock_code="000000",
            risk_level="低风险", risk_score=1,
            risk_detail="Apallg投研 Coze 通道测试通知",
            suggestion="如收到此消息，说明 Coze 通知通道已正常工作",
        )


_wecom: Optional[WeComNotifier] = None
_coze: Optional[CozeNotifier] = None


def get_wecom() -> WeComNotifier:
    global _wecom
    if _wecom is None:
        _wecom = WeComNotifier()
    return _wecom


def get_coze() -> CozeNotifier:
    global _coze
    if _coze is None:
        _coze = CozeNotifier()
    return _coze


def get_channel_status() -> dict:
    return {
        "wecom": get_wecom().enabled,
        "coze": get_coze().enabled,
    }


def send_risk_alert(stock_name: str, stock_code: str,
                    risk_level: str, risk_score: int,
                    risk_detail: str, suggestion: str,
                    shares: int = 0, cost_price: float = 0) -> bool:
    return send_risk_alert_result(
        stock_name, stock_code, risk_level, risk_score, risk_detail,
        suggestion, shares, cost_price,
    )["status"] == "sent"


def send_risk_alert_result(stock_name: str, stock_code: str,
                           risk_level: str, risk_score: int,
                           risk_detail: str, suggestion: str,
                           shares: int = 0, cost_price: float = 0) -> dict:
    """发送风控告警并区分失败、未配置和结果不确定"""
    wecom = get_wecom()
    if wecom.enabled:
        return wecom.send_alert_result(stock_name, stock_code, risk_level, risk_score,
                                       risk_detail, suggestion, shares, cost_price)
    coze = get_coze()
    if coze.enabled:
        return coze.send_alert_result(stock_name, stock_code, risk_level, risk_score,
                                      risk_detail, suggestion, shares, cost_price)
    logger.warning("未配置任何通知通道，跳过告警推送")
    return {"status": "unavailable", "channel": None, "error": "no_notification_channel_configured"}


def send_test_notification() -> dict:
    """发送测试通知到已配置的通道，返回各通道结果"""
    result = {}
    wecom = get_wecom()
    if wecom.enabled:
        result["wecom"] = wecom.send_test()
    coze = get_coze()
    if coze.enabled:
        result["coze"] = coze.send_test()
    if not result:
        result["error"] = "未配置任何通知通道（WECOM_BOT_KEY 或 COZE_API_TOKEN）"
    return result
