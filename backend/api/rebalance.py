"""调仓信号 API"""
import logging
import math
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import List, Literal, Optional
from uuid import UUID

import anyio
from fastapi import APIRouter, HTTPException, Body, Header
from pydantic import BaseModel, Field, StrictInt

router = APIRouter()
logger = logging.getLogger(__name__)


class T0CheckRequest(BaseModel):
    codes: List[str] = Field(..., min_length=1, max_length=200)


class OpeningRequest(BaseModel):
    opening_cash: Decimal
    holdings_fingerprint: str
    reason: str


class BrokerBindingRequest(BaseModel):
    expected_account_version: StrictInt


class OrderIntentRequest(BaseModel):
    intent_id: str
    signal_id: StrictInt
    expected_account_version: StrictInt
    quantity: StrictInt
    limit_price: Decimal
    fee_cap: Decimal


class BrokerReportResolutionRequest(BaseModel):
    reason: str


class BrokerOrderAttachRequest(BaseModel):
    broker_order_id: str


class BrokerFillRequest(BaseModel):
    broker_trade_id: str
    commission: Decimal
    stamp_duty: Decimal


class CloseRemainderRequest(BaseModel):
    expected_account_version: StrictInt
    reason: str


class ManualFillRequest(BaseModel):
    shares: StrictInt
    price: Decimal
    request_id: UUID
    expected_account_version: StrictInt
    commission: Decimal
    stamp_duty: Decimal
    broker_trade_id: Optional[str] = None
    date: Optional[str] = None
    notes: Optional[str] = None


class HistoricalFillRequest(BaseModel):
    request_id: UUID
    trade_date: date
    code: str = Field(min_length=6, max_length=6)
    side: Literal["buy", "sell"]
    shares: StrictInt = Field(gt=0, le=2147483647)
    price: Decimal
    expected_account_version: StrictInt = Field(gt=0, le=2147483647)
    commission: Decimal
    stamp_duty: Decimal
    broker_trade_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=8, max_length=500)
    source_signal_id: Optional[StrictInt] = Field(None, gt=0, le=2147483647)
    name: Optional[str] = Field(None, max_length=128)
    allow_odd_lot_sell: bool = False


class BatchManualFillItem(BaseModel):
    model_config = {"extra": "forbid"}

    signal_id: StrictInt = Field(gt=0, le=2147483647)
    shares: StrictInt = Field(gt=0, le=2147483647)
    price: Decimal
    request_id: UUID
    expected_account_version: StrictInt = Field(gt=0, le=2147483647)
    commission: Decimal
    stamp_duty: Decimal
    broker_trade_id: Optional[str] = None
    date: Optional[str] = None
    notes: Optional[str] = None


class BatchManualFillRequest(BaseModel):
    model_config = {"extra": "forbid"}

    signal_ids: List[BatchManualFillItem] = Field(..., max_length=200)


@router.get("/account/local")
async def get_local_account():
    from backend.service.local_ledger import LocalLedgerService
    try:
        result = await anyio.to_thread.run_sync(LocalLedgerService.get_account)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.post("/account/local/historical-fills")
async def record_local_historical_fill(
    req: HistoricalFillRequest,
    authorization: Optional[str] = Header(None),
):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    actor = require_admin_actor(authorization)
    detail = req.model_dump(mode="json")
    detail["request_id"] = str(req.request_id)
    detail["trade_date"] = req.trade_date.isoformat()
    detail["expected_account_version"] = req.expected_account_version
    detail["shares"] = req.shares
    detail["actor"] = actor
    try:
        result = await anyio.to_thread.run_sync(LocalLedgerService.record_historical_fill, detail)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        logger.exception("历史成交补录失败")
        raise HTTPException(503, "历史成交补录失败，账本事务已回滚") from error
    return {"data": result}


@router.post("/account/local/activate")
async def activate_local_account(req: OpeningRequest, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_ledger import LocalLedgerRepo

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalLedgerRepo.activate, req.opening_cash, req.holdings_fingerprint, actor, req.reason
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.get("/account/local/reconciliation")
async def reconcile_local_account():
    from backend.service.local_ledger import LocalLedgerService
    return {"data": await anyio.to_thread.run_sync(LocalLedgerService.reconcile)}


@router.post("/account/local/bind-broker")
async def bind_local_broker(req: BrokerBindingRequest, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalLedgerService.bind_broker, req.expected_account_version, actor
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.get("/account/local/intents")
async def list_local_intents(limit: int = 100, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_order_intent import LocalOrderIntentRepo

    require_admin_actor(authorization)
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit 必须介于1和500之间")
    return {"data": await anyio.to_thread.run_sync(LocalOrderIntentRepo.list_recent, limit)}


@router.post("/account/local/intents")
async def prepare_local_intent(req: OrderIntentRequest, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalLedgerService.prepare_intent, {**req.model_dump(), "actor": actor}
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.post("/account/local/intents/{intent_id}/cancel")
async def cancel_local_intent(intent_id: str, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_order_intent import LocalOrderIntentRepo

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(LocalOrderIntentRepo.cancel_prepared, intent_id, actor)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.post("/account/local/intents/{intent_id}/attach-order")
async def attach_local_broker_order(intent_id: str, req: BrokerOrderAttachRequest,
                                    authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_order_intent import LocalOrderIntentRepo

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalOrderIntentRepo.attach_broker_order, intent_id, req.broker_order_id, actor
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.get("/account/local/intents/{intent_id}/broker-evidence")
async def get_local_broker_evidence(intent_id: str, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_order_intent import LocalOrderIntentRepo

    require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(LocalOrderIntentRepo.get_broker_evidence, intent_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"data": result}


@router.post("/account/local/intents/{intent_id}/broker-fill")
async def confirm_local_broker_fill(intent_id: str, req: BrokerFillRequest,
                                    authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_order_intent import LocalOrderIntentRepo
    from core.domain.portfolio.decision import decision_fingerprint

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalOrderIntentRepo.record_broker_fill, intent_id, req.broker_trade_id,
            req.commission, req.stamp_duty, actor, decision_fingerprint,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"data": result}


@router.post("/account/local/intents/{intent_id}/finalize")
async def finalize_local_broker_order(intent_id: str, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_order_intent import LocalOrderIntentRepo

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(LocalOrderIntentRepo.finalize_terminal, intent_id, actor)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"data": result}


@router.get("/account/local/broker-reports/unresolved")
async def list_unresolved_broker_reports(limit: int = 100, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    require_admin_actor(authorization)
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit 必须介于1和500之间")
    try:
        result = await anyio.to_thread.run_sync(LocalLedgerService.list_unresolved_broker_reports, limit)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.post("/account/local/broker-reports/replay")
async def replay_broker_reports(authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(LocalLedgerService.replay_broker_reports)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.post("/account/local/broker-reports/{report_key}/resolve")
async def resolve_broker_report(report_key: str, req: BrokerReportResolutionRequest,
                                authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalLedgerService.resolve_broker_report, report_key, actor, req.reason
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.get("/account/local/broker-trades/unresolved")
async def list_unresolved_broker_trades(limit: int = 100, authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    require_admin_actor(authorization)
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit 必须介于1和500之间")
    try:
        result = await anyio.to_thread.run_sync(LocalLedgerService.list_unresolved_broker_trades, limit)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.post("/account/local/broker-trades/{trade_id}/resolve")
async def resolve_broker_trade_conflicts(trade_id: str, req: BrokerReportResolutionRequest,
                                         authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalLedgerService.resolve_broker_trade_conflicts, trade_id, actor, req.reason,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.get("/account/local/broker-report-history/{report_type}/{report_key}")
async def list_broker_report_history(report_type: str, report_key: str, limit: int = 100,
                                     authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    require_admin_actor(authorization)
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit 必须介于1和500之间")
    try:
        result = await anyio.to_thread.run_sync(
            LocalLedgerService.list_broker_report_history, report_type, report_key, limit,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.get("/account/local/quarantine")
async def get_local_quarantine(limit: int = 100, offset: int = 0,
                               authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from core.data.repos.local_ledger import LocalLedgerRepo

    require_admin_actor(authorization)
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit 必须介于1和500之间")
    if offset < 0:
        raise HTTPException(422, "offset 不能为负数")
    return {"data": await anyio.to_thread.run_sync(LocalLedgerRepo.get_quarantine, limit, offset)}


@router.post("/daily-check")
async def daily_risk_check(
    stop_loss_warn: Optional[float] = Body(None),
    stop_loss_full: Optional[float] = Body(None),
    take_profit_half: Optional[float] = Body(None),
    take_profit_full: Optional[float] = Body(None),
):
    from backend.service.rebalance import RebalanceService

    def _run():
        regime_info = None
        try:
            from core.domain.regime.classifier import get_current_regime
            raw = get_current_regime()
            if raw is not None:
                regime_info = raw.to_dict() if hasattr(raw, 'to_dict') else raw
        except Exception:
            pass
        return RebalanceService.daily_risk_check(
            stop_loss_warn=stop_loss_warn,
            stop_loss_full=stop_loss_full,
            take_profit_half=take_profit_half,
            take_profit_full=take_profit_full,
            regime_info=regime_info,
        )

    try:
        with anyio.fail_after(30):
            result = await anyio.to_thread.run_sync(_run, abandon_on_cancel=True)
    except TimeoutError:
        return {"data": {"status": "timeout", "message": "风控检查超时，请稍后重试"}}
    return {"data": result}


@router.post("/run")
async def run_rebalance(
    screening_save_id: int = Body(...),
    top_n: int = Body(5),
    sizing: str = Body("equal"),
    frequency: str = Body("weekly"),
    code_prefixes: Optional[list] = Body(None),
    stop_loss_pct: Optional[float] = Body(None),
    take_profit_half: Optional[float] = Body(None),
    take_profit_full: Optional[float] = Body(None),
    cooldown_stop_weeks: Optional[int] = Body(None),
    cooldown_profit_weeks: Optional[int] = Body(None),
    cash: Optional[float] = Body(None),
    decision_id: Optional[str] = Body(None),
    policy_id: str = Body("legacy"),
    profile_id: Optional[str] = Body(None),
    profile_version: Optional[int] = Body(None),
    profile_overrides: Optional[dict] = Body(None),
    enforce_strategy_profile: bool = Body(True),
    unmanaged_reason: Optional[str] = Body(None),
    preview_only: bool = Body(True),
    commit_request_id: Optional[str] = Body(None),
    expected_plan_fingerprint: Optional[str] = Body(None),
    authorization: Optional[str] = Header(None),
):
    from backend.service.rebalance import RebalanceService

    unmanaged_actor = None
    if not enforce_strategy_profile:
        from backend.api.auth import require_admin_actor
        unmanaged_actor = require_admin_actor(authorization)
        unmanaged_reason = str(unmanaged_reason or "").strip()
        if len(unmanaged_reason) < 8:
            raise HTTPException(422, "人工接管理由至少需要8个字符")
    if not preview_only and (not commit_request_id or not expected_plan_fingerprint):
        raise HTTPException(422, "提交调仓方案必须携带请求 ID 和预览指纹")

    def _run():
        if profile_id:
            from backend.service.strategy_profile import StrategyProfileService
            from core.data import ScreeningSaveRepo
            profile = StrategyProfileService.resolve(profile_id, profile_version)
            save = ScreeningSaveRepo.get_by_id(screening_save_id)
            if not save or (save.get("profile_id"), int(save.get("profile_version") or 0)) != (profile_id, int(profile_version or 0)):
                return {"error": "调仓请求Profile与选股保存不一致"}
            if save.get("profile_fingerprint") != profile["fingerprint"]:
                return {"error": "调仓请求Profile指纹与选股保存不一致"}
            try:
                StrategyProfileService.bind_operation(profile, "rebalance", profile_overrides)
            except ValueError as error:
                return {"error": str(error)}
        effective_decision_id = decision_id
        if not effective_decision_id:
            from backend.service.portfolio_decision import PortfolioDecisionService
            preview = PortfolioDecisionService.preview(
                screening_save_id, top_n, policy_id, code_prefixes,
            )
            if "error" in preview:
                return preview
            effective_decision_id = preview["decision_id"]
        return RebalanceService.run_rebalance(
            screening_save_id=screening_save_id,
            top_n=top_n,
            sizing=sizing,
            frequency=frequency,
            code_prefixes=code_prefixes,
            stop_loss_pct=stop_loss_pct,
            take_profit_half=take_profit_half,
            take_profit_full=take_profit_full,
            cooldown_stop_weeks=cooldown_stop_weeks,
            cooldown_profit_weeks=cooldown_profit_weeks,
            cash=cash,
            decision_id=effective_decision_id,
            enforce_strategy_profile=enforce_strategy_profile,
            unmanaged_reason=unmanaged_reason,
            unmanaged_actor=unmanaged_actor,
            preview_only=preview_only,
            commit_request_id=commit_request_id,
            expected_plan_fingerprint=expected_plan_fingerprint,
        )

    try:
        with anyio.fail_after(600):
            result = await anyio.to_thread.run_sync(_run, abandon_on_cancel=True)
    except TimeoutError:
        raise HTTPException(504, "调仓计算超时，请稍后重试")
    if "error" in result:
        headers = None
        if not preview_only:
            headers = {"X-Commit-Outcome": result.get("commit_outcome", "not_committed")}
        raise HTTPException(400, result["error"], headers=headers)
    return {"data": result}


@router.post("/decision-preview")
async def preview_decision(
    screening_save_id: int = Body(...),
    top_n: int = Body(5),
    policy_id: str = Body("legacy"),
    selected_codes: Optional[list] = Body(None),
    profile_id: Optional[str] = Body(None),
    profile_version: Optional[int] = Body(None),
):
    from backend.service.portfolio_decision import PortfolioDecisionService

    def _run():
        return PortfolioDecisionService.preview(
            screening_save_id, top_n, policy_id, selected_codes,
            profile_id=profile_id, profile_version=profile_version,
        )

    result = await anyio.to_thread.run_sync(_run)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"data": result}


@router.get("/decisions/{decision_id}")
async def get_decision(decision_id: str):
    from backend.service.portfolio_decision import PortfolioDecisionService

    result = await anyio.to_thread.run_sync(PortfolioDecisionService.get, decision_id)
    if not result:
        raise HTTPException(404, f"决策快照 {decision_id} 不存在")
    return {"data": result}


@router.get("/signals/pending")
async def get_pending_signals(
    check_type: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
):
    from backend.service.rebalance import RebalanceService
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(422, "待办分页参数无效")
    signals = RebalanceService.get_pending_signals(check_type=check_type, limit=limit, offset=offset)
    return {"data": signals}


@router.get("/signals/{signal_id}")
async def get_signal(signal_id: int):
    from core.data import RebalanceSignalRepo
    from core.data.repos.local_ledger import ACCOUNT_ID
    signal = RebalanceSignalRepo.get_by_id(signal_id, account_id=ACCOUNT_ID)
    if not signal:
        raise HTTPException(404, f"信号 #{signal_id} 不存在")
    return {"data": signal}


@router.put("/signals/{signal_id}/execute")
async def execute_signal(
    signal_id: int,
    req: ManualFillRequest,
    authorization: Optional[str] = Header(None),
):
    import anyio
    from backend.api.auth import require_admin_actor
    from backend.service.rebalance import RebalanceService

    actor = require_admin_actor(authorization)
    detail = {"shares": req.shares, "price": str(req.price), "request_id": str(req.request_id),
              "expected_account_version": req.expected_account_version,
              "commission": str(req.commission), "stamp_duty": str(req.stamp_duty), "actor": actor}
    if req.date:
        detail["date"] = req.date
    if req.broker_trade_id is not None:
        detail["broker_trade_id"] = req.broker_trade_id
    if req.notes:
        detail["notes"] = req.notes

    def _run():
        return RebalanceService.mark_executed(signal_id, detail if detail else None)

    result = await anyio.to_thread.run_sync(_run)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"data": result}


@router.put("/signals/{signal_id}/skip")
async def skip_signal(
    signal_id: int,
    reason: str = Body(""),
    authorization: Optional[str] = Header(None),
):
    import anyio
    from backend.api.auth import require_admin_actor
    from backend.service.rebalance import RebalanceService

    require_admin_actor(authorization)
    def _run():
        return RebalanceService.mark_skipped(signal_id, reason)

    result = await anyio.to_thread.run_sync(_run)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"data": result}


@router.post("/signals/{signal_id}/close-remainder")
async def close_signal_remainder(signal_id: int, req: CloseRemainderRequest,
                                 authorization: Optional[str] = Header(None)):
    from backend.api.auth import require_admin_actor
    from backend.service.local_ledger import LocalLedgerService

    actor = require_admin_actor(authorization)
    try:
        result = await anyio.to_thread.run_sync(
            LocalLedgerService.close_signal_remainder,
            signal_id, req.expected_account_version, actor, req.reason,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


@router.put("/signals/batch-execute")
async def batch_execute_signals(
    req: BatchManualFillRequest,
    authorization: Optional[str] = Header(None),
):
    """批量执行 — signal_ids: [{signal_id, shares?, price?, date?}]"""
    import anyio
    from backend.service.rebalance import RebalanceService
    from backend.api.auth import require_admin_actor

    actor = require_admin_actor(authorization)
    request_ids = [item.request_id for item in req.signal_ids]
    if len(request_ids) != len(set(request_ids)):
        raise HTTPException(422, "批量成交的请求 ID 必须唯一")
    signal_details = [{**item.model_dump(mode="json", exclude_none=True), "actor": actor}
                      for item in req.signal_ids]

    def _run():
        return RebalanceService.batch_execute(signal_details)

    result = await anyio.to_thread.run_sync(_run)
    return {"data": result}


@router.get("/history")
async def get_history(
    days: int = 30,
    limit: int = 50,
):
    from backend.service.rebalance import RebalanceService
    history = RebalanceService.get_history(days=days, limit=limit)
    return {"data": history}


@router.get("/batch/{batch_id}")
async def get_batch_signals(
    batch_id: str,
    execution_status: Optional[str] = None,
):
    from backend.service.rebalance import RebalanceService
    signals = RebalanceService.get_batch_signals(batch_id, execution_status=execution_status)
    return {"data": signals}


# ═══ 调仓历史 ═══

@router.get("/sessions")
async def list_sessions(days: int = 90, limit: int = 30):
    from core.data import RebalanceSessionRepo
    from core.data.repos.local_ledger import ACCOUNT_ID
    rows = RebalanceSessionRepo.get_recent(days=days, limit=limit, account_id=ACCOUNT_ID)
    return {"data": rows}


@router.get("/sessions/latest")
async def get_latest_session(check_type: Optional[str] = None):
    from core.data import RebalanceSessionRepo
    from core.data.repos.local_ledger import ACCOUNT_ID
    r = RebalanceSessionRepo.get_latest(check_type=check_type, account_id=ACCOUNT_ID)
    if not r:
        raise HTTPException(404, "无历史记录")
    return {"data": r}


@router.get("/sessions/{batch_id}")
async def get_session(batch_id: str):
    from core.data import RebalanceSessionRepo
    from core.data.repos.local_ledger import ACCOUNT_ID
    r = RebalanceSessionRepo.get_by_batch(batch_id, account_id=ACCOUNT_ID)
    if not r:
        raise HTTPException(404, f"会话 {batch_id} 不存在")
    return {"data": r}


# ═══ 现金余额管理 ═══

@router.get("/cash")
async def get_cash():
    from core.data import SettingsRepo
    cash = SettingsRepo.get("rebalance_cash", 0)
    try:
        cash = float(cash) if cash not in (None, "") else 0.0
    except (TypeError, ValueError):
        raise HTTPException(503, "保存的人工预算无效，请重新设置")
    if not math.isfinite(cash) or cash < 0:
        raise HTTPException(503, "保存的人工预算无效，请重新设置")
    return {"data": {"cash": cash}}


@router.put("/cash")
async def set_cash(cash: float = Body(...)):
    from core.data import SettingsRepo
    if not math.isfinite(cash) or cash < 0:
        raise HTTPException(422, "人工预算必须为有限非负金额")
    SettingsRepo.set("rebalance_cash", cash)
    return {"data": {"cash": cash}}


@router.get("/cooldown")
async def get_cooldown():
    from backend.service.rebalance import RebalanceService

    result = await anyio.to_thread.run_sync(RebalanceService.get_cooldown)
    return {"data": result}


# ═══ 日内回调检测 ═══

@router.post("/intraday-check")
async def intraday_check(
    codes: list = Body(...),
    ma_data: Optional[dict] = Body(None),
    sell_signals: Optional[list] = Body(None),
):
    """日内执行建议 — 买入目标回调检测 + 卖出信号执行决策"""
    from backend.service.rebalance import RebalanceService

    try:
        with anyio.fail_after(30):
            result = await anyio.to_thread.run_sync(
                lambda: RebalanceService.analyze_intraday_entry(
                    codes=codes,
                    ma_data=ma_data,
                    sell_signals=sell_signals,
                ),
                abandon_on_cancel=True,
            )
    except TimeoutError:
        return {"data": {"status": "timeout", "message": "日内检测超时，请稍后重试"}}
    return {"data": result}


@router.post("/t0-check")
async def t0_check(req: T0CheckRequest):
    """T+0 做T建议 — I(激进)/N(保守) 双档，使用本地权威账户快照。"""
    from backend.service.rebalance import RebalanceService

    try:
        with anyio.fail_after(120):
            result = await anyio.to_thread.run_sync(
                lambda: RebalanceService.analyze_t0(
                    codes=req.codes,
                ),
                abandon_on_cancel=True,
            )
    except TimeoutError:
        raise HTTPException(504, "T+0计算超时")
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"data": result}


# ═══ 选股预览 ═══

@router.post("/preview")
async def preview_stocks(
    codes: list = Body(...),
    reference_date: Optional[str] = Body(None),
    stop_mode: str = Body("fixed"),
):
    """
    选股预览 — 批量获取日K线、支撑压力位、建议买卖价位。
    codes: 股票代码列表
    reference_date: 参考日期 (可选)
    stop_mode: "fixed" (固定百分比) 或 "dynamic" (动态支撑压力)
    """
    from backend.service.rebalance import RebalanceService

    try:
        with anyio.fail_after(60):
            result = await anyio.to_thread.run_sync(
                lambda: RebalanceService.preview_stocks(
                    codes=codes,
                    reference_date=reference_date,
                    stop_mode=stop_mode,
                ),
                abandon_on_cancel=True,
            )
    except TimeoutError:
        return {"data": {"status": "timeout", "message": "预览数据获取超时"}}
    return {"data": result}


# ═══ 日内分时线 ═══

@router.get("/intraday-bars/{code}")
async def get_intraday_bars(code: str):
    """获取今日分时数据 (1分钟粒度)"""
    from core.data.ths_collector import THSCollector
    from core.shared.data_contract import make_data_envelope, revalidate_data_envelope
    from core.shared.minute_data import has_complete_minute_sequence, has_verified_minute_semantics

    def _run():
        collector = THSCollector()
        data = collector.fetch_intraday(code)
        if not data:
            return {"code": code, "bars": [], "message": "暂无分时数据", "freshness": "unavailable"}
        from core.shared.trading_time import A_SHARE_TIMEZONE

        now = datetime.now(A_SHARE_TIMEZONE)
        parsed_rows = []
        invalid_timestamp_count = 0
        for row in data:
            try:
                row_time = datetime.fromisoformat(str(row.get("bar_time", "")))
                if row_time.tzinfo is None:
                    row_time = row_time.replace(tzinfo=A_SHARE_TIMEZONE)
                else:
                    row_time = row_time.astimezone(A_SHARE_TIMEZONE)
                if row_time.date() != now.date():
                    invalid_timestamp_count += 1
                    continue
                parsed_rows.append((row_time, row))
            except (TypeError, ValueError):
                invalid_timestamp_count += 1

        source_times = [row_time for row_time, _ in parsed_rows]
        out_of_order = source_times != sorted(source_times)
        timestamp_counts = {}
        for row_time in source_times:
            timestamp_counts[row_time] = timestamp_counts.get(row_time, 0) + 1
        duplicate_times = {row_time for row_time, count in timestamp_counts.items() if count > 1}
        normalized_timed_rows = [
            (row_time, row) for row_time, row in sorted(parsed_rows, key=lambda item: item[0])
            if row_time not in duplicate_times
        ]
        timestamp_sequence_complete = not invalid_timestamp_count and not out_of_order and not duplicate_times
        freshness = "unavailable"
        age_seconds = None
        latest = None
        bars = []
        accepted_bar_times = []
        row_envelopes = [d.get("data_envelope") for d in data]
        shared_envelope = row_envelopes[0] if row_envelopes and all(
            envelope == row_envelopes[0] for envelope in row_envelopes
        ) else None
        semantic_verified = has_verified_minute_semantics(shared_envelope)
        invalid_bar_value_count = 0
        for row_time, d in normalized_timed_rows:
            t = str(d.get("bar_time", ""))
            try:
                price = float(d.get("price"))
                if not math.isfinite(price) or price <= 0:
                    invalid_bar_value_count += 1
                    continue
            except (TypeError, ValueError, OverflowError):
                invalid_bar_value_count += 1
                continue
            try:
                avg_price = float(d.get("avg_price", 0))
                if not math.isfinite(avg_price) or avg_price <= 0:
                    avg_price = None
            except (TypeError, ValueError, OverflowError):
                avg_price = None
            quantity_status = (
                "verified" if semantic_verified and d.get("quantity_status") == "verified" else "unknown"
            )
            volume = None
            amount = None
            if quantity_status == "verified":
                try:
                    volume_value = Decimal(str(d.get("volume")))
                    amount_value = Decimal(str(d.get("amount")))
                    amount_float = float(amount_value)
                    if (not volume_value.is_finite() or not amount_value.is_finite() or
                            volume_value < 0 or amount_value < 0 or
                            volume_value != volume_value.to_integral_value() or
                            ((volume_value == 0) != (amount_value == 0)) or
                            not math.isfinite(amount_float)):
                        raise ValueError("invalid verified minute quantities")
                    volume = int(volume_value)
                    amount = amount_float
                except (InvalidOperation, TypeError, ValueError, OverflowError):
                    quantity_status = "unknown"
            bars.append({
                "time": t.split(" ")[-1][:5] if " " in t else t,
                "price": round(price, 3),
                "volume": int(volume) if volume is not None else None,
                "amount": float(amount) if amount is not None else None,
                "avg_price": round(avg_price, 3) if quantity_status == "verified" and avg_price is not None else None,
                "quantity_status": quantity_status,
            })
            accepted_bar_times.append(row_time)
        latest_dt = max(accepted_bar_times, default=None)
        if latest_dt is not None:
            latest = latest_dt.isoformat(timespec="seconds")
            age_seconds = round((now - latest_dt).total_seconds(), 3)
            if latest_dt.date() != now.date() or age_seconds < -5 or age_seconds > 90:
                freshness = "stale"
            else:
                freshness = "fresh"
        sequence_complete = (
            timestamp_sequence_complete
            and invalid_bar_value_count == 0
            and has_complete_minute_sequence(accepted_bar_times)
        )
        quantity_verified = bool(
            semantic_verified
            and freshness == "fresh"
            and sequence_complete
            and bars
            and all(bar["quantity_status"] == "verified" for bar in bars)
        )
        returned_bars = bars if freshness == "fresh" else []
        envelope_status = "partial" if freshness == "fresh" else freshness
        if semantic_verified:
            data_envelope = dict(shared_envelope)
            if freshness != "fresh":
                data_envelope["status"] = freshness
            elif not quantity_verified or not sequence_complete:
                data_envelope["status"] = "partial"
            data_envelope["quantity_status"] = "verified" if quantity_verified else "unknown"
            data_envelope["coverage"] = {
                "requested": len(data), "accepted": len(returned_bars),
                "missing": len(data) - len(returned_bars),
            }
            if not quantity_verified:
                missing_reason = data_envelope.get("missing_reason") or []
                if isinstance(missing_reason, str):
                    missing_reason = [missing_reason]
                missing_reasons = [*missing_reason]
                if not sequence_complete:
                    missing_reasons.append("minute_sequence_integrity")
                if not semantic_verified or any(bar["quantity_status"] != "verified" for bar in bars):
                    missing_reasons.append("minute_quantity_rows_unverified")
                data_envelope["missing_reason"] = list(dict.fromkeys(missing_reasons))
        else:
            data_envelope = make_data_envelope(
                source="ths_intraday_api", observed_at=latest or None,
                available_at=datetime.now().isoformat(timespec="seconds"),
                as_of=latest[:10] if latest else None, version="unknown",
                unit={"price": "unknown", "volume": "unknown", "amount": "unknown"},
                price_basis="unknown", status="partial" if freshness == "fresh" and not sequence_complete else envelope_status,
                coverage={"requested": len(data), "accepted": len(returned_bars), "missing": len(data) - len(returned_bars)},
                missing_reason=[
                    "price_basis", "volume_unit", "amount_unit", "source_quantity_semantics",
                    *( ["minute_sequence_integrity"] if not sequence_complete else [] ),
                ],
            ) | {"source_quantity_semantics": "unknown"}
            data_envelope["quantity_status"] = "unknown"
        data_envelope = revalidate_data_envelope(data_envelope)
        unit = data_envelope.get("unit") or {}
        return {"code": code, "bars": returned_bars,
                "freshness": freshness, "age_seconds": age_seconds,
                "price_basis": data_envelope.get("price_basis", "unknown"),
                "quantity_status": "verified" if quantity_verified else "unknown",
                "sequence_status": "complete" if sequence_complete else "partial",
                "rejected_bar_count": len(data) - len(returned_bars),
                "semantic_status": "verified" if semantic_verified else "unverified",
                "volume_unit": unit.get("volume"), "amount_unit": unit.get("amount"),
                "data_envelope": data_envelope}

    try:
        with anyio.fail_after(15):
            result = await anyio.to_thread.run_sync(_run, abandon_on_cancel=True)
    except TimeoutError:
        return {"data": {"code": code, "bars": [], "message": "获取超时"}}
    return {"data": result}
