"""交易执行 API"""
import anyio
import logging
import math
import os
import uuid
import threading
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, StrictInt, field_validator
from backend.service.qlib_access import run_with_qlib_read_access

router = APIRouter()
logger = logging.getLogger(__name__)

_broker = None
_broker_lock = threading.RLock()
_API_KEY = os.getenv("TRADING_API_KEY", "")


def verify_trading_auth(x_api_key: str = Header(None, alias="X-Api-Key")):
    """交易端点认证——API Key 校验"""
    if not _API_KEY:
        return True  # 未配置 API Key 时允许本地访问 (开发模式)
    if x_api_key and x_api_key == _API_KEY:
        return True
    raise HTTPException(401, "未授权: 缺少有效 API Key (X-Api-Key)")


def _make_price_provider():
    def price_provider(symbol):
        try:
            from datetime import date, datetime
            from decimal import Decimal, ROUND_HALF_UP
            from backend.service.market_data import MarketDataService
            from core.data import Database
            from core.shared.market_rules import limit_pct_for_code

            response = MarketDataService.get_live_quotes([symbol], use_case="execution")
            quote = (response.get("data") or {}).get(symbol) or {}
            raw_price = quote.get("price") or quote.get("last_price") or 0
            result = {"price": float(raw_price or 0), "limit_up": None, "limit_down": None}
            if response.get("meta", {}).get("freshness") != "fresh":
                return {"price": 0, "limit_up": None, "limit_down": None}

            try:
                prev_close = Decimal(str(quote.get("prev_close")))
                name = str(quote.get("name") or "")
                code = str(symbol).split(".")[0]
                identity = Database.fetchone(
                    "SELECT list_status,list_date,delist_date FROM stock_identity WHERE symbol=%s LIMIT 1",
                    (code,),
                )
                if (not identity or str(identity.get("list_status") or "").upper() != "L" or
                        identity.get("delist_date") or not identity.get("list_date") or "退" in name):
                    return result
                listed = identity["list_date"]
                listed_date = listed.date() if isinstance(listed, datetime) else (
                    listed if isinstance(listed, date) else date.fromisoformat(str(listed)[:10]))
                if (date.today() - listed_date).days < 30 or not prev_close.is_finite() or prev_close <= 0:
                    return result
                rate = Decimal(str(limit_pct_for_code(code, is_st="ST" in name.upper())))
                result["limit_up"] = (prev_close * (1 + rate)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                result["limit_down"] = (prev_close * (1 - rate)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            except Exception:
                return result
            return result
        except Exception:
            return {"price": 0, "limit_up": None, "limit_down": None}
    return price_provider


def get_broker():
    with _broker_lock:
        return _get_broker_locked()


def _get_broker_locked():
    global _broker
    from core.shared.config import (BROKER_TYPE, BROKER_QMT, BROKER_FAKE, BROKER_EASYT,
                                    QMT_ACCOUNT, EASYTRAIDER_USER, EASYTRAIDER_ACCOUNT_ID)
    from core.adapters.broker import FakeBroker, QmtBroker, EasytraderBroker, create_broker
    stale_broker = _broker is not None and not (
        (BROKER_TYPE == BROKER_FAKE and isinstance(_broker, FakeBroker)) or
        (BROKER_TYPE == BROKER_QMT and QMT_ACCOUNT and isinstance(_broker, QmtBroker)
         and _broker.account_id == f"QMT-{QMT_ACCOUNT}") or
        (BROKER_TYPE == BROKER_EASYT and EASYTRAIDER_USER and isinstance(_broker, EasytraderBroker)
         and _broker._user == EASYTRAIDER_USER
         and _broker.account_id == f"EASYT-{str(EASYTRAIDER_ACCOUNT_ID or '').strip()}")
    )
    if stale_broker:
        try:
            _broker.disconnect()
        except Exception as error:
            logger.warning("旧 Broker 断开失败: %s", error)
        _broker = None
    if _broker is not None:
        if _broker.connected is not True:
            logger.warning("Broker 连接已断开，尝试重连...")
            try:
                ok = _broker.connect()
                if ok is True and _broker.connected is True:
                    logger.info("Broker 重连成功")
                else:
                    logger.warning("Broker 重连失败")
            except Exception as e:
                logger.warning(f"Broker 重连异常: {e}")
                ok = False
            if ok is not True or _broker.connected is not True:
                try:
                    _broker.disconnect()
                except Exception as error:
                    logger.warning("Broker 失败会话清理异常: %s", error)
                _broker = None
                if BROKER_TYPE == BROKER_FAKE:
                    raise RuntimeError("模拟券商连接不可用")
                raise RuntimeError("真实券商连接不可用")
        return _broker

    if BROKER_TYPE == BROKER_EASYT:
        from core.shared.config import (EASYTRAIDER_USER, EASYTRAIDER_PASSWORD,
                                        EASYTRAIDER_EXE_PATH, EASYTRAIDER_ACCOUNT_ID)
        if not EASYTRAIDER_USER:
            raise RuntimeError("EASYTRAIDER_USER 未配置，请在 .env 中设置同花顺账号")
        if not EASYTRAIDER_ACCOUNT_ID:
            raise RuntimeError("EASYTRAIDER_ACCOUNT_ID 未配置，拒绝连接未标识的券商账户")
        try:
            broker = create_broker(BROKER_EASYT,
                                   user=EASYTRAIDER_USER,
                                   password=EASYTRAIDER_PASSWORD,
                                   exe_path=EASYTRAIDER_EXE_PATH,
                                   broker_account_id=EASYTRAIDER_ACCOUNT_ID)
        except ValueError as error:
            raise RuntimeError("EASYTRAIDER_ACCOUNT_ID 格式无效") from error
        try:
            ok = broker.connect()
        except Exception as e:
            logger.warning(f"Easytrader 连接失败，实盘交易不可用: {e}")
            ok = False
        if ok is not True or broker.connected is not True:
            logger.warning("Easytrader 连接未成功或状态未就绪")
            try:
                broker.disconnect()
            except Exception as error:
                logger.warning("Easytrader 失败会话清理异常: %s", error)
            _broker = None
            raise RuntimeError("Easytrader 连接不可用")
        _broker = broker

    elif BROKER_TYPE == BROKER_QMT:
        from core.shared.config import QMT_USERDATA_DIR, QMT_ACCOUNT, QMT_SESSION_ID
        if not QMT_ACCOUNT:
            raise RuntimeError("QMT_ACCOUNT 未配置，请在 .env 中设置 QMT 资金账号")
        broker = create_broker(BROKER_QMT,
                                userdata_dir=QMT_USERDATA_DIR,
                                account=QMT_ACCOUNT,
                                session_id=QMT_SESSION_ID)
        connection_error = None
        try:
            ok = broker.connect()
        except Exception as error:
            connection_error = error
            logger.warning("QMT Broker 连接异常，实盘交易不可用: %s", error)
            ok = False
        if ok is not True or broker.connected is not True:
            try:
                broker.disconnect()
            except Exception as error:
                logger.warning("QMT Broker 失败会话清理异常: %s", error)
            logger.warning(
                "QMT Broker 连接失败（miniQMT 可能未启动或未登录），"
                "后续请求将自动重试连接。请确保：\n"
                "  1. 迅投极速交易终端 已启动\n"
                "  2. 账户已登录\n"
                "  3. userdata_mini 路径正确"
            )
            _broker = None
            if str(connection_error or "") == "QMT userdata_mini 目录不存在或不可用":
                raise RuntimeError("QMT userdata_mini 目录不存在或不可用") from connection_error
            raise RuntimeError("QMT 连接不可用")
        _broker = broker

    elif BROKER_TYPE == BROKER_FAKE:
        _broker = create_broker(
            BROKER_FAKE, price_provider=_make_price_provider(), price_limits_configured=True,
        )
        _broker.connect()

        try:
            from core.data import Database
            if Database.is_available():
                row = Database.fetchone(
                    "SELECT * FROM accounts WHERE id=%s", (_broker.account_id,))
                orders = Database.fetchall(
                    "SELECT * FROM orders WHERE account_id=%s ORDER BY created_at,id",
                    (_broker.account_id,),
                )
                trades = Database.fetchall(
                    "SELECT * FROM trades WHERE account_id=%s ORDER BY trade_time,id",
                    (_broker.account_id,),
                )
                expected_identity = (_broker.account_id, "fake", "simulation", "broker")
                if any(tuple(item.get(key) for key in
                               ("account_id", "broker", "environment", "source")) != expected_identity
                       for item in [*orders, *trades]):
                    raise RuntimeError("模拟账户订单或成交身份不一致，拒绝恢复模拟资金和持仓")
                if row:
                    _broker._cash = float(row["cash"])
                    _broker._frozen = float(row["frozen"])
                elif orders or trades:
                    raise RuntimeError("模拟账户资金记录缺失但仍存在委托或成交，拒绝重建")
                else:
                    _broker._save_account()
                _broker.restore(orders, trades)
                _broker._persistence_available = True
        except Exception as error:
            _broker.disconnect()
            _broker = None
            logger.error("模拟账户恢复失败，已暂停模拟交易: %s", error)
            raise RuntimeError("模拟账户状态恢复失败，已暂停模拟交易") from error

    else:
        raise RuntimeError(f"未知券商类型: {BROKER_TYPE}")

    return _broker


def is_fake_broker():
    from core.adapters.broker import is_fake_broker as check
    from core.shared.config import BROKER_TYPE, BROKER_FAKE
    if BROKER_TYPE != BROKER_FAKE:
        return False
    return check(get_broker())


class OrderRequest(BaseModel):
    symbol: str
    name: str = ""
    side: str  # "buy" | "sell"
    quantity: StrictInt
    price_type: str = "market"  # "market" | "limit"
    price: float = 0

    @field_validator("price", mode="before")
    @classmethod
    def validate_order_price(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("价格必须为JSON数字")
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise ValueError("价格必须为有限数值")
        return value


@router.get("/broker")
async def get_broker_info():
    def _run():
        broker = get_broker()
        from core.adapters.broker import is_fake_broker
        from core.shared.config import BROKER_TYPE, BROKER_FAKE
        is_fake = is_fake_broker(broker)
        account_type = broker.account_id.split("-")[0].lower()
        return {
            "type": "easytrader" if account_type == "easyt" else account_type,
            "is_fake": is_fake,
            "connected": broker.connected,
            "account_id": broker.account_id,
            "cash_enabled": is_fake and BROKER_TYPE == BROKER_FAKE and broker.connected,
            "supported_price_types": list(broker.supported_price_types),
        }
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/account")
async def get_account():
    def _run():
        broker = get_broker()
        return broker.get_account()
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/orders")
async def list_orders(status: str = ""):
    def _run():
        broker = get_broker()
        orders = broker.get_orders(status=status or None)
        return list(reversed(orders))
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/orders/{order_id}")
async def get_order(order_id: str):
    def _run():
        broker = get_broker()
        order = broker.get_order(order_id)
        if not order:
            raise HTTPException(404, "订单不存在")
        return order
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.post("/orders")
async def place_order(req: OrderRequest, _auth=Depends(verify_trading_auth)):
    from core.shared.config import BROKER_TYPE, BROKER_FAKE
    if BROKER_TYPE != BROKER_FAKE:
        from core.shared.order_validation import validate_real_limit_order
        try:
            validate_real_limit_order(req.symbol, req.side, req.quantity, req.price_type, req.price)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    else:
        if req.side not in ("buy", "sell"):
            raise HTTPException(400, "side 必须为 buy 或 sell")
        if req.price_type not in ("market", "limit"):
            raise HTTPException(400, "price_type 必须为 market 或 limit")
        if req.quantity <= 0:
            raise HTTPException(400, "数量必须大于 0")

    def _run():
        broker = get_broker()
        if req.price_type not in getattr(type(broker), "supported_price_types", ("limit",)):
            raise HTTPException(422, "当前券商适配器不支持该价格类型")
        from core.adapters.broker import is_fake_broker as check_fake_broker
        if not check_fake_broker(broker):
            from backend.service.local_ledger import LocalLedgerService
            reconciliation = LocalLedgerService.reconcile()
            if not reconciliation["can_execute"]:
                raise HTTPException(409, reconciliation["reason"] or "券商对账未通过")
            raise HTTPException(409, "实盘委托尚未绑定本地账本计划和委托预留，当前禁止直接下单")
        try:
            order_id = broker.place_order(
                req.symbol, req.name, req.side, req.quantity,
                req.price_type, req.price)
        except RuntimeError as e:
            raise HTTPException(503, str(e))

        order = broker.get_order(order_id)
        return order

    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str, _auth=Depends(verify_trading_auth)):
    def _run():
        broker = get_broker()
        from core.adapters.broker import is_fake_broker as check_fake_broker
        if not check_fake_broker(broker):
            raise HTTPException(409, "实盘撤单尚未绑定本地委托意图和券商回报，当前禁止直接撤单")
        ok = broker.cancel_order(order_id)
        if not ok:
            raise HTTPException(400, "订单无法撤销（可能已成交）")
        return True

    try:
        await anyio.to_thread.run_sync(_run)
        return {"status": "ok"}
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/trades")
async def list_trades(symbol: str = ""):
    def _run():
        broker = get_broker()
        trades = broker.get_trades(symbol=symbol or None)
        return list(reversed(trades))
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/positions")
async def get_positions():
    def _run():
        broker = get_broker()
        return broker.get_positions()
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


class CashRequest(BaseModel):
    amount: float


@router.post("/account/cash")
async def add_cash(req: CashRequest, _auth=Depends(verify_trading_auth)):
    if not is_fake_broker():
        raise HTTPException(400, "QMT 实盘不支持模拟出入金")
    def _run():
        broker = get_broker()
        return broker.add_cash(req.amount)
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/account/cash")
async def set_cash(req: CashRequest, _auth=Depends(verify_trading_auth)):
    if not is_fake_broker():
        raise HTTPException(400, "QMT 实盘不支持模拟出入金")
    def _run():
        broker = get_broker()
        return broker.set_cash(req.amount)
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/account/withdraw")
async def withdraw_cash(req: CashRequest, _auth=Depends(verify_trading_auth)):
    if not is_fake_broker():
        raise HTTPException(400, "QMT 实盘不支持模拟出入金")
    def _run():
        broker = get_broker()
        return broker.withdraw_cash(req.amount)
    try:
        return {"data": await anyio.to_thread.run_sync(_run)}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ─── Live Screening (量化实盘调仓) ───

_live_results: dict = {}
_live_lock = threading.Lock()
_MAX_LIVE_RESULTS = 50


class LiveScreeningRequest(BaseModel):
    pool: str = "a500"
    top_n: int = 5
    sizing: str = "score_weighted"
    max_position_pct: float = 0.20
    max_sector_pct: float = 0.30
    stop_loss_pct: float = -0.08
    stop_loss_warn: float = -0.05
    take_profit_half: float = 0.15
    take_profit_full: float = 0.30
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    stamp_duty: float = 0.0005
    use_timing: bool = True
    code_prefixes: list = []


def _run_live_screening(job_id, params):
    import datetime as _dt

    with _live_lock:
        _live_results[job_id] = {"status": "running", "started_at": _dt.datetime.now().isoformat()}

    try:
        from core.shared.trading_time import get_latest_trading_day
        today = get_latest_trading_day()
        today_str = today.isoformat()

        # 1. 获取持仓和账户
        from backend.service.local_ledger import LocalLedgerService

        snapshot = LocalLedgerService.get_advice_snapshot()
        account = snapshot["account"]
        holdings_state = snapshot["holdings"]

        # 3. 择时
        timing_ratio = 1.0
        if params.use_timing:
            try:
                from core.domain.timing.model import get_timing_ratio
                timing_ratio = get_timing_ratio(as_of=today)
                if timing_ratio is None:
                    timing_ratio = 1.0
            except Exception:
                pass

        panic = False
        try:
            from core.domain.portfolio.consistency import check_panic
            panic = check_panic()
        except Exception:
            pass

        # 5. 选股: 走完整 screen() 管线, 与回测一致
        from core.domain.screening._screener import get_screener
        screener = get_screener()
        screen_result = screener.screen(
            pool=params.pool,
            top_n=params.top_n,
            eval_date=today,
            regime="auto",
            skip_debate=True,
            code_prefixes=params.code_prefixes,
            macro_bias=True,
            sector_fusion=None,
            source_weights=None,
            qlib_model=None,
        )
        candidates = screen_result.get("candidates", [])
        regime_result = screen_result.get("regime", {})
        regime_name = regime_result.get("regime", "steady_up")
        active_direction = regime_result.get("direction", "")
        regime_confidence = regime_result.get("confidence", 0.5)

        # 6. 调仓建议
        from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor
        _sl = params.stop_loss_pct or -0.08
        advisor = WeeklyRebalanceAdvisor(
            top_n=params.top_n, top_k=params.top_n * 2,
            pool=params.pool, sizing=params.sizing,
            max_position_pct=params.max_position_pct,
            max_sector_pct=params.max_sector_pct,
            stop_loss_warn=_sl / 2,
            stop_loss_full=_sl,
            commission_rate=params.commission_rate,
            slippage_rate=params.slippage_rate,
            stamp_duty=params.stamp_duty,
        )

        screening_result = {
            "candidates": candidates,
            "regime": regime_result,
            "direction": active_direction,
        }

        action_plan = advisor.generate_advice_from_screening(
            screening_result=screening_result,
            holdings_data=holdings_state,
            account_data=account,
            timing_ratio=timing_ratio,
            panic=panic,
        )

        # 7. 辅助参考数据
        auxiliary = {}
        aux_codes = [c.get("code", "") for c in candidates] + [h["code"] for h in holdings_state]
        if aux_codes:
            try:
                from core.data.fetcher import DataFetcher
                fetcher = DataFetcher()
                for code in list(dict.fromkeys(aux_codes))[:20]:
                    info = {}
                    try:
                        fin = fetcher.get_stock_financial(code)
                        if fin is not None and hasattr(fin, 'iloc') and not fin.empty:
                            pe = fin.iloc[-1].get("市盈率-动态", 0)
                            if pe and pe > 0:
                                info["fundamental"] = f"PE {pe:.0f}"
                    except Exception:
                        pass
                    try:
                        cap = fetcher.get_capital_flow(code)
                        if cap is not None and hasattr(cap, 'iloc') and not cap.empty:
                            latest = cap.iloc[-1]
                            net = latest.get("主力净流入", 0)
                            info["flow"] = "流入" if net > 0 else "流出" if net < 0 else "持平"
                    except Exception:
                        pass
                    if info:
                        auxiliary[code] = info
            except Exception:
                pass

        result = {
            "status": "done",
            "pipeline_context": {
                "regime": regime_result,
                "direction": active_direction,
                "timing_ratio": timing_ratio,
                "panic": panic,
                "eval_date": today_str,
            },
            "screening": {
                "candidates": candidates,
            },
            "current_state": {
                "account": account,
                "holdings": holdings_state,
            },
            "action_plan": action_plan,
            "execution": {"executable": False, "reason": "该建议仅供预览，请从本地账本调仓计划生成可执行信号"},
            "auxiliary": auxiliary,
        }

        with _live_lock:
            _live_results[job_id] = result
            if len(_live_results) > _MAX_LIVE_RESULTS:
                old = sorted(_live_results.keys())[:len(_live_results) - _MAX_LIVE_RESULTS]
                for k in old:
                    del _live_results[k]

    except Exception as e:
        logger.exception(f"Live screening 失败: {e}")
        with _live_lock:
            _live_results[job_id] = {"status": "error", "error": str(e)}


@router.post("/live-screening")
async def start_live_screening(req: LiveScreeningRequest):
    job_id = uuid.uuid4().hex[:12]
    with _live_lock:
        _live_results[job_id] = {"status": "pending"}

    import threading as _th
    t = _th.Thread(target=run_with_qlib_read_access, args=(_run_live_screening, job_id, req), daemon=True)
    t.start()

    return {"data": {"job_id": job_id, "status": "pending"}}


@router.get("/live-screening/{job_id}")
async def get_live_screening(job_id: str):
    with _live_lock:
        result = _live_results.get(job_id)
    if result is None:
        raise HTTPException(404, "任务不存在")
    return {"data": result}
