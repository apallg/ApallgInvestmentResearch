"""持仓管理 API"""
from decimal import Decimal

import anyio
from fastapi import APIRouter, HTTPException, Body, Header
from pydantic import BaseModel, StrictBool, StrictInt, field_validator

router = APIRouter()


class HoldingCreate(BaseModel):
    code: str
    name: str
    shares: StrictInt
    cost_price: Decimal
    buy_date: str = ""
    alerts_enabled: StrictBool = True
    risk_threshold: StrictInt = 7

    @field_validator("cost_price", mode="before")
    @classmethod
    def reject_boolean_cost_price(cls, value):
        if isinstance(value, bool):
            raise ValueError("成本价必须为数值")
        if isinstance(value, float):
            raise ValueError("成本价请以十进制字符串提交，避免精度丢失")
        return value


@router.get("/holdings")
async def list_holdings():
    from core.data import HoldingsRepo
    from core.data.repos.local_ledger import ACCOUNT_ID
    result = await anyio.to_thread.run_sync(HoldingsRepo.get_all, ACCOUNT_ID)
    return {"data": result}


@router.get("/holdings/adjustments")
async def list_holding_adjustments(code: str = "", limit: int = 100):
    from core.data import HoldingsRepo
    from core.data.repos.local_ledger import ACCOUNT_ID
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit必须为1到500")
    try:
        rows = await anyio.to_thread.run_sync(HoldingsRepo.get_adjustments, code or None, limit, ACCOUNT_ID)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"data": rows}


@router.post("/holdings")
async def add_holding(h: HoldingCreate, authorization: str | None = Header(None)):
    from backend.api.auth import require_admin_actor
    require_admin_actor(authorization)

    def _run():
        from core.data import HoldingsRepo, Database
        if not Database.is_available():
            return False
        HoldingsRepo.add(h.code, h.name, h.shares, h.cost_price,
                         h.buy_date, h.alerts_enabled, h.risk_threshold)
        return True
    try:
        ok = await anyio.to_thread.run_sync(_run)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        import pymysql
        if isinstance(e, pymysql.err.IntegrityError) and e.args and e.args[0] == 1062:
            raise HTTPException(409, "该股票已存在，请使用编辑操作") from e
        raise
    if not ok:
        raise HTTPException(503, "数据库未连接")
    return {"status": "ok"}


@router.put("/holdings/{code}")
async def update_holding(code: str, data: dict, authorization: str | None = Header(None)):
    """更新持仓：risk_threshold, alerts_enabled 等"""
    from backend.api.auth import require_admin_actor
    require_admin_actor(authorization)

    from core.data import HoldingsRepo
    from core.data.repos.holdings import StaleHoldingError
    if isinstance(data.get("cost_price"), float):
        raise HTTPException(422, "成本价请以十进制字符串提交，避免精度丢失")
    try:
        updated = await anyio.to_thread.run_sync(lambda: HoldingsRepo.update(code, **data))
    except StaleHoldingError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if not updated:
        raise HTTPException(404, "持仓不存在")
    return {"status": "ok"}


@router.post("/holdings/prices")
async def get_holdings_prices(data: dict):
    """批量获取持仓股票的当前价格"""
    codes = data.get("codes", [])
    from backend.service.market_data import MarketDataService
    response = await anyio.to_thread.run_sync(MarketDataService.get_live_quotes, codes, "live_display")
    quotes = response.get("data", {})
    meta = dict(response.get("meta", {}))
    prices = {}
    for code, quote in quotes.items():
        if quote:
            price = quote.get("price") or quote.get("last_price")
        else:
            price = None
        if price:
            prices[code] = {
                "price": price,
                "pct_chg": quote.get("pct_chg"),
                "source": quote.get("source"),
                "observed_at": quote.get("observed_at"),
                "age_seconds": meta.get("quote_statuses", {}).get(code, {}).get("age_seconds"),
                "price_basis": meta.get("price_basis"),
            }
    meta["expected_count"] = meta.get("coverage", {}).get("requested", len(set(codes)))
    meta["priced_count"] = len(prices)
    meta["coverage_rate"] = round(len(prices) / meta["expected_count"], 4) if meta["expected_count"] else 0.0
    return {"data": prices, "meta": meta}


@router.delete("/holdings/{code}")
async def delete_holding(code: str, data: dict = Body(...), authorization: str | None = Header(None)):
    from backend.api.auth import require_admin_actor
    require_admin_actor(authorization)

    from core.data import HoldingsRepo
    from core.data.repos.holdings import StaleHoldingError
    try:
        deleted = await anyio.to_thread.run_sync(
            lambda: HoldingsRepo.delete(code, reason=data.get("reason"),
                                        expected_shares=data.get("expected_shares"),
                                        expected_cost_price=data.get("expected_cost_price"),
                                        expected_buy_date=data.get("expected_buy_date"),
                                        expected_id=data.get("expected_id"),
                                        expected_version=data.get("expected_version"))
        )
    except StaleHoldingError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if not deleted:
        raise HTTPException(404, "持仓不存在")
    return {"status": "ok"}
