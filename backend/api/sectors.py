"""板块选股 API"""
import datetime
import anyio
import logging
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from core.domain.screening.stock_scanner import SectorScanner
from core.shared.data_contract import make_data_envelope
from core.shared.persistence import now_iso
from backend.service.market_data import MarketDataService

router = APIRouter()
scanner = SectorScanner()
logger = logging.getLogger(__name__)


async def _load_verified_industry_members(sector_name, snapshot_id=None, limit=None):
    from core.data.repos.stock_industry_history import (
        get_industry_history_snapshot,
        get_industry_snapshot_member_count,
        get_industry_snapshot_members,
        get_latest_available_industry_snapshot,
    )
    from core.shared.trading_time import market_date

    as_of_date = market_date()
    snapshot = (
        await anyio.to_thread.run_sync(get_industry_history_snapshot, snapshot_id)
        if snapshot_id else
        await anyio.to_thread.run_sync(get_latest_available_industry_snapshot, as_of_date.isoformat())
    )
    if not snapshot or snapshot.get("status") != "complete":
        return None, [], 0
    if any(not snapshot.get(field) for field in (
        "query_date", "source_update_date", "available_after_date", "payload_sha256",
        "source_name", "classification_standard",
    )):
        return None, [], 0
    if int(snapshot.get("row_count") or 0) < 2500 or int(snapshot.get("unique_code_count") or 0) < 2500:
        return None, [], 0
    if any(str(snapshot[field])[:10] > as_of_date.isoformat()
           for field in ("query_date", "source_update_date", "available_after_date")):
        return None, [], 0
    snapshot_age = (as_of_date - datetime.date.fromisoformat(str(snapshot["available_after_date"])[:10])).days
    snapshot["membership_status"] = "stale" if snapshot_age > 14 else "available"
    member_count = await anyio.to_thread.run_sync(
        get_industry_snapshot_member_count, snapshot["payload_sha256"], sector_name,
    )
    members = await anyio.to_thread.run_sync(
        get_industry_snapshot_members, snapshot["payload_sha256"], sector_name, limit,
    ) if member_count else []
    return snapshot, members, member_count


@router.get("/{sector_name}/stocks")
async def get_sector_stocks(sector_name: str, snapshot_id: str = Query(None, min_length=32, max_length=32)):
    """获取板块下股票"""
    try:
        snapshot, members, member_count = await _load_verified_industry_members(sector_name, snapshot_id, 50)
        if not snapshot:
            return {
                "data": [],
                "meta": {
                    "status": "unavailable",
                    "sector_name": sector_name,
                    "sector_type": "industry",
                    "membership_envelope": make_data_envelope(source=None, status="unavailable",
                                                               missing_reason="verified_industry_snapshot_missing"),
                },
            }
        if not member_count:
            return {
                "data": [],
                "meta": {
                    "status": "unavailable",
                    "sector_name": sector_name,
                    "sector_type": "industry",
                    "snapshot_id": snapshot["snapshot_id"],
                    "membership_envelope": make_data_envelope(
                        source=snapshot["source_name"],
                        observed_at=snapshot.get("observed_at"),
                        available_at=snapshot["available_after_date"],
                        version=snapshot["snapshot_id"],
                        coverage={"members": 0},
                        status="unavailable",
                        missing_reason="sector_not_in_industry_snapshot",
                    ),
                },
            }
        codes = [str(row.get("code") or "") for row in members]
        quote_result = await anyio.to_thread.run_sync(
            MarketDataService.get_live_quotes, codes, "live_display",
        )
        quotes = quote_result.get("data") or {}
        quote_meta = quote_result.get("meta") or {}
        quote_statuses = quote_meta.get("quote_statuses") or {}
        rows = []
        for row in members:
            code = str(row.get("code") or "")
            quote = quotes.get(code) or {}
            status = quote_statuses.get(code) or {}
            rows.append({
                "rank": 0,
                "code": code,
                "name": row.get("name") or "",
                "price": quote.get("price"),
                "pct_chg": quote.get("pct_chg"),
                "volume": quote.get("volume"),
                "quote_status": status.get("freshness", "unavailable"),
                "missing_reason": status.get("missing_reason"),
            })
        rows.sort(key=lambda row: (
            row["volume"] is None,
            -(float(row["volume"]) if row["volume"] is not None else 0),
            row["code"],
        ))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        received_at = now_iso()
        coverage = dict(quote_meta.get("coverage") or {})
        coverage.update({"members": member_count, "returned": len(rows)})
        accepted = coverage.get("accepted", 0)
        status = "unavailable" if accepted == 0 else "available"
        available_at = snapshot["available_after_date"]
        membership_status = snapshot["membership_status"]
        envelope = make_data_envelope(
            source=quote_meta.get("source"),
            observed_at=quote_meta.get("observed_at"),
            available_at=quote_meta.get("received_at") or received_at,
            as_of=quote_meta.get("data_envelope", {}).get("as_of"),
            unit={"price": "CNY/share", "volume": "share"},
            price_basis="raw",
            coverage=coverage,
            status=status,
            missing_reason={
                code: value.get("missing_reason")
                for code, value in quote_statuses.items()
                if value.get("missing_reason")
            } or ("sector_quotes_unavailable" if accepted == 0 else None),
        )
        return {
            "data": rows,
            "meta": {
                "status": envelope["status"],
                "sector_name": sector_name,
                "sector_type": "industry",
                "snapshot_id": snapshot["snapshot_id"],
                "membership_source": snapshot["source_name"],
                "classification_standard": snapshot["classification_standard"],
                "membership_envelope": make_data_envelope(
                    source=snapshot["source_name"],
                    observed_at=snapshot.get("observed_at"),
                    available_at=available_at,
                    as_of=snapshot.get("query_date"),
                    version=snapshot["snapshot_id"],
                    coverage={"members": member_count, "returned": len(members)},
                    status=membership_status,
                    missing_reason="industry_snapshot_stale" if membership_status == "stale" else None,
                ),
                "quote_source": quote_meta.get("source"),
                "freshness": quote_meta.get("freshness", "unavailable"),
                "coverage": coverage,
                "data_envelope": envelope,
            },
        }
    except Exception as e:
        logger.error(f"板块API异常: {e}")
        received_at = now_iso()
        return JSONResponse(status_code=503, content={
            "data": [],
            "error": "sector_data_unavailable",
            "retry": True,
            "meta": {
                "status": "unavailable",
                "sector_name": sector_name,
                "sector_type": "industry",
                "data_envelope": make_data_envelope(
                    source=None,
                    available_at=received_at,
                    coverage={"members": 0, "requested": 0, "accepted": 0},
                    status="unavailable",
                    missing_reason="sector_data_unavailable",
                ),
            },
        })


@router.get("/{sector_name}/support-resistance")
async def scan_support_resistance(sector_name: str, snapshot_id: str = Query(None, min_length=32, max_length=32)):
    """批量扫描板块内接近支撑位/压力位的股票（布林带+均线法）"""
    try:
        snapshot, members, member_count = await _load_verified_industry_members(sector_name, snapshot_id)
        if not snapshot or not members:
            return JSONResponse(status_code=409, content={"data": [], "error": "verified_industry_snapshot_unavailable"})
        if snapshot["membership_status"] == "stale":
            return JSONResponse(status_code=409, content={"data": [], "error": "industry_membership_snapshot_stale",
                "meta": {"snapshot_id": snapshot["snapshot_id"], "membership_count": member_count,
                         "membership_status": "stale"}})
        results = await anyio.to_thread.run_sync(
            scanner.scan_support_resistance, sector_name, member_rows=members,
        )
        return {"data": results, "meta": {"snapshot_id": snapshot["snapshot_id"],
                "membership_count": member_count, "candidates": min(member_count, 80),
                "returned": len(results), "source": snapshot["source_name"],
                "membership_status": snapshot["membership_status"]}}
    except Exception as e:
        logger.error(f"板块API异常: {e}")
        return JSONResponse(status_code=503, content={"data": [], "error": str(e), "retry": True})


@router.get("/{sector_name}/dense-zones")
async def scan_dense_zones(
    sector_name: str,
    snapshot_id: str = Query(None, min_length=32, max_length=32),
    max_stocks: int = Query(80, ge=10, le=200),
    window: int = Query(40, ge=20, le=120),
    n_zones: int = Query(3, ge=1, le=10),
):
    """
    板块密集成交区扫描（成交量加权密度剖面算法）

    相比支撑/压力位扫描，此端点提供：
    - 量价加权的真实筹码分布
    - 多密集区同时检测（不只是最近支撑/压力）
    - 盈亏比计算
    - 板块级别聚合统计（支撑占比/压力占比/多空状态）
    """
    try:
        snapshot, members, member_count = await _load_verified_industry_members(sector_name, snapshot_id)
        if not snapshot or not members:
            return JSONResponse(status_code=409, content={"data": {}, "error": "verified_industry_snapshot_unavailable"})
        if snapshot["membership_status"] == "stale":
            return JSONResponse(status_code=409, content={"data": {}, "error": "industry_membership_snapshot_stale",
                "meta": {"snapshot_id": snapshot["snapshot_id"], "membership_count": member_count,
                         "membership_status": "stale"}})
        result = await anyio.to_thread.run_sync(
            scanner.scan_dense_zones,
            sector_name, max_stocks=max_stocks,
            window=window, n_zones=n_zones, member_rows=members,
        )
        return {"data": result, "meta": {"snapshot_id": snapshot["snapshot_id"],
                "membership_count": member_count, "candidates": min(member_count, max_stocks),
                "scanned": result.get("scanned"), "source": snapshot["source_name"],
                "membership_status": snapshot["membership_status"]}}
    except Exception as e:
        logger.error(f"板块API异常(dense-zones): {e}")
        return JSONResponse(status_code=503, content={"data": {}, "error": str(e), "retry": True})


def _scan_sector_heatmap(stock_count=200, window=40, n_zones=3):
    from core.data.repos.stock_industry_history import (
        get_industry_snapshot_members,
        get_latest_available_industry_snapshot,
    )
    from core.shared.trading_time import market_date

    snapshot = get_latest_available_industry_snapshot(market_date().isoformat())
    if not snapshot:
        return [], None
    snapshot_age = (market_date() - datetime.date.fromisoformat(str(snapshot["available_after_date"])[:10])).days
    if snapshot_age > 14:
        return [], snapshot
    members = get_industry_snapshot_members(snapshot["payload_sha256"])
    member_counts = {}
    for row in members:
        industry = row.get("industry")
        if industry:
            member_counts[industry] = member_counts.get(industry, 0) + 1
    sector_names = sorted(member_counts, key=lambda name: (-member_counts[name], name))[:15]
    per_sector = max(1, min(stock_count // max(len(sector_names), 1), 30))
    results = []
    for sector in sector_names:
        try:
            sector_members = [row for row in members if row.get("industry") == sector]
            s = scanner.scan_dense_zones(
                sector, max_stocks=per_sector, window=window, n_zones=n_zones,
                member_rows=sector_members,
            )
            if s:
                s["snapshot_id"] = snapshot["snapshot_id"]
                results.append(s)
        except Exception:
            pass
    return results, snapshot


@router.get("/heatmap")
async def sector_heatmap(
    stock_count: int = Query(200, ge=50, le=500),
    window: int = Query(40, ge=20, le=120),
    n_zones: int = Query(3, ge=1, le=10),
):
    """
    全市场板块热力图

    按行业聚合统计密集区支撑/压力分布，
    返回每个行业的多空状态、平均盈亏比、支撑占比等。
    """
    try:
        result, snapshot = await anyio.to_thread.run_sync(
            lambda: _scan_sector_heatmap(
                stock_count=stock_count, window=window, n_zones=n_zones),
        )
        if not snapshot:
            return {"data": [], "meta": {"status": "unavailable", "missing_reason": "verified_industry_snapshot_missing"}}
        from core.shared.trading_time import market_date
        snapshot_age = (market_date() - datetime.date.fromisoformat(str(snapshot["available_after_date"])[:10])).days
        status = "stale" if snapshot_age > 14 else "available"
        return {"data": result, "meta": {"status": status, "snapshot_id": snapshot["snapshot_id"],
                "source": snapshot["source_name"], "classification_standard": snapshot["classification_standard"],
                "as_of": snapshot["query_date"], "sectors": len(result),
                "missing_reason": None if result else "no_sector_scan_results"}}
    except Exception as e:
        logger.error(f"板块API异常: {e}")
        return JSONResponse(status_code=503, content={"data": [], "error": str(e), "retry": True})
