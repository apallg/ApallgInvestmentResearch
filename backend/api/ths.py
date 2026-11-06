"""同花顺实时数据 API"""
import json
import logging
import math
import anyio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from core.shared.infrastructure.redis_client import get_redis
from backend.api.instrument_identity import require_current_stock_code
from core.data import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ths", tags=["同花顺数据"])


def _safe_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


REALTIME_FIELDS = {
    "code": "代码", "name": "名称",
    "7": "最新价", "8": "开盘价", "9": "最高价", "10": "最低价",
    "24": "昨收", "13": "成交量", "19": "成交额",
    "30": "买一价", "31": "买一量", "32": "买二价", "33": "买二量",
    "34": "买三价", "35": "买三量", "36": "买四价", "37": "买四量",
    "38": "买五价", "39": "买五量",
    "40": "卖一价", "41": "卖一量", "42": "卖二价", "43": "卖二量",
    "44": "卖三价", "45": "卖三量", "46": "卖四价", "47": "卖四量",
    "48": "卖五价", "49": "卖五量",
    "69": "52周最高", "70": "52周最低",
    "199112": "涨跌幅", "264648": "涨跌额",
    "461256": "市盈率", "134152": "市净率",
    "3475914": "总市值", "3541450": "流通市值",
    "1968584": "换手率", "1771976": "量比",
    "1149395": "每股收益", "1378761": "均价",
    "223": "大单净流入", "224": "大单净流出",
    "237": "主力净流入", "238": "主力净流入占比",
    "stockStatus": "交易状态", "updateTime": "更新时间",
}


@router.get("/stock/{code}/realtime")
async def get_realtime(code: str = Depends(require_current_stock_code)):
    """严格实时行情，不读取历史或过期 Redis 快照。"""
    from backend.service.market_data import MarketDataService
    response = await anyio.to_thread.run_sync(
        MarketDataService.get_live_quotes, [code], "live_display"
    )
    quote = response.get("data", {}).get(code)
    if not quote:
        return {"data": None, "meta": response.get("meta", {})}
    meta = response.get("meta", {})
    return {
        "data": {
            "code": code,
            "名称": quote.get("name", ""),
            "最新价": quote.get("price"),
            "涨跌幅": quote.get("pct_chg", 0),
            "涨跌额": quote.get("change", 0),
            "开盘价": quote.get("open"),
            "最高价": quote.get("high"),
            "最低价": quote.get("low"),
            "昨收": quote.get("prev_close"),
            "成交量": quote.get("volume", 0),
            "成交额": quote.get("amount", 0),
            "source": quote.get("source"),
            "observed_at": quote.get("observed_at"),
            "received_at": quote.get("received_at"),
            "age_seconds": meta.get("age_seconds"),
            "买盘": [],
            "卖盘": [],
        },
        "meta": meta,
    }


@router.post("/collect/{code}")
async def trigger_collect(code: str = Depends(require_current_stock_code)):
    """手动触发单只股票的实时采集"""
    try:
        _warmup_sync(code)
        return {"ok": True, "code": code}
    except Exception as e:
        raise HTTPException(500, str(e))


def _format_realtime(items, code):
    result = {"code": code, "name": items.get("name", "")}
    for fid, cname in REALTIME_FIELDS.items():
        val = items.get(fid, "")
        if val and val != "-1" and val != "":
            result[cname] = _safe_float(val) if fid.isdigit() else val
    # 计算涨跌幅
    close_v = _safe_float(items.get("7"))
    prev = _safe_float(items.get("24"))
    if close_v and prev and prev > 0 and "涨跌幅" not in result:
        result["涨跌幅"] = round((close_v / prev - 1) * 100, 2)
    # 盘口
    bids = []
    for i in range(5):
        bp = items.get(str(30 + i * 2), "")
        bv = items.get(str(31 + i * 2), "")
        if bp and bp != "-1":
            bids.append({"price": _safe_float(bp), "volume": _safe_float(bv, 0)})
    asks = []
    for i in range(5):
        ap = items.get(str(40 + i * 2), "")
        av = items.get(str(41 + i * 2), "")
        if ap and ap != "-1":
            asks.append({"price": _safe_float(ap), "volume": _safe_float(av, 0)})
    result["买盘"] = bids
    result["卖盘"] = asks
    result["_raw_time"] = items.get("_time", "")
    return result


@router.get("/stock/{code}/intraday")
async def get_intraday(code: str = Depends(require_current_stock_code)):
    """当日分时数据 (直接来自 THS 实时接口，不读取 Redis 历史快照)"""
    from core.shared.data_contract import make_data_envelope, revalidate_data_envelope
    from core.shared.minute_data import (
        has_complete_minute_sequence,
        has_verified_minute_semantics,
        normalize_trade_quantity,
    )

    try:
        from backend.service.market_data import MarketDataService
        with anyio.fail_after(15):
            rows, source = await anyio.to_thread.run_sync(
                MarketDataService.get_live_intraday,
                code,
                abandon_on_cancel=True,
            )
    except TimeoutError:
        return {"data": [], "meta": {"data_type": "intraday_bar", "source": "ths", "freshness": "unavailable", "reason": "timeout"}}
    except Exception as e:
        logger.warning(f"THS 分时获取失败 {code}: {e}")
        return {"data": [], "meta": {"data_type": "intraday_bar", "source": "ths", "freshness": "unavailable"}}

    from core.shared.trading_time import A_SHARE_TIMEZONE

    rows = rows or []
    now = datetime.now(A_SHARE_TIMEZONE)
    parsed_rows = []
    invalid_row_count = 0
    for row in rows:
        try:
            row_time = datetime.fromisoformat(str(row.get("bar_time", "")).replace("Z", "+00:00"))
            if row_time.tzinfo is None:
                row_time = row_time.replace(tzinfo=A_SHARE_TIMEZONE)
            else:
                row_time = row_time.astimezone(A_SHARE_TIMEZONE)
            if row_time.date() != now.date():
                invalid_row_count += 1
                continue
            parsed_rows.append((row_time, row))
        except (AttributeError, TypeError, ValueError):
            invalid_row_count += 1
    source_times = [row_time for row_time, _ in parsed_rows]
    out_of_order = source_times != sorted(source_times)
    timestamp_counts = {}
    for row_time in source_times:
        timestamp_counts[row_time] = timestamp_counts.get(row_time, 0) + 1
    duplicate_times = {row_time for row_time, count in timestamp_counts.items() if count > 1}
    ordered_rows = [
        (row_time, row) for row_time, row in sorted(parsed_rows, key=lambda item: item[0])
        if row_time not in duplicate_times
    ]
    row_envelopes = [row.get("data_envelope") for row in rows if isinstance(row, dict)]
    shared_envelope = row_envelopes[0] if row_envelopes and all(
        envelope == row_envelopes[0] for envelope in row_envelopes
    ) else None
    semantic_verified = has_verified_minute_semantics(shared_envelope)
    display_rows = []
    valid_bar_times = []
    invalid_price_count = 0
    for row_time, row in ordered_rows:
        try:
            price = float(row.get("price"))
            if not math.isfinite(price) or price <= 0:
                invalid_price_count += 1
                continue
        except (TypeError, ValueError, OverflowError):
            invalid_price_count += 1
            continue
        quantity_status = (
            "verified" if semantic_verified and row.get("quantity_status") == "verified" else "unknown"
        )
        quantity = normalize_trade_quantity(row.get("volume"), row.get("amount")) if quantity_status == "verified" else None
        if quantity is None:
            quantity_status = "unknown"
        avg_price = None
        if quantity_status == "verified":
            try:
                value = float(row.get("avg_price"))
                if math.isfinite(value) and value > 0:
                    avg_price = value
            except (TypeError, ValueError, OverflowError):
                pass
        display_row = dict(row)
        display_row.pop("raw_fields", None)
        display_row["price"] = price
        display_row["volume"] = quantity[0] if quantity is not None else None
        display_row["amount"] = quantity[1] if quantity is not None else None
        display_row["avg_price"] = avg_price
        display_row["quantity_status"] = quantity_status
        display_rows.append(display_row)
        valid_bar_times.append(row_time)
    sequence_complete = bool(
        not invalid_row_count
        and not out_of_order
        and not duplicate_times
        and not invalid_price_count
        and has_complete_minute_sequence(source_times)
    )
    latest_dt = max(valid_bar_times, default=None)
    latest = latest_dt.isoformat(timespec="seconds") if latest_dt else None
    freshness = "unavailable"
    age_seconds = None
    if latest_dt is not None:
        age_seconds = round((now - latest_dt).total_seconds(), 3)
        freshness = "stale" if age_seconds < -5 or age_seconds > 90 else "fresh"
    quantity_verified = bool(
        semantic_verified and sequence_complete and freshness == "fresh" and display_rows
        and all(row["quantity_status"] == "verified" for row in display_rows)
    )
    returned_rows = display_rows if freshness == "fresh" else []
    unit = (shared_envelope or {}).get("unit") or {}
    if semantic_verified:
        data_envelope = dict(shared_envelope)
        if freshness != "fresh":
            data_envelope["status"] = freshness
        elif not quantity_verified or not sequence_complete:
            data_envelope["status"] = "partial"
        data_envelope["quantity_status"] = "verified" if quantity_verified else "unknown"
        data_envelope["coverage"] = {
            "requested": len(rows), "accepted": len(returned_rows),
            "missing": len(rows) - len(returned_rows),
        }
        if not quantity_verified:
            reasons = data_envelope.get("missing_reason") or []
            if isinstance(reasons, str):
                reasons = [reasons]
            if not sequence_complete:
                reasons = [*reasons, "minute_sequence_integrity"]
            if any(row["quantity_status"] != "verified" for row in display_rows):
                reasons = [*reasons, "minute_quantity_rows_unverified"]
            data_envelope["missing_reason"] = list(dict.fromkeys(reasons))
    else:
        data_envelope = make_data_envelope(
            source=source or "ths", observed_at=latest,
            available_at=now.isoformat(timespec="seconds"), as_of=now.date().isoformat(),
            version="unknown", unit={"price": "unknown", "volume": "unknown", "amount": "unknown"},
            price_basis="unknown", coverage={
                "requested": len(rows), "accepted": len(returned_rows),
                "missing": len(rows) - len(returned_rows),
            }, status="partial" if freshness == "fresh" else freshness,
            missing_reason=["price_basis", "volume_unit", "amount_unit", "source_quantity_semantics"],
        ) | {"source_quantity_semantics": "unknown", "quantity_status": "unknown"}
    data_envelope = revalidate_data_envelope(data_envelope)
    return {
        "data": returned_rows,
        "meta": {
            "data_type": "intraday_bar",
            "source": source,
            "freshness": freshness,
            "age_seconds": age_seconds,
            "session_date": now.date().isoformat(),
            "semantic_status": "verified" if semantic_verified else "unverified",
            "quantity_status": "verified" if quantity_verified else "unknown",
            "sequence_status": "complete" if sequence_complete else "partial",
            "rejected_bar_count": len(rows) - len(returned_rows),
            "price_basis": data_envelope.get("price_basis", "unknown"),
            "volume_unit": unit.get("volume"),
            "amount_unit": unit.get("amount"),
            "data_envelope": data_envelope,
        },
    }


@router.get("/stock/{code}/daily")
async def get_daily_kline(code: str = Depends(require_current_stock_code)):
    """日K线 (统一来自 daily_quotes 前复权口径)"""
    try:
        rows = Database.fetchall(
            "SELECT trade_date, open, high, low, close, volume, amount, pct_chg, "
            "turnover AS turnover_rate "
            "FROM daily_quotes WHERE code=%s ORDER BY trade_date DESC LIMIT 250",
            (code,)
        )
        return {
            "data": rows,
            "source": "daily_quotes",
            "price_basis": "qfq",
            "volume_unit": "share",
            "amount_unit": "yuan",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/stock/{code}/60min")
async def get_60min_kline(code: str = Depends(require_current_stock_code)):
    """60分钟K线 (来自MySQL持久化, 缺失时自动采集)"""
    from core.shared.data_contract import make_data_envelope
    from core.shared.trading_time import A_SHARE_TIMEZONE

    try:
        rows = Database.fetchall(
            "SELECT bar_time, open, high, low, close, volume, amount, pct_chg "
            "FROM ths_60min_kline WHERE code=%s ORDER BY bar_time DESC LIMIT 200",
            (code,)
        )
        if not rows:
            _warmup_sync(code)
            rows = Database.fetchall(
                "SELECT bar_time, open, high, low, close, volume, amount, pct_chg "
                "FROM ths_60min_kline WHERE code=%s ORDER BY bar_time DESC LIMIT 200",
                (code,)
            )
        rows = rows or []
        latest_bar = rows[0].get("bar_time") if rows else None
        observed_at = latest_bar.isoformat() if hasattr(latest_bar, "isoformat") else str(latest_bar) if latest_bar else None
        as_of = latest_bar.date().isoformat() if hasattr(latest_bar, "date") else str(latest_bar)[:10] if latest_bar else None
        now = datetime.now(A_SHARE_TIMEZONE)
        data_envelope = make_data_envelope(
            source="ths",
            observed_at=observed_at,
            available_at=now.isoformat(timespec="seconds"),
            as_of=as_of,
            version="ths-60min-v1",
            unit={"price": "unknown", "volume": "unknown", "amount": "unknown"},
            price_basis="unknown",
            coverage={"requested": len(rows), "accepted": len(rows)},
            status="partial" if rows else "unavailable",
            missing_reason=["price_basis", "volume_unit", "amount_unit", "source_quantity_semantics"] if rows else ["no_data"],
        ) | {"source_quantity_semantics": "unknown", "quantity_status": "unknown"}
        safe_rows = [
            {**row, "volume": None, "amount": None, "quantity_status": "unknown"}
            for row in rows
        ]
        return {
            "data": safe_rows,
            "meta": {
                "data_type": "60min_bar",
                "source": "ths",
                "semantic_status": "unverified",
                "quantity_status": "unknown",
                "price_basis": "unknown",
                "volume_unit": None,
                "amount_unit": None,
                "data_envelope": data_envelope,
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/signals/{code}")
async def get_signals(code: str = Depends(require_current_stock_code)):
    """读取采集任务生成的个股异动信号"""
    r = get_redis()
    if not r:
        raise HTTPException(503, "Redis 不可用")
    raw = r.get(f"ths:signals:{code}")
    if not raw:
        return {"data": {"code": code, "snapshots": 0, "detail": "等待采集任务生成异动数据"}}
    try:
        signals = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return {"data": {"code": code, "snapshots": 0, "detail": "异动结果暂不可用"}}
    if not isinstance(signals, dict):
        return {"data": {"code": code, "snapshots": 0, "detail": "异动结果暂不可用"}}
    if not signals or signals.get("snapshots", 0) < 2:
        return {"data": {"code": code, "snapshots": 0, "detail": "数据积累中，请等待几秒后刷新"}}
    return {"data": signals}


@router.get("/alerts")
async def get_alerts():
    """全局异动列表 (来自Redis缓存)"""
    r = get_redis()
    if not r:
        raise HTTPException(503, "Redis 不可用")
    raw = r.get("ths:alerts")
    if not raw:
        return {"data": []}
    alerts = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
    return {"data": alerts}


@router.get("/index/{code}/daily")
async def get_index_daily(code: str):
    """指数日K线对比数据 (返回累计收益率, 用于叠图对比)"""
    try:
        rows = Database.fetchall(
            "SELECT trade_date, close FROM ("
            "SELECT trade_date, close FROM ths_daily_kline WHERE code=%s "
            "ORDER BY trade_date DESC LIMIT 250"
            ") AS recent_index_rows ORDER BY trade_date ASC",
            (code,)
        )
        if not rows:
            _warmup_sync(code)
            rows = Database.fetchall(
                "SELECT trade_date, close FROM ("
                "SELECT trade_date, close FROM ths_daily_kline WHERE code=%s "
                "ORDER BY trade_date DESC LIMIT 250"
                ") AS recent_index_rows ORDER BY trade_date ASC",
                (code,)
            )
        return {"data": rows}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/stock/{code}/snapshots")
async def get_snapshots(code: str = Depends(require_current_stock_code), limit: int = 20):
    """历史实时快照 (用于查看盘中变化)"""
    try:
        rows = Database.fetchall(
            "SELECT snapshot_time, data_json FROM ths_realtime_snapshots "
            "WHERE code=%s ORDER BY snapshot_time DESC LIMIT %s",
            (code, limit)
        )
        return {"data": rows}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── 自动预热 ────────────────────────────────────────────────

def _warmup_sync(code: str):
    """首次访问时自动采集实时行情+日K线, 建立信号窗口"""
    from core.data.ths_collector import THSCollector, _get_cookie, ensure_tables
    ensure_tables()
    c = THSCollector()
    c.session.headers["Cookie"] = f"v={_get_cookie()}"
    for attempt in range(3):
        try:
            c.collect_realtime_batch([code])
            logger.info(f"实时行情预热成功: {code}")
            break
        except Exception as e:
            logger.warning(f"实时行情预热失败 (attempt {attempt+1}/3): {code} {e}")
            import time; time.sleep(1)
    try:
        rows = c.fetch_daily_kline(code)
        if rows:
            c._save_daily(code, rows)
            logger.info(f"日K线预热成功: {code} ({len(rows)}条)")
    except Exception as e:
        logger.warning(f"日K线预热失败: {code} {e}")
    try:
        rows = c.fetch_60min_kline(code)
        if rows:
            c._save_60min(code, rows)
            logger.info(f"60分钟K线预热成功: {code} ({len(rows)}条)")
    except Exception as e:
        logger.warning(f"60分钟K线预热失败: {code} {e}")
    try:
        rows = c.fetch_intraday(code)
        if rows:
            c._save_intraday(code, rows)
            logger.info(f"分时数据预热成功: {code} ({len(rows)}条)")
    except Exception as e:
        logger.warning(f"分时数据预热失败: {code} {e}")
