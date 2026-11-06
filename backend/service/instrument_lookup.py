import logging

from core.shared.instrument_identity import is_market_index_symbol, is_unambiguous_index_symbol, search_market_indexes

logger = logging.getLogger(__name__)


def resolve_current_stock_code(value):
    text = str(value or "").strip().upper()
    if not text or is_market_index_symbol(text) or is_unambiguous_index_symbol(text):
        return None
    exchange = None
    symbol = text
    if text.startswith(("SH", "SZ", "BJ")) and len(text) == 8:
        exchange, symbol = text[:2], text[2:]
    elif ":" in text:
        parts = text.split(":")
        if len(parts) != 3 or parts[0] != "STOCK":
            return None
        exchange, symbol = parts[1], parts[2]
    elif "." in text:
        parts = text.split(".")
        if len(parts) != 2:
            return None
        symbol, exchange = parts[0], parts[1]
    if not symbol.isdigit() or len(symbol) > 6:
        return None
    symbol = symbol.zfill(6)
    if exchange and exchange not in ("SH", "SZ", "BJ"):
        return None
    from core.data.repos.stock_identity import StockIdentityRepo

    identity = StockIdentityRepo.get_current_identity(symbol, exchange)
    if not identity:
        return None
    ts_code = str(identity.get("ts_code") or "").strip().upper()
    suffix = ts_code.rsplit(".", 1)[-1] if "." in ts_code else ""
    resolved = str(identity.get("symbol") or "").strip().zfill(6)
    if resolved != symbol or suffix not in ("SH", "SZ", "BJ"):
        return None
    return resolved


def search_instruments(query, limit=10):
    text = str(query or "").strip()
    if not text:
        return []
    if isinstance(limit, bool) or not isinstance(limit, int):
        return []
    limit = max(1, min(limit, 20))
    results = []
    try:
        from core.data.repos.stock_identity import StockIdentityRepo

        for row in StockIdentityRepo.search_current(text, limit):
            symbol = str(row.get("symbol") or "").strip().zfill(6)
            ts_code = str(row.get("ts_code") or "").strip().upper()
            exchange = ts_code.rsplit(".", 1)[-1] if "." in ts_code else str(row.get("exchange") or "").upper()
            if len(symbol) != 6 or not symbol.isdigit() or exchange not in ("SH", "SZ", "BJ"):
                continue
            results.append({
                "code": symbol,
                "name": str(row.get("name") or symbol),
                "type": "A股",
                "asset_type": "stock",
                "exchange": exchange,
                "instrument_id": f"stock:{exchange}:{symbol}",
            })
    except Exception as error:
        logger.warning("证券身份搜索不可用: %s", type(error).__name__)
    results.extend(search_market_indexes(text, limit))
    unique = {}
    for result in results:
        unique[result["instrument_id"]] = result
    return list(unique.values())[:limit]
