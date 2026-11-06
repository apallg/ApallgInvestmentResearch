MARKET_INDEXES = (
    ("sh000001", "上证指数", "000001", "SH"),
    ("sz399001", "深证成指", "399001", "SZ"),
    ("sz399006", "创业板指", "399006", "SZ"),
    ("sh000688", "科创50", "000688", "SH"),
    ("sh000300", "沪深300", "000300", "SH"),
    ("sh000905", "中证500", "000905", "SH"),
)

_INDEX_CODES = {code.upper() for code, _, _, _ in MARKET_INDEXES}
_INDEX_TSCODES = {f"{symbol}.{exchange}" for _, _, symbol, exchange in MARKET_INDEXES}
_UNAMBIGUOUS_INDEX_SYMBOLS = {
    symbol for _, _, symbol, _ in MARKET_INDEXES if symbol != "000001"
}


def is_market_index_symbol(value):
    code = str(value or "").strip().upper()
    if code.startswith("INDEX:"):
        return True
    return code in _INDEX_CODES or code in _INDEX_TSCODES


def is_unambiguous_index_symbol(value):
    return str(value or "").strip() in _UNAMBIGUOUS_INDEX_SYMBOLS


def search_market_indexes(query, limit):
    text = str(query or "").strip().casefold()
    if not text:
        return []
    matches = []
    for code, name, symbol, exchange in MARKET_INDEXES:
        if text in name.casefold() or text in code.casefold() or text in symbol:
            matches.append({
                "code": code,
                "name": name,
                "type": "指数",
                "asset_type": "index",
                "exchange": exchange,
                "instrument_id": f"index:{exchange}:{symbol}",
            })
    return matches[:limit]
