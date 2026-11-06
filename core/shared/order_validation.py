import re
from decimal import Decimal, InvalidOperation
from core.shared.instrument_identity import is_market_index_symbol, is_unambiguous_index_symbol


def normalize_a_share_symbol(symbol):
    code = str(symbol or "").strip().upper()
    if is_market_index_symbol(code):
        raise ValueError("指数身份不能用作股票代码")
    if not re.fullmatch(r"\d{6}(?:\.(?:SH|SZ|BJ))?", code):
        raise ValueError("证券代码必须为六位数字，可附交易所后缀")
    base, _, exchange = code.partition(".")
    if is_unambiguous_index_symbol(base):
        raise ValueError("指数代码不能用作股票代码")
    expected = ("SH" if base[0] in "56" else
                "SZ" if base[0] in "0123" else
                "BJ" if base[0] in "48" else None)
    if expected is None or (exchange and exchange != expected):
        raise ValueError("证券代码与交易所后缀不一致")
    return base


def validate_real_limit_order(symbol, side, quantity, price_type, price):
    code = str(symbol or "").strip().upper()
    normalize_a_share_symbol(code)
    if side not in ("buy", "sell"):
        raise ValueError("交易方向必须为 buy 或 sell")
    if price_type != "limit":
        raise ValueError("真实券商当前仅支持明确限价单，禁止市价单暗转限价")
    if isinstance(quantity, bool):
        raise ValueError("委托股数无效")
    try:
        decimal_shares = Decimal(str(quantity))
        limit_price = Decimal(str(price))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("委托股数或限价无效") from None
    if (not decimal_shares.is_finite() or decimal_shares != decimal_shares.to_integral_value() or
            decimal_shares <= 0 or decimal_shares > 100000000):
        raise ValueError("委托股数无效，A股买入须为100股整数倍")
    shares = int(decimal_shares)
    if shares <= 0 or shares > 100000000 or (side == "buy" and shares % 100):
        raise ValueError("委托股数无效，A股买入须为100股整数倍")
    if (not limit_price.is_finite() or limit_price <= 0 or
            limit_price >= Decimal("1000000000") or
            limit_price != limit_price.quantize(Decimal("0.001"))):
        raise ValueError("限价须大于0且最多三位小数")
    return code, shares, float(limit_price)
