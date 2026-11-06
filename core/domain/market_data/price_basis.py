import math


def qfq_to_live_raw_scale(last_qfq_close, last_adj_factor, last_trade_date,
                          expected_previous_date, raw_previous_close, factor_history):
    try:
        qfq_close = float(last_qfq_close)
        adj_factor = float(last_adj_factor)
        raw_close = float(raw_previous_close)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in (qfq_close, adj_factor, raw_close)):
        return None
    if str(last_trade_date)[:10] != str(expected_previous_date)[:10]:
        return None
    try:
        factors = [float(value) for value in factor_history]
    except (TypeError, ValueError):
        return None
    if not factors or any(not math.isfinite(value) or value <= 0 for value in factors):
        return None
    if abs(raw_close - qfq_close) > 0.005:
        return None
    return raw_close / qfq_close
