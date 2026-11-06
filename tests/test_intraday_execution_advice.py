from core.domain.portfolio.intraday import IntradayResult
from backend.service.rebalance import _execution_advice, _intraday_execution_blockers


def _quote(price, previous_close=10.0):
    return {
        "price": price,
        "prev_close": previous_close,
        "name": "平安银行",
    }


def test_stop_loss_preserves_market_intent_without_claiming_execution():
    result = IntradayResult(code="000001", current_price=9.0)

    advice = _execution_advice(
        result,
        {"close": 10.0},
        side="sell",
        urgency="stop_loss",
        phase="mid",
        blockers=[],
        quote=_quote(9.0),
    )

    assert advice["intent_action"] == "market"
    assert advice["can_execute"] is False
    assert advice["status"] == "blocked"
    assert any("仅支持限价单" in blocker for blocker in advice["blockers"])


def test_buy_fill_above_limit_up_is_blocked():
    result = IntradayResult(code="000001", current_price=10.98)

    advice = _execution_advice(
        result,
        {"close": 10.0},
        side="buy",
        phase="close",
        blockers=[],
        quote=_quote(10.98),
    )

    assert advice["action"] == "fill"
    assert advice["limit_price"] > 11.0
    assert advice["can_execute"] is False
    assert any("超出当日涨跌停范围" in blocker for blocker in advice["blockers"])


def test_sell_fill_below_limit_down_is_blocked():
    result = IntradayResult(code="000001", current_price=9.02)

    advice = _execution_advice(
        result,
        {"close": 10.0},
        side="sell",
        phase="close",
        blockers=[],
        quote=_quote(9.02),
    )

    assert advice["action"] == "fill"
    assert advice["limit_price"] < 9.0
    assert advice["can_execute"] is False
    assert any("超出当日涨跌停范围" in blocker for blocker in advice["blockers"])


def test_structurally_verified_bars_without_source_semantics_block_execution():
    import datetime
    import pandas as pd

    now = datetime.datetime(2026, 10, 30, 10, 0)
    df = pd.DataFrame({"close": [10.0]}, index=pd.DatetimeIndex(["2026-10-30 09:59:00"]))

    blockers = _intraday_execution_blockers(
        "000001", df, None, "2026-10-30", now, True, "buy",
        bars_verified=True, semantics_verified=False,
    )

    assert "分钟线未通过verified状态校验" not in blockers
    assert "分钟数据单位、累计口径或价格基准未完成语义核验" in blockers
