import pandas as pd
import pytest

from core.domain.screening.indicators import Analyzer


pytestmark = [pytest.mark.p0, pytest.mark.unit]


def _frame(direction=1):
    rows = []
    for index in range(25):
        close = 10 + direction * index * 0.1
        rows.append({
            "trade_date": pd.Timestamp("2026-08-01") + pd.Timedelta(days=index),
            "close": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "pct_chg": direction * 1.0,
            "volume": 1_500_000,
            "MA5": close - direction * 0.1,
            "MA10": close - direction * 0.2,
            "MA20": close - direction * 0.3,
            "MACD": direction * 0.2,
            "MACD_signal": direction * 0.1,
            "RSI14": 60 if direction > 0 else 40,
            "KDJ_K": 60 if direction > 0 else 40,
            "KDJ_D": 50,
            "BB_upper": close + 1,
            "BB_middle": close - direction * 0.2,
            "BB_lower": close - 1,
            "VOL_MA20": 1_000_000,
            "ATR14": 0.2,
        })
    return pd.DataFrame(rows)


def test_technical_signal_summary_reports_bullish_indicators():
    result = Analyzer.technical_signal_summary(_frame(1), current_price=12.5)

    assert result["overall"]["bias"] == "bullish"
    assert result["overall"]["bullish"] >= 5
    assert {item["key"] for item in result["items"]} == {
        "ma", "macd", "rsi", "kdj", "bollinger", "volume", "atr", "momentum",
    }
    assert result["as_of"] == "2026-08-25"


def test_technical_signal_summary_reports_bearish_indicators():
    result = Analyzer.technical_signal_summary(_frame(-1), current_price=7.5)

    assert result["overall"]["bias"] == "bearish"
    assert result["overall"]["bearish"] >= 5


def test_technical_signal_summary_marks_overbought_as_risk_not_simple_bullish():
    frame = _frame(1)
    frame.loc[frame.index[-1], "RSI14"] = 78
    frame.loc[frame.index[-1], ["KDJ_K", "KDJ_D"]] = [88, 84]

    result = Analyzer.technical_signal_summary(frame, current_price=14)
    statuses = {item["key"]: item["status"] for item in result["items"]}

    assert statuses["rsi"] == "risk"
    assert statuses["kdj"] == "risk"
    assert statuses["bollinger"] == "risk"


def test_volume_price_marks_breakout_only_above_prior_20_day_high():
    frame = _frame(1)
    last = frame.index[-1]
    frame.loc[last, "close"] = frame.loc[:last - 1, "high"].max() + 0.2
    frame.loc[last, "volume"] = 1_600_000

    result = Analyzer.technical_signal_summary(frame)
    volume = next(item for item in result["items"] if item["key"] == "volume")

    assert volume["status"] == "bullish"
    assert volume["value"].startswith("放量突破")
    assert "后续不跌回原区间确认" in volume["detail"]


def test_volume_price_treats_heavy_down_day_as_unconfirmed_distribution():
    frame = _frame(1)
    last = frame.index[-1]
    frame.loc[last, ["pct_chg", "volume"]] = [-2.0, 1_600_000]

    result = Analyzer.technical_signal_summary(frame)
    volume = next(item for item in result["items"] if item["key"] == "volume")

    assert volume["status"] == "bearish"
    assert volume["value"].startswith("放量下跌")
    assert "出货特征" in volume["detail"]
    assert "单日不能确认" in volume["detail"]


def test_volume_price_calls_shrinking_down_day_washout_only_above_ma20():
    frame = _frame(1)
    last = frame.index[-1]
    frame.loc[last, ["pct_chg", "volume", "MA20"]] = [-1.0, 600_000, frame.loc[last, "close"] - 0.3]

    result = Analyzer.technical_signal_summary(frame)
    volume = next(item for item in result["items"] if item["key"] == "volume")

    assert volume["status"] == "neutral"
    assert volume["value"].startswith("缩量回调")
    assert "具备洗盘特征但不能确认" in volume["detail"]

    frame.loc[last, "MA20"] = frame.loc[last, "close"] + 0.3
    result = Analyzer.technical_signal_summary(frame)
    volume = next(item for item in result["items"] if item["key"] == "volume")

    assert volume["status"] == "bearish"
    assert volume["value"].startswith("缩量阴跌")
    assert "不能仅按洗盘解释" in volume["detail"]
