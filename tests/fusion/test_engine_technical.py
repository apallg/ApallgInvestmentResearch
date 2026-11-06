import pytest
import pandas as pd
import numpy as np
from tests.factories import make_kline_df


def _make_signal_df_with_indicators(n_rows=60):
    """构建 SignalFusion 期望的列名格式的 DataFrame"""
    df = make_kline_df(n_rows=n_rows, with_indicators=True)
    # SignalFusion 使用大写列名
    df["MACD"] = df["macd"].values
    df["MACD_signal"] = df["macd_signal"].values
    df["RSI14"] = df["rsi14"].values
    df["MA5"] = df["ma5"].values
    df["MA10"] = df["ma10"].values
    df["MA20"] = df["ma20"].values
    df["KDJ_K"] = df["k"].values
    df["KDJ_D"] = df["d"].values
    df["BB_lower"] = df["bb_lower"].values
    df["BB_upper"] = df["bb_upper"].values
    df["VOL_MA20"] = df["vol_ma20"].values
    return df


class TestSignalFusion:
    @pytest.fixture(autouse=True)
    def setup(self):
        from core.domain.signals.engine import SignalFusion
        self.fusion = SignalFusion()

    def test_all_bullish(self):
        df = _make_signal_df_with_indicators(60)
        # 强制全看多
        df.at[df.index[-2], "MACD"] = -0.10
        df.at[df.index[-2], "MACD_signal"] = -0.05
        df.at[df.index[-1], "MACD"] = 0.05
        df.at[df.index[-1], "MACD_signal"] = 0.03
        df.at[df.index[-1], "RSI14"] = 25.0
        df.at[df.index[-1], "MA5"] = 15.0
        df.at[df.index[-1], "MA20"] = 13.0
        df.at[df.index[-1], "MA10"] = 14.0
        df.at[df.index[-2], "KDJ_K"] = 20.0
        df.at[df.index[-2], "KDJ_D"] = 28.0
        df.at[df.index[-1], "KDJ_K"] = 29.0
        df.at[df.index[-1], "KDJ_D"] = 25.0
        lower = df["BB_lower"].iloc[-1]
        df.at[df.index[-1], "close"] = lower
        df.at[df.index[-1], "volume"] = 5000000.0
        df.at[df.index[-1], "VOL_MA20"] = 2000000.0
        df.at[df.index[-1], "pct_chg"] = 2.0

        result = self.fusion.evaluate(df)
        assert result.action == "buy"
        assert result.confidence >= 0.3

    def test_all_bearish(self):
        df = _make_signal_df_with_indicators(60)
        # 强制全看空
        df.at[df.index[-2], "MACD"] = 0.10
        df.at[df.index[-2], "MACD_signal"] = 0.05
        df.at[df.index[-1], "MACD"] = 0.03
        df.at[df.index[-1], "MACD_signal"] = 0.08
        df.at[df.index[-1], "RSI14"] = 80.0
        df.at[df.index[-1], "MA5"] = 12.0
        df.at[df.index[-1], "MA20"] = 15.0
        df.at[df.index[-1], "MA10"] = 14.0
        df.at[df.index[-2], "KDJ_K"] = 80.0
        df.at[df.index[-2], "KDJ_D"] = 75.0
        df.at[df.index[-1], "KDJ_K"] = 73.0
        df.at[df.index[-1], "KDJ_D"] = 78.0
        upper = df["BB_upper"].iloc[-1]
        df.at[df.index[-1], "close"] = upper
        df.at[df.index[-1], "volume"] = 5000000.0
        df.at[df.index[-1], "VOL_MA20"] = 2000000.0
        df.at[df.index[-1], "pct_chg"] = -2.0

        result = self.fusion.evaluate(df)
        assert result.action == "sell"
        assert result.confidence >= 0.3

    def test_mixed_signals_hold(self):
        df = _make_signal_df_with_indicators(60)
        # MACD看多，RSI看空，MA看空 → 混合
        df.at[df.index[-2], "MACD"] = -0.10
        df.at[df.index[-2], "MACD_signal"] = -0.05
        df.at[df.index[-1], "MACD"] = 0.05
        df.at[df.index[-1], "MACD_signal"] = 0.03
        df.at[df.index[-1], "RSI14"] = 80.0
        df.at[df.index[-1], "MA5"] = 12.0
        df.at[df.index[-1], "MA20"] = 15.0
        df.at[df.index[-1], "MA10"] = 14.0
        # 其他指标不给信号
        df.at[df.index[-1], "volume"] = df["VOL_MA20"].iloc[-1]

        result = self.fusion.evaluate(df)
        assert result.action == "hold"

    def test_short_df_returns_hold(self):
        df = _make_signal_df_with_indicators(1)
        result = self.fusion.evaluate(df)
        assert result.action == "hold"
        assert result.score == 0.0
        assert result.confidence == 0.0
