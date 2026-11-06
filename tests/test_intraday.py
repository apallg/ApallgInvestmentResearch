import pytest
import numpy as np
import pandas as pd
from core.domain.portfolio.intraday import (
    IntradayPullbackDetector,
    IntradayResult,
    SupportLevel,
    PullbackSignal,
    RSI_OVERSOLD,
)


def _make_5min_df(prices, volumes=None):
    """构造标准 5分钟 OHLC DataFrame"""
    n = len(prices)
    data = {
        "open": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": volumes if volumes else [10000] * n,
        "amount": [p * 10000 for p in prices],
    }
    idx = pd.date_range("2026-01-15 09:35", periods=n, freq="5min")
    return pd.DataFrame(data, index=idx)


def _make_hammer_df():
    """构造最后含锤子线的 5分钟 DataFrame — 下影线长, 上影线极短, 实体小"""
    prices = [10.0] * 10
    volumes = [10000] * 10
    df = _make_5min_df(prices, volumes)
    df.iloc[-1, df.columns.get_loc("open")] = 9.98
    df.iloc[-1, df.columns.get_loc("high")] = 9.981
    df.iloc[-1, df.columns.get_loc("low")] = 9.90
    df.iloc[-1, df.columns.get_loc("close")] = 9.96
    return df


def test_vwap_consumer_rejects_minutes_without_verified_semantics(monkeypatch):
    from unittest.mock import Mock
    import backend.service.rebalance as rebalance_service

    df = _make_5min_df([10.0] * 10)
    df.attrs["data_envelope"] = {
        "schema": "data-envelope-v1",
        "source": "ths_live_minute",
        "status": "partial",
        "price_basis": "unknown",
        "unit": {"price": "CNY/share", "volume": "unknown", "amount": "unknown"},
        "source_quantity_semantics": "unknown",
    }
    calculate_vwap = Mock(return_value=10.0)
    monkeypatch.setattr(rebalance_service, "_try_ths_intraday_live", lambda *args: df)
    monkeypatch.setattr(IntradayPullbackDetector, "_calc_vwap", calculate_vwap)

    assert rebalance_service._get_intraday_vwap("000001", "2026-01-15") is None
    calculate_vwap.assert_not_called()


class TestSupportCalculation:

    def test_supports_sorted_by_distance(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([12.50] * 20)
        supports = detector._calc_supports(
            current_price=12.50, open_price=12.60,
            prev_day_low=12.35, vwap=12.52, day_low=12.38,
            ma5=12.30, ma10=12.10, ma20=11.80)
        assert len(supports) >= 4
        for i in range(len(supports) - 1):
            a = supports[i].distance_pct if supports[i].distance_pct >= 0 else float('inf')
            b = supports[i+1].distance_pct if supports[i+1].distance_pct >= 0 else float('inf')
            assert a <= b

    def test_supports_below_price_only(self):
        """MA 支撑位只在低于现价时出现"""
        detector = IntradayPullbackDetector()
        supports = detector._calc_supports(
            current_price=10.0, open_price=10.0,
            prev_day_low=9.5, vwap=None, day_low=None,
            ma5=11.0, ma10=12.0, ma20=9.0)
        names = {s.name for s in supports}
        assert "MA5" not in names
        assert "MA10" not in names
        assert "MA20" in names


class TestRSI:

    def test_rsi_perfect_gain(self):
        detector = IntradayPullbackDetector()
        closes = pd.Series([10.0 + i * 0.1 for i in range(20)])
        rsi = detector._calc_rsi(closes, 14)
        assert rsi == 100.0

    def test_rsi_perfect_loss(self):
        detector = IntradayPullbackDetector()
        closes = pd.Series([20.0 - i * 0.1 for i in range(20)])
        rsi = detector._calc_rsi(closes, 14)
        assert rsi < 1.0

    def test_rsi_insufficient_data(self):
        detector = IntradayPullbackDetector()
        closes = pd.Series([10.0] * 5)
        rsi = detector._calc_rsi(closes, 14)
        assert rsi is None

    def test_rsi_uses_latest_changes_when_only_the_tail_differs(self):
        detector = IntradayPullbackDetector()
        shared_prefix = [100.0 + i for i in range(15)]
        falling_tail = [shared_prefix[-1] - i for i in range(1, 15)]
        rising_tail = [shared_prefix[-1] + i for i in range(1, 15)]

        falling_rsi = detector._calc_rsi(pd.Series(shared_prefix + falling_tail), 14)
        rising_rsi = detector._calc_rsi(pd.Series(shared_prefix + rising_tail), 14)

        assert falling_rsi == 0.0
        assert rising_rsi == 100.0

    def test_rsi_oversold_signal(self):
        detector = IntradayPullbackDetector()
        closes = pd.Series([20.0 - i * 0.3 for i in range(20)])
        rsi = detector._calc_rsi(closes, 14)
        assert rsi is not None
        assert rsi < RSI_OVERSOLD


class TestHammer:

    def test_hammer_last_bar(self):
        detector = IntradayPullbackDetector()
        df = _make_hammer_df()
        result = detector._detect_hammer(df)
        assert result == "最新K线"

    def test_no_hammer_normal_bar(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([10.0] * 10)
        result = detector._detect_hammer(df)
        assert result is None

    def test_no_hammer_doji(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([10.0] * 10)
        df.iloc[-1, df.columns.get_loc("open")] = 10.0
        df.iloc[-1, df.columns.get_loc("close")] = 10.0
        df.iloc[-1, df.columns.get_loc("low")] = 9.95
        df.iloc[-1, df.columns.get_loc("high")] = 10.05
        result = detector._detect_hammer(df)
        assert result is None


class TestVolumeShrink:

    def test_volume_shrink_detected(self):
        detector = IntradayPullbackDetector()
        volumes = [20000] * 10 + [5000] * 5
        df = _make_5min_df([10.0] * 15, volumes)
        shrunk, ratio = detector._detect_volume_shrink(df)
        assert shrunk
        assert ratio < 1.0

    def test_volume_shrink_not_detected(self):
        detector = IntradayPullbackDetector()
        volumes = [10000] * 15
        df = _make_5min_df([10.0] * 15, volumes)
        shrunk, ratio = detector._detect_volume_shrink(df)
        assert not shrunk


class TestEntryScore:

    def test_excellent_entry(self):
        detector = IntradayPullbackDetector()
        signals = [
            PullbackSignal(type="near_support", description="接近前日低点", strength=2),
            PullbackSignal(type="volume_shrink", description="量能萎缩", strength=2),
            PullbackSignal(type="rsi_oversold", description="RSI超卖", strength=2),
            PullbackSignal(type="hammer", description="锤子线", strength=2),
        ]
        supports = [SupportLevel(name="前日低点", price=9.50, distance_pct=0.003)]
        score, label = detector._score_entry(supports, signals, 9.53, 9.55, 0.3, 18.0)
        assert score >= 70
        assert label == "优秀"

    def test_poor_entry(self):
        detector = IntradayPullbackDetector()
        signals = []
        supports = [SupportLevel(name="前日低点", price=9.0, distance_pct=0.05)]
        score, label = detector._score_entry(supports, signals, 9.45, 9.5, 1.0, 60.0)
        assert score < 30
        assert label == "观望"


class TestFallback:

    def test_empty_df_fallback(self):
        detector = IntradayPullbackDetector()
        result = detector.analyze(
            df_5min=None, prev_day_low=10.0, prev_day_high=11.0,
            prev_day_close=10.5, reference_date="2026-01-15",
            ma5=10.3, ma10=10.1, ma20=9.8)
        assert isinstance(result, IntradayResult)
        assert result.entry_quality_score <= 50
        # 缺日内数据不是日内风险, risk_warning=None 避免 decide_execution 否决买入
        assert result.risk_warning is None

    def test_empty_dataframe_graceful(self):
        detector = IntradayPullbackDetector()
        df = pd.DataFrame()
        result = detector.analyze(
            df_5min=df, prev_day_low=10.0, prev_day_high=11.0,
            prev_day_close=10.5, reference_date="2026-01-15")
        assert isinstance(result, IntradayResult)
        assert result.entry_quality_score <= 50


class TestAnalyzeIntegration:

    def test_full_analyze(self):
        detector = IntradayPullbackDetector()
        volumes = [20000] * 6 + [6000] * 10 + [5000] * 4
        df = _make_5min_df([10.5 - i * 0.02 for i in range(20)], volumes)
        result = detector.analyze(
            df_5min=df, prev_day_low=9.8, prev_day_high=11.0,
            prev_day_close=10.5, reference_date="2026-01-15",
            ma5=10.2, ma10=10.0, ma20=9.5)
        assert isinstance(result, IntradayResult)
        assert result.current_price > 0
        assert result.day_high >= result.day_low
        assert len(result.supports) > 0
        assert 0 <= result.entry_quality_score <= 100

    def test_recent_bars_format(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([10.0] * 50)
        result = detector.analyze(
            df_5min=df, prev_day_low=9.8, prev_day_high=11.0,
            prev_day_close=10.0, reference_date="2026-01-15")
        assert len(result.recent_bars) == 48
        bar = result.recent_bars[0]
        assert "time" in bar
        assert "open" in bar
        assert "close" in bar

    def test_vwap_calculation(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([10.0] * 10)
        vwap = detector._calc_vwap(df)
        assert 9.9 < vwap < 10.1


class TestNoLookahead:

    def test_deterministic(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([10.0 + i * 0.01 for i in range(30)])
        np.random.seed(42)
        result1 = detector.analyze(
            df_5min=df, prev_day_low=9.8, prev_day_high=10.5,
            prev_day_close=10.0, reference_date="2026-01-15")
        result2 = detector.analyze(
            df_5min=df, prev_day_low=9.8, prev_day_high=10.5,
            prev_day_close=10.0, reference_date="2026-01-15")
        assert result1.entry_quality_score == result2.entry_quality_score
        assert result1.signals == result2.signals


class TestRiskCheck:

    def test_high_open_low_close_risk(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([11.0 - i * 0.05 for i in range(20)])
        risk = detector._check_risk(df, 10.0, 11.0, 10.8, 10.5, None)
        assert risk is not None
        assert "高开低走" in risk

    def test_no_risk(self):
        detector = IntradayPullbackDetector()
        df = _make_5min_df([10.0 + i * 0.01 for i in range(20)])
        risk = detector._check_risk(df, 10.2, 10.0, 10.5, 10.0, None)
        assert risk is None
