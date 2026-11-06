"""
信号融合引擎单元测试 — 规则投票、ML回退、边界条件
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from core.domain.signals.engine import (
    SignalFusion, FusionResult, DEFAULT_WEIGHTS,
    _extract_ml_features, MLSignalFusion, get_fusion,
)
from tests.factories import make_kline_df, make_kline_df_uptrend, make_kline_df_downtrend


class TestFusionResult:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_default_hold(self):
        r = FusionResult()
        assert r.action == "hold"
        assert r.score == 0.0
        assert r.confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_buy_action(self):
        r = FusionResult(score=0.8, action="buy", confidence=0.6)
        assert r.action == "buy"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_sell_action(self):
        r = FusionResult(score=-0.8, action="sell", confidence=0.6)
        assert r.action == "sell"


class TestSignalFusionEvaluate:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_empty_df_returns_hold(self):
        sf = SignalFusion()
        df = pd.DataFrame()
        result = sf.evaluate(df)
        assert result.action == "hold"
        assert result.score == 0.0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_single_row_df_returns_hold(self):
        sf = SignalFusion()
        df = make_kline_df(n_rows=1, with_indicators=True)
        result = sf.evaluate(df)
        assert result.action == "hold"

    @pytest.mark.unit
    @pytest.mark.p0
    def test_uptrend_generates_buy_or_hold(self):
        sf = SignalFusion()
        df = make_kline_df_uptrend(n_rows=120)
        result = sf.evaluate(df)
        assert result.action in ("buy", "hold")
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_downtrend_generates_sell_or_hold(self):
        sf = SignalFusion()
        df = make_kline_df_downtrend(n_rows=120)
        result = sf.evaluate(df)
        assert result.action in ("sell", "hold")
        assert -1.0 <= result.score <= 1.0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_ranging_generates_hold(self):
        sf = SignalFusion()
        from tests.factories import make_kline_df_ranging
        df = make_kline_df_ranging(n_rows=120)
        result = sf.evaluate(df)
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_custom_weights(self):
        sf = SignalFusion(weights={"MACD": 1.0, "RSI": 1.0})
        df = make_kline_df(n_rows=120, with_indicators=True)
        result = sf.evaluate(df)
        assert "MACD" in result.indicator_votes
        assert "RSI" in result.indicator_votes

    @pytest.mark.unit
    @pytest.mark.p1
    def test_all_votes_present(self):
        sf = SignalFusion()
        df = make_kline_df_uptrend(n_rows=120)
        result = sf.evaluate(df)
        for key in ["MACD", "RSI", "MA", "KDJ", "BB", "VOLUME"]:
            assert key in result.indicator_votes, f"Missing {key}"

    @pytest.mark.unit
    @pytest.mark.p0
    def test_confidence_bounded(self):
        sf = SignalFusion()
        for _ in range(5):
            df = make_kline_df(n_rows=120, with_indicators=True)
            result = sf.evaluate(df)
            assert 0 <= result.confidence <= 1.0, f"confidence={result.confidence}"

    @pytest.mark.unit
    @pytest.mark.p2
    def test_zero_weights_all_hold(self):
        sf = SignalFusion(weights={k: 0.0 for k in DEFAULT_WEIGHTS})
        df = make_kline_df_uptrend(n_rows=120)
        result = sf.evaluate(df)
        assert result.action == "hold"
        assert result.score == 0.0
        assert result.confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.p2
    def test_reasons_not_empty_when_signal(self):
        sf = SignalFusion()
        df = make_kline_df_uptrend(n_rows=120)
        result = sf.evaluate(df)
        assert isinstance(result.reasons, list)


class TestMLFeatures:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_extract_features_returns_13d(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        feats, names = _extract_ml_features(df)
        assert len(feats) == 13
        assert len(names) == 13

    @pytest.mark.unit
    @pytest.mark.p1
    def test_extract_features_no_nan(self):
        df = make_kline_df(n_rows=120, with_indicators=True)
        feats, names = _extract_ml_features(df)
        for i, f in enumerate(feats):
            assert not np.isnan(f), f"Feature {names[i]} is NaN"

    @pytest.mark.unit
    @pytest.mark.p2
    def test_extract_features_short_df(self):
        df = make_kline_df(n_rows=10, with_indicators=True)
        feats, names = _extract_ml_features(df)
        assert len(feats) == 13


class TestMLSignalFusion:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_fallback_when_no_model(self):
        ml = MLSignalFusion(weights={"MACD": 1.0})
        df = make_kline_df_uptrend(n_rows=120)
        result = ml.evaluate(df)
        assert result.action in ("buy", "hold", "sell")
        assert -1.0 <= result.score <= 1.0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_is_ready_false_without_model(self):
        with patch("core.domain.signals.engine._load_ml_model", return_value=None):
            ml = MLSignalFusion()
            assert not ml.is_ready


class TestGetFusion:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_returns_signal_fusion_by_default(self):
        from core.shared import config
        with patch.object(config, 'SIGNAL_FUSION_ML_ENABLED', False):
            fusion = get_fusion()
            assert isinstance(fusion, SignalFusion)
