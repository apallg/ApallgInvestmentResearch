"""
指数择时模型单元测试 — 仓位映射、特征工程、标签生成
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock

from core.domain.timing.model import (
    prob_to_ratio, _make_features, _make_labels,
    IndexTimingModel, LGB_PARAMS, XGB_PARAMS, get_timing_ratio,
)
from tests.factories import make_kline_df


class TestProbToRatio:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_min_prob_gives_min_ratio(self):
        ratio = prob_to_ratio(0.0)
        assert 0.19 <= ratio <= 0.22

    @pytest.mark.unit
    @pytest.mark.p0
    def test_max_prob_gives_max_ratio(self):
        ratio = prob_to_ratio(1.0)
        assert 0.96 <= ratio <= 1.0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_monotonic_increasing(self):
        ratios = [prob_to_ratio(p) for p in np.linspace(0, 1, 11)]
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i-1], f"Not monotonic at i={i}"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_mid_prob_around_half(self):
        ratio = prob_to_ratio(0.5)
        assert 0.55 <= ratio <= 0.70

    @pytest.mark.unit
    @pytest.mark.p2
    def test_ratio_bounded(self):
        for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            r = prob_to_ratio(p)
            assert 0.19 <= r <= 0.99, f"p={p} → r={r}"


class TestMakeFeatures:

    @pytest.fixture
    def index_df(self):
        n = 300
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 4000 * (1 + rng.normal(0.0003, 0.012, n).cumsum())
        return pd.DataFrame({
            "date": dates,
            "open": close * (1 - rng.uniform(0, 0.005, n)),
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.lognormal(18, 0.5, n),
            "amount": rng.lognormal(22, 0.5, n),
        })

    @pytest.mark.unit
    @pytest.mark.p0
    def test_returns_dataframe(self, index_df):
        result = _make_features(index_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.unit
    @pytest.mark.p0
    def test_has_core_features(self, index_df):
        result = _make_features(index_df)
        expected = [
            "ret_1d", "ret_5d", "ret_10d", "ret_20d",
            "ma5_dev", "ma10_dev", "ma20_dev", "ma60_dev",
            "vol_5d", "vol_10d", "vol_20d",
            "vol_ratio", "rsi_14", "macd", "bb_position",
            "consecutive", "amplitude",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    @pytest.mark.unit
    @pytest.mark.p1
    def test_features_with_breadth(self, index_df):
        breadth = pd.DataFrame({
            "trade_date": index_df["date"].dt.strftime("%Y-%m-%d"),
            "up_ratio": np.random.uniform(0.3, 0.7, len(index_df)),
        })
        result = _make_features(index_df, breadth)
        assert "up_ratio" in result.columns

    @pytest.mark.unit
    @pytest.mark.p1
    def test_volume_ratio_finite(self, index_df):
        result = _make_features(index_df)
        valid = result["vol_ratio"].dropna()
        assert (valid > 0).all()

    @pytest.mark.unit
    @pytest.mark.p2
    def test_short_df_does_not_crash(self):
        df = make_kline_df(n_rows=10)
        df["date"] = pd.date_range("2025-01-01", periods=10, freq="B")
        result = _make_features(df)
        assert isinstance(result, pd.DataFrame)


class TestMakeLabels:

    @pytest.fixture
    def idx_df(self):
        n = 300
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 4000 * (1 + rng.normal(0.0003, 0.012, n).cumsum())
        return pd.DataFrame({
            "date": dates,
            "close": close,
        })

    @pytest.mark.unit
    @pytest.mark.p0
    def test_returns_binary(self, idx_df):
        labels = _make_labels(idx_df, horizon=5, threshold=0.015)
        assert set(labels.dropna().unique()).issubset({0, 1})

    @pytest.mark.unit
    @pytest.mark.p1
    def test_last_rows_are_zero(self, idx_df):
        labels = _make_labels(idx_df, horizon=5, threshold=0.015)
        # _make_labels returns int (0/1); last 5 rows have NaN fwd_ret
        # which the > comparison treats as False → 0
        assert (labels.iloc[-5:] == 0).all()

    @pytest.mark.unit
    @pytest.mark.p1
    def test_higher_threshold_fewer_ones(self, idx_df):
        labels_low = _make_labels(idx_df, horizon=5, threshold=0.0)
        labels_high = _make_labels(idx_df, horizon=5, threshold=0.05)
        n_low = labels_low.sum()
        n_high = labels_high.sum()
        assert n_high <= n_low

    @pytest.mark.unit
    @pytest.mark.p2
    def test_different_horizon(self, idx_df):
        labels_5 = _make_labels(idx_df, horizon=5, threshold=0.015)
        labels_10 = _make_labels(idx_df, horizon=10, threshold=0.015)
        # horizon=10: last 10 rows have NaN fwd_ret → 0
        assert (labels_10.iloc[-10:] == 0).all()


class TestModelParams:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_lgb_params_valid(self):
        assert LGB_PARAMS["objective"] == "binary"
        assert LGB_PARAMS["metric"] == "auc"
        assert LGB_PARAMS["max_depth"] >= 3

    @pytest.mark.unit
    @pytest.mark.p1
    def test_xgb_params_valid(self):
        assert XGB_PARAMS["objective"] == "binary:logistic"
        assert XGB_PARAMS["max_depth"] >= 3


class TestModelInit:

    @pytest.mark.unit
    @pytest.mark.p1
    def test_init_defaults(self):
        model = IndexTimingModel()
        assert model.lgb is None
        assert model.xgb is None
        assert model.label_horizon == 5
        assert model.threshold == 0.015

    @pytest.mark.unit
    @pytest.mark.p1
    def test_init_custom_params(self):
        model = IndexTimingModel(label_horizon=10, threshold=0.02)
        assert model.label_horizon == 10
        assert model.threshold == 0.02


class TestTimingPredictionContract:

    def setup_method(self):
        self.index_df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=240),
            "open": np.linspace(3990, 4190, 240),
            "high": np.linspace(4010, 4210, 240),
            "low": np.linspace(3980, 4180, 240),
            "close": np.linspace(4000, 4200, 240),
            "volume": np.linspace(1e8, 2e8, 240),
            "amount": np.linspace(1e10, 2e10, 240),
        })
        self.model = IndexTimingModel()
        self.model.lgb = Mock()
        self.model.xgb = Mock()
        self.model.feature_names = ["ret_1d"]
        self.model.turnover_feature_source = "volume_proxy"
        self.model.last_train_date = "2023-12-29"
        self.model.lgb.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.model.xgb.predict_proba.return_value = np.array([[0.5, 0.5]])

    @pytest.mark.unit
    def test_prediction_rejects_turnover_source_drift(self):
        self.model.turnover_feature_source = "turnover"
        with patch("core.data.fetcher.DataFetcher") as fetcher, \
                patch("core.domain.timing.breadth.get_breadth", return_value=None):
            fetcher.return_value.get_index_daily.return_value = self.index_df
            result = self.model.predict()
        assert result["prob"] is None
        assert "来源与模型训练契约不一致" in result["error"]
        assert result["sample_date"] == str(self.index_df["date"].iloc[-1].date())

    @pytest.mark.unit
    def test_prediction_rejects_legacy_model_without_feature_source_contract(self):
        self.model.turnover_feature_source = None
        result = self.model.predict()
        assert result["prob"] is None
        assert "请重新训练" in result["error"]

    @pytest.mark.unit
    def test_prediction_rejects_missing_required_breadth_feature(self):
        self.model.feature_names = ["turnover_median"]
        with patch("core.data.fetcher.DataFetcher") as fetcher, \
                patch("core.domain.timing.breadth.get_breadth", return_value=None):
            fetcher.return_value.get_index_daily.return_value = self.index_df
            result = self.model.predict()
        assert result["prob"] is None
        assert "turnover_median" in result["error"]

    @pytest.mark.unit
    def test_prediction_reports_actual_sample_and_model_dates(self):
        with patch("core.data.fetcher.DataFetcher") as fetcher, \
                patch("core.domain.timing.breadth.get_breadth", return_value=None):
            fetcher.return_value.get_index_daily.return_value = self.index_df
            result = self.model.predict()
        assert result["prob"] == 0.6
        assert result["as_of"] == str(self.index_df["date"].iloc[-1].date())
        assert result["sample_date"] == result["as_of"]
        assert result["model_last_train_date"] == "2023-12-29"
        assert result["turnover_feature_source"] == "volume_proxy"

    @pytest.mark.unit
    def test_prediction_rejects_non_finite_model_output(self):
        self.model.lgb.predict_proba.return_value = np.array([[np.nan, np.nan]])
        with patch("core.data.fetcher.DataFetcher") as fetcher, \
                patch("core.domain.timing.breadth.get_breadth", return_value=None):
            fetcher.return_value.get_index_daily.return_value = self.index_df
            result = self.model.predict()
        assert result["prob"] is None
        assert result["error"] == "模型输出概率无效"


class TestGetTimingRatio:

    @pytest.mark.unit
    @pytest.mark.p0
    def test_returns_position_ratio(self):
        mock_result = {"position_ratio": 0.8}
        with patch("core.domain.timing.model.IndexTimingModel.predict", return_value=mock_result):
            ratio = get_timing_ratio()
            assert 0.0 <= ratio <= 1.0

    @pytest.mark.unit
    @pytest.mark.p1
    def test_panic_returns_zero(self):
        with patch("core.domain.portfolio.consistency.check_panic", return_value=True):
            ratio = get_timing_ratio()
            assert ratio == 0.0

    @pytest.mark.unit
    @pytest.mark.p2
    def test_predict_error_propagates(self):
        with patch("core.domain.portfolio.consistency.check_panic", return_value=False):
            with patch("core.domain.timing.model.IndexTimingModel.predict", side_effect=RuntimeError):
                with pytest.raises(RuntimeError):
                    get_timing_ratio()
