"""
RegimePredictorV2 单元测试

覆盖:
  - 连续目标生成 (_make_regression_targets)
  - RBF 映射 (_ret_vol_to_pseudo_probs, _ret_vol_to_ratio, _ret_vol_to_weights)
  - 训练 (fit) — 正常/数据不足
  - 预测 (predict_regime, predict_raw, predict_from_df)
  - 持久化 (_save/_load)
  - 边界: NaN, 空数据, 极值, 并发, 特征异常
  - 返回值格式校验
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import warnings

warnings.filterwarnings("ignore")


# ── Fixtures ──

@pytest.fixture
def simple_index_df():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 4000 + np.cumsum(np.random.randn(n) * 25)
    close = np.maximum(close, 2000)

    df = pd.DataFrame({
        "date": dates,
        "open": close * (1 + np.random.randn(n) * 0.002),
        "high": close * (1 + np.abs(np.random.randn(n) * 0.008)),
        "low": close * (1 - np.abs(np.random.randn(n) * 0.008)),
        "close": close,
        "volume": np.random.randint(1e8, 5e8, n),
        "amount": close * np.random.randint(1e8, 5e8, n) / 1000,
    })
    return df


@pytest.fixture
def tmp_model_dir():
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ── 回归目标生成 ──

class TestRegressionTargets:

    def test_output_shapes(self, simple_index_df):
        from core.domain.regime.predictor_v2 import _make_regression_targets
        y_ret, y_vol, valid = _make_regression_targets(simple_index_df, horizon=10)
        assert len(y_ret) == len(simple_index_df)
        assert len(y_vol) == len(simple_index_df)
        assert len(valid) == len(simple_index_df)

    def test_last_horizon_days_invalid(self, simple_index_df):
        from core.domain.regime.predictor_v2 import _make_regression_targets
        _, _, valid = _make_regression_targets(simple_index_df, horizon=10)
        assert not valid[-10:].any()  # 最后10天无目标

    def test_all_nan_data(self):
        from core.domain.regime.predictor_v2 import _make_regression_targets
        df = pd.DataFrame({"close": [np.nan] * 30})
        y_ret, y_vol, valid = _make_regression_targets(df, horizon=10)
        assert not valid.any()

    def test_constant_price(self):
        from core.domain.regime.predictor_v2 import _make_regression_targets
        df = pd.DataFrame({"close": np.ones(50) * 4000})
        y_ret, y_vol, valid = _make_regression_targets(df, horizon=10)
        assert valid[:40].all()
        assert np.allclose(y_ret[:40], 0.0)  # 常数价→零收益

    def test_steep_uptrend(self):
        from core.domain.regime.predictor_v2 import _make_regression_targets
        n = 60
        df = pd.DataFrame({"close": np.linspace(1000, 2000, n)})
        y_ret, y_vol, valid = _make_regression_targets(df, horizon=10)
        # 暴涨 → ret 大幅正, vol 非零
        assert y_ret[0] > 0.1  # 前段涨超 10%
        assert y_vol[0] > 0


# ── RBF 映射函数 ──

class TestRbfMapping:

    def test_center_anchor_returns_high_prob(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_pseudo_probs
        # 在 anchor 0 (volatile_up: ret=3.8%, vol=22.5%) 附近
        probs = _ret_vol_to_pseudo_probs(0.038, 0.225)
        assert probs[0] > 0.3  # volatile_up 概率最高
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_extreme_ret_returns_uniform(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_pseudo_probs
        probs = _ret_vol_to_pseudo_probs(0.50, 0.20)
        # 极端收益 → 距离所有锚点都远 → 接近均匀
        assert max(probs) < 0.5
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_nan_returns_uniform(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_pseudo_probs
        probs = _ret_vol_to_pseudo_probs(np.nan, 0.20)
        assert np.allclose(probs, np.ones(6) / 6)

    def test_negative_vol_handled(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_pseudo_probs
        probs = _ret_vol_to_pseudo_probs(0.01, -0.05)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_zero_vol_ratio(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_ratio
        ratio = _ret_vol_to_ratio(0.01, 0.0)
        assert 0.2 <= ratio <= 0.9

    def test_bull_case_ratio(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_ratio
        ratio = _ret_vol_to_ratio(0.05, 0.10)
        assert ratio > 0.60

    def test_bear_case_ratio(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_ratio
        ratio = _ret_vol_to_ratio(-0.05, 0.25)
        assert ratio < 0.40

    def test_nan_ratio_default(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_ratio
        assert _ret_vol_to_ratio(np.nan, 0.1) == 0.50
        assert _ret_vol_to_ratio(0.01, np.nan) == 0.50

    def test_weights_return_dict(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_weights
        w = _ret_vol_to_weights(0.02, 0.15)
        assert isinstance(w, dict)
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_weights_extreme_ret(self):
        from core.domain.regime.predictor_v2 import _ret_vol_to_weights
        w = _ret_vol_to_weights(0.50, 0.50)
        assert isinstance(w, dict)
        assert len(w) > 0
        assert abs(sum(w.values()) - 1.0) < 0.01


# ── 训练 ──

class TestFit:

    def test_insufficient_data_raises(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        with pytest.raises(RuntimeError):
            model.fit("2020-01-01", "2020-01-05", "2020-01-06", "2020-01-10")

    def test_is_loaded_before_fit(self):
        from core.domain.regime.predictor_v2 import RegimePredictorV2
        model = RegimePredictorV2()
        assert not model._is_loaded()

    def test_predict_without_model(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=99)
        result = model.predict_regime()
        assert result.get("error") is not None
        assert result["position_ratio"] is None

    def test_fit_saves_model(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        info = model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        assert info["n_train"] > 200
        assert model._is_loaded()
        assert (tmp_model_dir / "regime_v2_h10.pkl").exists()

    def test_metrics_in_result(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        info = model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        assert "ret_r2" in info
        assert "vol_r2" in info
        assert "ret_mae" in info
        assert "vol_mae" in info


# ── 预测格式 ──

class TestPredictOutput:

    def test_unloaded_returns_error_dict(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=99)
        result = model.predict_regime()
        assert result.get("error") is not None
        assert result["probs"] is None
        assert result["pred_ret"] is None
        assert result["pred_vol"] is None

    def test_loaded_output_keys(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        result = model.predict_regime()
        required_keys = {"probs", "position_ratio", "signal_weights",
                        "confidence", "top_state", "pred_ret", "pred_vol",
                        "model_type", "as_of"}
        assert required_keys.issubset(set(result.keys()))
        assert result["model_type"] == "regression"
        assert result.get("error") is None

    def test_probs_sum_to_one(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        result = model.predict_regime()
        probs = result["probs"]
        assert probs is not None
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_position_in_bounds(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        result = model.predict_regime()
        assert 0.2 <= result["position_ratio"] <= 0.9

    def test_pred_ret_clipped(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        result = model.predict_regime()
        assert result["pred_ret"] is not None
        assert -0.16 <= result["pred_ret"] <= 0.16  # clipped

    def test_pred_vol_positive(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        result = model.predict_regime()
        assert result["pred_vol"] > 0


# ── 持久化 ──

class TestPersistence:

    def test_save_load_roundtrip(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)

        model1 = rpv2.RegimePredictorV2(label_horizon=10)
        model1.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        pred1 = model1.predict_regime()

        model2 = rpv2.RegimePredictorV2(label_horizon=10)
        model2._load()
        pred2 = model2.predict_regime()

        assert pred1["pred_ret"] == pred2["pred_ret"]
        assert pred1["pred_vol"] == pred2["pred_vol"]
        assert pred1["position_ratio"] == pred2["position_ratio"]

    def test_load_nonexistent_returns_false(self):
        from core.domain.regime.predictor_v2 import RegimePredictorV2
        model = RegimePredictorV2(label_horizon=99)
        assert not model._is_loaded()

    def test_load_path_nonexistent(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2()
        model._load_path(tmp_model_dir / "nonexistent.pkl")
        assert not model._is_loaded()


# ── predict_from_df ──

class TestPredictFromDf:

    def test_unloaded_returns_none(self, simple_index_df, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=99)
        ratio, weights = model.predict_from_df(simple_index_df)
        assert ratio is None
        assert weights is None

    def test_loaded_returns_values(self, simple_index_df, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        ratio, weights = model.predict_from_df(simple_index_df)
        assert ratio is not None
        assert 0.2 <= ratio <= 0.9
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_empty_df(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        # 空 DataFrame 不应该崩溃
        empty_df = pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns]"),
            "close": pd.Series([], dtype=float),
            "open": pd.Series([], dtype=float),
            "high": pd.Series([], dtype=float),
            "low": pd.Series([], dtype=float),
            "volume": pd.Series([], dtype=float),
            "amount": pd.Series([], dtype=float),
        })
        ratio, _ = model.predict_from_df(empty_df)
        # 空数据 → 返回 None
        assert ratio is None or (0.2 <= ratio <= 0.9)


# ── 并发 ──

class TestConcurrency:

    def test_singleton_lock_creation(self):
        from core.domain.regime.predictor_v2 import _get_regime_v2_lock
        lock1 = _get_regime_v2_lock()
        lock2 = _get_regime_v2_lock()
        assert lock1 is lock2

    def test_get_regime_predictor_v2_idempotent(self):
        from core.domain.regime.predictor_v2 import get_regime_predictor_v2
        p1 = get_regime_predictor_v2()
        p2 = get_regime_predictor_v2()
        # 两次调用返回同一个实例
        assert p1 is p2


# ── 异常路径 ──

class TestExceptionPaths:

    def test_predict_raw_unloaded(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=99)
        ret, vol = model.predict_raw()
        assert ret is None
        assert vol is None

    def test_empty_df_all_nan_features(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        # 全是 NaN 的 df — 不应崩溃
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100),
            "open": [np.nan] * 100,
            "high": [np.nan] * 100,
            "low": [np.nan] * 100,
            "close": [np.nan] * 100,
            "volume": [np.nan] * 100,
            "amount": [np.nan] * 100,
        })
        ratio, _ = model.predict_from_df(df)
        assert ratio is None or (0.2 <= ratio <= 0.9)

    def test_single_row_df(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        df = pd.DataFrame({
            "date": [pd.Timestamp("2023-06-15")],
            "close": [4000.0], "open": [3990.0], "high": [4010.0], "low": [3980.0],
            "volume": [1e8], "amount": [4e10],
        })
        ratio, _ = model.predict_from_df(df)
        assert ratio is None or (0.2 <= ratio <= 0.9)


# ── 边界值: predict_raw 返回值范围 ──

class TestRawPrediction:

    def test_clipping_applied(self, tmp_model_dir, monkeypatch):
        import core.regime_predictor_v2 as rpv2
        monkeypatch.setattr(rpv2, 'MODEL_DIR', tmp_model_dir)
        model = rpv2.RegimePredictorV2(label_horizon=10)
        model.fit("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31")
        # _predict_continuous 内部做了 clip, 这里验证公开 API
        ret, vol = model.predict_raw()
        if ret is not None:
            assert -0.16 <= ret <= 0.16
        if vol is not None:
            assert 0.02 <= vol <= 0.50
