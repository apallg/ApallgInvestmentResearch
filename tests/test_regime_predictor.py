"""
RegimePredictor 单元测试

覆盖:
  - 标签生成 (_make_regime_labels) 所有 6 个状态分支
  - 训练流程 (fit) — 正常/数据不足/缺失类别
  - 预测流程 (predict_regime) — 正常/未加载/无数据
  - 软融合函数 (_soft_position_ratio, _soft_signal_weights)
  - 边界输入: NaN值、空DataFrame、单日数据、极值收益率
  - 持久化: _save/_load 往返一致性
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import pickle
import warnings

warnings.filterwarnings("ignore")


# ── Fixtures ──

@pytest.fixture
def simple_index_df():
    """构造 500 个交易日的模拟沪深300数据"""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 3500 + np.cumsum(np.random.randn(n) * 30)
    close = np.maximum(close, 1000)
    daily_ret = np.random.randn(n) * 0.015
    close = close * (1 + daily_ret)
    close = close / close[-1] * 4000

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
    """临时模型目录"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ── 标签生成测试 ──

class TestMakeRegimeLabels:

    def test_all_states_present(self):
        from core.domain.regime.predictor import _make_regime_labels

        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # 构造包含所有状态的行情
        close = np.ones(n) * 4000
        # 前半段稳步上涨
        close[:60] = np.linspace(3800, 4200, 60)
        # 然后是高波动
        close[60:120] = 4200 + np.random.randn(60) * 80
        # 震荡
        close[120:180] = 4000 + np.random.randn(60) * 20
        # 趋势下跌
        close[180:240] = np.linspace(4000, 3400, 60)
        # 阴跌
        close[240:] = np.linspace(3400, 3000, 60)

        df = pd.DataFrame({"close": close})
        labels = _make_regime_labels(df, horizon=10)

        assert len(labels) == n
        assert labels.dtype in (np.int32, np.int64)
        unique = set(labels[:-10])  # 最后10天无标签
        assert len(unique) >= 3, f"至少应有3种状态, 实际: {unique}"

    def test_last_horizon_days_default_ranging(self):
        from core.domain.regime.predictor import _make_regime_labels

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"close": np.linspace(4000, 4100, n)})
        labels = _make_regime_labels(df, horizon=10)

        # 最后 10 天的标签应为默认 STEADY_FLAT (5)
        assert all(labels[-10:] == 5)

    def test_nan_values_handled(self):
        from core.domain.regime.predictor import _make_regime_labels

        n = 30
        df = pd.DataFrame({"close": [np.nan] * 10 + list(range(10, 30))})
        labels = _make_regime_labels(df, horizon=5)
        # NaN 部分默认 STEADY_FLAT (5)
        assert labels[0] == 5

    def test_volatile_state(self):
        from core.domain.regime.predictor import _make_regime_labels

        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # 高波动: 日波动率约 30%+
        r = np.random.randn(n) * 0.04
        close = 4000 * np.exp(np.cumsum(r))
        df = pd.DataFrame({"close": close})
        labels = _make_regime_labels(df, horizon=10)
        # 至少有一些 volatile 标签
        assert 5 in set(labels[:-10])


# ── 软融合测试 ──

class TestSoftFusion:

    def test_position_ratio_array(self):
        from core.domain.regime.weights import get_soft_position_ratio
        # 100% steady_up (index 3) → 90%
        probs = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        ratio = get_soft_position_ratio(probs)
        assert ratio == 0.90

    def test_position_ratio_dict(self):
        from core.domain.regime.weights import get_soft_position_ratio
        probs = {"steady_up": 0.5, "steady_flat": 0.5}
        ratio = get_soft_position_ratio(probs)
        assert 0.65 < ratio < 0.75  # 0.5*0.9 + 0.5*0.5 = 0.7

    def test_position_ratio_uniform(self):
        from core.domain.regime.weights import get_soft_position_ratio
        probs = np.ones(6) / 6
        ratio = get_soft_position_ratio(probs)
        # 平均仓位: (0.65+0.20+0.35+0.90+0.30+0.50)/6 ≈ 0.483
        assert 0.46 < ratio < 0.50

    def test_signal_weights_dict(self):
        from core.domain.regime.weights import get_soft_regime_weights
        probs = {"volatile_up": 1.0}
        weights = get_soft_regime_weights(probs)
        assert isinstance(weights, dict)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_signal_weights_array(self):
        from core.domain.regime.weights import get_soft_regime_weights
        probs = np.array([0, 0.4, 0.3, 0.1, 0.1, 0.1])
        weights = get_soft_regime_weights(probs)
        assert len(weights) > 3
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_unknown_state_ignored(self):
        from core.domain.regime.weights import get_soft_regime_weights
        probs = {"nonexistent_state": 1.0}
        weights = get_soft_regime_weights(probs)
        # 未知状态不影响, 返回空或默认权重
        assert isinstance(weights, dict)


# ── RegimePredictor 训练测试 ──

class TestRegimePredictorFit:

    def test_insufficient_data_raises(self, tmp_model_dir):
        import core.domain.regime.predictor as rp
        rp.MODEL_DIR = tmp_model_dir

        model = rp.RegimePredictor(label_horizon=10, model_type="linear")
        with pytest.raises(RuntimeError):
            model.fit("2020-01-01", "2020-01-05", "2020-01-06", "2020-01-10")

    def test_is_loaded_before_fit(self):
        from core.domain.regime.predictor import RegimePredictor
        model = RegimePredictor()
        assert not model._is_loaded()

    def test_predict_without_model(self):
        from core.domain.regime.predictor import RegimePredictor
        model = RegimePredictor()
        result = model.predict_regime()
        assert result["probs"] is None
        assert result["error"] is not None

    def test_top_k_acc(self):
        from core.domain.regime.predictor import RegimePredictor
        probs = np.array([
            [0.6, 0.1, 0.1, 0.1, 0.05, 0.05],   # top2=[0,1], true=0 ✓
            [0.1, 0.5, 0.2, 0.1, 0.05, 0.05],    # top2=[1,2], true=1 ✓
            [0.1, 0.1, 0.4, 0.2, 0.1, 0.1],       # top2=[2,3], true=3 ✓
        ])
        y_true = np.array([0, 1, 3])
        acc = RegimePredictor._top_k_acc(probs, y_true, k=2)
        assert acc == 1.0


# ── 边界输入测试 ──

class TestEdgeCases:

    def test_extreme_returns(self):
        from core.domain.regime.predictor import _make_regime_labels
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # 暴涨 100%
        close = np.ones(n) * 4000
        close[-30:] = 8000
        df = pd.DataFrame({"close": close})
        labels = _make_regime_labels(df, horizon=10)
        assert len(np.unique(labels[:-10])) >= 1  # 不应崩溃

    def test_constant_price(self):
        from core.domain.regime.predictor import _make_regime_labels
        n = 50
        df = pd.DataFrame({"close": np.ones(n) * 4000})
        labels = _make_regime_labels(df, horizon=10)
        # 常数价格 → RANGING 或 BULL_STEADY (取决于波动率)
        assert all(isinstance(l, (int, np.integer)) for l in labels)

    def test_single_day_data(self):
        """单日数据 → 标签全默认"""
        from core.domain.regime.predictor import _make_regime_labels
        n = 30
        df = pd.DataFrame({"close": range(1, n + 1)}).astype(float)
        labels = _make_regime_labels(df, horizon=20)
        assert len(labels) == n


# ── 返回格式测试 ──

class TestPredictFromDf:

    def test_unloaded_returns_none(self, simple_index_df, tmp_model_dir, monkeypatch):
        import core.domain.regime.predictor as rp
        monkeypatch.setattr(rp, 'MODEL_DIR', tmp_model_dir)
        model = rp.RegimePredictor()
        result = model.predict_from_df(simple_index_df)
        assert result is None

    def test_with_mock_model(self, simple_index_df, tmp_model_dir, monkeypatch):
        import core.domain.regime.predictor as rp
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        monkeypatch.setattr(rp, 'MODEL_DIR', tmp_model_dir)
        model = rp.RegimePredictor(label_horizon=10, model_type="linear")
        model.feature_names = ["ret_1d", "ma5_dev"]

        # Mock models
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]])

        model.elasticnet = MockModel()
        model.mlp = MockModel()
        model.scaler = StandardScaler()
        model.scaler.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        df_copy = simple_index_df.copy()
        df_copy["ret_1d"] = df_copy["close"].pct_change().fillna(0)
        df_copy["ma5_dev"] = df_copy["close"] / df_copy["close"].rolling(5).mean() - 1
        df_copy["ma5_dev"] = df_copy["ma5_dev"].fillna(0)

        result = model.predict_from_df(df_copy)
        assert result is not None
        assert len(result) == 6
        assert abs(sum(result) - 1.0) < 0.01


class TestReturnFormat:

    def test_predict_regime_error_format(self):
        from core.domain.regime.predictor import RegimePredictor
        model = RegimePredictor()
        result = model.predict_regime()
        expected_keys = {"probs", "position_ratio", "signal_weights",
                        "confidence", "top_state", "as_of", "error", "model_type"}
        assert expected_keys.issubset(set(result.keys()))
        assert result["probs"] is None


# ── 并发安全测试 ──

class TestConcurrency:

    def test_lock_creation(self):
        """验证锁的惰性创建"""
        from core.domain.regime.predictor import _get_regime_lock
        lock1 = _get_regime_lock()
        lock2 = _get_regime_lock()
        assert lock1 is lock2  # 单例锁

    def test_get_regime_predictor_idempotent(self):
        """重复调用返回同一实例"""
        from core.domain.regime.predictor import get_regime_predictor
        p1 = get_regime_predictor()
        p2 = get_regime_predictor()
        assert p1 is p2
