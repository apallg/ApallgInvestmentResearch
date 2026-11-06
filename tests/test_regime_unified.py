"""统一市场状态入口 (get_unified_regime) 单元测试 — 零 I/O。

回归重点:
  - RuleBasedRegimeClassifier 回退输出 6 状态
  - get_unified_regime 主模型失败时回退规则, apply_macro=False 不带 macro
  - _score_regime_bias 扩展到 6 状态键
  - EMA 平滑保持概率和=1
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def index_df():
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 3500 + np.cumsum(np.random.randn(n) * 20)
    close = np.maximum(close, 1000)
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.random.randint(1e8, 5e8, n),
    })


class TestRegimeUnified:
    def test_rule_based_returns_six_state(self, index_df):
        from core.domain.regime.classifier import RuleBasedRegimeClassifier
        rc = RuleBasedRegimeClassifier(index_df)
        rr = rc.classify(as_of=index_df["date"].iloc[300])
        assert rr.regime.value in (
            "volatile_up", "volatile_down", "volatile_flat",
            "steady_up", "steady_down", "steady_flat",
        )

    def test_unified_fallback_and_no_macro(self, monkeypatch, index_df):
        import core.domain.regime.unified as unified
        monkeypatch.setattr(unified, "_predictor_probs", lambda *a, **k: None)
        as_of = index_df["date"].iloc[300]
        rr = unified.get_unified_regime(as_of=as_of, index_df=index_df, apply_macro=False)
        assert rr.regime.value in unified.STATE_NAMES
        assert rr.details.get("method") == "rule_based"
        assert "macro" not in rr.details

    def test_score_regime_bias_six_states(self):
        from core.domain.macro.analyzer import MacroAnalyzer
        from core.domain.macro.data import MacroSnapshot
        snap = MacroSnapshot(
            as_of_date="2024-01-01", pmi=52.0, m1_m2_scissors=2.0,
            ppi_yoy=1.0, cpi_yoy=1.0, shibor_on=1.5, spread_1y_on=0.5,
        )
        bias = MacroAnalyzer()._score_regime_bias(snap)
        assert set(bias.keys()) == {
            "volatile_up", "volatile_down", "volatile_flat",
            "steady_up", "steady_down", "steady_flat",
        }
        # 宏观偏多: up 状态分数应高于 down
        assert bias["steady_up"] > bias["steady_down"]
        assert bias["volatile_up"] > bias["volatile_down"]

    def test_crf_smooth_keeps_sum_one(self):
        import core.domain.regime.unified as unified
        unified._prior_probs = []
        unified._last_date = None
        probs = {"volatile_up": 0.5, "volatile_down": 0.1, "volatile_flat": 0.1,
                 "steady_up": 0.1, "steady_down": 0.1, "steady_flat": 0.1}
        out = unified._crf_smooth(probs)
        assert set(out.keys()) == set(unified.STATE_NAMES)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-3)

    def test_crf_smooth_same_day_dedup(self):
        import core.domain.regime.unified as unified
        unified._prior_probs = []
        unified._last_date = None
        probs = {"volatile_up": 0.5, "volatile_down": 0.1, "volatile_flat": 0.1,
                 "steady_up": 0.1, "steady_down": 0.1, "steady_flat": 0.1}
        for _ in range(5):
            unified._crf_smooth(probs)
        # 同日去重: 只保留 1 个观测, 不按调用次数累积
        assert len(unified._prior_probs) == 1

    def test_forward_filter_suppresses_isolated_flip(self):
        import numpy as np
        from core.domain.regime.unified import _forward_filter, _PRIOR_TRANS, STATE_NAMES
        steady_up = np.array([0.05, 0.05, 0.05, 0.75, 0.05, 0.05])
        flip = np.array([0.05, 0.75, 0.05, 0.05, 0.05, 0.05])
        obs = np.array([steady_up] * 5 + [flip])
        alpha = _forward_filter(obs, _PRIOR_TRANS)
        up_idx = STATE_NAMES.index("steady_up")
        down_idx = STATE_NAMES.index("volatile_down")
        # 转移矩阵对角占优: 孤立翻转被历史 steady_up 抑制
        assert alpha[-1][up_idx] > alpha[-1][down_idx]

    def test_estimate_transition_matrix_row_normalized(self):
        import numpy as np
        from core.domain.regime.predictor import _estimate_transition_matrix
        np.random.seed(42)
        labels = np.random.randint(0, 6, 500)
        trans = _estimate_transition_matrix(labels)
        assert trans.shape == (6, 6)
        assert np.allclose(trans.sum(axis=1), 1.0)
        assert (np.diag(trans) <= 0.95 + 1e-9).all()

    def test_estimate_transition_matrix_learns_self_prob(self):
        import numpy as np
        from core.domain.regime.predictor import _estimate_transition_matrix
        labels = np.full(100, 3, dtype=int)  # 全 steady_up 连续
        trans = _estimate_transition_matrix(labels)
        # 连续同状态 → 自转移被 clip 到上限 0.95 附近
        assert trans[3, 3] == pytest.approx(0.95, abs=0.02)

    def test_transition_matrix_fallback_to_prior(self, monkeypatch):
        import numpy as np
        import core.domain.regime.unified as unified
        unified._trans = None
        unified._trans_loaded = False
        import core.domain.regime.predictor as predictor
        monkeypatch.setattr(predictor, "get_regime_predictor", lambda: None)
        trans = unified._get_transition_matrix()
        # 回退先验: 对角占优 (self_prob=0.9)
        assert (np.diag(trans) > 0.8).all()

    def test_crf_smooth_sequence(self):
        import core.domain.regime.unified as unified
        steady_up = [0.05, 0.05, 0.05, 0.75, 0.05, 0.05]
        flip = [0.05, 0.75, 0.05, 0.05, 0.05, 0.05]
        obs = [steady_up] * 5 + [flip]
        out = unified.crf_smooth_sequence(obs, unified._PRIOR_TRANS)
        assert out is not None
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-3)
        # 孤立翻转被抑制 (回测复用同一纯函数)
        assert out["steady_up"] > out["volatile_down"]
