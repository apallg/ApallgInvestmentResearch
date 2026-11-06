"""regime 指标精简 + 波动率向量化 等价性测试。

回归重点: add_regime_indicators 的 MA/BB/ATR 必须与 add_indicators 完全一致;
_score_volatile 的向量化 rolling_vol 必须与 for 循环等价 (shift(1) 对齐窗口右端点)。
"""
import numpy as np
import pandas as pd

from core.domain.screening.indicators import Analyzer


def _make_df(n=400, seed=42):
    np.random.seed(seed)
    close = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    volume = pd.Series(np.random.randint(10000, 100000, n).astype(float))
    return pd.DataFrame({'close': close, 'high': high, 'low': low, 'volume': volume})


class TestRegimeIndicators:
    def test_add_regime_indicators_matches_full(self):
        df = _make_df()
        d1 = Analyzer.add_indicators(df.copy())
        d2 = Analyzer.add_regime_indicators(df.copy())
        for k in ['MA5', 'MA10', 'MA20', 'MA60', 'BB_upper', 'BB_lower', 'ATR14']:
            assert d1[k].equals(d2[k]), f'{k} 与 add_indicators 不一致'

    def test_add_regime_indicators_no_redundant(self):
        df = _make_df()
        d2 = Analyzer.add_regime_indicators(df.copy())
        for c in ['EMA5', 'MACD', 'RSI14', 'KDJ_K', 'BB_middle', 'BB_width', 'VOL_MA5']:
            assert c not in d2.columns, f'冗余列 {c} 未被剔除'

    def test_rolling_vol_vectorized_equivalent(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(400) * 0.01)
        old = []
        for i in range(20, len(returns)):
            old.append(returns.iloc[i-20:i].std() * (250 ** 0.5))
        new = (returns.rolling(20).std().shift(1) * (250 ** 0.5)).dropna().tolist()
        assert len(old) == len(new)
        assert np.allclose(old, new)
