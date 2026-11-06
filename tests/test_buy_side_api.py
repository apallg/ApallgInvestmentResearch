"""
BuySideScreener 公开 API 测试
覆盖: get_qlib_predictions 封装, 无模型降级, weekly_rebalance 调用链
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import datetime

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGetQlibPredictions:
    """get_qlib_predictions() — 公开 API 封装 _find + _load"""

    def test_returns_empty_when_no_model(self):
        """无可用模型时返回 (None, {})"""
        from core.domain.screening._screener import BuySideScreener
        screener = BuySideScreener()
        with patch.object(screener, '_find_best_qlib_model', return_value=None):
            model, scores = screener.get_qlib_predictions()
        assert model is None
        assert scores == {}

    def test_delegates_to_load_when_model_found(self):
        """模型存在时返回 (model, scores) 元组"""
        from core.domain.screening._screener import BuySideScreener
        screener = BuySideScreener()
        mock_model = ("exp_1", "run_abc123")
        mock_scores = {"000001.SZ": 0.05, "600519.SH": 0.12}
        today = datetime.date(2026, 8, 3)

        with patch.object(screener, '_find_best_qlib_model', return_value=mock_model):
            with patch.object(screener, '_load_qlib_predictions', return_value=mock_scores) as mock_load:
                model, scores = screener.get_qlib_predictions(as_of=today, pool="all")

        assert model == mock_model
        assert scores == mock_scores
        mock_load.assert_called_once_with(mock_model, as_of=today, pool="all")

    def test_default_parameters(self):
        """默认参数: as_of=None, pool=csi300"""
        from core.domain.screening._screener import BuySideScreener
        screener = BuySideScreener()
        mock_model = ("exp_1", "run_abc")
        with patch.object(screener, '_find_best_qlib_model', return_value=mock_model):
            with patch.object(screener, '_load_qlib_predictions', return_value={}) as mock_load:
                model, scores = screener.get_qlib_predictions()
        assert model == mock_model
        assert scores == {}
        mock_load.assert_called_once_with(mock_model, as_of=None, pool="csi300")

    def test_load_returns_empty_scores(self):
        """模型存在但预测返回空 (过时数据)"""
        from core.domain.screening._screener import BuySideScreener
        screener = BuySideScreener()
        with patch.object(screener, '_find_best_qlib_model', return_value=("e", "r")):
            with patch.object(screener, '_load_qlib_predictions', return_value={}):
                model, scores = screener.get_qlib_predictions()
        assert model == ("e", "r")
        assert scores == {}

    def test_method_exists_on_class(self):
        """确认 get_qlib_predictions 是公开方法 (无下划线前缀)"""
        from core.domain.screening._screener import BuySideScreener
        assert hasattr(BuySideScreener, 'get_qlib_predictions')
        assert not hasattr(BuySideScreener, '_get_qlib_predictions')


class TestWeeklyRebalanceDecoupling:
    """weekly_rebalance 不再直接调用私有方法"""

    def test_weekly_rebalance_imports(self):
        """weekly_rebalance 模块能正常导入"""
        from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor
        assert WeeklyRebalanceAdvisor is not None

    def test_uses_public_api(self):
        """weekly_rebalance 的 generate_advice 使用 get_qlib_predictions"""
        import ast
        import re
        from pathlib import Path

        rebalance_path = Path(__file__).parent.parent / "core" / "weekly_rebalance.py"
        source = rebalance_path.read_text(encoding="utf-8")

        # 私有方法不应出现在 weekly_rebalance.py 中
        assert '._find_best_qlib_model' not in source, \
            "weekly_rebalance.py 仍包含 _find_best_qlib_model 调用"
        assert '._load_qlib_predictions(' not in source, \
            "weekly_rebalance.py 仍包含 _load_qlib_predictions 调用"

        # 公开 API 应该出现
        assert '.get_qlib_predictions(' in source, \
            "weekly_rebalance.py 未使用 get_qlib_predictions"
