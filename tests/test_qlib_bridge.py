"""
Qlib Bridge 测试 — 价格调整校验
verify_price_adjustment 内部懒导入 Database/get_pro, 需 mock 源模块
"""
import datetime
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVerifyPriceAdjustment:
    """verify_price_adjustment — factor=1.0 安全性校验"""

    _DB_PATH = "core.data.Database"
    _PRO_PATH = "core.data.sources.tushare.get_pro"

    def test_skip_when_db_unavailable(self):
        """DB 不可用时跳过"""
        from core.adapters.qlib.bridge import verify_price_adjustment
        with patch(f"{self._DB_PATH}.is_available", return_value=False):
            result = verify_price_adjustment()
        assert result is None

    def test_skip_when_no_codes(self):
        """daily_quotes 为空时跳过"""
        from core.adapters.qlib.bridge import verify_price_adjustment
        with patch(f"{self._DB_PATH}.is_available", return_value=True), \
             patch(f"{self._DB_PATH}.fetchall", return_value=[]):
            result = verify_price_adjustment()
        assert result is None

    def test_skip_when_tushare_unavailable(self):
        """Tushare 不可用时跳过"""
        from core.adapters.qlib.bridge import verify_price_adjustment
        with patch(f"{self._DB_PATH}.is_available", return_value=True), \
             patch(f"{self._DB_PATH}.fetchall", side_effect=[
                [{"code": "600519"}],
             ]), \
             patch(self._PRO_PATH, return_value=None):
            result = verify_price_adjustment()
        assert result is None

    def _make_mock_pro(self):
        import pandas as pd
        mock_pro = MagicMock()
        mock_pro.daily.return_value = pd.DataFrame({
            "trade_date": ["20260802", "20260801"],
            "close": [200.0, 100.0],
        })
        mock_pro.adj_factor.return_value = pd.DataFrame({
            "trade_date": ["20260802", "20260801"],
            "adj_factor": [2.0, 1.0],
        })
        return mock_pro

    def test_ok_when_price_match(self):
        """MySQL 与 Tushare 前复权价格一致 (<0.5%)"""
        from core.adapters.qlib.bridge import verify_price_adjustment
        mock_pro = self._make_mock_pro()

        with patch(f"{self._DB_PATH}.is_available", return_value=True), \
             patch(f"{self._DB_PATH}.fetchall", side_effect=[
                [{"code": "600519"}],
                [
                    {"trade_date": datetime.date(2026, 8, 2), "close": 200.0},
                    {"trade_date": datetime.date(2026, 8, 1), "close": 50.0},
                ],
             ]), \
             patch(self._PRO_PATH, return_value=mock_pro):
            result = verify_price_adjustment()
        assert result is None

    def test_warn_when_price_deviation(self):
        """价格偏差 1% 产生警告 (>0.5%, <5%)"""
        from core.adapters.qlib.bridge import verify_price_adjustment
        mock_pro = self._make_mock_pro()

        with patch(f"{self._DB_PATH}.is_available", return_value=True), \
             patch(f"{self._DB_PATH}.fetchall", side_effect=[
                [{"code": "600519"}],
                [
                    {"trade_date": datetime.date(2026, 8, 2), "close": 200.0},
                    {"trade_date": datetime.date(2026, 8, 1), "close": 51.0},
                ],
             ]), \
             patch(self._PRO_PATH, return_value=mock_pro), \
             patch("core.adapters.qlib.bridge.logger") as mock_logger:
            verify_price_adjustment()
        mock_logger.warning.assert_called()

    def test_error_when_serious_deviation(self):
        """价格偏差 >=5% 报 ERROR"""
        from core.adapters.qlib.bridge import verify_price_adjustment
        mock_pro = self._make_mock_pro()

        with patch(f"{self._DB_PATH}.is_available", return_value=True), \
             patch(f"{self._DB_PATH}.fetchall", side_effect=[
                [{"code": "600519"}],
                [
                    {"trade_date": datetime.date(2026, 8, 2), "close": 200.0},
                    {"trade_date": datetime.date(2026, 8, 1), "close": 60.0},
                ],
             ]), \
             patch(self._PRO_PATH, return_value=mock_pro), \
             patch("core.adapters.qlib.bridge.logger") as mock_logger:
            verify_price_adjustment()
        mock_logger.error.assert_called()
