"""PIT 服务 — 基本面 PIT 同步/校验编排。"""
from __future__ import annotations

import logging
from backend.service.qlib_access import qlib_read_locked

logger = logging.getLogger(__name__)


class PitService:

    @staticmethod
    def sync_fundamental_pit(verify_samples=10, batch_size=500, refresh_valuation=True):
        """同步基本面 PIT 数据到 qlib 特征目录。"""
        from core.adapters.qlib.generation import assert_no_active_historical_release
        from core.shared import config

        assert_no_active_historical_release(config.QLIB_DATA_ROOT)
        from core.adapters.qlib.pit_bridge import PitBridge
        result = PitBridge.sync_fundamental_pit(
            verify_samples=verify_samples,
            batch_size=batch_size,
            refresh_valuation=refresh_valuation,
        )
        return {"data": result or {"message": "无基本面数据可同步"}}

    @staticmethod
    @qlib_read_locked
    def check_pit():
        """检查基本面 PIT 文件是否已生成且字段完整。"""
        from core.adapters.qlib.generation import assert_no_active_historical_release
        from core.shared import config
        from pathlib import Path

        assert_no_active_historical_release(config.QLIB_DATA_ROOT)
        data_dir = Path(config.QLIB_DATA_DIR)
        features_dir = data_dir / "features"
        fields = ["roe", "net_profit_yoy", "revenue_yoy", "debt_to_assets", "eps", "pb", "pe"]
        checks = []
        passed = True
        if not features_dir.exists():
            return {"data": {"passed": False, "checks": [{"item": "features", "ok": False, "msg": "features 目录不存在"}]}}

        sample_codes = ["sh600519", "sz000001", "sh600036", "sz000858"]
        for code in sample_codes:
            stock_dir = features_dir / code
            if not stock_dir.exists():
                checks.append({"item": f"{code}", "ok": False, "msg": "目录不存在"})
                passed = False
                continue
            missing = [f for f in fields if not (stock_dir / f"{f}.day.bin").exists()]
            if missing:
                checks.append({"item": code, "ok": False, "msg": f"缺失字段: {', '.join(missing)}"})
                passed = False
            else:
                checks.append({"item": code, "ok": True, "msg": "PIT 字段完整"})
        return {"data": {"passed": passed, "checks": checks}}
