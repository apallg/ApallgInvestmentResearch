"""PIT 数据桥接 — 封装基本面 PIT 写入/校验，供服务层调用。"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PitBridge:
    """调用现有 scripts.dump_fundamental_to_qlib 完成基本面 PIT 同步。"""

    @staticmethod
    def sync_fundamental_pit(data_dir=None, verify_samples=10, batch_size=500, refresh_valuation=True):
        """将 MySQL fundamental_data 按公告日/法定截止日写入 qlib 特征目录。

        该逻辑已由 scripts/dump_fundamental_to_qlib.py 实现：
        - 优先 ann_date
        - 缺失时按报告期法定披露截止日保守偏移
        - 写入后抽样校验“生效日前不可见”
        """
        from scripts.dump_fundamental_to_qlib import dump_fundamental
        return dump_fundamental(
            data_dir=data_dir,
            verify_samples=verify_samples,
            batch_size=batch_size,
            refresh_valuation=refresh_valuation,
        )

    @staticmethod
    def pit_field(field: str) -> str:
        """返回 qlib 可用的 PIT 因子字段名。

        目前项目按日频 forward-fill 写入普通 features 目录，
        字段名与 FUND_FIELDS 保持一致。
        """
        from core.data.pit_utils import FUND_FIELDS
        allowed = set(FUND_FIELDS)
        if field not in allowed:
            raise ValueError(f"暂不支持 PIT 字段: {field}")
        return field
