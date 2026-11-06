"""Qlib 多模型集成适配器 — 封装 qlib.model.ens 相关能力。"""
from __future__ import annotations

import pandas as pd

from core.domain.ensemble.average import weighted_average
from core.domain.ensemble.weights import ic_weights


class QlibEnsembleAdapter:
    """将 qlib 模型预测结果转换为项目统一的多模型集成结果。"""

    @staticmethod
    def merge_predictions(predictions: dict, ic_map: dict | None = None, strategy: str = "ic") -> pd.Series:
        """合并多个 qlib 预测 Series。

        strategy:
          - "equal": 等权平均
          - "ic": 按验证集 IC 加权
        """
        if strategy == "equal":
            from core.domain.ensemble.average import equal_average
            return equal_average(predictions)
        if strategy == "ic":
            weights = ic_weights(ic_map or {})
            return weighted_average(predictions, weights)
        raise ValueError(f"未知集成策略: {strategy}")
