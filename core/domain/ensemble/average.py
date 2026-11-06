"""多模型预测合并 — 纯 pandas 计算，不依赖 qlib。"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def weighted_average(predictions: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """按权重合并多个预测 Series。

    要求各 Series 使用相同的 MultiIndex（datetime, instrument）。
    缺失值按 0 处理，保证合并结果完整。
    """
    if not predictions:
        raise ValueError("predictions 不能为空")
    if not weights:
        raise ValueError("weights 不能为空")

    result = None
    for name, pred in predictions.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        p = pred.fillna(0.0)
        if result is None:
            result = p * w
        else:
            result = result.add(p * w, fill_value=0.0)
    if result is None:
        raise ValueError("weights 全为 0，无法合并")
    return result


def equal_average(predictions: Dict[str, pd.Series]) -> pd.Series:
    names = list(predictions.keys())
    weights = {name: 1.0 / len(names) for name in names}
    return weighted_average(predictions, weights)
