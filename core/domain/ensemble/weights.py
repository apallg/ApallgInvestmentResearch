"""多模型集成权重计算 — 纯 L3 逻辑，不依赖 qlib/DB/网络。"""
from __future__ import annotations

import math
from typing import Dict, Optional


def ic_weights(ic_map: Dict[str, Optional[float]], lo: float = 0.1, hi: float = 0.7) -> Dict[str, float]:
    """根据各子模型 IC 计算集成权重。

    - 任一 IC 缺失/NaN/Inf 或全部非正时，回退等权
    - 权重按 IC 正比分配，并 clip 到 [lo, hi]，最后归一化
    """
    clean = {}
    for name, ic in ic_map.items():
        if ic is None:
            continue
        try:
            v = float(ic)
        except (TypeError, ValueError):
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        clean[name] = max(v, 0.0)

    if not clean:
        return _equal_weights(ic_map)
    total = sum(clean.values())
    if total <= 0:
        return _equal_weights(ic_map)

    raw = {name: max(v / total, lo) for name, v in clean.items()}
    raw = {name: min(v, hi) for name, v in raw.items()}
    s = sum(raw.values())
    if s <= 0:
        return _equal_weights(ic_map)
    return {name: v / s for name, v in raw.items()}


def equal_weights(names) -> Dict[str, float]:
    if not names:
        return {}
    w = 1.0 / len(names)
    return {name: w for name in names}


def _equal_weights(ic_map: Dict[str, Optional[float]]) -> Dict[str, float]:
    return equal_weights(list(ic_map.keys()))
