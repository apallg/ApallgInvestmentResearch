"""
多源信号融合引擎 — 将 5 类异构信号标准化并加权融合为统一评分
"""

__all__ = ["MetaSignalFusion", "MetaVote", "FusionResult",
           "BUY_SIDE_WEIGHTS", "HOLD_SIDE_WEIGHTS"]

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

from core.shared import config

logger = logging.getLogger(__name__)

# ── 两套预设权重 ──

BUY_SIDE_WEIGHTS = {
    "tech_fusion":       getattr(config, "FUSION_BUY_WEIGHT_TECH", 0.18),
    "qlib":              getattr(config, "FUSION_BUY_WEIGHT_QLIB", 0.35),
    "strategy_cluster":  getattr(config, "FUSION_BUY_WEIGHT_STRATEGY", 0.20),
    "sentiment":         getattr(config, "FUSION_BUY_WEIGHT_SENTIMENT", 0.10),
    "north_flow":        getattr(config, "FUSION_BUY_WEIGHT_NORTH_FLOW", 0.07),
    "size_turnover":     getattr(config, "FUSION_BUY_WEIGHT_SIZE_TURNOVER", 0.10),
}

HOLD_SIDE_WEIGHTS = {
    "tech_fusion":       getattr(config, "FUSION_HOLD_WEIGHT_TECH", 0.20),
    "qlib":              getattr(config, "FUSION_HOLD_WEIGHT_QLIB", 0.10),
    "strategy_cluster":  getattr(config, "FUSION_HOLD_WEIGHT_STRATEGY", 0.15),
    "sentiment":         getattr(config, "FUSION_HOLD_WEIGHT_SENTIMENT", 0.07),
    "north_flow":        getattr(config, "FUSION_HOLD_WEIGHT_NORTH_FLOW", 0.05),
    "size_turnover":     getattr(config, "FUSION_HOLD_WEIGHT_SIZE_TURNOVER", 0.08),
    "pnl_pressure":      getattr(config, "FUSION_HOLD_WEIGHT_PNL", 0.25),
    "ai_debate":         getattr(config, "FUSION_HOLD_WEIGHT_AI_DEBATE", 0.10),
}

MAX_BUY_WEIGHT = sum(BUY_SIDE_WEIGHTS.values())
MAX_HOLD_WEIGHT = sum(HOLD_SIDE_WEIGHTS.values())


@dataclass
class MetaVote:
    source: str
    direction: int        # -1 看空 / 0 中性 / +1 看多
    score: float          # -1.0 ~ +1.0
    confidence: float     # 0.0 ~ 1.0
    weight: float = 0.0   # 融合权重(运行时填充)
    detail: dict = field(default_factory=dict)


@dataclass
class FusionResult:
    score: float = 0.0
    direction: str = "neutral"   # bullish / bearish / neutral
    confidence: float = 0.0
    action: str = "hold"         # buy / sell / hold
    size_ratio: float = 0.0
    votes: List[MetaVote] = field(default_factory=list)
    coverage: float = 0.0
    reason: str = ""


class MetaSignalFusion:
    def __init__(self, mode="buy_side", custom_weights=None):
        self.mode = mode
        if custom_weights:
            self.weights = dict(custom_weights)
        elif mode == "hold_side":
            self.weights = dict(HOLD_SIDE_WEIGHTS)
        else:
            self.weights = dict(BUY_SIDE_WEIGHTS)

        self._max_total_weight = sum(self.weights.values())
        self._bullish_bias = 1.2 if mode == "buy_side" else 1.0
        self._bearish_bias = 1.3 if mode == "hold_side" else 1.0
        self._votes: List[MetaVote] = []

    def add(self, source: str, score: float, confidence: float, detail: Optional[Dict] = None):
        if source not in self.weights:
            logger.warning(f"未知信号源: {source}, 跳过")
            return
        direction = 1 if score > 0.05 else -1 if score < -0.05 else 0
        self._votes.append(MetaVote(
            source=source,
            direction=direction,
            score=max(-1.0, min(1.0, score)),
            confidence=max(0.0, min(1.0, confidence)),
            weight=self.weights.get(source, 0),
            detail=detail or {},
        ))

    def fuse(self) -> FusionResult:
        active = [v for v in self._votes if v.weight > 0]
        if not active:
            return FusionResult(reason="无有效信号源")

        total_weight = sum(v.weight for v in active)
        raw_score = sum(v.score * v.weight for v in active) / total_weight

        if self._bullish_bias != 1.0 and raw_score > 0:
            raw_score = min(1.0, raw_score * self._bullish_bias)
        if self._bearish_bias != 1.0 and raw_score < 0:
            raw_score = max(-1.0, raw_score * self._bearish_bias)

        raw_conf = sum(v.confidence * v.weight for v in active) / total_weight
        coverage = total_weight / self._max_total_weight if self._max_total_weight > 0 else 0
        confidence = raw_conf * coverage

        if raw_score > 0.05 and confidence >= 0.25:
            action = "buy"
        elif raw_score < -0.05 and confidence >= 0.25:
            action = "sell"
        else:
            action = "hold"

        size_ratio = min(1.0, abs(raw_score) * confidence * 1.5)

        direction = "bullish" if raw_score > 0.05 else "bearish" if raw_score < -0.05 else "neutral"

        reason_parts = []
        for v in sorted(active, key=lambda x: abs(x.score), reverse=True):
            reason_parts.append(f"{v.source}={v.score:+.2f}")
        reason = " | ".join(reason_parts)

        return FusionResult(
            score=round(raw_score, 3),
            direction=direction,
            confidence=round(confidence, 3),
            action=action,
            size_ratio=round(size_ratio, 2),
            votes=active,
            coverage=round(coverage, 3),
            reason=reason,
        )

    @staticmethod
    def qlib_to_score(pred_value: float, all_predictions: List[float]) -> float:
        """将 Qlib 预测值映射到 -1~+1 分位分数"""
        if not all_predictions or pred_value is None:
            return 0.0
        if len(all_predictions) < 2:
            return 0.5 if pred_value > 0 else -0.5 if pred_value < 0 else 0
        import numpy as np
        arr = np.array(all_predictions)
        pct = (arr < pred_value).mean()
        return (pct - 0.5) * 2.0

    @staticmethod
    def calc_pnl_pressure(cost_price: float, current_price: float, shares: int,
                          total_asset: float, hold_days: int = 0) -> float:
        """计算持仓盈亏压力 (0~1, 越高压力越大需减仓)"""
        if not cost_price or not current_price or cost_price <= 0:
            return 0.5
        pnl_pct = (current_price - cost_price) / cost_price
        if pnl_pct > 0.20:
            pnl_score = 0.8
        elif pnl_pct > 0.10:
            pnl_score = 0.6
        elif pnl_pct > 0:
            pnl_score = 0.3
        elif pnl_pct > -0.05:
            pnl_score = 0.5
        elif pnl_pct > -0.10:
            pnl_score = 0.7
        else:
            pnl_score = 0.9

        time_score = 0.0
        if hold_days > 30:
            time_score = 0.3
        elif hold_days > 10:
            time_score = 0.2
        elif hold_days > 3:
            time_score = 0.1

        concentration = 0.0
        if total_asset > 0 and current_price > 0:
            position_value = abs(current_price * shares)
            position_pct = position_value / total_asset
            if position_pct > 0.20:
                concentration = 0.2
            elif position_pct > 0.10:
                concentration = 0.1

        return round(min(1.0, pnl_score * 0.6 + time_score * 0.25 + concentration * 0.15), 3)
