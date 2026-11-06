"""
宏观分析引擎 — 4个方向:
1. 市场状态修正 (adjust_regime)
2. 行业轮动信号 (get_sector_rotation_signals)
3. 择时风控评分 (calculate_macro_risk_score)
4. 因子宏观归因 (get_factor_macro_fitness)
"""
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from core.domain.macro.data import MacroDataAccess, MacroSnapshot

logger = logging.getLogger(__name__)

# 行业轮动映射: (宏观条件, 偏多板块列表)
SECTOR_ROTATION_RULES = [
    ("LPR下降(宽松)", ["房地产开发", "汽车", "家用电器", "建筑材料", "家居用品"],
     lambda s: _trend(s, "lpr_5y", -0.15, 90)),
    ("LPR上升(收紧)", ["银行", "保险", "多元金融"],
     lambda s: _trend(s, "lpr_5y", 0.10, 90)),
    ("PPI上行", ["煤炭", "有色金属", "钢铁", "基础化工", "石油石化"],
     lambda s: _trend(s, "ppi_yoy", 1.0, 90)),
    ("消费通胀", ["食品饮料", "医药生物", "农林牧渔", "消费者服务"],
     lambda s: _above(s, "cpi_yoy", 2.0)),
    ("PMI新订单扩张", ["机械设备", "电子", "交通运输", "电力设备"],
     lambda s: _above(s, "pmi_new_order", 52)),
    ("M1-M2剪刀差扩大", ["证券", "计算机", "传媒"],
     lambda s: _trend(s, "m1_m2_scissors", 1.0, 90)),
    ("Shibor走高(流动性紧)", ["银行", "公用事业", "煤炭"],
     lambda s: _trend(s, "m3_rate", 0.3, 60)),
    ("流动性宽松", ["通信", "国防军工", "电子"],
     lambda s: _trend(s, "on_rate", -0.2, 60) and _below(s, "shibor_on", 1.5)),
]


def _above(s, key, threshold):
    v = getattr(s, key, None)
    return v is not None and v > threshold


def _below(s, key, threshold):
    v = getattr(s, key, None)
    return v is not None and v < threshold


def _trend(s, key, delta, days):
    """近N天变化超过delta"""
    chg = MacroDataAccess.get_macro_change(key, days, as_of=getattr(s, 'as_of_date', None))
    return chg is not None and chg > delta if delta > 0 else chg is not None and chg < delta


class MacroAnalyzer:

    def __init__(self):
        self._data = MacroDataAccess()

    # ================================================================
    # 方向1: 市场状态修正
    # ================================================================
    def adjust_regime(self, regime_result, macro_snapshot=None):
        """在技术面 regime 判定后做宏观修正。返回可能修改过的 RegimeResult。"""
        if regime_result is None:
            return regime_result

        if macro_snapshot is None:
            macro_snapshot = self._data.get_macro_snapshot()
        if macro_snapshot is None:
            return regime_result

        regime_bias = self._score_regime_bias(macro_snapshot)
        current_regime = regime_result.regime.value
        macro_favor = max(regime_bias, key=regime_bias.get)
        macro_score = regime_bias[current_regime]

        if macro_score >= 0.6:
            regime_result.confidence = min(0.95, regime_result.confidence + 0.10)
            regime_result.details["macro"] = {"action": "confirm", "bias": regime_bias}
        elif macro_score >= 0.3:
            regime_result.confidence = max(0.25, round(regime_result.confidence * 0.85, 3))
            regime_result.details["macro"] = {"action": "weaken", "bias": regime_bias}
        else:
            regime_result.confidence = max(0.15, round(regime_result.confidence * 0.65, 3))
            regime_result.details["macro"] = {"action": "weaken_strong", "bias": regime_bias,
                                              "macro_favor": macro_favor}

        return regime_result

    def _score_regime_bias(self, s: MacroSnapshot) -> Dict[str, float]:
        scores = {"trending_up": 0.3, "trending_down": 0.3, "ranging": 0.3, "volatile": 0.3}

        # trending_up 支撑
        up = 0.3
        if s.pmi is not None:
            if s.pmi > 51: up += 0.20
            elif s.pmi > 50: up += 0.10
            else: up -= 0.10
        if s.m1_m2_scissors is not None:
            if s.m1_m2_scissors > 0: up += 0.15
            elif s.m1_m2_scissors < -3: up -= 0.10
        if s.ppi_yoy is not None:
            if s.ppi_yoy > 0: up += 0.10
            elif s.ppi_yoy < -2: up -= 0.10
        if s.shibor_on is not None and s.spread_1y_on is not None and s.spread_1y_on < 0.3:
            up += 0.10
        scores["trending_up"] = max(0.05, min(0.95, up))

        # trending_down
        down = 0.3
        if s.pmi is not None:
            if s.pmi < 49.5: down += 0.20
            elif s.pmi < 50: down += 0.10
            else: down -= 0.10
        if s.m1_m2_scissors is not None:
            if s.m1_m2_scissors < -3: down += 0.15
            elif s.m1_m2_scissors > 0: down -= 0.10
        if s.cpi_yoy is not None:
            if s.cpi_yoy < 0: down += 0.15
            elif s.cpi_yoy > 3: down += 0.10
        scores["trending_down"] = max(0.05, min(0.95, down))

        # ranging
        ranging = 0.3
        if s.pmi is not None and 49.5 <= s.pmi <= 50.5: ranging += 0.15
        if s.m1_m2_scissors is not None and -1 <= s.m1_m2_scissors <= 1: ranging += 0.15
        if s.spread_1y_on is not None and 0.2 <= s.spread_1y_on <= 0.6: ranging += 0.10
        scores["ranging"] = max(0.05, min(0.95, ranging))

        # volatile
        volatile = 0.3
        if s.spread_1y_on is not None and (s.spread_1y_on > 1.5 or s.spread_1y_on < -0.1): volatile += 0.25
        if s.cpi_yoy is not None and s.cpi_yoy > 3.5: volatile += 0.15
        if s.m1_m2_scissors is not None and abs(s.m1_m2_scissors) > 8: volatile += 0.15
        scores["volatile"] = max(0.05, min(0.95, volatile))

        # 4 状态 → 6 状态映射: 方向(trending_up/down/ranging) × 波动(volatile/steady)
        vol = scores["volatile"]
        steady = 1.0 - vol

        def _combine(base, v):
            return max(0.05, min(0.95, 0.3 + (base - 0.3) + (v - 0.3)))

        return {
            "volatile_up": _combine(scores["trending_up"], vol),
            "volatile_down": _combine(scores["trending_down"], vol),
            "volatile_flat": _combine(scores["ranging"], vol),
            "steady_up": _combine(scores["trending_up"], steady),
            "steady_down": _combine(scores["trending_down"], steady),
            "steady_flat": _combine(scores["ranging"], steady),
        }

    # ================================================================
    # 方向2: 行业轮动信号
    # ================================================================
    def get_sector_rotation_signals(self, macro_snapshot=None, as_of=None):
        if macro_snapshot is None:
            macro_snapshot = self._data.get_macro_snapshot(as_of=as_of)
        if macro_snapshot is None:
            return []

        signals = []
        for condition, sectors, check in SECTOR_ROTATION_RULES:
            try:
                if check(macro_snapshot):
                    signals.append({"sectors": sectors, "condition": condition, "bias": "偏多"})
            except Exception:
                pass
        return signals

    # ================================================================
    # 方向3: 择时风控评分
    # ================================================================
    def calculate_macro_risk_score(self, macro_snapshot=None, as_of=None):
        if macro_snapshot is None:
            macro_snapshot = self._data.get_macro_snapshot(as_of=as_of)
        if macro_snapshot is None:
            return {"score": 0.0, "level": "normal", "signals": []}

        chg_as_of = as_of or getattr(macro_snapshot, 'as_of_date', None)
        items = []
        total = 0.0

        # PMI 持续收缩
        if macro_snapshot.pmi is not None and macro_snapshot.pmi < 50:
            chg_90 = MacroDataAccess.get_macro_change("pmi", 90, as_of=chg_as_of)
            if chg_90 is not None and chg_90 < -0.5:
                items.append("PMI连续3月下行(荣枯线下方)")
                total += 0.30
            else:
                items.append("PMI低于荣枯线")
                total += 0.15

        # M1-M2 急剧逆转
        chg_m1m2_90 = MacroDataAccess.get_macro_change("m1_m2_scissors", 90, as_of=chg_as_of)
        if chg_m1m2_90 is not None and abs(chg_m1m2_90) > 3:
            items.append(f"M1-M2剪刀差90天变动{chg_m1m2_90:+.1f}pp")
            total += 0.20

        # CPI/PPI 剪刀差异常
        if macro_snapshot.cpi_ppi_spread is not None:
            if macro_snapshot.cpi_ppi_spread > 5:
                items.append(f"CPI-PPI价差异常拉大({macro_snapshot.cpi_ppi_spread:.1f}pp)")
                total += 0.15
            elif macro_snapshot.cpi_ppi_spread < -3:
                items.append(f"PPI大幅高于CPI({macro_snapshot.cpi_ppi_spread:.1f}pp,企业利润受挤)")
                total += 0.10

        # LPR 近期上调
        chg_lpr_60 = MacroDataAccess.get_macro_change("lpr_5y", 60, as_of=chg_as_of)
        if chg_lpr_60 is not None and chg_lpr_60 > 0.15:
            items.append(f"LPR5Y近60天上调{chg_lpr_60:.2f}pp")
            total += 0.15

        # Shibor 短端急升
        chg_shibor_30 = MacroDataAccess.get_macro_change("on_rate", 30, as_of=chg_as_of)
        if chg_shibor_30 is not None and chg_shibor_30 > 0.5:
            items.append(f"Shibor隔夜30天急升{chg_shibor_30:.2f}pp")
            total += 0.20

        score = min(0.95, total)
        if score < 0.4:
            level = "normal"
        elif score < 0.7:
            level = "warning"
        else:
            level = "danger"

        return {"score": round(score, 3), "level": level, "signals": items}

    # ================================================================
    # 方向4: 因子宏观归因
    # ================================================================
    def get_factor_macro_fitness(self, factor_keys, macro_snapshot=None, as_of=None):
        """返回 {factor_key: {fitness_score, label}} 基于当前宏观状态的因子适配度。
        优先从 factor_macro_ic_history 表读历史IC分档，无数据时回退启发式规则。
        """
        if macro_snapshot is None:
            macro_snapshot = self._data.get_macro_snapshot(as_of=as_of)
        if macro_snapshot is None:
            return {k: {"fitness": 0.5, "label": "中", "reason": "无宏观数据"} for k in factor_keys}

        # 确定当前宏观分档
        pmi_level = "扩张" if (macro_snapshot.pmi and macro_snapshot.pmi > 51) else \
                    ("收缩" if (macro_snapshot.pmi and macro_snapshot.pmi < 49) else "温和")
        cpi_level = "高通胀" if (macro_snapshot.cpi_yoy and macro_snapshot.cpi_yoy > 3) else \
                    ("低通胀" if (macro_snapshot.cpi_yoy is not None and macro_snapshot.cpi_yoy < 0) else "温和")
        if macro_snapshot.m1_m2_scissors is None:
            m1m2_dir = "未知"
        elif macro_snapshot.m1_m2_scissors > 0:
            m1m2_dir = "正剪刀差"
        elif macro_snapshot.m1_m2_scissors < 0:
            m1m2_dir = "负剪刀差"
        else:
            m1m2_dir = "零剪刀差"

        result = {}
        for key in factor_keys:
            ic_avg, n = self._query_macro_ic(key, pmi_level, cpi_level, m1m2_dir)
            if n >= 5:
                fitness = min(0.95, max(0.10, 0.50 + ic_avg * 5))
                label = "高" if fitness >= 0.7 else ("低" if fitness < 0.4 else "中")
                result[key] = {"fitness": round(fitness, 2), "label": label,
                               "reason": f"历史IC={ic_avg:.4f} (n={n}, PMI={pmi_level}, CPI={cpi_level}, M1M2={m1m2_dir})"}
            else:
                fitness, reason = self._heuristic_fitness(key, macro_snapshot)
                label = "高" if fitness >= 0.7 else ("低" if fitness < 0.4 else "中")
                result[key] = {"fitness": round(fitness, 2), "label": label,
                               "reason": f"{reason} (历史数据不足 n={n})"}
        return result

    def _query_macro_ic(self, factor_key, pmi_level, cpi_level, m1m2_dir):
        """查询因子在特定宏观分档下的历史平均IC"""
        try:
            from core.data import Database
            if not Database.is_available():
                return 0.0, 0
            rows = Database.fetchall(
                "SELECT AVG(ic) as avg_ic, COUNT(*) as n FROM factor_macro_ic_history "
                "WHERE factor_key=%s AND pmi_level=%s AND cpi_level=%s AND m1m2_dir=%s",
                (factor_key, pmi_level, cpi_level, m1m2_dir))
            if rows and rows[0]["n"] and rows[0]["n"] >= 3:
                return float(rows[0]["avg_ic"] or 0), int(rows[0]["n"])
        except Exception:
            pass
        return 0.0, 0

    def _heuristic_fitness(self, key: str, s: MacroSnapshot) -> Tuple[float, str]:
        """基于领域知识的启发式适配度。后续可用 overfitting_checker 历史IC替换。"""
        key_lower = key.lower()

        # 动量因子: PMI>50 扩张环境更好
        if any(kw in key_lower for kw in ("momentum", "trend", "roc")):
            if s.pmi is not None and s.pmi > 51:
                return 0.85, f"PMI={s.pmi:.1f}扩张,动量环境有利"
            elif s.pmi is not None and s.pmi > 50:
                return 0.65, f"PMI={s.pmi:.1f}温和,动量尚可"
            else:
                return 0.35, f"PMI偏弱,动量因子易失效"

        # 反转/均值回归: 高波动或市场恐慌时更强
        if any(kw in key_lower for kw in ("reversal", "mean_reversion", "bollinger")):
            if s.spread_1y_on is not None and (s.spread_1y_on < -0.1 or s.spread_1y_on > 1.5):
                return 0.80, f"利差异常({s.spread_1y_on:.2f}),反转效应增强"
            if s.m1_m2_scissors is not None and abs(s.m1_m2_scissors) > 5:
                return 0.70, f"M1-M2大幅偏离,市场极端"
            return 0.50, "宏观中性"

        # 低波动: 利率上行环境有利防御
        if any(kw in key_lower for kw in ("volatility", "low_vol")):
            if s.shibor_3m is not None and s.shibor_3m > 2.0:
                return 0.80, f"Shibor3M={s.shibor_3m:.1f}%,利率高,防御有利"
            if s.pmi is not None and s.pmi < 49.5:
                return 0.75, "PMI收缩,低波防御"
            return 0.55, "宏观中性"

        # 质量/基本面: 通胀温和+增长稳定
        if any(kw in key_lower for kw in ("fundamental", "value", "quality", "roe")):
            if s.cpi_yoy is not None and 1 < s.cpi_yoy < 3 and s.pmi is not None and s.pmi > 50:
                return 0.80, f"温和通胀(CPI={s.cpi_yoy:.1f}%)+PMI扩张,基本面因子有利"
            if s.cpi_yoy is not None and s.cpi_yoy > 3.5:
                return 0.35, f"高通胀(CPI={s.cpi_yoy:.1f}%),基本面估值受挤压"
            return 0.55, "宏观中性"

        # 流动性/成交量: 宽松货币环境
        if any(kw in key_lower for kw in ("turnover", "volume", "liquidity", "size")):
            if s.shibor_on is not None and s.shibor_on < 1.5:
                return 0.80, f"隔夜利率低({s.shibor_on:.1f}%),流动性充裕"
            if s.m2_yoy is not None and s.m2_yoy > 10:
                return 0.75, f"M2高增({s.m2_yoy:.1f}%),流动性宽"
            return 0.50, "宏观中性"

        # 成长/科技: 低利率环境
        if any(kw in key_lower for kw in ("growth", "tech", "innovation")):
            if s.shibor_1y is not None and s.shibor_1y < 2.0:
                return 0.80, f"低利率(1Y={s.shibor_1y:.1f}%),成长估值受益"
            if s.ppi_yoy is not None and s.ppi_yoy > 3:
                return 0.40, f"PPI高企(={s.ppi_yoy:.1f}%),周期占优压制成长"
            return 0.55, "宏观中性"

        # 小盘: 流动性+风险偏好
        if any(kw in key_lower for kw in ("small_cap", "micro_cap")):
            if s.pmi is not None and s.pmi > 51 and s.shibor_on is not None and s.shibor_on < 1.5:
                return 0.85, "PMI扩张+低利率,小盘弹性强"
            if s.spread_1y_on is not None and s.spread_1y_on < 0:
                return 0.30, "收益率倒挂,小盘风险高"
            return 0.50, "宏观中性"

        return 0.50, "暂无此类因子的宏观先验"
