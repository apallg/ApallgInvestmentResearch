"""
宏观修正 + 板块背离融合 — 选股页面和方向模型回测共用
"""

import logging

logger = logging.getLogger(__name__)


SECTOR_INDUSTRY_KEYWORDS = {
    "房地产开发": ("房地产",),
    "汽车": ("汽车制造",),
    "家用电器": ("电气机械和器材制造",),
    "建筑材料": ("非金属矿物制品",),
    "家居用品": ("家具制造",),
    "银行": ("货币金融",),
    "保险": ("保险",),
    "多元金融": ("其他金融",),
    "煤炭": ("煤炭",),
    "有色金属": ("有色金属",),
    "钢铁": ("黑色金属",),
    "基础化工": ("化学原料", "化学制品"),
    "石油石化": ("石油", "炼焦", "核燃料"),
    "食品饮料": ("农副食品", "食品制造", "酒、饮料", "酒饮料"),
    "医药生物": ("医药制造",),
    "农林牧渔": ("农业", "林业", "畜牧业", "渔业"),
    "消费者服务": ("餐饮业", "居民服务", "商务服务"),
    "机械设备": ("通用设备", "专用设备", "仪器仪表"),
    "电子": ("计算机、通信和其他电子设备制造", "电子设备制造"),
    "交通运输": ("铁路运输", "道路运输", "水上运输", "航空运输"),
    "电力设备": ("电气机械和器材制造",),
    "证券": ("资本市场服务",),
    "计算机": ("软件和信息技术服务", "互联网和相关服务", "计算机、通信和其他电子设备制造"),
    "传媒": ("新闻和出版", "广播、电视、电影", "文化艺术"),
    "公用事业": ("电力、热力", "燃气生产和供应", "水的生产和供应"),
    "通信": ("电信、广播电视和卫星传输", "计算机、通信和其他电子设备制造"),
    "国防军工": ("铁路、船舶、航空航天和其他运输设备制造",),
}


def sector_bias_bonus(industry, sector_bias):
    if not industry or not sector_bias:
        return 0.0
    industry = str(industry)
    if industry in sector_bias:
        return float(sector_bias[industry])
    return max(
        (
            float(bonus)
            for sector, bonus in sector_bias.items()
            if any(keyword in industry for keyword in SECTOR_INDUSTRY_KEYWORDS.get(sector, (sector,)))
        ),
        default=0.0,
    )


def apply_sector_bias(scores, sector_bias, industry_map):
    adjusted = dict(scores)
    for code, score in scores.items():
        value = industry_map.get(code)
        industry = value.get("industry") if isinstance(value, dict) else value
        bonus = sector_bias_bonus(industry, sector_bias)
        if bonus:
            adjusted[code] = min(1.0, score + bonus)
    return adjusted


def apply_macro_sector_bias(scores, eval_date, industry_map=None):
    """
    宏观行业偏好 — 对 macro-favored 行业的股票加 L0 加分。
    输入: {code: score} 原始分数
    输出: {code: adjusted_score} 调整后分数, macro_sector_bias dict
    """
    macro_sector_bias = {}
    if not scores:
        return scores, macro_sector_bias

    try:
        from core.domain.macro.analyzer import MacroAnalyzer
        m = MacroAnalyzer()
        sector_signals = m.get_sector_rotation_signals(as_of=eval_date)
        for sig in sector_signals:
            for sec in sig.get("sectors", []):
                macro_sector_bias[sec] = 0.15
        if not macro_sector_bias:
            return scores, macro_sector_bias

        if industry_map is None:
            from core.data.fetcher import DataFetcher
            industry_map = DataFetcher().get_industry_map() or {}

        adjusted = apply_sector_bias(scores, macro_sector_bias, industry_map)

        logger.info(f"[Macro] 行业偏好: {list(macro_sector_bias.keys())[:8]} "
                    f"({len(macro_sector_bias)} 个板块)")
        return adjusted, macro_sector_bias
    except Exception as e:
        logger.warning(f"[Macro] 行业偏好计算失败: {e}")
        return scores, {}


def apply_sector_fusion(scores, eval_date, pool="a500", quotes_df=None, dir_scores=None,
                        neutralize_qlib=None):
    """
    板块背离融合 — 对背离板块的股票, 用板块专属 Qlib 模型重新打分后融合。
    委托选股管线的完整实现 (BuySideScreener._apply_sector_fusion), 与选股页面共用同一套代码。
    输入: {code: score} 原始 Qlib 市场模型分数
    输出: {code: fused_score} 融合后分数, fusion_info dict
    quotes_df: 预拉取的全市场 daily_quotes (回测复用)。
    dir_scores: 预计算的 {方向: {code: score}} (回测复用 pred_cache)。
    """
    if not scores:
        return scores, {"divergent": {}, "models": {}, "attr": {}, "escort": []}
    try:
        from core.domain.screening._screener import BuySideScreener
        return BuySideScreener()._apply_sector_fusion(
            scores, eval_date, pool, quotes_df=quotes_df, dir_scores=dir_scores,
            neutralize_qlib=neutralize_qlib)
    except Exception as e:
        logger.warning(f"[SectorFusion] 融合失败: {e}")
        return scores, {"divergent": {}, "models": {}, "attr": {}, "escort": []}
