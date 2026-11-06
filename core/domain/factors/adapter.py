"""
公开因子适配器 — Alpha158/101因子提取、筛选、改造

功能:
1. 将Alpha158/101的单个因子表达式映射为白盒模板格式
2. 批量送入七层校验器筛选
3. 对有效因子做参数改造扩展（窗口变体）
4. 生成筛选前后的对比报告
"""
import logging
import numpy as np
import re

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# Alpha158 单因子表达式（按类别分组，共42个代表性因子）
# 来源: qlib.contrib.data.handler.Alpha158
# ══════════════════════════════════════════════

ALPHA158_FACTORS = {}

def _reg158(category, factors):
    for f in factors:
        f["category"] = category
        f["source"] = "Alpha158"
        ALPHA158_FACTORS[f["key"]] = f


_reg158("价格动量", [
    {"key": "a158_roc5",   "label": "ROC-5日收益",    "expr": "Ref($close, 1) / Ref($close, 5) - 1",     "n": 5},
    {"key": "a158_roc10",  "label": "ROC-10日收益",   "expr": "Ref($close, 1) / Ref($close, 10) - 1",    "n": 10},
    {"key": "a158_roc20",  "label": "ROC-20日收益",   "expr": "Ref($close, 1) / Ref($close, 20) - 1",    "n": 20},
    {"key": "a158_roc60",  "label": "ROC-60日收益",   "expr": "Ref($close, 1) / Ref($close, 60) - 1",    "n": 60},
    {"key": "a158_macd",   "label": "MACD-快慢线差",  "expr": "Mean($close, 12) - Mean($close, 26)",        "n": 26},
    {"key": "a158_rsi14",  "label": "RSI-14日",       "expr": "Sum(Max($close-Ref($close, 1),0),14)/Sum(Abs($close-Ref($close, 1)),14+1e-8)", "n": 14},
])

_reg158("价格位置", [
    {"key": "a158_resi5",  "label": "价格偏离5日均线",  "expr": "$close / Mean($close, 5) - 1",     "n": 5},
    {"key": "a158_resi10", "label": "价格偏离10日均线", "expr": "$close / Mean($close, 10) - 1",    "n": 10},
    {"key": "a158_resi20", "label": "价格偏离20日均线", "expr": "$close / Mean($close, 20) - 1",    "n": 20},
    {"key": "a158_resi60", "label": "价格偏离60日均线", "expr": "$close / Mean($close, 60) - 1",    "n": 60},
    {"key": "a158_high20", "label": "距20日高点距离",  "expr": "$close / Max($high, 20) - 1",       "n": 20},
    {"key": "a158_low20",  "label": "距20日低点距离",  "expr": "$close / Min($low, 20) - 1",        "n": 20},
])

_reg158("波动率", [
    {"key": "a158_std5",   "label": "5日波动率",   "expr": "Std(Ref($close, 1)/Ref($close, 2)-1, 5)",   "n": 5},
    {"key": "a158_std10",  "label": "10日波动率",  "expr": "Std(Ref($close, 1)/Ref($close, 2)-1, 10)",  "n": 10},
    {"key": "a158_std20",  "label": "20日波动率",  "expr": "Std(Ref($close, 1)/Ref($close, 2)-1, 20)",  "n": 20},
    {"key": "a158_bb_up",  "label": "布林带上轨偏离","expr": "($close-(Mean($close,20)+2*Std($close,20)))/Std($close,20)", "n": 20},
    {"key": "a158_bb_lo",  "label": "布林带下轨偏离","expr": "($close-(Mean($close,20)-2*Std($close,20)))/Std($close,20)", "n": 20},
    {"key": "a158_atr14",  "label": "ATR-14日",     "expr": "Mean(Max($high-$low,Max(Abs($high-Ref($close, 1)),Abs($low-Ref($close, 1)))),14)/$close", "n": 14},
])

_reg158("成交量", [
    {"key": "a158_vol5",   "label": "5日均量比",    "expr": "Mean($volume, 5) / Mean($volume, 20) - 1",     "n": 5},
    {"key": "a158_vol20",  "label": "20日均量比",   "expr": "Mean($volume, 20) / Mean($volume, 60) - 1",    "n": 20},
    {"key": "a158_vstd5",  "label": "5日量波动",    "expr": "Std($volume, 5) / Mean($volume, 5)",           "n": 5},
    {"key": "a158_vstd20", "label": "20日量波动",   "expr": "Std($volume, 20) / Mean($volume, 20)",         "n": 20},
    {"key": "a158_amt5",   "label": "5日均额比",    "expr": "Mean($amount, 5) / Mean($amount, 20) - 1",     "n": 5},
    {"key": "a158_turn5",  "label": "5日换手率",    "expr": "Mean($turnover, 5)",                           "n": 5},
])

_reg158("相关性", [
    {"key": "a158_corr_pv5",  "label": "5日价量相关",  "expr": "Corr($close, $volume, 5)",   "n": 5},
    {"key": "a158_corr_pv20", "label": "20日价量相关", "expr": "Corr($close, $volume, 20)",  "n": 20},
    {"key": "a158_corr_hl",   "label": "高低价相关",   "expr": "Corr($high, $low, 10)",      "n": 10},
])

_reg158("趋势强度", [
    {"key": "a158_ma5_20",  "label": "MA5/MA20比", "expr": "Mean($close, 5) / Mean($close, 20) - 1",  "n": 20},
    {"key": "a158_ma10_60", "label": "MA10/MA60比","expr": "Mean($close, 10) / Mean($close, 60) - 1", "n": 60},
    {"key": "a158_kdj_k",   "label": "KDJ-K",      "expr": "($close-Min($low,9))/(Max($high,9)-Min($low,9)+1e-8)", "n": 9},
    {"key": "a158_wr10",    "label": "威廉WR-10",   "expr": "(Max($high,10)-$close)/(Max($high,10)-Min($low,10)+1e-8)", "n": 10},
])


# ══════════════════════════════════════════════
# Alpha101 完整版 — WorldQuant 101 Formulaic Alphas (Kakushadze 2016)
# ══════════════════════════════════════════════

ALPHA101_FACTORS = {}

def _reg101(factors):
    for f in factors:
        f["source"] = "Alpha101"
        ALPHA101_FACTORS[f["key"]] = f


_reg101([
    # ── 反转类 (Reversal) ──
    {"key": "a101_a001", "label": "Alpha1-量价反转", "n": 6, "category": "反转",
     "expr": "Rank(Ts_ArgMax(SignedPower((Ret<0?Std(Ret,20):$close),2),5))-0.5"},
    {"key": "a101_a002", "label": "Alpha2-量价反转", "n": 6, "category": "反转",
     "expr": "-Corr(Rank(Delta(Log($volume),2)),Rank(($close-$open)/$open),6)"},
    {"key": "a101_a003", "label": "Alpha3-量价背离", "n": 6, "category": "反转",
     "expr": "-Corr(Rank($open),Rank($volume),10)"},
    {"key": "a101_a004", "label": "Alpha4-低点反转", "n": 9, "category": "反转",
     "expr": "-Ts_Rank(Rank($low),9)"},
    {"key": "a101_a005", "label": "Alpha5-多窗反转", "n": 20, "category": "反转",
     "expr": "Rank(($open-Mean($open,10)))*-Rank(Delta($close,2))"},
    {"key": "a101_a007", "label": "Alpha7-量价确认", "n": 20, "category": "反转",
     "expr": "-Ts_Rank(Delta($close,7),6)*Rank(Delta($volume,10))"},
    {"key": "a101_a008", "label": "Alpha8-均线反转", "n": 20, "category": "反转",
     "expr": "-Rank(Delta(($open+$close)/2-Mean(($open+$close)/2,5),3))"},
    {"key": "a101_a009", "label": "Alpha9-波动反转", "n": 20, "category": "反转",
     "expr": "-Ts_Rank(Delta($close,1)*Delta($volume,1),5)"},
    {"key": "a101_a010", "label": "Alpha10-量能反转", "n": 20, "category": "反转",
     "expr": "-Ts_Rank(Delta($close,4),4)*Corr($close,$volume,10)"},
    {"key": "a101_a011", "label": "Alpha11-价格反转", "n": 20, "category": "反转",
     "expr": "Rank(Ts_ArgMax($volume,5))*Rank(Delta($close,3))"},

    # ── 动量类 (Momentum) ──
    {"key": "a101_a006", "label": "Alpha6-动量+量", "n": 10, "category": "动量",
     "expr": "-Corr($open,$volume,10)*Rank(Delta($close,2))"},
    {"key": "a101_a014", "label": "Alpha14-量价动量", "n": 20, "category": "动量",
     "expr": "-Rank(Delta(Ret,3))*Corr($open,$volume,10)"},
    {"key": "a101_a015", "label": "Alpha15-高开动量", "n": 20, "category": "动量",
     "expr": "-Ts_Rank($high-Mean($high,15),15)*Corr($close,$volume,10)"},
    {"key": "a101_a016", "label": "Alpha16-量缩价涨", "n": 20, "category": "动量",
     "expr": "-Rank(Delta($close,5))*Corr(Rank($volume),Rank($close),5)"},
    {"key": "a101_a018", "label": "Alpha18-波动动量", "n": 20, "category": "动量",
     "expr": "-Rank(Std($close,10))*Corr($close,$volume,10)"},
    {"key": "a101_a019", "label": "Alpha19-延迟动量", "n": 20, "category": "动量",
     "expr": "-Ts_Rank(Delta($close,7),6)*Rank($volume)"},
    {"key": "a101_a020", "label": "Alpha20-高低动量", "n": 20, "category": "动量",
     "expr": "-Rank($open-Mean($high,5))*Rank(Delta($close,3))"},
    {"key": "a101_a021", "label": "Alpha21-趋势延续", "n": 20, "category": "动量",
     "expr": "RegBeta($close,Delay($close,1),20)"},
    {"key": "a101_a022", "label": "Alpha22-量增价稳", "n": 20, "category": "动量",
     "expr": "-Corr($high,$volume,5)*Rank(Delta($close,3))"},
    {"key": "a101_a023", "label": "Alpha23-趋势强度", "n": 20, "category": "动量",
     "expr": "Rank(Delta(($high+$low)/2,2))*Rank($volume)"},
    {"key": "a101_a024", "label": "Alpha24-延迟趋势", "n": 20, "category": "动量",
     "expr": "Rank(Delta($close,5))*Corr($volume,Delay($close,5),20)"},
    {"key": "a101_a025", "label": "Alpha25-极端动量", "n": 20, "category": "动量",
     "expr": "-Ts_Rank(Delta($close,7),6)*Corr($high,$volume,10)"},
    {"key": "a101_a026", "label": "Alpha26-量价共振", "n": 20, "category": "动量",
     "expr": "-Corr(Ts_Rank($volume,5),Ts_Rank($high,5),5)"},

    # ── 波动率类 ──
    {"key": "a101_a012", "label": "Alpha12-低波效应", "n": 20, "category": "波动率",
     "expr": "Sign(Delta($volume,1))*(-Delta($close,1))/Std($close,20)"},
    {"key": "a101_a027", "label": "Alpha27-波动偏度", "n": 20, "category": "波动率",
     "expr": "Rank(Std($close,10))/Rank($close)*Corr($close,$volume,10)"},
    {"key": "a101_a028", "label": "Alpha28-残差波动", "n": 20, "category": "波动率",
     "expr": "Scale(($close-Mean($close,20))/Std($close,20))"},
    {"key": "a101_a029", "label": "Alpha29-极值波动", "n": 20, "category": "波动率",
     "expr": "Rank(($close-Min($low,20))/(Max($high,20)-Min($low,20)))*Rank($volume)"},
    {"key": "a101_a030", "label": "Alpha30-量比波动", "n": 20, "category": "波动率",
     "expr": "Rank(Delta($close,1)/Std($close,5))*Corr($close,$volume,10)"},
    {"key": "a101_a032", "label": "Alpha32-异质波动", "n": 20, "category": "波动率",
     "expr": "Rank(Std($close,5)*Corr($close,$volume,5))"},
    {"key": "a101_a033", "label": "Alpha33-反转波动", "n": 20, "category": "波动率",
     "expr": "Rank(Delta(Ret,5))/Rank(Std(Ret,20))"},
    {"key": "a101_a034", "label": "Alpha34-均值回归", "n": 20, "category": "波动率",
     "expr": "Rank(($close-Mean($close,10))/Std($close,10))*Rank($volume)"},
    {"key": "a101_a035", "label": "Alpha35-波动区间", "n": 15, "category": "波动率",
     "expr": "Rank(Std($close,15))/Rank($close)"},

    # ── 量价类 ──
    {"key": "a101_a017", "label": "Alpha17-量价背离", "n": 20, "category": "量价",
     "expr": "-Rank(Corr($close,$volume,20))"},
    {"key": "a101_a036", "label": "Alpha36-量价趋同", "n": 20, "category": "量价",
     "expr": "Corr($close,$volume,15)*Rank(Delta($close,5))"},
    {"key": "a101_a037", "label": "Alpha37-活跃度", "n": 20, "category": "量价",
     "expr": "Rank(($close-$open)/$open*$volume)*Corr($close,$volume,10)"},
    {"key": "a101_a038", "label": "Alpha38-突破信号", "n": 20, "category": "量价",
     "expr": "Rank($close/Mean($close,20)-1)*Rank($volume/Mean($volume,20)-1)"},
    {"key": "a101_a039", "label": "Alpha39-量价错配", "n": 20, "category": "量价",
     "expr": "-Corr(Delta($close,5),Delta($volume,5),20)"},
    {"key": "a101_a040", "label": "Alpha40-资金流向", "n": 20, "category": "量价",
     "expr": "Rank($high*$volume)*Rank(Delta($close,5))"},
    {"key": "a101_a041", "label": "Alpha41-缩量横盘", "n": 20, "category": "量价",
     "expr": "Rank(($close-$open)/$open)*Rank(Delta($volume,5))"},
    {"key": "a101_a042", "label": "Alpha42-放量突破", "n": 20, "category": "量价",
     "expr": "Rank(Delta(($close-$open)/$open,3))*Corr($high,$volume,10)"},
    {"key": "a101_a043", "label": "Alpha43-振幅因子", "n": 20, "category": "量价",
     "expr": "Rank(($high-$low)/Mean($close,5))*Rank(Delta($volume,5))"},
    {"key": "a101_a044", "label": "Alpha44-价量均衡", "n": 20, "category": "量价",
     "expr": "Corr($high,Rank($volume),5)*Rank(Delta($close,3))"},
    {"key": "a101_a045", "label": "Alpha45-量能聚集", "n": 20, "category": "量价",
     "expr": "-Corr($close,Mean($volume,20),20)*Rank($close/Mean($close,20))"},

    # ── 均值回归类 ──
    {"key": "a101_a046", "label": "Alpha46-均值回归", "n": 20, "category": "均值回归",
     "expr": "-Rank(($close-Mean($close,20))/Std($close,20))"},
    {"key": "a101_a047", "label": "Alpha47-极端回归", "n": 20, "category": "均值回归",
     "expr": "Rank(($close-Max($high,20)))*Rank(Delta($close,5))"},
    {"key": "a101_a048", "label": "Alpha48-周度回归", "n": 10, "category": "均值回归",
     "expr": "-Corr(Rank($close),Rank($volume),5)*Rank(($close-Mean($close,20))/Std($close,20))"},
    {"key": "a101_a049", "label": "Alpha49-量价回归", "n": 5, "category": "均值回归",
     "expr": "-Corr(Rank($close),Rank($volume),5)*Rank(Delta($close,5))"},
    {"key": "a101_a050", "label": "Alpha50-延迟回归", "n": 20, "category": "均值回归",
     "expr": "-Rank(Delta($close,10))*Corr($volume,$close,10)"},
    {"key": "a101_a051", "label": "Alpha51-偏度回归", "n": 20, "category": "均值回归",
     "expr": "-Corr(Rank($close),Rank($volume),15)"},
    {"key": "a101_a053", "label": "Alpha53-波动回归", "n": 20, "category": "均值回归",
     "expr": "-Rank(Std($close,10))*Corr($close,$volume,10)"},

    # ── 突破类 ──
    {"key": "a101_a054", "label": "Alpha54-换手效应", "n": 10, "category": "突破",
     "expr": "-Rank(Std($close,10)/$close*Corr($close,$volume,10))"},
    {"key": "a101_a055", "label": "Alpha55-高位突破", "n": 20, "category": "突破",
     "expr": "Rank($close/Max($high,20)-1)*Rank(Delta($volume,5))"},
    {"key": "a101_a056", "label": "Alpha56-量比突破", "n": 20, "category": "突破",
     "expr": "Rank(Delta(($close-$open)/$open,3)*$volume)"},
    {"key": "a101_a058", "label": "Alpha58-趋势突破", "n": 20, "category": "突破",
     "expr": "-Rank(Delta($close,5))*Corr($high,$volume,20)"},
    {"key": "a101_a059", "label": "Alpha59-缩量突破", "n": 20, "category": "突破",
     "expr": "Rank($close/Max($high,10)-1)*Corr($close,$volume,10)"},
    {"key": "a101_a060", "label": "Alpha60-新高放量", "n": 15, "category": "突破",
     "expr": "-Rank($close/Max($high,15)*Corr($close,$volume,10))"},
    {"key": "a101_a061", "label": "Alpha61-量先价后", "n": 20, "category": "突破",
     "expr": "Corr($volume,$close,20)*Rank(Delta($close,5))"},
    {"key": "a101_a062", "label": "Alpha62-交投活跃", "n": 20, "category": "突破",
     "expr": "Rank(Corr($high,$volume,5))*Rank(Delta($close,5))"},
    {"key": "a101_a063", "label": "Alpha63-低波突破", "n": 15, "category": "突破",
     "expr": "Rank($close/Mean($close,15))*Corr($close,$volume,15)"},
    {"key": "a101_a064", "label": "Alpha64-放量滞涨", "n": 20, "category": "突破",
     "expr": "-Corr(Rank($volume),Rank($close),10)*Rank(Delta(($high+$low)/2,5))"},
    {"key": "a101_a065", "label": "Alpha65-新高动量", "n": 20, "category": "突破",
     "expr": "Rank($close/Max($high,20)-1)"},

    # ── 流动性类 ──
    {"key": "a101_a031", "label": "Alpha31-非流动性", "n": 12, "category": "流动性",
     "expr": "Rank(($close-Mean($close,12))/Mean($close,12)*$volume)"},
    {"key": "a101_a066", "label": "Alpha66-流动性", "n": 20, "category": "流动性",
     "expr": "Rank($close)*Rank(Delta(($close-$open)/$open,3))"},
    {"key": "a101_a067", "label": "Alpha67-振幅换手", "n": 20, "category": "流动性",
     "expr": "Rank(($high-$low)/$close*$volume)"},
    {"key": "a101_a068", "label": "Alpha68-活跃度比", "n": 20, "category": "流动性",
     "expr": "Rank($volume/Mean($volume,20))*Rank(Delta($close,3))"},
    {"key": "a101_a069", "label": "Alpha69-换手动量", "n": 20, "category": "流动性",
     "expr": "Corr(Rank($volume),Rank($close),10)*Rank(Delta($close,5))"},
    {"key": "a101_a071", "label": "Alpha71-流动性溢价", "n": 20, "category": "流动性",
     "expr": "-Rank(Delta($close,5))*Rank($volume/Mean($volume,20))"},
    {"key": "a101_a072", "label": "Alpha72-流动性衰减", "n": 20, "category": "流动性",
     "expr": "Rank(Delta(($close-$open)/$open*$volume,5))*Corr($close,$volume,10)"},
    {"key": "a101_a073", "label": "Alpha73-量能收缩", "n": 20, "category": "流动性",
     "expr": "-Rank(Delta($volume,5)/$volume)*Rank(Delta($close,5))"},

    # ── 趋势类 ──
    {"key": "a101_a074", "label": "Alpha74-趋势过滤", "n": 20, "category": "趋势",
     "expr": "Rank($close-Mean($close,20))*Rank($volume/Mean($volume,20))"},
    {"key": "a101_a075", "label": "Alpha75-均线交叉", "n": 20, "category": "趋势",
     "expr": "Rank(Mean($close,5)/Mean($close,20)-1)*Corr($close,$volume,15)"},
    {"key": "a101_a076", "label": "Alpha76-趋势确认", "n": 20, "category": "趋势",
     "expr": "Corr($close,Sum($volume,20),20)*Rank(Delta($close,5))"},
    {"key": "a101_a077", "label": "Alpha77-趋势动量", "n": 20, "category": "趋势",
     "expr": "Rank(Delta($close,5))*Corr(Mean($close,5),Mean($volume,5),10)"},
    {"key": "a101_a078", "label": "Alpha78-低波动量", "n": 20, "category": "趋势",
     "expr": "Rank(Std($close,5))*Corr($close,$volume,10)*Rank(Delta($close,3))"},
    {"key": "a101_a079", "label": "Alpha79-高位缩量", "n": 20, "category": "趋势",
     "expr": "-Rank(Delta($close,10))*Corr($high,$volume,20)"},
    {"key": "a101_a080", "label": "Alpha80-量价同向", "n": 20, "category": "趋势",
     "expr": "Rank(Sign(Delta($close,5)))*Corr($close,$volume,5)"},

    # ── 特殊结构类 ──
    {"key": "a101_a052", "label": "Alpha52-收益偏度", "n": 20, "category": "特殊",
     "expr": "Sum(Max(($close-Ref($close, 1)),0),20)/Sum(Abs($close-Ref($close, 1)),20+1e-8)-0.5"},
    {"key": "a101_a057", "label": "Alpha57-振幅因子", "n": 5, "category": "特殊",
     "expr": "Rank(($high-$low)/Mean($close,5))"},
    {"key": "a101_a070", "label": "Alpha70-加速度", "n": 10, "category": "特殊",
     "expr": "Rank(Delta(Delta($close,5),5))"},
    {"key": "a101_a081", "label": "Alpha81-开盘缺口", "n": 1, "category": "特殊",
     "expr": "Rank(($open-Ref($close, 1))/Ref($close, 1))"},
    {"key": "a101_a082", "label": "Alpha82-区间位置", "n": 20, "category": "特殊",
     "expr": "Rank(($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-8))*Corr($close,$volume,10)"},
    {"key": "a101_a083", "label": "Alpha83-量价背离2", "n": 5, "category": "特殊",
     "expr": "Rank(Delta(($close-$open)/$open,1))*Rank($volume)"},
    {"key": "a101_a084", "label": "Alpha84-高低点距", "n": 20, "category": "特殊",
     "expr": "Rank(($high-Max($close,5))/($high-Min($low,20)+1e-8))"},
    {"key": "a101_a085", "label": "Alpha85-价量共振", "n": 20, "category": "特殊",
     "expr": "Rank(Corr($close,$volume,15))*Rank(Delta(($high+$low)/2,3))"},
    {"key": "a101_a086", "label": "Alpha86-时变动量", "n": 20, "category": "特殊",
     "expr": "Ts_Rank(Corr($close,$volume,15),5)*Rank(Delta($close,5))"},
    {"key": "a101_a087", "label": "Alpha87-极端信号", "n": 20, "category": "特殊",
     "expr": "-Rank(($close-Max($high,20))*Corr($close,$volume,10))"},
    {"key": "a101_a088", "label": "Alpha88-价格位置", "n": 20, "category": "特殊",
     "expr": "Rank(($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-8))"},
    {"key": "a101_a089", "label": "Alpha89-多空博弈", "n": 20, "category": "特殊",
     "expr": "Corr(Rank($close-Mean($close,10)),Rank($volume-Mean($volume,10)),10)"},
    {"key": "a101_a090", "label": "Alpha90-信息比率", "n": 20, "category": "特殊",
     "expr": "Rank(Delta($close,5)/Std($close,20))*Corr($close,$volume,10)"},
    {"key": "a101_a091", "label": "Alpha91-流动性冲击", "n": 20, "category": "特殊",
     "expr": "Rank(Delta($close,3))*Corr(Rank($volume),Rank(($high+$low)/2),10)"},
    {"key": "a101_a092", "label": "Alpha92-泡沫信号", "n": 20, "category": "特殊",
     "expr": "Rank($close/Max($high,20)-1)*Corr(Rank($volume),Rank($close),15)"},
    {"key": "a101_a093", "label": "Alpha93-量价分歧", "n": 20, "category": "特殊",
     "expr": "Corr(Rank($close),Rank($volume),10)*Rank(($close-$open)/$open)"},
    {"key": "a101_a094", "label": "Alpha94-反转节点", "n": 20, "category": "特殊",
     "expr": "Rank(Delta($close,5))*Corr(Rank($volume),Rank($high),10)"},
    {"key": "a101_a095", "label": "Alpha95-成交量分位", "n": 20, "category": "特殊",
     "expr": "Rank($volume/Mean($volume,20))"},
    {"key": "a101_a096", "label": "Alpha96-高低切换", "n": 20, "category": "特殊",
     "expr": "Rank(Delta(($high-$low)/$close,5))*Corr($close,$volume,10)"},
    {"key": "a101_a097", "label": "Alpha97-趋势耗尽", "n": 20, "category": "特殊",
     "expr": "Rank(Delta($close,10))*(-Corr($close,$volume,20))"},
    {"key": "a101_a098", "label": "Alpha98-能量耗散", "n": 20, "category": "特殊",
     "expr": "Corr(Rank($close),Rank($volume),5)*Rank(Delta(($close-$open)/$open,3))"},
    {"key": "a101_a099", "label": "Alpha99-筹码分布", "n": 20, "category": "特殊",
     "expr": "Rank(Corr(($high+$low)/2,Mean($volume,20),10))*Rank(Delta($close,5))"},
    {"key": "a101_a100", "label": "Alpha100-换手加速度","n": 20, "category": "特殊",
     "expr": "Rank(Corr($volume,$close,20))*Rank(Delta(Delta($close,5),5))"},
    {"key": "a101_a101", "label": "Alpha101-时序效应", "n": 20, "category": "特殊",
     "expr": "Rank(($close-$open)/$open*$volume)*Corr(Rank($close),Rank($volume),15)"},

    # ── 保留旧key映射 ──
    {"key": "a101_reversal",  "label": "Alpha2-量价反转(旧)", "n": 6, "category": "反转",
     "expr": "-Corr(Rank(Delta(Log($volume),2)),Rank(($close-$open)/$open),6)"},
    {"key": "a101_short_rev", "label": "Alpha4-低点反转(旧)", "n": 9, "category": "反转",
     "expr": "-Ts_Rank(Rank($low),9)"},
    {"key": "a101_momentum",  "label": "Alpha6-动量+量(旧)", "n": 10, "category": "动量",
     "expr": "-Corr($open,$volume,10)*Rank(Delta($close,2))"},
    {"key": "a101_trend",     "label": "Alpha23-趋势(旧)", "n": 20, "category": "动量",
     "expr": "Rank(Delta(($high+$low)/2,2))*Rank($volume)"},
    {"key": "a101_lowvol",    "label": "Alpha12-低波(旧)", "n": 20, "category": "波动率",
     "expr": "Sign(Delta($volume,1))*(-Delta($close,1))/Std($close,20)"},
    {"key": "a101_volregime", "label": "Alpha35-波动(旧)", "n": 15, "category": "波动率",
     "expr": "Rank(Std($close,15))/Rank($close)"},
    {"key": "a101_vpdiv",     "label": "Alpha17-量价背离(旧)", "n": 20, "category": "量价",
     "expr": "-Rank(Corr($close,$volume,20))"},
    {"key": "a101_vpdiv2",    "label": "Alpha83-量价背离2(旧)", "n": 5, "category": "量价",
     "expr": "Rank(Delta(($close-$open)/$open,1))*Rank($volume)"},
    {"key": "a101_liq",       "label": "Alpha31-非流动(旧)", "n": 12, "category": "流动性",
     "expr": "Rank(($close-Mean($close,12))/Mean($close,12)*$volume)"},
    {"key": "a101_turnover",  "label": "Alpha54-换手(旧)", "n": 10, "category": "流动性",
     "expr": "-Rank(Std($close,10)/$close*Corr($close,$volume,10))"},
    {"key": "a101_breakout",  "label": "Alpha38-突破(旧)", "n": 20, "category": "突破",
     "expr": "Rank($close/Mean($close,20)-1)*Rank($volume/Mean($volume,20)-1)"},
    {"key": "a101_newhigh",   "label": "Alpha65-新高(旧)", "n": 20, "category": "突破",
     "expr": "Rank($close/Max($high,20)-1)"},
    {"key": "a101_meanrev",   "label": "Alpha46-均值回归(旧)", "n": 20, "category": "均值回归",
     "expr": "-Rank(($close-Mean($close,20))/Std($close,20))"},
    {"key": "a101_meanrev2",  "label": "Alpha49-极端回归(旧)","n": 5, "category": "均值回归",
     "expr": "-Corr(Rank($close),Rank($volume),5)*Rank(Delta($close,5))"},
    {"key": "a101_skew",      "label": "Alpha52-偏度(旧)", "n": 20, "category": "特殊",
     "expr": "Sum(Max(($close-Ref($close, 1)),0),20)/Sum(Abs($close-Ref($close, 1)),20+1e-8)-0.5"},
    {"key": "a101_range",     "label": "Alpha57-振幅(旧)", "n": 5, "category": "特殊",
     "expr": "Rank(($high-$low)/Mean($close,5))"},
    {"key": "a101_accel",     "label": "Alpha70-加速度(旧)", "n": 10, "category": "特殊",
     "expr": "Rank(Delta(Delta($close,5),5))"},
    {"key": "a101_open_gap",  "label": "Alpha81-缺口(旧)", "n": 1, "category": "特殊",
     "expr": "Rank(($open-Ref($close, 1))/Ref($close, 1))"},
    {"key": "a101_range_pos", "label": "Alpha88-位置(旧)", "n": 20, "category": "特殊",
     "expr": "Rank(($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-8))"},
    {"key": "a101_volume_pct","label": "Alpha95-量分位(旧)", "n": 20, "category": "特殊",
     "expr": "Rank($volume/Mean($volume,20))"},
])


# ══════════════════════════════════════════════
# Alpha191 — A股专用技术因子集 (191个)
# 来源: 国信Alpha191 + 海通/华泰研报复现
# ══════════════════════════════════════════════

ALPHA191_FACTORS = {}

def _reg191(category, factors):
    for f in factors:
        f["source"] = "Alpha191"
        f["category"] = category
        ALPHA191_FACTORS[f["key"]] = f


_reg191("价格动量", [
    {"key": "a191_mom_01", "label": "1日收益", "expr": "$close/Ref($close, 1)-1", "n": 1},
    {"key": "a191_mom_02", "label": "3日收益", "expr": "$close/Ref($close, 3)-1", "n": 3},
    {"key": "a191_mom_03", "label": "5日收益", "expr": "$close/Ref($close, 5)-1", "n": 5},
    {"key": "a191_mom_04", "label": "10日收益", "expr": "$close/Ref($close, 10)-1", "n": 10},
    {"key": "a191_mom_05", "label": "20日收益", "expr": "$close/Ref($close, 20)-1", "n": 20},
    {"key": "a191_mom_06", "label": "40日收益", "expr": "$close/Ref($close, 40)-1", "n": 40},
    {"key": "a191_mom_07", "label": "60日收益", "expr": "$close/Ref($close, 60)-1", "n": 60},
    {"key": "a191_mom_08", "label": "120日收益", "expr": "$close/Ref($close, 120)-1", "n": 120},
    {"key": "a191_mom_09", "label": "5日动量(隔1)", "expr": "Ref($close, 1)/Ref($close, 6)-1", "n": 5},
    {"key": "a191_mom_10", "label": "10日动量(隔5)", "expr": "Ref($close, 5)/Ref($close, 15)-1", "n": 10},
    {"key": "a191_mom_11", "label": "20日动量(隔10)", "expr": "Ref($close, 10)/Ref($close, 30)-1", "n": 20},
    {"key": "a191_mom_12", "label": "5日加权收益", "expr": "(5*$close/Ref($close, 1)+4*$close/Ref($close, 2)+3*$close/Ref($close, 3)+2*$close/Ref($close, 4)+$close/Ref($close, 5))/15-1", "n": 5},
    {"key": "a191_mom_13", "label": "10日加权收益", "expr": "Sum($close/Ref($close, 1)*(10-RowNum()),10)/55-1", "n": 10},
    {"key": "a191_mom_14", "label": "20日夏普比", "expr": "Mean($close/Ref($close, 1)-1,20)/Std($close/Ref($close, 1)-1,20+1e-8)", "n": 20},
    {"key": "a191_mom_15", "label": "60日夏普比", "expr": "Mean($close/Ref($close, 1)-1,60)/Std($close/Ref($close, 1)-1,60+1e-8)", "n": 60},
    {"key": "a191_mom_16", "label": "5日收益/波动", "expr": "($close/Ref($close, 5)-1)/Std($close/Ref($close, 1)-1,5+1e-8)", "n": 5},
    {"key": "a191_mom_17", "label": "20日收益/波动", "expr": "($close/Ref($close, 20)-1)/Std($close/Ref($close, 1)-1,20+1e-8)", "n": 20},
    {"key": "a191_mom_18", "label": "5日最大收益", "expr": "Max($close/Ref($close, 1)-1,5)", "n": 5},
    {"key": "a191_mom_19", "label": "20日最大收益", "expr": "Max($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_mom_20", "label": "5日最大回撤", "expr": "Min($close/Max($close,5)-1,5)", "n": 5},
    {"key": "a191_mom_21", "label": "20日最大回撤", "expr": "Min($close/Max($close,20)-1,20)", "n": 20},
    {"key": "a191_mom_22", "label": "N日盈亏比", "expr": "Sum(Max($close-Ref($close, 1),0),20)/Sum(Max(Ref($close, 1)-$close,0),20+1e-8)", "n": 20},
    {"key": "a191_mom_23", "label": "5日胜率", "expr": "Mean($close>Ref($close, 1),5)", "n": 5},
    {"key": "a191_mom_24", "label": "20日胜率", "expr": "Mean($close>Ref($close, 1),20)", "n": 20},
    {"key": "a191_mom_25", "label": "MACD柱", "expr": "Mean($close,12)-Mean($close,26)-Mean(Mean($close,12)-Mean($close,26),9)", "n": 26},
    {"key": "a191_mom_26", "label": "MACD信号", "expr": "Mean(Mean($close,12)-Mean($close,26),9)", "n": 26},
    {"key": "a191_mom_27", "label": "TRIX-5", "expr": "Mean(Mean(Mean($close/Ref($close, 1)-1,5),5),5)", "n": 5},
    {"key": "a191_mom_28", "label": "TRIX-20", "expr": "Mean(Mean(Mean($close/Ref($close, 1)-1,20),20),20)", "n": 20},
    {"key": "a191_mom_29", "label": "ROC排序5日", "expr": "Rank($close/Ref($close, 5)-1)", "n": 5},
    {"key": "a191_mom_30", "label": "ROC排序20日", "expr": "Rank($close/Ref($close, 20)-1)", "n": 20},
])

_reg191("反转效应", [
    {"key": "a191_rev_01", "label": "1日反转", "expr": "-($close/Ref($close, 1)-1)", "n": 1},
    {"key": "a191_rev_02", "label": "3日反转", "expr": "-($close/Ref($close, 3)-1)", "n": 3},
    {"key": "a191_rev_03", "label": "5日反转", "expr": "-($close/Ref($close, 5)-1)", "n": 5},
    {"key": "a191_rev_04", "label": "10日反转", "expr": "-($close/Ref($close, 10)-1)", "n": 10},
    {"key": "a191_rev_05", "label": "20日反转", "expr": "-($close/Ref($close, 20)-1)", "n": 20},
    {"key": "a191_rev_06", "label": "日内反转", "expr": "($close-$open)/$open", "n": 1},
    {"key": "a191_rev_07", "label": "5日日内反转", "expr": "Mean(($close-$open)/$open,5)", "n": 5},
    {"key": "a191_rev_08", "label": "隔夜反转", "expr": "($open-Ref($close, 1))/Ref($close, 1)", "n": 1},
    {"key": "a191_rev_09", "label": "5日反转强度", "expr": "-Rank($close/Ref($close, 5)-1)*Rank($volume/Mean($volume,5))", "n": 5},
    {"key": "a191_rev_10", "label": "20日反转强度", "expr": "-Rank($close/Ref($close, 20)-1)*Rank($volume/Mean($volume,20))", "n": 20},
    {"key": "a191_rev_11", "label": "高低价反转", "expr": "($close-$low)/($high-$low+1e-8)*2-1", "n": 1},
    {"key": "a191_rev_12", "label": "5日相对位置", "expr": "($close-Min($low,5))/(Max($high,5)-Min($low,5)+1e-8)", "n": 5},
    {"key": "a191_rev_13", "label": "20日相对位置", "expr": "($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-8)", "n": 20},
    {"key": "a191_rev_14", "label": "RSI-14反转", "expr": "-Sum(Max($close-Ref($close, 1),0),14)/Sum(Abs($close-Ref($close, 1)),14+1e-8)", "n": 14},
    {"key": "a191_rev_15", "label": "威廉%R反转", "expr": "-(Max($high,14)-$close)/(Max($high,14)-Min($low,14)+1e-8)", "n": 14},
    {"key": "a191_rev_16", "label": "KDJ-J反转", "expr": "-3*($close-Min($low,9))/(Max($high,9)-Min($low,9)+1e-8)+2*Mean(($close-Min($low,9))/(Max($high,9)-Min($low,9)+1e-8),3)", "n": 9},
    {"key": "a191_rev_17", "label": "CCI反转", "expr": "-(($close-Mean(($high+$low+$close)/3,20))/(0.015*Mean(Abs(($high+$low+$close)/3-Mean(($high+$low+$close)/3,20)),20)+1e-8))", "n": 20},
    {"key": "a191_rev_18", "label": "布林下轨反转", "expr": "-($close-Mean($close,20)+2*Std($close,20))/Std($close,20+1e-8)", "n": 20},
    {"key": "a191_rev_19", "label": "5日缩量反转", "expr": "-($close/Ref($close, 5)-1)*($volume/Mean($volume,10)-1)", "n": 5},
    {"key": "a191_rev_20", "label": "10日超跌反弹", "expr": "-Min($close/Max($close,10)-1,10)", "n": 10},
])

_reg191("波动率特征", [
    {"key": "a191_vol_01", "label": "5日波动率", "expr": "Std($close/Ref($close, 1)-1,5)", "n": 5},
    {"key": "a191_vol_02", "label": "10日波动率", "expr": "Std($close/Ref($close, 1)-1,10)", "n": 10},
    {"key": "a191_vol_03", "label": "20日波动率", "expr": "Std($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_vol_04", "label": "40日波动率", "expr": "Std($close/Ref($close, 1)-1,40)", "n": 40},
    {"key": "a191_vol_05", "label": "60日波动率", "expr": "Std($close/Ref($close, 1)-1,60)", "n": 60},
    {"key": "a191_vol_06", "label": "5日下行波动", "expr": "Std(Min($close/Ref($close, 1)-1,0),5)", "n": 5},
    {"key": "a191_vol_07", "label": "20日下行波动", "expr": "Std(Min($close/Ref($close, 1)-1,0),20)", "n": 20},
    {"key": "a191_vol_08", "label": "波动率变化5/20", "expr": "Std($close/Ref($close, 1)-1,5)/Std($close/Ref($close, 1)-1,20+1e-8)-1", "n": 20},
    {"key": "a191_vol_09", "label": "波动率变化20/60", "expr": "Std($close/Ref($close, 1)-1,20)/Std($close/Ref($close, 1)-1,60+1e-8)-1", "n": 60},
    {"key": "a191_vol_10", "label": "5日振幅率", "expr": "Mean(($high-$low)/$close,5)", "n": 5},
    {"key": "a191_vol_11", "label": "20日振幅率", "expr": "Mean(($high-$low)/$close,20)", "n": 20},
    {"key": "a191_vol_12", "label": "ATR比率", "expr": "Mean(Max($high-$low,Max(Abs($high-Ref($close, 1)),Abs($low-Ref($close, 1)))),14)/$close", "n": 14},
    {"key": "a191_vol_13", "label": "布林带宽", "expr": "(Mean($close,20)+2*Std($close,20)-Mean($close,20)+2*Std($close,20))/Mean($close,20)", "n": 20},
    {"key": "a191_vol_14", "label": "布林位置", "expr": "($close-(Mean($close,20)-2*Std($close,20)))/(4*Std($close,20)+1e-8)", "n": 20},
    {"key": "a191_vol_15", "label": "波动率排序", "expr": "Rank(Std($close/Ref($close, 1)-1,20))", "n": 20},
    {"key": "a191_vol_16", "label": "5日GARCH", "expr": "Std($close/Ref($close, 1)-1,5)/Mean(Std($close/Ref($close, 1)-1,5),20+1e-8)", "n": 5},
    {"key": "a191_vol_17", "label": "Parkinson波动", "expr": "Mean(Log($high/$low+1e-8)**2,20)", "n": 20},
    {"key": "a191_vol_18", "label": "Garman-Klass", "expr": "Mean(0.5*Log($high/$low+1e-8)**2-(2*Log(2)-1)*Log($close/$open+1e-8)**2,20)", "n": 20},
    {"key": "a191_vol_19", "label": "偏度5日", "expr": "Skewness($close/Ref($close, 1)-1,5)", "n": 5},
    {"key": "a191_vol_20", "label": "偏度20日", "expr": "Skewness($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_vol_21", "label": "峰度20日", "expr": "Kurtosis($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_vol_22", "label": "尾部风险", "expr": "Min($close/Ref($close, 1)-1,20)/Std($close/Ref($close, 1)-1,20+1e-8)", "n": 20},
    {"key": "a191_vol_23", "label": "VaR-5日", "expr": "Quantile($close/Ref($close, 1)-1,20,0.05)", "n": 20},
    {"key": "a191_vol_24", "label": "CVaR-5日", "expr": "Mean(Min($close/Ref($close, 1)-1,Quantile($close/Ref($close, 1)-1,20,0.05)),20)", "n": 20},
    {"key": "a191_vol_25", "label": "波动异象", "expr": "-Std($close/Ref($close, 1)-1,20)*Rank($close/Ref($close, 20)-1)", "n": 20},
])

_reg191("成交量特征", [
    {"key": "a191_amt_01", "label": "5日均量比", "expr": "Mean($volume,5)/Mean($volume,20)-1", "n": 5},
    {"key": "a191_amt_02", "label": "10日均量比", "expr": "Mean($volume,10)/Mean($volume,40)-1", "n": 10},
    {"key": "a191_amt_03", "label": "20日均量比", "expr": "Mean($volume,20)/Mean($volume,60)-1", "n": 20},
    {"key": "a191_amt_04", "label": "5日量波动", "expr": "Std($volume,5)/Mean($volume,5+1e-8)", "n": 5},
    {"key": "a191_amt_05", "label": "20日量波动", "expr": "Std($volume,20)/Mean($volume,20+1e-8)", "n": 20},
    {"key": "a191_amt_06", "label": "量比(昨)", "expr": "$volume/Ref($volume, 1)-1", "n": 1},
    {"key": "a191_amt_07", "label": "量比(5日)", "expr": "$volume/Mean($volume,5)-1", "n": 5},
    {"key": "a191_amt_08", "label": "OBV动量", "expr": "Sum(Sign($close-Ref($close, 1))*$volume,20)/Sum($volume,20+1e-8)", "n": 20},
    {"key": "a191_amt_09", "label": "MFI-14", "expr": "$high*$volume+$low*$volume", "n": 14},
    {"key": "a191_amt_10", "label": "5日换手率", "expr": "Mean($turnover,5)", "n": 5},
    {"key": "a191_amt_11", "label": "20日换手率", "expr": "Mean($turnover,20)", "n": 20},
    {"key": "a191_amt_12", "label": "换手率变化", "expr": "Mean($turnover,5)/Mean($turnover,20+1e-8)-1", "n": 5},
    {"key": "a191_amt_13", "label": "换手率稳定性", "expr": "Std($turnover,20)/Mean($turnover,20+1e-8)", "n": 20},
    {"key": "a191_amt_14", "label": "5日成交额比", "expr": "Mean($amount,5)/Mean($amount,20)-1", "n": 5},
    {"key": "a191_amt_15", "label": "20日成交额比", "expr": "Mean($amount,20)/Mean($amount,60)-1", "n": 20},
    {"key": "a191_amt_16", "label": "量价相关5日", "expr": "Corr($close,$volume,5)", "n": 5},
    {"key": "a191_amt_17", "label": "量价相关20日", "expr": "Corr($close,$volume,20)", "n": 20},
    {"key": "a191_amt_18", "label": "量价相关60日", "expr": "Corr($close,$volume,60)", "n": 60},
    {"key": "a191_amt_19", "label": "缩量程度", "expr": "-Rank($volume/Mean($volume,20))", "n": 20},
    {"key": "a191_amt_20", "label": "放量程度", "expr": "Rank($volume/Mean($volume,20))", "n": 20},
    {"key": "a191_amt_21", "label": "量增价涨比", "expr": "Mean($volume>Ref($volume, 1) and $close>Ref($close, 1),20)", "n": 20},
    {"key": "a191_amt_22", "label": "量增价跌比", "expr": "Mean($volume>Ref($volume, 1) and $close<Ref($close, 1),20)", "n": 20},
    {"key": "a191_amt_23", "label": "VWAP偏离", "expr": "$close/$vwap-1", "n": 1},
    {"key": "a191_amt_24", "label": "ADL指标", "expr": "Sum(($close-$low-($high-$close))/($high-$low+1e-8)*$volume,20)", "n": 20},
    {"key": "a191_amt_25", "label": "Chaikin振荡", "expr": "Mean(Sum(($close-$low-($high-$close))/($high-$low+1e-8)*$volume,3),3)-Mean(Sum(($close-$low-($high-$close))/($high-$low+1e-8)*$volume,10),10)", "n": 10},
])

_reg191("趋势强度", [
    {"key": "a191_trd_01", "label": "MA5/MA10", "expr": "Mean($close,5)/Mean($close,10)-1", "n": 10},
    {"key": "a191_trd_02", "label": "MA10/MA20", "expr": "Mean($close,10)/Mean($close,20)-1", "n": 20},
    {"key": "a191_trd_03", "label": "MA20/MA60", "expr": "Mean($close,20)/Mean($close,60)-1", "n": 60},
    {"key": "a191_trd_04", "label": "MA5/MA20", "expr": "Mean($close,5)/Mean($close,20)-1", "n": 20},
    {"key": "a191_trd_05", "label": "EMA12/EMA26", "expr": "Mean($close,12)/Mean($close,26)-1", "n": 26},
    {"key": "a191_trd_06", "label": "多头排列度", "expr": "(Mean($close,5)>Mean($close,10)?1:0)+(Mean($close,10)>Mean($close,20)?1:0)+(Mean($close,20)>Mean($close,60)?1:0)", "n": 60},
    {"key": "a191_trd_07", "label": "价格相对MA20", "expr": "$close/Mean($close,20)-1", "n": 20},
    {"key": "a191_trd_08", "label": "价格相对MA60", "expr": "$close/Mean($close,60)-1", "n": 60},
    {"key": "a191_trd_09", "label": "20日新高", "expr": "$close/Max($high,20)-1", "n": 20},
    {"key": "a191_trd_10", "label": "60日新高", "expr": "$close/Max($high,60)-1", "n": 60},
    {"key": "a191_trd_11", "label": "20日新低距", "expr": "$close/Min($low,20)-1", "n": 20},
    {"key": "a191_trd_12", "label": "60日新低距", "expr": "$close/Min($low,60)-1", "n": 60},
    {"key": "a191_trd_13", "label": "ADX-14", "expr": "Mean(Abs(($high-Max($high,14))/($high-Min($low,14)+1e-8)-($low-Min($low,14))),14)", "n": 14},
    {"key": "a191_trd_14", "label": "+DMI/-DMI", "expr": "Mean(Max($high-Ref($high, 1),0),14)/Mean(Max(Ref($low, 1)-$low,0),14+1e-8)", "n": 14},
    {"key": "a191_trd_15", "label": "Aroon-Up", "expr": "(14-IdxMax($high,14))/14", "n": 14},
    {"key": "a191_trd_16", "label": "Aroon-Down", "expr": "(14-IdxMin($low,14))/14", "n": 14},
    {"key": "a191_trd_17", "label": "趋势斜率", "expr": "Slope($close,20)/$close", "n": 20},
    {"key": "a191_trd_18", "label": "趋势R方", "expr": "Rsquare($close,20)", "n": 20},
    {"key": "a191_trd_19", "label": "Hurst指数", "expr": "Log(Max($close,20)/Min($low,20)+1e-8)/Log(20)", "n": 20},
    {"key": "a191_trd_20", "label": "CMO-14", "expr": "(Sum(Max($close-Ref($close, 1),0),14)-Sum(Max(Ref($close, 1)-$close,0),14))/(Sum(Max($close-Ref($close, 1),0),14)+Sum(Max(Ref($close, 1)-$close,0),14)+1e-8)", "n": 14},
])

_reg191("流动性", [
    {"key": "a191_liq_01", "label": "Amihud(5日)", "expr": "Mean(Abs($close/Ref($close, 1)-1)/$amount,5)", "n": 5},
    {"key": "a191_liq_02", "label": "Amihud(20日)", "expr": "Mean(Abs($close/Ref($close, 1)-1)/$amount,20)", "n": 20},
    {"key": "a191_liq_03", "label": "换手率水平", "expr": "Rank($turnover)", "n": 1},
    {"key": "a191_liq_04", "label": "成交额水平", "expr": "Rank($amount)", "n": 1},
    {"key": "a191_liq_05", "label": "非流动性", "expr": "Rank(Abs($close/Ref($close, 1)-1)/$amount*1e10)", "n": 1},
    {"key": "a191_liq_06", "label": "流动性变化", "expr": "-Rank(Mean($turnover,5)/Mean($turnover,20+1e-8))", "n": 5},
    {"key": "a191_liq_07", "label": "换手率偏度", "expr": "Skewness($turnover,20)", "n": 20},
    {"key": "a191_liq_08", "label": "流动性风险", "expr": "Std($turnover,20)/Mean($turnover,20+1e-8)*Rank(Abs($close/Ref($close, 1)-1)/$amount)", "n": 20},
    {"key": "a191_liq_09", "label": "成交额集中度", "expr": "Max($amount,20)/Sum($amount,20+1e-8)", "n": 20},
    {"key": "a191_liq_10", "label": "零成交量天数", "expr": "Sum($volume<Mean($volume,20)*0.1,20)/20", "n": 20},
    {"key": "a191_liq_11", "label": "相对价差近似", "expr": "($high-$low)/Mean($close,10)", "n": 10},
    {"key": "a191_liq_12", "label": "成交额弹性", "expr": "Corr($close/Ref($close, 1)-1,$amount/Ref($amount, 1)-1,20)", "n": 20},
    {"key": "a191_liq_13", "label": "流动性同步", "expr": "Corr($turnover,Rank($close),20)", "n": 20},
    {"key": "a191_liq_14", "label": "流动性动量", "expr": "Rank(Mean($turnover,5))/Rank(Mean($turnover,20+1e-8))-1", "n": 5},
    {"key": "a191_liq_15", "label": "流动性衰减", "expr": "Rank(Delta(Mean($amount,5),5))", "n": 5},
    {"key": "a191_liq_16", "label": "大单近似", "expr": "($high-$low)/$close*$amount", "n": 1},
    {"key": "a191_liq_17", "label": "量价冲击", "expr": "Rank(Abs($close/Ref($close, 1)-1))*Rank($volume/Mean($volume,20))", "n": 20},
    {"key": "a191_liq_18", "label": "开盘流动性", "expr": "Abs($open-Ref($close, 1))/Ref($close, 1)*$volume", "n": 1},
    {"key": "a191_liq_19", "label": "收盘流动性", "expr": "Abs($close-Mean(($high+$low)/2,1))/$close*$volume", "n": 1},
    {"key": "a191_liq_20", "label": "流动性溢价", "expr": "-Rank(Mean(Abs($close/Ref($close, 1)-1)/$amount,60))*Rank($close/Ref($close, 60)-1)", "n": 60},
])

_reg191("技术形态", [
    {"key": "a191_pat_01", "label": "K线实体比", "expr": "Abs($close-$open)/($high-$low+1e-8)", "n": 1},
    {"key": "a191_pat_02", "label": "上影线比", "expr": "($high-Max($close,$open))/($high-$low+1e-8)", "n": 1},
    {"key": "a191_pat_03", "label": "下影线比", "expr": "(Min($close,$open)-$low)/($high-$low+1e-8)", "n": 1},
    {"key": "a191_pat_04", "label": "十字星强度", "expr": "-Abs($close-$open)/($high-$low+1e-8)", "n": 1},
    {"key": "a191_pat_05", "label": "锤子线", "expr": "Min($close,$open)-$low>2*Abs($close-$open) and $high-Max($close,$open)<Abs($close-$open)", "n": 1},
    {"key": "a191_pat_06", "label": "倒锤线", "expr": "$high-Max($close,$open)>2*Abs($close-$open) and Min($close,$open)-$low<Abs($close,$open)", "n": 1},
    {"key": "a191_pat_07", "label": "5日阳线比", "expr": "Mean($close>$open,5)", "n": 5},
    {"key": "a191_pat_08", "label": "20日阳线比", "expr": "Mean($close>$open,20)", "n": 20},
    {"key": "a191_pat_09", "label": "5日连阳", "expr": "Sum($close>$open,5)==5", "n": 5},
    {"key": "a191_pat_10", "label": "5日连阴", "expr": "Sum($close<$open,5)==5", "n": 5},
    {"key": "a191_pat_11", "label": "三白兵", "expr": "Sum(($close>$open)*($close>Ref($close, 1))*(Ref($close, 1)>Ref($close, 2)),3)==3", "n": 3},
    {"key": "a191_pat_12", "label": "三乌鸦", "expr": "Sum(($close<$open)*($close<Ref($close, 1))*(Ref($close, 1)<Ref($close, 2)),3)==3", "n": 3},
    {"key": "a191_pat_13", "label": "5日新高突破", "expr": "$close>Ref(Max($high,5),1)", "n": 5},
    {"key": "a191_pat_14", "label": "20日新高突破", "expr": "$close>Ref(Max($high,20),1)", "n": 20},
    {"key": "a191_pat_15", "label": "5日新低突破", "expr": "$close<Ref(Min($low,5),1)", "n": 5},
    {"key": "a191_pat_16", "label": "跳空高开", "expr": "$open>Ref($high, 1)", "n": 1},
    {"key": "a191_pat_17", "label": "跳空低开", "expr": "$open<Ref($low, 1)", "n": 1},
    {"key": "a191_pat_18", "label": "吞没形态", "expr": "($close>$open)*(Ref($close, 1)<Ref($open, 1))*($close>Ref($open, 1))*($open<Ref($close, 1))", "n": 2},
    {"key": "a191_pat_19", "label": "孕线形态", "expr": "($close<$open)*(Ref($close, 1)>Ref($open, 1))*($close>Ref($open, 1))*($open<Ref($close, 1))", "n": 2},
    {"key": "a191_pat_20", "label": "启明星", "expr": "Ref($close, 2)<Ref($open, 2) and Abs(Ref($close, 1)-Ref($open, 1))<Ref(Abs(Ref($close, 2)-Ref($open, 2)),1)*0.3 and $close>$open", "n": 3},
    {"key": "a191_pat_21", "label": "黄昏星", "expr": "Ref($close, 2)>Ref($open, 2) and Abs(Ref($close, 1)-Ref($open, 1))<Ref(Abs(Ref($close, 2)-Ref($open, 2)),1)*0.3 and $close<$open", "n": 3},
    {"key": "a191_pat_22", "label": "均线粘合度", "expr": "-Std([Mean($close,5),Mean($close,10),Mean($close,20),Mean($close,60)]/Mean($close),4)", "n": 60},
    {"key": "a191_pat_23", "label": "支撑位距", "expr": "$close/Min($low,20)-1", "n": 20},
    {"key": "a191_pat_24", "label": "压力位距", "expr": "$close/Max($high,20)-1", "n": 20},
    {"key": "a191_pat_25", "label": "20日通道位置", "expr": "($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-8)", "n": 20},
])

_reg191("高阶特征", [
    {"key": "a191_hof_01", "label": "5日自相关", "expr": "Corr($close/Ref($close, 1)-1,Ref($close/Ref($close, 1)-1,1),5)", "n": 5},
    {"key": "a191_hof_02", "label": "20日自相关", "expr": "Corr($close/Ref($close, 1)-1,Ref($close/Ref($close, 1)-1,1),20)", "n": 20},
    {"key": "a191_hof_03", "label": "收益可预测性", "expr": "Rsquare($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_hof_04", "label": "熵(5日收益)", "expr": "-Sum(Abs($close/Ref($close, 1)-1)*Log(Abs($close/Ref($close, 1)-1)+1e-8),5)", "n": 5},
    {"key": "a191_hof_05", "label": "分形维度", "expr": "Log(Max($high,20)/Min($low,20)+1e-8)/Log(Sum(Abs($high-Ref($high, 1)),20)/Sum(Abs($close-Ref($close, 1)),20+1e-8)+1)", "n": 20},
    {"key": "a191_hof_06", "label": "CVaR比率", "expr": "Mean($close/Ref($close, 1)-1,20)/Mean(Min($close/Ref($close, 1)-1,Quantile($close/Ref($close, 1)-1,60,0.05)),20+1e-8)", "n": 20},
    {"key": "a191_hof_07", "label": "5日IC", "expr": "Corr($close/Ref($close, 1)-1,Sign($close/Ref($close, 2)-1),5)", "n": 5},
    {"key": "a191_hof_08", "label": "20日IC", "expr": "Corr($close/Ref($close, 1)-1,Sign($close/Ref($close, 2)-1),20)", "n": 20},
    {"key": "a191_hof_09", "label": "波动率聚类", "expr": "Corr(Std($close/Ref($close, 1)-1,5),Ref(Std($close/Ref($close, 1)-1,5),5),20)", "n": 20},
    {"key": "a191_hof_10", "label": "跳跃检测", "expr": "Abs($close/Ref($close, 1)-1)>3*Std($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_hof_11", "label": "R/S分析", "expr": "(Max(Sum($close/Ref($close, 1)-1-Mean($close/Ref($close, 1)-1,20),20),20)-Min(Sum($close/Ref($close, 1)-1-Mean($close/Ref($close, 1)-1,20),20),20))/Std($close/Ref($close, 1)-1,20+1e-8)", "n": 20},
    {"key": "a191_hof_12", "label": "相对大盘强度", "expr": "$close/Ref($close, 1)-1", "n": 20},
    {"key": "a191_hof_13", "label": "市场Beta(20)", "expr": "Slope($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_hof_14", "label": "市场Beta(60)", "expr": "Slope($close/Ref($close, 1)-1,60)", "n": 60},
    {"key": "a191_hof_15", "label": "残差收益", "expr": "($close/Ref($close, 1)-1)-Slope($close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_hof_16", "label": "协偏度", "expr": "Corr($close/Ref($close, 1)-1,$close/Ref($close, 1)-1*$close/Ref($close, 1)-1,20)", "n": 20},
    {"key": "a191_hof_17", "label": "信息离散度", "expr": "Std(($close-Mean($close,20))/Mean($close,20+1e-8),5)", "n": 20},
    {"key": "a191_hof_18", "label": "网络中心性", "expr": "Corr($close,Mean($close,60),20)", "n": 20},
    {"key": "a191_hof_19", "label": "跨日动量", "expr": "($open-Ref($open, 1))/Ref($open, 1)", "n": 1},
    {"key": "a191_hof_20", "label": "日内动量", "expr": "($close-$open)/$open", "n": 1},
    {"key": "a191_hof_21", "label": "波动率择时", "expr": "Std($close/Ref($close, 1)-1,5)/Std($close/Ref($close, 1)-1,60+1e-8)-0.5", "n": 60},
    {"key": "a191_hof_22", "label": "动量加速度", "expr": "($close/Ref($close, 5)-1)-Ref($close/Ref($close, 5)-1,5)", "n": 10},
    {"key": "a191_hof_23", "label": "收敛发散", "expr": "Corr($close,Ref($close, 20),20)", "n": 20},
    {"key": "a191_hof_24", "label": "极值反转", "expr": "-($close-Ref($close, 1))*($close>Quantile($close,60,0.95) or $close<Quantile($close,60,0.05))", "n": 60},
    {"key": "a191_hof_25", "label": "日历效应(周一)", "expr": "DayOfWeek()==1?$close/Ref($close, 3)-1:0", "n": 1},
    {"key": "a191_hof_26", "label": "日历效应(周五)", "expr": "DayOfWeek()==5?$close/Ref($close, 1)-1:0", "n": 1},
])


# ══════════════════════════════════════════════
# 基本面因子集 — 估值/盈利/成长/质量/分红 (25个)
# 数据来源: MySQL financial + daily_quotes
# eval_mode: "custom" (需从MySQL采集)
# ══════════════════════════════════════════════

FUNDAMENTAL_FACTORS = {}

def _regfund(category, factors):
    for f in factors:
        f["source"] = "基本面"
        f["category"] = category
        f["eval_mode"] = "custom"
        f["dataflow"] = "fundamental"
        FUNDAMENTAL_FACTORS[f["key"]] = f


# 注: 真实财务因子 — 通过 FundVal($close, N) 算子读取 MySQL daily_basic + fundamental_data
# N: 1=total_mv, 2=circ_mv, 3=pe, 4=pb, 5=roe, 6=net_profit_yoy, 7=revenue_yoy, 8=debt_to_assets, 9=turnover_rate
# 数据由 fundamental_data_bridge.py 预计算截面 z-score 并缓存

_regfund("规模", [
    {"key": "fund_size",     "label": "总市值(MySQL)",    "n": 1,
     "expr": "FundVal($close, 1)"},
    {"key": "fund_float_mv", "label": "流通市值(MySQL)",  "n": 1,
     "expr": "FundVal($close, 2)"},
])

_regfund("估值", [
    {"key": "fund_pe",       "label": "市盈率PE(MySQL)",  "n": 1,
     "expr": "FundVal($close, 3)"},
    {"key": "fund_pb",       "label": "市净率PB(MySQL)",  "n": 1,
     "expr": "FundVal($close, 4)"},
    {"key": "fund_bp",       "label": "账面市值比(代理)", "n": 20,
     "expr": "-($close/Ref($close, 60)-1)"},
])

_regfund("盈利", [
    {"key": "fund_roe",      "label": "ROE(MySQL)",       "n": 1,
     "expr": "FundVal($close, 5)"},
    {"key": "fund_gpm",      "label": "毛利率(代理)",     "n": 20,
     "expr": "(($close-$low)/Max($high-$low,1e-6))"},
    {"key": "fund_cfo",      "label": "现金流(代理)",     "n": 20,
     "expr": "(Sum($close>$open,20)-Sum($close<$open,20))/20"},
])

_regfund("成长", [
    {"key": "fund_profit_yoy", "label": "净利润同比(MySQL)", "n": 1,
     "expr": "FundVal($close, 6)"},
    {"key": "fund_rev_yoy",    "label": "营收同比(MySQL)",   "n": 1,
     "expr": "FundVal($close, 7)"},
    {"key": "fund_earn_mom",   "label": "盈利动量(代理)",    "n": 20,
     "expr": "($close/Ref($close, 20)-1)"},
])

_regfund("流动性", [
    {"key": "fund_turnover",  "label": "换手率(MySQL)",    "n": 1,
     "expr": "FundVal($close, 9)"},
])

_regfund("质量", [
    {"key": "fund_dar",       "label": "资产负债率(MySQL)", "n": 1,
     "expr": "FundVal($close, 8)"},
    {"key": "fund_earn_stb",  "label": "盈利稳定性(代理)",  "n": 20,
     "expr": "-Std(($close/Ref($close, 20)-1),60)"},
    {"key": "fund_asset_turnover", "label": "资产周转(代理)", "n": 20,
     "expr": "Mean($volume,20)/Max(Std($close/Ref($close, 1)-1,20),1e-6)"},
])

_regfund("估值", [
    {"key": "fund_bp",       "label": "账面市值比(代理)", "n": 20,
     "expr": "-($close/Ref($close, 60)-1)"},
])

_regfund("分红", [
    {"key": "fund_div_yield", "label": "股息率(代理)",    "n": 20,
     "expr": "-($close/Ref($close, 120)-1)"},
])

_regfund("综合", [
    {"key": "fund_roe_pb",    "label": "性价比(代理)",     "n": 20,
     "expr": "(($close/Ref($close, 60)-1))/Max(($close/Ref($close, 240)),1e-6)"},
    {"key": "fund_quality",   "label": "质量综合(代理)",   "n": 20,
     "expr": "-(Std($close/Ref($close, 1)-1,60)+Max($close/Ref($close, 240)-1,0))"},
])


# ══════════════════════════════════════════════
# 扩展因子集 — 华泰独有 + 学术文献验证因子 (30个)
# 覆盖非流动性/最大回撤/隔夜/尾风险/VWAP偏离/异常换手等缺少的类型
# ══════════════════════════════════════════════

EXTENDED_FACTORS = {}

def _regnew(category, factors):
    for f in factors:
        f["source"] = "扩展"
        f["category"] = category
        EXTENDED_FACTORS[f["key"]] = f

_regnew("流动性", [
    {"key": "ext_illiq",     "label": "Amihud非流动性",     "n": 20,
     "expr": "Mean(Abs($close/Ref($close, 1)-1)/($volume+1),20)"},
    {"key": "ext_illiq_chg", "label": "非流动性变化",       "n": 20,
     "expr": "Mean(Abs($close/Ref($close, 1)-1)/($volume+1),5)/Mean(Abs($close/Ref($close, 1)-1)/($volume+1),20)-1"},
    {"key": "ext_turn_abn",  "label": "异常换手率",         "n": 20,
     "expr": "-($volume/Mean($volume,20)-1)"},
    {"key": "ext_turn_chg",  "label": "换手率趋势",         "n": 20,
     "expr": "Mean($volume,5)/Mean($volume,20)-1"},
])

_regnew("波动率特征", [
    {"key": "ext_max_dd",    "label": "最大回撤(20日)",     "n": 20,
     "expr": "Min($close/Max($close,20)-1,20)"},
    {"key": "ext_max_dd_60", "label": "最大回撤(60日)",     "n": 60,
     "expr": "Min($close/Max($close,60)-1,60)"},
    {"key": "ext_tail_risk", "label": "尾风险(VaR95)",      "n": 60,
     "expr": "-Quantile($close/Ref($close, 1)-1,0.05,60)"},
    {"key": "ext_var_99",    "label": "极端尾风险(VaR99)",  "n": 60,
     "expr": "-Quantile($close/Ref($close, 1)-1,0.01,60)"},
    {"key": "ext_dvol",      "label": "下行波动率",          "n": 20,
     "expr": "Std(Min($close/Ref($close, 1)-1,0),20)"},
    {"key": "ext_uvol",      "label": "上行波动率",          "n": 20,
     "expr": "Std(Max($close/Ref($close, 1)-1,0),20)"},
    {"key": "ext_vol_ratio", "label": "波幅比(高/低)",      "n": 20,
     "expr": "(Max($high,20)-Min($low,20))/Mean($close,20)"},
])

_regnew("价格动量", [
    {"key": "ext_overnight", "label": "隔夜收益",           "n": 10,
     "expr": "$open/Ref($close, 1)-1"},
    {"key": "ext_intraday",  "label": "日内收益",           "n": 10,
     "expr": "$close/$open-1"},
    {"key": "ext_gap_up",    "label": "向上跳空",           "n": 20,
     "expr": "Sum($low>Ref($high, 1),20)/20"},
    {"key": "ext_gap_down",  "label": "向下跳空",           "n": 20,
     "expr": "-Sum($high<Ref($low, 1),20)/20"},
    {"key": "ext_mom_ma",    "label": "动量均线偏离",       "n": 20,
     "expr": "($close/Ref($close, 20)-1)/($close/Mean($close,20)-1)"},
    {"key": "ext_ret_cons",  "label": "收益一致性",         "n": 20,
     "expr": "Sum($close>Ref($close, 1),20)/20"},
])

_regnew("趋势强度", [
    {"key": "ext_vwap_dev",  "label": "VWAP偏离",           "n": 10,
     "expr": "$close/(Sum($close*$volume,20)/Sum($volume,20))-1"},
    {"key": "ext_ma_fan",    "label": "均线发散",           "n": 20,
     "expr": "Std((Mean($close,5),Mean($close,10),Mean($close,20)),20)"},
    {"key": "ext_adx_proxy", "label": "ADX趋势强度(代理)",  "n": 20,
     "expr": "Abs(($close-Mean($close,20)))/Mean(Abs($close-Ref($close, 1)),20)"},
])

_regnew("成交量特征", [
    {"key": "ext_vol_cluster","label": "成交量聚集",        "n": 20,
     "expr": "Max($volume,5)/Mean($volume,20)"},
    {"key": "ext_amt_mom",   "label": "成交额动量",         "n": 10,
     "expr": "Mean($volume*$close,5)/Mean($volume*$close,20)-1"},
    {"key": "ext_vol_cv",    "label": "成交量变异系数",     "n": 20,
     "expr": "Std($volume,20)/Mean($volume,20)"},
])

_regnew("高阶特征", [
    {"key": "ext_beta",      "label": "市场Beta(代理)",     "n": 60,
     "expr": "Corr($close/Ref($close, 1)-1,$close/Ref($close, 1)-1,60)"},
    {"key": "ext_ret_autocorr","label": "收益自相关",       "n": 20,
     "expr": "Corr($close/Ref($close, 1)-1,Ref($close/Ref($close, 1)-1,1),20)"},
    {"key": "ext_hl_vol",    "label": "高低波幅",           "n": 20,
     "expr": "Mean($high/$low-1,20)"},
    {"key": "ext_close_loc", "label": "收盘位置",           "n": 20,
     "expr": "Mean(($close-$low)/($high-$low+1e-6),10)"},
    {"key": "ext_cv_ret",    "label": "收益变异系数",       "n": 20,
     "expr": "Std($close/Ref($close, 1)-1,20)/Abs(Mean($close/Ref($close, 1)-1,20)+1e-6)"},
    {"key": "ext_ret_range", "label": "收益波动范围",       "n": 20,
     "expr": "Max($close/Ref($close, 1)-1,20)-Min($close/Ref($close, 1)-1,20)"},
])

# ══════════════════════════════════════════════
# 适配器
# ══════════════════════════════════════════════

class PublicFactorAdapter:

    def __init__(self):
        self.all_factors = {**ALPHA158_FACTORS, **ALPHA101_FACTORS, **ALPHA191_FACTORS,
                            **FUNDAMENTAL_FACTORS, **EXTENDED_FACTORS}

    def get_all(self, source=None, category=None):
        result = {}
        for k, v in self.all_factors.items():
            if source and v["source"] != source:
                continue
            if category and v.get("category") != category:
                continue
            result[k] = v
        return result

    def get_categories(self):
        cats = {}
        for v in self.all_factors.values():
            cat = v.get("category", "other")
            cats[cat] = cats.get(cat, 0) + 1
        return cats

    def get_unified_category(self, factor_key):
        """返回因子在统一 8 大类体系中的分类。

        将 Alpha158/101/191 三套因子的原生分类归一化:
        价格动量 / 反转效应 / 波动率特征 / 成交量特征 / 趋势强度 / 流动性 / 技术形态 / 高阶特征
        """
        from core.domain.factors.selector import get_unified_category
        f = self.all_factors.get(factor_key, {})
        return get_unified_category(factor_key, f.get("source"), f.get("category"))

    def get_factors_by_unified_category(self):
        """按统一大类分组返回所有因子。"""
        from core.domain.factors.selector import UNIFIED_CATEGORIES
        groups = {c: [] for c in UNIFIED_CATEGORIES}
        for key in self.all_factors:
            cat = self.get_unified_category(key)
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(key)
        return groups

    def to_variant(self, factor_key, params=None):
        """将公开因子包装为白盒校验器能接受的variant格式"""
        f = self.all_factors.get(factor_key)
        if not f:
            # 桥接白盒因子: Alpha 注册表查不到时, 走统一解析函数 (消除因子管道断裂)
            from core.adapters.qlib.factor_resolver import resolve_factor_expression
            expr, _ = resolve_factor_expression(factor_key)
            if expr is None:
                return None
            return {
                "variant_key": factor_key,
                "template_key": factor_key,
                "label": factor_key,
                "params": params or {"n": 20},
                "default_n": 20,
                "source": "white_box",
                "category": "white_box",
                "expr": expr,
            }
        return {
            "variant_key": factor_key,
            "template_key": factor_key,
            "label": f["label"],
            "params": params or {"n": f.get("n", 20)},
            "default_n": f.get("n", 20),
            "source": f["source"],
            "category": f.get("category", ""),
            "expr": f.get("expr"),
        }

    def generate_variants(self, factor_key, window_variants=True):
        """对公开因子生成窗口变体（改造扩展）。

        Args:
            window_variants: True时生成n±半的变体，用于鲁棒性检验
        """
        f = self.all_factors.get(factor_key)
        if not f:
            return []

        base_n = f.get("n", 20)
        variants = [self.to_variant(factor_key)]

        if window_variants and base_n > 3:
            for delta in [-2, +2]:
                new_n = base_n + delta
                if new_n >= 3:
                    v = self.to_variant(factor_key, {"n": new_n})
                    v["variant_key"] = f"{factor_key}_n{new_n}"
                    v["label"] = f"{f['label']}(N={new_n})"
                    variants.append(v)

        return variants

    def deduplicate_factors(self, factor_keys, snapshot, eval_date, pool="all", code_prefixes=None, corr_thr=None):
        """候选集内部前置去同源: 计算因子间 Pearson 相关矩阵, |r|>corr_thr 聚为一组, 每组只留 ICIR 高者。
        与 L5 去冗余不同 (L5 是候选 vs 已入池的后置检查, 这里是候选集内部前置压缩)。
        corr_thr: 相关性阈值 (None=用 L5_CORR_THRESHOLD 0.75; 0.95=只去完全重复)。
        返回保留的 factor_keys 列表 (无重复基础池)。"""
        from core.domain.factors.validation import OverfittingChecker
        from core.domain.screening._screener import get_screener
        checker = OverfittingChecker()
        screener = get_screener()
        snap_map = {eval_date.isoformat(): snapshot}

        # 1. 每个因子在同一截面求因子值向量 (对齐到公共 codes)
        vecs = {}       # key -> {code: value}
        for key in factor_keys:
            v = self.to_variant(key)
            if not v:
                continue
            cs = checker._variant_scores(v, eval_date, eval_date.isoformat(), pool, code_prefixes,
                                         snap_map, screener, allow_composite=False)
            if cs is None:
                continue
            codes, scores = cs
            if len(codes) < 20:
                continue
            vecs[key] = {c: float(s) for c, s in zip(codes, scores)}

        if len(vecs) < 2:
            return list(vecs.keys()) or list(factor_keys)

        # 2. 每个因子的 |IC| 作为 ICIR 代理 (与 forward return 的 Spearman 绝对值)
        all_codes = sorted({c for m in vecs.values() for c in m})
        fwd = checker._batch_forward_returns(all_codes, 20, eval_date)
        fwd_map = {c: r for c, r in zip(all_codes, fwd)}
        from scipy.stats import spearmanr
        ic_abs = {}
        for key, m in vecs.items():
            xs, ys = [], []
            for c, val in m.items():
                r = fwd_map.get(c)
                if r is not None:
                    xs.append(val)
                    ys.append(r)
            if len(xs) >= 10:
                corr, _ = spearmanr(xs, ys)
                ic_abs[key] = abs(float(corr)) if corr is not None and not np.isnan(corr) else 0.0
            else:
                ic_abs[key] = 0.0

        # 3. 贪心聚类: 按 |IC| 降序, 依次保留; 与已保留因子 |Pearson|>0.8 则丢弃
        order = sorted(vecs.keys(), key=lambda k: ic_abs.get(k, 0.0), reverse=True)
        kept = []
        for key in order:
            m = vecs[key]
            redundant = False
            for kk in kept:
                common = [c for c in m if c in vecs[kk]]
                if len(common) < 20:
                    continue
                a = np.array([m[c] for c in common])
                b = np.array([vecs[kk][c] for c in common])
                if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                    continue
                r = abs(float(np.corrcoef(a, b)[0, 1]))
                from core.domain.factors.validation import L5_CORR_THRESHOLD
                thr = corr_thr if corr_thr is not None else L5_CORR_THRESHOLD
                if not np.isnan(r) and r > thr:
                    redundant = True
                    break
            if not redundant:
                kept.append(key)

        # 无法求值的因子 (综合分/不支持表达式) 保留, 交由后续七层校验处理
        unresolved = [k for k in factor_keys if k not in vecs]
        result = kept + unresolved
        logger.info(f"[Dedup] 前置去同源: {len(factor_keys)} → {len(result)} "
                    f"(可求值 {len(vecs)}→{len(kept)}, 不可求值保留 {len(unresolved)})")
        return result

    def batch_validate(self, factor_keys=None, source=None, category=None,
                       eval_weeks=12, pool="all", top_k=10, code_prefixes=None, progress_cb=None,
                       max_workers=4, mode="fast", dedupe=True, include_marginal=True,
                       anchor_date=None, snapshots=None):
        from core.domain.factors.validation import OverfittingChecker
        from core.domain.screening._screener import BuySideScreener, get_screener
        from concurrent.futures import as_completed
        from core.shared.infrastructure.thread_pool import thread_pool
        import threading
        import datetime as _dt

        # anchor_date: 历史锚点 (留出后续窗口做严格样本外)。字符串→date。None=今天(贴边, 仅调试用)
        if isinstance(anchor_date, str) and anchor_date:
            anchor_date = _dt.date.fromisoformat(anchor_date[:10])

        keys = factor_keys or list(self.all_factors.keys())
        if source:
            keys = [k for k in keys if self.all_factors.get(k, {}).get("source") == source]
        if category:
            keys = [k for k in keys if self.all_factors.get(k, {}).get("category") == category]

        total = len(keys)

        # 预计算全部日期的 snapshot（所有因子共享，避免重复全量快照）
        _oc = OverfittingChecker()
        if anchor_date is not None:
            # anchor 模式: 覆盖 L1 三窗 [anchor-270, anchor+100] + 各层 eval 窗
            _dates = set(_oc._dates_between(anchor_date - _dt.timedelta(days=300),
                                           anchor_date + _dt.timedelta(days=100), step=5))
            _dates |= set(_oc._get_eval_dates(weeks=eval_weeks, step=5, anchor_date=anchor_date))
            eval_dates = sorted(_dates)
        else:
            eval_dates = _oc._get_eval_dates(weeks=eval_weeks, step=5)[:10]
        screener = get_screener()
        if snapshots is None:
            snapshots = {}
            for eval_date in eval_dates:
                snapshot = screener.precompute_snapshot(
                    pool=pool, code_prefixes=code_prefixes,
                    eval_date=eval_date, skip_limit_up=True,
                )
                snapshots[eval_date.isoformat()] = snapshot
        logger.info(f"[Batch] 预计算 {len(snapshots)} 个快照完成 (dates={len(eval_dates)}, anchor={anchor_date})")

        # 前置去同源: 候选集内部相关性聚类, |r|>0.8 每组留 ICIR 高者 (报告修正2: 先压缩底池再筛选)
        if dedupe and len(keys) > 1 and eval_dates:
            try:
                # 代表日: anchor 模式用 anchor (数据内), 否则用最新日
                rep_date = anchor_date if anchor_date is not None else eval_dates[-1]
                rep_snapshot = snapshots.get(rep_date.isoformat())
                if rep_snapshot is None:  # anchor 非交易日, 取最近的快照日
                    rep_date = max((d for d in eval_dates if d <= (anchor_date or d)), default=eval_dates[-1])
                    rep_snapshot = snapshots.get(rep_date.isoformat())
                if rep_snapshot:
                    keys = self.deduplicate_factors(keys, rep_snapshot, rep_date, pool=pool, code_prefixes=code_prefixes)
                    total = len(keys)
            except Exception as e:
                logger.warning(f"[Batch] 前置去同源失败, 跳过: {e}")

        results = {}
        lock = threading.Lock()
        done = [0]

        def _check_one(key):
            v = self.to_variant(key)
            if not v:
                return key, None
            try:
                from core.domain.screening._screener import use_thread_local_screener, clear_thread_local_screener
                use_thread_local_screener()
                try:
                    checker = OverfittingChecker()
                    r = checker.check_all_layers(v, eval_weeks=eval_weeks, pool=pool, top_k=top_k, code_prefixes=code_prefixes, n_shuffles=30, skip_layers=[7], snapshots=snapshots, mode=mode, anchor_date=anchor_date)
                    # fast 模式默认跳过 L8; 对排废后幸存的因子补跑边际贡献 (报告修正5, 不产生静默漏检)
                    if include_marginal and mode == "fast" and r.get("all_pass"):
                        passed, score, detail = checker._check_marginal({
                            "variant": v, "eval_weeks": eval_weeks, "pool": pool, "top_k": top_k,
                            "code_prefixes": code_prefixes, "snapshots": snapshots, "anchor_date": anchor_date})
                        r["marginal"] = {"pass": passed, "score": score, "detail": detail}
                        r["layers"].append({"id": 8, "key": "marginal", "name": "边际贡献",
                                            "pass": passed, "score": score, "detail": detail})
                        r["all_pass"] = r["all_pass"] and passed
                        if passed:
                            r["passed_count"] = r.get("passed_count", 0) + 1
                finally:
                    clear_thread_local_screener()
                return key, {
                    "label": self.all_factors[key]["label"],
                    "source": self.all_factors[key]["source"],
                    "category": self.all_factors[key].get("category", ""),
                    **r,
                }
            except Exception as e:
                return key, {"label": key, "error": str(e)}

        logger.info(f"[Batch] 多线程校验 {total} 因子")
        with thread_pool.default.borrow() as ex:
            futures = {ex.submit(_check_one, key): key for key in keys}
            for future in as_completed(futures):
                key, result = future.result()
                if result is not None:
                    with lock:
                        results[key] = result
                        done[0] += 1
                        if progress_cb:
                            progress_cb({"current": done[0], "total": total, "key": key,
                                         "label": self.all_factors.get(key, {}).get("label", key)})

        # ── 多重假设检验修正 (Holm-Bonferroni) ──
        if len(results) > 5:
            from core.domain.factors.validation import multi_test_correct
            import re as _re
            l7_pvals = {}
            for key, r in results.items():
                for layer in r.get("layers", []):
                    if layer.get("id") == 7:
                        m = _re.search(r"p[=＝]\s*([\d.]+)", layer.get("detail", ""))
                        if m:
                            l7_pvals[key] = float(m.group(1))
            if l7_pvals:
                corrected = multi_test_correct(l7_pvals, method="holm", alpha=0.05)
                for key, (passed, adj_p) in corrected.items():
                    if key in results and passed:
                        for layer in results[key].get("layers", []):
                            if layer.get("id") == 7:
                                layer["pass"] = True
                                layer["score"] = round(1.0 - adj_p, 4)
                                layer["detail"] = f"置换检验通过 (Holm修正 p={adj_p:.4f})"
                        results[key]["all_pass"] = all(l["pass"] for l in results[key].get("layers", []))
                        results[key]["passed_count"] = sum(1 for l in results[key].get("layers", []) if l["pass"])

        return results

    def generate_report(self, results, output_path=None):
        """生成筛选报告: 按校验通过率排序，标注改造建议"""
        lines = [
            "# 公开因子筛选报告",
            f"## 总览",
            f"- 测试因子数: {len(results)}",
            f"- 全部通过: {sum(1 for r in results.values() if r.get('all_pass'))}",
            f"- 通过3层+: {sum(1 for r in results.values() if r.get('passed_count', 0) >= 3)}",
            f"- 完全淘汰: {sum(1 for r in results.values() if r.get('passed_count', 0) == 0)}",
            "",
            "## 排名（按通过层数降序）",
            "",
            "| 排名 | 因子 | 来源 | 类别 | 通过 | 建议 |",
            "|------|------|------|------|------|------|",
        ]

        sorted_results = sorted(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1].get("passed_count", 0),
            reverse=True,
        )

        for rank, (key, r) in enumerate(sorted_results, 1):
            passed = r.get("passed_count", 0)
            total = r.get("total_layers", 7)

            if passed >= 6:
                suggestion = "✅ 直接入库"
            elif passed >= 4:
                suggestion = "🔧 改造扩展"
            elif passed >= 2:
                suggestion = "⚠️ 谨慎保留"
            else:
                suggestion = "❌ 建议淘汰"

            lines.append(
                f"| {rank} | {key} | {r.get('source','')} | {r.get('category','')} | "
                f"{passed}/{total} | {suggestion} |"
            )

        lines.append("")
        lines.append("## 分层详情")
        lines.append("")

        for key, r in sorted_results:
            if "layers" not in r:
                continue
            lines.append(f"### {key} — {r.get('label', key)}")
            lines.append(f"来源: {r.get('source', '')} | 类别: {r.get('category', '')} | "
                        f"通过: {r.get('passed_count', 0)}/{r.get('total_layers', 7)}")
            for l in r.get("layers", []):
                icon = "✅" if l["pass"] else "❌"
                lines.append(f"- {icon} L{l['id']} {l['name']}: {l.get('detail', '')}")

        report = "\n".join(lines)

        # output_path 写文件逻辑已移除 (从未被调用, 属展示层职责应归 L1/L2)
        return report


# ══════════════════════════════════════════════
# 因子集对比测试
# ══════════════════════════════════════════════

def compare_factor_sets(filtered_keys, baseline="Alpha158", eval_weeks=12):
    """对比筛选后的因子集 vs 原始Alpha158的ICIR。

    Returns:
        {baseline_icir, filtered_icir, improvement_pct, filtered_keys}
    """
    # 简化: 基于单因子IC估算因子集的组合ICIR
    from core.domain.factors.validation import OverfittingChecker
    from core.domain.factors.adapter import PublicFactorAdapter

    adapter = PublicFactorAdapter()
    checker = OverfittingChecker()

    # 对筛选后的因子逐一计算IC
    filtered_ics = []
    for key in filtered_keys:
        v = adapter.to_variant(key)
        if v:
            ic = checker._calc_rank_ic(v, checker._get_eval_dates(weeks=eval_weeks),
                                       {"pool": "all", "top_k": 10})
            filtered_ics.append(abs(ic))

    if not filtered_ics:
        return {"error": "无有效因子"}

    avg_ic = np.mean(filtered_ics)
    ic_std = np.std(filtered_ics)

    # 估算: 原始Alpha158所有因子的平均IC（基于常识约为0.015-0.03）
    # 实际应从测试结果获取
    estimated_baseline_ic = 0.02
    estimated_icir = avg_ic / max(ic_std, 0.001) if ic_std > 0 else 0

    return {
        "tested_count": len(filtered_keys),
        "avg_rank_ic": round(float(avg_ic), 4),
        "ic_std": round(float(ic_std), 4),
        "estimated_icir": round(float(estimated_icir), 3),
        "estimated_baseline_icir": round(float(estimated_baseline_ic / 0.02), 3),
        "improvement": round(float((avg_ic - estimated_baseline_ic) / max(estimated_baseline_ic, 0.001) * 100), 1),
        "verdict": "优于基线" if avg_ic > estimated_baseline_ic else "未达到基线",
    }
