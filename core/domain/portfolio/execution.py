"""
日内执行决策 — 把调仓信号翻译成盘中动作 + 挂单价 + 理由。

职责: 输入调仓信号(side/urgency) + 盘中实时状态(现价/VWAP/评分/风险/时段),
     输出"现在该怎么做" (挂单/观望/市价/兜底) + 具体挂单价 + 理由。
不负责: 选股、下单、取数、涨跌停判断 (那是 broker/回测引擎的事)。

纯 L3 函数——不调 DB/HTTP/文件/时间, 所有状态显式传入。

核心纪律:
  1. 早盘优先保证价格 (不追高), 尾盘优先保证成交 (兜底)。
  2. 买入永远不高于 VWAP (除非尾盘兜底)。
  3. 止损必须确定性成交 (市价), 止盈可以等好价 (限价)。
  4. 日内风险触发 (高开低走/放量下跌/触跌停) 否决买入, 不管时段。
"""
from dataclasses import dataclass
from typing import Optional

# ── 时段 ──
PHASE_OPEN = "open"    # 早盘 9:30-10:00
PHASE_MID = "mid"      # 盘中 10:00-14:30
PHASE_CLOSE = "close"  # 尾盘 14:30-15:00
PHASE_CLOSED = "closed"

# ── 动作 ──
ACTION_LIMIT = "limit"    # 挂限价单
ACTION_WAIT = "wait"      # 观望, 不动作
ACTION_MARKET = "market"  # 市价单 (止损用)
ACTION_FILL = "fill"      # 尾盘兜底, 略越价确保成交

HIGH_OPEN_THRESHOLD = 0.02   # 高开阈值 (>2% 视为高开)
FILL_PREMIUM = 0.005         # 兜底溢价 (买入挂高/卖出挂低 0.5% 确保成交)
LOW_SCORE_DISCOUNT = 0.003   # 评分弱时 VWAP 下浮 0.3% 才买
SCORE_GOOD = 50
SCORE_WATCH = 30


@dataclass
class ExecutionDecision:
    action: str = ACTION_WAIT
    limit_price: Optional[float] = None
    reason: str = ""


def resolve_phase(time_str):
    """按 HH:MM 解析时段。纯函数, 供 L2 用当前时间转换。"""
    try:
        hh, mm = str(time_str).split(":")
        h, m = int(hh), int(mm)
    except (ValueError, AttributeError):
        return PHASE_CLOSED
    if h < 0 or h > 23 or m < 0 or m > 59:
        return PHASE_CLOSED
    t = h * 60 + m
    if 9 * 60 + 30 <= t < 10 * 60:
        return PHASE_OPEN
    if 14 * 60 + 30 <= t < 15 * 60:
        return PHASE_CLOSE
    if 10 * 60 <= t <= 11 * 60 + 30 or 13 * 60 <= t < 14 * 60 + 30:
        return PHASE_MID
    return PHASE_CLOSED


def decide_execution(side, urgency="normal", phase=PHASE_MID,
                     current_price=0.0, vwap=None, prev_close=0.0,
                     open_price=None, entry_score=None, risk_warning=None,
                     support_price=None, must_fill=True):
    """日内执行决策。

    Args:
        side: "buy"/"add"/"sell"/"reduce"
        urgency: "normal"/"stop_loss"/"take_profit"
        phase: PHASE_OPEN/PHASE_MID/PHASE_CLOSE/PHASE_CLOSED
        current_price: 现价
        vwap: 当日 VWAP (None 则退化为现价)
        prev_close: 前收盘价 (判高开用)
        open_price: 今开价 (判高开用)
        entry_score: 日内入场评分 0-100 (None 默认按"一般"处理)
        risk_warning: 风险提示 (非空则否决买入)
        support_price: 日内支撑位 (可选)
        must_fill: 是否必成交 (尾盘兜底判断用)

    Returns:
        ExecutionDecision(action, limit_price, reason)
    """
    if phase == PHASE_CLOSED:
        return ExecutionDecision(ACTION_WAIT, None, "非交易时段，不具备执行条件")
    if not current_price or current_price <= 0:
        return ExecutionDecision(ACTION_WAIT, None, "无有效现价, 无法决策")

    is_sell = side in ("sell", "reduce")

    # ── 卖出侧 ──
    if is_sell:
        if urgency == "stop_loss":
            return ExecutionDecision(ACTION_MARKET, None, "止损要确定性, 市价出, 不挂限价")
        if phase == PHASE_CLOSE and must_fill:
            price = current_price * (1 - FILL_PREMIUM)
            return ExecutionDecision(ACTION_FILL, round(price, 3),
                                     "尾盘兜底卖出, 挂低0.5%确保当日成交")
        if urgency == "take_profit":
            price = current_price * 1.002
            return ExecutionDecision(ACTION_LIMIT, round(price, 3), "止盈挂限价, 等冲高不追")
        anchor = vwap if vwap and vwap > 0 else current_price
        return ExecutionDecision(ACTION_LIMIT, round(anchor, 3), "减仓挂限价于VWAP")

    # ── 买入侧 ──
    if risk_warning:
        return ExecutionDecision(ACTION_WAIT, None, f"日内风险触发({risk_warning}), 放弃今日买入")

    anchor = vwap if vwap and vwap > 0 else current_price

    if phase == PHASE_OPEN:
        if open_price and prev_close and open_price > prev_close * (1 + HIGH_OPEN_THRESHOLD):
            return ExecutionDecision(ACTION_LIMIT, round(prev_close, 3),
                                     "高开不追, 挂前收价等回落")
        base = support_price or prev_close or anchor
        price = min(base, anchor) if base and anchor else base
        return ExecutionDecision(ACTION_LIMIT, round(price, 3), "早盘挂支撑位限价, 不追高")

    if phase == PHASE_CLOSE:
        if must_fill:
            price = current_price * (1 + FILL_PREMIUM)
            return ExecutionDecision(ACTION_FILL, round(price, 3),
                                     "尾盘兜底买入, 挂高0.5%确保当日成交")
        return ExecutionDecision(ACTION_WAIT, None, "尾盘非必成交单, 放弃")

    # 盘中
    score = entry_score if entry_score is not None else SCORE_GOOD
    if score >= SCORE_GOOD:
        price = anchor
        reason = "评分好, VWAP平价挂单"
    elif score >= SCORE_WATCH:
        price = min(anchor, support_price) if support_price and support_price > 0 else anchor
        reason = "评分一般, 挂VWAP与支撑的较低者"
    else:
        price = anchor * (1 - LOW_SCORE_DISCOUNT)
        reason = "评分弱, 挂VWAP下方0.3%坚持好价"
    return ExecutionDecision(ACTION_LIMIT, round(price, 3), reason)
