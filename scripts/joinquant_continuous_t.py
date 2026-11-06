from jqdata import *
import math
import numpy as np
import pandas as pd


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_order_cost(
        OrderCost(
            close_tax=0.0005,
            open_commission=0.0003,
            close_commission=0.0003,
            min_commission=5,
        ),
        type='stock',
    )
    set_slippage(PriceRelatedSlippage(0.0001), type='stock')

    g.target_stocks = ['000001.XSHE']
    g.auto_build = True
    g.target_total_ratio = 0.70
    g.max_t_ratio = 0.30
    g.max_round_ratio = 0.15
    g.max_round_trips = 2
    g.min_trade_value = 5000
    g.min_edge = 0.006
    g.atr_edge_multiple = 0.65
    g.take_profit_buffer = 0.004
    g.min_net_profit_rate = 0.0025
    g.stop_loss = 0.012
    g.signal_interval_minutes = 5
    g.open_time = (9, 45)
    g.last_open_time = (14, 15)
    g.force_close_time = (14, 45)
    g.market_trend_limit = 0.012
    g.stock_trend_limit = 0.030

    g.states = {}
    g.cost_basis = {}
    g.total_t_profit = {}
    g.total_t_rounds = 0
    g.t_wins = 0
    g.t_losses = 0
    g.engine_bar_calls = 0

    run_daily(build_initial_positions, time='09:40')
    run_daily(run_t_engine, time='every_bar', reference_security='000300.XSHG')


def before_trading_start(context):
    g.engine_bar_calls = 0
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        old_state = g.states.get(stock, {})
        pending = old_state.get('pending')
        start_closeable = int(position.closeable_amount // 100 * 100)
        max_t_shares = int(start_closeable * g.max_t_ratio // 100 * 100)
        g.states[stock] = {
            'start_closeable': start_closeable,
            'max_t_shares': max_t_shares,
            'sell_used': 0,
            'round_trips': 0,
            'pending': pending,
            'last_action_dt': None,
            'day_profit': 0.0,
        }
        if position.total_amount > 0 and stock not in g.cost_basis:
            g.cost_basis[stock] = float(position.avg_cost * position.total_amount)
        g.total_t_profit.setdefault(stock, 0.0)


def build_initial_positions(context):
    if not g.auto_build or not g.target_stocks:
        return

    current = get_current_data()
    target_value = context.portfolio.total_value * g.target_total_ratio / len(g.target_stocks)
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        data = current[stock]
        price = float(data.last_price)
        if position.total_amount > 0:
            continue
        if data.paused or data.is_st or not np.isfinite(price) or price <= 0:
            continue
        if price >= data.high_limit or price <= data.low_limit:
            continue

        amount = int(target_value / price // 100 * 100)
        if amount < 100:
            continue
        fill = submit_order(stock, amount)
        if fill is not None:
            log.info('%s 初始建仓 %d 股，成交价 %.3f' % (stock, fill['amount'], fill['price']))


def run_t_engine(context):
    g.engine_bar_calls += 1
    now = context.current_dt
    clock = (now.hour, now.minute)
    if clock < g.open_time or clock > g.force_close_time:
        return
    if now.minute % g.signal_interval_minutes != 0:
        return

    market_state = get_market_state()
    current = get_current_data()
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        state = g.states.get(stock)
        if state is None or position.total_amount <= 0:
            continue

        data = current[stock]
        price = float(data.last_price)
        if data.paused or data.is_st or not np.isfinite(price) or price <= 0:
            continue

        force_close = clock >= g.force_close_time
        pending = state['pending']
        if pending is not None:
            close_pending_cycle(context, stock, price, data, state, force_close)
            continue

        if clock > g.last_open_time:
            continue
        if state['round_trips'] >= g.max_round_trips:
            continue
        if state['last_action_dt'] is not None:
            elapsed = (now - state['last_action_dt']).total_seconds()
            if elapsed < g.signal_interval_minutes * 60:
                continue

        signal = get_stock_signal(stock, price, now)
        if signal is None:
            continue

        amount = get_trade_amount(stock, price, state)
        if amount < 100:
            continue

        if signal['side'] == 'buy_low':
            if market_state == 'down' or signal['trend_return'] <= -g.stock_trend_limit:
                continue
            needed_cash = amount * price * 1.002
            if context.portfolio.available_cash < needed_cash:
                continue
            if position.closeable_amount < amount:
                continue
            fill = submit_order(stock, amount)
            if fill is None:
                continue
            state['pending'] = {
                'kind': 'buy_low',
                'amount': fill['amount'],
                'entry_price': fill['price'],
                'entry_fee_per_share': fill['commission'] / fill['amount'],
                'edge': signal['edge'],
                'close_edge': get_close_edge(fill['amount'] * fill['price'], signal['edge']),
                'realized_profit': 0.0,
                'opened_at': now,
            }
            state['last_action_dt'] = now
            log.info('%s 低吸开T %d 股，成交价 %.3f，VWAP %.3f' % (
                stock, fill['amount'], fill['price'], signal['vwap']))
        else:
            if market_state == 'up' or signal['trend_return'] >= g.stock_trend_limit:
                continue
            if position.closeable_amount < amount:
                continue
            fill = submit_order(stock, -amount)
            if fill is None:
                continue
            state['sell_used'] += fill['amount']
            state['pending'] = {
                'kind': 'sell_high',
                'amount': fill['amount'],
                'entry_price': fill['price'],
                'entry_fee_per_share': fill['commission'] / fill['amount'],
                'edge': signal['edge'],
                'close_edge': get_close_edge(fill['amount'] * fill['price'], signal['edge']),
                'realized_profit': 0.0,
                'opened_at': now,
            }
            state['last_action_dt'] = now
            log.info('%s 高抛开T %d 股，成交价 %.3f，VWAP %.3f' % (
                stock, fill['amount'], fill['price'], signal['vwap']))


def close_pending_cycle(context, stock, price, data, state, force_close):
    pending = state['pending']
    amount = int(pending['amount'] // 100 * 100)
    if amount < 100:
        state['pending'] = None
        return

    edge = pending.get('close_edge', max(g.take_profit_buffer, pending['edge'] * 0.45))
    if pending['kind'] == 'buy_low':
        take_profit = price >= pending['entry_price'] * (1 + edge)
        stop_loss = price <= pending['entry_price'] * (1 - g.stop_loss)
        if not (take_profit or stop_loss or force_close):
            return
        if price <= data.low_limit or context.portfolio.positions[stock].closeable_amount < amount:
            return
        fill = submit_order(stock, -amount)
        if fill is None:
            return
        state['sell_used'] += fill['amount']
        entry_cost = (pending['entry_price'] + pending['entry_fee_per_share']) * fill['amount']
        exit_value = fill['price'] * fill['amount'] - fill['commission']
        profit = exit_value - entry_cost
    else:
        take_profit = price <= pending['entry_price'] * (1 - edge)
        stop_loss = price >= pending['entry_price'] * (1 + g.stop_loss)
        if not (take_profit or stop_loss or force_close):
            return
        if price >= data.high_limit:
            return
        needed_cash = amount * price * 1.002
        if context.portfolio.available_cash < needed_cash:
            return
        fill = submit_order(stock, amount)
        if fill is None:
            return
        entry_value = (pending['entry_price'] - pending['entry_fee_per_share']) * fill['amount']
        exit_cost = fill['price'] * fill['amount'] + fill['commission']
        profit = entry_value - exit_cost

    pending['amount'] -= fill['amount']
    pending['realized_profit'] = pending.get('realized_profit', 0.0) + profit
    state['day_profit'] += profit
    g.total_t_profit[stock] += profit
    state['last_action_dt'] = context.current_dt
    if pending['amount'] < 100:
        cycle_profit = pending['realized_profit']
        state['pending'] = None
        state['round_trips'] += 1
        g.total_t_rounds += 1
        if cycle_profit > 0:
            g.t_wins += 1
        else:
            g.t_losses += 1

    reason = '止盈' if take_profit else ('止损' if stop_loss else '尾盘回补')
    log.info('%s %s平T %d 股，成交价 %.3f，本轮收益 %.2f' % (
        stock, reason, fill['amount'], fill['price'], profit))


def get_stock_signal(stock, price, now):
    bars = get_bars(
        stock,
        count=42,
        unit='5m',
        fields=['date', 'high', 'low', 'close', 'volume', 'money'],
        include_now=False,
        df=True,
    )
    if bars is None or len(bars) < 24:
        return None

    bars = bars.dropna()
    if len(bars) < 24:
        return None
    today = bars[pd.to_datetime(bars['date']).dt.date == now.date()]
    today = today[today['volume'] > 0]
    if len(today) < 3 or today['volume'].sum() <= 0:
        return None

    closes = bars['close'].astype(float)
    highs = bars['high'].astype(float)
    lows = bars['low'].astype(float)
    previous = closes.shift(1)
    true_range = pd.concat([
        highs - lows,
        (highs - previous).abs(),
        (lows - previous).abs(),
    ], axis=1).max(axis=1)
    atr_pct = float(true_range.tail(14).mean() / price)
    edge = max(g.min_edge, atr_pct * g.atr_edge_multiple)

    delta = closes.diff()
    gain = delta.clip(lower=0).tail(7).mean()
    loss = -delta.clip(upper=0).tail(7).mean()
    if loss <= 0:
        rsi = 100.0
    else:
        rsi = float(100 - 100 / (1 + gain / loss))

    vwap = float(today['money'].sum() / today['volume'].sum())
    ema = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    fair_price = 0.60 * vwap + 0.40 * ema
    trend_return = float(closes.iloc[-1] / closes.iloc[-20] - 1)
    turned_up = closes.iloc[-1] > closes.iloc[-2] and closes.iloc[-2] <= closes.iloc[-3]
    turned_down = closes.iloc[-1] < closes.iloc[-2] and closes.iloc[-2] >= closes.iloc[-3]

    if price <= fair_price * (1 - edge) and rsi <= 38 and turned_up:
        side = 'buy_low'
    elif price >= fair_price * (1 + edge) and rsi >= 62 and turned_down:
        side = 'sell_high'
    else:
        return None

    return {
        'side': side,
        'edge': edge,
        'vwap': vwap,
        'rsi': rsi,
        'trend_return': trend_return,
    }


def get_market_state():
    bars = get_bars(
        '000300.XSHG',
        count=24,
        unit='5m',
        fields=['close'],
        include_now=False,
        df=True,
    )
    if bars is None or len(bars) < 20:
        return 'flat'
    closes = bars['close'].dropna().astype(float)
    if len(closes) < 20:
        return 'flat'
    trend_return = float(closes.iloc[-1] / closes.iloc[-20] - 1)
    if trend_return >= g.market_trend_limit:
        return 'up'
    if trend_return <= -g.market_trend_limit:
        return 'down'
    return 'flat'


def get_trade_amount(stock, price, state):
    remaining = state['max_t_shares'] - state['sell_used']
    per_round_cap = int(state['start_closeable'] * g.max_round_ratio // 100 * 100)
    upper = int(min(remaining, per_round_cap) // 100 * 100)
    minimum = int(math.ceil(g.min_trade_value / price / 100.0) * 100)
    if upper < minimum:
        return 0
    return upper


def get_close_edge(trade_value, signal_edge):
    buy_commission = max(trade_value * 0.0003, 5.0)
    sell_commission = max(trade_value * 0.0003, 5.0)
    stamp_duty = trade_value * 0.0005
    slippage = trade_value * 0.0002
    cost_rate = (buy_commission + sell_commission + stamp_duty + slippage) / trade_value
    return max(g.take_profit_buffer, signal_edge * 0.45, cost_rate + g.min_net_profit_rate)


def submit_order(stock, amount):
    order_result = order(stock, int(amount))
    if order_result is None or order_result.filled <= 0:
        return None

    filled = int(order_result.filled)
    price = float(order_result.price)
    commission = float(order_result.commission or 0.0)
    g.cost_basis.setdefault(stock, 0.0)
    if amount > 0:
        g.cost_basis[stock] += price * filled + commission
    else:
        g.cost_basis[stock] -= price * filled - commission
    return {
        'amount': filled,
        'price': price,
        'commission': commission,
    }


def after_trading_end(context):
    if g.engine_bar_calls < 100:
        raise RuntimeError(
            '本策略必须选择分钟回测；当前每个交易日仅触发 %d 次 every_bar' %
            g.engine_bar_calls
        )
    day_profit = 0.0
    day_rounds = 0
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        state = g.states.get(stock, {})
        day_profit += state.get('day_profit', 0.0)
        day_rounds += state.get('round_trips', 0)
        if position.total_amount <= 0:
            continue
        basis = g.cost_basis.get(stock, position.avg_cost * position.total_amount)
        effective_cost = basis / position.total_amount
        pending = state.get('pending')
        pending_text = '无' if pending is None else '%s %d股' % (
            pending['kind'], pending['amount'])
        log.info(
            '%s 收盘持仓 %d 股，经济成本 %.3f，累计做T收益 %.2f，未闭合循环 %s' % (
                stock,
                position.total_amount,
                effective_cost,
                g.total_t_profit.get(stock, 0.0),
                pending_text,
            )
        )
    total_rounds = g.t_wins + g.t_losses
    win_rate = float(g.t_wins) / float(total_rounds) if total_rounds else 0.0
    total_profit = 0.0
    for value in g.total_t_profit.values():
        total_profit += float(value)
    record(
        t_day_profit=float(day_profit),
        t_cum_profit=float(total_profit),
        t_day_rounds=float(day_rounds),
        t_win_rate=float(win_rate),
    )
    log.info('做T汇总：完成 %d 轮，盈利 %d，亏损 %d，胜率 %.2f%%，累计收益 %.2f' % (
        g.total_t_rounds,
        g.t_wins,
        g.t_losses,
        win_rate * 100,
        total_profit,
    ))
