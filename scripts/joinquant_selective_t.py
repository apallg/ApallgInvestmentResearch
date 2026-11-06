from jqdata import *
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
    g.enable_t = True
    g.target_total_ratio = 0.70
    g.max_t_ratio = 0.30
    g.target_trade_value = 20000
    g.min_trade_value = 15000
    g.max_round_trips = 1

    g.min_entry_deviation = 0.006
    g.atr_entry_multiple = 0.90
    g.take_profit_gross = 0.0075
    g.stop_loss_gross = 0.0045
    g.min_rsi = 65
    g.min_intraday_range = 0.010
    g.min_daily_range = 0.015
    g.min_peak_volume_ratio = 1.00
    g.market_up_limit = 0.008
    g.stock_up_limit = 0.012
    g.daily_5d_up_limit = 0.040
    g.daily_ma20_gap_limit = 0.060

    g.signal_interval_minutes = 5
    g.manage_start_time = (9, 35)
    g.open_time = (10, 0)
    g.last_open_time = (13, 45)
    g.force_close_time = (14, 45)

    g.states = {}
    g.daily_filters = {}
    g.cost_basis = {}
    g.total_t_profit = {}
    g.total_t_rounds = 0
    g.t_wins = 0
    g.t_losses = 0
    g.engine_bar_calls = 0
    g.trading_days = 0
    g.filter_counts = {
        'daily': 0,
        'bars': 0,
        'range': 0,
        'deviation': 0,
        'rsi': 0,
        'reversal': 0,
        'volume': 0,
        'market': 0,
    }
    g.reason_stats = {
        '止盈': {'rounds': 0, 'profit': 0.0},
        '止损': {'rounds': 0, 'profit': 0.0},
        '尾盘回补': {'rounds': 0, 'profit': 0.0},
        '隔夜回补': {'rounds': 0, 'profit': 0.0},
    }

    run_daily(build_initial_positions, time='09:40')
    run_daily(run_t_engine, time='every_bar', reference_security='000300.XSHG')


def before_trading_start(context):
    g.engine_bar_calls = 0
    g.daily_filters = {}
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        old_state = g.states.get(stock, {})
        start_closeable = int(position.closeable_amount // 100 * 100)
        g.states[stock] = {
            'start_closeable': start_closeable,
            'max_t_shares': int(start_closeable * g.max_t_ratio // 100 * 100),
            'round_trips': 0,
            'pending': old_state.get('pending'),
            'last_action_dt': None,
            'day_profit': 0.0,
        }
        g.daily_filters[stock] = get_daily_filter(stock)
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
            log.info('%s 初始建仓 %d 股，成交价 %.3f' % (
                stock, fill['amount'], fill['price']))


def run_t_engine(context):
    g.engine_bar_calls += 1
    now = context.current_dt
    clock = (now.hour, now.minute)
    if clock < g.manage_start_time or clock > g.force_close_time:
        return
    if now.minute % g.signal_interval_minutes != 0:
        return

    current = get_current_data()
    market_state = get_market_state()
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        state = g.states.get(stock)
        if state is None or position.total_amount <= 0:
            continue

        data = current[stock]
        price = float(data.last_price)
        if data.paused or data.is_st or not np.isfinite(price) or price <= 0:
            continue

        pending = state['pending']
        if pending is not None:
            overnight = pending['opened_at'].date() < now.date()
            force_close = clock >= g.force_close_time
            close_pending_cycle(
                context, stock, price, data, state, force_close, overnight)
            continue

        if not g.enable_t or clock < g.open_time or clock > g.last_open_time:
            continue
        if state['round_trips'] >= g.max_round_trips:
            continue
        if market_state == 'up':
            g.filter_counts['market'] += 1
            continue

        signal = get_stock_signal(stock, price, now)
        if signal is None:
            continue

        amount = get_trade_amount(price, state)
        if amount < 100 or position.closeable_amount < amount:
            continue

        fill = submit_order(stock, -amount)
        if fill is None:
            continue

        trade_value = fill['amount'] * fill['price']
        state['pending'] = {
            'amount': fill['amount'],
            'entry_price': fill['price'],
            'entry_fee_per_share': fill['commission'] / fill['amount'],
            'take_profit_edge': get_take_profit_edge(trade_value),
            'stop_loss_edge': g.stop_loss_gross,
            'realized_profit': 0.0,
            'opened_at': now,
        }
        state['last_action_dt'] = now
        log.info(
            '%s 高抛开T %d 股，成交价 %.3f，VWAP %.3f，目标价差 %.2f%%' % (
                stock,
                fill['amount'],
                fill['price'],
                signal['vwap'],
                state['pending']['take_profit_edge'] * 100,
            )
        )


def close_pending_cycle(context, stock, price, data, state, force_close, overnight):
    pending = state['pending']
    amount = int(pending['amount'] // 100 * 100)
    if amount < 100:
        state['pending'] = None
        return

    take_profit = price <= pending['entry_price'] * (1 - pending['take_profit_edge'])
    stop_loss = price >= pending['entry_price'] * (1 + pending['stop_loss_edge'])
    if not (take_profit or stop_loss or force_close or overnight):
        return
    if price >= data.high_limit:
        return

    needed_cash = amount * price * 1.002
    if context.portfolio.available_cash < needed_cash:
        return
    fill = submit_order(stock, amount)
    if fill is None:
        return

    entry_value = (
        pending['entry_price'] - pending['entry_fee_per_share']
    ) * fill['amount']
    exit_cost = fill['price'] * fill['amount'] + fill['commission']
    profit = entry_value - exit_cost
    pending['amount'] -= fill['amount']
    pending['realized_profit'] += profit
    state['day_profit'] += profit
    g.total_t_profit[stock] += profit
    state['last_action_dt'] = context.current_dt

    if take_profit:
        reason = '止盈'
    elif stop_loss:
        reason = '止损'
    elif overnight:
        reason = '隔夜回补'
    else:
        reason = '尾盘回补'

    if pending['amount'] < 100:
        cycle_profit = pending['realized_profit']
        state['pending'] = None
        state['round_trips'] += 1
        g.total_t_rounds += 1
        g.reason_stats[reason]['rounds'] += 1
        g.reason_stats[reason]['profit'] += cycle_profit
        if cycle_profit > 0:
            g.t_wins += 1
        else:
            g.t_losses += 1

    log.info('%s %s平T %d 股，成交价 %.3f，本轮收益 %.2f' % (
        stock, reason, fill['amount'], fill['price'], profit))


def get_stock_signal(stock, price, now):
    daily_filter = g.daily_filters.get(stock)
    if daily_filter is None or not daily_filter['allowed']:
        g.filter_counts['daily'] += 1
        return None

    bars = get_bars(
        stock,
        count=48,
        unit='5m',
        fields=['date', 'open', 'high', 'low', 'close', 'volume', 'money'],
        include_now=False,
        df=True,
    )
    if bars is None or len(bars) < 24:
        g.filter_counts['bars'] += 1
        return None
    bars = bars.dropna()
    if len(bars) < 24:
        g.filter_counts['bars'] += 1
        return None

    today = bars[pd.to_datetime(bars['date']).dt.date == now.date()]
    today = today[today['volume'] > 0]
    if len(today) < 6 or today['volume'].sum() <= 0:
        g.filter_counts['bars'] += 1
        return None

    intraday_range = float(today['high'].max() / today['low'].min() - 1)
    required_range = max(
        g.min_intraday_range,
        daily_filter['median_range'] * 0.55,
    )
    if intraday_range < required_range:
        g.filter_counts['range'] += 1
        return None

    closes = bars['close'].astype(float)
    opens = bars['open'].astype(float)
    highs = bars['high'].astype(float)
    lows = bars['low'].astype(float)
    volumes = bars['volume'].astype(float)
    previous = closes.shift(1)
    true_range = pd.concat([
        highs - lows,
        (highs - previous).abs(),
        (lows - previous).abs(),
    ], axis=1).max(axis=1)
    atr_pct = float(true_range.tail(14).mean() / price)
    entry_deviation = max(
        g.min_entry_deviation,
        min(0.012, atr_pct * g.atr_entry_multiple),
    )
    vwap = float(today['money'].sum() / today['volume'].sum())
    if price < vwap * (1 + entry_deviation * 0.75):
        g.filter_counts['deviation'] += 1
        return None

    delta = closes.diff()
    gain = delta.clip(lower=0).tail(7).mean()
    loss = -delta.clip(upper=0).tail(7).mean()
    rsi = 100.0 if loss <= 0 else float(100 - 100 / (1 + gain / loss))
    if rsi < g.min_rsi:
        g.filter_counts['rsi'] += 1
        return None

    trend_return = float(closes.iloc[-1] / closes.iloc[-20] - 1)
    peak_high = float(highs.iloc[-2])
    peak_range = max(peak_high - float(lows.iloc[-2]), price * 0.001)
    upper_wick = peak_high - max(float(opens.iloc[-2]), float(closes.iloc[-2]))
    upper_wick_ratio = upper_wick / peak_range
    local_peak = peak_high >= float(highs.iloc[-7:-1].max())
    turned_down = closes.iloc[-1] < closes.iloc[-2]
    if (
        trend_return >= g.stock_up_limit
        or not local_peak
        or not turned_down
        or upper_wick_ratio < 0.20
    ):
        g.filter_counts['reversal'] += 1
        return None

    recent_volume = float(volumes.iloc[-12:-2].median())
    peak_volume_ratio = (
        float(volumes.iloc[-2]) / recent_volume if recent_volume > 0 else 0.0
    )
    if peak_volume_ratio < g.min_peak_volume_ratio:
        g.filter_counts['volume'] += 1
        return None

    return {
        'vwap': vwap,
        'rsi': rsi,
        'trend_return': trend_return,
        'entry_deviation': entry_deviation,
        'peak_volume_ratio': peak_volume_ratio,
    }


def get_daily_filter(stock):
    bars = get_bars(
        stock,
        count=21,
        unit='1d',
        fields=['high', 'low', 'close'],
        include_now=False,
        df=True,
    )
    if bars is None or len(bars) < 21:
        return None
    bars = bars.dropna()
    if len(bars) < 21:
        return None

    closes = bars['close'].astype(float)
    previous = closes.shift(1)
    ranges = ((bars['high'] - bars['low']) / previous).dropna()
    median_range = float(ranges.tail(20).median())
    five_day_return = float(closes.iloc[-1] / closes.iloc[-6] - 1)
    ma20 = float(closes.tail(20).mean())
    ma20_gap = float(closes.iloc[-1] / ma20 - 1)
    allowed = (
        median_range >= g.min_daily_range
        and five_day_return < g.daily_5d_up_limit
        and ma20_gap < g.daily_ma20_gap_limit
    )
    return {
        'allowed': allowed,
        'median_range': median_range,
        'five_day_return': five_day_return,
        'ma20_gap': ma20_gap,
    }


def get_market_state():
    bars = get_bars(
        '000300.XSHG',
        count=20,
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
    return 'up' if trend_return >= g.market_up_limit else 'flat'


def get_trade_amount(price, state):
    target_shares = int(g.target_trade_value / price // 100 * 100)
    amount = int(min(state['max_t_shares'], target_shares) // 100 * 100)
    if amount < 100 or amount * price < g.min_trade_value:
        return 0
    return amount


def get_take_profit_edge(trade_value):
    buy_commission = max(trade_value * 0.0003, 5.0)
    sell_commission = max(trade_value * 0.0003, 5.0)
    stamp_duty = trade_value * 0.0005
    slippage = trade_value * 0.0002
    cost_rate = (
        buy_commission + sell_commission + stamp_duty + slippage
    ) / trade_value
    return max(g.take_profit_gross, cost_rate * 3.0)


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

    g.trading_days += 1
    day_profit = 0.0
    day_rounds = 0
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        state = g.states.get(stock, {})
        day_profit += float(state.get('day_profit', 0.0))
        day_rounds += int(state.get('round_trips', 0))
        if position.total_amount <= 0:
            continue

        basis = g.cost_basis.get(stock, position.avg_cost * position.total_amount)
        effective_cost = basis / position.total_amount
        pending = state.get('pending')
        pending_text = '无' if pending is None else '%d股' % pending['amount']
        log.info(
            '%s 收盘持仓 %d 股，经济成本 %.3f，累计做T收益 %.2f，未闭合循环 %s' % (
                stock,
                position.total_amount,
                effective_cost,
                g.total_t_profit.get(stock, 0.0),
                pending_text,
            )
        )

    total_rounds = int(g.t_wins) + int(g.t_losses)
    win_rate = (
        float(g.t_wins) / float(total_rounds) if total_rounds else 0.0
    )
    total_profit = 0.0
    for value in g.total_t_profit.values():
        total_profit += float(value)

    record(
        t_day_profit=float(day_profit),
        t_cum_profit=float(total_profit),
        t_day_rounds=float(day_rounds),
        t_win_rate=float(win_rate),
        t_take_profit=float(g.reason_stats['止盈']['profit']),
        t_stop_profit=float(g.reason_stats['止损']['profit']),
        t_force_profit=float(g.reason_stats['尾盘回补']['profit']),
        t_overnight_profit=float(g.reason_stats['隔夜回补']['profit']),
    )
    log.info(
        '做T汇总：完成 %d 轮，盈利 %d，亏损 %d，胜率 %.2f%%，累计收益 %.2f' % (
            g.total_t_rounds,
            g.t_wins,
            g.t_losses,
            win_rate * 100,
            total_profit,
        )
    )

    if g.trading_days % 20 == 0:
        log.info(
            '平仓归因：止盈 %d轮/%.2f，止损 %d轮/%.2f，尾盘 %d轮/%.2f，隔夜 %d轮/%.2f' % (
                g.reason_stats['止盈']['rounds'],
                g.reason_stats['止盈']['profit'],
                g.reason_stats['止损']['rounds'],
                g.reason_stats['止损']['profit'],
                g.reason_stats['尾盘回补']['rounds'],
                g.reason_stats['尾盘回补']['profit'],
                g.reason_stats['隔夜回补']['rounds'],
                g.reason_stats['隔夜回补']['profit'],
            )
        )
        log.info(
            '信号过滤：日线%d，数据%d，振幅%d，偏离%d，RSI%d，反转%d，量能%d，大盘%d' % (
                g.filter_counts['daily'],
                g.filter_counts['bars'],
                g.filter_counts['range'],
                g.filter_counts['deviation'],
                g.filter_counts['rsi'],
                g.filter_counts['reversal'],
                g.filter_counts['volume'],
                g.filter_counts['market'],
            )
        )
