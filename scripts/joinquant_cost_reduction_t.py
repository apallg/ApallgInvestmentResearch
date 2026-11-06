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
    g.target_gross_edge = 0.0075
    g.min_net_profit_rate = 0.0020
    g.min_cycle_profit = 30.0
    g.confirmation_rebound_allowance = 0.0020
    g.slippage_rate = 0.0001
    g.min_rsi = 65
    g.min_intraday_range = 0.010
    g.min_daily_range = 0.015
    g.min_peak_volume_ratio = 1.00
    g.min_high_position = 0.78
    g.min_peak_upswing = 0.004
    g.min_upper_wick_ratio = 0.20
    g.min_lower_wick_ratio = 0.15
    g.market_up_limit = 0.008
    g.stock_up_limit = 0.012
    g.daily_5d_up_limit = 0.040
    g.daily_ma20_gap_limit = 0.060

    g.signal_interval_minutes = 5
    g.manage_start_time = (9, 35)
    g.open_time = (10, 0)
    g.last_open_time = (13, 45)
    g.manage_end_time = (14, 55)

    g.states = {}
    g.daily_filters = {}
    g.core_shares = {}
    g.core_initial_cost = {}
    g.total_t_profit = {}
    g.total_t_rounds = 0
    g.t_wins = 0
    g.t_losses = 0
    g.floor_touches = 0
    g.low_confirmed = 0
    g.engine_bar_calls = 0
    g.trading_days = 0
    g.filter_counts = {
        'daily': 0,
        'bars': 0,
        'range': 0,
        'deviation': 0,
        'rsi': 0,
        'location': 0,
        'reversal': 0,
        'volume': 0,
        'market': 0,
        'rebuy_bars': 0,
        'rebuy_bottom': 0,
        'rebuy_profit': 0,
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
        if position.total_amount > 0 and stock not in g.core_shares:
            g.core_shares[stock] = int(position.total_amount)
            g.core_initial_cost[stock] = float(position.avg_cost)
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
        if fill is None:
            continue
        g.core_shares[stock] = fill['amount']
        g.core_initial_cost[stock] = (
            fill['price'] * fill['amount'] + fill['commission']
        ) / fill['amount']
        log.info('%s 初始建仓 %d 股，成交价 %.3f' % (
            stock, fill['amount'], fill['price']))


def run_t_engine(context):
    g.engine_bar_calls += 1
    now = context.current_dt
    clock = (now.hour, now.minute)
    if clock < g.manage_start_time or clock > g.manage_end_time:
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

        if state['pending'] is not None:
            close_pending_cycle(context, stock, price, data, state)
            continue

        if not g.enable_t or clock < g.open_time or clock > g.last_open_time:
            continue
        if state['round_trips'] >= g.max_round_trips:
            continue
        if market_state == 'up':
            g.filter_counts['market'] += 1
            continue
        if price <= data.low_limit:
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

        sell_value = fill['amount'] * fill['price']
        target_edge = get_target_edge(sell_value, fill['commission'])
        state['pending'] = {
            'amount': fill['amount'],
            'entry_price': fill['price'],
            'sell_value': sell_value,
            'net_sell_proceeds': sell_value - fill['commission'],
            'realized_profit': 0.0,
            'target_edge': target_edge,
            'target_price': fill['price'] * (1 - target_edge),
            'target_touched': False,
            'lowest_price': fill['price'],
            'opened_at': now,
        }
        state['last_action_dt'] = now
        log.info(
            '%s 高抛开T %d股，成交价%.3f，目标触发价%.3f，VWAP %.3f，高位%.1f%%' % (
                stock,
                fill['amount'],
                fill['price'],
                state['pending']['target_price'],
                signal['vwap'],
                signal['high_position'] * 100,
            )
        )


def close_pending_cycle(context, stock, price, data, state):
    pending = state['pending']
    amount = int(pending['amount'] // 100 * 100)
    if amount < 100:
        state['pending'] = None
        return
    if price >= data.high_limit:
        return

    rebuy_signal = get_rebuy_signal(stock, price, pending)
    if rebuy_signal is None:
        return

    needed_cash = amount * price * 1.002
    if context.portfolio.available_cash < needed_cash:
        return

    fill = submit_order(stock, amount)
    if fill is None:
        return

    remaining_before = pending['amount']
    filled_ratio = float(fill['amount']) / float(remaining_before)
    sell_proceeds = pending['net_sell_proceeds'] * filled_ratio
    allocated_sell_value = pending['sell_value'] * filled_ratio
    buy_cost = fill['price'] * fill['amount'] + fill['commission']
    profit = sell_proceeds - buy_cost
    pending['amount'] -= fill['amount']
    pending['net_sell_proceeds'] -= sell_proceeds
    pending['sell_value'] -= allocated_sell_value
    pending['realized_profit'] += profit
    state['day_profit'] += profit
    g.total_t_profit[stock] += profit
    state['last_action_dt'] = context.current_dt

    if pending['amount'] < 100:
        cycle_profit = pending['realized_profit']
        if cycle_profit > 0:
            g.t_wins += 1
        else:
            g.t_losses += 1
            log.error('%s 回补成交后未降低成本，本轮收益 %.2f' % (stock, cycle_profit))
        state['pending'] = None
        state['round_trips'] += 1
        g.total_t_rounds += 1
        g.low_confirmed += 1

    reduction = profit / fill['amount']
    log.info(
        '%s 低点确认回补 %d股，成交价%.3f，本轮收益%.2f，每股降本%.4f，低点%.3f' % (
            stock,
            fill['amount'],
            fill['price'],
            profit,
            reduction,
            pending['lowest_price'],
        )
    )


def get_rebuy_signal(stock, price, pending):
    bars = get_bars(
        stock,
        count=24,
        unit='5m',
        fields=['date', 'open', 'high', 'low', 'close', 'volume'],
        include_now=False,
        df=True,
    )
    if bars is None or len(bars) < 8:
        g.filter_counts['rebuy_bars'] += 1
        return None
    bars = bars.dropna()
    if len(bars) < 8:
        g.filter_counts['rebuy_bars'] += 1
        return None

    bar_times = pd.to_datetime(bars['date'])
    after_entry = bars[bar_times > pending['opened_at']]
    if len(after_entry) > 0:
        observed_low = float(after_entry['low'].min())
        pending['lowest_price'] = min(pending['lowest_price'], observed_low, price)
        if observed_low <= pending['target_price'] or price <= pending['target_price']:
            if not pending['target_touched']:
                g.floor_touches += 1
            pending['target_touched'] = True
    if not pending['target_touched']:
        return None

    closes = bars['close'].astype(float)
    opens = bars['open'].astype(float)
    highs = bars['high'].astype(float)
    lows = bars['low'].astype(float)
    bottom_time = bar_times.iloc[-2].to_pydatetime()
    if bottom_time <= pending['opened_at']:
        g.filter_counts['rebuy_bars'] += 1
        return None

    bottom_low = float(lows.iloc[-2])
    bottom_range = max(
        float(highs.iloc[-2]) - bottom_low,
        pending['entry_price'] * 0.001,
    )
    lower_wick = min(float(opens.iloc[-2]), float(closes.iloc[-2])) - bottom_low
    lower_wick_ratio = lower_wick / bottom_range
    local_bottom = bottom_low <= float(lows.iloc[-7:-1].min())
    turned_up = (
        closes.iloc[-1] > closes.iloc[-2]
        and closes.iloc[-1] >= opens.iloc[-1]
    )
    reclaimed_mean = closes.iloc[-1] >= closes.iloc[-3:].mean()
    if (
        not local_bottom
        or not turned_up
        or not reclaimed_mean
        or lower_wick_ratio < g.min_lower_wick_ratio
    ):
        g.filter_counts['rebuy_bottom'] += 1
        return None

    expected_profit = get_expected_cycle_profit(
        price,
        amount=pending['amount'],
        pending=pending,
    )
    required_profit = max(
        g.min_cycle_profit,
        pending['sell_value'] * g.min_net_profit_rate,
    )
    if expected_profit < required_profit:
        g.filter_counts['rebuy_profit'] += 1
        return None

    return {
        'expected_profit': expected_profit,
        'lower_wick_ratio': lower_wick_ratio,
        'bottom_low': bottom_low,
    }


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

    today_high = float(today['high'].max())
    today_low = float(today['low'].min())
    intraday_range = float(today_high / today_low - 1)
    required_range = max(
        g.min_intraday_range,
        daily_filter['median_range'] * 0.55,
    )
    if intraday_range < required_range:
        g.filter_counts['range'] += 1
        return None

    high_position = (price - today_low) / max(today_high - today_low, price * 0.001)
    if high_position < g.min_high_position:
        g.filter_counts['location'] += 1
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
    prior_high = float(highs.iloc[-7:-2].max())
    local_peak = peak_high > prior_high
    turned_down = (
        closes.iloc[-1] < closes.iloc[-2]
        and closes.iloc[-1] <= opens.iloc[-1]
    )
    peak_upswing = float(closes.iloc[-2] / closes.iloc[-7:-2].min() - 1)
    if (
        trend_return >= g.stock_up_limit
        or not local_peak
        or not turned_down
        or peak_upswing < g.min_peak_upswing
        or upper_wick_ratio < g.min_upper_wick_ratio
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
        'high_position': high_position,
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


def get_target_edge(sell_value, sell_commission):
    buy_commission = max(sell_value * 0.0003, 5.0)
    cost_rate = (sell_commission + buy_commission) / sell_value
    protected_edge = (
        cost_rate
        + g.min_net_profit_rate
        + g.confirmation_rebound_allowance
        + g.slippage_rate
    )
    return max(g.target_gross_edge, protected_edge)


def get_expected_cycle_profit(price, amount, pending):
    expected_price = price * (1 + g.slippage_rate)
    buy_value = expected_price * amount
    buy_commission = max(buy_value * 0.0003, 5.0)
    return pending['net_sell_proceeds'] - buy_value - buy_commission


def submit_order(stock, amount):
    order_result = order(stock, int(amount))
    if order_result is None or order_result.filled <= 0:
        return None

    return {
        'amount': int(order_result.filled),
        'price': float(order_result.price),
        'commission': float(order_result.commission or 0.0),
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
    pending_count = 0
    max_pending_days = 0
    total_initial_basis = 0.0
    total_core_shares = 0
    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        state = g.states.get(stock, {})
        day_profit += float(state.get('day_profit', 0.0))
        day_rounds += int(state.get('round_trips', 0))
        pending = state.get('pending')
        if pending is not None:
            pending_count += 1
            pending_days = (context.current_dt.date() - pending['opened_at'].date()).days
            max_pending_days = max(max_pending_days, pending_days)

        core_shares = int(g.core_shares.get(stock, 0))
        initial_cost = float(g.core_initial_cost.get(stock, 0.0))
        total_initial_basis += core_shares * initial_cost
        total_core_shares += core_shares
        if core_shares <= 0:
            continue

        total_profit = float(g.total_t_profit.get(stock, 0.0))
        effective_cost = initial_cost - total_profit / core_shares
        pending_text = '无'
        if pending is not None:
            pending_text = '%d股/%d天/目标%.3f' % (
                pending['amount'],
                (context.current_dt.date() - pending['opened_at'].date()).days,
                pending['target_price'],
            )
        log.info(
            '%s 收盘持仓%d股，基准成本%.3f，降本后成本%.3f，累计降本%.2f，待回补%s' % (
                stock,
                position.total_amount,
                initial_cost,
                effective_cost,
                total_profit,
                pending_text,
            )
        )

    total_profit = 0.0
    for value in g.total_t_profit.values():
        total_profit += float(value)
    effective_cost = 0.0
    cost_reduction = 0.0
    if total_core_shares > 0:
        effective_cost = (total_initial_basis - total_profit) / total_core_shares
        cost_reduction = total_profit / total_core_shares

    total_rounds = int(g.t_wins) + int(g.t_losses)
    win_rate = float(g.t_wins) / float(total_rounds) if total_rounds else 0.0
    record(
        t_day_profit=float(day_profit),
        t_cum_profit=float(total_profit),
        t_day_rounds=float(day_rounds),
        t_win_rate=float(win_rate),
        t_cost_reduction=float(cost_reduction),
        t_effective_cost=float(effective_cost),
        t_pending_count=float(pending_count),
        t_pending_days=float(max_pending_days),
        t_floor_touches=float(g.floor_touches),
        t_low_confirmed=float(g.low_confirmed),
    )
    log.info(
        '降本汇总：完成%d轮，盈利%d，异常亏损%d，累计降本%.2f，每股降本%.4f，待回补%d' % (
            g.total_t_rounds,
            g.t_wins,
            g.t_losses,
            total_profit,
            cost_reduction,
            pending_count,
        )
    )

    if g.trading_days % 20 == 0:
        log.info(
            '信号过滤：日线%d，数据%d，振幅%d，偏离%d，RSI%d，位置%d，反转%d，量能%d，大盘%d' % (
                g.filter_counts['daily'],
                g.filter_counts['bars'],
                g.filter_counts['range'],
                g.filter_counts['deviation'],
                g.filter_counts['rsi'],
                g.filter_counts['location'],
                g.filter_counts['reversal'],
                g.filter_counts['volume'],
                g.filter_counts['market'],
            )
        )
        log.info(
            '回补过滤：数据%d，未确认低点%d，净利润不足%d，触底%d，确认%d' % (
                g.filter_counts['rebuy_bars'],
                g.filter_counts['rebuy_bottom'],
                g.filter_counts['rebuy_profit'],
                g.floor_touches,
                g.low_confirmed,
            )
        )
