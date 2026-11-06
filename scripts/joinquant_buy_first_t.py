from jqdata import *
import numpy as np
import pandas as pd


EXPERIMENT_GROUP = 'I'
# 股票池与核心仓仓位（改这里即可做多标的/降仓实验，其余逻辑全自动适配）
TARGET_STOCKS = ['000001.XSHE']
CORE_TOTAL_RATIO = 0.70

# 同一 tick 内 get_bars 结果缓存：多标的/多处调用共享，行为与直接调用一致
_tick_cache = {}
_tick_key = None


def reset_tick_cache(now):
    global _tick_cache, _tick_key
    _tick_key = (now.date(), now.hour, now.minute)
    _tick_cache = {}


def cached_get_bars(stock, count, unit, fields):
    key = (_tick_key, stock, unit, count, tuple(fields))
    if key not in _tick_cache:
        _tick_cache[key] = get_bars(
            stock,
            count=count,
            unit=unit,
            fields=list(fields),
            include_now=False,
            df=True,
        )
    return _tick_cache[key]
OPEN_COMMISSION_RATE = 0.0003
CLOSE_COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.0005
MIN_COMMISSION = 5.0


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_order_cost(
        OrderCost(
            close_tax=STAMP_DUTY_RATE,
            open_commission=OPEN_COMMISSION_RATE,
            close_commission=CLOSE_COMMISSION_RATE,
            min_commission=MIN_COMMISSION,
        ),
        type='stock',
    )
    set_slippage(PriceRelatedSlippage(0.0001), type='stock')

    g.target_stocks = list(TARGET_STOCKS)
    g.auto_build = True
    g.enable_t = True
    g.enable_sell_first = False
    g.experiment_group = EXPERIMENT_GROUP.upper()
    if g.experiment_group not in ('A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'):
        raise ValueError('实验组必须是 A、B、C、D、E、G、H、I、J、K、L、M 或 N')
    g.use_pressure_filter = g.experiment_group in ('B', 'D')
    g.use_opening_path_filter = g.experiment_group in ('C', 'D')
    g.use_opening_breakdown_filter = g.experiment_group in ('E', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N')
    g.use_adaptive_band_entry = g.experiment_group in ('G', 'H', 'I', 'J', 'K', 'L', 'M', 'N')
    g.use_strict_adaptive_entry = g.experiment_group in ('H', 'I', 'J', 'K', 'L', 'M', 'N')
    g.use_confirmed_close_stop = g.experiment_group == 'H'
    g.use_emergency_only_stop = g.experiment_group in ('I', 'J', 'K', 'L', 'M', 'N')
    g.use_time_decay_target = g.experiment_group in ('J', 'L')
    g.use_trailing_take_profit = g.experiment_group in ('K', 'L')
    # L 组 = J + K + 分批止盈 / 量能确认 / 日内熔断；M 组 = I + 尾盘出场管理
    # N 组 = I + 核心仓风控（只减不加：破位砍波段仓，保留 T 底仓）
    g.use_scale_out = g.experiment_group == 'L'
    g.use_confirm_volume = g.experiment_group == 'L'
    g.use_intraday_breakers = g.experiment_group == 'L'
    g.use_afternoon_exit = g.experiment_group == 'M'
    g.use_core_risk_mgmt = g.experiment_group == 'N'
    g.collect_experiment_features = g.experiment_group in ('E', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N')
    g.use_pressure_context = (
        g.use_pressure_filter or g.collect_experiment_features
    )
    g.target_total_ratio = CORE_TOTAL_RATIO
    g.max_t_ratio = 0.30
    g.max_sell_first_ratio = 0.15
    g.target_trade_value = 15000
    g.min_trade_value = 12000
    g.min_sell_first_trade_value = 7000
    g.max_round_trips = 1

    g.entry_atr_multiple = 1.20
    g.entry_daily_atr_multiple = 0.35
    g.rebound_atr_multiple = 0.40
    g.high_pullback_atr_multiple = 0.30
    g.stop_atr_buffer = 0.25
    g.reward_risk_multiple = 1.50
    g.target_atr_multiple = 1.00
    g.min_net_profit_rate = 0.0020
    g.max_stop_loss = 0.0080
    g.slippage_rate = 0.0001
    g.max_candidate_rsi = 35
    g.max_confirm_rsi = 45
    g.min_high_candidate_rsi = 65
    g.min_high_confirm_rsi = 55
    g.confirm_bars_required = 2
    g.high_confirm_bars_required = 1
    g.max_candidate_bars = 6
    g.min_intraday_range = 0.009
    g.min_daily_range = 0.012
    g.max_low_position = 0.25
    g.min_high_position = 0.75
    g.max_intraday_decline = -0.015
    g.max_intraday_rise = 0.020
    g.market_down_limit = -0.012
    g.market_up_limit = 0.012
    g.min_vol_percentile = 0.20
    g.max_vol_percentile = 0.85
    g.gap_atr_multiple = 0.25
    g.min_gap_threshold = 0.004
    g.max_gap_threshold = 0.010

    g.pressure_lookback = 60
    g.min_pressure_volume_percentile = 0.70
    g.opening_dip_atr_multiple = 0.20
    g.opening_min_dip = 0.0030
    g.opening_breakdown_atr_multiple = 0.75
    g.opening_max_decline = 0.0120
    g.opening_max_efficiency = 0.70
    g.opening_min_close_position = 0.20

    g.adaptive_vwap_bars = 12
    g.adaptive_vol_span = 12
    g.adaptive_atr_multiple_low = 0.70
    g.adaptive_atr_multiple_high = 1.00
    g.adaptive_ewma_multiple = 1.00
    g.adaptive_reentry_atr_multiple = 0.15
    g.adaptive_confirm_bars_required = 1
    g.h_reentry_atr_multiple = 0.30
    g.h_confirm_bars_required = 2
    g.h_stop_confirm_bars = 2
    g.h_emergency_stop_loss = 0.010

    # L 组（J + K + 优化叠加）参数
    g.scale_out_ratio = 0.50
    g.scale_out_target_multiple = 1.80
    g.confirm_volume_ratio = 1.20
    g.intraday_day_loss_limit = 0.010
    g.intraday_core_loss_limit = 0.020
    g.time_decay_start_time = (13, 30)
    g.time_decay_late_time = (14, 0)
    g.time_decay_mid_factor = 0.80
    g.time_decay_late_factor = 0.60
    g.trailing_activation_atr = 0.50
    g.trailing_distance_atr = 0.80

    # M 组（I + 尾盘出场管理）：14:10 浮亏即平，14:25 浮盈即锁
    # 注意：实测"尾盘止损"会割在尾盘反弹前（14:10→14:45 低吸仓平均回抽），
    # 建议下一轮把 afternoon_loss_cut_enabled 改为 False 只测"仅锁盈"版本。
    g.afternoon_loss_cut_time = (14, 10)
    g.afternoon_win_lock_time = (14, 25)
    g.afternoon_loss_cut_enabled = True
    g.afternoon_win_lock_enabled = True

    # N 组（I + 核心仓风控）：即时波段仓 + 死区防漂移微调
    # core_base_ratio = T 底仓（始终持有，做 T 的标的）
    # core_swing_ratio = 波段仓（破 MA20 减到 hold_fast、破 MA60 减到 hold_slow，个股/大盘任一触发）
    # swing_hold_fast/slow = 减仓后的保留比例：0.75/0.25 比 0.5/0.0 更保上升参与度（实测 2019 少赚28pp）
    # swing_deadband_ratio = 目标与持仓差额小于该比例则不调（消除每日价格漂移微调）
    # swing_warmup_days = 回测起步 N 个交易日不执行减仓（消除"起点处于熊市regime"的起始偏差；实盘建议 0）
    g.core_base_ratio = 0.30
    g.core_swing_ratio = 0.40
    g.swing_ma_fast = 20
    g.swing_ma_slow = 60
    g.swing_market_ma = 20
    g.swing_deadband_ratio = 0.05
    g.swing_hold_fast = 0.75
    g.swing_hold_slow = 0.25
    g.swing_warmup_days = 0

    g.min_daily_5d_return = -0.040
    g.max_daily_5d_return = 0.040
    g.min_daily_ma20_gap = -0.030
    g.max_daily_ma20_gap = 0.040
    g.min_daily_ma60_gap = 0.000
    g.min_ma20_slope = 0.000
    g.max_efficiency_ratio = 0.45

    g.max_consecutive_losses = 3
    g.consecutive_loss_pause_days = 5
    g.rolling_cycle_count = 10
    g.rolling_loss_limit = -300.0
    g.rolling_pause_days = 10

    g.signal_interval_minutes = 5
    g.manage_start_time = (9, 35)
    g.open_time = (10, 0)
    g.last_candidate_time = (13, 5)
    g.last_open_time = (13, 15)
    g.force_close_time = (14, 45)
    g.manage_end_time = (14, 55)

    g.states = {}
    g.daily_filters = {}
    g.core_shares = {}
    g.core_initial_cost = {}
    g.total_t_profit = {}
    g.reason_stats = {
        '止盈': {'rounds': 0, 'profit': 0.0},
        '止损': {'rounds': 0, 'profit': 0.0},
        '尾盘平仓': {'rounds': 0, 'profit': 0.0},
        '隔夜平仓': {'rounds': 0, 'profit': 0.0},
        '尾盘止损': {'rounds': 0, 'profit': 0.0},
        '尾盘锁盈': {'rounds': 0, 'profit': 0.0},
    }
    g.direction_stats = {
        'buy_first': {'rounds': 0, 'wins': 0, 'profit': 0.0},
        'sell_first': {'rounds': 0, 'wins': 0, 'profit': 0.0},
    }
    g.mode_counts = {
        'buy_first': 0,
        'sell_first': 0,
        'neutral': 0,
        'unavailable': 0,
    }
    g.cycle_profits = []
    g.total_t_rounds = 0
    g.candidates_created = 0
    g.candidates_confirmed = 0
    g.candidates_expired = 0
    g.t_wins = 0
    g.t_losses = 0
    g.consecutive_losses = 0
    g.pause_days_remaining = 0
    g.pause_today = False
    g.engine_bar_calls = 0
    g.trading_days = 0
    g.pressure_days = 0
    g.opening_path_days = 0
    g.opening_breakdown_pass_days = 0
    g.confirmed_close_stops = 0
    g.emergency_stops = 0
    g.scale_out_count = 0
    g.breaker_days = 0
    g.time_decay_exits = 0
    g.trailing_exits = 0
    g.sum_win = 0.0
    g.sum_loss = 0.0
    g.win_rounds = 0
    g.loss_rounds = 0
    g.marked_peak = 0.0
    g.marked_mdd = 0.0
    g.core_reduce_days = 0
    g.core_sell_shares = 0
    g.last_trade_mfe = 0.0
    g.last_trade_mae = 0.0
    g.last_holding_minutes = 0.0
    g.last_exit_minute = 0.0
    g.last_trade_profit = 0.0
    g.last_exit_reason_code = 0.0
    g.filter_counts = {
        'daily': 0,
        'daily_buy': 0,
        'daily_sell': 0,
        'bars': 0,
        'range': 0,
        'vol_low': 0,
        'vol_high': 0,
        'candidate': 0,
        'confirm': 0,
        'risk': 0,
        'rsi': 0,
        'location': 0,
        'bottom': 0,
        'market': 0,
        'neutral': 0,
        'pause': 0,
        'pressure': 0,
        'opening_path': 0,
        'opening_breakdown': 0,
        'volume': 0,
        'breaker': 0,
    }

    run_daily(build_initial_positions, time='09:40')
    run_daily(rebalance_core_position, time='09:40')
    run_daily(run_t_engine, time='every_bar', reference_security='000300.XSHG')


def before_trading_start(context):
    g.engine_bar_calls = 0
    g.daily_filters = {}
    g.pause_today = g.pause_days_remaining > 0
    if g.pause_days_remaining > 0:
        g.pause_days_remaining -= 1

    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        old_state = g.states.get(stock, {})
        pending = old_state.get('pending')
        if old_state.get('candidate') is not None:
            g.candidates_expired += 1
        start_total = int(position.total_amount // 100 * 100)
        start_closeable = int(position.closeable_amount // 100 * 100)
        base_shares = start_total
        if pending is not None:
            base_shares = int(pending['base_shares'])

        g.states[stock] = {
            'base_shares': base_shares,
            'start_closeable': start_closeable,
            'sellable_remaining': start_closeable,
            'max_t_shares': int(base_shares * g.max_t_ratio // 100 * 100),
            'max_sell_first_shares': int(
                base_shares * g.max_sell_first_ratio // 100 * 100
            ),
            'round_trips': 0,
            'pending': pending,
            'candidate': None,
            'day_mode': None,
            'open_regime': None,
            'open_gap': 0.0,
            'gap_threshold': 0.0,
            'opening_path': None,
            'adaptive_last_bar_time': None,
            'adaptive_was_below': False,
            'adaptive_anchor': 0.0,
            'adaptive_lower_band': 0.0,
            'adaptive_band_distance': 0.0,
            'last_action_dt': None,
            'day_profit': 0.0,
            'intraday_breaker': False,
        }
        daily_filter = get_daily_filter(stock)
        g.daily_filters[stock] = daily_filter
        if daily_filter is not None and daily_filter['pressure_allowed']:
            g.pressure_days += 1
        if position.total_amount > 0 and stock not in g.core_shares:
            g.core_shares[stock] = base_shares
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
        state = g.states.get(stock)
        if state is not None:
            state['base_shares'] = fill['amount']
            state['max_t_shares'] = int(
                fill['amount'] * g.max_t_ratio // 100 * 100
            )
            state['max_sell_first_shares'] = int(
                fill['amount'] * g.max_sell_first_ratio // 100 * 100
            )
        log.info('%s 初始建仓 %d股，成交价%.3f' % (
            stock, fill['amount'], fill['price']))


def get_core_swing_factor(stock):
    """N 组：核心仓波段因子（即时版，浅减仓档位）。
    factor: 1.0=满仓；破MA20 → min(..., swing_hold_fast)；破MA60 → min(..., swing_hold_slow)。
    个股与大盘取更严的一档（min 组合）；每日按昨日收盘即时计算，无未来函数。
    实测：0.5/0.0 档位在 2019 年 V 型反弹中少赚 28pp，故保留档位改浅为 0.75/0.25。"""
    factor = 1.0
    bars = get_bars(
        stock,
        count=g.swing_ma_slow + 10,
        unit='1d',
        fields=['close'],
        include_now=False,
        df=True,
    )
    if bars is not None and len(bars) >= g.swing_ma_slow:
        closes = bars['close'].dropna().astype(float)
        prev_close = float(closes.iloc[-1])
        ma_fast = float(closes.tail(g.swing_ma_fast).mean())
        ma_slow = float(closes.tail(g.swing_ma_slow).mean())
        if prev_close < ma_slow:
            factor = min(factor, g.swing_hold_slow)
        elif prev_close < ma_fast:
            factor = min(factor, g.swing_hold_fast)
    market = get_bars(
        '000300.XSHG',
        count=g.swing_ma_slow + 10,
        unit='1d',
        fields=['close'],
        include_now=False,
        df=True,
    )
    if market is not None and len(market) >= 60:
        mcloses = market['close'].dropna().astype(float)
        m_prev = float(mcloses.iloc[-1])
        m_ma20 = float(mcloses.tail(g.swing_market_ma).mean())
        m_ma60 = float(mcloses.tail(60).mean())
        if m_prev < m_ma60:
            factor = min(factor, g.swing_hold_slow)
        elif m_prev < m_ma20:
            factor = min(factor, g.swing_hold_fast)
    return factor


def rebalance_core_position(context):
    """N 组：核心仓风控（即时波段仓 + 死区）。每日 09:40 按昨日日线信号调整波段仓。
    回撤控制 = 破位时把波段仓换成现金；T 底仓保留，T 逻辑自动跟随持仓缩量。"""
    if not g.use_core_risk_mgmt:
        return
    if g.trading_days == 0:
        return  # 首日刚建仓，跳过
    if g.swing_warmup_days > 0 and g.trading_days <= g.swing_warmup_days:
        return  # 起步暖机期不执行核心仓减仓
    current = get_current_data()
    for stock in g.target_stocks:
        if stock not in g.core_shares:
            continue
        state = g.states.get(stock)
        if state is not None and state.get('pending') is not None:
            continue  # 有在途 T 仓位不动核心
        data = current[stock]
        price = float(data.last_price)
        if data.paused or data.is_st or not np.isfinite(price) or price <= 0:
            continue
        position = context.portfolio.positions[stock]
        swing_factor = get_core_swing_factor(stock)
        target_value = context.portfolio.total_value * (
            g.core_base_ratio + g.core_swing_ratio * swing_factor
        ) / len(g.target_stocks)
        target_shares = int(target_value / price // 100 * 100)
        current_shares = int(position.total_amount // 100 * 100)
        diff_shares = target_shares - current_shares
        position_value = float(position.total_amount) * price
        trade_value = abs(diff_shares) * price
        if trade_value < max(position_value * g.swing_deadband_ratio, 5000.0):
            log.info('%s 核心风控维持：因子%.2f，持仓%d股（漂移%d股忽略）' % (
                stock, swing_factor, current_shares, diff_shares))
            continue
        if diff_shares >= 100:
            cash_needed = diff_shares * price * 1.002
            if context.portfolio.available_cash - cash_needed >= g.target_trade_value:
                fill = submit_order(stock, diff_shares)
                if fill is not None:
                    g.core_shares[stock] = int(position.total_amount)
                    log.info('%s 核心风控加回 %d股 @%.3f' % (
                        stock, fill['amount'], fill['price']))
        elif diff_shares <= -100:
            sell_shares = int(min(-diff_shares, position.closeable_amount) // 100 * 100)
            if sell_shares >= 100:
                fill = submit_order(stock, -sell_shares)
                if fill is not None:
                    g.core_shares[stock] = int(position.total_amount)
                    g.core_reduce_days += 1
                    g.core_sell_shares += int(fill['amount'])
                    log.info('%s 核心风控减仓 %d股 @%.3f（因子%.2f，目标%d股）' % (
                        stock, fill['amount'], fill['price'],
                        swing_factor, target_shares))
        else:
            log.info('%s 核心风控维持：因子%.2f，持仓%d股' % (
                stock, swing_factor, current_shares))


def run_t_engine(context):
    g.engine_bar_calls += 1
    now = context.current_dt
    clock = (now.hour, now.minute)
    if clock < g.manage_start_time or clock > g.manage_end_time:
        return
    if now.minute % g.signal_interval_minutes != 0:
        return

    reset_tick_cache(now)
    current = get_current_data()
    market_state = None
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
                context,
                stock,
                price,
                data,
                state,
                force_close,
                overnight,
            )
            continue

        if not g.enable_t or clock < g.open_time or clock > g.last_open_time:
            continue
        if state['round_trips'] >= g.max_round_trips:
            continue
        if g.pause_today:
            g.filter_counts['pause'] += 1
            continue
        if state['day_mode'] is None:
            set_day_mode(data, state, g.daily_filters.get(stock))
        mode = state['day_mode']
        if market_state is None:
            market_state = get_market_state()
        if (
            (mode == 'buy_first' and market_state == 'down')
            or (mode == 'sell_first' and market_state == 'up')
        ):
            g.filter_counts['market'] += 1
            expire_candidate(state)
            continue
        if price >= data.high_limit or price <= data.low_limit:
            continue
        if g.use_intraday_breakers:
            if state.get('intraday_breaker'):
                g.filter_counts['breaker'] += 1
                continue
            if state['day_profit'] <= -g.target_trade_value * g.intraday_day_loss_limit:
                state['intraday_breaker'] = True
                g.breaker_days += 1
                g.filter_counts['breaker'] += 1
                expire_candidate(state)
                continue
            initial_cost = g.core_initial_cost.get(stock, 0.0)
            if initial_cost > 0 and price <= initial_cost * (1 - g.intraday_core_loss_limit):
                state['intraday_breaker'] = True
                g.breaker_days += 1
                g.filter_counts['breaker'] += 1
                expire_candidate(state)
                continue
        if state['candidate'] is None and clock > g.last_candidate_time:
            continue

        if mode == 'buy_first':
            signal = get_stock_signal(stock, price, now, state)
        else:
            signal = get_sell_first_signal(stock, price, now, state)
        if signal is None:
            continue

        if mode == 'buy_first':
            amount = get_trade_amount(context, stock, price, state)
        else:
            amount = get_sell_first_amount(context, stock, price, state)
        if amount < 100:
            continue

        order_amount = amount if mode == 'buy_first' else -amount
        fill = submit_order(stock, order_amount)
        if fill is None:
            continue

        if mode == 'buy_first':
            buy_value = fill['price'] * fill['amount']
            buy_cost = buy_value + fill['commission']
            minimum_edge = get_minimum_target_edge(
                buy_value,
                fill['commission'],
            )
            if g.use_strict_adaptive_entry:
                stop_price = signal['candidate_low']
                emergency_stop_price = (
                    fill['price'] * (1 - g.h_emergency_stop_loss)
                )
            else:
                fixed_stop = fill['price'] * (1 - g.max_stop_loss)
                technical_stop = (
                    signal['candidate_low']
                    - g.stop_atr_buffer * signal['atr_abs']
                )
                stop_price = max(fixed_stop, technical_stop)
                emergency_stop_price = stop_price
            risk_distance = max(fill['price'] - stop_price, 0.0)
            target_distance = max(
                risk_distance * g.reward_risk_multiple,
                signal['atr_abs'] * g.target_atr_multiple,
                fill['price'] * minimum_edge,
            )
            state['pending'] = {
                'direction': 'buy_first',
                'amount': fill['amount'],
                'entry_price': fill['price'],
                'remaining_buy_cost': buy_cost,
                'realized_profit': 0.0,
                'target_price': fill['price'] + target_distance,
                'original_target_price': fill['price'] + target_distance,
                'profit_floor_price': fill['price'] * (1 + minimum_edge),
                'active_target_price': fill['price'] + target_distance,
                'atr_abs': signal['atr_abs'],
                'time_decay_target': g.use_time_decay_target,
                'trailing_take_profit': g.use_trailing_take_profit,
                'trailing_active': False,
                'trailing_stop_price': 0.0,
                'take_profit_trigger': None,
                'stop_price': stop_price,
                'emergency_stop_price': emergency_stop_price,
                'confirmed_close_stop': g.use_confirmed_close_stop,
                'emergency_only_stop': g.use_emergency_only_stop,
                'stop_close_count': 0,
                'stop_last_bar_time': None,
                'stop_trigger': None,
                'max_seen_price': fill['price'],
                'min_seen_price': fill['price'],
                'base_shares': state['base_shares'],
                'opened_at': now,
                'exit_started': False,
                'exit_reason': None,
                'scale_out_done': False,
            }
            if state.get('open_regime') == 'buy_first':
                entry_name = '低开先买'
            elif state.get('open_regime') == 'sell_first':
                entry_name = '高开回落低吸'
            else:
                entry_name = '平开低吸'
            log.info(
                '%s %s %d股，成交价%.3f，止盈%.3f，止损%.3f，紧急%.3f，跳空%.2f%%' % (
                    stock,
                    entry_name,
                    fill['amount'],
                    fill['price'],
                    state['pending']['target_price'],
                    state['pending']['stop_price'],
                    state['pending']['emergency_stop_price'],
                    state['open_gap'] * 100,
                )
            )
        else:
            sell_value = fill['price'] * fill['amount']
            sell_proceeds = sell_value - fill['commission']
            minimum_edge = get_minimum_rebuy_edge(
                sell_value,
                fill['commission'],
            )
            fixed_stop = fill['price'] * (1 + g.max_stop_loss)
            technical_stop = (
                signal['candidate_high'] + g.stop_atr_buffer * signal['atr_abs']
            )
            stop_price = min(fixed_stop, technical_stop)
            risk_distance = max(stop_price - fill['price'], 0.0)
            target_distance = max(
                risk_distance * g.reward_risk_multiple,
                signal['atr_abs'] * g.target_atr_multiple,
                fill['price'] * minimum_edge,
            )
            state['pending'] = {
                'direction': 'sell_first',
                'amount': fill['amount'],
                'entry_price': fill['price'],
                'remaining_sell_proceeds': sell_proceeds,
                'realized_profit': 0.0,
                'target_price': fill['price'] - target_distance,
                'base_target_distance': target_distance,
                'stop_price': stop_price,
                'base_shares': state['base_shares'],
                'opened_at': now,
                'exit_started': False,
                'exit_reason': None,
                'scale_out_done': False,
            }
            state['sellable_remaining'] -= fill['amount']
            log.info(
                '%s 高开先卖 %d股，成交价%.3f，回补目标%.3f，止损回补%.3f，跳空%.2f%%' % (
                    stock,
                    fill['amount'],
                    fill['price'],
                    state['pending']['target_price'],
                    state['pending']['stop_price'],
                    state['open_gap'] * 100,
                )
            )
        state['candidate'] = None
        state['last_action_dt'] = now


def close_pending_cycle(context, stock, price, data, state, force_close, overnight):
    pending = state['pending']
    direction = pending.get('direction', 'buy_first')
    amount = int(pending['amount'] // 100 * 100)
    if amount < 100:
        finish_cycle(stock, state, pending)
        return
    if direction == 'buy_first' and price <= data.low_limit:
        return
    if direction == 'sell_first' and price >= data.high_limit:
        return

    update_pending_excursions(stock, price, pending)

    scale_out = False
    if overnight:
        pending['exit_started'] = True
        pending['exit_reason'] = '隔夜平仓'
    elif not pending['exit_started']:
        clock = (context.current_dt.hour, context.current_dt.minute)
        stop_trigger = None
        take_profit_trigger = 'fixed'
        if direction == 'buy_first':
            take_profit_trigger = get_buy_first_take_profit_trigger(
                price,
                clock,
                pending,
            )
            take_profit = take_profit_trigger is not None
            stop_trigger = get_buy_first_stop_trigger(
                stock,
                price,
                pending,
            )
            stop_loss = stop_trigger is not None
        else:
            take_profit = price <= pending['target_price']
            stop_loss = price >= pending['stop_price']
        if g.use_afternoon_exit:
            marked_profit = get_pending_marked_profit(price, pending)
        else:
            marked_profit = 0.0
        if take_profit:
            reason = '止盈'
            pending['take_profit_trigger'] = take_profit_trigger
        elif (
            g.use_afternoon_exit
            and g.afternoon_loss_cut_enabled
            and clock >= g.afternoon_loss_cut_time
            and marked_profit < 0
        ):
            reason = '尾盘止损'
        elif (
            g.use_afternoon_exit
            and g.afternoon_win_lock_enabled
            and clock >= g.afternoon_win_lock_time
            and marked_profit > 0
        ):
            reason = '尾盘锁盈'
        elif stop_loss:
            reason = '止损'
            pending['stop_trigger'] = stop_trigger
            if stop_trigger == 'confirmed_close':
                g.confirmed_close_stops += 1
            elif stop_trigger == 'emergency':
                g.emergency_stops += 1
        elif force_close:
            reason = '尾盘平仓'
        else:
            return
        scale_out = (
            g.use_scale_out
            and reason == '止盈'
            and not pending.get('scale_out_done', False)
            and pending['amount'] >= 200
        )
        if not scale_out:
            if reason == '止盈' and take_profit_trigger == 'time_decay':
                g.time_decay_exits += 1
            elif reason == '止盈' and take_profit_trigger == 'trailing':
                g.trailing_exits += 1
            pending['exit_started'] = True
            pending['exit_reason'] = reason

    remaining_before = pending['amount']
    if direction == 'buy_first':
        position = context.portfolio.positions[stock]
        close_amount = int(min(
            amount,
            state['sellable_remaining'],
            position.closeable_amount,
        ) * (g.scale_out_ratio if scale_out else 1.0) // 100 * 100)
        if close_amount < 100:
            return
        fill = submit_order(stock, -close_amount)
        if fill is None:
            return
        filled_ratio = float(fill['amount']) / float(remaining_before)
        allocated_buy_cost = pending['remaining_buy_cost'] * filled_ratio
        sell_proceeds = fill['price'] * fill['amount'] - fill['commission']
        profit = sell_proceeds - allocated_buy_cost
        pending['remaining_buy_cost'] -= allocated_buy_cost
        state['sellable_remaining'] -= fill['amount']
    else:
        cash_amount = get_max_affordable_buy_amount(
            context.portfolio.available_cash,
            price,
            amount,
        )
        close_amount = int(min(amount, cash_amount) * (
            g.scale_out_ratio if scale_out else 1.0
        ) // 100 * 100)
        if close_amount < 100:
            return
        fill = submit_order(stock, close_amount)
        if fill is None:
            return
        filled_ratio = float(fill['amount']) / float(remaining_before)
        allocated_sell_proceeds = (
            pending['remaining_sell_proceeds'] * filled_ratio
        )
        buy_cost = fill['price'] * fill['amount'] + fill['commission']
        profit = allocated_sell_proceeds - buy_cost
        pending['remaining_sell_proceeds'] -= allocated_sell_proceeds

    pending['amount'] -= fill['amount']
    pending['realized_profit'] += profit
    state['day_profit'] += profit
    state['last_action_dt'] = context.current_dt
    g.total_t_profit[stock] += profit

    direction_name = '先买后卖' if direction == 'buy_first' else '先卖后买'
    if scale_out:
        pending['scale_out_done'] = True
        if direction == 'buy_first':
            base_distance = abs(
                float(pending['original_target_price'])
                - float(pending['entry_price'])
            )
            second_target = (
                float(pending['entry_price'])
                + base_distance * g.scale_out_target_multiple
            )
            pending['target_price'] = second_target
            pending['original_target_price'] = second_target
        else:
            base_distance = float(pending.get('base_target_distance', 0.0))
            second_target = (
                float(pending['entry_price'])
                - base_distance * g.scale_out_target_multiple
            )
            pending['target_price'] = second_target
        g.scale_out_count += 1
        log.info('%s %s 分批止盈(第1笔) %d股，成交价%.3f，本次收益%.2f，剩余目标%.3f' % (
            stock,
            direction_name,
            fill['amount'],
            fill['price'],
            profit,
            second_target,
        ))
        return
    exit_name = pending['exit_reason']
    if pending.get('stop_trigger') == 'confirmed_close':
        exit_name += '-双收盘破位'
    elif pending.get('stop_trigger') == 'emergency':
        exit_name += '-紧急止损'
    elif pending.get('take_profit_trigger') == 'time_decay':
        exit_name += '-时间衰减'
    elif pending.get('take_profit_trigger') == 'trailing':
        exit_name += '-移动止盈'
    log.info('%s %s/%s %d股，成交价%.3f，本次收益%.2f' % (
        stock,
        direction_name,
        exit_name,
        fill['amount'],
        fill['price'],
        profit,
    ))

    if pending['amount'] < 100:
        finish_cycle(stock, state, pending)


def get_effective_buy_target(clock, pending):
    original_target = float(pending['original_target_price'])
    if not pending.get('time_decay_target', False):
        return original_target
    if clock >= g.time_decay_late_time:
        factor = g.time_decay_late_factor
    elif clock >= g.time_decay_start_time:
        factor = g.time_decay_mid_factor
    else:
        factor = 1.0
    entry_price = float(pending['entry_price'])
    decayed_target = entry_price + (original_target - entry_price) * factor
    return max(decayed_target, float(pending['profit_floor_price']))


def get_buy_first_take_profit_trigger(price, clock, pending):
    active_target = get_effective_buy_target(clock, pending)
    pending['active_target_price'] = active_target
    if pending.get('trailing_take_profit', False):
        entry_price = float(pending['entry_price'])
        atr_abs = float(pending['atr_abs'])
        max_seen_price = float(pending['max_seen_price'])
        activation_price = entry_price + atr_abs * g.trailing_activation_atr
        if max_seen_price >= activation_price:
            pending['trailing_active'] = True
            pending['trailing_stop_price'] = max(
                float(pending['profit_floor_price']),
                max_seen_price - atr_abs * g.trailing_distance_atr,
            )
            if price <= pending['trailing_stop_price']:
                return 'trailing'
            return None
    if price < active_target:
        return None
    if active_target < float(pending['original_target_price']):
        return 'time_decay'
    return 'fixed'


def get_confirmed_buy_stop_trigger(stock, price, pending):
    if price <= pending['emergency_stop_price']:
        return 'emergency'
    bars = cached_get_bars(stock, 2, '5m', ['date', 'close'])
    if bars is None or len(bars) == 0:
        return None
    bars = bars.dropna()
    if len(bars) == 0:
        return None
    bar_time = pd.to_datetime(bars['date'].iloc[-1])
    if bar_time == pending.get('stop_last_bar_time'):
        return None
    pending['stop_last_bar_time'] = bar_time
    latest_close = float(bars['close'].iloc[-1])
    if latest_close < pending['stop_price']:
        pending['stop_close_count'] += 1
    else:
        pending['stop_close_count'] = 0
    if pending['stop_close_count'] >= g.h_stop_confirm_bars:
        return 'confirmed_close'
    return None


def get_buy_first_stop_trigger(stock, price, pending):
    if pending.get('confirmed_close_stop', False):
        return get_confirmed_buy_stop_trigger(stock, price, pending)
    if pending.get('emergency_only_stop', False):
        if price <= pending['emergency_stop_price']:
            return 'emergency'
        return None
    if price <= pending['stop_price']:
        return 'fixed'
    return None


def update_pending_excursions(stock, price, pending):
    high_price = price
    low_price = price
    bars = cached_get_bars(stock, 1, '5m', ['high', 'low'])
    if bars is not None and len(bars) > 0:
        bars = bars.dropna()
        if len(bars) > 0:
            high_price = max(high_price, float(bars['high'].iloc[-1]))
            low_price = min(low_price, float(bars['low'].iloc[-1]))
    pending['max_seen_price'] = max(
        float(pending.get('max_seen_price', pending['entry_price'])),
        high_price,
    )
    pending['min_seen_price'] = min(
        float(pending.get('min_seen_price', pending['entry_price'])),
        low_price,
    )


def finish_cycle(stock, state, pending):
    reason = pending['exit_reason'] or '尾盘平仓'
    cycle_profit = float(pending['realized_profit'])
    direction = pending.get('direction', 'buy_first')
    entry_price = float(pending['entry_price'])
    max_seen_price = float(pending.get('max_seen_price', entry_price))
    min_seen_price = float(pending.get('min_seen_price', entry_price))
    if direction == 'buy_first':
        g.last_trade_mfe = max_seen_price / entry_price - 1
        g.last_trade_mae = min_seen_price / entry_price - 1
    else:
        g.last_trade_mfe = entry_price / min_seen_price - 1
        g.last_trade_mae = entry_price / max_seen_price - 1
    closed_at = state.get('last_action_dt') or pending['opened_at']
    g.last_holding_minutes = max(
        (closed_at - pending['opened_at']).total_seconds() / 60.0,
        0.0,
    )
    g.last_exit_minute = float(closed_at.hour * 60 + closed_at.minute)
    g.last_trade_profit = cycle_profit
    g.last_exit_reason_code = float({
        '止盈': 1,
        '止损': 2,
        '尾盘平仓': 3,
        '隔夜平仓': 4,
        '尾盘止损': 5,
        '尾盘锁盈': 6,
    }[reason])
    state['pending'] = None
    state['round_trips'] += 1
    g.total_t_rounds += 1
    g.reason_stats[reason]['rounds'] += 1
    g.reason_stats[reason]['profit'] += cycle_profit
    g.direction_stats[direction]['rounds'] += 1
    g.direction_stats[direction]['profit'] += cycle_profit
    g.cycle_profits.append(cycle_profit)
    if len(g.cycle_profits) > 30:
        g.cycle_profits = g.cycle_profits[-30:]

    if cycle_profit > 0:
        g.t_wins += 1
        g.direction_stats[direction]['wins'] += 1
        g.consecutive_losses = 0
        g.sum_win += cycle_profit
        g.win_rounds += 1
    else:
        g.t_losses += 1
        g.consecutive_losses += 1
        g.sum_loss += -cycle_profit
        g.loss_rounds += 1

    if g.consecutive_losses >= g.max_consecutive_losses:
        g.pause_days_remaining = max(
            g.pause_days_remaining,
            g.consecutive_loss_pause_days,
        )
        g.pause_today = True

    if len(g.cycle_profits) >= g.rolling_cycle_count:
        rolling_profit = 0.0
        for value in g.cycle_profits[-g.rolling_cycle_count:]:
            rolling_profit += float(value)
        if rolling_profit <= g.rolling_loss_limit:
            g.pause_days_remaining = max(
                g.pause_days_remaining,
                g.rolling_pause_days,
            )
            g.pause_today = True

    direction_name = '先买后卖' if direction == 'buy_first' else '先卖后买'
    log.info('%s 完成一轮%s，收益%.2f，连续亏损%d，暂停剩余%d天' % (
        stock,
        direction_name,
        cycle_profit,
        g.consecutive_losses,
        g.pause_days_remaining,
    ))


def set_day_mode(data, state, daily_filter):
    mode = 'buy_first'
    open_regime = 'unavailable' if daily_filter is None else 'neutral'
    open_gap = 0.0
    threshold = 0.0
    if daily_filter is not None:
        day_open = float(getattr(data, 'day_open', np.nan))
        previous_close = float(daily_filter['previous_close'])
        threshold = max(
            g.min_gap_threshold,
            min(
                g.max_gap_threshold,
                daily_filter['daily_atr_pct'] * g.gap_atr_multiple,
            ),
        )
        if np.isfinite(day_open) and day_open > 0 and previous_close > 0:
            open_gap = day_open / previous_close - 1
            if open_gap >= threshold:
                open_regime = 'sell_first'
                if g.enable_sell_first:
                    mode = 'sell_first'
            elif open_gap <= -threshold:
                mode = 'buy_first'
                open_regime = 'buy_first'
    state['day_mode'] = mode
    state['open_regime'] = open_regime
    state['open_gap'] = open_gap
    state['gap_threshold'] = threshold
    g.mode_counts[open_regime] += 1


def expire_candidate(state):
    if state.get('candidate') is None:
        return
    state['candidate'] = None
    g.candidates_expired += 1


def get_stock_signal(stock, price, now, state):
    daily_filter = g.daily_filters.get(stock)
    if daily_filter is None:
        g.filter_counts['daily'] += 1
        return None
    if not daily_filter['buy_allowed']:
        if daily_filter['high_vol_down']:
            g.filter_counts['vol_high'] += 1
        elif daily_filter['vol_percentile'] < g.min_vol_percentile:
            g.filter_counts['vol_low'] += 1
        else:
            g.filter_counts['daily_buy'] += 1
        expire_candidate(state)
        return None
    if g.use_pressure_filter and not daily_filter['pressure_allowed']:
        g.filter_counts['pressure'] += 1
        expire_candidate(state)
        return None

    bars = cached_get_bars(
        stock,
        48,
        '5m',
        ['date', 'open', 'high', 'low', 'close', 'volume', 'money'],
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

    if g.use_opening_path_filter or g.use_opening_breakdown_filter:
        if state['opening_path'] is None:
            opening_path = evaluate_opening_path(
                today.iloc[:6],
                daily_filter['daily_atr_pct'],
                state['open_gap'],
            )
            state['opening_path'] = opening_path
            if opening_path['allowed']:
                g.opening_path_days += 1
            if not opening_path['one_way_breakdown']:
                g.opening_breakdown_pass_days += 1
        reject_reason = get_opening_reject_reason(state['opening_path'])
        if reject_reason is not None:
            g.filter_counts[reject_reason] += 1
            expire_candidate(state)
            return None

    today_high = float(today['high'].max())
    today_low = float(today['low'].min())
    intraday_range = float(today_high / today_low - 1)
    required_range = max(
        g.min_intraday_range,
        daily_filter['median_range'] * 0.50,
    )
    if intraday_range < required_range:
        g.filter_counts['range'] += 1
        return None

    closes = bars['close'].astype(float)
    opens = bars['open'].astype(float)
    highs = bars['high'].astype(float)
    lows = bars['low'].astype(float)
    previous = closes.shift(1)
    true_range = pd.concat([
        highs - lows,
        (highs - previous).abs(),
        (lows - previous).abs(),
    ], axis=1).max(axis=1)
    atr_abs = float(true_range.tail(14).mean())
    if not np.isfinite(atr_abs) or atr_abs <= 0:
        g.filter_counts['bars'] += 1
        return None
    atr_pct = atr_abs / price
    log_returns = np.log(closes / closes.shift(1)).dropna()
    rv60 = float(np.sqrt((log_returns.tail(12) ** 2).sum()))
    cost_rate = estimate_round_cost_rate(g.target_trade_value)
    expected_move = max(rv60, daily_filter['daily_atr_pct'] * 0.25)
    if expected_move < cost_rate + g.min_net_profit_rate:
        g.filter_counts['vol_low'] += 1
        expire_candidate(state)
        return None

    if g.use_adaptive_band_entry:
        adaptive_levels = calculate_adaptive_buy_levels(
            today,
            price,
            atr_abs,
            cost_rate,
            daily_filter['vol_percentile'],
        )
        vwap = adaptive_levels['anchor']
        entry_threshold = adaptive_levels['lower_band']
        state['adaptive_anchor'] = vwap
        state['adaptive_lower_band'] = entry_threshold
        state['adaptive_band_distance'] = adaptive_levels['band_distance']
    else:
        adaptive_levels = None
        vwap = float(today['money'].sum() / today['volume'].sum())
        entry_distance = max(
            atr_abs * g.entry_atr_multiple,
            price * daily_filter['daily_atr_pct'] * g.entry_daily_atr_multiple,
            price * (cost_rate + g.min_net_profit_rate),
        )
        entry_threshold = vwap - entry_distance
    low_position = (price - today_low) / max(today_high - today_low, price * 0.001)
    trend_return = float(closes.iloc[-1] / closes.iloc[-20] - 1)
    bar_time = pd.to_datetime(bars['date'].iloc[-1])
    candidate = state['candidate']

    if g.use_adaptive_band_entry:
        latest_close = float(closes.iloc[-1])
        latest_low = float(lows.iloc[-1])
        if candidate is None:
            if bar_time == state.get('adaptive_last_bar_time'):
                g.filter_counts['candidate'] += 1
                return None
            state['adaptive_last_bar_time'] = bar_time
            below_band = latest_close <= entry_threshold
            crossed_below = below_band and not state.get(
                'adaptive_was_below',
                False,
            )
            state['adaptive_was_below'] = below_band
            if not crossed_below:
                g.filter_counts['candidate'] += 1
                return None
            if trend_return <= g.max_intraday_decline:
                g.filter_counts['bottom'] += 1
                return None
            state['candidate'] = {
                'direction': 'buy_first',
                'low': min(latest_low, price),
                'atr_abs': atr_abs,
                'last_bar_time': bar_time,
                'observed_bars': 0,
                'bars_without_new_low': 0,
                'vwap': vwap,
                'lower_band': entry_threshold,
            }
            g.candidates_created += 1
            g.filter_counts['confirm'] += 1
            return None

        if bar_time == candidate['last_bar_time']:
            g.filter_counts['confirm'] += 1
            return None

        candidate['last_bar_time'] = bar_time
        candidate['observed_bars'] += 1
        state['adaptive_last_bar_time'] = bar_time
        state['adaptive_was_below'] = latest_close <= entry_threshold
        if latest_low < candidate['low']:
            candidate['low'] = latest_low
            candidate['atr_abs'] = atr_abs
            candidate['bars_without_new_low'] = 0
        else:
            candidate['bars_without_new_low'] += 1

        if candidate['observed_bars'] > g.max_candidate_bars:
            expire_candidate(state)
            g.filter_counts['confirm'] += 1
            return None
        confirm_bars_required = (
            g.h_confirm_bars_required
            if g.use_strict_adaptive_entry
            else g.adaptive_confirm_bars_required
        )
        if candidate['bars_without_new_low'] < confirm_bars_required:
            g.filter_counts['confirm'] += 1
            return None

        if g.use_confirm_volume:
            latest_volume = float(bars['volume'].iloc[-1])
            prev_volume = float(bars['volume'].iloc[-4:-1].mean())
            if (
                not np.isfinite(latest_volume)
                or not np.isfinite(prev_volume)
                or latest_volume < prev_volume * g.confirm_volume_ratio
            ):
                g.filter_counts['volume'] += 1
                return None

        reentry_atr_multiple = (
            g.h_reentry_atr_multiple
            if g.use_strict_adaptive_entry
            else g.adaptive_reentry_atr_multiple
        )
        reentry_price = (
            entry_threshold
            + candidate['atr_abs'] * reentry_atr_multiple
        )
        turned_up = latest_close > float(closes.iloc[-2])
        if g.use_strict_adaptive_entry:
            turned_up = turned_up and latest_close > float(highs.iloc[-2])
        if latest_close < reentry_price or not turned_up:
            if latest_close >= vwap:
                expire_candidate(state)
                state['adaptive_was_below'] = False
            g.filter_counts['confirm'] += 1
            return None

        expected_buy_price = price * (1 + g.slippage_rate)
        if g.use_strict_adaptive_entry:
            technical_stop = candidate['low']
        else:
            technical_stop = (
                candidate['low'] - g.stop_atr_buffer * candidate['atr_abs']
            )
        risk_pct = (expected_buy_price - technical_stop) / expected_buy_price
        if risk_pct <= 0 or risk_pct > g.max_stop_loss:
            expire_candidate(state)
            g.filter_counts['risk'] += 1
            return None

        signal = {
            'vwap': vwap,
            'rsi': calculate_rsi(closes, 7),
            'trend_return': trend_return,
            'atr_pct': atr_pct,
            'atr_abs': candidate['atr_abs'],
            'rv60': rv60,
            'low_position': low_position,
            'candidate_low': candidate['low'],
            'lower_band': entry_threshold,
            'reentry_price': reentry_price,
        }
        state['candidate'] = None
        state['adaptive_was_below'] = False
        g.candidates_confirmed += 1
        return signal

    if candidate is None:
        if low_position > g.max_low_position:
            g.filter_counts['location'] += 1
            return None
        if price > entry_threshold:
            g.filter_counts['candidate'] += 1
            return None
        rsi = calculate_rsi(closes, 7)
        if rsi > g.max_candidate_rsi:
            g.filter_counts['rsi'] += 1
            return None
        recent_low = float(lows.iloc[-1])
        if recent_low > float(lows.iloc[-6:-1].min()):
            g.filter_counts['bottom'] += 1
            return None
        if trend_return <= g.max_intraday_decline:
            g.filter_counts['bottom'] += 1
            return None
        state['candidate'] = {
            'direction': 'buy_first',
            'low': min(recent_low, price),
            'atr_abs': atr_abs,
            'last_bar_time': bar_time,
            'observed_bars': 0,
            'bars_without_new_low': 0,
            'vwap': vwap,
        }
        g.candidates_created += 1
        g.filter_counts['confirm'] += 1
        return None

    if bar_time == candidate['last_bar_time']:
        g.filter_counts['confirm'] += 1
        return None

    candidate['last_bar_time'] = bar_time
    candidate['observed_bars'] += 1
    latest_low = float(lows.iloc[-1])
    if latest_low < candidate['low']:
        candidate['low'] = latest_low
        candidate['atr_abs'] = atr_abs
        candidate['bars_without_new_low'] = 0
    else:
        candidate['bars_without_new_low'] += 1

    if candidate['observed_bars'] > g.max_candidate_bars or price >= vwap:
        expire_candidate(state)
        g.filter_counts['confirm'] += 1
        return None
    if candidate['bars_without_new_low'] < g.confirm_bars_required:
        g.filter_counts['confirm'] += 1
        return None

    if g.use_confirm_volume:
        latest_volume = float(bars['volume'].iloc[-1])
        prev_volume = float(bars['volume'].iloc[-4:-1].mean())
        if (
            not np.isfinite(latest_volume)
            or not np.isfinite(prev_volume)
            or latest_volume < prev_volume * g.confirm_volume_ratio
        ):
            g.filter_counts['volume'] += 1
            return None

    rsi = calculate_rsi(closes, 7)
    previous_rsi = calculate_rsi(closes.iloc[:-1], 7)
    rebound = price - candidate['low']
    turned_up = (
        closes.iloc[-1] > opens.iloc[-1]
        and closes.iloc[-1] > closes.iloc[-2]
        and closes.iloc[-1] >= closes.tail(3).mean()
    )
    if (
        rsi > g.max_confirm_rsi
        or rsi <= previous_rsi
        or rebound < candidate['atr_abs'] * g.rebound_atr_multiple
        or not turned_up
    ):
        g.filter_counts['confirm'] += 1
        return None

    expected_buy_price = price * (1 + g.slippage_rate)
    technical_stop = candidate['low'] - g.stop_atr_buffer * candidate['atr_abs']
    risk_pct = (expected_buy_price - technical_stop) / expected_buy_price
    if risk_pct <= 0 or risk_pct > g.max_stop_loss:
        expire_candidate(state)
        g.filter_counts['risk'] += 1
        return None

    signal = {
        'vwap': vwap,
        'rsi': rsi,
        'trend_return': trend_return,
        'atr_pct': atr_pct,
        'atr_abs': candidate['atr_abs'],
        'rv60': rv60,
        'low_position': low_position,
        'candidate_low': candidate['low'],
    }
    state['candidate'] = None
    g.candidates_confirmed += 1
    return signal


def get_sell_first_signal(stock, price, now, state):
    daily_filter = g.daily_filters.get(stock)
    if daily_filter is None:
        g.filter_counts['daily'] += 1
        return None
    if not daily_filter['sell_allowed']:
        if daily_filter['vol_percentile'] < g.min_vol_percentile:
            g.filter_counts['vol_low'] += 1
        else:
            g.filter_counts['daily_sell'] += 1
        expire_candidate(state)
        return None

    bars = cached_get_bars(
        stock,
        48,
        '5m',
        ['date', 'open', 'high', 'low', 'close', 'volume', 'money'],
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
        daily_filter['median_range'] * 0.50,
    )
    if intraday_range < required_range:
        g.filter_counts['range'] += 1
        return None

    closes = bars['close'].astype(float)
    opens = bars['open'].astype(float)
    highs = bars['high'].astype(float)
    lows = bars['low'].astype(float)
    previous = closes.shift(1)
    true_range = pd.concat([
        highs - lows,
        (highs - previous).abs(),
        (lows - previous).abs(),
    ], axis=1).max(axis=1)
    atr_abs = float(true_range.tail(14).mean())
    if not np.isfinite(atr_abs) or atr_abs <= 0:
        g.filter_counts['bars'] += 1
        return None
    atr_pct = atr_abs / price
    log_returns = np.log(closes / closes.shift(1)).dropna()
    rv60 = float(np.sqrt((log_returns.tail(12) ** 2).sum()))
    estimated_trade_value = min(
        g.target_trade_value,
        state['max_sell_first_shares'] * price,
    )
    if estimated_trade_value <= 0:
        return None
    cost_rate = estimate_round_cost_rate(estimated_trade_value)
    expected_move = max(rv60, daily_filter['daily_atr_pct'] * 0.25)
    if expected_move < cost_rate + g.min_net_profit_rate:
        g.filter_counts['vol_low'] += 1
        expire_candidate(state)
        return None

    vwap = float(today['money'].sum() / today['volume'].sum())
    entry_distance = max(
        atr_abs * g.entry_atr_multiple,
        price * daily_filter['daily_atr_pct'] * g.entry_daily_atr_multiple,
        price * (cost_rate + g.min_net_profit_rate),
    )
    entry_threshold = vwap + entry_distance
    high_position = (
        (price - today_low) / max(today_high - today_low, price * 0.001)
    )
    trend_return = float(closes.iloc[-1] / closes.iloc[-20] - 1)
    bar_time = pd.to_datetime(bars['date'].iloc[-1])
    candidate = state['candidate']

    if candidate is None:
        if high_position < g.min_high_position:
            g.filter_counts['location'] += 1
            return None
        if price < entry_threshold:
            g.filter_counts['candidate'] += 1
            return None
        rsi = calculate_rsi(closes, 7)
        if rsi < g.min_high_candidate_rsi:
            g.filter_counts['rsi'] += 1
            return None
        recent_high = float(highs.iloc[-1])
        if recent_high < float(highs.iloc[-6:-1].max()):
            g.filter_counts['bottom'] += 1
            return None
        if trend_return >= g.max_intraday_rise:
            g.filter_counts['bottom'] += 1
            return None
        state['candidate'] = {
            'direction': 'sell_first',
            'high': max(recent_high, price),
            'atr_abs': atr_abs,
            'last_bar_time': bar_time,
            'observed_bars': 0,
            'bars_without_new_high': 0,
        }
        g.candidates_created += 1
        g.filter_counts['confirm'] += 1
        return None

    if bar_time == candidate['last_bar_time']:
        g.filter_counts['confirm'] += 1
        return None

    candidate['last_bar_time'] = bar_time
    candidate['observed_bars'] += 1
    latest_high = float(highs.iloc[-1])
    if latest_high > candidate['high']:
        candidate['high'] = latest_high
        candidate['atr_abs'] = atr_abs
        candidate['bars_without_new_high'] = 0
    else:
        candidate['bars_without_new_high'] += 1

    if candidate['observed_bars'] > g.max_candidate_bars or price <= vwap:
        expire_candidate(state)
        g.filter_counts['confirm'] += 1
        return None
    if candidate['bars_without_new_high'] < g.high_confirm_bars_required:
        g.filter_counts['confirm'] += 1
        return None

    if g.use_confirm_volume:
        latest_volume = float(bars['volume'].iloc[-1])
        prev_volume = float(bars['volume'].iloc[-4:-1].mean())
        if (
            not np.isfinite(latest_volume)
            or not np.isfinite(prev_volume)
            or latest_volume < prev_volume * g.confirm_volume_ratio
        ):
            g.filter_counts['volume'] += 1
            return None

    rsi = calculate_rsi(closes, 7)
    previous_rsi = calculate_rsi(closes.iloc[:-1], 7)
    pullback = candidate['high'] - price
    turned_down = (
        closes.iloc[-1] < opens.iloc[-1]
        and closes.iloc[-1] < closes.iloc[-2]
        and closes.iloc[-1] <= closes.tail(3).mean()
    )
    if (
        rsi < g.min_high_confirm_rsi
        or rsi >= previous_rsi
        or pullback < candidate['atr_abs'] * g.high_pullback_atr_multiple
        or not turned_down
    ):
        g.filter_counts['confirm'] += 1
        return None

    expected_sell_price = price * (1 - g.slippage_rate)
    technical_stop = candidate['high'] + g.stop_atr_buffer * candidate['atr_abs']
    risk_pct = (technical_stop - expected_sell_price) / expected_sell_price
    if risk_pct <= 0 or risk_pct > g.max_stop_loss:
        expire_candidate(state)
        g.filter_counts['risk'] += 1
        return None

    signal = {
        'vwap': vwap,
        'rsi': rsi,
        'trend_return': trend_return,
        'atr_pct': atr_pct,
        'atr_abs': candidate['atr_abs'],
        'rv60': rv60,
        'high_position': high_position,
        'candidate_high': candidate['high'],
    }
    state['candidate'] = None
    g.candidates_confirmed += 1
    return signal


def get_daily_filter(stock):
    fields = ['high', 'low', 'close']
    if g.use_pressure_context:
        fields.extend(['volume', 'money'])
    bars = get_bars(
        stock,
        count=81,
        unit='1d',
        fields=fields,
        include_now=False,
        df=True,
    )
    if bars is None or len(bars) < 81:
        return None
    bars = bars.dropna(subset=['high', 'low', 'close'])
    if len(bars) < 81:
        return None

    closes = bars['close'].astype(float)
    previous = closes.shift(1)
    highs = bars['high'].astype(float)
    lows = bars['low'].astype(float)
    true_range = pd.concat([
        highs - lows,
        (highs - previous).abs(),
        (lows - previous).abs(),
    ], axis=1).max(axis=1)
    ranges = ((highs - lows) / previous).dropna()
    median_range = float(ranges.tail(20).median())
    daily_atr_series = true_range.rolling(14).mean() / closes
    daily_atr_values = daily_atr_series.dropna().tail(60)
    daily_atr_pct = float(daily_atr_values.iloc[-1])
    vol_percentile = float(
        (daily_atr_values <= daily_atr_pct).sum()
    ) / float(len(daily_atr_values))
    five_day_return = float(closes.iloc[-1] / closes.iloc[-6] - 1)
    ma20 = float(closes.tail(20).mean())
    prior_ma20 = float(closes.iloc[-25:-5].mean())
    ma60 = float(closes.tail(60).mean())
    ma20_gap = float(closes.iloc[-1] / ma20 - 1)
    ma60_gap = float(closes.iloc[-1] / ma60 - 1)
    ma20_slope = float(ma20 / prior_ma20 - 1)
    path = closes.tail(21).diff().abs().dropna()
    path_length = 0.0
    for value in path:
        path_length += float(value)
    efficiency_ratio = (
        abs(float(closes.iloc[-1] - closes.iloc[-21])) / path_length
        if path_length > 0 else 1.0
    )
    previous_day_return = 0.0
    volume_percentile = 0.0
    money_percentile = 0.0
    pressure_allowed = False
    if g.use_pressure_context:
        volumes = bars['volume'].astype(float)
        money = bars['money'].astype(float)
        previous_day_return = float(closes.iloc[-1] / closes.iloc[-2] - 1)
        volume_percentile = percentile_rank_last(
            volumes.tail(g.pressure_lookback)
        )
        money_percentile = percentile_rank_last(
            money.tail(g.pressure_lookback)
        )
        pressure_allowed = is_pressure_reversal_day(
            previous_day_return,
            volume_percentile,
        )

    high_vol_down = (
        vol_percentile > g.max_vol_percentile
        and (ma20_gap < 0 or ma20_slope <= 0)
    )
    buy_allowed = (
        median_range >= g.min_daily_range
        and vol_percentile >= g.min_vol_percentile
        and not high_vol_down
        and five_day_return >= g.min_daily_5d_return
        and five_day_return <= g.max_daily_5d_return
        and ma20_gap >= g.min_daily_ma20_gap
        and ma20_gap <= g.max_daily_ma20_gap
        and ma60_gap >= g.min_daily_ma60_gap
        and ma20_slope >= g.min_ma20_slope
        and efficiency_ratio <= g.max_efficiency_ratio
    )
    strong_uptrend = ma20_gap > 0.040 and ma20_slope > 0.020
    sell_allowed = (
        median_range >= g.min_daily_range
        and vol_percentile >= g.min_vol_percentile
        and five_day_return >= -0.080
        and five_day_return <= 0.060
        and ma20_gap <= 0.060
        and efficiency_ratio <= 0.65
        and not strong_uptrend
    )
    return {
        'buy_allowed': buy_allowed,
        'sell_allowed': sell_allowed,
        'previous_close': float(closes.iloc[-1]),
        'median_range': median_range,
        'daily_atr_pct': daily_atr_pct,
        'vol_percentile': vol_percentile,
        'high_vol_down': high_vol_down,
        'five_day_return': five_day_return,
        'ma20_gap': ma20_gap,
        'ma60_gap': ma60_gap,
        'ma20_slope': ma20_slope,
        'efficiency_ratio': efficiency_ratio,
        'previous_day_return': previous_day_return,
        'volume_percentile': volume_percentile,
        'money_percentile': money_percentile,
        'pressure_allowed': pressure_allowed,
    }


def percentile_rank_last(values):
    latest = float(values.iloc[-1])
    if not np.isfinite(latest) or latest < 0:
        return 0.0
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean >= 0]
    if len(clean) == 0:
        return 0.0
    return float((clean <= latest).sum()) / float(len(clean))


def is_pressure_reversal_day(previous_day_return, volume_percentile):
    return (
        np.isfinite(previous_day_return)
        and np.isfinite(volume_percentile)
        and previous_day_return < 0
        and volume_percentile >= g.min_pressure_volume_percentile
    )


def get_opening_reject_reason(opening_path):
    if g.use_opening_path_filter and not opening_path['allowed']:
        return 'opening_path'
    if (
        g.use_opening_breakdown_filter
        and opening_path['one_way_breakdown']
    ):
        return 'opening_breakdown'
    return None


def evaluate_opening_path(opening_bars, daily_atr_pct, open_gap):
    first_open = float(opening_bars['open'].iloc[0])
    first_high = float(opening_bars['high'].max())
    first_low = float(opening_bars['low'].min())
    first_close = float(opening_bars['close'].iloc[-1])
    closes = opening_bars['close'].astype(float)
    values = [first_open]
    values.extend([float(value) for value in closes])
    path_length = 0.0
    for index in range(1, len(values)):
        path_length += abs(values[index] - values[index - 1])
    first_return = first_close / first_open - 1
    opening_drawdown = first_low / first_open - 1
    opening_range = first_high / first_low - 1
    close_position = (
        (first_close - first_low) / (first_high - first_low)
        if first_high > first_low else 0.5
    )
    efficiency = (
        abs(first_close - first_open) / path_length
        if path_length > 0 else 1.0
    )
    required_dip = max(
        g.opening_min_dip,
        daily_atr_pct * g.opening_dip_atr_multiple,
    )
    breakdown_limit = max(
        g.opening_max_decline,
        daily_atr_pct * g.opening_breakdown_atr_multiple,
    )
    has_dip = (
        opening_drawdown <= -required_dip
        or open_gap <= -g.min_gap_threshold
    )
    one_way_breakdown = (
        first_return <= -breakdown_limit
        and efficiency >= g.opening_max_efficiency
        and close_position <= g.opening_min_close_position
    )
    return {
        'allowed': bool(has_dip and not one_way_breakdown),
        'has_dip': bool(has_dip),
        'one_way_breakdown': bool(one_way_breakdown),
        'first_return': first_return,
        'opening_drawdown': opening_drawdown,
        'opening_range': opening_range,
        'close_position': close_position,
        'efficiency': efficiency,
        'required_dip': required_dip,
        'breakdown_limit': breakdown_limit,
    }


def get_market_state():
    bars = cached_get_bars('000300.XSHG', 20, '5m', ['close'])
    if bars is None or len(bars) < 20:
        return 'flat'
    closes = bars['close'].dropna().astype(float)
    if len(closes) < 20:
        return 'flat'
    trend_return = float(closes.iloc[-1] / closes.iloc[-20] - 1)
    if trend_return <= g.market_down_limit:
        return 'down'
    if trend_return >= g.market_up_limit:
        return 'up'
    return 'flat'


def get_trade_amount(context, stock, price, state):
    target_shares = int(g.target_trade_value / price // 100 * 100)
    position = context.portfolio.positions[stock]
    max_amount = int(min(
        state['max_t_shares'],
        target_shares,
        state['sellable_remaining'],
        position.closeable_amount,
    ) // 100 * 100)
    amount = get_max_affordable_buy_amount(
        context.portfolio.available_cash,
        price,
        max_amount,
    )
    if amount < 100 or amount * price < g.min_trade_value:
        return 0
    return amount


def get_sell_first_amount(context, stock, price, state):
    target_shares = int(g.target_trade_value / price // 100 * 100)
    position = context.portfolio.positions[stock]
    amount = int(min(
        state['max_sell_first_shares'],
        target_shares,
        state['sellable_remaining'],
        position.closeable_amount,
    ) // 100 * 100)
    if amount < 100 or amount * price < g.min_sell_first_trade_value:
        return 0
    return amount


def get_max_affordable_buy_amount(available_cash, price, max_amount):
    if available_cash <= 0 or price <= 0 or max_amount < 100:
        return 0
    expected_price = price * (1 + g.slippage_rate)
    amount = int(max_amount // 100 * 100)
    while amount >= 100:
        trade_value = expected_price * amount
        cash_required = trade_value + estimate_order_cost(
            trade_value,
            is_sell=False,
        )
        if cash_required <= available_cash:
            return amount
        amount -= 100
    return 0


def calculate_rsi(closes, period):
    delta = closes.diff()
    gain = float(delta.clip(lower=0).tail(period).mean())
    loss = float(-delta.clip(upper=0).tail(period).mean())
    if loss <= 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def calculate_adaptive_buy_levels(
    today,
    price,
    atr_abs,
    cost_rate,
    daily_vol_percentile,
):
    recent = today.tail(g.adaptive_vwap_bars)
    anchor = float(recent['money'].sum() / recent['volume'].sum())
    recent_closes = recent['close'].astype(float)
    log_returns = np.log(recent_closes / recent_closes.shift(1)).dropna()
    if len(log_returns) > 0:
        ewma_var = float(
            (log_returns ** 2).ewm(
                span=g.adaptive_vol_span,
                adjust=False,
            ).mean().iloc[-1]
        )
        ewma_vol = float(np.sqrt(max(ewma_var, 0.0)))
    else:
        ewma_vol = 0.0
    vol_range = max(g.max_vol_percentile - g.min_vol_percentile, 0.01)
    vol_position = float(np.clip(
        (daily_vol_percentile - g.min_vol_percentile) / vol_range,
        0.0,
        1.0,
    ))
    atr_multiple = (
        g.adaptive_atr_multiple_low
        + (
            g.adaptive_atr_multiple_high
            - g.adaptive_atr_multiple_low
        ) * vol_position
    )
    band_distance = max(
        atr_abs * atr_multiple,
        price * ewma_vol * g.adaptive_ewma_multiple,
        price * (cost_rate + g.min_net_profit_rate),
    )
    return {
        'anchor': anchor,
        'lower_band': anchor - band_distance,
        'band_distance': band_distance,
        'ewma_vol': ewma_vol,
        'atr_multiple': atr_multiple,
    }


def estimate_order_cost(trade_value, is_sell):
    commission_rate = (
        CLOSE_COMMISSION_RATE if is_sell else OPEN_COMMISSION_RATE
    )
    cost = max(trade_value * commission_rate, MIN_COMMISSION)
    if is_sell:
        cost += trade_value * STAMP_DUTY_RATE
    return cost


def estimate_round_cost_rate(trade_value):
    buy_cost = estimate_order_cost(trade_value, is_sell=False)
    sell_cost = estimate_order_cost(trade_value, is_sell=True)
    return (
        buy_cost + sell_cost
    ) / trade_value + g.slippage_rate * 2


def get_minimum_target_edge(buy_value, buy_cost):
    sell_cost = estimate_order_cost(buy_value, is_sell=True)
    cost_rate = (buy_cost + sell_cost) / buy_value
    return cost_rate + g.min_net_profit_rate + g.slippage_rate


def get_minimum_rebuy_edge(sell_value, sell_cost):
    buy_cost = estimate_order_cost(sell_value, is_sell=False)
    cost_rate = (sell_cost + buy_cost) / sell_value
    return cost_rate + g.min_net_profit_rate + g.slippage_rate


def submit_order(stock, amount):
    order_result = order(stock, int(amount))
    if order_result is None or order_result.filled <= 0:
        return None
    return {
        'amount': int(order_result.filled),
        'price': float(order_result.price),
        'commission': float(order_result.commission or 0.0),
    }


def get_pending_marked_profit(price, pending):
    amount = int(pending['amount'])
    if amount <= 0:
        return 0.0
    if pending.get('direction', 'buy_first') == 'sell_first':
        expected_price = price * (1 + g.slippage_rate)
        buy_value = expected_price * amount
        buy_cost = estimate_order_cost(buy_value, is_sell=False)
        return (
            pending['remaining_sell_proceeds']
            - buy_value
            - buy_cost
        )
    expected_price = price * (1 - g.slippage_rate)
    sell_value = expected_price * amount
    sell_cost = estimate_order_cost(sell_value, is_sell=True)
    return (
        sell_value
        - sell_cost
        - pending['remaining_buy_cost']
    )


def after_trading_end(context):
    if g.engine_bar_calls < 100:
        raise RuntimeError(
            '本策略必须选择分钟回测；当前每个交易日仅触发 %d 次 every_bar' %
            g.engine_bar_calls
        )

    g.trading_days += 1
    current = get_current_data()
    day_profit = 0.0
    day_rounds = 0
    pending_count = 0
    pending_shares = 0
    marked_pending_profit = 0.0
    inventory_drift = 0
    total_initial_basis = 0.0
    total_core_shares = 0

    for stock in g.target_stocks:
        position = context.portfolio.positions[stock]
        state = g.states.get(stock, {})
        expire_candidate(state)
        day_profit += float(state.get('day_profit', 0.0))
        day_rounds += int(state.get('round_trips', 0))
        pending = state.get('pending')
        base_shares = int(state.get('base_shares', position.total_amount))

        if pending is not None:
            pending_count += 1
            pending_shares += int(pending['amount'])
            price = float(current[stock].last_price)
            if np.isfinite(price) and price > 0:
                marked_pending_profit += get_pending_marked_profit(price, pending)
            if pending.get('direction', 'buy_first') == 'sell_first':
                expected_amount = base_shares - int(pending['amount'])
            else:
                expected_amount = base_shares + int(pending['amount'])
        else:
            expected_amount = base_shares
        inventory_drift += int(position.total_amount - expected_amount)

        core_shares = int(g.core_shares.get(stock, 0))
        initial_cost = float(g.core_initial_cost.get(stock, 0.0))
        total_initial_basis += core_shares * initial_cost
        total_core_shares += core_shares
        if core_shares <= 0:
            continue

        realized_profit = float(g.total_t_profit.get(stock, 0.0))
        effective_cost = initial_cost - realized_profit / core_shares
        pending_direction = ''
        if pending is not None:
            pending_direction = (
                '先卖后买' if pending.get('direction') == 'sell_first'
                else '先买后卖'
            )
        log.info(
            '%s 收盘持仓%d股，基准%d股，累计T收益%.2f，降本后成本%.3f，待处理%s%d股' % (
                stock,
                position.total_amount,
                base_shares,
                realized_profit,
                effective_cost,
                pending_direction,
                int(pending['amount']) if pending is not None else 0,
            )
        )

    total_profit = 0.0
    for value in g.total_t_profit.values():
        total_profit += float(value)
    marked_total_profit = total_profit + marked_pending_profit
    cost_reduction = 0.0
    marked_cost_reduction = 0.0
    effective_cost = 0.0
    if total_core_shares > 0:
        cost_reduction = total_profit / total_core_shares
        marked_cost_reduction = marked_total_profit / total_core_shares
        effective_cost = (
            total_initial_basis - marked_total_profit
        ) / total_core_shares

    if marked_total_profit > g.marked_peak:
        g.marked_peak = marked_total_profit
    if g.marked_peak - marked_total_profit > g.marked_mdd:
        g.marked_mdd = g.marked_peak - marked_total_profit

    total_rounds = int(g.t_wins) + int(g.t_losses)
    win_rate = float(g.t_wins) / float(total_rounds) if total_rounds else 0.0
    buy_stats = g.direction_stats['buy_first']
    sell_stats = g.direction_stats['sell_first']
    buy_win_rate = (
        float(buy_stats['wins']) / float(buy_stats['rounds'])
        if buy_stats['rounds'] else 0.0
    )
    sell_win_rate = (
        float(sell_stats['wins']) / float(sell_stats['rounds'])
        if sell_stats['rounds'] else 0.0
    )
    primary_stock = g.target_stocks[0]
    feature_daily = g.daily_filters.get(primary_stock) or {}
    feature_state = g.states.get(primary_stock) or {}
    feature_opening = feature_state.get('opening_path') or {}
    record(
        t_day_profit=float(day_profit),
        t_cum_profit=float(total_profit),
        t_marked_profit=float(marked_total_profit),
        t_day_rounds=float(day_rounds),
        t_win_rate=float(win_rate),
        t_cost_reduction=float(cost_reduction),
        t_marked_cost_reduction=float(marked_cost_reduction),
        t_effective_cost=float(effective_cost),
        t_pending_count=float(pending_count),
        t_pending_shares=float(pending_shares),
        t_inventory_drift=float(inventory_drift),
        t_pause_days=float(g.pause_days_remaining),
        t_candidate_count=float(g.candidates_created),
        t_confirmed_count=float(g.candidates_confirmed),
        t_expired_count=float(g.candidates_expired),
        t_buy_first_profit=float(buy_stats['profit']),
        t_buy_first_rounds=float(buy_stats['rounds']),
        t_buy_first_win_rate=float(buy_win_rate),
        t_sell_first_profit=float(sell_stats['profit']),
        t_sell_first_rounds=float(sell_stats['rounds']),
        t_sell_first_win_rate=float(sell_win_rate),
        t_low_open_days=float(g.mode_counts['buy_first']),
        t_high_open_days=float(g.mode_counts['sell_first']),
        t_neutral_days=float(g.mode_counts['neutral']),
        t_unavailable_days=float(g.mode_counts['unavailable']),
        t_take_profit=float(g.reason_stats['止盈']['profit']),
        t_stop_profit=float(g.reason_stats['止损']['profit']),
        t_force_profit=float(g.reason_stats['尾盘平仓']['profit']),
        t_overnight_profit=float(g.reason_stats['隔夜平仓']['profit']),
        t_experiment_group=float(
            {
                'A': 1,
                'B': 2,
                'C': 3,
                'D': 4,
                'E': 5,
                'G': 6,
                'H': 7,
                'I': 8,
                'J': 9,
                'K': 10,
                'L': 11,
                'M': 12,
                'N': 13,
            }[
                g.experiment_group
            ]
        ),
        t_pressure_days=float(g.pressure_days),
        t_opening_path_days=float(g.opening_path_days),
        t_opening_breakdown_pass_days=float(g.opening_breakdown_pass_days),
        t_pressure_rejects=float(g.filter_counts['pressure']),
        t_opening_path_rejects=float(g.filter_counts['opening_path']),
        t_opening_breakdown_rejects=float(
            g.filter_counts['opening_breakdown']
        ),
        t_prev_day_return=float(
            feature_daily.get('previous_day_return', 0.0)
        ),
        t_volume_percentile=float(
            feature_daily.get('volume_percentile', 0.0)
        ),
        t_money_percentile=float(
            feature_daily.get('money_percentile', 0.0)
        ),
        t_opening_feature_ready=float(bool(feature_opening)),
        t_opening_first_return=float(
            feature_opening.get('first_return', 0.0)
        ),
        t_opening_drawdown=float(
            feature_opening.get('opening_drawdown', 0.0)
        ),
        t_opening_efficiency=float(
            feature_opening.get('efficiency', 0.0)
        ),
        t_opening_close_position=float(
            feature_opening.get('close_position', 0.0)
        ),
        t_opening_one_way_breakdown=float(
            feature_opening.get('one_way_breakdown', False)
        ),
        t_adaptive_anchor=float(
            feature_state.get('adaptive_anchor', 0.0)
        ),
        t_adaptive_lower_band=float(
            feature_state.get('adaptive_lower_band', 0.0)
        ),
        t_adaptive_band_distance=float(
            feature_state.get('adaptive_band_distance', 0.0)
        ),
        t_confirmed_close_stops=float(g.confirmed_close_stops),
        t_emergency_stops=float(g.emergency_stops),
        t_time_decay_exits=float(g.time_decay_exits),
        t_trailing_exits=float(g.trailing_exits),
        t_scale_out_count=float(g.scale_out_count),
        t_breaker_days=float(g.breaker_days),
        t_volume_rejects=float(g.filter_counts['volume']),
        t_t_mdd=float(g.marked_mdd),
        t_profit_factor=float(g.sum_win / g.sum_loss if g.sum_loss > 0 else 0.0),
        t_avg_win=float(g.sum_win / g.win_rounds if g.win_rounds else 0.0),
        t_avg_loss=float(-g.sum_loss / g.loss_rounds if g.loss_rounds else 0.0),
        t_tp_rounds=float(g.reason_stats['止盈']['rounds']),
        t_sl_rounds=float(g.reason_stats['止损']['rounds']),
        t_force_rounds=float(g.reason_stats['尾盘平仓']['rounds']),
        t_on_rounds=float(g.reason_stats['隔夜平仓']['rounds']),
        t_as_cut_rounds=float(g.reason_stats['尾盘止损']['rounds']),
        t_as_lock_rounds=float(g.reason_stats['尾盘锁盈']['rounds']),
        t_as_cut_profit=float(g.reason_stats['尾盘止损']['profit']),
        t_as_lock_profit=float(g.reason_stats['尾盘锁盈']['profit']),
        t_f_daily=float(g.filter_counts['daily']),
        t_f_bars=float(g.filter_counts['bars']),
        t_f_range=float(g.filter_counts['range']),
        t_f_vol_low=float(g.filter_counts['vol_low']),
        t_f_vol_high=float(g.filter_counts['vol_high']),
        t_f_candidate=float(g.filter_counts['candidate']),
        t_f_confirm=float(g.filter_counts['confirm']),
        t_f_risk=float(g.filter_counts['risk']),
        t_f_rsi=float(g.filter_counts['rsi']),
        t_f_location=float(g.filter_counts['location']),
        t_f_bottom=float(g.filter_counts['bottom']),
        t_f_market=float(g.filter_counts['market']),
        t_f_pause=float(g.filter_counts['pause']),
        t_f_opening_path=float(g.filter_counts['opening_path']),
        t_f_opening_breakdown=float(g.filter_counts['opening_breakdown']),
        t_f_volume=float(g.filter_counts['volume']),
        t_f_breaker=float(g.filter_counts['breaker']),
        t_core_reduce_days=float(g.core_reduce_days),
        t_core_sell_shares=float(g.core_sell_shares),
        t_last_trade_mfe=float(g.last_trade_mfe),
        t_last_trade_mae=float(g.last_trade_mae),
        t_last_holding_minutes=float(g.last_holding_minutes),
        t_last_exit_minute=float(g.last_exit_minute),
        t_last_trade_profit=float(g.last_trade_profit),
        t_last_exit_reason=float(g.last_exit_reason_code),
    )
    log.info(
        '做T汇总：完成%d轮，盈利%d，亏损%d，累计%.2f，盯市%.2f，每股真实降本%.4f，待平仓%d，T腿回撤%.2f，盈亏比%.2f' % (
            g.total_t_rounds,
            g.t_wins,
            g.t_losses,
            total_profit,
            marked_total_profit,
            marked_cost_reduction,
            pending_count,
            g.marked_mdd,
            g.sum_win / g.sum_loss if g.sum_loss > 0 else 0.0,
        )
    )
    log.info(
        '分方向：先买后卖%d轮/%.2f/胜率%.1f%%，先卖后买%d轮/%.2f/胜率%.1f%%' % (
            buy_stats['rounds'],
            buy_stats['profit'],
            buy_win_rate * 100,
            sell_stats['rounds'],
            sell_stats['profit'],
            sell_win_rate * 100,
        )
    )
    log.info(
        '实验组%s：前日压力通过%d天/拦截%d次，开盘路径通过%d天/拦截%d次，崩落安全%d天/拦截%d次' % (
            g.experiment_group,
            g.pressure_days,
            g.filter_counts['pressure'],
            g.opening_path_days,
            g.filter_counts['opening_path'],
            g.opening_breakdown_pass_days,
            g.filter_counts['opening_breakdown'],
        )
    )

    if g.trading_days % 20 == 0:
        log.info(
            '平仓归因：止盈%d轮/%.2f(时间衰减%d/移动止盈%d)，止损%d轮/%.2f，尾盘%d轮/%.2f，尾盘止损%d轮/%.2f，尾盘锁盈%d轮/%.2f，隔夜%d轮/%.2f，分批止盈%d次' % (
                g.reason_stats['止盈']['rounds'],
                g.reason_stats['止盈']['profit'],
                g.time_decay_exits,
                g.trailing_exits,
                g.reason_stats['止损']['rounds'],
                g.reason_stats['止损']['profit'],
                g.reason_stats['尾盘平仓']['rounds'],
                g.reason_stats['尾盘平仓']['profit'],
                g.reason_stats['尾盘止损']['rounds'],
                g.reason_stats['尾盘止损']['profit'],
                g.reason_stats['尾盘锁盈']['rounds'],
                g.reason_stats['尾盘锁盈']['profit'],
                g.reason_stats['隔夜平仓']['rounds'],
                g.reason_stats['隔夜平仓']['profit'],
                g.scale_out_count,
            )
        )
        log.info(
            '信号过滤：日线%d，低开日线%d，高开日线%d，数据%d，振幅%d，低波%d，高波%d，候选%d，确认等待%d，风险%d，RSI%d，位置%d，极值%d，大盘%d，中性开盘%d，暂停%d，前日压力%d，开盘路径%d，单边崩落%d，量能%d，熔断%d' % (
                g.filter_counts['daily'],
                g.filter_counts['daily_buy'],
                g.filter_counts['daily_sell'],
                g.filter_counts['bars'],
                g.filter_counts['range'],
                g.filter_counts['vol_low'],
                g.filter_counts['vol_high'],
                g.filter_counts['candidate'],
                g.filter_counts['confirm'],
                g.filter_counts['risk'],
                g.filter_counts['rsi'],
                g.filter_counts['location'],
                g.filter_counts['bottom'],
                g.filter_counts['market'],
                g.filter_counts['neutral'],
                g.filter_counts['pause'],
                g.filter_counts['pressure'],
                g.filter_counts['opening_path'],
                g.filter_counts['opening_breakdown'],
                g.filter_counts['volume'],
                g.filter_counts['breaker'],
            )
        )
