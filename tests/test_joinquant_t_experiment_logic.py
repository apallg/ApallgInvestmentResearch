import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.unit


def load_strategy():
    sys.modules.setdefault('jqdata', types.ModuleType('jqdata'))
    path = Path(__file__).parents[1] / 'scripts' / 'joinquant_buy_first_t.py'
    spec = importlib.util.spec_from_file_location('joinquant_buy_first_t', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.g = types.SimpleNamespace(
        min_pressure_volume_percentile=0.70,
        opening_min_dip=0.0030,
        opening_dip_atr_multiple=0.20,
        opening_max_decline=0.0120,
        opening_breakdown_atr_multiple=0.75,
        opening_max_efficiency=0.70,
        opening_min_close_position=0.20,
        min_gap_threshold=0.004,
        rebound_atr_multiple=0.40,
        adaptive_vwap_bars=12,
        adaptive_vol_span=12,
        adaptive_atr_multiple_low=0.70,
        adaptive_atr_multiple_high=1.00,
        adaptive_ewma_multiple=1.00,
        adaptive_reentry_atr_multiple=0.15,
        adaptive_confirm_bars_required=1,
        h_reentry_atr_multiple=0.30,
        h_confirm_bars_required=2,
        h_stop_confirm_bars=2,
        h_emergency_stop_loss=0.010,
        min_vol_percentile=0.20,
        max_vol_percentile=0.85,
        min_net_profit_rate=0.0020,
        slippage_rate=0.0001,
        time_decay_start_time=(13, 30),
        time_decay_late_time=(14, 0),
        time_decay_mid_factor=0.80,
        time_decay_late_factor=0.60,
        trailing_activation_atr=0.50,
        trailing_distance_atr=0.80,
    )
    return module


def opening_frame(closes, lows, highs):
    return pd.DataFrame({
        'open': [10.0] + closes[:-1],
        'high': highs,
        'low': lows,
        'close': closes,
    })


@pytest.mark.parametrize(
    (
        'group',
        'pressure',
        'opening',
        'breakdown',
        'adaptive',
        'strict_entry',
        'confirmed_stop',
        'emergency_only',
        'time_decay',
        'trailing',
        'context',
    ),
    [
        ('A', False, False, False, False, False, False, False, False, False, False),
        ('B', True, False, False, False, False, False, False, False, False, True),
        ('C', False, True, False, False, False, False, False, False, False, False),
        ('D', True, True, False, False, False, False, False, False, False, True),
        ('E', False, False, True, False, False, False, False, False, False, True),
        ('G', False, False, True, True, False, False, False, False, False, True),
        ('H', False, False, True, True, True, True, False, False, False, True),
        ('I', False, False, True, True, True, False, True, False, False, True),
        ('J', False, False, True, True, True, False, True, True, False, True),
        ('K', False, False, True, True, True, False, True, False, True, True),
    ],
)
def test_experiment_group_enables_only_its_filters(
    group,
    pressure,
    opening,
    breakdown,
    adaptive,
    strict_entry,
    confirmed_stop,
    emergency_only,
    time_decay,
    trailing,
    context,
):
    strategy = load_strategy()
    strategy.EXPERIMENT_GROUP = group
    strategy.g = types.SimpleNamespace()
    strategy.set_benchmark = lambda *args, **kwargs: None
    strategy.set_option = lambda *args, **kwargs: None
    strategy.set_order_cost = lambda *args, **kwargs: None
    strategy.set_slippage = lambda *args, **kwargs: None
    strategy.OrderCost = lambda **kwargs: kwargs
    strategy.PriceRelatedSlippage = lambda *args, **kwargs: None
    strategy.run_daily = lambda *args, **kwargs: None
    strategy.initialize(None)
    assert strategy.g.use_pressure_filter is pressure
    assert strategy.g.use_opening_path_filter is opening
    assert strategy.g.use_opening_breakdown_filter is breakdown
    assert strategy.g.use_adaptive_band_entry is adaptive
    assert strategy.g.use_strict_adaptive_entry is strict_entry
    assert strategy.g.use_confirmed_close_stop is confirmed_stop
    assert strategy.g.use_emergency_only_stop is emergency_only
    assert strategy.g.use_time_decay_target is time_decay
    assert strategy.g.use_trailing_take_profit is trailing
    assert strategy.g.use_pressure_context is context


def test_pressure_filter_requires_negative_return_and_high_volume():
    strategy = load_strategy()
    assert strategy.is_pressure_reversal_day(-0.01, 0.70)
    assert not strategy.is_pressure_reversal_day(0.01, 0.90)
    assert not strategy.is_pressure_reversal_day(-0.01, 0.69)


def test_e_only_rejects_one_way_breakdown():
    strategy = load_strategy()
    strategy.g.use_opening_path_filter = False
    strategy.g.use_opening_breakdown_filter = True
    no_early_dip = {'allowed': False, 'one_way_breakdown': False}
    breakdown = {'allowed': False, 'one_way_breakdown': True}
    assert strategy.get_opening_reject_reason(no_early_dip) is None
    assert strategy.get_opening_reject_reason(breakdown) == 'opening_breakdown'


def test_c_keeps_original_opening_path_gate():
    strategy = load_strategy()
    strategy.g.use_opening_path_filter = True
    strategy.g.use_opening_breakdown_filter = False
    no_early_dip = {'allowed': False, 'one_way_breakdown': False}
    assert strategy.get_opening_reject_reason(no_early_dip) == 'opening_path'


def test_missing_daily_filter_is_not_counted_as_neutral_open():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        mode_counts={
            'buy_first': 0,
            'sell_first': 0,
            'neutral': 0,
            'unavailable': 0,
        },
    )
    state = {}
    strategy.set_day_mode(types.SimpleNamespace(), state, None)
    assert state['open_regime'] == 'unavailable'
    assert strategy.g.mode_counts['neutral'] == 0
    assert strategy.g.mode_counts['unavailable'] == 1


def test_market_state_is_computed_once_per_engine_tick():
    strategy = load_strategy()
    stocks = ['000001.XSHE', '600000.XSHG']
    strategy.g = types.SimpleNamespace(
        engine_bar_calls=0,
        manage_start_time=(9, 35),
        manage_end_time=(14, 55),
        signal_interval_minutes=5,
        target_stocks=stocks,
        states={stock: {
            'pending': None,
            'round_trips': 0,
            'day_mode': 'buy_first',
            'candidate': None,
        } for stock in stocks},
        enable_t=True,
        open_time=(10, 0),
        last_open_time=(13, 15),
        max_round_trips=1,
        use_intraday_breakers=False,
        pause_today=False,
        daily_filters={stock: {} for stock in stocks},
        filter_counts={'market': 0},
        last_candidate_time=(13, 5),
    )
    position = types.SimpleNamespace(total_amount=1000)
    context = types.SimpleNamespace(
        current_dt=pd.Timestamp('2024-01-02 10:00'),
        portfolio=types.SimpleNamespace(
            positions={stock: position for stock in stocks},
        ),
    )
    market_data = {
        stock: types.SimpleNamespace(
            paused=False,
            is_st=False,
            last_price=10.0,
            high_limit=11.0,
            low_limit=9.0,
        )
        for stock in stocks
    }
    calls = {'market': 0}
    strategy.get_current_data = lambda: market_data
    strategy.get_stock_signal = lambda *args, **kwargs: None

    def market_state():
        calls['market'] += 1
        return 'flat'

    strategy.get_market_state = market_state
    strategy.run_t_engine(context)
    assert calls['market'] == 1


def test_percentile_rank_uses_latest_observation():
    strategy = load_strategy()
    values = pd.Series([1.0, 2.0, 4.0, 3.0])
    assert strategy.percentile_rank_last(values) == pytest.approx(0.75)
    assert strategy.percentile_rank_last(pd.Series([1.0, 2.0, float('nan')])) == 0.0


def test_adaptive_buy_levels_use_rolling_vwap_and_cost_floor():
    strategy = load_strategy()
    today = pd.DataFrame({
        'close': [10.00, 9.98, 9.96, 9.94],
        'volume': [100.0, 100.0, 100.0, 100.0],
        'money': [1000.0, 998.0, 996.0, 994.0],
    })
    result = strategy.calculate_adaptive_buy_levels(
        today,
        price=9.94,
        atr_abs=0.01,
        cost_rate=0.0015,
        daily_vol_percentile=0.20,
    )
    expected_anchor = today['money'].sum() / today['volume'].sum()
    expected_floor = 9.94 * (0.0015 + 0.0020)
    assert result['anchor'] == pytest.approx(expected_anchor)
    assert result['band_distance'] >= expected_floor
    assert result['lower_band'] == pytest.approx(
        result['anchor'] - result['band_distance']
    )


def test_adaptive_buy_levels_widen_with_daily_volatility():
    strategy = load_strategy()
    today = pd.DataFrame({
        'close': [10.0, 10.0, 10.0, 10.0],
        'volume': [100.0, 100.0, 100.0, 100.0],
        'money': [1000.0, 1000.0, 1000.0, 1000.0],
    })
    low_vol = strategy.calculate_adaptive_buy_levels(
        today,
        price=10.0,
        atr_abs=0.10,
        cost_rate=0.001,
        daily_vol_percentile=0.20,
    )
    high_vol = strategy.calculate_adaptive_buy_levels(
        today,
        price=10.0,
        atr_abs=0.10,
        cost_rate=0.001,
        daily_vol_percentile=0.85,
    )
    assert low_vol['atr_multiple'] == pytest.approx(0.70)
    assert high_vol['atr_multiple'] == pytest.approx(1.00)
    assert high_vol['band_distance'] > low_vol['band_distance']


def test_g_entry_requires_cross_then_reentry_without_new_low():
    strategy = load_strategy()
    filter_keys = [
        'daily', 'daily_buy', 'vol_high', 'vol_low', 'pressure', 'bars',
        'opening_path', 'opening_breakdown', 'range', 'candidate',
        'confirm', 'bottom', 'risk',
    ]
    strategy.g = types.SimpleNamespace(
        use_pressure_filter=False,
        use_opening_path_filter=False,
        use_opening_breakdown_filter=False,
        use_adaptive_band_entry=True,
        use_strict_adaptive_entry=False,
        use_confirmed_close_stop=False,
        use_emergency_only_stop=False,
        use_confirm_volume=False,
        min_vol_percentile=0.20,
        min_intraday_range=0.0,
        target_trade_value=15000,
        min_net_profit_rate=0.0020,
        slippage_rate=0.0001,
        max_intraday_decline=-0.015,
        max_candidate_bars=6,
        adaptive_confirm_bars_required=1,
        adaptive_reentry_atr_multiple=0.15,
        stop_atr_buffer=0.25,
        max_stop_loss=0.008,
        candidates_created=0,
        candidates_confirmed=0,
        candidates_expired=0,
        filter_counts={key: 0 for key in filter_keys},
        daily_filters={
            '000001.XSHE': {
                'buy_allowed': True,
                'high_vol_down': False,
                'vol_percentile': 0.50,
                'median_range': 0.0,
                'daily_atr_pct': 0.02,
            },
        },
    )
    dates = pd.date_range('2024-01-02 09:35', periods=24, freq='5min')
    closes = [10.00] * 20 + [9.99, 9.98, 9.97, 9.94]
    bars = pd.DataFrame({
        'date': dates,
        'open': closes,
        'high': [value + 0.02 for value in closes],
        'low': [value - 0.01 for value in closes],
        'close': closes,
        'volume': [100.0] * len(closes),
        'money': [value * 100.0 for value in closes],
    })
    current_bars = [bars]
    strategy.get_bars = lambda *args, **kwargs: current_bars[0]
    strategy.calculate_adaptive_buy_levels = lambda *args, **kwargs: {
        'anchor': 10.0,
        'lower_band': 9.95,
        'band_distance': 0.05,
        'ewma_vol': 0.0,
        'atr_multiple': 0.70,
    }
    state = {
        'candidate': None,
        'opening_path': None,
        'open_gap': 0.0,
        'adaptive_last_bar_time': None,
        'adaptive_was_below': False,
    }
    first = strategy.get_stock_signal(
        '000001.XSHE',
        9.94,
        pd.Timestamp('2024-01-02 11:30'),
        state,
    )
    assert first is None
    assert state['candidate'] is not None
    assert strategy.g.candidates_created == 1

    next_bar = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-02 11:35')],
        'open': [9.94],
        'high': [10.00],
        'low': [9.94],
        'close': [9.98],
        'volume': [100.0],
        'money': [998.0],
    })
    current_bars[0] = pd.concat([bars, next_bar], ignore_index=True)
    strategy.reset_tick_cache(pd.Timestamp('2024-01-02 11:35'))
    signal = strategy.get_stock_signal(
        '000001.XSHE',
        9.98,
        pd.Timestamp('2024-01-02 11:35'),
        state,
    )
    assert signal is not None
    assert signal['lower_band'] == pytest.approx(9.95)
    assert state['candidate'] is None
    assert strategy.g.candidates_confirmed == 1


def test_h_entry_waits_for_two_bars_and_previous_high_break():
    strategy = load_strategy()
    filter_keys = [
        'daily', 'daily_buy', 'vol_high', 'vol_low', 'pressure', 'bars',
        'opening_path', 'opening_breakdown', 'range', 'candidate',
        'confirm', 'bottom', 'risk',
    ]
    strategy.g = types.SimpleNamespace(
        use_pressure_filter=False,
        use_opening_path_filter=False,
        use_opening_breakdown_filter=False,
        use_adaptive_band_entry=True,
        use_strict_adaptive_entry=True,
        use_confirmed_close_stop=True,
        use_emergency_only_stop=False,
        use_confirm_volume=False,
        min_vol_percentile=0.20,
        min_intraday_range=0.0,
        target_trade_value=15000,
        min_net_profit_rate=0.0020,
        slippage_rate=0.0001,
        max_intraday_decline=-0.015,
        max_candidate_bars=6,
        adaptive_confirm_bars_required=1,
        adaptive_reentry_atr_multiple=0.15,
        h_confirm_bars_required=2,
        h_reentry_atr_multiple=0.30,
        stop_atr_buffer=0.25,
        max_stop_loss=0.008,
        candidates_created=1,
        candidates_confirmed=0,
        candidates_expired=0,
        filter_counts={key: 0 for key in filter_keys},
        daily_filters={
            '000001.XSHE': {
                'buy_allowed': True,
                'high_vol_down': False,
                'vol_percentile': 0.50,
                'median_range': 0.0,
                'daily_atr_pct': 0.02,
            },
        },
    )
    dates = pd.date_range('2024-01-02 09:35', periods=24, freq='5min')
    closes = [10.00] * 20 + [9.99, 9.98, 9.97, 9.94]
    bars = pd.DataFrame({
        'date': dates,
        'open': closes,
        'high': [value + 0.02 for value in closes],
        'low': [value - 0.01 for value in closes],
        'close': closes,
        'volume': [100.0] * len(closes),
        'money': [value * 100.0 for value in closes],
    })
    state = {
        'candidate': {
            'direction': 'buy_first',
            'low': 9.93,
            'atr_abs': 0.03,
            'last_bar_time': dates[-1],
            'observed_bars': 0,
            'bars_without_new_low': 0,
            'vwap': 10.0,
            'lower_band': 9.95,
        },
        'opening_path': None,
        'open_gap': 0.0,
        'adaptive_last_bar_time': dates[-1],
        'adaptive_was_below': True,
    }
    current_bars = [bars]
    strategy.get_bars = lambda *args, **kwargs: current_bars[0]
    strategy.calculate_adaptive_buy_levels = lambda *args, **kwargs: {
        'anchor': 10.0,
        'lower_band': 9.95,
        'band_distance': 0.05,
        'ewma_vol': 0.0,
        'atr_multiple': 0.70,
    }

    first_rebound = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-02 11:35')],
        'open': [9.94], 'high': [9.98], 'low': [9.94], 'close': [9.97],
        'volume': [100.0], 'money': [997.0],
    })
    current_bars[0] = pd.concat([bars, first_rebound], ignore_index=True)
    first_signal = strategy.get_stock_signal(
        '000001.XSHE', 9.97, pd.Timestamp('2024-01-02 11:35'), state
    )
    assert first_signal is None
    assert state['candidate']['bars_without_new_low'] == 1

    second_rebound = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-02 11:40')],
        'open': [9.97], 'high': [10.01], 'low': [9.96], 'close': [10.00],
        'volume': [100.0], 'money': [1000.0],
    })
    current_bars[0] = pd.concat(
        [current_bars[0], second_rebound],
        ignore_index=True,
    )
    strategy.reset_tick_cache(pd.Timestamp('2024-01-02 11:40'))
    signal = strategy.get_stock_signal(
        '000001.XSHE', 10.00, pd.Timestamp('2024-01-02 11:40'), state
    )
    assert signal is not None
    assert signal['reentry_price'] > 9.95
    assert strategy.g.candidates_confirmed == 1


def test_h_stop_requires_two_completed_closes_or_emergency_price():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(h_stop_confirm_bars=2)
    pending = {
        'stop_price': 10.00,
        'emergency_stop_price': 9.90,
        'stop_close_count': 0,
        'stop_last_bar_time': None,
    }
    current_bars = [pd.DataFrame({
        'date': [pd.Timestamp('2024-01-02 11:35')],
        'close': [9.99],
    })]
    strategy.get_bars = lambda *args, **kwargs: current_bars[0]
    assert strategy.get_confirmed_buy_stop_trigger(
        '000001.XSHE', 9.99, pending
    ) is None
    assert pending['stop_close_count'] == 1
    assert strategy.get_confirmed_buy_stop_trigger(
        '000001.XSHE', 9.98, pending
    ) is None
    assert pending['stop_close_count'] == 1

    current_bars[0] = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-02 11:40')],
        'close': [9.98],
    })
    strategy.reset_tick_cache(pd.Timestamp('2024-01-02 11:40'))
    assert strategy.get_confirmed_buy_stop_trigger(
        '000001.XSHE', 9.98, pending
    ) == 'confirmed_close'
    assert strategy.get_confirmed_buy_stop_trigger(
        '000001.XSHE', 9.89, pending
    ) == 'emergency'


def test_i_ignores_normal_stop_and_only_uses_emergency_price():
    strategy = load_strategy()
    pending = {
        'confirmed_close_stop': False,
        'emergency_only_stop': True,
        'stop_price': 10.00,
        'emergency_stop_price': 9.90,
    }
    assert strategy.get_buy_first_stop_trigger(
        '000001.XSHE', 9.95, pending
    ) is None
    assert strategy.get_buy_first_stop_trigger(
        '000001.XSHE', 9.89, pending
    ) == 'emergency'


def test_cost_edges_are_symmetric_when_actual_entry_costs_include_tax():
    strategy = load_strategy()
    trade_value = 15000.0
    buy_cost = strategy.estimate_order_cost(trade_value, is_sell=False)
    sell_cost = strategy.estimate_order_cost(trade_value, is_sell=True)
    buy_first_edge = strategy.get_minimum_target_edge(
        trade_value,
        buy_cost,
    )
    sell_first_edge = strategy.get_minimum_rebuy_edge(
        trade_value,
        sell_cost,
    )
    assert buy_cost == pytest.approx(5.0)
    assert sell_cost == pytest.approx(12.5)
    assert buy_first_edge == pytest.approx(sell_first_edge)
    assert strategy.estimate_round_cost_rate(trade_value) == pytest.approx(
        (buy_cost + sell_cost) / trade_value
        + strategy.g.slippage_rate * 2
    )


def test_affordable_buy_amount_uses_slippage_and_minimum_commission():
    strategy = load_strategy()
    amount = strategy.get_max_affordable_buy_amount(
        available_cash=10000.0,
        price=10.0,
        max_amount=1000,
    )
    assert amount == 900
    required = 10.0 * 1.0001 * 1000
    required += strategy.estimate_order_cost(required, is_sell=False)
    assert strategy.get_max_affordable_buy_amount(
        available_cash=required,
        price=10.0,
        max_amount=1000,
    ) == 1000


def test_j_time_decay_never_lowers_target_below_profit_floor():
    strategy = load_strategy()
    pending = {
        'entry_price': 10.0,
        'original_target_price': 10.10,
        'profit_floor_price': 10.07,
        'time_decay_target': True,
        'trailing_take_profit': False,
        'active_target_price': 10.10,
    }
    assert strategy.get_effective_buy_target((13, 29), pending) == pytest.approx(10.10)
    assert strategy.get_effective_buy_target((13, 30), pending) == pytest.approx(10.08)
    assert strategy.get_effective_buy_target((14, 0), pending) == pytest.approx(10.07)
    assert strategy.get_buy_first_take_profit_trigger(
        10.081,
        (13, 30),
        pending,
    ) == 'time_decay'


def test_k_trailing_take_profit_activates_and_locks_rebound():
    strategy = load_strategy()
    pending = {
        'entry_price': 10.0,
        'original_target_price': 10.20,
        'profit_floor_price': 10.03,
        'time_decay_target': False,
        'trailing_take_profit': True,
        'active_target_price': 10.20,
        'atr_abs': 0.10,
        'max_seen_price': 10.15,
        'trailing_active': False,
        'trailing_stop_price': 0.0,
    }
    assert strategy.get_buy_first_take_profit_trigger(
        10.08,
        (11, 0),
        pending,
    ) is None
    assert pending['trailing_active']
    assert pending['trailing_stop_price'] == pytest.approx(10.07)
    assert strategy.get_buy_first_take_profit_trigger(
        10.06,
        (11, 5),
        pending,
    ) == 'trailing'


def test_pending_excursions_use_completed_bar_high_and_low():
    strategy = load_strategy()
    pending = {
        'entry_price': 10.0,
        'max_seen_price': 10.0,
        'min_seen_price': 10.0,
    }
    strategy.get_bars = lambda *args, **kwargs: pd.DataFrame({
        'high': [10.20],
        'low': [9.80],
    })
    strategy.update_pending_excursions('000001.XSHE', 10.10, pending)
    assert pending['max_seen_price'] == pytest.approx(10.20)
    assert pending['min_seen_price'] == pytest.approx(9.80)


def test_finish_cycle_records_trade_excursion_and_exit_metadata():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        last_trade_mfe=0.0,
        last_trade_mae=0.0,
        last_holding_minutes=0.0,
        last_exit_minute=0.0,
        last_trade_profit=0.0,
        last_exit_reason_code=0.0,
        total_t_rounds=0,
        sum_win=0.0,
        sum_loss=0.0,
        win_rounds=0,
        loss_rounds=0,
        reason_stats={
            '止盈': {'rounds': 0, 'profit': 0.0},
            '止损': {'rounds': 0, 'profit': 0.0},
            '尾盘平仓': {'rounds': 0, 'profit': 0.0},
            '隔夜平仓': {'rounds': 0, 'profit': 0.0},
        },
        direction_stats={
            'buy_first': {'rounds': 0, 'wins': 0, 'profit': 0.0},
            'sell_first': {'rounds': 0, 'wins': 0, 'profit': 0.0},
        },
        cycle_profits=[],
        t_wins=0,
        t_losses=0,
        consecutive_losses=0,
        max_consecutive_losses=3,
        pause_days_remaining=0,
        consecutive_loss_pause_days=5,
        pause_today=False,
        rolling_cycle_count=10,
        rolling_loss_limit=-300.0,
        rolling_pause_days=10,
    )
    strategy.log = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    opened_at = pd.Timestamp('2024-01-02 10:00')
    state = {
        'pending': object(),
        'round_trips': 0,
        'last_action_dt': pd.Timestamp('2024-01-02 14:45'),
    }
    pending = {
        'exit_reason': '尾盘平仓',
        'realized_profit': 25.0,
        'direction': 'buy_first',
        'entry_price': 10.0,
        'max_seen_price': 10.30,
        'min_seen_price': 9.80,
        'opened_at': opened_at,
    }
    strategy.finish_cycle('000001.XSHE', state, pending)
    assert strategy.g.last_trade_mfe == pytest.approx(0.03)
    assert strategy.g.last_trade_mae == pytest.approx(-0.02)
    assert strategy.g.last_holding_minutes == pytest.approx(285.0)
    assert strategy.g.last_exit_minute == pytest.approx(885.0)
    assert strategy.g.last_trade_profit == pytest.approx(25.0)
    assert strategy.g.last_exit_reason_code == pytest.approx(3.0)


def test_opening_path_accepts_dip_with_recovery():
    strategy = load_strategy()
    frame = opening_frame(
        [9.96, 9.93, 9.91, 9.95, 9.98, 10.00],
        [9.97, 9.92, 9.90, 9.91, 9.94, 9.96],
        [10.02, 9.98, 9.95, 9.97, 10.00, 10.02],
    )
    result = strategy.evaluate_opening_path(frame, 0.02, 0.0)
    assert result['allowed']
    assert result['opening_drawdown'] == pytest.approx(-0.01)


def test_opening_path_rejects_day_without_early_dip():
    strategy = load_strategy()
    frame = opening_frame(
        [10.00, 10.01, 10.02, 10.01, 10.02, 10.03],
        [9.99, 10.00, 10.00, 10.00, 10.01, 10.01],
        [10.02, 10.03, 10.04, 10.03, 10.04, 10.05],
    )
    result = strategy.evaluate_opening_path(frame, 0.02, 0.0)
    assert not result['allowed']
    assert not result['one_way_breakdown']


def test_opening_path_accepts_low_gap_as_opening_pressure():
    strategy = load_strategy()
    frame = opening_frame(
        [10.00, 10.01, 10.02, 10.01, 10.02, 10.03],
        [9.99, 10.00, 10.00, 10.00, 10.01, 10.01],
        [10.02, 10.03, 10.04, 10.03, 10.04, 10.05],
    )
    result = strategy.evaluate_opening_path(frame, 0.02, -0.005)
    assert result['allowed']


def test_opening_path_rejects_one_way_breakdown():
    strategy = load_strategy()
    frame = opening_frame(
        [9.97, 9.94, 9.91, 9.88, 9.84, 9.80],
        [9.96, 9.93, 9.90, 9.87, 9.83, 9.79],
        [10.01, 9.98, 9.95, 9.92, 9.89, 9.85],
    )
    result = strategy.evaluate_opening_path(frame, 0.02, 0.0)
    assert not result['allowed']
    assert result['one_way_breakdown']
    assert result['efficiency'] >= 0.70
