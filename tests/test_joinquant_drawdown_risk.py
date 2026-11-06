import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.unit


def load_strategy():
    sys.modules.setdefault('jqdata', types.ModuleType('jqdata'))
    path = Path(__file__).parents[1] / 'research' / 'joinquant' / 'market_sector_gate_strategy.py'
    spec = importlib.util.spec_from_file_location('market_sector_gate_strategy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_profiles_keep_prior_controls_and_isolate_v9_crash_pause():
    strategy = load_strategy()

    assert strategy.EXPERIMENT_CONFIGS[6] == {
        'min_hold_days': 20,
        'rebalance_every_weeks': 2,
    }
    assert strategy.EXPERIMENT_CONFIGS[7] == {
        'min_hold_days': 0,
        'rebalance_every_weeks': 2,
    }
    assert strategy.EXPERIMENT_CONFIGS[8] == {
        'min_hold_days': 20,
        'rebalance_every_weeks': 2,
    }
    assert strategy.EXPERIMENT_CONFIGS[9] == {
        'min_hold_days': 20,
        'rebalance_every_weeks': 2,
    }
    assert strategy.RANK_VARIANTS == {
        6: 'legacy', 7: 'legacy', 8: 'robust', 9: 'legacy'}
    assert strategy.EXPERIMENT_VERSION == 9
    assert strategy.MIN_HOLD_DAYS == 20
    assert strategy.REBALANCE_EVERY_WEEKS == 2
    assert strategy.RANK_VARIANT == 'legacy'
    assert strategy.RANK_LOOKBACK_DAYS == 61
    assert strategy.RISK_VARIANT == 'crash_pause'
    assert strategy.USE_PORTFOLIO_RISK is True
    assert strategy.CRASH_DRAWDOWN_TRIGGER == pytest.approx(-0.18)
    assert strategy.CRASH_PAUSE_DAYS == 5
    assert strategy.FORCE_FULLY_INVESTED is True


def test_robust_ranking_ignores_a_recent_spike_and_prefers_persistent_momentum():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        rank_variant='robust',
        min_median_money=0,
    )
    index = range(121)
    persistent = [100 + i * 0.4 for i in index]
    recent_spike = [100.0] * 101 + [100 + i * 3 for i in range(20)]
    falling = [150 - i * 0.3 for i in index]
    closes = pd.DataFrame({
        'persistent': persistent,
        'recent_spike': recent_spike,
        'falling': falling,
    })
    money = pd.DataFrame({
        code: [20000000.0] * 20 for code in closes.columns
    })

    ranked, features = strategy.rank_candidates(
        list(closes.columns), closes, money)

    assert ranked[0] == 'persistent'
    assert features['persistent']['medium_momentum'] > 0
    assert features['recent_spike']['medium_momentum'] == pytest.approx(0.0)


def test_robust_ranking_does_not_count_missing_prices_as_down_days():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        rank_variant='robust',
        min_median_money=0,
    )
    prices = [100 + i for i in range(121)]
    prices[50] = float('nan')
    closes = pd.DataFrame({'stock': prices})
    money = pd.DataFrame({'stock': [20000000.0] * 20})

    ranked, features = strategy.rank_candidates(['stock'], closes, money)

    assert ranked == ['stock']
    assert features['stock']['positive_day_ratio'] == pytest.approx(1.0)
    assert features['stock']['downside_volatility'] == pytest.approx(0.0)


def test_robust_ranking_skips_an_incomplete_lookback_window():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        rank_variant='robust',
        min_median_money=0,
    )
    closes = pd.DataFrame({'stock': range(100, 220)})
    money = pd.DataFrame({'stock': [20000000.0] * 20})

    assert strategy.rank_candidates(['stock'], closes, money) == ([], {})


def test_fixed_risk_action_keeps_two_stage_behavior():
    strategy = load_strategy()
    assert strategy.fixed_risk_action(-0.05, False)[0] == 'half'
    assert strategy.fixed_risk_action(-0.10, False)[0] == 'full'
    assert strategy.fixed_risk_action(0.15, False)[0] == 'half'
    assert strategy.fixed_risk_action(0.30, False)[0] == 'full'
    assert strategy.fixed_risk_action(0.20, True)[0] is None


def test_hybrid_stop_has_hard_floor_and_only_ratchets_up():
    strategy = load_strategy()
    cost = 100.0
    stop0, _ = strategy.hybrid_stop_price(cost, 100.0)
    stop8, _ = strategy.hybrid_stop_price(cost, 108.0)
    stop15, _ = strategy.hybrid_stop_price(cost, 115.0)
    stop25, _ = strategy.hybrid_stop_price(cost, 125.0)
    assert stop0 == pytest.approx(92.0)
    assert stop8 >= 101.0
    assert stop15 >= 107.0
    assert stop25 >= 115.0
    assert stop0 < stop8 < stop15 < stop25


def test_portfolio_risk_scale_deleverages_immediately_and_recovers_with_hysteresis():
    strategy = load_strategy()
    assert strategy.portfolio_risk_scale(-0.061, 1.0) == 0.70
    assert strategy.portfolio_risk_scale(-0.101, 0.70) == 0.35
    assert strategy.portfolio_risk_scale(-0.141, 0.35) == 0.0
    assert strategy.portfolio_risk_scale(-0.05, 0.35) == 0.70
    assert strategy.portfolio_risk_scale(-0.02, 0.70) == 1.0
    assert strategy.portfolio_risk_scale(-0.04, 0.70) == 0.70


def test_hybrid_v2_stop_is_wider_and_trailing_starts_late():
    strategy = load_strategy()
    cost = 100.0
    assert strategy.hybrid_v2_stop_price(cost, 112.0, 4)[0] == pytest.approx(88.0)
    assert strategy.hybrid_v2_stop_price(cost, 111.99, 10)[0] == pytest.approx(88.0)
    assert strategy.hybrid_v2_stop_price(cost, 112.0, 10)[0] == pytest.approx(102.0)
    assert strategy.hybrid_v2_stop_price(cost, 120.0, 10)[0] == pytest.approx(108.0)
    assert strategy.hybrid_v2_stop_price(cost, 130.0, 10)[0] == pytest.approx(118.0)


def test_hybrid_v3_exits_failed_entries_and_locks_profit_tighter():
    strategy = load_strategy()
    cost = 100.0

    assert strategy.hybrid_v3_risk_action(cost, 94.0, 102.0, 4)[0] is None
    assert strategy.hybrid_v3_risk_action(cost, 94.0, 102.0, 5) == (
        'full', '失败入场退出')
    assert strategy.hybrid_v3_risk_action(cost, 89.0, 100.0, 20) == (
        'full', '固定灾难止损')

    assert strategy.hybrid_v3_risk_action(cost, 113.0, 120.0, 10)[0] is None
    assert strategy.hybrid_v3_risk_action(cost, 112.0, 120.0, 10) == (
        'full', '收紧阶梯移动止盈')
    assert strategy.hybrid_v3_risk_action(cost, 119.0, 130.0, 10) == (
        'full', '收紧阶梯移动止盈')


def test_atr_v4_stop_expands_with_volatility_and_trails_winners():
    strategy = load_strategy()
    cost = 100.0

    quiet_stop, quiet_reason = strategy.atr_v4_stop_price(
        cost, 100.0, 0.02, 20)
    volatile_stop, volatile_reason = strategy.atr_v4_stop_price(
        cost, 100.0, 0.05, 20)
    assert quiet_stop == pytest.approx(90.0)
    assert volatile_stop == pytest.approx(85.0)
    assert quiet_reason == volatile_reason == '动态ATR止损'

    trailing_stop, reason = strategy.atr_v4_stop_price(
        cost, 120.0, 0.02, 20)
    assert trailing_stop == pytest.approx(108.0)
    assert reason == '动态ATR移动止盈'


def test_atr_v5_only_activates_profit_exit_after_large_gain():
    strategy = load_strategy()
    cost = 100.0

    stop, reason, active = strategy.atr_v5_stop_price(
        cost, 129.0, 0.02, 20)
    assert stop == pytest.approx(90.0)
    assert reason == '动态ATR止损'
    assert active is False

    stop, reason, active = strategy.atr_v5_stop_price(
        cost, 130.0, 0.02, 20)
    assert stop == pytest.approx(110.5)
    assert reason == '动态ATR高位止盈'
    assert active is True

    _, _, active = strategy.atr_v5_stop_price(
        cost, 140.0, 0.08, 20)
    assert active is False
    _, _, active = strategy.atr_v5_stop_price(
        cost, 145.0, 0.08, 20)
    assert active is True


def test_atr_v4_rebalance_runs_every_two_weeks(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        rebalance_calls=0,
        rebalance_every_weeks=2,
        universe_index='000905.XSHG',
        top50=6,
        top_n=2,
        hold_buffer=2,
        use_context_gate=False,
        position_holding_days={},
        min_hold_days=20,
        last_audit={},
    )
    context = types.SimpleNamespace(
        previous_date=types.SimpleNamespace(),
        current_dt=types.SimpleNamespace(date=lambda: '2025-01-06'),
        portfolio=types.SimpleNamespace(positions={}),
    )
    ranked = ['A', 'B', 'C', 'D', 'E', 'F']
    calls = []
    monkeypatch.setattr(
        strategy, 'get_index_stocks', lambda *args, **kwargs: ranked, raising=False)
    monkeypatch.setattr(strategy, 'filter_universe', lambda stocks, previous_date: stocks)
    monkeypatch.setattr(strategy, 'history', lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, 'rank_candidates', lambda *args: (ranked, {}))
    monkeypatch.setattr(strategy, 'load_industries', lambda *args: {})
    monkeypatch.setattr(strategy, 'evaluate_market', lambda *args: {
        'severe_stress': False,
        'breadth_5d': 0.5,
        'trend_votes': 2,
        'es_change': 0.0,
    })
    monkeypatch.setattr(strategy, 'allocate_weights', lambda targets, industries: {
        code: 0.5 for code in targets
    })
    monkeypatch.setattr(
        strategy, 'execute_targets',
        lambda context, weights, features, preserve_codes=None: calls.append(list(weights)))
    monkeypatch.setattr(strategy, 'record', lambda **kwargs: None, raising=False)
    strategy.log = types.SimpleNamespace(info=lambda *args: None, warn=lambda *args: None)

    strategy.rebalance(context)
    strategy.rebalance(context)
    strategy.rebalance(context)

    assert calls == [['A', 'B'], ['A', 'B']]


def test_atr_v5_winner_is_not_resized_by_regular_rebalance(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(risk_scale=1.0, max_participation=0.05)
    position = types.SimpleNamespace(total_amount=100)
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(
        total_value=100000.0,
        positions={'A': position},
    ))
    current = {
        'A': types.SimpleNamespace(
            paused=False,
            last_price=10.0,
            low_limit=9.0,
            high_limit=11.0,
        ),
    }
    orders = []
    monkeypatch.setattr(
        strategy, 'get_current_data', lambda: current, raising=False)
    monkeypatch.setattr(
        strategy, 'order_target_value',
        lambda code, value: orders.append((code, value)),
        raising=False,
    )

    strategy.execute_targets(
        context,
        {'A': 0.18},
        {'A': {'median_money': 10000000}},
        preserve_codes=['A'],
    )

    assert orders == []


def test_force_full_weights_use_all_capital_and_ignore_sector_caps():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        force_fully_invested=True,
        full_investment_ratio=0.995,
    )
    targets = ['A', 'B', 'C', 'D', 'E']

    weights = strategy.allocate_weights(
        targets,
        {code: 'SAME' for code in targets},
    )

    assert set(weights) == set(targets)
    assert sum(weights.values()) == pytest.approx(0.995)
    assert all(weight == pytest.approx(0.199) for weight in weights.values())


def test_record_exposure_reports_actual_invested_ratio(monkeypatch):
    strategy = load_strategy()
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(
        total_value=100.0,
        positions={
            'A': types.SimpleNamespace(total_amount=100, value=60.0),
            'B': types.SimpleNamespace(total_amount=100, value=30.0),
        },
    ))
    captured = []
    monkeypatch.setattr(
        strategy, 'record', lambda **values: captured.append(values), raising=False)

    strategy.record_exposure(context)

    assert captured == [{
        'gross_exposure_pct': 90.0,
        'cash_pct': pytest.approx(10.0),
        'experiment_version': 9.0,
        'rank_version': 1.0,
        'rank_lookback_days': 61.0,
        'held_count': 2.0,
        'average_holding_days': 0.0,
        'rebalance_events': 0.0,
        'planned_add_events': 0.0,
        'planned_drop_events': 0.0,
        'crash_pause_active': 0.0,
        'crash_pause_days_remaining': 0.0,
        'crash_pause_triggers': 0.0,
    }]


def test_crash_pause_only_triggers_at_extreme_portfolio_drawdown(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='crash_pause',
        crash_lock_days=0,
        crash_trigger=-0.18,
        crash_pause_days=5,
        crash_resume_pending=False,
        pending_full_targets={'stale': True},
        risk_scale=1.0,
        portfolio_peak=100.0,
        risk_stats={
            'crash_pause_triggers': 0,
            'crash_exit_orders': 0,
        },
    )
    position = types.SimpleNamespace(total_amount=100)
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(
        positions={'A': position},
    ))
    current = {'A': object()}
    orders = []
    monkeypatch.setattr(
        strategy,
        'submit_risk_target',
        lambda code, held, data, ratio, reason: orders.append(
            (code, ratio, reason)) or True,
    )
    strategy.log = types.SimpleNamespace(warn=lambda *args: None)

    assert strategy.apply_crash_pause_risk(context, current, 82.1) is False
    assert orders == []
    assert strategy.apply_crash_pause_risk(context, current, 81.9) is True

    assert orders == [('A', 0.0, '极端组合回撤熔断')]
    assert strategy.g.crash_lock_days == 5
    assert strategy.g.risk_scale == 0.0
    assert strategy.g.pending_full_targets is None
    assert strategy.g.risk_stats['crash_pause_triggers'] == 1
    assert strategy.g.risk_stats['crash_exit_orders'] == 1


def test_crash_pause_keeps_five_trading_days_and_then_requests_resume():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_exits_today=set(),
        cooldowns={},
        crash_lock_days=5,
        crash_resume_pending=False,
        risk_variant='crash_pause',
        portfolio_peak=80.0,
        risk_scale=0.0,
        position_holding_days={},
        position_risk={},
    )
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={}))

    for _ in range(4):
        strategy.before_trading_start(context)
    assert strategy.g.crash_lock_days == 1
    assert strategy.g.crash_resume_pending is False

    strategy.before_trading_start(context)
    assert strategy.g.crash_lock_days == 0
    assert strategy.g.crash_resume_pending is True
    assert strategy.g.portfolio_peak == 0.0
    assert strategy.g.risk_scale == 0.0


def test_crash_resume_forces_a_fresh_rebalance(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='crash_pause',
        crash_resume_pending=True,
        crash_lock_days=0,
        pending_full_targets=None,
        position_holding_days={'OLD': 30},
        position_risk={'OLD': {'hold_days': 30}},
        risk_stats={'crash_resume_events': 0},
    )
    calls = []

    def fake_rebalance(context, force=False):
        calls.append(force)
        strategy.g.pending_full_targets = {'targets': True}

    monkeypatch.setattr(strategy, 'rebalance', fake_rebalance)

    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={}))
    strategy.resume_after_crash(context)

    assert calls == [True]
    assert strategy.g.crash_resume_pending is False
    assert strategy.g.risk_scale == 1.0
    assert strategy.g.position_holding_days == {}
    assert strategy.g.position_risk == {}
    assert strategy.g.risk_stats['crash_resume_events'] == 1


def test_crash_resume_waits_until_blocked_positions_are_really_closed(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='crash_pause',
        crash_resume_pending=True,
        crash_lock_days=0,
        pending_full_targets=None,
        risk_scale=0.0,
        risk_stats={'crash_resume_events': 0},
    )
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={
        'LIMIT_DOWN': types.SimpleNamespace(total_amount=100),
    }))
    calls = []
    monkeypatch.setattr(
        strategy, 'rebalance', lambda context, force=False: calls.append(force))

    strategy.resume_after_crash(context)

    assert calls == []
    assert strategy.g.crash_resume_pending is True
    assert strategy.g.risk_scale == 0.0


def test_crash_pause_retries_liquidation_after_countdown(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='crash_pause',
        crash_resume_pending=True,
        crash_lock_days=0,
        risk_scale=0.0,
        risk_stats={'crash_exit_orders': 0},
    )
    position = types.SimpleNamespace(total_amount=100)
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={
        'LIMIT_DOWN': position,
    }))
    submitted = []
    monkeypatch.setattr(
        strategy,
        'submit_risk_target',
        lambda code, held, data, ratio, reason: submitted.append(code) or True,
    )

    assert strategy.apply_crash_pause_risk(
        context, {'LIMIT_DOWN': object()}, signal_total_value=None) is True
    assert submitted == ['LIMIT_DOWN']
    assert strategy.g.risk_stats['crash_exit_orders'] == 1


def test_force_full_rebalance_sells_before_delayed_buy(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        force_fully_invested=True,
        pending_full_targets=None,
        risk_scale=1.0,
        max_participation=0.05,
    )
    old_position = types.SimpleNamespace(total_amount=100, value=1000.0)
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(
        total_value=100000.0,
        positions={'OLD': old_position},
    ))
    current = {
        'OLD': types.SimpleNamespace(
            paused=False, last_price=10.0, low_limit=9.0, high_limit=11.0),
        'NEW': types.SimpleNamespace(
            paused=False, last_price=10.0, low_limit=9.0, high_limit=11.0),
    }
    orders = []
    monkeypatch.setattr(
        strategy, 'get_current_data', lambda: current, raising=False)
    monkeypatch.setattr(
        strategy, 'order_target_value',
        lambda code, value: orders.append((code, value)),
        raising=False,
    )

    strategy.execute_targets(
        context,
        {'NEW': 0.995},
        {'NEW': {'median_money': 10000000}},
    )
    assert orders == [('OLD', 0)]
    assert strategy.g.pending_full_targets is not None

    context.portfolio.positions = {}
    strategy.execute_pending_full_targets(context)
    assert orders == [('OLD', 0), ('NEW', 99000.0)]
    assert strategy.g.pending_full_targets is None


def test_atr_v4_requires_two_closes_and_adds_stop_cooldown(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='atr_v4',
        risk_check_mode='daily_close',
        position_risk={},
        risk_exits_today=set(),
        cooldowns={},
        stop_cooldown_days=10,
        profit_cooldown_days=5,
        risk_stats={
            'atr_v4_full': 0,
            'atr_v4_hard_stop': 0,
            'atr_v4_profit_exit': 0,
        },
    )
    position = types.SimpleNamespace(total_amount=100, avg_cost=100.0, value=8900.0)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={'000001.XSHE': position}))
    current = {'000001.XSHE': types.SimpleNamespace(last_price=89.0)}
    submitted = []
    monkeypatch.setattr(
        strategy, 'submit_risk_target',
        lambda code, position, data, ratio, reason: submitted.append(reason) or True)

    strategy.apply_position_risk(
        context, current, {'000001.XSHE': 89.0}, {'000001.XSHE': 0.02})
    assert submitted == []

    strategy.apply_position_risk(
        context, current, {'000001.XSHE': 89.0}, {'000001.XSHE': 0.02})
    assert submitted == ['动态ATR止损']
    assert strategy.g.cooldowns == {'000001.XSHE': 10}
    assert strategy.g.risk_stats['atr_v4_hard_stop'] == 1


def test_hybrid_v2_portfolio_scale_uses_crossings_instead_of_repeated_caps():
    strategy = load_strategy()
    scale, zone, recovered = strategy.hybrid_v2_portfolio_scale(-0.081, 0, 1.0, 0.0, 0)
    assert (scale, zone, recovered) == (0.75, 1, False)

    scale, zone, recovered = strategy.hybrid_v2_portfolio_scale(-0.10, 1, scale, 0.0, 20)
    assert (scale, zone, recovered) == (0.75, 1, False)

    scale, zone, recovered = strategy.hybrid_v2_portfolio_scale(-0.121, 1, scale, 0.0, 0)
    assert (scale, zone, recovered) == (0.50, 2, False)

    scale, zone, recovered = strategy.hybrid_v2_portfolio_scale(-0.181, 2, scale, 0.0, 0)
    assert (scale, zone, recovered) == (0.25, 3, False)


def test_hybrid_v2_portfolio_scale_recovers_one_step_after_stability():
    strategy = load_strategy()
    assert strategy.hybrid_v2_portfolio_scale(-0.15, 3, 0.25, 0.03, 9) == (0.25, 2, False)
    assert strategy.hybrid_v2_portfolio_scale(-0.15, 2, 0.25, 0.03, 10) == (0.50, 2, True)
    assert strategy.hybrid_v2_portfolio_scale(-0.15, 2, 0.50, 0.0, 1) == (0.50, 2, False)
    assert strategy.hybrid_v2_portfolio_scale(-0.15, 2, 0.25, 0.029, 20) == (0.25, 2, False)


def test_hybrid_v2_keeps_global_peak_while_recovery_anchor_moves():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='hybrid_v2',
        portfolio_peak=0.0,
        portfolio_trough=0.0,
        recovery_anchor=0.0,
        portfolio_zone=0,
        days_since_portfolio_low=0,
        recovery_wait_days=10,
        recovery_step_return=0.03,
        risk_scale=1.0,
        cash_reserve=0.10,
        crash_lock_days=0,
        risk_stats={'portfolio_delever': 0, 'portfolio_recover': 0},
    )
    strategy.log = types.SimpleNamespace(warn=lambda *args: None)
    portfolio = types.SimpleNamespace(total_value=100.0, positions={})
    context = types.SimpleNamespace(portfolio=portfolio)

    strategy.apply_portfolio_risk(context, {}, set())
    portfolio.total_value = 91.0
    strategy.apply_portfolio_risk(context, {}, set())
    assert strategy.g.risk_scale == 0.75
    assert strategy.g.portfolio_peak == 100.0

    strategy.g.days_since_portfolio_low = 10
    portfolio.total_value = 93.73
    strategy.apply_portfolio_risk(context, {}, set())
    assert strategy.g.risk_scale == 1.0
    assert strategy.g.recovery_anchor == pytest.approx(93.73)
    assert strategy.g.portfolio_peak == 100.0


def test_hybrid_v3_tracks_drawdown_without_scaling_portfolio():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='hybrid_v3',
        use_portfolio_risk=False,
        portfolio_peak=0.0,
        portfolio_zone=0,
        risk_scale=1.0,
        risk_stats={'portfolio_delever': 0},
    )
    portfolio = types.SimpleNamespace(total_value=100.0, positions={})
    context = types.SimpleNamespace(portfolio=portfolio)

    strategy.apply_portfolio_risk(context, {}, set())
    portfolio.total_value = 80.0
    strategy.apply_portfolio_risk(context, {}, set())

    assert strategy.g.portfolio_peak == 100.0
    assert strategy.g.portfolio_zone == 3
    assert strategy.g.risk_scale == 1.0
    assert strategy.g.risk_stats['portfolio_delever'] == 0


def test_hybrid_v2_hard_stop_uses_one_completed_daily_close(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='hybrid_v2',
        risk_check_mode='daily_close',
        position_risk={},
        risk_exits_today=set(),
        cooldowns={},
        stop_cooldown_days=10,
        profit_cooldown_days=5,
        risk_stats={
            'hybrid_v2_full': 0,
            'hybrid_v2_hard_stop': 0,
            'hybrid_v2_profit_exit': 0,
        },
    )
    position = types.SimpleNamespace(total_amount=100, avg_cost=100.0, value=8700.0)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={'000001.XSHE': position}))
    current = {
        '000001.XSHE': types.SimpleNamespace(last_price=87.0),
    }
    submitted = []
    monkeypatch.setattr(
        strategy, 'submit_risk_target',
        lambda code, position, data, ratio, reason: submitted.append(code) or True)

    actioned = strategy.apply_position_risk(
        context, current, {'000001.XSHE': 95.0})
    assert actioned == set()
    assert submitted == []

    actioned = strategy.apply_position_risk(
        context, current, {'000001.XSHE': 87.0})
    assert actioned == {'000001.XSHE'}
    assert submitted == ['000001.XSHE']
    assert strategy.g.risk_exits_today == {'000001.XSHE'}
    assert strategy.g.risk_stats['hybrid_v2_hard_stop'] == 1
    assert strategy.g.risk_stats['hybrid_v2_profit_exit'] == 0


def test_hybrid_v3_failure_exit_updates_separate_counter(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='hybrid_v3',
        risk_check_mode='daily_close',
        position_risk={'000001.XSHE': {
            'peak': 102.0,
            'already_half': False,
            'loss_trigger_count': 0,
            'loss_trigger_key': None,
            'hold_days': 5,
            'amount': 100,
        }},
        risk_exits_today=set(),
        cooldowns={},
        stop_cooldown_days=10,
        profit_cooldown_days=5,
        risk_stats={
            'hybrid_v3_full': 0,
            'hybrid_v3_failure_exit': 0,
            'hybrid_v3_hard_stop': 0,
            'hybrid_v3_profit_exit': 0,
        },
    )
    position = types.SimpleNamespace(total_amount=100, avg_cost=100.0, value=9400.0)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={'000001.XSHE': position}))
    current = {'000001.XSHE': types.SimpleNamespace(last_price=94.0)}
    submitted = []
    monkeypatch.setattr(
        strategy, 'submit_risk_target',
        lambda code, position, data, ratio, reason: submitted.append(reason) or True)

    actioned = strategy.apply_position_risk(
        context, current, {'000001.XSHE': 94.0})

    assert actioned == {'000001.XSHE'}
    assert submitted == ['失败入场退出']
    assert strategy.g.risk_stats['hybrid_v3_full'] == 1
    assert strategy.g.risk_stats['hybrid_v3_failure_exit'] == 1
    assert strategy.g.cooldowns == {}


def test_daily_close_portfolio_value_uses_cash_and_completed_closes():
    strategy = load_strategy()
    positions = {
        '000001.XSHE': types.SimpleNamespace(total_amount=100),
        '600000.XSHG': types.SimpleNamespace(total_amount=200),
    }
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(
        cash=1000.0, positions=positions))
    prices = {'000001.XSHE': 10.0, '600000.XSHG': 20.0}

    assert strategy.daily_close_portfolio_value(context, prices) == 6000.0
    assert strategy.daily_close_portfolio_value(
        context, {'000001.XSHE': 10.0}) is None


def test_daily_stop_confirmation_preserves_fixed_two_stage_thresholds(monkeypatch):
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_variant='fixed',
        risk_check_mode='daily_close',
        position_risk={},
        risk_exits_today=set(),
        cooldowns={},
        stop_cooldown_days=10,
        profit_cooldown_days=5,
        risk_stats={'fixed_half': 0, 'fixed_full': 0},
    )
    position = types.SimpleNamespace(total_amount=100, avg_cost=100.0, value=9400.0)
    context = types.SimpleNamespace(
        portfolio=types.SimpleNamespace(positions={'000001.XSHE': position}))
    current = {'000001.XSHE': types.SimpleNamespace(last_price=94.0)}
    submitted = []
    monkeypatch.setattr(
        strategy, 'submit_risk_target',
        lambda code, position, data, ratio, reason: submitted.append(ratio) or True)

    strategy.apply_position_risk(context, current, {'000001.XSHE': 94.0})
    assert submitted == []
    strategy.apply_position_risk(context, current, {'000001.XSHE': 94.0})
    assert submitted == [0.5]

    strategy.apply_position_risk(context, current, {'000001.XSHE': 89.0})
    assert submitted == [0.5]
    strategy.apply_position_risk(context, current, {'000001.XSHE': 89.0})
    assert submitted == [0.5, 0.0]


def test_rebalance_protects_positions_held_less_than_ten_days(monkeypatch):
    strategy = load_strategy()
    held = types.SimpleNamespace(total_amount=100)
    strategy.g = types.SimpleNamespace(
        universe_index='000905.XSHG',
        top50=6,
        top_n=2,
        hold_buffer=2,
        use_context_gate=False,
        position_holding_days={'A': 5},
        min_hold_days=10,
        last_audit={},
    )
    context = types.SimpleNamespace(
        previous_date=types.SimpleNamespace(),
        current_dt=types.SimpleNamespace(date=lambda: '2025-01-06'),
        portfolio=types.SimpleNamespace(positions={'A': held}),
    )
    ranked = ['B', 'C', 'D', 'E', 'F', 'A']
    captured = []
    monkeypatch.setattr(
        strategy, 'get_index_stocks', lambda *args, **kwargs: ranked, raising=False)
    monkeypatch.setattr(strategy, 'filter_universe', lambda stocks, previous_date: stocks)
    monkeypatch.setattr(strategy, 'history', lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(strategy, 'rank_candidates', lambda *args: (ranked, {}))
    monkeypatch.setattr(strategy, 'load_industries', lambda *args: {})
    monkeypatch.setattr(strategy, 'evaluate_market', lambda *args: {
        'severe_stress': False,
        'breadth_5d': 0.5,
        'trend_votes': 2,
        'es_change': 0.0,
    })
    monkeypatch.setattr(strategy, 'allocate_weights', lambda targets, industries: {
        code: 0.5 for code in targets
    })
    monkeypatch.setattr(
        strategy, 'execute_targets',
        lambda context, weights, features, preserve_codes=None: captured.append(list(weights)))
    monkeypatch.setattr(strategy, 'record', lambda **kwargs: None, raising=False)
    strategy.log = types.SimpleNamespace(info=lambda *args: None, warn=lambda *args: None)

    strategy.rebalance(context)
    assert captured[-1] == ['A', 'B']
    assert strategy.g.last_audit['protected'] == ['A']

    strategy.g.position_holding_days['A'] = 10
    strategy.rebalance(context)
    assert captured[-1] == ['B', 'C']
    assert strategy.g.last_audit['protected'] == []

    strategy.g.position_holding_days['A'] = 5
    monkeypatch.setattr(
        strategy, 'filter_universe',
        lambda stocks, previous_date: [code for code in stocks if code != 'A'],
    )
    strategy.rebalance(context)
    assert captured[-1] == ['B', 'C']
    assert strategy.g.last_audit['protected'] == []

    strategy.g.risk_variant = 'atr_v5'
    strategy.g.position_risk = {'A': {'profit_active': True}}
    strategy.g.position_holding_days['A'] = 30
    monkeypatch.setattr(strategy, 'filter_universe', lambda stocks, previous_date: stocks)
    strategy.rebalance(context)
    assert captured[-1] == ['A', 'B']
    assert strategy.g.last_audit['winner_protected'] == ['A']


def test_before_trading_start_tracks_holding_days_and_removes_closed_positions():
    strategy = load_strategy()
    strategy.g = types.SimpleNamespace(
        risk_exits_today={'OLD'},
        cooldowns={},
        crash_lock_days=0,
        risk_variant='hybrid_v2',
        portfolio_trough=0.0,
        position_holding_days={'CLOSED': 4},
        position_risk={
            'A': {'hold_days': 2},
            'CLOSED': {'hold_days': 4},
        },
    )
    context = types.SimpleNamespace(portfolio=types.SimpleNamespace(positions={
        'A': types.SimpleNamespace(total_amount=100),
        'CLOSED': types.SimpleNamespace(total_amount=0),
    }))

    strategy.before_trading_start(context)
    assert strategy.g.risk_exits_today == set()
    assert strategy.g.position_holding_days == {'A': 1}
    assert strategy.g.position_risk == {'A': {'hold_days': 3}}

    strategy.before_trading_start(context)
    assert strategy.g.position_holding_days == {'A': 2}
    assert strategy.g.position_risk == {'A': {'hold_days': 4}}
