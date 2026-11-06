import pytest
from core.domain.portfolio.execution import (
    decide_execution,
    resolve_phase,
    ExecutionDecision,
    PHASE_OPEN, PHASE_MID, PHASE_CLOSE, PHASE_CLOSED,
    ACTION_LIMIT, ACTION_WAIT, ACTION_MARKET, ACTION_FILL,
)


class TestResolvePhase:

    def test_open_phase(self):
        assert resolve_phase("09:30") == PHASE_OPEN
        assert resolve_phase("09:59") == PHASE_OPEN

    def test_close_phase(self):
        assert resolve_phase("14:30") == PHASE_CLOSE
        assert resolve_phase("14:59") == PHASE_CLOSE

    def test_mid_phase(self):
        assert resolve_phase("10:00") == PHASE_MID
        assert resolve_phase("13:30") == PHASE_MID

    def test_invalid(self):
        assert resolve_phase("abc") == PHASE_CLOSED
        assert resolve_phase(None) == PHASE_CLOSED

    def test_non_trading_periods_are_closed(self):
        for value in ("09:29", "11:31", "12:30", "15:00", "15:01", "24:00"):
            assert resolve_phase(value) == PHASE_CLOSED


class TestSell:

    def test_stop_loss_market_any_phase(self):
        for phase in (PHASE_OPEN, PHASE_MID, PHASE_CLOSE):
            d = decide_execution("sell", urgency="stop_loss", phase=phase,
                                 current_price=10.0, vwap=10.1)
            assert d.action == ACTION_MARKET
            assert d.limit_price is None

    def test_take_profit_mid_limit_above(self):
        d = decide_execution("sell", urgency="take_profit", phase=PHASE_MID,
                             current_price=10.0)
        assert d.action == ACTION_LIMIT
        assert d.limit_price > 10.0

    def test_take_profit_close_fill_below(self):
        d = decide_execution("sell", urgency="take_profit", phase=PHASE_CLOSE,
                             current_price=10.0, must_fill=True)
        assert d.action == ACTION_FILL
        assert d.limit_price < 10.0

    def test_reduce_limit_at_vwap(self):
        d = decide_execution("reduce", phase=PHASE_MID,
                             current_price=10.0, vwap=9.95)
        assert d.action == ACTION_LIMIT
        assert d.limit_price == pytest.approx(9.95)


class TestBuyRiskVeto:

    def test_risk_warning_blocks_buy(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=10.0,
                             vwap=10.0, risk_warning="放量下跌")
        assert d.action == ACTION_WAIT

    def test_risk_warning_blocks_even_close(self):
        d = decide_execution("buy", phase=PHASE_CLOSE, current_price=10.0,
                             risk_warning="触及跌停价", must_fill=True)
        assert d.action == ACTION_WAIT

    def test_no_risk_allows_buy(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=10.0,
                             vwap=10.0, risk_warning=None)
        assert d.action != ACTION_WAIT


class TestBuyPhases:

    def test_open_high_open_no_chase(self):
        d = decide_execution("buy", phase=PHASE_OPEN, current_price=10.4,
                             vwap=10.35, prev_close=10.0, open_price=10.4)
        assert d.action == ACTION_LIMIT
        assert d.limit_price == pytest.approx(10.0)  # 挂前收, 不追高

    def test_open_flat_limits_at_support(self):
        d = decide_execution("buy", phase=PHASE_OPEN, current_price=10.05,
                             vwap=10.0, prev_close=10.0, open_price=10.02,
                             support_price=9.9)
        assert d.action == ACTION_LIMIT
        assert d.limit_price == pytest.approx(9.9)

    def test_close_must_fill_buy_above(self):
        d = decide_execution("buy", phase=PHASE_CLOSE, current_price=10.0,
                             vwap=9.9, must_fill=True)
        assert d.action == ACTION_FILL
        assert d.limit_price > 10.0

    def test_close_not_must_fill_waits(self):
        d = decide_execution("buy", phase=PHASE_CLOSE, current_price=10.0,
                             must_fill=False)
        assert d.action == ACTION_WAIT


class TestBuyMidScore:

    def test_good_score_vwap_anchor(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=10.2,
                             vwap=10.0, entry_score=70)
        assert d.action == ACTION_LIMIT
        assert d.limit_price == pytest.approx(10.0)

    def test_weak_score_vwap_below(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=10.0,
                             vwap=10.0, entry_score=20)
        assert d.action == ACTION_LIMIT
        assert d.limit_price < 10.0  # VWAP 下方

    def test_mid_score_min_of_vwap_and_support(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=10.1,
                             vwap=10.0, support_price=9.8, entry_score=40)
        assert d.limit_price == pytest.approx(9.8)


class TestEdge:

    def test_closed_phase_never_returns_execution_action(self):
        d = decide_execution("sell", urgency="stop_loss", phase=PHASE_CLOSED,
                             current_price=10.0)
        assert d.action == ACTION_WAIT
        assert "非交易时段" in d.reason

    def test_no_price_waits(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=0.0)
        assert d.action == ACTION_WAIT

    def test_missing_vwap_falls_back_to_price(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=10.0,
                             vwap=None, entry_score=80)
        assert d.action == ACTION_LIMIT
        assert d.limit_price == pytest.approx(10.0)

    def test_returns_decision_type(self):
        d = decide_execution("buy", phase=PHASE_MID, current_price=10.0, vwap=10.0)
        assert isinstance(d, ExecutionDecision)
