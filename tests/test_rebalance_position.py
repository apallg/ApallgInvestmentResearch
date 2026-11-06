import pytest

from backend.service.rebalance import _resolve_position_ratio, _round_buy_order


pytestmark = [pytest.mark.p0, pytest.mark.unit]


def test_unified_regime_position_overrides_timing_fallback():
    ratio, source = _resolve_position_ratio(
        None, 1.0, {"details": {"position_ratio": 0.25}},
    )

    assert ratio == pytest.approx(0.25)
    assert source == "unified_regime"


def test_position_ratio_is_capped_at_ninety_five_percent():
    ratio, source = _resolve_position_ratio(None, 1.0, None)

    assert ratio == pytest.approx(0.95)
    assert source == "timing_fallback"


def test_explicit_position_remains_highest_priority():
    ratio, source = _resolve_position_ratio(
        0.4, 0.8, {"position": 0.25},
    )

    assert ratio == pytest.approx(0.4)
    assert source == "explicit"


def test_buy_order_rounds_down_to_board_lot_before_capacity_check():
    shares, amount = _round_buy_order(25_050, 25)

    assert shares == 1000
    assert amount == pytest.approx(25_000)


def test_buy_order_below_one_lot_is_not_forced_to_one_lot():
    shares, amount = _round_buy_order(2_000, 25)

    assert shares == 0
    assert amount == 0
