import pytest
from pydantic import ValidationError

from backend.api.backtest import DirectionBacktestRequest


@pytest.mark.unit
@pytest.mark.p0
@pytest.mark.parametrize("field,value", [
    ("top_n", 0),
    ("commission_rate", -0.0001),
    ("slippage_rate", -0.0001),
    ("stamp_duty", -0.0001),
    ("max_position_pct", 1.01),
    ("max_sector_pct", 0),
    ("dynamic_lookback", 1),
    ("dynamic_trail", 0),
    ("cooldown_stop_weeks", -1),
])
def test_direction_backtest_request_rejects_invalid_risk_parameters(field, value):
    with pytest.raises(ValidationError):
        DirectionBacktestRequest(start="2025-01-01", end="2025-12-31", **{field: value})


@pytest.mark.unit
@pytest.mark.p0
def test_direction_backtest_request_rejects_inverted_dates_and_caps():
    with pytest.raises(ValidationError):
        DirectionBacktestRequest(start="2025-12-31", end="2025-01-01")
    with pytest.raises(ValidationError):
        DirectionBacktestRequest(
            start="2025-01-01", end="2025-12-31",
            target_exposure=0.5, max_position_pct=0.6, max_sector_pct=0.7,
        )
    with pytest.raises(ValidationError):
        DirectionBacktestRequest(
            start="2025-01-01", end="2025-12-31",
            max_position_pct=0.4, max_sector_pct=0.3,
        )


@pytest.mark.unit
@pytest.mark.p0
def test_direction_backtest_request_rejects_invalid_stop_order():
    with pytest.raises(ValidationError):
        DirectionBacktestRequest(
            start="2025-01-01", end="2025-12-31",
            stop_loss_warn=-0.10, stop_loss_full=-0.05,
        )
    with pytest.raises(ValidationError):
        DirectionBacktestRequest(
            start="2025-01-01", end="2025-12-31",
            take_profit_half=0.30, take_profit_full=0.20,
        )
