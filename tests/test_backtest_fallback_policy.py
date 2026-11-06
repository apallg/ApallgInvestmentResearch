import pytest

from backend.api.backtest import DirectionBacktestRequest
from backend.service.backtest import BacktestService


class _Runner:
    def __init__(self, **kwargs):
        pass

    def run(self, **kwargs):
        return {"runner_params": kwargs}


@pytest.mark.unit
@pytest.mark.p0
def test_conditional_switch_fails_closed_when_environment_is_missing(monkeypatch):
    monkeypatch.setattr("core.domain.backtest.engine.DirectionBacktestRunner", _Runner)
    monkeypatch.setattr(
        "backend.service.market_data.MarketDataService.get_conditional_market_env",
        lambda **kwargs: {},
    )

    request = DirectionBacktestRequest(
        start="2025-01-01",
        end="2025-02-01",
        use_vol_switch=True,
        use_conditional_switch=True,
    )
    with pytest.raises(RuntimeError, match="条件切换环境特征为空"):
        BacktestService.run_direction(request)


@pytest.mark.unit
@pytest.mark.p0
def test_explicit_binary_fallback_is_recorded(monkeypatch):
    monkeypatch.setattr("core.domain.backtest.engine.DirectionBacktestRunner", _Runner)
    monkeypatch.setattr(
        "backend.service.market_data.MarketDataService.get_conditional_market_env",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("source unavailable")),
    )

    result = BacktestService.run_direction(DirectionBacktestRequest(
        start="2025-01-01",
        end="2025-02-01",
        use_vol_switch=True,
        use_conditional_switch=True,
        conditional_fallback_policy="binary",
    ))

    assert result["runner_params"]["conditional_market_env"] == {}
    assert result["conditional_switch"] == {
        "requested": "conditional",
        "effective": "binary",
        "fallback_applied": True,
        "fallback_reason": "source unavailable",
    }
