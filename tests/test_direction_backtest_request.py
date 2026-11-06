import pytest
from pydantic import ValidationError

from backend.api.backtest import DirectionBacktestRequest
from backend.service.backtest import BacktestService


BASE = {"start": "2026-01-01", "end": "2026-03-31"}


@pytest.mark.p0
def test_atr_request_defaults():
    req = DirectionBacktestRequest(**BASE, stop_source="atr")
    assert req.atr_period == 14
    assert req.atr_mult == 2.0


@pytest.mark.p0
def test_fixed_model_request_is_supported():
    req = DirectionBacktestRequest(**BASE, fixed_model=["exp", "run"])
    assert req.fixed_model == ["exp", "run"]


@pytest.mark.p0
def test_direction_backtest_defaults():
    req = DirectionBacktestRequest(**BASE)
    assert req.code_prefixes == [
        "600", "601", "603", "605", "000", "001", "002", "003"]
    assert req.neutralize_qlib is False
    assert req.macro_bias is False
    assert req.macro_regime is False
    assert req.sector_fusion is False
    assert req.decision_day_filter is True
    assert req.use_hmm is False
    assert req.use_crf is False
    assert req.use_vol_switch is False
    assert req.full_position is True
    assert req.use_timing is False
    assert req.cash_yield is False
    assert req.directions == ["trend_momentum"]
    assert req.sizing == "equal"
    assert req.target_exposure == pytest.approx(0.95)
    assert req.max_one_way_turnover == pytest.approx(0.30)
    assert req.stop_source == "off"


@pytest.mark.p0
def test_score_overlays_can_be_overridden():
    req = DirectionBacktestRequest(
        **BASE, code_prefixes=None, neutralize_qlib=False, macro_bias=True,
        macro_regime=False, decision_day_filter=False,
        sector_fusion=False, use_hmm=False, use_crf=True, use_vol_switch=True,
        full_position=False, stop_source="fixed")
    assert req.code_prefixes is None
    assert req.neutralize_qlib is False
    assert req.macro_bias is True
    assert req.macro_regime is False
    assert req.sector_fusion is False
    assert req.decision_day_filter is False
    assert req.use_hmm is False
    assert req.use_crf is True
    assert req.use_vol_switch is True
    assert req.full_position is False
    assert req.stop_source == "fixed"


@pytest.mark.p0
@pytest.mark.parametrize("values", [
    {"stop_source": "unknown"},
    {"atr_period": 1},
    {"atr_mult": 0},
])
def test_invalid_atr_request_rejected(values):
    with pytest.raises(ValidationError):
        DirectionBacktestRequest(**BASE, **values)


@pytest.mark.p0
def test_profile_injects_locked_backtest_contract(monkeypatch):
    profile = {
        "profile_id": "baseline", "version": 1, "status": "challenger", "fingerprint": "f" * 64,
        "config": {
            "model": {
                "refs": [{"experiment_id": "exp", "run_id": "production"}],
                "oos_schedule": [{
                    "window_start": "2026-01-01", "window_end": "2026-03-31",
                    "experiment_id": "exp", "run_id": "fold-1", "artifact_fingerprint": "a" * 64,
                }],
            },
            "windows": {"oos_start": "2026-01-01", "oos_end": "2026-03-31"},
        },
    }
    monkeypatch.setattr(
        "backend.service.strategy_profile.StrategyProfileService.resolve",
        lambda *args, **kwargs: profile,
    )
    monkeypatch.setattr(
        "core.adapters.qlib.model_selection.get_model_for_factor_set",
        lambda *args, **kwargs: {"artifact_fingerprint": "a" * 64},
    )
    req = DirectionBacktestRequest(**BASE, profile_id="baseline", profile_version=1)
    params = BacktestService.validate_direction_request(req)
    assert params["pool"] == "stable_history_500d"
    assert params["top_n"] == 5
    assert params["fixed_model"] == ("exp", "fold-1")
    assert params["_strategy_profile_model"]["run_id"] == "fold-1"
    assert params["sizing"] == "equal"
    assert params["target_exposure"] == pytest.approx(0.95)
    assert params["max_one_way_turnover"] == pytest.approx(0.30)
    assert params["use_hmm"] is False
    assert params["neutralize_qlib"] is False


@pytest.mark.p0
def test_profile_rejects_explicit_locked_override(monkeypatch):
    profile = {
        "profile_id": "baseline", "version": 1, "status": "challenger", "fingerprint": "f" * 64,
        "config": {
            "model": {
                "refs": [{"experiment_id": "exp", "run_id": "production"}],
                "oos_schedule": [{
                    "window_start": "2026-01-01", "window_end": "2026-03-31",
                    "experiment_id": "exp", "run_id": "fold-1", "artifact_fingerprint": "a" * 64,
                }],
            },
            "windows": {"oos_start": "2026-01-01", "oos_end": "2026-03-31"},
        },
    }
    monkeypatch.setattr(
        "backend.service.strategy_profile.StrategyProfileService.resolve",
        lambda *args, **kwargs: profile,
    )
    req = DirectionBacktestRequest(**BASE, profile_id="baseline", profile_version=1, top_n=10)
    with pytest.raises(ValueError, match="Profile"):
        BacktestService.validate_direction_request(req)


@pytest.mark.p0
def test_profile_rejects_backtest_window_without_frozen_fold_model(monkeypatch):
    profile = {
        "profile_id": "baseline", "version": 1, "status": "challenger", "fingerprint": "f" * 64,
        "config": {
            "model": {"refs": [{"experiment_id": "exp", "run_id": "production"}], "oos_schedule": []},
            "windows": {"oos_start": "2026-01-01", "oos_end": "2026-12-31"},
        },
    }
    monkeypatch.setattr(
        "backend.service.strategy_profile.StrategyProfileService.resolve",
        lambda *args, **kwargs: profile,
    )
    req = DirectionBacktestRequest(**BASE, profile_id="baseline", profile_version=1)
    with pytest.raises(ValueError, match="walk-forward"):
        BacktestService.validate_direction_request(req)
