import inspect

import pytest

from backend.api.screening import run_screening
from core.domain.screening._screener import normalize_qlib_scores
from core.domain.screening.macro import apply_sector_bias


@pytest.mark.p0
def test_screening_defaults():
    signature = inspect.signature(run_screening)
    assert signature.parameters["pool"].default.default == "stable_history_500d"
    assert signature.parameters["code_prefixes"].default.default == [
        "600", "601", "603", "605", "000", "001", "002", "003"]
    assert signature.parameters["neutralize_qlib"].default.default is False
    assert signature.parameters["macro_bias"].default.default is False
    assert signature.parameters["macro_regime"].default.default is False
    assert signature.parameters["sector_fusion"].default.default is False
    assert signature.parameters["decision_day_filter"].default.default is True
    assert signature.parameters["smooth_regime"].default.default is False
    assert signature.parameters["use_vol_switch"].default.default is False


@pytest.mark.p0
def test_percentile_mapping_remains_when_neutralization_is_off():
    scores = {f"{i:06d}": float(i) for i in range(10)}
    mapped = normalize_qlib_scores(scores, "2026-08-28", neutralize=False)

    assert mapped["000000"] == pytest.approx(-1.0)
    assert mapped["000009"] == pytest.approx(0.8)
    assert sorted(mapped, key=mapped.get) == list(scores)


@pytest.mark.p0
def test_score_processing_reports_actual_fallback(monkeypatch):
    from core.domain.factors import market_cap, neutralize

    scores = {f"{i:06d}": float(i) for i in range(10)}
    monkeypatch.setattr(market_cap, "get_market_cap_map", lambda codes, ref_date: {})

    def fail(*args, **kwargs):
        raise RuntimeError("neutralizer failed")

    monkeypatch.setattr(neutralize, "cross_sectional_neutralize", fail)

    result = normalize_qlib_scores(
        scores, "2026-08-28", industry_map={}, neutralize=True,
        return_details=True,
    )

    assert result["audit"]["neutralize_status"] == "fallback"
    assert result["audit"]["neutralize_qlib_applied"] is False
    assert result["audit"]["neutralize_error"] == "neutralizer failed"
    assert result["ledger"]["000009"]["raw_model_score"] == 9.0


@pytest.mark.p0
def test_macro_sector_bias_changes_only_matching_industry_scores():
    adjusted = apply_sector_bias(
        {"000001": 0.2, "000002": 0.4},
        {"银行": 0.15},
        {"000001": {"industry": "银行"}, "000002": {"industry": "电子"}},
    )

    assert adjusted == {"000001": pytest.approx(0.35), "000002": 0.4}


@pytest.mark.p0
def test_macro_sector_bias_matches_baostock_industry_names():
    adjusted = apply_sector_bias(
        {"000001": 0.2, "600036": 0.3, "600519": 0.4},
        {"银行": 0.15, "电子": 0.15},
        {
            "000001": {"industry": "J66 货币金融服务"},
            "600036": {"industry": "C39 计算机、通信和其他电子设备制造业"},
            "600519": {"industry": "C15 酒、饮料和精制茶制造业"},
        },
    )

    assert adjusted == {
        "000001": pytest.approx(0.35),
        "600036": pytest.approx(0.45),
        "600519": 0.4,
    }


@pytest.mark.parametrize("neutralize", [False, True])
def test_sector_fusion_forwards_neutralization_override(monkeypatch, neutralize):
    from unittest.mock import Mock
    from core.domain.screening._screener import BuySideScreener
    from core.domain.screening.macro import apply_sector_fusion

    fuse = Mock(return_value=({"000001": 0.5}, {}))
    monkeypatch.setattr(BuySideScreener, "_apply_sector_fusion", fuse)
    apply_sector_fusion({"000001": 0.2}, "2026-08-28", neutralize_qlib=neutralize)

    assert fuse.call_args.kwargs["neutralize_qlib"] is neutralize
