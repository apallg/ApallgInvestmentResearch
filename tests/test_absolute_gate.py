import pytest

from core.domain.screening.absolute_gate import apply_absolute_gate, fit_absolute_gate


pytestmark = [pytest.mark.unit, pytest.mark.p0]


def oof_rows():
    rows = []
    for day in range(24):
        for rank in range(8):
            rows.append({
                "decision_date": f"2024-01-{day + 1:02d}",
                "code": f"{rank:06d}",
                "raw_model_score": float(rank),
                "realized_tradable_return": (rank - 3.5) * 0.002,
                "market_state": "steady",
                "is_oof": True,
            })
    return rows


def test_absolute_gate_only_accepts_oof_and_independent_dates():
    rows = oof_rows()
    rows[0]["is_oof"] = False
    with pytest.raises(ValueError, match="OOF"):
        fit_absolute_gate(rows)
    with pytest.raises(ValueError, match="20"):
        fit_absolute_gate(oof_rows()[:80])


def test_absolute_gate_is_shadow_and_preserves_raw_order():
    model = fit_absolute_gate(oof_rows(), bins=8)
    raw_scores = {"000001": 3.0, "000002": 1.0, "000003": 2.0}
    result = apply_absolute_gate(model, raw_scores)
    assert model["shadow"] is True
    assert result["shadow"] is True
    assert list(raw_scores) == ["000001", "000002", "000003"]
    assert result["predictions"]["000001"]["changes_ranking"] is False
    assert "expected_tradable_return" in result["predictions"]["000001"]
