import datetime as dt
from types import SimpleNamespace

from backend.api import macro


def test_macro_overview_marks_missing_snapshot_unavailable(monkeypatch):
    monkeypatch.setattr("core.domain.macro.data.MacroDataAccess.get_macro_snapshot", lambda: None)

    result = macro._build()

    assert result["data_status"] == "unavailable"
    assert result["risk_score"] is None
    assert result["risk_level"] == "unknown"
    assert result["coverage"]["accepted"] == 0
    assert result["data_envelope"]["status"] == "unavailable"


def test_macro_overview_exposes_period_dates_units_and_unknown_release_times(monkeypatch):
    snapshot = SimpleNamespace(
        as_of_date="2026-10-30",
        pmi=50.2,
        lpr_5y=3.5,
        raw={
            "pmi": {"trade_date": dt.date(2026, 9, 1)},
            "lpr": {
                "trade_date": dt.date(2026, 10, 20),
                "source_observed_at": dt.datetime(2026, 10, 20, 9, 30),
            },
        },
    )

    class FakeAnalyzer:
        def _score_regime_bias(self, snap):
            return {"steady_up": 0.5}

        def calculate_macro_risk_score(self, snap):
            return {"score": 0.25, "level": "normal", "signals": []}

    monkeypatch.setattr("core.domain.macro.data.MacroDataAccess.get_macro_snapshot", lambda: snapshot)
    monkeypatch.setattr("core.domain.macro.analyzer.MacroAnalyzer", FakeAnalyzer)
    monkeypatch.setattr("core.domain.macro.data.MacroDataAccess.get_macro_change", lambda *args, **kwargs: None)

    result = macro._build()

    assert result["as_of_semantics"] == "query_cutoff_not_release_date"
    assert result["data_envelope"]["status"] == "partial"
    assert result["coverage"]["accepted"] == 2
    assert result["indicators"]["制造业PMI"]["source_date"] == "2026-09-01"
    assert result["indicators"]["制造业PMI"]["date_semantics"] == "report_period_label"
    assert result["indicators"]["制造业PMI"]["unit"] == "index_point"
    assert result["indicators"]["LPR5年"]["source_available_at"] is None
    assert result["indicators"]["LPR5年"]["source_observed_at"] == "2026-10-20T09:30:00"
    assert result["indicator_statuses"]["制造业PMI"]["source_availability_status"] == "unknown"
