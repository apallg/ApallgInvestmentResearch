import pytest

from core.shared.data_contract import (
    data_envelope_allows_use,
    make_data_envelope,
    make_live_quote_envelope,
    revalidate_data_envelope,
)


def test_verified_envelope_requires_complete_row_coverage():
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:00+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        version="v1",
        unit={"price": "CNY/share"},
        price_basis="raw",
        coverage={"requested": 5, "accepted": 5, "missing": 0},
        status="verified",
    )

    assert envelope["status"] == "verified"
    assert envelope["missing_reason"] is None
    assert envelope["schema"] == "data-envelope-v2"
    assert envelope["validation_errors"] == []


@pytest.mark.parametrize("missing_reason", ["missing source field", ["stale_quote"], {"000001": "missing_price"}])
def test_verified_envelope_with_missing_reason_becomes_partial(missing_reason):
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:00+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        version="v1",
        coverage={"requested": 5, "accepted": 5, "missing": 0},
        status="verified",
        missing_reason=missing_reason,
    )

    assert envelope["status"] == "partial"
    assert envelope["missing_reason"] == missing_reason


@pytest.mark.parametrize("coverage", [
    None,
    {"requested": 5, "accepted": 4, "missing": 1},
    {"requested": 5, "accepted": 5, "missing": 1},
    {"requested": True, "accepted": True, "missing": 0},
    {"requested": 5, "accepted": 6, "missing": -1},
])
def test_verified_envelope_with_missing_or_invalid_coverage_becomes_partial(coverage):
    envelope = make_data_envelope(source="provider", coverage=coverage, status="verified")

    assert envelope["status"] == "partial"
    assert envelope["missing_reason"] == "coverage_incomplete"


@pytest.mark.parametrize("coverage", [
    {"requested": 5, "accepted": 4},
    {"requested": 5, "accepted": 6},
    {"requested": 5, "accepted": 5, "missing": 1},
    {"requested": True, "accepted": True},
])
def test_available_envelope_with_invalid_row_coverage_becomes_partial(coverage):
    envelope = make_data_envelope(source="provider", coverage=coverage, status="available")

    assert envelope["status"] == "partial"
    assert envelope["missing_reason"] == "coverage_incomplete"


def test_available_envelope_keeps_non_row_coverage_contract():
    envelope = make_data_envelope(
        source="provider", observed_at="2026-10-30", available_at="2026-10-30",
        as_of="2026-10-30", coverage={"members": 12, "returned": 12}, status="available",
    )

    assert envelope["status"] == "available"


@pytest.mark.parametrize("missing_reason", [
    "daily_quote_coverage_incomplete",
    ["source_availability_time_unknown"],
    {"000001": "missing_price"},
])
def test_available_envelope_with_missing_reason_becomes_partial(missing_reason):
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:00+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        version="v1",
        coverage={"requested": 5, "accepted": 5, "missing": 0},
        status="available",
        missing_reason=missing_reason,
    )

    assert envelope["status"] == "partial"
    assert envelope["missing_reason"] == missing_reason


@pytest.mark.parametrize("field", ["source", "observed_at", "available_at", "as_of"])
def test_available_envelope_requires_provenance_fields(field):
    values = {
        "source": "provider",
        "observed_at": "2026-10-30T15:00:00+08:00",
        "available_at": "2026-10-30T15:00:01+08:00",
        "as_of": "2026-10-30",
    }
    values[field] = None

    envelope = make_data_envelope(**values, status="available")

    assert envelope["status"] == "partial"
    assert f"{field}_missing" in envelope["validation_errors"]


def test_verified_envelope_requires_version_and_known_declared_units():
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:00+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        unit={"price": "unknown"},
        price_basis="unknown",
        coverage={"requested": 1, "accepted": 1},
        status="verified",
    )

    assert envelope["status"] == "partial"
    assert "version_missing" in envelope["validation_errors"]
    assert "unit_price_unknown" in envelope["validation_errors"]
    assert "price_basis_unknown" in envelope["validation_errors"]


def test_revalidation_catches_status_mutation_after_envelope_creation():
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:00+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        coverage={"requested": 1, "accepted": 1},
        status="partial",
    )
    envelope["status"] = "verified"

    checked = revalidate_data_envelope(envelope)

    assert checked["status"] == "partial"
    assert checked["missing_reason"] == "version_missing"


def test_live_use_gate_rejects_legacy_schema():
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:00+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        coverage={"requested": 1, "accepted": 1, "missing": 0},
        status="available",
    )
    envelope["schema"] = "data-envelope-v1"

    assert data_envelope_allows_use(envelope, "execution") is False


@pytest.mark.parametrize("use_case", ["typo_execution", "broker", "", [], {}])
def test_live_use_gate_rejects_unknown_use_case(use_case):
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:00+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        coverage={"requested": 1, "accepted": 1, "missing": 0},
        status="available",
    )

    assert data_envelope_allows_use(envelope, use_case) is False


def test_live_display_is_an_explicitly_supported_non_execution_use_case():
    envelope = make_data_envelope(status="unavailable")

    assert data_envelope_allows_use(envelope, "live_display") is True


def test_live_quote_envelope_uses_row_provenance_for_risk_gate():
    quote = {
        "price": 10.0,
        "source": "provider",
        "observed_at": "2026-10-30T10:00:00+08:00",
        "received_at": "2026-10-30T10:00:01+08:00",
        "session_date": "2026-10-30",
        "price_basis": "raw",
    }

    assert data_envelope_allows_use(make_live_quote_envelope(quote), "risk") is True
    quote.pop("received_at")
    assert data_envelope_allows_use(make_live_quote_envelope(quote), "risk") is False


@pytest.mark.parametrize("field", ["observed_at", "available_at", "as_of"])
def test_available_envelope_rejects_unparseable_time_fields(field):
    values = {
        "source": "provider",
        "observed_at": "2026-10-30T15:00:00+08:00",
        "available_at": "2026-10-30T15:00:01+08:00",
        "as_of": "2026-10-30",
    }
    values[field] = "not-a-time"

    envelope = make_data_envelope(**values, status="available")

    assert envelope["status"] == "partial"
    assert f"{field}_invalid" in envelope["validation_errors"]


def test_available_envelope_rejects_available_time_before_observation_time():
    envelope = make_data_envelope(
        source="provider",
        observed_at="2026-10-30T15:00:02+08:00",
        available_at="2026-10-30T15:00:01+08:00",
        as_of="2026-10-30",
        status="available",
    )

    assert envelope["status"] == "partial"
    assert "available_before_observed" in envelope["validation_errors"]
