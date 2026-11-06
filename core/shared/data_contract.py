from datetime import date, datetime, timezone


_UNKNOWN_VALUES = {"", "unknown", "unverified", "n/a", "na", "none", "null"}
SUPPORTED_DATA_USE_CASES = frozenset({"live_display", "execution", "risk"})


def _is_known(value):
    if isinstance(value, str):
        return value.strip().lower() not in _UNKNOWN_VALUES
    return callable(getattr(value, "isoformat", None))


def _parse_contract_time(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if "T" not in text and " " not in text:
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None
    if callable(getattr(value, "isoformat", None)):
        return _parse_contract_time(value.isoformat())
    return None


def _time_before(left, right):
    if isinstance(left, datetime) and isinstance(right, datetime):
        if (left.tzinfo is None) != (right.tzinfo is None):
            return None
        if left.tzinfo is not None:
            return left.astimezone(timezone.utc) < right.astimezone(timezone.utc)
        return left < right
    left_date = left.date() if isinstance(left, datetime) else left
    right_date = right.date() if isinstance(right, datetime) else right
    return left_date < right_date


def make_data_envelope(
    source=None,
    observed_at=None,
    available_at=None,
    as_of=None,
    version=None,
    unit=None,
    price_basis=None,
    coverage=None,
    status="unknown",
    missing_reason=None,
):
    coverage_data = dict(coverage or {})
    unit_data = dict(unit or {})
    requested = coverage_data.get("requested")
    accepted = coverage_data.get("accepted")
    has_row_coverage = "requested" in coverage_data or "accepted" in coverage_data
    valid_row_coverage = (
        type(requested) is int
        and type(accepted) is int
        and requested > 0
        and 0 <= accepted <= requested
        and (
            "missing" not in coverage_data
            or (
                type(coverage_data["missing"]) is int
                and coverage_data["missing"] == requested - accepted
            )
        )
    )
    errors = []
    incomplete_verified_coverage = status == "verified" and (
        not valid_row_coverage or accepted != requested
    )
    incomplete_available_coverage = status == "available" and has_row_coverage and (
        not valid_row_coverage or accepted != requested
    )
    if incomplete_verified_coverage or incomplete_available_coverage:
        errors.append("coverage_incomplete")
    requested_provenance = {
        "source": source,
        "observed_at": observed_at,
        "available_at": available_at,
        "as_of": as_of,
    }
    if status in ("available", "verified"):
        errors.extend(f"{name}_missing" for name, value in requested_provenance.items()
                      if not _is_known(value))
        for name, value in (("observed_at", observed_at), ("available_at", available_at), ("as_of", as_of)):
            if _is_known(value) and _parse_contract_time(value) is None:
                errors.append(f"{name}_invalid")
        parsed_observed_at = _parse_contract_time(observed_at)
        parsed_available_at = _parse_contract_time(available_at)
        if parsed_observed_at is not None and parsed_available_at is not None:
            available_before_observed = _time_before(parsed_available_at, parsed_observed_at)
            if available_before_observed is None:
                errors.append("time_timezone_mismatch")
            elif available_before_observed:
                errors.append("available_before_observed")
        errors.extend(f"unit_{name}_unknown" for name, value in unit_data.items()
                      if not _is_known(value))
        if status == "verified" and not _is_known(version):
            errors.append("version_missing")
        if "price" in unit_data and not _is_known(price_basis):
            errors.append("price_basis_unknown")
    if status in ("verified", "available") and missing_reason:
        errors.append("missing_reason_present")
    if errors:
        status = "partial"
        if missing_reason is None:
            missing_reason = errors[0]
    return {
        "schema": "data-envelope-v2",
        "source": source,
        "observed_at": observed_at,
        "available_at": available_at,
        "as_of": as_of,
        "version": version,
        "unit": unit_data,
        "price_basis": price_basis,
        "coverage": coverage_data,
        "status": status,
        "missing_reason": missing_reason,
        "validation_errors": list(dict.fromkeys(errors)),
    }


def make_live_quote_envelope(quote):
    if not isinstance(quote, dict):
        return make_data_envelope(
            coverage={"requested": 1, "accepted": 0, "missing": 1},
            status="unavailable",
            missing_reason="invalid_quote",
        )
    return make_data_envelope(
        source=quote.get("source"),
        observed_at=quote.get("observed_at"),
        available_at=quote.get("received_at"),
        as_of=quote.get("session_date"),
        unit={"price": "CNY/share"},
        price_basis=quote.get("price_basis"),
        coverage={"requested": 1, "accepted": 1, "missing": 0},
        status="available",
    )


def revalidate_data_envelope(envelope):
    if not isinstance(envelope, dict):
        return make_data_envelope(status="unavailable", missing_reason="invalid_envelope")
    previous_errors = envelope.get("validation_errors") or []
    if not isinstance(previous_errors, (list, tuple)):
        previous_errors = [previous_errors] if isinstance(previous_errors, str) else []
    normalized = make_data_envelope(
        source=envelope.get("source"),
        observed_at=envelope.get("observed_at"),
        available_at=envelope.get("available_at"),
        as_of=envelope.get("as_of"),
        version=envelope.get("version"),
        unit=envelope.get("unit"),
        price_basis=envelope.get("price_basis"),
        coverage=envelope.get("coverage"),
        status=envelope.get("status", "unknown"),
        missing_reason=envelope.get("missing_reason"),
    )
    normalized.update({
        key: value for key, value in envelope.items()
        if key not in normalized and key != "schema"
    })
    normalized["validation_errors"] = list(dict.fromkeys([
        *previous_errors,
        *(normalized.get("validation_errors") or []),
    ]))
    return normalized


def data_envelope_allows_use(envelope, use_case):
    if not isinstance(use_case, str) or use_case not in SUPPORTED_DATA_USE_CASES:
        return False
    if use_case == "live_display":
        return True
    if not isinstance(envelope, dict) or envelope.get("schema") != "data-envelope-v2":
        return False
    checked = revalidate_data_envelope(envelope)
    if checked.get("status") not in ("available", "verified"):
        return False
    if checked.get("missing_reason") or checked.get("validation_errors"):
        return False
    coverage = checked.get("coverage") or {}
    requested = coverage.get("requested")
    accepted = coverage.get("accepted")
    return (
        type(requested) is int
        and type(accepted) is int
        and requested > 0
        and accepted == requested
        and coverage.get("missing", 0) == 0
    )
