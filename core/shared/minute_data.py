import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation


_CERTIFIED_MINUTE_CONTRACTS = frozenset()


def parse_a_share_minute_time(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if parsed.tzinfo is not None and parsed.utcoffset() != timedelta(hours=8):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.second or parsed.microsecond:
        return None
    return parsed


def has_complete_minute_sequence(values):
    if not values or any(not isinstance(value, datetime) for value in values):
        return False
    if any(parse_a_share_minute_time(value) is None for value in values):
        return False
    if any(value.tzinfo != values[0].tzinfo for value in values):
        return False
    if values[0].tzinfo is not None:
        try:
            if any(value.utcoffset() != timedelta(hours=8) for value in values):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        if values != sorted(values) or len(set(values)) != len(values):
            return False
    except TypeError:
        return False
    first = values[0]
    last = values[-1]
    if any(value.date() != first.date() or value.second or value.microsecond for value in values):
        return False
    expected = []
    current = first
    while current <= last:
        current_time = current.time().replace(tzinfo=None)
        if time(9, 30) <= current_time <= time(11, 30) or time(13, 0) <= current_time <= time(15, 0):
            expected.append(current)
        elif time(11, 30) < current_time < time(13, 0):
            pass
        else:
            return False
        current += timedelta(minutes=1)
    return values == expected


def normalize_trade_quantity(volume, amount):
    try:
        volume = Decimal(str(volume))
        amount = Decimal(str(amount))
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return None
    if (
        not volume.is_finite()
        or volume < 0
        or volume != volume.to_integral_value()
        or volume > Decimal("9223372036854775807")
        or not amount.is_finite()
        or amount < 0
        or ((volume == 0) != (amount == 0))
    ):
        return None
    amount_value = float(amount)
    if not math.isfinite(amount_value):
        return None
    return int(volume), amount_value


def has_verified_minute_semantics(envelope):
    if not isinstance(envelope, dict) or envelope.get("schema") not in ("data-envelope-v1", "data-envelope-v2"):
        return False
    if envelope.get("status") != "verified" or envelope.get("missing_reason") or envelope.get("validation_errors"):
        return False
    coverage = envelope.get("coverage")
    if not isinstance(coverage, dict):
        return False
    requested = coverage.get("requested")
    accepted = coverage.get("accepted")
    if type(requested) is not int or type(accepted) is not int or requested <= 0 or accepted != requested:
        return False
    if "missing" in coverage and (type(coverage["missing"]) is not int or coverage["missing"] != 0):
        return False
    if any(not isinstance(envelope.get(key), str) or not envelope[key].strip()
           for key in ("source", "observed_at", "available_at", "as_of", "version",
                       "semantic_certification_id")):
        return False
    if (envelope["source"].strip(), envelope["version"].strip(),
            envelope["semantic_certification_id"].strip()) not in _CERTIFIED_MINUTE_CONTRACTS:
        return False
    try:
        observed_at = datetime.fromisoformat(envelope["observed_at"].replace("Z", "+00:00"))
        available_at = datetime.fromisoformat(envelope["available_at"].replace("Z", "+00:00"))
        as_of = envelope["as_of"]
        if "T" in as_of or " " in as_of:
            datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        else:
            date.fromisoformat(as_of)
        if (observed_at.tzinfo is None) != (available_at.tzinfo is None):
            return False
        if available_at < observed_at:
            return False
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    if envelope.get("price_basis") not in ("raw", "qfq", "hfq"):
        return False
    unit = envelope.get("unit")
    if not isinstance(unit, dict) or any(unit.get(key) != expected for key, expected in {
        "price": "CNY/share",
        "volume": "share",
        "amount": "CNY",
    }.items()):
        return False
    semantics = str(envelope.get("source_quantity_semantics") or "").strip().lower()
    return semantics == "incremental-per-minute"
