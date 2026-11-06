from datetime import datetime
import math
from zoneinfo import ZoneInfo

_A_SHARE_TIMEZONE = ZoneInfo("Asia/Shanghai")

def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_A_SHARE_TIMEZONE)
    return parsed.astimezone(_A_SHARE_TIMEZONE)


def assess_quote(code, quote, now, max_age, expected_session_date=None):
    if not quote:
        return {
            "code": code,
            "usable": False,
            "freshness": "unavailable",
            "age_seconds": None,
            "missing_reason": "missing_quote",
        }
    price = quote.get("price") or quote.get("last_price") or quote.get("最新价") or 0
    try:
        price = float(price)
    except (TypeError, ValueError):
        price = 0

    if not math.isfinite(price) or price <= 0:
        return {
            "code": code,
            "usable": False,
            "freshness": "unavailable",
            "age_seconds": None,
            "missing_reason": "invalid_price",
        }

    observed_at = _parse_datetime(quote.get("observed_at"))
    if observed_at is None:
        return {
            "code": code,
            "usable": False,
            "freshness": "unavailable",
            "age_seconds": None,
            "missing_reason": "missing_or_invalid_observed_at",
        }

    now = now.replace(tzinfo=_A_SHARE_TIMEZONE) if now.tzinfo is None else now.astimezone(_A_SHARE_TIMEZONE)

    age = (now - observed_at).total_seconds()
    if age < -5:
        return {
            "code": code,
            "usable": False,
            "freshness": "unavailable",
            "age_seconds": round(age, 3),
            "missing_reason": "future_observed_at",
        }
    age = max(0.0, age)
    session_date = quote.get("session_date") or observed_at.date().isoformat()
    if expected_session_date and session_date != expected_session_date:
        return {
            "code": code,
            "usable": False,
            "freshness": "stale",
            "age_seconds": round(age, 3),
            "session_date": session_date,
            "missing_reason": "wrong_session_date",
        }

    usable = age <= max_age
    return {
        "code": code,
        "usable": usable,
        "freshness": "fresh" if usable else "stale",
        "age_seconds": round(age, 3),
        "session_date": session_date,
        "missing_reason": None if usable else "stale_quote",
    }


def assess_batch(quotes, codes, now, max_age, expected_session_date=None, require_complete=True):
    statuses = {}
    accepted = {}
    missing = []
    for code in codes:
        quote = quotes.get(code) or quotes.get(str(code))
        status = assess_quote(code, quote, now, max_age, expected_session_date)
        statuses[code] = status
        if status["usable"]:
            accepted[code] = quote
        else:
            missing.append(code)

    usable = bool(accepted) and (not require_complete or not missing)
    ages = [item["age_seconds"] for item in statuses.values() if item["age_seconds"] is not None]
    return {
        "usable": usable,
        "accepted": accepted if usable else {},
        "missing_codes": missing,
        "partial": bool(accepted) and bool(missing),
        "statuses": statuses,
        "coverage": {
            "requested": len(codes),
            "accepted": len(accepted),
            "missing": len(missing),
        },
        "max_age_seconds": round(max(ages), 3) if ages else None,
        "freshness": "fresh" if usable else ("degraded" if accepted else "unavailable"),
    }
