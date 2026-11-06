"""宏观经济数据 API"""
import logging
from datetime import date, datetime

import anyio
from fastapi import APIRouter
from core.shared.data_contract import make_data_envelope

router = APIRouter()
logger = logging.getLogger(__name__)

_cache = {}
_cache_time = {}


def _cached(key, ttl, factory):
    import time
    now = time.time()
    if key in _cache and (now - _cache_time.get(key, 0)) < ttl:
        return _cache[key]
    from core.shared.infrastructure import redis_client as rds
    rk = f"macro:{key}"
    raw = rds._get(rk)
    if raw is not None:
        _cache[key] = raw
        _cache_time[key] = now
        return raw
    data = factory()
    rds._set(rk, data, expire=ttl)
    _cache[key] = data
    _cache_time[key] = now
    return data


def _build():
    from core.domain.macro.data import MacroDataAccess
    from core.domain.macro.analyzer import MacroAnalyzer
    from core.shared.trading_time import A_SHARE_TIMEZONE

    indicator_specs = [
        ("Shibor隔夜", "shibor_on", "shibor", "macro_shibor", "observation_date", "%"),
        ("Shibor1年", "shibor_1y", "shibor", "macro_shibor", "observation_date", "%"),
        ("Shibor3月", "shibor_3m", "shibor", "macro_shibor", "observation_date", "%"),
        ("期限利差", "spread_1y_on", "shibor", "macro_shibor", "observation_date", "percentage_point"),
        ("LPR1年", "lpr_1y", "lpr", "macro_lpr", "rate_record_date", "%"),
        ("LPR5年", "lpr_5y", "lpr", "macro_lpr", "rate_record_date", "%"),
        ("M1同比", "m1_yoy", "money", "macro_money_supply", "report_period_label", "%"),
        ("M2同比", "m2_yoy", "money", "macro_money_supply", "report_period_label", "%"),
        ("M1-M2剪刀差", "m1_m2_scissors", "money", "macro_money_supply", "report_period_label", "percentage_point"),
        ("CPI同比", "cpi_yoy", "cpi", "macro_cpi", "report_period_label", "%"),
        ("核心CPI", "core_cpi_yoy", "cpi", "macro_cpi", "report_period_label", "%"),
        ("PPI同比", "ppi_yoy", "ppi", "macro_ppi", "report_period_label", "%"),
        ("PPI生产资料", "ppi_mp_yoy", "ppi", "macro_ppi", "report_period_label", "%"),
        ("CPI-PPI价差", "cpi_ppi_spread", "cpi+ppi", "macro_cpi+macro_ppi", "report_period_label", "percentage_point"),
        ("制造业PMI", "pmi", "pmi", "macro_pmi", "report_period_label", "index_point"),
        ("PMI新订单", "pmi_new_order", "pmi", "macro_pmi", "report_period_label", "index_point"),
        ("非制造业PMI", "pmi_non_mfg", "pmi", "macro_pmi", "report_period_label", "index_point"),
    ]
    generated_at = datetime.now(A_SHARE_TIMEZONE).isoformat(timespec="seconds")
    snap = MacroDataAccess.get_macro_snapshot()
    if snap is None:
        cutoff = date.today().isoformat()
        missing = {name: "no_macro_snapshot" for name, *_ in indicator_specs}
        coverage = {"requested": len(indicator_specs), "accepted": 0, "missing": len(indicator_specs)}
        return {
            "as_of": cutoff,
            "as_of_semantics": "query_cutoff_not_release_date",
            "generated_at": generated_at,
            "data_status": "unavailable",
            "coverage": coverage,
            "indicator_statuses": {
                name: {"status": "unavailable", "missing_reason": "no_macro_snapshot"}
                for name, *_ in indicator_specs
            },
            "data_envelope": make_data_envelope(
                source="macro_* tables",
                as_of=cutoff,
                unit={},
                coverage=coverage,
                status="unavailable",
                missing_reason=missing,
            ),
            "indicators": {},
            "regime_bias": {},
            "risk_score": None,
            "risk_score_status": "unavailable",
            "risk_level": "unknown",
            "risk_signals": [],
            "style": {},
        }

    analyzer = MacroAnalyzer()
    bias = analyzer._score_regime_bias(snap)
    risk = analyzer.calculate_macro_risk_score(snap)

    indicators = {}
    indicator_statuses = {}
    for name, key, raw_key, source_table, date_semantics, unit in indicator_specs:
        v = getattr(snap, key, None)
        source_records = {
            source_key: snap.raw.get(source_key) or {}
            for source_key in raw_key.split("+")
        }
        source_dates = {}
        source_available_at = {}
        source_observed_at = {}
        for source_key, record in source_records.items():
            source_date = record.get("trade_date")
            source_dates[source_key] = (
                source_date.isoformat() if hasattr(source_date, "isoformat")
                else (str(source_date) if source_date else None)
            )
            available_at = record.get("published_at") or record.get("available_at")
            observed_at = record.get("source_observed_at")
            source_available_at[source_key] = (
                available_at.isoformat(timespec="seconds") if isinstance(available_at, datetime)
                else (available_at.isoformat() if hasattr(available_at, "isoformat")
                else (str(available_at) if available_at else None)
                )
            )
            source_observed_at[source_key] = (
                observed_at.isoformat(timespec="seconds") if isinstance(observed_at, datetime)
                else (observed_at.isoformat() if hasattr(observed_at, "isoformat")
                else (str(observed_at) if observed_at else None)
                )
            )
        source_date = next(iter(source_dates.values())) if len(source_dates) == 1 else source_dates
        source_available_at_value = next(iter(source_available_at.values())) if len(source_available_at) == 1 else source_available_at
        source_observed_at_value = next(iter(source_observed_at.values())) if len(source_observed_at) == 1 else source_observed_at
        source_availability_status = "known" if all(source_available_at.values()) else "unknown"
        indicator_statuses[name] = {
            "status": "available" if v is not None else "unavailable",
            "source_table": source_table,
            "source_date_field": "trade_date",
            "source_date": source_date,
            "date_semantics": date_semantics,
            "source_available_at": source_available_at_value,
            "source_observed_at": source_observed_at_value,
            "source_availability_status": source_availability_status,
            "missing_reason": None if v is not None else "value_missing",
        }
        if v is not None:
            chg_30 = MacroDataAccess.get_macro_change(key, 30)
            trend = ""
            if chg_30 is not None:
                trend = "up" if chg_30 > 0 else ("down" if chg_30 < 0 else "flat")
            indicators[name] = {"value": round(v, 4), "trend": trend,
                                 "change_30d": round(chg_30, 4) if chg_30 is not None else None,
                                 **indicator_statuses[name], "unit": unit}

    available_count = len(indicators)
    coverage = {
        "requested": len(indicator_specs),
        "accepted": available_count,
        "missing": len(indicator_specs) - available_count,
    }
    missing_reason = {
        name: status["missing_reason"] or "source_availability_time_unknown"
        for name, status in indicator_statuses.items()
        if status["missing_reason"] or status["source_availability_status"] == "unknown"
    }
    data_status = "partial" if available_count else "unavailable"
    return {
        "as_of": snap.as_of_date,
        "as_of_semantics": "query_cutoff_not_release_date",
        "generated_at": generated_at,
        "data_status": data_status,
        "coverage": coverage,
        "indicator_statuses": indicator_statuses,
        "data_envelope": make_data_envelope(
            source="macro_* tables",
            as_of=snap.as_of_date,
            unit={"indicator_values": "mixed; see indicators[].unit"},
            coverage=coverage,
            status=data_status,
            missing_reason=missing_reason or None,
        ),
        "indicators": indicators,
        "regime_bias": bias,
        "risk_score": risk["score"] if available_count else None,
        "risk_score_status": "partial" if available_count else "unavailable",
        "risk_level": risk["level"] if available_count else "unknown",
        "risk_signals": risk["signals"] if available_count else [],
    }


@router.get("/overview")
async def macro_overview():
    """最新宏观快照 + regime偏向 + 风险评分"""
    result = await anyio.to_thread.run_sync(lambda: _cached("overview:v2", 3600, _build))
    return {"data": result}
