from datetime import datetime
import datetime as datetime_module
import logging
import time

import pandas as pd

from core.data.sources.realtime import RealtimeEngine
from core.data.sources.base import BaseRealtimeSource
from core.data.repos.quotes import DailyQuotesRepo
from core.domain.market_data.freshness import assess_batch
from core.shared import config
from core.shared.data_contract import SUPPORTED_DATA_USE_CASES, data_envelope_allows_use, make_data_envelope
from backend.service.qlib_access import qlib_read_locked

logger = logging.getLogger(__name__)


class MarketDataService:
    _qlib_readiness_cache = {}

    @staticmethod
    def get_screening_decision_date(requested_date=None):
        from core.shared.trading_time import get_latest_trading_day

        if requested_date is not None:
            return get_latest_trading_day(requested_date)
        return (
            DailyQuotesRepo.get_latest_complete_date(config.COLLECT_MIN_DAILY_ROWS)
            or get_latest_trading_day()
        )

    @staticmethod
    @qlib_read_locked
    def get_qlib_readiness(decision_date):
        from core.adapters.qlib.bridge import QlibDataBridge
        requested = str(decision_date)[:10]
        bridge = QlibDataBridge(config.QLIB_DATA_DIR)
        status = bridge.get_status()
        latest = status.get("last_date")
        expected_codes = DailyQuotesRepo.get_tradable_codes_for_date(requested)
        daily_rows = DailyQuotesRepo.get_valid_row_count_for_date(requested)
        cache_key = (requested, latest, daily_rows, len(expected_codes), type(bridge))
        cached = MarketDataService._qlib_readiness_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 60:
            return dict(cached[1])
        calendar_ready = bool(status.get("synced") and latest and latest >= requested)
        daily_complete = daily_rows >= config.COLLECT_MIN_DAILY_ROWS
        coverage = bridge.get_date_coverage(
            requested, expected_codes,
        ) if calendar_ready else {
            "expected": len(expected_codes), "available": 0, "coverage_rate": 0.0,
            "missing_codes": expected_codes[:100],
        }
        result = {
            "ready": bool(daily_complete and calendar_ready and coverage["expected"] > 0
                          and coverage["available"] == coverage["expected"]),
            "decision_date": requested,
            "latest_date": latest,
            "daily_rows": daily_rows,
            "minimum_rows": config.COLLECT_MIN_DAILY_ROWS,
            **coverage,
        }
        MarketDataService._qlib_readiness_cache[cache_key] = (time.monotonic(), result)
        return dict(result)

    @staticmethod
    def get_conditional_market_env(start_date=None, end_date=None):
        if end_date is None:
            end_date = DailyQuotesRepo.get_latest_date()
        if end_date is None:
            return {}
        end = pd.Timestamp(end_date)
        start = (pd.Timestamp(start_date) if start_date is not None else end) - pd.Timedelta(days=90)
        rows = DailyQuotesRepo.get_market_daily_aggregates(start.date(), end.date())
        if not rows:
            return {}
        frame = pd.DataFrame(rows)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values("trade_date").drop_duplicates("trade_date")
        frame["breadth"] = pd.to_numeric(frame["breadth"], errors="coerce").fillna(0.5)
        market_ret = pd.to_numeric(frame["mean_ret"], errors="coerce").fillna(0.0) / 100.0
        market_level = (1.0 + market_ret).cumprod()
        oversold = (-market_level.pct_change(20, fill_method=None)).clip(-1, 1).fillna(0.0)
        return {
            row.trade_date.strftime("%Y-%m-%d"): (
                float(row.breadth), float(oversold.iloc[index])
            )
            for index, row in enumerate(frame.itertuples())
        }

    @staticmethod
    def get_live_intraday(code):
        for source in config.REALTIME_INTRADAY_CHAIN:
            if source != "ths":
                continue
            try:
                from core.data.ths_collector import THSCollector
                rows = THSCollector().fetch_intraday(code) or []
                if rows:
                    return rows, source
            except Exception:
                continue
        return [], None

    @staticmethod
    def get_live_sectors():
        from core.shared.persistence import now_iso
        from core.shared.trading_time import market_date
        from core.data.repos.stock_industry_history import (
            get_industry_snapshot_market_stats,
            get_latest_available_industry_snapshot,
        )

        received_at = now_iso()
        try:
            as_of_date = market_date()
            snapshot = get_latest_available_industry_snapshot(as_of_date.isoformat())
            if not snapshot:
                return {
                    "data": [],
                    "meta": {
                        "data_type": "sector",
                        "source": None,
                        "freshness": "unavailable",
                        "received_at": received_at,
                        "data_envelope": make_data_envelope(
                            source=None,
                            available_at=received_at,
                            coverage={"requested": 0, "accepted": 0},
                            status="unavailable",
                            missing_reason="verified_industry_snapshot_missing",
                        ),
                    },
                }
            latest_trade_date = DailyQuotesRepo.get_latest_complete_date(
                config.COLLECT_MIN_DAILY_ROWS, as_of_date=as_of_date,
            )
            if latest_trade_date is None:
                raise RuntimeError("无可用完整日线交易日")
            trade_date = latest_trade_date.isoformat() if hasattr(latest_trade_date, "isoformat") else str(latest_trade_date)[:10]
            rows = get_industry_snapshot_market_stats(snapshot["payload_sha256"], trade_date)
            requested = sum(int(row.get("member_count") or 0) for row in rows)
            accepted = sum(int(row.get("quote_count") or 0) for row in rows)
            available_at = str(snapshot["available_after_date"])
            available_date = datetime_module.date.fromisoformat(available_at[:10])
            trade_day = datetime_module.date.fromisoformat(trade_date)
            freshness = "fresh" if (as_of_date - available_date).days <= 14 and (as_of_date - trade_day).days <= 7 else "stale"
            source = snapshot["source_name"]
            for row in rows:
                member_count = int(row.get("member_count") or 0)
                quote_count = int(row.get("quote_count") or 0)
                pct_chg = row.get("pct_chg")
                coverage = quote_count / member_count if member_count else 0.0
                rank_eligible = member_count > 0 and quote_count == member_count and freshness == "fresh" and pct_chg is not None
                row.update({
                    "板块名称": row["sector_name"],
                    "name": row["sector_name"],
                    "涨跌幅": float(pct_chg) if pct_chg is not None and rank_eligible else None,
                    "pct_chg": float(pct_chg) if pct_chg is not None and rank_eligible else None,
                    "observed_pct_chg": float(pct_chg) if pct_chg is not None else None,
                    "membership_count": member_count,
                    "quote_count": quote_count,
                    "coverage": coverage,
                    "rank_eligible": rank_eligible,
                    "rank_coverage_threshold": 1.0,
                    "snapshot_id": snapshot["snapshot_id"],
                    "classification_standard": snapshot["classification_standard"],
                    "trade_date": trade_date,
                })
            status = "unavailable" if accepted == 0 else "available"
            return {
                "data": rows,
                "meta": {
                    "data_type": "sector",
                    "source": source,
                    "freshness": freshness,
                    "received_at": received_at,
                    "as_of": trade_date,
                    "snapshot_id": snapshot["snapshot_id"],
                    "classification_standard": snapshot["classification_standard"],
                    "data_envelope": make_data_envelope(
                        source=source,
                        observed_at=snapshot.get("observed_at"),
                        available_at=available_at,
                        as_of=trade_date,
                        version=snapshot["snapshot_id"],
                        unit={"pct_chg": "%", "volume": "share"},
                        coverage={"requested": requested, "accepted": accepted,
                                  "sectors": len(rows), "membership_snapshot": snapshot["snapshot_id"]},
                        status="stale" if freshness == "stale" else status,
                        missing_reason=("industry_or_daily_snapshot_stale" if freshness == "stale" else
                                        "daily_quote_coverage_incomplete" if accepted < requested
                                        else "industry_snapshot_no_members" if not rows else
                                        None),
                    ),
                },
            }
        except Exception as error:
            logger.warning("获取板块行情失败: %s", error)
            reason = "industry_sector_data_unavailable"
            return {
                "data": [],
                "meta": {
                    "data_type": "sector",
                    "source": "stock_industry_history+daily_quotes",
                    "freshness": "unavailable",
                    "received_at": received_at,
                    "data_envelope": make_data_envelope(
                            source="stock_industry_history+daily_quotes",
                        available_at=received_at,
                        coverage={"returned": 0},
                        status="unavailable",
                        missing_reason=reason,
                    ),
                },
            }

    @staticmethod
    def get_live_ranking(ascending=False, top_n=20):
        return RealtimeEngine().get_live_ranking(asc=1 if ascending else 0, top_n=top_n)

    @staticmethod
    def get_live_quotes(codes, use_case="live_display", data_type="quote"):
        request_codes = list(dict.fromkeys(str(code) for code in (codes or []) if code))
        expected_codes = list(dict.fromkeys(
            BaseRealtimeSource.clean_symbol(code) for code in request_codes
        ))
        source_chain = config.REALTIME_INDEX_CHAIN if data_type == "index" else config.REALTIME_QUOTE_CHAIN
        max_age = config.LIVE_QUOTE_MAX_AGE if use_case in ("execution", "risk") else config.LIVE_DISPLAY_MAX_AGE
        if not isinstance(use_case, str) or use_case not in SUPPORTED_DATA_USE_CASES:
            coverage = {
                "requested": len(expected_codes),
                "accepted": 0,
                "missing": len(expected_codes),
            }
            return {
                "data": {},
                "meta": {
                    "data_type": data_type,
                    "source": None,
                    "source_chain": list(source_chain),
                    "price_basis": "raw",
                    "volume_unit": "share",
                    "amount_unit": "yuan",
                    "max_age_seconds": max_age,
                    "age_seconds": None,
                    "freshness": "unavailable",
                    "partial": False,
                    "coverage": coverage,
                    "missing_codes": list(expected_codes),
                    "quote_statuses": {
                        code: {
                            "usable": False,
                            "freshness": "unavailable",
                            "age_seconds": None,
                            "missing_reason": "unsupported_use_case",
                        }
                        for code in expected_codes
                    },
                    "observed_at": None,
                    "received_at": None,
                    "use_case": use_case,
                    "use_case_eligible": False,
                    "data_envelope": make_data_envelope(
                        coverage=coverage,
                        status="unavailable",
                        missing_reason="unsupported_use_case",
                    ),
                },
            }
        if not expected_codes:
            coverage = {
                "requested": len(request_codes),
                "accepted": 0,
                "missing": len(request_codes),
            }
            missing_reasons = {
                code: "invalid_or_missing_symbol" for code in request_codes
            }
            return {
                "data": {},
                "meta": {
                    "data_type": data_type,
                    "source": None,
                    "source_chain": list(source_chain),
                    "price_basis": "raw",
                    "volume_unit": "share",
                    "amount_unit": "yuan",
                    "max_age_seconds": max_age,
                    "age_seconds": None,
                    "freshness": "unavailable",
                    "partial": False,
                    "coverage": coverage,
                    "missing_codes": request_codes,
                    "quote_statuses": {
                        code: {
                            "usable": False,
                            "freshness": "unavailable",
                            "age_seconds": None,
                            "missing_reason": "invalid_or_missing_symbol",
                        }
                        for code in request_codes
                    },
                    "observed_at": None,
                    "received_at": None,
                    "use_case": use_case,
                    "data_envelope": make_data_envelope(
                        source=None,
                        unit={"price": "CNY/share", "volume": "share", "amount": "yuan"},
                        price_basis="raw",
                        coverage=coverage,
                        status="unavailable",
                        missing_reason=missing_reasons or None,
                    ),
                },
            }

        source_codes = request_codes if data_type == "index" else expected_codes
        quote_fetch_failed = False
        try:
            quotes = RealtimeEngine().get_quotes(source_codes, source_chain=source_chain) or {}
            if not isinstance(quotes, dict):
                raise TypeError("实时行情返回值必须是字典")
        except Exception as exc:
            logger.warning("实时行情读取失败: %s", exc)
            quotes = {}
            quote_fetch_failed = True
        from core.shared.trading_time import A_SHARE_TIMEZONE
        now = datetime.now(A_SHARE_TIMEZONE)
        expected_session_date = None
        try:
            from core.shared.trading_time import is_trading_day
            if is_trading_day():
                expected_session_date = now.date().isoformat()
        except Exception:
            expected_session_date = None

        checked = assess_batch(
            quotes,
            expected_codes,
            now,
            max_age,
            expected_session_date=expected_session_date,
            require_complete=use_case in ("execution", "risk"),
        )
        if quote_fetch_failed:
            for status in checked["statuses"].values():
                status["missing_reason"] = "quote_fetch_failed"
        accepted = checked["accepted"]
        sources = sorted({q.get("source") for q in accepted.values() if q.get("source")})
        meta = {
            "data_type": data_type,
            "source": sources[0] if len(sources) == 1 else ("mixed" if sources else None),
            "source_chain": list(source_chain),
            "price_basis": "raw",
            "volume_unit": "share",
            "amount_unit": "yuan",
            "max_age_seconds": max_age,
            "age_seconds": checked["max_age_seconds"],
            "freshness": checked["freshness"],
            "partial": checked["partial"],
            "coverage": checked["coverage"],
            "missing_codes": checked["missing_codes"],
            "quote_statuses": checked["statuses"],
            "observed_at": max((q.get("observed_at", "") for q in accepted.values()), default=None),
            "received_at": max((q.get("received_at", "") for q in accepted.values()), default=None),
            "use_case": use_case,
        }
        meta["data_envelope"] = make_data_envelope(
            source=meta["source"],
            observed_at=meta["observed_at"],
            available_at=meta["received_at"],
            as_of=expected_session_date or max(
                (str(quote.get("session_date")) for quote in accepted.values() if quote.get("session_date")),
                default=None,
            ),
            unit={"price": "CNY/share", "volume": "share", "amount": "yuan"},
            price_basis="raw",
            coverage=checked["coverage"],
            status={
                "fresh": "available", "degraded": "partial", "unavailable": "unavailable",
            }[checked["freshness"]],
            missing_reason={
                code: status["missing_reason"]
                for code, status in checked["statuses"].items()
                if status["missing_reason"]
            } or None,
        )
        if use_case in ("execution", "risk"):
            row_contract_failures = {}
            for code, quote in accepted.items():
                row_envelope = make_data_envelope(
                    source=quote.get("source"),
                    observed_at=quote.get("observed_at"),
                    available_at=quote.get("received_at"),
                    as_of=quote.get("session_date"),
                    unit={"price": "CNY/share"},
                    price_basis=quote.get("price_basis"),
                    coverage={"requested": 1, "accepted": 1, "missing": 0},
                    status="available",
                )
                if not data_envelope_allows_use(row_envelope, use_case):
                    row_contract_failures[code] = row_envelope.get("validation_errors") or [
                        row_envelope.get("missing_reason") or "data_contract_unverified"
                    ]
            eligible = (
                data_envelope_allows_use(meta["data_envelope"], use_case)
                and not row_contract_failures
            )
            meta["use_case_eligible"] = eligible
            meta["row_contract_failures"] = row_contract_failures
            if not eligible:
                meta["source_coverage"] = dict(meta["coverage"])
                meta["data_envelope"]["source_coverage"] = dict(meta["coverage"])
                meta["data_envelope"]["status"] = "unavailable"
                if row_contract_failures:
                    meta["data_envelope"]["validation_errors"] = list(dict.fromkeys([
                        *(meta["data_envelope"].get("validation_errors") or []),
                        *(f"{code}:{error}" for code, errors in row_contract_failures.items()
                          for error in errors),
                    ]))
                    if not meta["data_envelope"].get("missing_reason"):
                        meta["data_envelope"]["missing_reason"] = row_contract_failures
                if not meta["data_envelope"].get("missing_reason"):
                    meta["data_envelope"]["missing_reason"] = "data_contract_unverified"
                meta["data_envelope"]["use_case_blocked"] = use_case
                meta["quote_statuses"] = {
                    code: {
                        **status,
                        "usable": False,
                        "missing_reason": row_contract_failures.get(code, ["data_contract_unverified"])[0],
                    }
                    for code, status in meta["quote_statuses"].items()
                }
                meta["coverage"] = {
                    "requested": len(expected_codes),
                    "accepted": 0,
                    "missing": len(expected_codes),
                }
                meta["data_envelope"]["coverage"] = dict(meta["coverage"])
                meta["missing_codes"] = list(expected_codes)
                meta["freshness"] = "unavailable"
                meta["partial"] = False
                accepted = {}
        elif use_case == "live_display":
            meta["use_case_eligible"] = True
        return {"data": accepted, "meta": meta}

    @staticmethod
    def get_live_prices(codes, use_case="execution"):
        response = MarketDataService.get_live_quotes(codes, use_case=use_case)
        if response["meta"]["freshness"] != "fresh":
            return {}, response["meta"]
        prices = {}
        for code, quote in response["data"].items():
            price = quote.get("price") or quote.get("last_price") or 0
            if price and float(price) > 0:
                prices[code] = float(price)
        return prices, response["meta"]
