import pandas as pd

import pytest

from core.domain.portfolio.t0 import T0Engine


pytestmark = pytest.mark.unit


def test_t0_returns_empty_variants_when_intraday_data_is_missing():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]

    result = T0Engine().compute("000001", None, daily)

    assert result["intraday_ok"] is False
    assert result["aggressive"] is None
    assert result["conservative"] is None


def test_t0_rejects_intraday_data_without_verified_quantity_semantics():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute("000001", intraday, daily, quantity_status="unknown")

    assert result["quantity_status"] == "unknown"
    assert result["intraday_ok"] is False
    assert result["aggressive"] is None
    assert result["conservative"] is None
    assert "分钟量额或来源语义未核验" in result["daily_flags"]


@pytest.mark.parametrize("amounts", [[None] * 6, [0] * 6])
def test_t0_rejects_verified_rows_with_incomplete_or_inconsistent_amount(amounts):
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": amounts,
    })

    result = T0Engine().compute("000001", intraday, daily, quantity_status="verified", semantic_status="verified")

    assert result["quantity_status"] == "unknown"
    assert result["aggressive"] is None
    assert result["conservative"] is None
    assert "分钟量额数值无效" in result["daily_flags"]


def test_t0_rejects_fractional_verified_minute_volume():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000.5] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute("000001", intraday, daily, quantity_status="verified", semantic_status="verified")

    assert result["quantity_status"] == "unknown"
    assert result["aggressive"] is None
    assert result["conservative"] is None
    assert "分钟量额数值无效" in result["daily_flags"]


def test_t0_accepts_complete_verified_quantity_rows():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute("000001", intraday, daily, quantity_status="verified", semantic_status="verified")

    assert result["quantity_status"] == "verified"
    assert result["aggressive"] is not None
    assert result["conservative"] is not None


def test_t0_does_not_offer_trade_quantity_without_verified_source_semantics():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute(
        "000001", intraday, daily, total_shares=1000, available_cash=10000,
        sellable_shares=500, quantity_status="verified",
    )

    assert result["quantity_status"] == "unknown"
    assert result["semantic_status"] == "unverified"
    assert result["current_price"] is None
    assert result["vwap"] is None
    assert result["day_high"] is None
    assert result["day_low"] is None
    assert result["atr_abs"] is None
    assert result["rsi"] is None
    assert result["intraday_ok"] is False
    assert result["aggressive"] is None
    assert result["conservative"] is None
    assert "分钟量额或来源语义未核验" in result["daily_flags"]


@pytest.mark.parametrize("field,value,flag", [
    ("open", 0, "分钟价格数据无效"),
    ("close", -1, "分钟价格数据无效"),
    ("high", 9.8, "分钟OHLC价格关系无效"),
    ("low", 10.2, "分钟OHLC价格关系无效"),
])
def test_t0_rejects_nonpositive_or_inconsistent_minute_prices(field, value, flag):
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })
    intraday.loc[0, field] = value

    result = T0Engine().compute("000001", intraday, daily, quantity_status="verified", semantic_status="verified")

    assert result["intraday_ok"] is False
    assert result["aggressive"] is None
    assert result["conservative"] is None
    assert flag in result["daily_flags"]


@pytest.mark.parametrize("current_price", [0, -1, float("nan"), float("inf"), float("-inf"), True])
def test_t0_rejects_invalid_explicit_current_price(current_price):
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute(
        "000001", intraday, daily, current_price=current_price, quantity_status="verified", semantic_status="verified",
    )

    assert result["quantity_status"] == "verified"
    assert result["intraday_ok"] is False
    assert result["aggressive"] is None
    assert result["conservative"] is None
    assert "当前价格无效" in result["daily_flags"]


@pytest.mark.parametrize("field,value", [
    ("close", 0),
    ("low", 10.3),
    ("high", 9.9),
    ("volume", float("nan")),
])
def test_t0_rejects_invalid_daily_price_or_quantity_rows(field, value):
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    daily[-1][field] = value
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute("000001", intraday, daily, quantity_status="verified", semantic_status="verified")

    assert result["intraday_ok"] is False
    assert result["aggressive"] is None
    assert result["conservative"] is None
    assert "日K价格或数量数据无效" in result["daily_flags"]


def test_t0_share_suggestion_respects_cash_for_one_code():
    shares = T0Engine()._suggested_shares(
        price=10.0,
        total_shares=10000,
        available_cash=1000.0,
    )

    assert shares * 10.0 * 1.002 <= 1000.0


@pytest.mark.parametrize("cash, holdings", [(0, 10000), (0.1, 10000), (10000, 0)])
def test_t0_preserves_explicit_zero_capacity(cash, holdings):
    engine = T0Engine()
    shares = engine._suggested_shares(10.0, holdings, cash)
    variant = engine._variant("I", "激进", 1.0, {"allowed": True}, shares)

    assert shares == 0
    assert variant["allowed"] is False


@pytest.mark.parametrize("cash", [None, float("nan"), float("inf"), -1, True, "invalid"])
def test_t0_unknown_or_invalid_cash_blocks_buy_quantity(cash):
    assert T0Engine()._suggested_shares(10.0, 1000, cash) == 0


def test_t0_missing_available_cash_blocks_both_t0_legs():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute(
        "000001", intraday, daily, total_shares=1000, sellable_shares=500,
        quantity_status="verified", semantic_status="verified",
    )

    assert result["aggressive"]["suggested_shares"] == 0
    assert result["aggressive"]["suggested_sell_shares"] == 0
    assert result["aggressive"]["allowed"] is False
    assert "本地可用资金缺失或无效，不能生成做T数量" in result["daily_flags"]


def test_t0_sell_suggestion_respects_broker_sellable_quantity():
    engine = T0Engine()

    assert engine._suggested_sell_shares(10.0, 1000, 300) == 300
    assert engine._suggested_sell_shares(10.0, 1000, 0) == 0
    assert engine._suggested_sell_shares(10.0, 1000, 300.5) == 0


@pytest.mark.parametrize("sellable_shares", [None, -1, 1001, 100.5, True])
def test_t0_rejects_missing_or_invalid_sellable_quantity_for_both_legs(sellable_shares):
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute(
        "000001", intraday, daily, total_shares=1000, available_cash=10000,
        sellable_shares=sellable_shares, quantity_status="verified", semantic_status="verified",
    )

    assert result["aggressive"]["suggested_shares"] == 0
    assert result["aggressive"]["suggested_sell_shares"] == 0
    assert result["aggressive"]["allowed"] is False
    assert "当日可卖股数缺失或无效，不能生成做T数量" in result["daily_flags"]


def test_t0_buy_and_sell_legs_are_bounded_by_cash_and_same_day_sellable_shares():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute(
        "000001", intraday, daily, total_shares=1000, available_cash=1500,
        sellable_shares=300, quantity_status="verified", semantic_status="verified",
    )
    variant = result["aggressive"]

    no_cash = T0Engine().compute(
        "000001", intraday, daily, total_shares=1000, available_cash=0,
        sellable_shares=300, quantity_status="verified", semantic_status="verified",
    )["aggressive"]

    assert variant["suggested_shares"] <= 100
    assert variant["suggested_sell_shares"] <= 100
    assert variant["suggested_shares"] == variant["suggested_sell_shares"]
    assert no_cash["suggested_shares"] == 0
    assert no_cash["suggested_sell_shares"] == 0
    assert no_cash["allowed"] is False


def test_t0_with_no_sellable_shares_does_not_offer_buy_only_t0():
    daily = [
        {
            "trade_date": date.strftime("%Y-%m-%d"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000000,
        }
        for date in pd.bdate_range("2025-01-01", periods=60)
    ]
    intraday = pd.DataFrame({
        "open": [10.0] * 6,
        "high": [10.1] * 6,
        "low": [9.9] * 6,
        "close": [10.0] * 6,
        "volume": [1000] * 6,
        "amount": [10000] * 6,
    })

    result = T0Engine().compute(
        "000001", intraday, daily, total_shares=1000, available_cash=10000,
        sellable_shares=0, quantity_status="verified", semantic_status="verified",
    )

    assert result["aggressive"]["suggested_shares"] == 0
    assert result["aggressive"]["suggested_sell_shares"] == 0
    assert result["aggressive"]["allowed"] is False
    assert "当日无可卖股数，无法完成T+0" in result["daily_flags"]


@pytest.mark.parametrize("reference_date", [None, "2025-01-02"])
def test_t0_selects_live_or_historical_source(monkeypatch, reference_date):
    from unittest.mock import Mock
    import backend.service.rebalance as service
    from core.data.repos.intraday import IntradayDataRepo

    live = Mock(return_value=pd.DataFrame(
        {"close": [10.0]},
        index=pd.DatetimeIndex(["2026-10-30 10:00:00"]),
    ))
    history = Mock(return_value=[{
        "bar_time": "2025-01-02 10:00:00",
        "close": 10.0,
        "quantity_status": "verified",
    }])
    compute = Mock(return_value={"code": "000001"})
    monkeypatch.setattr(service, "_try_ths_intraday_live", live)
    monkeypatch.setattr(IntradayDataRepo, "get_history_bars", history)
    monkeypatch.setattr(IntradayDataRepo, "get_daily_kline", lambda *args: [])
    monkeypatch.setattr(T0Engine, "compute", compute)
    if reference_date is None:
        monkeypatch.setattr(
            "backend.service.local_ledger.LocalLedgerService.get_t0_account_snapshot",
            lambda: {
                "account": {"account_id": "LOCAL-LIVE", "version": 3,
                            "status": "active", "available_cash": 1000},
                "holdings": [{"code": "000001", "shares": 1000, "sellable_shares": 700}],
                "sellable_source": "local_ledger_t1",
            },
        )
        monkeypatch.setattr(
            "backend.service.local_ledger.LocalLedgerService.get_account",
            lambda: {"status": "active", "version": 3, "broker_account_id": None},
        )

    result = service.RebalanceService.analyze_t0(
        ["000001"], reference_date=reference_date,
        holdings=[{"code": "000001", "shares": 0}], available_cash=0,
    )

    assert result["results"][0]["code"] == "000001"
    assert live.call_count == int(reference_date is None)
    assert history.call_count == int(reference_date is not None)
    assert compute.call_args.kwargs["total_shares"] == (1000 if reference_date is None else 0)
    assert compute.call_args.kwargs["available_cash"] == (1000 if reference_date is None else 0)
    assert compute.call_args.kwargs["sellable_shares"] == (700 if reference_date is None else None)
    assert compute.call_args.kwargs["quantity_status"] == "verified"
    assert compute.call_args.kwargs["semantic_status"] == "unverified"
    assert result["data_status"] == "partial"
    assert result["semantic_status"] == "unverified"
    assert result["coverage"] == {"requested": 1, "available": 1, "missing": 0}
    assert result["data_envelope"]["schema"] == "data-envelope-v2"
    assert result["data_envelope"]["status"] == "partial"
    assert result["mode"] == ("live" if reference_date is None else "historical_preview")
    assert result["data_source"] == ("ths_live_minute" if reference_date is None else "ths_minute_history")
    assert result["results"][0]["data_provenance"]["observed_at"] == (
        "2026-10-30T10:00:00+08:00" if reference_date is None else "2025-01-02T10:00:00+08:00"
    )
    assert result["results"][0]["data_provenance"]["price_basis"] == "unknown"
    assert result["results"][0]["data_provenance"]["unit"]["volume"] == "unknown"


def test_analyze_t0_splits_cash_budget_across_codes(monkeypatch):
    import backend.service.rebalance as rebalance_service
    from core.data.repos.intraday import IntradayDataRepo

    calls = []

    class FakeEngine:
        def compute(self, **kwargs):
            calls.append((kwargs["available_cash"], kwargs["sellable_shares"]))
            return {
                "code": kwargs["code"],
                "aggressive": {},
                "conservative": {},
            }

    monkeypatch.setattr("core.domain.portfolio.t0.T0Engine", FakeEngine)
    monkeypatch.setattr(
        rebalance_service,
        "_try_ths_intraday_live",
        lambda code, trade_date: pd.DataFrame({"close": [10.0]}),
    )
    monkeypatch.setattr(
        IntradayDataRepo,
        "get_daily_kline",
        lambda code, start_date, end_date: [{"trade_date": end_date}],
    )
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_t0_account_snapshot",
        lambda: {
            "account": {"account_id": "LOCAL-LIVE", "version": 3,
                        "status": "active", "available_cash": 1000},
            "holdings": [
                {"code": "000001", "shares": 1000, "sellable_shares": 300},
                {"code": "000002", "shares": 500, "sellable_shares": 0},
            ],
            "sellable_source": "local_ledger_t1",
        },
    )
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_account",
        lambda: {"status": "active", "version": 3, "broker_account_id": None},
    )

    result = rebalance_service.RebalanceService.analyze_t0(
        ["000001", "000002"], holdings=[], available_cash=1000.0,
    )

    assert len(result["results"]) == 2
    assert calls == [(500.0, 300), (500.0, 0)]
    assert result["account_id"] == "LOCAL-LIVE"
    assert result["account_version"] == 3
    assert result["sellable_source"] == "local_ledger_t1"
    assert result["data_status"] == "partial"
    assert result["results"][0]["data_provenance"]["semantic_status"] == "unverified"
    assert result["results"][0]["aggressive"]["allowed"] is False


def test_analyze_t0_reports_missing_local_holding_and_data_state(monkeypatch):
    import backend.service.rebalance as rebalance_service

    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_t0_account_snapshot",
        lambda: {
            "account": {"account_id": "LOCAL-LIVE", "version": 4,
                        "status": "active", "available_cash": 1000},
            "holdings": [],
            "sellable_source": "local_ledger_t1",
        },
    )
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_account",
        lambda: {"status": "active", "version": 4, "broker_account_id": None},
    )

    result = rebalance_service.RebalanceService.analyze_t0(["000001"])

    assert result["data_status"] == "unavailable"
    assert result["mode"] == "live"
    assert result["unavailable"] == ["000001"]
    assert result["coverage"] == {"requested": 1, "available": 0, "missing": 1}
    assert result["data_envelope"]["status"] == "unavailable"
    assert result["unavailable_details"]["000001"]["reason_code"] == "local_holding_missing"


@pytest.mark.parametrize("field,value", [
    ("shares", 1000.5),
    ("shares", True),
    ("sellable_shares", 300.5),
    ("sellable_shares", True),
])
def test_analyze_t0_preserves_invalid_share_values_for_domain_rejection(monkeypatch, field, value):
    import backend.service.rebalance as rebalance_service

    captured = {}

    class CapturingEngine:
        def compute(self, **kwargs):
            captured.update(kwargs)
            return {"code": kwargs["code"], "aggressive": None, "conservative": None}

    monkeypatch.setattr("core.domain.portfolio.t0.T0Engine", CapturingEngine)
    monkeypatch.setattr(
        rebalance_service,
        "_try_ths_intraday_live",
        lambda code, trade_date: pd.DataFrame({"close": [10.0]}),
    )
    holding = {"code": "000001", "shares": 1000, "sellable_shares": 300}
    holding[field] = value
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_t0_account_snapshot",
        lambda: {
            "account": {"account_id": "LOCAL-LIVE", "version": 3,
                        "status": "active", "available_cash": 1000},
            "holdings": [holding],
            "sellable_source": "local_ledger_t1",
        },
    )
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_account",
        lambda: {"status": "active", "version": 3, "broker_account_id": None},
    )

    rebalance_service.RebalanceService.analyze_t0(["000001"])

    assert captured["total_shares"] == holding["shares"]
    assert captured["sellable_shares"] == holding["sellable_shares"]


def test_analyze_t0_rejects_account_version_changed_during_calculation(monkeypatch):
    import backend.service.rebalance as rebalance_service

    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_t0_account_snapshot",
        lambda: {
            "account": {"account_id": "LOCAL-LIVE", "version": 3,
                        "status": "active", "available_cash": 1000,
                        "broker_account_id": None},
            "holdings": [],
            "sellable_source": "local_ledger_t1",
        },
    )
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_account",
        lambda: {"status": "active", "version": 4, "broker_account_id": None},
    )

    with pytest.raises(RuntimeError, match=r"本地账户在T\+0计算期间已变化"):
        rebalance_service.RebalanceService.analyze_t0(["000001"])


def test_analyze_t0_rejects_broker_sellable_change_during_calculation(monkeypatch):
    import backend.service.rebalance as rebalance_service
    from core.data.repos.intraday import IntradayDataRepo

    class FakeEngine:
        def compute(self, **kwargs):
            return {"code": kwargs["code"], "aggressive": {}, "conservative": {}}

    monkeypatch.setattr("core.domain.portfolio.t0.T0Engine", FakeEngine)
    monkeypatch.setattr(rebalance_service, "_try_ths_intraday_live", lambda code, trade_date: pd.DataFrame({"close": [10.0]}))
    monkeypatch.setattr(IntradayDataRepo, "get_daily_kline", lambda code, start_date, end_date: [{"trade_date": end_date}])
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_t0_account_snapshot",
        lambda: {
            "account": {"account_id": "LOCAL-LIVE", "version": 3, "status": "active",
                        "available_cash": 1000, "broker_account_id": "QMT-1"},
            "holdings": [{"code": "000001", "shares": 1000, "sellable_shares": 300}],
            "sellable_source": "broker_reconciliation",
        },
    )
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.get_account",
        lambda: {"status": "active", "version": 3, "broker_account_id": "QMT-1"},
    )
    monkeypatch.setattr(
        "backend.service.local_ledger.LocalLedgerService.reconcile",
        lambda: {"can_execute": True, "local_version": 3, "broker_account_id": "QMT-1",
                 "broker_sellable": {"000001": 200}},
    )

    with pytest.raises(RuntimeError, match=r"券商可卖数量在T\+0计算期间已变化"):
        rebalance_service.RebalanceService.analyze_t0(["000001"])


def test_live_t0_snapshot_uses_ledger_buys_and_same_day_opening_holding(monkeypatch):
    import datetime
    from backend.service.local_ledger import LocalLedgerService
    from core.data.repos.local_ledger import LocalLedgerRepo
    from core.data.connection import Database

    today = datetime.date.today().isoformat()
    account = {"account_id": "LOCAL-LIVE", "status": "active", "version": 5,
               "cash": 5000, "frozen": 0, "broker_account_id": None,
               "available_cash": 5000}
    snapshot = {
        "account": {**account},
        "holdings": [
            {"code": "000001", "shares": 1000, "buy_date": "2020-01-01"},
            {"code": "000002", "shares": 1000, "buy_date": today},
            {"code": "000003", "shares": 1000, "buy_date": None},
            {"code": "000004", "shares": 1000, "buy_date": "unknown"},
        ],
    }
    monkeypatch.setattr(LocalLedgerService, "get_advice_snapshot", lambda: snapshot)
    monkeypatch.setattr(Database, "fetchall", lambda *args: [
        {"code": "000001", "bought_today": 300},
    ])
    monkeypatch.setattr(LocalLedgerRepo, "get_account", lambda: account)

    result = LocalLedgerService.get_t0_account_snapshot()

    assert result["holdings"][0]["sellable_shares"] == 700
    assert result["holdings"][1]["sellable_shares"] == 0
    assert result["holdings"][2]["sellable_shares"] == 0
    assert result["holdings"][3]["sellable_shares"] == 0
    assert result["sellable_source"] == "local_ledger_t1"


def test_live_t0_uses_a_share_market_date_for_local_t1_sellability(monkeypatch):
    import datetime
    from backend.service.local_ledger import LocalLedgerService
    from core.data.repos.local_ledger import LocalLedgerRepo
    from core.data.connection import Database

    host_date = datetime.date.today()
    trading_date = host_date + datetime.timedelta(days=1)
    account = {"account_id": "LOCAL-LIVE", "status": "active", "version": 5,
               "cash": 5000, "frozen": 0, "broker_account_id": None,
               "available_cash": 5000}
    snapshot = {"account": {**account}, "holdings": [
        {"code": "000001", "shares": 1000, "buy_date": host_date.isoformat()},
    ]}
    monkeypatch.setattr(LocalLedgerService, "get_advice_snapshot", lambda: snapshot)
    monkeypatch.setattr(Database, "fetchall", lambda *args: [])
    monkeypatch.setattr(LocalLedgerRepo, "get_account", lambda: account)
    monkeypatch.setattr("core.shared.trading_time.market_date", lambda: trading_date)

    result = LocalLedgerService.get_t0_account_snapshot()

    assert result["holdings"][0]["sellable_shares"] == 1000


@pytest.mark.parametrize("can_execute", [True, False])
def test_live_t0_snapshot_uses_broker_sellable_quantity_when_account_is_bound(monkeypatch, can_execute):
    from backend.service.local_ledger import LocalLedgerService
    from core.data.repos.local_ledger import LocalLedgerRepo
    from core.data.connection import Database
    from backend.service.market_data import MarketDataService

    account = {"account_id": "LOCAL-LIVE", "status": "active", "version": 5,
               "cash": 5000, "frozen": 0, "available": 5000,
               "broker_account_id": "QMT-ACCOUNT-1"}
    monkeypatch.setattr(LocalLedgerRepo, "get_account", lambda: account.copy())
    monkeypatch.setattr(LocalLedgerRepo, "audit_integrity", lambda: {"ok": True})
    monkeypatch.setattr(Database, "fetchall", lambda *args: [{
        "code": "000001", "name": "平安银行", "shares": 1000,
        "cost_price": 10, "buy_date": "2020-01-01",
    }])
    monkeypatch.setattr(MarketDataService, "get_live_quotes", lambda *args, **kwargs: {
        "meta": {"freshness": "fresh"},
        "data": {"000001": {"price": 10, "pct_chg": 0}},
    })
    reconciliation = {
        "can_execute": can_execute, "local_version": 5,
        "broker_account_id": "QMT-ACCOUNT-1",
        "broker_sellable": {"000001": 300},
    }
    monkeypatch.setattr(LocalLedgerService, "reconcile", lambda: reconciliation)

    if not can_execute:
        with pytest.raises(RuntimeError, match="券商对账状态不可用"):
            LocalLedgerService.get_t0_account_snapshot()
        return

    result = LocalLedgerService.get_t0_account_snapshot()

    assert result["sellable_source"] == "broker_reconciliation"
    assert result["holdings"][0]["sellable_shares"] == 300
