import asyncio
import datetime
import sys
from decimal import Decimal
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.adapters.broker.easytrader_broker import EasytraderBroker
from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING
from core.adapters.broker.qmt_broker import QmtBroker, STATUS_MAP, _TraderCallback
from core.shared.order_validation import validate_real_limit_order
from backend.service.local_ledger import LocalLedgerService
from core.data.repos.broker_trade_report import BrokerTradeReportRepo


@pytest.fixture(autouse=True)
def isolate_broker_submission_storage(monkeypatch):
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    monkeypatch.setattr(
        BrokerOrderReportRepo,
        "record_submission",
        classmethod(lambda cls, broker, account_id, order, client_request, broker_request, broker_response: {
            "client_request": client_request, "broker_request": broker_request,
            "broker_response": broker_response,
        }),
    )
    monkeypatch.setattr(
        BrokerOrderReportRepo,
        "list_submission_metadata",
        classmethod(lambda cls, broker, account_id, order_ids: {}),
    )
@pytest.fixture(autouse=True)
def isolate_easytrader_report_storage(monkeypatch, request):
    if request.node.name.startswith("test_easytrader"):
        from core.data.repos.broker_order_report import BrokerOrderReportRepo

        monkeypatch.setattr(
            BrokerOrderReportRepo,
            "ingest",
            classmethod(lambda cls, broker, account_id, order, source, raw_status=None, raw_fields=None: {
                "disposition": "applied", "status": order["status"], "filled_qty": order["filled_qty"],
            }),
        )
        monkeypatch.setattr(
            BrokerOrderReportRepo,
            "record_submission",
            classmethod(lambda cls, broker, account_id, order, client_request, broker_request, broker_response: {
                "client_request": client_request, "broker_request": broker_request,
                "broker_response": broker_response,
            }),
        )
        monkeypatch.setattr(
            BrokerOrderReportRepo,
            "list_submission_metadata",
            classmethod(lambda cls, broker, account_id, order_ids: {}),
        )
        monkeypatch.setattr(
            BrokerTradeReportRepo,
            "ingest",
            classmethod(lambda cls, broker, account_id, trade, source, raw_fields=None: {
                "trade_id": trade["id"], "status": "recorded", "fees_complete": False,
            }),
        )


def test_disconnected_qmt_does_not_return_cached_orders_trades_or_positions():
    broker = QmtBroker("unused", "test")
    broker._orders = {"1": {"id": "1", "status": "pending"}}
    broker._trades = [{"id": "T-1"}]
    broker._positions = {"000001.SZ": {"stock_code": "000001.SZ", "volume": 100}}
    broker._positions_observed_at = datetime.datetime.now()
    for read in (lambda: broker.get_order("1"), broker.get_orders, broker.get_trades, broker.get_positions):
        with pytest.raises(RuntimeError, match="不可用"):
            read()


def test_qmt_connect_fails_before_sdk_when_userdata_directory_is_missing(tmp_path):
    broker = QmtBroker(str(tmp_path / "missing-userdata"), "test")
    with patch.dict(sys.modules, {"xtquant.xttrader": None}):
        with pytest.raises(RuntimeError, match="userdata_mini 目录不存在"):
            broker.connect()
    assert broker.connected is False
    assert broker._trader is None


def test_qmt_connect_rejects_missing_account_before_sdk(tmp_path):
    broker = QmtBroker(str(tmp_path), "")
    with patch.dict(sys.modules, {"xtquant.xttrader": None}):
        with pytest.raises(RuntimeError, match="资金账号未配置"):
            broker.connect()
    assert broker.connected is False
    assert broker._trader is None


def test_qmt_missing_positions_response_is_not_an_empty_portfolio():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.query_stock_positions.return_value = None
    with pytest.raises(RuntimeError, match="持仓查询不可用"):
        broker.get_positions()


def test_easytrader_missing_positions_response_is_not_an_empty_portfolio():
    broker = EasytraderBroker()
    with pytest.raises(RuntimeError, match="持仓查询不可用"):
        broker.get_positions()
    broker._connected = True
    broker._trader = Mock()
    broker._trader.position = None
    with patch.object(broker, "_dismiss_captcha"):
        with pytest.raises(RuntimeError, match="持仓查询不可用"):
            broker.get_positions()
    broker._trader.position = []
    assert broker.get_positions() == []


@pytest.mark.parametrize("bad_position", [
    {"account_id": "other", "stock_code": "000002.SZ", "volume": 100},
    {"account_id": "test", "stock_code": "", "volume": 100},
    {"account_id": "test", "stock_code": "000002.SZ", "volume": "1.5"},
    {"account_id": "test", "stock_code": "000001.SZ", "volume": 100},
])
def test_qmt_invalid_position_row_rejects_entire_snapshot(bad_position):
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    previous = {"000003": {"account_id": "test", "stock_code": "000003.SZ", "volume": 100}}
    broker._positions = previous
    broker._positions_observed_at = datetime.datetime.now()
    broker._trader.query_stock_positions.return_value = [
        {"account_id": "test", "stock_code": "000001.SZ", "volume": 100},
        bad_position,
    ]
    with pytest.raises(RuntimeError, match="持仓查询不可用"):
        broker.get_positions()
    assert broker._positions is previous
    assert broker._positions_observed_at is None


def test_qmt_integral_decimal_position_quantity_is_preserved():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.query_stock_positions.return_value = [
        {"account_id": "test", "stock_code": "000001.SZ", "volume": "100.0"},
    ]
    assert broker.get_positions()[0]["shares"] == 100


@pytest.mark.parametrize("bad_position", [
    {"股票余额": 100},
    {"证券代码": "000001", "股票余额": None},
    {"证券代码": "000001", "股票余额": "1.5"},
    {"证券代码": "000001.SH", "股票余额": 100},
    {"证券代码": "000001", "股票余额": -1},
])
def test_easytrader_invalid_position_row_is_not_silently_dropped(bad_position):
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    broker._trader.position = [bad_position]
    with pytest.raises(RuntimeError, match="持仓证券代码或股数无效"):
        broker.get_positions()


def test_easytrader_position_maps_documented_total_and_sellable_shares():
    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock(position=[{
        "证券代码": "600000", "股份余额": "1200", "当前持仓": 1200,
        "股份可用": "300", "参考成本价": 10.2,
    }])

    position = broker.get_positions()[0]

    assert position["shares"] == 1200
    assert position["sellable"] == 300
    assert position["avg_cost"] == 10.2


@pytest.mark.parametrize("sellable", [None, -1, "1.5", 101])
def test_easytrader_invalid_sellable_quantity_stays_unknown(sellable):
    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock(position=[{
        "证券代码": "600000", "当前持仓": 100, "股份可用": sellable,
    }])

    assert broker.get_positions()[0]["sellable"] is None


def test_positions_api_reports_broker_failure_as_unavailable():
    from backend.api import trading
    from fastapi import HTTPException

    broker = Mock()
    broker.get_positions.side_effect = RuntimeError("券商断连")
    with patch.object(trading, "get_broker", return_value=broker):
        with pytest.raises(HTTPException) as error:
            asyncio.run(trading.get_positions())
    assert error.value.status_code == 503


@pytest.mark.parametrize("field,value", [
    ("quantity", True), ("quantity", 100.5), ("price", True),
    ("price", "10.0"), ("price", 10 ** 400),
])
def test_order_api_rejects_coerced_numeric_inputs_before_broker_access(field, value):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import trading

    app = FastAPI()
    app.include_router(trading.router)
    app.dependency_overrides[trading.verify_trading_auth] = lambda: True
    payload = {
        "symbol": "000001",
        "side": "buy",
        "quantity": 100,
        "price_type": "limit",
        "price": 10,
    }
    payload[field] = value
    with patch.object(trading, "get_broker", side_effect=AssertionError("无效请求不得读取券商")):
        response = TestClient(app).post("/orders", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("price", [float("nan"), float("inf"), float("-inf")])
def test_order_request_rejects_non_finite_numeric_price(price):
    from pydantic import ValidationError
    from backend.api.trading import OrderRequest

    with pytest.raises(ValidationError):
        OrderRequest.model_validate({
            "symbol": "000001",
            "side": "buy",
            "quantity": 100,
            "price_type": "limit",
            "price": price,
        })


@pytest.mark.parametrize("changes", [
    {"price_type": "market", "price": 0},
    {"price_type": "limit", "price": 0},
    {"price_type": "limit", "price": 10, "quantity": 50},
    {"price_type": "limit", "price": 10, "symbol": "000001.SH"},
])
def test_real_order_semantics_rejected_before_broker_access(changes):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import trading
    from core.shared import config

    app = FastAPI()
    app.include_router(trading.router)
    app.dependency_overrides[trading.verify_trading_auth] = lambda: True
    payload = {
        "symbol": "000001",
        "side": "buy",
        "quantity": 100,
        "price_type": "limit",
        "price": 10,
    }
    payload.update(changes)
    with patch.object(config, "BROKER_TYPE", config.BROKER_QMT), \
         patch.object(trading, "get_broker", side_effect=AssertionError("无效实盘委托不得初始化券商")):
        response = TestClient(app).post("/orders", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("broker_type", ["qmt", "easytrader"])
@pytest.mark.parametrize("reconciliation,expected_message", [
    ({"can_execute": False, "reason": "券商对账存在资金差异"}, "券商对账存在资金差异"),
    ({"can_execute": True, "reason": None}, "实盘委托尚未绑定本地账本计划和委托预留"),
])
def test_direct_real_order_route_never_submits_without_local_intent(broker_type, reconciliation, expected_message):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import trading
    from core.shared import config

    app = FastAPI()
    app.include_router(trading.router)
    app.dependency_overrides[trading.verify_trading_auth] = lambda: True
    broker = Mock(connected=True, supported_price_types=("limit",), place_order=Mock())
    payload = {
        "symbol": "000001", "side": "buy", "quantity": 100,
        "price_type": "limit", "price": 10,
    }
    with patch.object(config, "BROKER_TYPE", broker_type), \
         patch.object(config, "QMT_ACCOUNT", "test"), \
         patch.object(trading, "get_broker", return_value=broker), \
         patch("core.adapters.broker.is_fake_broker", return_value=False), \
         patch("backend.service.local_ledger.LocalLedgerService.reconcile",
               return_value=reconciliation):
        response = TestClient(app).post("/orders", json=payload)

    assert response.status_code == 409
    assert expected_message in response.json()["detail"]
    broker.place_order.assert_not_called()


def test_broker_adapters_publish_supported_price_types():
    assert FakeBroker.supported_price_types == ("market", "limit")
    assert QmtBroker.supported_price_types == ("limit",)
    assert EasytraderBroker.supported_price_types == ("limit",)


def test_qmt_status_constants_distinguish_partial_cancel_and_rejection():
    assert [STATUS_MAP[code] for code in (48, 49, 50, 51, 52, 53, 54, 55, 56, 57)] == [
        "pending", "pending", "pending", "pending", "partial", "cancelled",
        "cancelled", "partial", "filled", "rejected",
    ]


@pytest.mark.parametrize("status,filled_qty", [("pending", 0), ("partial", 40)])
def test_qmt_cancel_acceptance_does_not_finalize_order(status, filled_qty):
    from unittest.mock import Mock, patch

    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.cancel_order.return_value = 0
    order = {
        "id": "123", "symbol": "000001.SZ", "side": "buy",
        "quantity": 100, "filled_qty": filled_qty, "status": status,
    }
    broker._orders["123"] = order

    with patch.object(broker, "_persist_order") as persist:
        assert broker.cancel_order("123") is True

    assert broker._orders["123"] == order
    persist.assert_not_called()


@pytest.mark.parametrize("raw_order_type,expected_side", [(23, "buy"), (24, "sell")])
def test_qmt_order_callback_uses_actual_buy_sell_type(raw_order_type, expected_side):
    broker = QmtBroker("unused", "test")
    raw_order = {"account_id": "test", "order_id": 123, "stock_code": "000001.SZ",
                 "order_type": raw_order_type, "order_volume": 100, "order_status": 48,
                 "price_type": 11, "price": 10, "traded_volume": 0, "traded_price": 0}
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest",
               return_value={"disposition": "applied", "status": "pending", "filled_qty": 0}) as ingest, \
         patch.object(broker, "_persist_qmt_order", return_value=True), patch.object(broker, "_sync_positions"):
        _TraderCallback(broker).on_stock_order(raw_order)
    assert ingest.call_args.args[2]["side"] == expected_side


def test_qmt_unknown_price_type_stays_unknown_in_callback():
    broker = QmtBroker("unused", "test")
    raw_order = {"account_id": "test", "order_id": 123, "stock_code": "000001.SZ",
                 "order_type": 23, "order_volume": 100, "order_status": 48,
                 "price_type": 999, "price": 10, "traded_volume": 0, "traded_price": 0}
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest",
               return_value={"disposition": "conflict", "status": "unknown", "filled_qty": 0}) as ingest, \
         patch.object(broker, "_persist_order"), patch.object(broker, "_sync_positions"):
        _TraderCallback(broker).on_stock_order(raw_order)
    assert ingest.call_args.args[2]["price_type"] == "unknown"


def test_qmt_order_callback_preserves_fractional_quantities_for_validation():
    broker = QmtBroker("unused", "test")
    raw_order = {"account_id": "test", "order_id": 123, "stock_code": "000001.SZ",
                 "order_type": 23, "order_volume": 100.5, "order_status": 48,
                 "price_type": 11, "price": 10, "traded_volume": 0.5, "traded_price": 0}
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest",
               return_value={"disposition": "conflict", "status": "unknown", "filled_qty": 0}) as ingest, \
         patch.object(broker, "_persist_order") as persist, patch.object(broker, "_sync_positions"):
        _TraderCallback(broker).on_stock_order(raw_order)
    recorded = ingest.call_args.args[2]
    assert recorded["quantity"] == 100.5
    assert recorded["filled_qty"] == 0.5
    assert broker._orders == {}
    persist.assert_not_called()


@pytest.mark.parametrize("raw_order_type,expected_side", [(23, "buy"), (24, "sell")])
def test_qmt_trade_callback_uses_actual_buy_sell_type(raw_order_type, expected_side):
    broker = QmtBroker("unused", "test")
    raw_trade = {"account_id": "test", "traded_id": "T-1", "order_id": 123,
                 "stock_code": "000001.SZ", "order_type": raw_order_type,
                 "traded_price": 10, "traded_volume": 100, "traded_amount": 1000,
                 "traded_time": "10:00:00"}
    with patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.ingest",
               return_value={"status": "recorded", "fees_complete": False}) as ingest, \
         patch.object(broker, "_persist_trade"):
        _TraderCallback(broker).on_stock_trade(raw_trade)
    assert ingest.call_args.args[2]["side"] == expected_side


def test_qmt_trade_callback_preserves_fractional_quantity_for_validation():
    broker = QmtBroker("unused", "test")
    raw_trade = {"account_id": "test", "traded_id": "T-1", "order_id": 123,
                 "stock_code": "000001.SZ", "order_type": 23, "traded_price": 10,
                 "traded_volume": 40.5, "traded_amount": 405, "traded_time": "10:00:00"}
    with patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.ingest",
               return_value={"status": "conflict", "fees_complete": False}) as ingest:
        _TraderCallback(broker).on_stock_trade(raw_trade)
    assert ingest.call_args.args[2]["quantity"] == 40.5
    assert broker._trades == []


def test_qmt_unknown_order_type_does_not_enter_account_history():
    broker = QmtBroker("unused", "test")
    raw_order = {"account_id": "test", "order_id": 123, "stock_code": "000001.SZ",
                 "order_type": 99, "order_volume": 100, "order_status": 48,
                 "price_type": 11, "price": 10, "traded_volume": 0, "traded_price": 0}
    with patch.object(broker, "_persist_order") as persist:
        _TraderCallback(broker).on_stock_order(raw_order)
    assert broker._orders == {}
    assert broker._report_persistence_error is not None
    persist.assert_not_called()


def test_qmt_asset_cash_is_available_and_frozen_is_separate():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_asset.return_value = {
        "account_id": "test", "cash": 1000, "frozen_cash": 200,
        "market_value": 3000, "total_asset": 4200,
    }
    with patch.object(broker, "_should_sync", return_value=True), \
         patch.object(broker, "_sync_positions"):
        account = broker.get_account()
    assert account["status"] == "fresh"
    assert account["cash"] == 1200
    assert account["frozen"] == 200
    assert account["available"] == 1000


def test_qmt_asset_other_account_cannot_become_verified_snapshot():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_asset.return_value = {
        "account_id": "other", "cash": 1000, "frozen_cash": 0,
        "market_value": 3000, "total_asset": 4000,
    }
    with patch.object(broker, "_should_sync", return_value=True), \
         patch.object(broker, "_sync_positions"):
        account = broker.get_account()
    assert account["status"] == "unavailable"
    assert account["available"] is None


def test_qmt_reconciliation_maps_installed_xtquant_sdk_objects_without_connecting():
    xttype = pytest.importorskip("xtquant.xttype")
    from xtquant import xtconstant

    now = datetime.datetime.now()
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = xttype.StockAccount("test")
    broker._asset = xttype.XtAsset("test", 1000, 50, 1000, 2050, 950)
    broker._asset_observed_at = now
    broker._asset_sync_ok = True
    position = xttype.XtPosition(
        "test", "000001.SZ", 100, 60, 9.5, 1000, 40, 0, 100, 9.5, 0, 10, 0.05,
        "SH123", "测试股",
    )
    broker._positions = {"000001.SZ": position}
    broker._positions_observed_at = now
    broker._trader.query_stock_positions.return_value = [position]
    broker._trader.query_stock_orders.return_value = [xttype.XtOrder(
        "test", "000001.SZ", 123, "sys-123", 0, xtconstant.STOCK_BUY, 100,
        xtconstant.FIX_PRICE, 10, 0, 0, xtconstant.ORDER_UNREPORTED,
        "", "", "", 0, 0, "SH123", "测试股",
    )]

    with patch.object(broker, "_sync_asset"), patch.object(broker, "_sync_positions"):
        snapshot = broker.get_reconciliation_snapshot(check_report_integrity=False)

    assert snapshot["identity_verified"] is True
    assert snapshot["account"]["account_id"] == "QMT-test"
    assert snapshot["account"]["cash"] == 1050
    assert snapshot["account"]["available"] == 1000
    assert snapshot["account"]["frozen"] == 50
    assert snapshot["account"]["unrealized_pnl"] == 50
    assert snapshot["positions"] == [{
        "symbol": "000001.SZ", "shares": 100, "sellable": 60,
    }]
    assert snapshot["orders"] == [{
        "id": "123", "symbol": "000001.SZ", "side": "buy", "status": "pending",
        "quantity": 100, "price_type": "limit", "price": 10, "filled_qty": 0,
    }]
    broker._trader.order_stock.assert_not_called()


def test_qmt_report_from_other_account_is_never_journaled():
    broker = QmtBroker("unused", "test")
    order = {"id": "123", "symbol": "000001.SZ", "side": "buy",
             "quantity": 100, "filled_qty": 0, "status": "pending",
             "price_type": "limit", "price": 10}
    trade = {"id": "T-1", "order_id": "123", "symbol": "000001.SZ", "side": "buy",
             "quantity": 100, "price": 10, "amount": 1000,
             "commission": None, "stamp_duty": None}
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest") as order_ingest, \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.ingest") as trade_ingest:
        assert broker._record_order_report(order, "callback", 48, {"account_id": "other"}) is None
        assert broker._record_trade_report(trade, "callback", {"account_id": "other"}) is None
    order_ingest.assert_not_called()
    trade_ingest.assert_not_called()
    assert broker._report_persistence_error is not None


def test_qmt_other_account_order_callback_is_not_cached_or_persisted():
    broker = QmtBroker("unused", "test")
    raw_order = {"account_id": "other", "order_id": 123, "stock_code": "000001.SZ",
                 "order_type": 23, "order_volume": 100, "order_status": 48,
                 "price_type": 11, "price": 10, "traded_volume": 0, "traded_price": 0}
    with patch.object(broker, "_persist_order") as persist:
        _TraderCallback(broker).on_stock_order(raw_order)
    assert broker._orders == {}
    assert broker._report_persistence_error is not None
    persist.assert_not_called()


def test_qmt_reconciliation_rejects_cross_account_positions():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_positions.return_value = [
        {"account_id": "other", "stock_code": "000001.SZ", "volume": 100, "can_use_volume": 100},
    ]
    broker._trader.query_stock_orders.return_value = []
    with patch.object(broker, "get_account", return_value={"status": "fresh"}):
        with pytest.raises(RuntimeError, match="资金账号不一致"):
            broker.get_reconciliation_snapshot(check_report_integrity=False)


def test_qmt_reconciliation_rejects_asset_that_aged_during_queries():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_positions.return_value = []
    broker._trader.query_stock_orders.return_value = []
    old_time = (datetime.datetime.now() - datetime.timedelta(seconds=8)).isoformat()
    with patch.object(broker, "get_account", return_value={"status": "fresh", "as_of": old_time}):
        with pytest.raises(RuntimeError, match="资产已过期"):
            broker.get_reconciliation_snapshot(check_report_integrity=False)


def test_qmt_reconciliation_snapshot_is_read_only_and_does_not_replay_reports():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_positions.return_value = []
    broker._trader.query_stock_orders.return_value = []
    account = {
        "status": "fresh",
        "as_of": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with patch.object(broker, "replay_order_reports", side_effect=RuntimeError("must not replay")) as replay, \
         patch.object(broker, "get_account", return_value=account), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.unresolved_count", return_value=0), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.get_health",
               return_value={"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0}):
        snapshot = broker.get_reconciliation_snapshot()
    replay.assert_not_called()
    assert snapshot["identity_verified"] is True
    assert snapshot["orders"] == []


def test_qmt_reconciliation_reads_current_orders_without_persisting_projection():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_positions.return_value = []
    broker._trader.query_stock_orders.return_value = [
        {"account_id": "test", "order_id": 123, "stock_code": "000001.SZ",
         "order_type": 24, "order_volume": 100, "order_status": 54,
         "price_type": 11, "price": 10, "traded_volume": 0},
    ]
    with patch.object(broker, "replay_order_reports", side_effect=RuntimeError("must not replay")) as replay, \
         patch.object(broker, "get_account", return_value={
             "status": "fresh",
             "as_of": datetime.datetime.now().isoformat(timespec="seconds"),
         }), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.unresolved_count", return_value=0), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.get_health",
               return_value={"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0}), \
         patch.object(broker, "_record_order_report", return_value={"disposition": "stale"}) as record:
        snapshot = broker.get_reconciliation_snapshot()
    replay.assert_not_called()
    record.assert_not_called()
    assert snapshot["orders"][0]["id"] == "123"
    assert snapshot["orders"][0]["side"] == "sell"


def test_qmt_trade_conflict_resolution_uses_current_unique_sell_trade():
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_trades.return_value = [
        {"account_id": "test", "traded_id": "T-1", "order_id": 123,
         "stock_code": "000001.SZ", "order_type": 24,
         "traded_volume": 100, "traded_price": 10, "traded_amount": 1000},
    ]
    with patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalLedgerService, "_get_report_scope", return_value=("qmt", broker.account_id)), \
         patch.object(BrokerTradeReportRepo, "resolve_conflicts", return_value={
             "trade_id": "T-1", "resolved_count": 1,
         }) as resolve:
        result = LocalLedgerService.resolve_broker_trade_conflicts(
            "T-1", "admin", "已核对券商逐笔卖出成交记录",
        )
    assert result["resolved_count"] == 1
    assert resolve.call_args.args[3]["side"] == "sell"
    assert resolve.call_args.args[3]["amount"] == 1000


def test_easytrader_trade_conflict_resolution_uses_verified_current_trade_evidence():
    from core.adapters.broker.easytrader_broker import EasytraderBroker
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-1")
    broker.get_trade_reconciliation_evidence = Mock(return_value={
        "id": "T-1", "order_id": "O-1", "symbol": "000001", "side": "buy",
        "quantity": 100, "price": 10, "amount": 1000,
        "commission": "0.30", "stamp_duty": "0.00",
    })
    with patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalLedgerService, "_get_report_scope", return_value=("easytrader", broker.account_id)), \
         patch.object(BrokerTradeReportRepo, "resolve_conflicts", return_value={
             "trade_id": "T-1", "resolved_count": 1,
         }) as resolve:
        result = LocalLedgerService.resolve_broker_trade_conflicts(
            "T-1", "admin", "已核对Easytrader逐笔成交回报",
        )

    assert result["resolved_count"] == 1
    broker.get_trade_reconciliation_evidence.assert_called_once_with("T-1")
    assert resolve.call_args.args[:3] == ("easytrader", broker.account_id, "T-1")


def test_easytrader_order_report_resolution_requires_verified_terminal_evidence():
    from core.adapters.broker.easytrader_broker import EasytraderBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-1")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "identity_verified": True, "account": {"account_id": broker.account_id},
    })
    broker.get_order_reconciliation_evidence = Mock(return_value={
        "id": "O-1", "symbol": "000001", "side": "buy", "quantity": 100,
        "price_type": "limit", "price": 10, "filled_qty": 40, "status": "cancelled",
    })
    with patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalLedgerService, "_get_report_scope", return_value=("easytrader", broker.account_id)), \
         patch.object(BrokerOrderReportRepo, "get_by_key", return_value={
             "broker": "easytrader", "broker_account_id": broker.account_id,
             "order_id": "O-1", "resolved_at": None,
         }), \
         patch.object(BrokerOrderReportRepo, "resolve", return_value={"already_resolved": False}) as resolve:
        result = LocalLedgerService.resolve_broker_report("key", "admin", "核对Easytrader终态委托")

    assert result["already_resolved"] is False
    broker.get_order_reconciliation_evidence.assert_called_once_with("O-1")
    assert resolve.call_args.args[1:3] == ("easytrader", broker.account_id)


@pytest.mark.parametrize("evidence_method,source_method,item_key,item", [
    ("get_trade_reconciliation_evidence", "get_trades", "id", {
        "id": "T-1", "order_id": "O-1", "symbol": "000001", "side": "buy",
        "quantity": 100, "price": 10, "amount": 1000,
    }),
    ("get_order_reconciliation_evidence", "get_orders", "id", {
        "id": "O-1", "symbol": "000001", "side": "buy", "quantity": 100,
        "price_type": "limit", "price": 10, "filled_qty": 40, "status": "cancelled",
    }),
])
def test_easytrader_report_evidence_requires_verified_configured_account(
        evidence_method, source_method, item_key, item):
    from core.adapters.broker.easytrader_broker import EasytraderBroker

    broker = EasytraderBroker(broker_account_id="ACCT-1")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "identity_verified": True, "account": {
            "account_id": broker.account_id, "cash": 1000, "frozen": 0, "available": 1000,
        },
    })
    setattr(broker, source_method, Mock(return_value=[item]))
    broker._read_balance_snapshot = Mock(return_value={
        "account_id": "ACCT-1", "cash": 1000, "frozen": 0, "available": 1000,
    })
    result = getattr(broker, evidence_method)(item[item_key])
    assert result == item
    getattr(broker, source_method).assert_called_once_with(persist_reports=False)

    broker.get_reconciliation_snapshot.return_value = {
        "identity_verified": False, "account": {"account_id": broker.account_id},
    }
    with pytest.raises(RuntimeError, match="账户身份未通过"):
        getattr(broker, evidence_method)(item[item_key])
    assert getattr(broker, source_method).call_count == 1


@pytest.mark.parametrize("evidence_method,source_method,item_key,item", [
    ("get_trade_reconciliation_evidence", "get_trades", "id", {
        "id": "T-1", "order_id": "O-1", "symbol": "000001", "side": "buy",
        "quantity": 100, "price": 10, "amount": 1000,
    }),
    ("get_order_reconciliation_evidence", "get_orders", "id", {
        "id": "O-1", "symbol": "000001", "side": "buy", "quantity": 100,
        "price_type": "limit", "price": 10, "filled_qty": 40, "status": "cancelled",
    }),
])
def test_easytrader_report_evidence_rejects_account_change_after_read(
        evidence_method, source_method, item_key, item):
    from core.adapters.broker.easytrader_broker import EasytraderBroker

    broker = EasytraderBroker(broker_account_id="ACCT-1")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "identity_verified": True, "account": {
            "account_id": broker.account_id, "cash": 1000, "frozen": 0, "available": 1000,
        },
    })
    setattr(broker, source_method, Mock(return_value=[item]))
    broker._read_balance_snapshot = Mock(return_value={
        "account_id": "ACCT-OTHER", "cash": 1000, "frozen": 0, "available": 1000,
    })
    with pytest.raises(RuntimeError, match="账户或资金快照发生变化"):
        getattr(broker, evidence_method)(item[item_key])
    getattr(broker, source_method).assert_called_once_with(persist_reports=False)


def test_qmt_trade_conflict_resolution_rejects_cross_account_trade():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_trades.return_value = [
        {"account_id": "other", "traded_id": "T-1"},
    ]
    with pytest.raises(RuntimeError, match="资金账号不一致"):
        broker.get_trade_reconciliation_evidence("T-1")


@pytest.mark.parametrize("symbol,side,quantity,price_type,price", [
    ("000001", "buy", 100, "market", 10),
    ("000001", "buy", 100, "limit", 0),
    ("000001", "buy", 99, "limit", 10),
    ("bad", "buy", 100, "limit", 10),
    ("000001", "buy", 100, "limit", "nan"),
    ("000001", "buy", 100, "limit", 10.1234),
    ("000001", "buy", 100, "limit", "10.0000000000000000000000000000001"),
    ("000001", "buy", "1e100000", "limit", 10),
    ("000001", "short", 100, "limit", 10),
    ("000001.SH", "buy", 100, "limit", 10),
    ("999999", "buy", 100, "limit", 10),
])
def test_real_order_rejects_invalid_inputs(symbol, side, quantity, price_type, price):
    with pytest.raises(ValueError):
        validate_real_limit_order(symbol, side, quantity, price_type, price)


@pytest.mark.parametrize("field,value", [
    ("quantity", "100.5"),
    ("price", "10.0000000000000000000000000000001"),
    ("amount", "1000.0000000000000000000000000001"),
    ("commission", "0.30000000000000000000000000001"),
    ("symbol", "000001.SH"),
])
def test_broker_trade_rejects_unrepresentable_amounts_before_db(field, value):
    trade = {"id": "T-1", "order_id": "O-1", "symbol": "000001.SZ", "side": "buy",
             "quantity": 100, "price": 10, "amount": 1000, "commission": "0.30", "stamp_duty": 0}
    trade[field] = value
    with patch("core.data.repos.broker_trade_report.Database.transaction") as transaction:
        with pytest.raises(ValueError):
            BrokerTradeReportRepo.ingest("qmt", "QMT-TEST", trade, "query")
    transaction.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("limit_price", "10.0000000000000000000000000000001"),
    ("fee_cap", "0.30000000000000000000000000001"),
])
def test_local_intent_rejects_unrepresentable_money_before_reconciliation(field, value):
    detail = {"intent_id": "00000000-0000-4000-8000-000000000001", "signal_id": 1,
              "expected_account_version": 1, "quantity": 100, "limit_price": "10",
              "fee_cap": "1", "actor": "admin"}
    detail[field] = value
    with patch("core.data.repos.local_order_intent.LocalOrderIntentRepo.get_by_id") as existing:
        with pytest.raises(ValueError):
            LocalLedgerService.prepare_intent(detail)
    existing.assert_not_called()


def test_qmt_never_converts_market_to_fixed_price():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    with pytest.raises(ValueError, match="市价单暗转"):
        broker.place_order("000001", "test", "buy", 100, "market", 0)
    broker._trader.order_stock.assert_not_called()


def test_qmt_limit_order_records_actual_sent_parameters():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._stock_account = object()
    broker._trader = Mock()
    broker._trader.order_stock.return_value = 123
    broker._trader.query_stock_orders.return_value = []
    xtconstant = SimpleNamespace(STOCK_BUY=23, STOCK_SELL=24, FIX_PRICE=11)
    with patch.dict(sys.modules, {"xtquant": SimpleNamespace(xtconstant=xtconstant)}), \
         patch.object(broker, "_normalize_symbol", return_value="000001.SZ"), \
         patch.object(broker, "_persist_order"):
        order_id = broker.place_order("000001", "test", "buy", 100, "limit", 10.12)
    assert order_id == "123"
    assert broker._trader.order_stock.call_args.args[4:6] == (11, 10.12)
    order = broker._orders["123"]
    assert order["submission"]["client_request"]["price_type"] == "limit"
    assert order["submission"]["broker_request"]["order_type"] == 23
    assert order["submission"]["broker_request"]["price_type_name"] == "FIX_PRICE"
    assert order["submission"]["broker_request"]["price"] == 10.12
    assert order["submission"]["broker_response"]["order_id"] == "123"


def test_qmt_order_query_restores_submission_metadata_after_restart():
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._stock_account = object()
    raw_order = SimpleNamespace(
        account_id="test", order_id=456, stock_code="000001.SZ", order_type=23,
        order_status=48, order_volume=100, price_type=11, price=10.12,
        traded_volume=0, traded_price=0,
    )
    broker._trader = Mock()
    broker._trader.query_stock_orders.return_value = [raw_order]
    submission = {"client_request": {"price_type": "limit"},
                  "broker_request": {"method": "order_stock", "price_type": 11},
                  "broker_response": {"order_id": "456"}}
    durable_order = {
        "id": "456", "symbol": "000001.SZ", "name": "test", "side": "buy",
        "quantity": 100, "price_type": "limit", "price": 10.12, "filled_qty": 0,
        "filled_price": 0, "status": "pending", "submission": submission,
        "created_at": "2026-10-30 10:00:00", "updated_at": "2026-10-30 10:00:00",
    }
    with (
        patch.object(BrokerOrderReportRepo, "list_submission_metadata", return_value={"456": submission}),
        patch.object(BrokerOrderReportRepo, "list_current_states", return_value=[durable_order]),
        patch.object(BrokerOrderReportRepo, "list_unprojected", return_value=[]),
        patch.object(broker, "_record_order_report", return_value={
            "disposition": "applied", "status": "pending", "filled_qty": 0,
        }),
        patch.object(broker, "_persist_qmt_order", return_value=True),
    ):
        restored = broker.get_order("456")

    assert restored["submission"] == submission


def test_qmt_does_not_hide_submission_persistence_failure_as_order_failure():
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._stock_account = object()
    broker._trader = Mock()
    broker._trader.order_stock.return_value = 123
    xtconstant = SimpleNamespace(STOCK_BUY=23, STOCK_SELL=24, FIX_PRICE=11)
    with (
        patch.dict(sys.modules, {"xtquant": SimpleNamespace(xtconstant=xtconstant)}),
        patch.object(broker, "_normalize_symbol", return_value="000001.SZ"),
        patch.object(BrokerOrderReportRepo, "record_submission", side_effect=RuntimeError("db unavailable")),
        patch.object(broker, "_persist_order") as persist,
    ):
        with pytest.raises(RuntimeError, match="禁止直接重试"):
            broker.place_order("000001", "test", "buy", 100, "limit", 10.12)

    broker._trader.order_stock.assert_called_once()
    persist.assert_not_called()


def test_easytrader_limit_never_calls_market_method():
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    broker._trader.buy.return_value = "123"
    with patch.object(broker, "_persist_order"):
        order_id = broker.place_order("000001.SZ", "test", "buy", 100, "limit", 10.12)
    assert order_id == "123"
    broker._trader.buy.assert_called_once_with("000001", 10.12, 100)
    broker._trader.market_buy.assert_not_called()


def test_easytrader_surfaces_submission_persistence_failure_for_manual_reconciliation():
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.buy.return_value = "123"
    with (
        patch.object(BrokerOrderReportRepo, "record_submission", side_effect=RuntimeError("db unavailable")),
        patch.object(broker, "_persist_order") as persist,
    ):
        with pytest.raises(RuntimeError, match="禁止直接重试"):
            broker.place_order("000001.SZ", "test", "buy", 100, "limit", 10.12)

    broker._trader.buy.assert_called_once_with("000001", 10.12, 100)
    persist.assert_not_called()


def test_easytrader_requires_explicit_broker_account_identity_before_connect():
    broker = EasytraderBroker()

    with pytest.raises(RuntimeError, match="ACCOUNT_ID 未配置"):
        broker.connect()

    configured = EasytraderBroker(broker_account_id="ACCT-001")
    assert configured.account_id == "EASYT-ACCT-001"


def test_easytrader_account_uses_verified_balance_row_for_cash_components():
    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock(balance=[{
        "资金帐号": "ACCT-001", "可用资金": 1000, "参考市值": 3000,
        "总资产": 4200, "股份参考盈亏": 50,
    }])

    account = broker.get_account()

    assert account["status"] == "fresh"
    assert account["cash"] == 1200
    assert account["frozen"] == 200
    assert account["available"] == 1000
    assert account["market_value"] == 3000
    assert account["total_assets"] == 4200
    assert account["unrealized_pnl"] == 50


@pytest.mark.parametrize("balance", [
    [{"资金帐号": "OTHER", "可用资金": 100, "参考市值": 0, "总资产": 100}],
    [{"资金帐号": "ACCT-001", "可用资金": 101, "参考市值": 0, "总资产": 100}],
    [{"资金帐号": "ACCT-001", "可用资金": 100, "参考市值": 101, "总资产": 100}],
    [{"资金帐号": "ACCT-001", "可用资金": None, "参考市值": 0, "总资产": 100}],
    [{"资金帐号": "ACCT-001", "可用资金": 100, "参考市值": 0, "总资产": 100},
     {"资金帐号": "ACCT-001", "可用资金": 100, "参考市值": 0, "总资产": 100}],
])
def test_easytrader_account_rejects_unmatched_or_inconsistent_balance(balance):
    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock(balance=balance)

    account = broker.get_account()

    assert account["status"] == "unavailable"
    assert account["cash"] is None
    assert account["available"] is None


def test_easytrader_order_query_persists_raw_and_normalized_report():
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock()
    raw = {
        "委托编号": "O-1", "证券代码": "000001", "操作": "买入", "委托数量": 100,
        "成交数量": 0, "委托价格": 10, "成交价格": 0, "状态": "已报",
    }
    broker._trader.today_entrusts = [raw]
    submission = {"client_request": {"price_type": "limit"},
                  "broker_request": {"method": "buy", "args": ["000001", 10, 100]},
                  "broker_response": {"order_id": "O-1"}}
    with patch.object(BrokerOrderReportRepo, "ingest", return_value={
        "disposition": "applied", "status": "pending", "filled_qty": 0,
    }) as ingest, patch.object(BrokerOrderReportRepo, "list_submission_metadata",
                               return_value={"O-1": submission}), \
         patch.object(BrokerOrderReportRepo, "unresolved_count", return_value=0), \
         patch.object(BrokerOrderReportRepo, "list_current_states", return_value=[]), \
         patch.object(BrokerOrderReportRepo, "list_unprojected", return_value=[]):
        orders = broker.get_orders()

    assert orders[0]["status"] == "pending"
    assert orders[0]["report_disposition"] == "applied"
    assert orders[0]["recovered_from_persisted_report"] is False
    assert orders[0]["submission"] == submission
    ingest.assert_called_once()
    args, kwargs = ingest.call_args
    assert args[:2] == ("easytrader", "EASYT-ACCT-001")
    assert args[2]["status"] == "pending"
    assert args[3] == "query"
    assert kwargs == {"raw_status": "已报", "raw_fields": raw}


def test_easytrader_stale_order_report_is_marked_as_persisted_recovery():
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.today_entrusts = [{
        "委托编号": "O-STALE", "证券代码": "000001", "操作": "买入", "委托数量": 100,
        "成交数量": 0, "委托价格": 10, "成交价格": 0, "状态": "已报",
    }]
    durable = {
        "id": "O-STALE", "symbol": "000001", "name": "", "side": "buy", "quantity": 100,
        "price_type": "limit", "price": 10, "filled_qty": 0, "filled_price": 0,
        "status": "pending", "submission": None, "created_at": "2026-10-31 10:00:00",
        "updated_at": "2026-10-31 10:00:00",
    }
    with patch.object(BrokerOrderReportRepo, "ingest", return_value={
        "disposition": "stale", "status": "pending", "filled_qty": 0,
    }), patch.object(BrokerOrderReportRepo, "list_submission_metadata", return_value={}), \
         patch.object(BrokerOrderReportRepo, "unresolved_count", return_value=0), \
         patch.object(BrokerOrderReportRepo, "list_current_states", return_value=[durable]), \
         patch.object(BrokerOrderReportRepo, "list_unprojected", return_value=[]):
        orders = broker.get_orders()

    assert orders[0]["report_disposition"] == "stale"
    assert orders[0]["recovered_from_persisted_report"] is True


def test_easytrader_order_query_exposes_report_persistence_failure():
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.today_entrusts = [{
        "委托编号": "O-1", "证券代码": "000001", "操作": "买入", "委托数量": 100,
        "成交数量": 0, "委托价格": 10, "成交价格": 0, "状态": "已报",
    }]
    with patch.object(BrokerOrderReportRepo, "ingest", side_effect=RuntimeError("report database unavailable")):
        with pytest.raises(RuntimeError, match="report database unavailable"):
            broker.get_orders()


def test_easytrader_trade_query_persists_raw_report():
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock()
    raw = {
        "成交编号": "T-1", "委托编号": "O-1", "证券代码": "000001", "操作": "买入",
        "成交数量": 100, "成交价格": 10, "成交金额": 1000, "成交时间": "10:00:00",
    }
    broker._trader.today_trades = [raw]
    with patch.object(BrokerTradeReportRepo, "ingest", return_value={
        "trade_id": "T-1", "status": "recorded", "fees_complete": False,
    }) as ingest:
        trades = broker.get_trades()

    assert trades[0]["commission"] is None
    ingest.assert_called_once_with("easytrader", "EASYT-ACCT-001", trades[0], "query", raw_fields=raw)


@pytest.mark.parametrize("report_status", ["conflict", "unknown"])
def test_easytrader_trade_query_rejects_untrusted_persisted_report(report_status):
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker = EasytraderBroker(broker_account_id="ACCT-001")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.today_trades = [{
        "成交编号": "T-1", "委托编号": "O-1", "证券代码": "000001", "操作": "买入",
        "成交数量": 100, "成交价格": 10, "成交金额": 1000,
    }]
    with patch.object(BrokerTradeReportRepo, "ingest", return_value={"status": report_status}):
        with pytest.raises(RuntimeError, match="未核对或未知状态"):
            broker.get_trades()


def test_fake_broker_uses_stable_account_and_project_fee_defaults():
    from core.shared import config

    first = FakeBroker()
    second = FakeBroker()
    assert first.account_id == second.account_id == "FAKE-LOCAL"
    assert first._stamp_duty == config.PORTFOLIO_STAMP_DUTY
    assert first._commission_rate == config.PORTFOLIO_COMMISSION_RATE
    first.connect()
    assert first.get_account()["status"] == "degraded"
    assert first.get_account()["persistence"] == "memory_only"


def test_fake_broker_missing_quote_cannot_create_zero_price_fill():
    broker = FakeBroker(price_provider=lambda symbol: 0)
    broker.connect()
    with pytest.raises(RuntimeError, match="缺少有效实时价格"):
        broker.place_order("000001", "test", "buy", 100, "market", 0)
    assert broker.get_orders() == []
    assert broker.get_trades() == []
    assert broker._cash == 1000000


@pytest.mark.parametrize("symbol,side,quantity,price_type,price", [
    ("bad", "buy", 100, "limit", 10),
    ("000001", "buy", True, "limit", 10),
    ("000001", "buy", 100.5, "limit", 10),
    ("000001", "buy", 99, "limit", 10),
    ("000001", "buy", 100, "limit", 0),
    ("000001", "buy", 100, "limit", "nan"),
    ("000001", "buy", 100, "market", 10),
])
def test_fake_broker_rejects_invalid_order_before_mutation(symbol, side, quantity, price_type, price):
    broker = FakeBroker(price_provider=lambda code: 10)
    broker.connect()
    with pytest.raises(ValueError):
        broker.place_order(symbol, "test", side, quantity, price_type, price)
    assert broker.get_orders() == []
    assert broker.get_trades() == []


def test_fake_broker_reservation_includes_fees_and_cash_cannot_underfund_frozen():
    broker = FakeBroker(initial_cash=5000, price_provider=lambda code: 10)
    broker.connect()
    order_id = broker.place_order("000001", "test", "buy", 100, "limit", 8)
    order = broker.get_order(order_id)
    assert order["status"] == STATUS_PENDING
    assert order["reserved_cash"] == 805
    assert broker._frozen == 805
    with pytest.raises(ValueError, match="低于已冻结金额"):
        broker.set_cash(804.99)
    assert broker._cash == 5000
    broker.set_cash(805)
    assert broker._cash == 805
    assert broker.cancel_order(order_id) is True
    assert broker._frozen == 0


def test_fake_broker_enforces_limit_bands_and_marketable_limit_uses_quote_price():
    bounds = lambda code: {"limit_up": 11, "limit_down": 9}
    at_upper = FakeBroker(initial_cash=5000, price_provider=lambda code: 11,
                          price_limit_provider=bounds)
    at_upper.connect()
    upper_order_id = at_upper.place_order("000001", "test", "buy", 100, "market", 0)
    assert at_upper.get_order(upper_order_id)["status"] == "rejected"
    assert at_upper.get_trades() == []
    assert at_upper._cash == 5000

    sealed_buy_id = at_upper.place_order("000001", "test", "buy", 100, "limit", 11)
    assert at_upper.get_order(sealed_buy_id)["status"] == "pending"
    assert at_upper._frozen == at_upper.get_order(sealed_buy_id)["reserved_cash"]
    assert at_upper.cancel_order(sealed_buy_id) is True
    assert at_upper._frozen == 0

    at_lower = FakeBroker(initial_cash=5000, price_provider=lambda code: 9,
                          price_limit_provider=bounds)
    at_lower.connect()
    old_trade = {"id": "T-OLD", "order_id": "O-OLD", "symbol": "000001",
                 "name": "test", "side": "buy", "price": 10, "quantity": 100,
                 "amount": 1000, "commission": 5, "stamp_duty": 0,
                 "trade_time": "2026-10-28 10:00:00"}
    old_order = {"id": "O-OLD", "symbol": "000001", "name": "test", "side": "buy",
                 "quantity": 100, "filled_qty": 100, "filled_price": 10, "status": "filled"}
    at_lower.restore([old_order], [old_trade])
    sell_id = at_lower.place_order("000001", "test", "sell", 100, "market", 0)
    assert at_lower.get_order(sell_id)["status"] == "rejected"
    assert at_lower._holdings["000001"]["shares"] == 100
    assert at_lower.get_trades() == [{"id": "T-OLD", "order_id": "O-OLD", "symbol": "000001",
                                      "name": "test", "side": "buy", "price": 10, "quantity": 100,
                                      "amount": 1000, "commission": 5, "stamp_duty": 0,
                                      "trade_time": "2026-10-28 10:00:00"}]
    sell_limit_id = at_lower.place_order("000001", "test", "sell", 100, "limit", 9)
    assert at_lower.get_order(sell_limit_id)["status"] == "pending"
    assert at_lower._frozen_shares["000001"] == 100
    assert at_lower.cancel_order(sell_limit_id) is True
    assert at_lower._frozen_shares["000001"] == 0

    marketable = FakeBroker(initial_cash=5000, price_provider=lambda code: 10,
                            price_limit_provider=bounds)
    marketable.connect()
    filled_id = marketable.place_order("000001", "test", "buy", 100, "limit", 11)
    assert marketable.get_order(filled_id)["status"] == "filled"
    assert marketable.get_order(filled_id)["filled_price"] == 10

    invalid_limit_id = marketable.place_order("000001", "test", "buy", 100, "limit", 12)
    assert marketable.get_order(invalid_limit_id)["status"] == "rejected"


def test_fake_broker_uses_price_and_bands_from_one_quote_mapping():
    broker = FakeBroker(
        initial_cash=5000,
        price_provider=lambda code: {"price": 10, "limit_up": 11, "limit_down": 9},
        price_limits_configured=True,
    )
    broker.connect()
    order_id = broker.place_order("000001", "test", "buy", 100, "market", 0)
    assert broker.get_order(order_id)["status"] == "filled"
    assert broker.get_order(order_id)["filled_price"] == 10
    assert broker.get_trades()[0]["price"] == 10


def test_fake_broker_fails_closed_when_price_limit_snapshot_is_unavailable():
    broker = FakeBroker(initial_cash=5000, price_provider=lambda code: 10,
                        price_limit_provider=lambda code: None)
    broker.connect()
    order_id = broker.place_order("000001", "test", "buy", 100, "market", 0)
    assert broker.get_order(order_id)["status"] == "rejected"
    assert broker.get_trades() == []
    assert broker._cash == 5000

    limit_id = broker.place_order("000001", "test", "buy", 100, "limit", 9)
    assert broker.get_order(limit_id)["status"] == "pending"
    assert broker._frozen > 0
    assert broker.cancel_order(limit_id) is True
    assert broker._frozen == 0


@pytest.mark.parametrize("code,name,expected", [
    ("000001", "平安银行", ("110.00", "90.00")),
    ("000001", "ST平安", ("105.00", "95.00")),
    ("300001", "创业板ST样本", ("120.00", "80.00")),
    ("688001", "科创板ST样本", ("120.00", "80.00")),
    ("830001", "北交所样本", ("130.00", "70.00")),
])
def test_fake_price_provider_uses_one_fresh_quote_for_price_and_shared_board_rules(code, name, expected):
    from backend.api.trading import _make_price_provider

    provider = _make_price_provider()
    identity = {"list_status": "L", "list_date": date.today() - datetime.timedelta(days=60),
                "delist_date": None}
    with patch("backend.service.market_data.MarketDataService.get_live_quotes", return_value={
        "meta": {"freshness": "fresh"},
        "data": {code: {"price": 101, "prev_close": 100, "name": name}},
    }) as get_quotes, patch("core.data.Database.fetchone", return_value=identity):
        quote = provider(code)

    assert quote["price"] == 101
    assert (str(quote["limit_up"]), str(quote["limit_down"])) == expected
    get_quotes.assert_called_once_with([code], use_case="execution")


def test_fake_price_provider_fails_closed_for_recent_listing_and_stale_quote():
    from backend.api.trading import _make_price_provider

    provider = _make_price_provider()
    identity = {"list_status": "L", "list_date": date.today() - datetime.timedelta(days=10),
                "delist_date": None}
    with patch("backend.service.market_data.MarketDataService.get_live_quotes", return_value={
        "meta": {"freshness": "fresh"}, "data": {"000001": {"price": 100, "prev_close": 100, "name": "测试"}},
    }), patch("core.data.Database.fetchone", return_value=identity):
        quote = provider("000001")
        assert quote["price"] == 100
        assert quote["limit_up"] is None and quote["limit_down"] is None
    with patch("backend.service.market_data.MarketDataService.get_live_quotes", return_value={
        "meta": {"freshness": "stale"}, "data": {"000001": {"price": 100, "prev_close": 100, "name": "测试"}},
    }):
        quote = provider("000001")
        assert quote["price"] == 0
        assert quote["limit_up"] is None and quote["limit_down"] is None


def test_fake_broker_rejects_same_day_sell_and_uses_project_stamp_duty():
    broker = FakeBroker(initial_cash=5000, price_provider=lambda code: 10,
                        price_limit_provider=lambda code: {"limit_up": 11, "limit_down": 9})
    broker.connect()
    with patch.object(broker, "_persist_order"), patch.object(broker, "_persist_trade"):
        buy_id = broker.place_order("000001", "test", "buy", 100, "market", 0)
        sell_id = broker.place_order("000001", "test", "sell", 100, "market", 0)
    assert broker.get_order(buy_id)["status"] == "filled"
    assert broker.get_order(sell_id)["status"] == "rejected"
    assert broker._holdings["000001"]["avg_cost"] == 10.05
    broker._today_bought.clear()
    with patch.object(broker, "_persist_order"), patch.object(broker, "_persist_trade"):
        next_day_sell_id = broker.place_order("000001", "test", "sell", 100, "market", 0)
    trade = broker.get_trades()[-1]
    assert broker.get_order(next_day_sell_id)["status"] == "filled"
    assert trade["stamp_duty"] == 0.5


def test_fake_broker_restores_holdings_orders_and_today_t_plus_one_state():
    broker = FakeBroker(price_provider=lambda code: 10)
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trade = {"id": "T-1", "order_id": "O-1", "symbol": "000001", "name": "test",
             "side": "buy", "price": 10, "quantity": 100, "amount": 1000,
             "commission": 5, "stamp_duty": 0, "trade_time": created_at}
    filled_order = {"id": "O-1", "symbol": "000001", "name": "test", "side": "buy",
                    "quantity": 100, "filled_qty": 100, "filled_price": 10, "status": "filled"}
    broker.restore([filled_order], [trade])
    assert broker._holdings["000001"] == {"shares": 100, "avg_cost": 10.05}
    assert broker._today_bought["000001"] == 100
    broker._frozen = 805
    broker.restore([filled_order,
                    {"id": "O-2", "symbol": "000001", "name": "test", "side": "buy",
                     "quantity": 100, "price_type": "limit", "price": 8,
                     "filled_qty": 0, "filled_price": 0, "status": "pending"}], [trade])
    assert broker.get_order("O-2")["reserved_cash"] == 805


def test_trade_persistence_insert_matches_trade_schema_columns():
    from core.data import Database

    broker = FakeBroker()
    trade = {"id": "T-1", "order_id": "O-1", "symbol": "000001", "name": "test",
             "side": "buy", "price": 10, "quantity": 100, "amount": 1000,
             "commission": 5, "stamp_duty": 0, "trade_time": "2026-10-29 10:00:00"}
    cursor = Mock()
    cursor.fetchone.return_value = None
    with patch.object(Database, "is_available", return_value=True), \
         patch.object(Database, "transaction", side_effect=lambda operation: operation(cursor)):
        broker._persist_trade(trade)
    query, values = cursor.execute.call_args.args
    assert query.count("%s") == 15
    assert len(values) == 15


def test_broker_order_persistence_insert_matches_order_schema_columns():
    from core.adapters.broker.base import BaseBroker
    from core.data import Database

    class TestBroker(BaseBroker):
        @property
        def account_id(self):
            return "QMT-TEST"

    broker = TestBroker()
    order = {"id": "O-1", "symbol": "000001.SZ", "name": "PingAn", "side": "buy",
             "quantity": 100, "price_type": "limit", "price": 10.5, "filled_qty": 0,
             "filled_price": 0, "status": "pending", "created_at": "2026-10-29 10:00:00",
             "updated_at": "2026-10-29 10:00:00"}
    cursor = Mock()
    cursor.fetchone.return_value = None
    with patch.object(Database, "is_available", return_value=True), \
         patch.object(Database, "transaction", side_effect=lambda operation: operation(cursor)):
        broker._persist_order(order)
    query, values = cursor.execute.call_args_list[1].args
    assert query.count("%s") == 16
    assert len(values) == 16
    assert values == ("O-1", "000001.SZ", "PingAn", "buy", 100, "limit", 10.5,
                      0, 0, "pending", "2026-10-29 10:00:00", "2026-10-29 10:00:00",
                      "QMT-TEST", "qmt", "live", "broker")


def test_qmt_failed_callback_report_replays_from_broker_query():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    raw_order = {"account_id": "test", "order_id": 123, "stock_code": "000001.SZ", "order_type": 23,
                 "order_volume": 100, "order_status": 48, "price": 10,
                 "traded_volume": 0, "traded_price": 0}
    broker._trader.query_stock_orders.return_value = [raw_order]
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest",
               side_effect=[RuntimeError("isolated write failure"),
                            {"disposition": "applied", "status": "pending", "filled_qty": 0}]) as ingest, \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_current_states", return_value=[]), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unprojected", return_value=[]), \
         patch.object(broker, "_persist_qmt_order", return_value=True), patch.object(broker, "_sync_positions"), \
         patch.object(broker, "_should_sync", return_value=True):
        _TraderCallback(broker).on_stock_order(raw_order)
        assert broker._report_persistence_error is not None
        assert broker._report_failed_orders == {"123"}
        broker._sync_orders()
    assert broker._report_persistence_error is None
    assert broker._report_failed_orders == set()
    assert ingest.call_count == 2
    assert ingest.call_args.args[3] == "query"


def test_qmt_stale_query_keeps_terminal_projection():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_orders.return_value = [{
        "account_id": "test", "order_id": 123, "stock_code": "000001.SZ", "order_type": 23,
        "order_volume": 100, "order_status": 48, "price": 10,
        "traded_volume": 0, "traded_price": 0,
    }]
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest",
               return_value={"disposition": "stale", "status": "cancelled", "filled_qty": 40}), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_current_states", return_value=[]), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unprojected", return_value=[]), \
         patch.object(broker, "_persist_qmt_order", return_value=True), \
         patch.object(broker, "_should_sync", return_value=True):
        broker._sync_orders()
    order = broker._orders["123"]
    assert order["status"] == "cancelled"
    assert order["filled_qty"] == 40


def test_qmt_order_sync_preserves_fractional_quantities_for_validation():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_orders.return_value = [{
        "account_id": "test", "order_id": 123, "stock_code": "000001.SZ", "order_type": 23,
        "order_volume": 100.5, "order_status": 48, "price_type": 11, "price": 10,
        "traded_volume": 0.5, "traded_price": 0,
    }]
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest",
               return_value={"disposition": "conflict", "status": "unknown", "filled_qty": 0}) as ingest, \
         patch.object(broker, "_should_sync", return_value=True):
        broker._sync_orders()
    recorded = ingest.call_args.args[2]
    assert recorded["quantity"] == 100.5
    assert recorded["filled_qty"] == 0.5
    assert broker._orders == {}


def test_qmt_reconciliation_snapshot_preserves_raw_order_quantities():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_positions.return_value = []
    broker._trader.query_stock_orders.return_value = [{
        "account_id": "test", "order_id": 123, "stock_code": "000001.SZ", "order_type": 23,
        "order_volume": 100.5, "order_status": 48, "price_type": 11, "price": 10,
        "traded_volume": 0.5,
    }]
    with patch.object(broker, "get_account", return_value={
        "status": "fresh", "as_of": datetime.datetime.now().isoformat(),
    }):
        snapshot = broker.get_reconciliation_snapshot(check_report_integrity=False)
    assert snapshot["orders"][0]["quantity"] == 100.5
    assert snapshot["orders"][0]["filled_qty"] == 0.5


@pytest.mark.parametrize("can_use_volume", [80, None])
def test_qmt_reconciliation_snapshot_maps_sellable_quantity_without_fallback(can_use_volume):
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_positions.return_value = [{
        "account_id": "test", "stock_code": "000001.SZ", "volume": 100,
        "can_use_volume": can_use_volume,
    }]
    broker._trader.query_stock_orders.return_value = []
    with patch.object(broker, "get_account", return_value={
        "status": "fresh", "as_of": datetime.datetime.now().isoformat(),
    }):
        snapshot = broker.get_reconciliation_snapshot(check_report_integrity=False)

    assert snapshot["positions"] == [{
        "symbol": "000001.SZ", "shares": 100, "sellable": can_use_volume,
    }]


def test_report_resolution_requires_current_terminal_broker_snapshot():
    broker = QmtBroker("unused", "test")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "identity_verified": True,
        "account": {"account_id": broker.account_id},
        "orders": [{"id": "123", "status": "pending", "filled_qty": 40}],
    })
    with patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalLedgerService, "_get_report_scope", return_value=("qmt", broker.account_id)), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.get_by_key",
               return_value={"broker": "qmt", "broker_account_id": broker.account_id,
                             "order_id": "123", "resolved_at": None}), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.resolve") as resolve:
        with pytest.raises(ValueError, match="尚未终结"):
            LocalLedgerService.resolve_broker_report("key", "admin", "核对券商终态")
    resolve.assert_not_called()


@pytest.mark.parametrize("raw_price_type,expected_price_type", [(11, "limit"), (44, "market")])
def test_qmt_replay_recovers_missed_terminal_order_report(raw_price_type, expected_price_type):
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_orders.return_value = [{
        "account_id": "test", "order_id": 123, "stock_code": "000001.SZ", "order_type": 23,
        "order_volume": 100, "order_status": 56, "price_type": raw_price_type, "price": 10,
        "traded_volume": 100, "traded_price": 9.9,
    }]
    broker._trader.query_stock_trades.return_value = []
    with patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.ingest",
               return_value={"disposition": "applied", "status": "filled", "filled_qty": 100}) as ingest, \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_current_states", return_value=[]), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unprojected", return_value=[]), \
         patch.object(broker, "_persist_qmt_order", return_value=True), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.unresolved_count",
               return_value=0), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.get_health",
               return_value={"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0}):
        result = broker.replay_order_reports()
    assert result == {"observed_orders": 1, "observed_trades": 0,
                      "unresolved_reports": 0,
                      "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0,
                                              "unassigned": 0}}
    assert broker._orders["123"]["status"] == "filled"
    assert ingest.call_args.args[2]["price_type"] == expected_price_type


def test_qmt_trade_callback_keeps_unknown_fees_and_persists_once():
    broker = QmtBroker("unused", "test")
    raw_trade = {"account_id": "test", "traded_id": "T-1", "order_id": 123, "stock_code": "000001.SZ",
                 "order_type": 23, "traded_price": 9.9, "traded_volume": 40,
                 "traded_amount": 396, "traded_time": "10:00:00"}
    with patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.ingest",
               side_effect=[{"status": "recorded", "fees_complete": False},
                            {"status": "duplicate", "fees_complete": False}]) as ingest, \
         patch.object(broker, "_persist_trade") as persist:
        callback = _TraderCallback(broker)
        callback.on_stock_trade(raw_trade)
        callback.on_stock_trade(raw_trade)
    assert ingest.call_count == 2
    assert len(broker._trades) == 1
    assert broker._trades[0]["commission"] is None
    assert broker._trades[0]["stamp_duty"] is None
    persist.assert_called_once()


def test_qmt_trade_callback_and_reconciliation_use_xttrade_commission():
    xttype = pytest.importorskip("xtquant.xttype")
    trade = xttype.XtTrade(
        "test", "000001.SZ", 23, "T-2", 20261029103045, 9.9, 40, 396,
        123, "sys-123", "", "", 0, 0, 1.23, "SH123", "测试股",
    )
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = xttype.StockAccount("test")
    broker._trader.query_stock_trades.return_value = [trade]

    with patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.ingest",
               return_value={"status": "recorded", "fees_complete": False}) as ingest, \
         patch.object(broker, "_persist_trade"):
        _TraderCallback(broker).on_stock_trade(trade)
        evidence = broker.get_trade_reconciliation_evidence("T-2")

    assert broker._trades[0]["commission"] == 1.23
    assert broker._trades[0]["stamp_duty"] is None
    assert broker._trades[0]["trade_time"] is None
    assert evidence["commission"] == 1.23
    assert evidence["stamp_duty"] is None
    callback_raw_fields = ingest.call_args_list[0].args[4]
    assert callback_raw_fields["commission"] == 1.23
    assert callback_raw_fields["traded_time"] == 20261029103045


def test_qmt_trade_query_sync_maps_xttrade_commission():
    xttype = pytest.importorskip("xtquant.xttype")
    trade = xttype.XtTrade(
        "test", "000001.SZ", 23, "T-3", 20261029103100, 9.9, 40, 396,
        124, "sys-124", "", "", 0, 0, 0.8, "SH123", "测试股",
    )
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = xttype.StockAccount("test")
    broker._trader.query_stock_trades.return_value = [trade]

    with patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.ingest",
               return_value={"status": "recorded", "fees_complete": False}) as ingest, \
         patch.object(broker, "_persist_trade") as persist:
        broker._sync_trades()

    assert broker._trades[0]["commission"] == 0.8
    assert broker._trades[0]["trade_time"] is None
    assert ingest.call_args.args[4]["commission"] == 0.8
    persist.assert_called_once()


def test_easytrader_unknown_order_status_does_not_become_pending():
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    broker._trader.position = []
    broker._trader.today_entrusts = [{
        "委托编号": "123", "证券代码": "000001", "操作": "买入",
        "委托数量": 100, "成交数量": 0, "委托价格": 10, "成交价格": 0,
        "状态": "未识别状态",
    }]
    assert broker.get_orders(persist_reports=False)[0]["status"] == "unknown"
    broker._trader.today_entrusts = None
    with pytest.raises(RuntimeError, match="委托查询不可用"):
        broker.get_orders()


@pytest.mark.parametrize("updates", [
    {"委托数量": 100.5},
    {"委托数量": True},
    {"委托数量": None},
    {"成交数量": None},
    {"成交数量": 101},
    {"操作": None},
    {"操作": "买入卖出"},
])
def test_easytrader_invalid_order_report_rejects_entire_snapshot(updates):
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    row = {"委托编号": "123", "证券代码": "000001", "操作": "买入",
           "委托数量": 100, "成交数量": 0, "委托价格": 10, "成交价格": 0,
           "状态": "已报"}
    row.update(updates)
    broker._trader.today_entrusts = [row]
    with pytest.raises(RuntimeError, match="委托证券、方向、股数或价格无效"):
        broker.get_orders()


@pytest.mark.parametrize("status,filled_qty", [
    ("已成", 0),
    ("已成", 50),
    ("部成", 0),
    ("部成", 100),
])
def test_easytrader_order_status_must_match_fill_quantity(status, filled_qty):
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    broker._trader.position = []
    broker._trader.today_entrusts = [{
        "委托编号": "123", "证券代码": "000001", "操作": "买入",
        "委托数量": 100, "成交数量": filled_qty, "委托价格": 10,
        "成交价格": 10 if filled_qty else 0, "状态": status,
    }]

    with patch.object(broker, "get_account", return_value={"status": "fresh"}), \
         pytest.raises(RuntimeError, match="委托证券、方向、股数或价格无效"):
        broker.get_reconciliation_snapshot()


@pytest.mark.parametrize("raw_status,filled_qty,expected_status", [
    ("已报", 0, "pending"),
    ("部成", 40, "partial"),
    ("已成", 100, "filled"),
])
def test_easytrader_valid_order_status_matches_fill_quantity(raw_status, filled_qty, expected_status):
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    broker._trader.today_entrusts = [{
        "委托编号": "123", "证券代码": "000001", "操作": "买入",
        "委托数量": 100, "成交数量": filled_qty, "委托价格": 10,
        "成交价格": 10 if filled_qty else 0, "状态": raw_status,
    }]

    assert broker.get_orders(persist_reports=False)[0]["status"] == expected_status


@pytest.mark.parametrize("updates", [
    {"成交数量": 40.5},
    {"操作": None},
    {"操作": "买入卖出"},
    {"成交编号": None, "合同编号": None},
    {"委托编号": None},
])
def test_easytrader_invalid_trade_report_rejects_entire_snapshot(updates):
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    row = {"成交编号": "T-1", "委托编号": "123", "证券代码": "000001",
           "操作": "买入", "成交数量": 40, "成交价格": 10, "成交金额": 400}
    row.update(updates)
    broker._trader.today_trades = [row]
    with pytest.raises(RuntimeError, match="成交证券、方向、股数、金额或费用无效"):
        broker.get_trades()


def test_easytrader_trade_report_keeps_missing_fees_unknown_and_filters_requested_symbol():
    broker = EasytraderBroker()
    broker._connected = True
    broker._trader = Mock()
    broker._trader.today_trades = [
        {"成交编号": "T-1", "委托编号": "123", "证券代码": "000001",
         "操作": "买入", "成交数量": 40, "成交价格": 10, "成交金额": 400},
        {"成交编号": "T-2", "委托编号": "124", "证券代码": "000002",
         "操作": "卖出", "成交数量": 100, "成交价格": 11, "成交金额": 1100,
         "手续费": 0.33, "印花税": 1.1},
    ]
    trades = broker.get_trades(symbol="000001")
    assert len(trades) == 1
    assert trades[0]["id"] == "T-1"
    assert trades[0]["commission"] is None
    assert trades[0]["stamp_duty"] is None


@pytest.mark.parametrize("row_name,row", [
    ("position", {"证券代码": "000001", "股票余额": "100.5"}),
    ("order", {"委托编号": "123", "证券代码": "000001", "操作": "买入",
                "委托数量": 100.5, "成交数量": 0, "委托价格": 10, "成交价格": 0,
                "状态": "已报"}),
])
def test_easytrader_reconciliation_snapshot_does_not_bypass_report_validation(row_name, row):
    broker = EasytraderBroker(broker_account_id="TEST-004")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.balance = [{
        "资金帐号": "TEST-004", "可用资金": 1000, "参考市值": 0, "总资产": 1000,
    }]
    broker._trader.position = [row] if row_name == "position" else []
    broker._trader.today_entrusts = [row] if row_name == "order" else []
    with pytest.raises(RuntimeError, match="Easytrader (持仓证券代码或股数无效|委托证券、方向、股数或价格无效)"):
        broker.get_reconciliation_snapshot()


def test_easytrader_reconciliation_snapshot_verifies_reported_funding_account():
    broker = EasytraderBroker(broker_account_id="TEST-005")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.balance = [{
        "资金帐号": "TEST-005", "可用资金": 1000, "参考市值": 0, "总资产": 1000,
    }]
    broker._trader.position = [{"证券代码": "000001", "股票余额": "100.0", "股份可用": 80}]
    broker._trader.today_entrusts = [{
        "委托编号": "123", "证券代码": "000001", "操作": "买入",
        "委托数量": 100, "成交数量": 0, "委托价格": 10, "成交价格": 0,
        "状态": "未识别状态",
    }]
    broker._trader.today_trades = []
    with patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.get_health", return_value={
        "conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0,
    }):
        snapshot = broker.get_reconciliation_snapshot()
    assert snapshot["identity_verified"] is True
    assert snapshot["positions"] == [{"symbol": "000001", "shares": 100, "sellable": 80}]
    assert snapshot["orders"] == [{"id": "123", "status": "unknown"}]


def test_easytrader_reconciliation_rejects_cash_change_during_snapshot():
    broker = EasytraderBroker(broker_account_id="TEST-007")
    broker._connected = True
    broker._trader = Mock(position=[], today_entrusts=[], today_trades=[])
    first = {
        "account_id": "TEST-007", "cash": Decimal("1000"), "frozen": Decimal("0"),
        "available": Decimal("1000"), "market_value": Decimal("0"),
        "total_assets": Decimal("1000"), "unrealized_pnl": None,
    }
    second = {**first, "cash": Decimal("900"), "available": Decimal("900"),
              "total_assets": Decimal("900")}
    with patch.object(broker, "_read_balance_snapshot", side_effect=[first, second]):
        with pytest.raises(RuntimeError, match="对账期间资金余额发生变化"):
            broker.get_reconciliation_snapshot()


@pytest.mark.parametrize("balance_rows", [
    [],
    [{}],
    [{"资金帐号": "OTHER-ACCOUNT"}],
    [{"资金帐号": "TEST-006"}, {"资金帐号": "OTHER-ACCOUNT"}],
])
def test_easytrader_reconciliation_rejects_missing_or_mismatched_funding_account(balance_rows):
    broker = EasytraderBroker(broker_account_id="TEST-006")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.balance = balance_rows

    with pytest.raises(RuntimeError, match="Easytrader 资产快照不可用"):
        broker.get_reconciliation_snapshot()

    broker._trader.position.assert_not_called()


def test_qmt_order_sync_cannot_clear_trade_query_failure():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_orders.return_value = []
    broker._trader.query_stock_trades.return_value = None
    with patch.object(broker, "_should_sync", return_value=True), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_current_states", return_value=[]), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unprojected", return_value=[]):
        broker._sync_trades()
        assert "trades" in broker._report_query_errors
        broker._sync_orders()
        assert broker._report_persistence_error is not None
        broker._trader.query_stock_trades.return_value = []
        broker._sync_trades()
    assert broker._report_persistence_error is None


def test_qmt_trade_query_replays_failed_projection_even_when_report_is_duplicate():
    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    raw_trade = SimpleNamespace(
        account_id="test", traded_id="T-REPLAY", order_id="O-1", stock_code="000001.SZ",
        order_type=23, traded_price=10, traded_volume=100, traded_amount=1000,
        commission=1, traded_time="10:00:00",
    )
    broker._trader.query_stock_trades.side_effect = [[raw_trade], []]
    unprojected = {"id": "T-REPLAY", "order_id": "O-1", "symbol": "000001.SZ", "side": "buy",
                   "price": Decimal("10.000"), "quantity": 100, "amount": Decimal("1000.00"),
                   "commission": Decimal("1.00"), "stamp_duty": None, "trade_time": None}
    with patch.object(broker, "_should_sync", return_value=True), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.ingest",
               return_value={"status": "duplicate", "fees_complete": True}), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.list_unprojected",
               side_effect=[[], [unprojected]]), \
         patch.object(broker, "_persist_trade", side_effect=[RuntimeError("projection write failed"), None]) as persist:
        broker._sync_trades()
        assert broker._report_persistence_error == "RuntimeError: QMT 成交投影未持久化"
        assert "trade:T-REPLAY" in broker._report_failed_orders

        broker._sync_trades()

    assert persist.call_count == 2
    assert persist.call_args.kwargs == {"required": True}
    assert broker._report_persistence_error is None
    assert "trade:T-REPLAY" not in broker._report_failed_orders


def test_live_screening_uses_local_ledger_and_remains_preview_only():
    from backend.api import trading

    params = SimpleNamespace(use_timing=False, pool="a500", top_n=5,
                             code_prefixes=[], sizing="score_weighted",
                             max_position_pct=0.2, max_sector_pct=0.3,
                             stop_loss_pct=-0.08, commission_rate=0.0003,
                             slippage_rate=0.0001, stamp_duty=0.0005)
    holding = {"code": "000001", "name": "测试", "shares": 100, "cost_price": 9,
               "buy_date": None}
    screener = Mock()
    screener.screen.return_value = {"candidates": [], "regime": {}}
    advisor = Mock()
    advisor.generate_advice_from_screening.return_value = {"orders": []}
    with patch("core.data.repos.local_ledger.LocalLedgerRepo.get_account",
               return_value={"cash": Decimal("1200"), "frozen": Decimal("200"),
                             "available": Decimal("1000"), "status": "active", "version": 3}), \
         patch("core.shared.trading_time.get_latest_trading_day", return_value=date(2026, 10, 27)), \
         patch("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity", return_value={"ok": True}), \
         patch("core.data.connection.Database.fetchall", return_value=[holding]) as holdings_query, \
         patch("backend.api.trading.get_broker", side_effect=AssertionError("不得读取券商余额")), \
         patch("backend.service.market_data.MarketDataService.get_live_quotes",
               return_value={"meta": {"freshness": "fresh"},
                             "data": {"000001": {"price": 10, "pct_chg": 0}}}), \
         patch("core.domain.portfolio.consistency.check_panic", return_value=False), \
         patch("core.domain.screening._screener.get_screener", return_value=screener), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", return_value=advisor), \
         patch("core.data.fetcher.DataFetcher", return_value=Mock()):
        trading._run_live_screening("local-ledger-preview", params)
    result = trading._live_results.pop("local-ledger-preview")
    assert result["status"] == "done"
    assert result["current_state"]["account"]["cash"] == 1000
    assert result["current_state"]["account"]["total_assets"] == 2200
    assert result["current_state"]["holdings"][0]["weight"] == pytest.approx(1000 / 2200)
    assert result["execution"]["executable"] is False
    assert holdings_query.call_args.args[1] == ("LOCAL-LIVE",)
    assert advisor.generate_advice_from_screening.call_args.kwargs["account_data"]["cash"] == 1000


def test_live_screening_stops_when_local_holding_has_no_fresh_price():
    from backend.api import trading

    params = SimpleNamespace(use_timing=False)
    with patch("core.data.repos.local_ledger.LocalLedgerRepo.get_account",
               return_value={"cash": Decimal("1000"), "available": Decimal("1000"),
                             "status": "active", "version": 1}), \
         patch("core.shared.trading_time.get_latest_trading_day", return_value=date(2026, 10, 27)), \
         patch("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity", return_value={"ok": True}), \
         patch("core.data.connection.Database.fetchall",
               return_value=[{"code": "000001", "shares": 100}]), \
         patch("backend.service.market_data.MarketDataService.get_live_quotes",
               return_value={"meta": {"freshness": "stale"}, "data": {"000001": {"price": 10}}}), \
         patch("backend.api.trading.get_broker", side_effect=AssertionError("不得读取券商余额")):
        trading._run_live_screening("missing-local-price", params)
    result = trading._live_results.pop("missing-local-price")
    assert result["status"] == "error"
    assert "本地持仓行情不完整" in result["error"]


def test_local_advice_snapshot_rejects_account_change_during_price_read():
    first = {"cash": Decimal("1200"), "frozen": Decimal("200"),
             "available": Decimal("1000"), "status": "active", "version": 3}
    second = {**first, "cash": Decimal("1300"), "version": 4}
    with patch("core.data.repos.local_ledger.LocalLedgerRepo.get_account",
               side_effect=[first, second]), \
         patch("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity", return_value={"ok": True}), \
         patch("core.data.connection.Database.fetchall",
               return_value=[{"code": "000001", "name": "测试", "shares": 100,
                              "cost_price": 9, "buy_date": None}]), \
         patch("backend.service.market_data.MarketDataService.get_live_quotes",
               return_value={"meta": {"freshness": "fresh"},
                             "data": {"000001": {"price": 10}}}):
        with pytest.raises(RuntimeError, match="读取建议期间已变化"):
            LocalLedgerService.get_advice_snapshot()


def test_local_advice_snapshot_exposes_total_available_and_frozen_cash_separately():
    account = {"cash": Decimal("1200"), "frozen": Decimal("200"),
               "available": Decimal("1000"), "status": "active", "version": 3}
    with patch("core.data.repos.local_ledger.LocalLedgerRepo.get_account",
               side_effect=[account, account.copy()]), \
         patch("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity",
               return_value={"ok": True, "account_version": 3}), \
         patch("core.data.connection.Database.fetchall",
               return_value=[{"code": "000001", "name": "测试", "shares": 100,
                              "cost_price": 9, "buy_date": None}]), \
         patch("backend.service.market_data.MarketDataService.get_live_quotes",
               return_value={"meta": {"freshness": "fresh"},
                             "data": {"000001": {"price": 10}}}):
        snapshot = LocalLedgerService.get_advice_snapshot()

    assert snapshot["account"] == {
        "account_id": "LOCAL-LIVE", "cash": 1000.0, "available_cash": 1000.0,
        "total_cash": 1200.0, "frozen_cash": 200.0, "market_value": 1000.0,
        "total_assets": 2200.0, "status": "active", "quote_status": "fresh",
        "cash_source": "local_ledger", "version": 3, "broker_account_id": None,
    }


def test_weekly_advice_reads_local_snapshot_and_is_preview_only():
    from backend.api.screening import generate_weekly_advice

    snapshot = {"account": {"cash": 1000, "total_assets": 2200, "status": "active", "version": 3},
                "holdings": [{"symbol": "000001", "shares": 100}]}
    advisor = Mock()
    advisor.generate_advice.return_value = {"to_buy": [], "to_sell": []}
    with patch("backend.service.local_ledger.LocalLedgerService.get_advice_snapshot",
               return_value=snapshot), \
         patch("core.shared.trading_time.get_latest_trading_day", return_value=date(2026, 10, 27)), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", return_value=advisor):
        result = asyncio.run(generate_weekly_advice(async_mode=False, eval_date=None))
    assert result["data"]["execution"]["executable"] is False
    assert result["data"]["account_source"] == "local_ledger"
    assert result["data"]["account_version"] == 3
    assert advisor.generate_advice.call_args.kwargs["account_snapshot"] == snapshot


def test_async_weekly_advice_reads_local_snapshot_and_is_preview_only():
    from backend.api import screening

    snapshot = {"account": {"cash": 1000, "total_assets": 2200, "status": "active", "version": 3},
                "holdings": [{"symbol": "000001", "shares": 100}]}
    advisor = Mock()
    advisor.generate_advice.return_value = {"to_buy": [], "to_sell": []}
    with patch("backend.service.local_ledger.LocalLedgerService.get_advice_snapshot",
               return_value=snapshot), \
         patch("core.shared.trading_time.get_latest_trading_day", return_value=date(2026, 10, 27)), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", return_value=advisor), \
         patch("backend.api.screening.thread_pool.default.submit",
               side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)), \
         patch("backend.api.screening.threading.Timer"):
        started = asyncio.run(screening.generate_weekly_advice(async_mode=True, eval_date=None))
    job = screening._rebalance_jobs.pop(started["data"]["job_id"])
    assert job["status"] == "done"
    assert job["result"]["execution"]["executable"] is False
    assert job["result"]["account_source"] == "local_ledger"
    assert job["result"]["account_version"] == 3
    assert advisor.generate_advice.call_args.kwargs["account_snapshot"] == snapshot


def test_weekly_advice_rejects_historical_date_without_account_snapshot():
    from fastapi import HTTPException
    from backend.api.screening import generate_weekly_advice

    with patch("core.shared.trading_time.get_latest_trading_day", return_value=date(2026, 10, 27)):
        with pytest.raises(HTTPException) as error:
            asyncio.run(generate_weekly_advice(async_mode=False, eval_date="2026-10-26"))
    assert error.value.status_code == 409
    assert "历史日期缺少当时账户" in error.value.detail


def test_weekly_advisor_injected_local_snapshot_skips_broker_read():
    from core.domain.portfolio.rebalance import WeeklyRebalanceAdvisor

    advisor = WeeklyRebalanceAdvisor.__new__(WeeklyRebalanceAdvisor)
    advisor.pool = "a500"
    advisor.sectors = None
    advisor.code_prefixes = []
    advisor.sizing = "score_weighted"
    advisor.max_turnover = 0.3
    advisor._allocator = Mock(config=SimpleNamespace(max_turnover=0.3))
    advisor._cooldown_cleanup = Mock()
    advisor._ensure_broker = Mock(side_effect=AssertionError("不得读取券商账户"))
    advisor.screener = Mock()
    advisor.screener.get_qlib_predictions.return_value = (None, {})
    result = advisor.generate_advice(
        account_snapshot={"account": {"status": "active", "cash": 1000,
                                     "total_assets": 2000}, "holdings": []},
    )
    assert "未找到可用的 Qlib" in result["error"]
    advisor._ensure_broker.assert_not_called()


def test_broker_report_history_is_scoped_to_current_qmt_account():
    broker = QmtBroker("unused", "TEST")
    report_key = "a" * 64
    with patch("core.data.repos.local_ledger.LocalLedgerRepo.get_account", return_value={
        "broker_account_id": broker.account_id,
    }), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[{"action": "resolved"}]) as query:
        history = LocalLedgerService.list_broker_report_history("trade", report_key, 20)
    assert history == [{"action": "resolved"}]
    assert query.call_args.args[1] == ("qmt", "QMT-TEST", "trade", report_key, 20)
    with pytest.raises(ValueError, match="类型或编号无效"):
        LocalLedgerService.list_broker_report_history("trade", "not-a-key")
