import asyncio
import datetime
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.service.local_ledger import LocalLedgerService
from backend.service.rebalance import RebalanceService, _check_risk_signal
from core.data.connection import Database
from core.data.repos.local_ledger import LocalLedgerRepo, MISSING_SIGNAL_PLAN
from core.data.repos.local_order_intent import LocalOrderIntentRepo
from core.data.repos.rebalance import RebalanceSignalRepo


DETAIL = {
    "shares": 100, "price": 10, "commission": "0.30", "stamp_duty": "0",
    "request_id": "00000000-0000-4000-8000-000000000001",
    "expected_account_version": 1, "actor": "admin",
}


def _broker_adapter_mock(account_id):
    from core.adapters.broker import EasytraderBroker, QmtBroker

    broker_type = EasytraderBroker if account_id.startswith("EASYT-") else QmtBroker
    broker = broker_type.__new__(broker_type)
    broker._account_id = account_id
    broker.get_reconciliation_snapshot = Mock()
    return broker


def _ready_reconciliation(**overrides):
    now = datetime.datetime.now()
    snapshot = {
        "can_bind": True,
        "can_execute": True,
        "broker_account_id": "QMT-ONE",
        "local_version": 2,
        "broker_status": "fresh",
        "broker_identity_verified": True,
        "missing_items": [],
        "position_differences": [],
        "sellable_missing_codes": [],
        "pending_orders": [],
        "pending_journal_orders": [],
        "unresolved_order_reports": [],
        "pending_local_intent": False,
        "broker_positions": {},
        "broker_sellable": {},
        "local_positions": {},
        "cash_difference": 0,
        "cash_balance_difference": 0,
        "frozen_difference": 0,
        "broker_cash": 1000,
        "broker_frozen": 0,
        "broker_available": 1000,
        "broker_trade_report_health": {
            "conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0,
        },
        "broker_received_at": now.isoformat(),
        "broker_as_of": now.isoformat(),
        "broker_snapshot_started_at": (now - datetime.timedelta(seconds=1)).isoformat(),
        "broker_snapshot_completed_at": now.isoformat(),
    }
    snapshot.update(overrides)
    return snapshot


@pytest.mark.parametrize("broker_account_id,broker_key", [
    ("QMT-ACCOUNT-1", "qmt"),
    ("EASYT-ACCOUNT-1", "easytrader"),
])
def test_local_intent_broker_key_uses_bound_account_namespace(broker_account_id, broker_key):
    assert LocalOrderIntentRepo._broker_key(broker_account_id) == broker_key


@pytest.mark.parametrize("broker_account_id", ["", "BROKER-1", "EASYT-UNVERIFIED"])
def test_local_intent_broker_key_rejects_unknown_account_namespace(broker_account_id):
    with pytest.raises(ValueError, match="券商账号标识无效"):
        LocalOrderIntentRepo._broker_key(broker_account_id)


def test_easytrader_broker_evidence_reads_easytrader_report_namespace():
    intent = {
        "intent_id": "intent-1", "account_id": "LOCAL-LIVE",
        "broker_account_id": "EASYT-ACCOUNT-1", "broker_order_id": "O-1",
    }
    with patch.object(LocalOrderIntentRepo, "get_by_id", return_value=intent), \
         patch.object(Database, "fetchone", side_effect=[{"order_id": "O-1"}, {"n": 0}, {"n": 0}]) as fetchone, \
         patch.object(Database, "fetchall", return_value=[]):
        evidence = LocalOrderIntentRepo.get_broker_evidence("intent-1")

    assert evidence["order"] == {"order_id": "O-1"}
    assert "broker=%s" in fetchone.call_args_list[0].args[0]
    assert fetchone.call_args_list[0].args[1] == ("easytrader", "EASYT-ACCOUNT-1", "O-1")
    assert evidence["order_conflicts"] == evidence["trade_conflicts"] == evidence["conflicts"] == 0
    assert fetchone.call_args_list[2].args[1] == ("easytrader", "EASYT-ACCOUNT-1", "O-1")


def test_easytrader_order_attach_reads_easytrader_report_namespace():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1},
        {"account_id": "LOCAL-LIVE", "broker_account_id": "EASYT-ACCOUNT-1",
         "broker_order_id": None, "state": "prepared", "account_version": 1,
         "side": "buy", "code": "000001", "quantity": 100, "limit_price": 10,
         "intent_age_seconds": 10},
        None,
    ]
    with patch("core.data.repos.local_order_intent.Database.transaction",
               side_effect=lambda operation: operation(cursor)), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         pytest.raises(ValueError, match="身份、限价或数量"):
        LocalOrderIntentRepo.attach_broker_order("intent-1", "O-1", "admin")

    assert cursor.execute.call_args_list[0].args[1] == ("LOCAL-LIVE",)
    assert cursor.execute.call_args_list[2].args[1] == ("easytrader", "EASYT-ACCOUNT-1", "O-1")


def test_attach_broker_order_invalidates_other_old_version_signals():
    cursor = Mock()
    cursor.rowcount = 1
    created_at = datetime.datetime.now() - datetime.timedelta(seconds=10)
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 4, "cash": 1000},
        {"account_id": "LOCAL-LIVE", "broker_account_id": "QMT-ACCOUNT-1",
         "broker_order_id": None, "state": "prepared", "account_version": 4,
         "side": "buy", "code": "000001", "quantity": 100, "limit_price": 10,
         "intent_age_seconds": 10, "created_at": created_at, "signal_id": 7},
        {"symbol": "000001", "side": "buy", "quantity": 100,
         "price_type": "limit", "limit_price": 10, "status": "submitted"},
        {"n": 0},
        {"first_seen": created_at},
        {"batch_id": "batch-1", "account_id": "LOCAL-LIVE", "execution_status": "pending",
         "account_version": 4, "code": "000001", "planned_shares": 100, "filled_shares": 0},
    ]
    details = [{"signal_id": 92, "code": "600000", "side": "buy",
                "execution_status": "pending", "planned_shares": 200,
                "filled_shares": 0, "invalidated_remaining_shares": 200}]
    with patch("core.data.repos.local_order_intent.Database.transaction",
               side_effect=lambda operation: operation(cursor)), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch.object(RebalanceSignalRepo, "invalidate_stale_account_signals", return_value=details) as invalidate:
        result = LocalOrderIntentRepo.attach_broker_order("intent-1", "O-1", "admin")

    assert result["state"] == "submitted"
    assert result["invalidated_signal_ids"] == [92]
    assert result["invalidated_signal_details"] == details
    invalidate.assert_called_once_with(cursor, "LOCAL-LIVE", 5)
    event_insert = next(call for call in cursor.execute.call_args_list
                        if "INSERT INTO local_ledger_events" in call.args[0])
    assert event_insert.args[1][-2] == "admin"
    assert json.loads(event_insert.args[1][-1])["invalidated_signal_details"] == details


@pytest.mark.parametrize("intent_age_seconds", [61, -6, None, True])
def test_broker_order_attach_rejects_expired_or_unverifiable_prepared_intent(intent_age_seconds):
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1},
        {"account_id": "LOCAL-LIVE", "broker_account_id": "QMT-ACCOUNT-1",
         "broker_order_id": None, "state": "prepared", "account_version": 1,
         "intent_age_seconds": intent_age_seconds},
    ]
    with patch("core.data.repos.local_order_intent.Database.transaction",
               side_effect=lambda operation: operation(cursor)), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         pytest.raises(ValueError, match="委托预留已过期或时间不可核验"):
        LocalOrderIntentRepo.attach_broker_order("intent-1", "O-1", "admin")

    assert len(cursor.execute.call_args_list) == 2


def test_easytrader_fill_reads_easytrader_trade_report_namespace():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1},
        {"account_id": "LOCAL-LIVE", "broker_account_id": "EASYT-ACCOUNT-1",
         "broker_order_id": "O-1"},
        None,
    ]
    with patch("core.data.repos.local_order_intent.Database.transaction",
               side_effect=lambda operation: operation(cursor)), \
         pytest.raises(ValueError, match="成交回报不存在"):
        LocalOrderIntentRepo.record_broker_fill(
            "intent-1", "T-1", "0.30", "0", "admin", lambda rows: "fp",
        )

    assert cursor.execute.call_args_list[2].args[1] == ("easytrader", "EASYT-ACCOUNT-1", "T-1")


def test_easytrader_terminal_check_reads_easytrader_order_namespace():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1},
        {"account_id": "LOCAL-LIVE", "broker_account_id": "EASYT-ACCOUNT-1",
         "broker_order_id": "O-1", "state": "submitted", "account_version": 1},
        None,
    ]
    with patch("core.data.repos.local_order_intent.Database.transaction",
               side_effect=lambda operation: operation(cursor)), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         pytest.raises(ValueError, match="尚无可确认的撤单或拒单终态"):
        LocalOrderIntentRepo.finalize_terminal("intent-1", "admin")

    assert cursor.execute.call_args_list[2].args[1] == ("easytrader", "EASYT-ACCOUNT-1", "O-1")


def test_market_date_uses_shanghai_date_when_utc_is_still_previous_day():
    from core.shared.trading_time import A_SHARE_TIMEZONE, market_date, time_until_close

    utc_instant = datetime.datetime(2026, 10, 28, 16, 30, tzinfo=datetime.timezone.utc)
    assert market_date(utc_instant) == datetime.date(2026, 10, 29)
    shanghai_instant = datetime.datetime(2026, 10, 29, 14, 30, tzinfo=A_SHARE_TIMEZONE)
    assert time_until_close(shanghai_instant) == 30 * 60


@pytest.fixture(autouse=True)
def isolated_integrity_snapshot():
    with patch.object(LocalLedgerRepo, "audit_integrity", side_effect=lambda *args: {
        "ok": True, "issues": [], "account_version": LocalLedgerRepo.get_account()["version"],
    }), patch("core.shared.config.BROKER_TYPE", "qmt"):
        yield


@pytest.mark.parametrize("field,value", [
    ("signal_id", 7.9), ("signal_id", True),
    ("expected_account_version", 2.5), ("expected_account_version", False),
    ("quantity", 100.5), ("quantity", True), ("quantity", "NaN"),
])
def test_prepare_intent_rejects_non_integer_service_inputs(field, value):
    detail = {
        "intent_id": "00000000-0000-4000-8000-000000000002",
        "signal_id": 7,
        "expected_account_version": 1,
        "quantity": 100,
        "limit_price": "10.00",
        "fee_cap": "5.00",
        "actor": "admin",
    }
    detail[field] = value
    with pytest.raises(ValueError, match="委托参数无效"):
        LocalLedgerService.prepare_intent(detail)


def test_local_account_read_retries_when_ledger_version_changes_mid_snapshot():
    def account(version):
        return {"status": "active", "cash": 1000, "frozen": 0, "version": version,
                "broker_account_id": None, "available": 1000}

    with patch.object(LocalLedgerRepo, "get_account", side_effect=[
        account(1), account(2), account(2), account(2),
    ]) as get_account, \
         patch.object(LocalLedgerRepo, "audit_integrity", side_effect=[
             {"ok": True, "account_version": 2}, {"ok": True, "account_version": 2},
         ]), \
         patch.object(LocalLedgerRepo, "get_opening_snapshot", return_value={
             "fingerprint": "f" * 64, "holdings": [],
         }), \
         patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={"orders": 0}):
        result = LocalLedgerService.get_account()

    assert result["version"] == result["integrity"]["account_version"] == 2
    assert result["holdings_fingerprint"] == "f" * 64
    assert get_account.call_count == 4


def test_local_account_read_fails_closed_after_repeated_version_changes():
    def account(version):
        return {"status": "active", "cash": 1000, "frozen": 0, "version": version,
                "broker_account_id": None, "available": 1000}

    with patch.object(LocalLedgerRepo, "get_account", side_effect=[
        account(1), account(2), account(2), account(3),
    ]), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True, "account_version": 1}), \
         patch.object(LocalLedgerRepo, "get_opening_snapshot", return_value={
             "fingerprint": "f" * 64, "holdings": [],
         }):
        with pytest.raises(RuntimeError, match="持续变化"):
            LocalLedgerService.get_account()


def test_local_account_route_reports_unstable_snapshot_as_service_unavailable():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    with patch.object(LocalLedgerService, "get_account", side_effect=RuntimeError("本地账本在读取期间持续变化，请刷新后重试")):
        response = TestClient(app).get("/account/local")

    assert response.status_code == 503
    assert "持续变化" in response.json()["detail"]


def test_holdings_route_reads_only_local_live_account():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.portfolio import router
    from core.data.repos.local_ledger import ACCOUNT_ID

    app = FastAPI()
    app.include_router(router)
    with patch("core.data.HoldingsRepo.get_all", return_value=[]) as get_all:
        response = TestClient(app).get("/holdings")

    assert response.status_code == 200
    get_all.assert_called_once_with(ACCOUNT_ID)


def test_opening_snapshot_change_blocks_activation():
    cursor = Mock()
    cursor.fetchone.return_value = {"status": "uninitialized"}
    cursor.fetchall.return_value = [{"code": "000001", "shares": 100, "cost_price": 10}]
    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="持仓已变化"):
            LocalLedgerRepo.activate("1000.00", "0" * 64, "admin", "确认期初余额和持仓")
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_opening_snapshot_includes_unassigned_legacy_holdings():
    rows = [
        {"code": "000001", "shares": 100, "cost_price": 10, "account_id": None},
        {"code": "600000", "shares": 200, "cost_price": 8,
         "buy_date": datetime.date(2026, 10, 29), "account_id": "LOCAL-LIVE"},
    ]
    with patch("core.data.repos.local_ledger.Database.fetchall", return_value=rows) as fetchall:
        result = LocalLedgerRepo.get_opening_snapshot()

    fetchall.assert_called_once()
    assert "account_id IS NULL" in fetchall.call_args.args[0]
    assert result["holdings"] == [
        {"code": "000001", "shares": 100, "cost_price": 10.0, "buy_date": None,
         "source_account_id": None},
        {"code": "600000", "shares": 200, "cost_price": 8.0, "buy_date": "2026-10-29",
         "source_account_id": "LOCAL-LIVE"},
    ]
    assert result["pending_holdings"] == [result["holdings"][0]]


def test_opening_approval_binds_fingerprinted_legacy_holdings_atomically():
    from core.domain.portfolio.decision import decision_fingerprint

    cursor = Mock()
    cursor.fetchone.return_value = {"status": "uninitialized"}
    cursor.fetchall.return_value = [
        {"code": "000001", "shares": 100, "cost_price": 10,
         "buy_date": datetime.date(2026, 10, 29), "account_id": None},
    ]
    cursor.rowcount = 1
    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.activate(
            "1000.00", decision_fingerprint([{
                "code": "000001", "shares": 100, "cost_price": 10.0, "buy_date": "2026-10-29",
                "source_account_id": None,
            }]),
            "admin", "核准旧持仓并确认期初余额",
        )

    statements = [call.args[0] for call in cursor.execute.call_args_list]
    claim_index = next(i for i, sql in enumerate(statements) if sql.startswith("UPDATE holdings SET account_id"))
    account_index = next(i for i, sql in enumerate(statements) if sql.startswith("UPDATE local_ledger_accounts"))
    event_index = next(i for i, sql in enumerate(statements) if sql.startswith("INSERT INTO local_ledger_events"))
    assert claim_index < account_index < event_index
    assert cursor.execute.call_args_list[claim_index].args[1] == ("LOCAL-LIVE",)
    event_payload = cursor.execute.call_args_list[event_index].args[1][-1]
    assert '"pending_holdings_approved": 1' in event_payload
    assert '"buy_date": "2026-10-29"' in event_payload
    assert result["status"] == "active"


def test_opening_approval_rejects_buy_date_changed_after_fingerprint():
    from core.domain.portfolio.decision import decision_fingerprint

    cursor = Mock()
    cursor.fetchone.return_value = {"status": "uninitialized"}
    cursor.fetchall.return_value = [{
        "code": "000001", "shares": 100, "cost_price": 10,
        "buy_date": datetime.date(2026, 10, 30), "account_id": None,
    }]
    expected = decision_fingerprint([{
        "code": "000001", "shares": 100, "cost_price": 10.0, "buy_date": "2026-10-29",
        "source_account_id": None,
    }])

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        with pytest.raises(ValueError, match="持仓已变化"):
            LocalLedgerRepo.activate("1000.00", expected, "admin", "核准期初余额和买入日期")

    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_opening_approval_rejects_account_assignment_changed_after_fingerprint():
    from core.domain.portfolio.decision import decision_fingerprint

    cursor = Mock()
    cursor.fetchone.return_value = {"status": "uninitialized"}
    cursor.fetchall.return_value = [{
        "code": "000001", "shares": 100, "cost_price": 10,
        "buy_date": datetime.date(2026, 10, 29), "account_id": "LOCAL-LIVE",
    }]
    expected = decision_fingerprint([{
        "code": "000001", "shares": 100, "cost_price": 10.0, "buy_date": "2026-10-29",
        "source_account_id": None,
    }])

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        with pytest.raises(ValueError, match="持仓已变化"):
            LocalLedgerRepo.activate("1000.00", expected, "admin", "核准期初归属状态")

    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_rebalance_holdings_fingerprint_normalizes_and_binds_buy_date():
    from core.domain.portfolio.decision import decision_fingerprint, holdings_fingerprint_rows

    typed_date = holdings_fingerprint_rows([{
        "code": "000001", "shares": 100, "cost_price": 10,
        "buy_date": datetime.date(2026, 10, 29),
    }])
    string_date = holdings_fingerprint_rows([{
        "code": "000001", "shares": 100, "cost_price": 10,
        "buy_date": "2026-10-29",
    }])
    changed_date = holdings_fingerprint_rows([{
        "code": "000001", "shares": 100, "cost_price": 10,
        "buy_date": "2026-10-30",
    }])

    assert typed_date == string_date
    assert decision_fingerprint(typed_date) != decision_fingerprint(changed_date)


@pytest.mark.parametrize("holding,expected_error", [
    ({"code": "000001", "shares": 0, "cost_price": 1}, "股数或成本无效"),
    ({"code": "000001", "shares": -100, "cost_price": 1}, "股数或成本无效"),
    ({"code": "000001", "shares": 100.5, "cost_price": 1}, "股数或成本无效"),
    ({"code": "000001", "shares": 2147483648, "cost_price": 1}, "股数或成本无效"),
    ({"code": "000001", "shares": 100, "cost_price": -1}, "股数或成本无效"),
    ({"code": "000001", "shares": 100, "cost_price": None}, "股数或成本无效"),
    ({"code": "000001", "shares": 100, "cost_price": "NaN"}, "股数或成本无效"),
    ({"code": "bad-code", "shares": 100, "cost_price": 1}, "无效证券代码"),
    ({"code": "000001.SZ", "shares": 100, "cost_price": 1}, "六位标准代码"),
])
def test_invalid_opening_holdings_block_activation(holding, expected_error):
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.return_value = {"status": "uninitialized", "version": 0,
                                   "cash": None, "frozen": 0}
    cursor.fetchall.return_value = [holding]
    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match=expected_error):
            LocalLedgerRepo.activate("1000.00", "0" * 64, "admin", "核准期初快照并记录来源")
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)
    assert not any("INSERT INTO local_ledger_events" in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("account,expected_error", [
    ({"status": "uninitialized", "version": 0, "cash": None, "frozen": 0}, "期初尚未核准"),
    ({"status": "active", "version": 2, "cash": 1000, "frozen": 0}, "账户版本已变化"),
])
def test_unapproved_or_stale_account_blocks_fill(account, expected_error):
    cursor = Mock()
    cursor.fetchone.side_effect = [account, None, None]
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match=expected_error):
            RebalanceSignalRepo.record_execution(7, DETAIL, lambda rows: "fingerprint")
    assert not any("UPDATE holdings" in call.args[0] for call in cursor.execute.call_args_list)


def test_bound_broker_account_blocks_unverified_manual_fill_in_transaction():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0,
         "broker_account_id": "QMT-ACCOUNT-1"},
        None,
    ]
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="必须通过本地委托意图关联券商订单"):
            RebalanceSignalRepo.record_execution(7, {**DETAIL, "broker_trade_id": "FAKE-TRADE"},
                                                 lambda rows: "fingerprint")
    assert cursor.execute.call_count == 2
    assert not any("broker_trade_reports" in call.args[0] for call in cursor.execute.call_args_list)
    assert not any("UPDATE holdings" in call.args[0] or "INSERT INTO holdings" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_overfill_blocks_before_holdings_write():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0}, None, None,
        {"account_id": "LOCAL-LIVE", "planned_shares": 100, "filled_shares": 50,
         "execution_status": "partial", "account_version": 1,
         "created_at": datetime.datetime.now()},
    ]
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="超过剩余计划量"):
            RebalanceSignalRepo.record_execution(7, DETAIL, lambda rows: "fingerprint")
    assert cursor.fetchall.call_count == 0


def test_corrupt_ledger_blocks_manual_fill_before_holdings_write():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0}, None,
    ]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": False}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="账本完整性校验失败"):
            RebalanceSignalRepo.record_execution(7, DETAIL, lambda rows: "fingerprint")
    assert not any("UPDATE holdings" in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("field,value", [
    ("signal_id", 7.5), ("signal_id", True),
    ("shares", 100.5), ("shares", False),
    ("expected_account_version", 1.5), ("expected_account_version", True),
])
def test_manual_fill_repository_rejects_non_integer_identity_and_quantity(field, value):
    detail = {**DETAIL}
    signal_id = 7
    if field == "signal_id":
        signal_id = value
    else:
        detail[field] = value
    with patch("core.data.repos.rebalance.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="成交股数、信号编号或账户版本无效"):
            RebalanceSignalRepo.record_execution(signal_id, detail, lambda rows: "fingerprint")
    transaction.assert_not_called()


@pytest.mark.parametrize("request_id", [None, "", "manual-1"])
def test_manual_fill_repository_rejects_non_uuid_request_id_before_transaction(request_id):
    detail = {**DETAIL, "request_id": request_id}
    with patch("core.data.repos.rebalance.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="成交请求 ID 必须是有效 UUID"):
            RebalanceSignalRepo.record_execution(7, detail, lambda rows: "fingerprint")
    transaction.assert_not_called()


@pytest.mark.parametrize("version", [1.5, True, "NaN"])
def test_manual_fill_service_rejects_non_integer_account_version(version):
    signal = {
        "account_id": "LOCAL-LIVE", "planned_shares": 100,
        "execution_status": "pending", "created_at": datetime.datetime.now(),
    }
    detail = {
        "shares": 100, "price": "10.00", "request_id": DETAIL["request_id"],
        "expected_account_version": version, "commission": "0.30",
        "stamp_duty": "0.00", "actor": "admin",
    }
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value=signal), \
         patch("core.data.RebalanceSignalRepo.record_execution") as record:
        result = RebalanceService.mark_executed(7, detail)
    assert "账户版本" in result["error"]
    record.assert_not_called()


@pytest.mark.parametrize("account_id,planned_shares,error", [
    (None, 100, "仅可确认本地实盘账户的信号"),
    ("OTHER-ACCOUNT", 100, "仅可确认本地实盘账户的信号"),
    ("LOCAL-LIVE", None, "旧信号缺少账户或明确计划股数"),
    ("LOCAL-LIVE", 0, "旧信号缺少账户或明确计划股数"),
    ("LOCAL-LIVE", -100, "旧信号缺少账户或明确计划股数"),
    ("LOCAL-LIVE", 100.5, "旧信号缺少账户或明确计划股数"),
])
def test_manual_fill_service_rejects_unbound_foreign_or_invalid_plan_before_followup_reads(
        account_id, planned_shares, error):
    signal = {"account_id": account_id, "planned_shares": planned_shares,
              "created_at": datetime.datetime.now(), "side": "buy"}
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value=signal), \
         patch("core.data.RebalanceSignalRepo.record_execution") as record, \
         patch("backend.service.strategy_profile.StrategyProfileService.ensure_execution_allowed") as profile_check, \
         patch("core.data.HoldingsRepo.get_all") as read_holdings:
        result = RebalanceService.mark_executed(7, DETAIL)
    assert error in result["error"]
    record.assert_not_called()
    profile_check.assert_not_called()
    read_holdings.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("price", "NaN"), ("price", "0"),
    ("commission", "-0.01"), ("commission", "0.001"),
    ("stamp_duty", "Infinity"),
])
def test_manual_fill_repository_rejects_invalid_actual_price_or_fees_before_transaction(field, value):
    detail = {**DETAIL, field: value}
    with patch("core.data.repos.rebalance.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="成交价格或实际费用无效"):
            RebalanceSignalRepo.record_execution(7, detail, lambda rows: "fingerprint")
    transaction.assert_not_called()


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_manual_fill_transaction_rejects_fees_above_gross_amount(side):
    cursor = Mock()
    cursor.rowcount = 0
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0},
        None,
        None,
        {"account_id": "LOCAL-LIVE", "planned_shares": 100, "filled_shares": 0,
         "execution_status": "pending", "account_version": 1,
         "created_at": datetime.datetime.now(), "input_fingerprint": "fingerprint",
         "code": "000001", "side": side, "batch_id": "batch"},
    ]
    cursor.fetchall.return_value = []
    detail = {**DETAIL, "price": "0.01", "commission": "2.00", "stamp_duty": "0.00"}
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}):
        with pytest.raises(ValueError, match="费用不能超过成交金额"):
            RebalanceSignalRepo.record_execution(7, detail, lambda rows: "fingerprint")
    assert not any("UPDATE holdings" in call.args[0] or "INSERT INTO holdings" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_corrupt_ledger_blocks_executable_plan_creation():
    cursor = Mock()
    cursor.fetchone.return_value = {"status": "active", "version": 1}
    signals = [{"code": "000001", "side": "buy", "planned_shares": 100}]
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": False}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="账本完整性校验失败"):
            RebalanceSignalRepo.batch_create(
                "batch", 1, "weekly", signals, account_id="LOCAL-LIVE", account_version=1,
                replace_pending=True, input_fingerprint="fp", fingerprint_fn=lambda rows: "fp",
            )
    assert not any("INSERT INTO rebalance_signals" in call.args[0] for call in cursor.execute.call_args_list)


def test_reused_batch_id_cannot_return_another_account_session():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 4}, None,
        {"account_id": "OTHER-ACCOUNT", "account_version": 4,
         "check_type": "rebalance", "screening_save_id": 1, "source_mode": "live"},
    ]
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="其他账户或版本"):
            RebalanceSignalRepo.batch_create(
                "reused-batch", 1, "rebalance",
                [{"code": "000001", "side": "buy", "planned_shares": 100}],
                replace_pending=True, input_fingerprint="fp", fingerprint_fn=lambda rows: "fp",
                session_payload={"batch_id": "reused-batch", "check_type": "rebalance",
                                 "screening_save_id": 1, "source_mode": "live"},
                account_id="LOCAL-LIVE", account_version=4,
            )
    assert not any("INSERT INTO rebalance_signals" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_buy_with_insufficient_cash_does_not_write_holdings():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 500, "frozen": 0}, None, None,
        {"account_id": "LOCAL-LIVE", "planned_shares": 100, "filled_shares": 0,
         "execution_status": "pending", "account_version": 1,
         "created_at": datetime.datetime.now(), "input_fingerprint": "fingerprint",
         "code": "000001", "side": "buy"},
    ]
    cursor.fetchall.return_value = []
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="可用现金不足"):
            RebalanceSignalRepo.record_execution(7, DETAIL, lambda rows: "fingerprint")
    assert not any("INSERT INTO holdings" in call.args[0] for call in cursor.execute.call_args_list)


def test_manual_sell_blocks_same_day_buys_from_t1_sellable_quantity():
    today = datetime.date.today()
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0},
        None, None,
        {"account_id": "LOCAL-LIVE", "planned_shares": 100, "filled_shares": 0,
         "execution_status": "pending", "account_version": 1,
         "created_at": datetime.datetime.combine(today, datetime.time(9, 0)),
         "created_date": today, "database_date": today,
         "input_fingerprint": "fingerprint", "code": "000001", "side": "sell",
         "batch_id": "batch"},
        {"shares": 50},
    ]
    cursor.fetchall.return_value = [{
        "code": "000001", "shares": 100, "cost_price": 9, "buy_date": today - datetime.timedelta(days=1),
    }]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)), \
         pytest.raises(ValueError, match=r"T\+1可卖量"):
        RebalanceSignalRepo.record_execution(
            7, {**DETAIL, "shares": 100, "price": 10}, lambda rows: "fingerprint",
        )

    assert any("DATE(occurred_at)=%s" in call.args[0] for call in cursor.execute.call_args_list)
    assert not any("UPDATE holdings" in call.args[0] or "DELETE FROM holdings" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_manual_sell_fails_closed_when_holding_acquisition_date_is_missing():
    today = datetime.date.today()
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0},
        None, None,
        {"account_id": "LOCAL-LIVE", "planned_shares": 100, "filled_shares": 0,
         "execution_status": "pending", "account_version": 1,
         "created_at": datetime.datetime.combine(today, datetime.time(9, 0)),
         "created_date": today, "database_date": today,
         "input_fingerprint": "fingerprint", "code": "000001", "side": "sell",
         "batch_id": "batch"},
    ]
    cursor.fetchall.return_value = [{"code": "000001", "shares": 100, "cost_price": 9, "buy_date": None}]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)), \
         pytest.raises(ValueError, match="买入日期未核验"):
        RebalanceSignalRepo.record_execution(
            7, {**DETAIL, "shares": 100, "price": 10}, lambda rows: "fingerprint",
        )

    assert not any("UPDATE holdings" in call.args[0] or "DELETE FROM holdings" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_fill_invalidates_stale_pending_signals_across_batches_and_records_remaining_shares():
    siblings = [
        {"id": 8, "code": "000002", "side": "sell", "planned_shares": 500,
         "filled_shares": 100, "execution_status": "partial"},
        {"id": 9, "code": "000003", "side": "buy", "planned_shares": 300,
         "filled_shares": 0, "execution_status": "pending"},
    ]
    cursor = Mock()
    cursor.rowcount = 1
    cursor.execute.side_effect = lambda sql, params=None: setattr(
        cursor, "rowcount", len(siblings) if "SET execution_status='cancelled'" in sql else 1,
    )
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 5000, "frozen": 0}, None, None,
        {"account_id": "LOCAL-LIVE", "planned_shares": 100, "filled_shares": 0,
         "execution_status": "pending", "account_version": 1,
         "created_at": datetime.datetime.now(), "input_fingerprint": "fingerprint",
         "code": "000001", "side": "buy", "batch_id": "current-batch"},
    ]
    cursor.fetchall.side_effect = [
        [],
        [{"code": "000001", "shares": 100, "cost_price": 10}],
        siblings,
    ]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = RebalanceSignalRepo.record_execution(7, DETAIL, lambda rows: "fingerprint")

    assert result["invalidated_signal_ids"] == [8, 9]
    assert [item["invalidated_remaining_shares"] for item in result["invalidated_signal_details"]] == [400, 300]
    selection_query = next(call for call in cursor.execute.call_args_list
                           if call.args[0].startswith("SELECT id,code,side,planned_shares,filled_shares,execution_status "))
    assert selection_query.args[1] == ("LOCAL-LIVE", 2, 7)
    invalidation_query = next(call for call in cursor.execute.call_args_list
                              if "SET execution_status='cancelled'" in call.args[0])
    assert invalidation_query.args[1] == (
        "账户资金或持仓已变化，旧调仓计划已失效，请重新生成", "LOCAL-LIVE", 8, 9, 2,
    )
    event_insert = next(call for call in cursor.execute.call_args_list
                        if "INSERT INTO local_ledger_events" in call.args[0])
    event_payload = json.loads(event_insert.args[1][-1])
    assert event_payload["invalidated_signal_details"] == result["invalidated_signal_details"]


def test_concurrent_duplicate_request_records_one_event():
    lock = threading.Lock()
    cursors = []
    signal = {"account_id": "LOCAL-LIVE", "planned_shares": 100, "filled_shares": 0,
              "execution_status": "pending", "account_version": 1,
              "created_at": datetime.datetime.now(), "input_fingerprint": "fingerprint",
              "code": "000001", "side": "sell", "batch_id": "batch"}

    def transaction(operation):
        with lock:
            cursor = Mock()
            cursor.rowcount = 1
            cursor.execute.side_effect = lambda sql, params=None: setattr(
                cursor, "rowcount", 0 if "execution_status='cancelled'" in sql else 1,
            )
            if not cursors:
                cursor.fetchone.side_effect = [
                    {"status": "active", "version": 1, "cash": 1000, "frozen": 0}, None, None, signal,
                    {"shares": 0}, None,
                ]
                cursor.fetchall.side_effect = [
                    [{"code": "000001", "shares": 100, "cost_price": 9,
                      "buy_date": datetime.date.today() - datetime.timedelta(days=1)}], [], [],
                ]
            else:
                cursor.fetchone.side_effect = [
                    {"status": "active", "version": 2, "cash": 1999.70, "frozen": 0},
                    {"signal_id": 7, "payload_json": {
                        **DETAIL, "execution_status": "executed", "remaining_plan_shares": 0,
                    }, "account_version": 2},
                ]
            cursors.append(cursor)
            return operation(cursor)

    with patch("core.data.repos.rebalance.Database.transaction", side_effect=transaction):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda _: RebalanceSignalRepo.record_execution(7, DETAIL, lambda rows: "fingerprint"),
                range(2),
            ))
    assert sorted(result["already_recorded"] for result in results) == [False, True]
    assert sum("INSERT INTO local_ledger_events" in call.args[0]
               for cursor in cursors for call in cursor.execute.call_args_list) == 1


@pytest.mark.parametrize("account", [
    {"status": "opening_required", "version": 1},
    {"status": "reconciliation_blocked", "version": 2},
    None,
])
def test_duplicate_fill_returns_original_result_even_if_account_is_not_active(account):
    cursor = Mock()
    cursor.fetchone.side_effect = [
        account,
        {"signal_id": 7, "payload_json": {
            **DETAIL, "execution_status": "executed", "remaining_plan_shares": 0,
        }, "account_version": 2},
    ]
    with patch("core.data.repos.rebalance.Database.transaction",
               side_effect=lambda operation: operation(cursor)):
        result = RebalanceSignalRepo.record_execution(7, DETAIL, lambda rows: "fingerprint")

    assert result["already_recorded"] is True
    assert result["execution_status"] == "executed"
    assert result["account_version"] == 2
    assert cursor.execute.call_count == 2


def test_completed_buy_retry_reaches_idempotency_before_stale_preflight():
    signal = {"id": 7, "account_id": "LOCAL-LIVE", "planned_shares": 100,
              "execution_status": "executed", "created_at": "invalid-timestamp",
              "code": "000001", "side": "buy", "input_fingerprint": "old",
              "planned_price": 10, "amount": 1000}
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value=signal), \
         patch("core.data.HoldingsRepo.get_all", return_value=[{
             "code": "000001", "shares": 100, "cost_price": 10,
         }]), patch("backend.service.strategy_profile.StrategyProfileService.ensure_execution_allowed",
                    side_effect=ValueError("策略已变化")), \
         patch("core.data.RebalanceSignalRepo.record_execution", return_value={
             "execution_status": "executed", "already_recorded": True,
             "remaining_plan_shares": 0, "account_version": 2,
         }) as record:
        result = RebalanceService.mark_executed(7, DETAIL)
    assert result["already_recorded"] is True
    assert "executed" in record.call_args.kwargs["fresh_error"]


@pytest.mark.parametrize("snapshot,expected_reason", [
    ({"account": {"account_id": "QMT-1", "as_of": "2026-10-27 10:00:00", "cash": 1000, "frozen": 0, "available": 1000},
      "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [{"symbol": "000001", "shares": 90}], "orders": []}, "持仓不一致"),
    ({"account": {"account_id": "QMT-1", "as_of": "2026-10-27 10:00:00", "cash": 1000, "frozen": 0, "available": 1000},
      "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [{"symbol": "000001", "shares": 100, "sellable": 100}],
      "orders": [{"id": "1", "status": "submitted"}]}, "未确认委托"),
])
def test_reconciliation_blocks_mismatch_and_pending_order(snapshot, expected_reason):
    broker = _broker_adapter_mock("QMT-1")
    broker.get_reconciliation_snapshot.return_value = snapshot
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0, "available": 1000,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[{"code": "000001", "shares": 100}]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert expected_reason in result["reason"]
    broker.get_reconciliation_snapshot.assert_called_once_with(check_report_integrity=False)


def test_qmt_reconciliation_snapshot_does_not_replay_or_persist_reports():
    from core.adapters.broker import QmtBroker
    from core.shared.trading_time import A_SHARE_TIMEZONE

    broker = QmtBroker.__new__(QmtBroker)
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = Mock()
    broker._account = "123456"
    broker._account_id = "QMT-123456"
    broker._report_persistence_error = None
    broker._lock = threading.RLock()
    broker.get_account = Mock(return_value={
        "status": "fresh", "account_id": "QMT-123456",
        "as_of": datetime.datetime.now(A_SHARE_TIMEZONE).isoformat(),
        "cash": 1000, "frozen": 0, "available": 1000,
    })
    broker._trader.query_stock_positions.return_value = []
    broker._trader.query_stock_orders.return_value = [{
        "account_id": "123456", "order_id": "done-1", "stock_code": "000001.SZ",
        "order_type": 23, "order_status": 56, "order_volume": 100,
        "price_type": 11, "price": 10, "traded_volume": 100,
    }]
    health = {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0,
              "legacy_quarantined": 0}

    with patch.object(broker, "replay_order_reports") as replay, \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.get_health",
               return_value=health) as get_health, \
         patch.object(broker, "_record_order_report") as record_order:
        snapshot = broker.get_reconciliation_snapshot()

    replay.assert_not_called()
    record_order.assert_not_called()
    get_health.assert_called_once_with("qmt", "QMT-123456")
    assert snapshot["trade_report_health"] == health


@pytest.mark.parametrize("sellable", [None, 101])
def test_reconciliation_blocks_execution_when_broker_sellable_quantity_is_missing_or_invalid(sellable):
    broker = _broker_adapter_mock("QMT-1")
    position = {"symbol": "000001", "shares": 100}
    if sellable is not None:
        position["sellable"] = sellable
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-1", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [position], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0, "available": 1000,
        "broker_account_id": "QMT-1",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[{"code": "000001", "shares": 100}]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()

    assert result["can_execute"] is False
    assert result["can_bind"] is False
    assert result["sellable_missing_codes"] == ["000001"]
    assert "position.sellable" in result["missing_items"]
    assert "可卖股数缺失" in result["reason"]


def test_reconciliation_reports_account_identity_mismatch():
    broker = _broker_adapter_mock("QMT-CONNECTED")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-RETURNED", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
        "positions": [],
        "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0, "available": 1000,
        "broker_account_id": "QMT-CONNECTED",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()

    assert result["broker_account_id"] == "QMT-RETURNED"
    assert result["broker_identity_verified"] is True
    assert result["broker_status"] == "unavailable"
    assert result["can_execute"] is False
    assert result["reason"] == "券商返回资金账号与当前连接不一致"
    assert result["missing_items"] == ["account.identity"]


def test_reconciliation_refuses_execution_if_local_ledger_changes_during_broker_snapshot():
    broker = _broker_adapter_mock("QMT-1")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-1", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
        "positions": [{"symbol": "000001", "shares": 100, "sellable": 100}],
        "orders": [],
    }
    account = {"status": "active", "version": 1, "cash": 1000, "frozen": 0,
               "available": 1000, "broker_account_id": "QMT-1"}
    changed = {**account, "version": 2, "cash": 900, "available": 900}
    with patch.object(LocalLedgerRepo, "get_account", side_effect=[account, changed]), \
         patch.object(LocalLedgerRepo, "audit_integrity", side_effect=[
             {"ok": True, "account_version": 1}, {"ok": True, "account_version": 2},
         ]), \
         patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[
             {"code": "000001", "shares": 100},
         ]), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unfinished", return_value=[]), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unresolved", return_value=[]), \
         patch("backend.service.local_ledger.LocalOrderIntentRepo.has_unresolved", return_value=False), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()

    assert result["local_version"] == 1
    assert result["can_execute"] is False
    assert result["reason"] == "对账期间本地账户已变化，请重新核对"


def test_reconciliation_blocks_disconnected_broker():
    broker = Mock()
    broker.get_reconciliation_snapshot.side_effect = RuntimeError("断连")
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0, "available": 1000,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert result["broker_status"] == "unavailable"


def test_reconciliation_rejects_broker_without_report_namespace_before_snapshot():
    broker = Mock()
    broker.account_id = "UNSUPPORTED-1"
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": "UNSUPPORTED-1",
    }), patch.object(LocalLedgerRepo, "audit_integrity", return_value={
        "ok": True, "account_version": 1,
    }), patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch.object(Database, "fetchall", return_value=[]), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.BROKER_FAKE", "fake"), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()

    assert result["broker_status"] == "unavailable"
    assert result["can_execute"] is False
    assert result["can_bind"] is False
    assert result["reason"] == "券商适配器缺少受支持的回报命名空间"
    assert "broker.report_namespace" in result["missing_items"]
    broker.get_reconciliation_snapshot.assert_not_called()


@pytest.mark.parametrize("error,reason,missing_item", [
    ("QMT_ACCOUNT 未配置，请在 .env 中设置 QMT 资金账号",
     "QMT 资金账号未配置", "broker.configuration.account"),
    ("QMT userdata_mini 目录不存在或不可用",
     "QMT userdata_mini 目录不可用", "broker.configuration.userdata_dir"),
    ("EASYTRAIDER_USER 未配置，请在 .env 中设置同花顺账号",
     "Easytrader 登录账号未配置", "broker.configuration.user"),
    ("EASYTRAIDER_ACCOUNT_ID 未配置，拒绝连接未标识的券商账户",
     "Easytrader 资金账号未配置", "broker.configuration.easytrader_account"),
    ("EASYTRAIDER_ACCOUNT_ID 格式无效",
     "Easytrader 资金账号格式无效", "broker.configuration.easytrader_account"),
])
def test_reconciliation_exposes_actionable_broker_configuration_gaps(error, reason, missing_item):
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0, "available": 1000,
        "broker_account_id": None,
    }), patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch.object(Database, "fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", side_effect=RuntimeError(error)):
        result = LocalLedgerService.reconcile()

    assert result["broker_status"] == "unavailable"
    assert result["can_execute"] is False
    assert result["reason"] == reason
    assert missing_item in result["missing_items"]


def test_simulation_reconciliation_does_not_initialize_broker():
    with patch("core.shared.config.BROKER_TYPE", "fake"), \
         patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
             "status": "uninitialized", "version": 0, "cash": None, "frozen": 0,
             "available": None, "broker_account_id": None,
         }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker") as get_broker:
        result = LocalLedgerService.reconcile()
    get_broker.assert_not_called()
    assert result["broker_status"] == "simulation"
    assert result["broker_account_id"] is None
    assert result["can_execute"] is False


def test_reconciliation_does_not_truncate_fractional_local_shares():
    account = {"status": "active", "version": 1, "cash": 1000, "frozen": 0,
               "available": 1000, "broker_account_id": None, "updated_at": None}
    with patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={
             "ok": False, "issues": ["持仓数量无效"], "account_version": 1,
         }), \
         patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={
             "orders": 0, "trades": 0, "signals": 0,
         }), \
         patch("core.data.connection.Database.fetchall", return_value=[
             {"code": "000001", "shares": "100.5"},
         ]), \
         patch("core.shared.config.BROKER_TYPE", "fake"):
        result = LocalLedgerService.reconcile()
    assert result["local_positions"] == {"000001": None}
    assert result["position_differences"] == []
    assert "local.position.shares" in result["missing_items"]
    assert result["can_execute"] is False


def test_reconciliation_blocks_cash_and_frozen_mismatch_even_if_available_matches():
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1100, "frozen": 100, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": "QMT-ONE",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False):
        result = LocalLedgerService.reconcile()
    assert result["cash_difference"] == 0
    assert result["cash_balance_difference"] == 100
    assert result["frozen_difference"] == 100
    assert result["can_execute"] is False
    assert "现金、冻结额" in result["reason"]


@pytest.mark.parametrize("cash,frozen,available", [
    ("1000.01", "0.00", "1000.01"),
    ("1000.01", "0.01", "1000.00"),
    ("1000.00", "0.01", "999.99"),
])
def test_reconciliation_blocks_one_cent_difference(cash, frozen, available):
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": cash, "frozen": frozen, "available": available},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": "QMT-ONE",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert "现金、冻结额" in result["reason"]


@pytest.mark.parametrize("cash,frozen,available,missing", [
    (None, 0, 1000, ["account.cash"]),
    (1000, 100, 1000, ["account.fund_consistency"]),
    ("1000.01", "0.00", "1000.00", ["account.fund_consistency"]),
])
def test_reconciliation_rejects_missing_or_inconsistent_broker_funds(cash, frozen, available, missing):
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": cash, "frozen": frozen, "available": available},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": "QMT-ONE",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert result["missing_items"] == missing
    assert "资金分项" in result["broker_error"]


@pytest.mark.parametrize("identity_verified", [False, None, "false", "true", 1])
def test_unverified_broker_identity_cannot_authorize_binding(identity_verified):
    broker = _broker_adapter_mock("EASYT-YH")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "EASYT-YH", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": identity_verified, "positions": [], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0, "available": 1000,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert result["broker_identity_verified"] is False
    assert "资金账号" in result["reason"]


@pytest.mark.parametrize("age_seconds", [180, -10])
def test_stale_broker_snapshot_cannot_authorize_new_intent(age_seconds):
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE",
                    "as_of": (datetime.datetime.now() - datetime.timedelta(seconds=age_seconds)).isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0, "available": 1000,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert "过期" in result["reason"]
    assert result["broker_status"] == "stale"


def test_unbound_account_can_bind_but_is_not_marked_executable():
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": None,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.audit_integrity", return_value={
        "ok": True, "account_version": 1,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False):
        result = LocalLedgerService.reconcile()

    assert result["can_bind"] is True
    assert result["can_execute"] is False
    assert "绑定" in result["reason"]


def test_unbound_account_with_unresolved_intent_cannot_bind_broker():
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 1, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": None,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.audit_integrity", return_value={
        "ok": True, "account_version": 1,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=True):
        result = LocalLedgerService.reconcile()

    assert result["can_bind"] is False
    assert result["can_execute"] is False
    assert result["pending_local_intent"] is True
    assert "未完成的本地委托意图" in result["reason"]


def test_bind_service_uses_bind_readiness_without_execution_readiness():
    reconciliation = {
        "can_bind": True,
        "can_execute": False,
        "broker_account_id": "QMT-ONE",
    }
    with patch.object(LocalLedgerService, "reconcile", return_value=reconciliation), \
         patch.object(LocalOrderIntentRepo, "bind_broker", return_value={"already_bound": False}) as bind:
        result = LocalLedgerService.bind_broker(1, "admin")

    assert result["already_bound"] is False
    bind.assert_called_once_with("QMT-ONE", 1, "admin", reconciliation)


def test_broker_binding_rejects_execution_only_reconciliation_flag():
    cursor = Mock()
    cursor.fetchone.return_value = {
        "status": "active", "version": 2, "broker_account_id": None, "cash": 1000,
    }
    reconciliation = {
        "can_execute": True, "broker_account_id": "QMT-ONE", "local_version": 2,
        "broker_received_at": datetime.datetime.now().isoformat(),
        "broker_as_of": datetime.datetime.now().isoformat(),
    }
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="券商对账未通过"):
            LocalOrderIntentRepo.bind_broker("QMT-ONE", 2, "admin", reconciliation)

    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("sellable,message", [(None, "不可核验"), (50, "超过券商当前可卖")])
def test_sell_intent_requires_broker_sellable_shares(sellable, message):
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "expected_account_version": 2, "quantity": 100,
              "limit_price": "10", "fee_cap": "0", "actor": "admin"}
    signal = {"side": "sell", "code": "000001"}
    with patch.object(LocalOrderIntentRepo, "get_by_id", return_value=None), \
         patch.object(RebalanceSignalRepo, "get_by_id", return_value=signal), \
         patch.object(LocalLedgerService, "reconcile", return_value={
             "can_execute": True, "broker_account_id": "QMT-ONE",
             "broker_sellable": {"000001": sellable},
         }), patch.object(LocalOrderIntentRepo, "prepare") as prepare:
        with pytest.raises(ValueError, match=message):
            LocalLedgerService.prepare_intent(detail)
    prepare.assert_not_called()


@pytest.mark.parametrize("sellable,expected_sellable,missing_codes,missing_items", [
    (80, {"000001": 80}, [], []),
    (None, {"000001": None}, ["000001"], ["position.sellable"]),
    (-1, {"000001": None}, ["000001"], ["position.sellable"]),
    (101, {"000001": None}, ["000001"], ["position.sellable"]),
    (1.5, {"000001": None}, ["000001"], ["position.sellable"]),
])
def test_reconciliation_reports_broker_sellable_shares_separately(sellable, expected_sellable,
                                                                   missing_codes, missing_items):
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
        "positions": [{"symbol": "000001.SZ", "shares": 100, "sellable": sellable}],
        "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0, "available": 1000,
        "broker_account_id": "QMT-ONE",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[
             {"code": "000001", "shares": 100},
         ]), patch("backend.api.trading.get_broker", return_value=broker), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unfinished", return_value=[]), \
         patch("core.data.repos.broker_order_report.BrokerOrderReportRepo.list_unresolved", return_value=[]), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is (sellable == 80)
    assert result["can_bind"] is False
    assert result["broker_positions"] == {"000001": 100}
    assert result["broker_sellable"] == expected_sellable
    assert result["sellable_missing_codes"] == missing_codes
    assert result["missing_items"] == missing_items


def test_reconciliation_keeps_sellable_codes_when_later_order_data_is_invalid():
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
        "positions": [{"symbol": "000001.SZ", "shares": 100, "sellable": None}],
        "orders": [{"id": None, "status": "submitted"}],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0, "available": 1000,
        "broker_account_id": "QMT-ONE",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()

    assert result["can_execute"] is False
    assert result["sellable_missing_codes"] == ["000001"]
    assert result["missing_items"] == ["position.sellable", "order.id", "order.status"]
    assert result["broker_account_id"] == "QMT-ONE"
    assert result["broker_cash"] == 1000
    assert result["broker_available"] == 1000
    assert result["broker_positions"] == {"000001": 100}
    assert result["broker_sellable"] == {"000001": None}
    assert result["position_differences"] == [
        {"code": "000001", "local_shares": 0, "broker_shares": 100}
    ]
    assert result["broker_as_of"] == broker.get_reconciliation_snapshot.return_value["account"]["as_of"]


def test_reconciliation_rejects_wrong_exchange_suffix_even_when_shares_match():
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
        "positions": [{"symbol": "000001.SH", "shares": 100, "sellable": 100}],
        "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0, "available": 1000,
        "broker_account_id": "QMT-ONE",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[
             {"code": "000001", "shares": 100},
         ]), patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert result["missing_items"] == ["position.symbol"]


def test_reconciliation_rejects_duplicate_broker_position_rows():
    broker = _broker_adapter_mock("QMT-ONE")
    broker.get_reconciliation_snapshot.return_value = {
        "account": {"account_id": "QMT-ONE", "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
        "positions": [{"symbol": "000001.SZ", "shares": 50, "sellable": 50},
                      {"symbol": "000001", "shares": 50, "sellable": 50}],
        "orders": [],
    }
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": "QMT-ONE",
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[
             {"code": "000001", "shares": 100},
         ]), patch("backend.api.trading.get_broker", return_value=broker):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert result["missing_items"] == ["position.duplicate_code"]


def test_reconciliation_blocks_unseen_unfinished_order_journal():
    from core.adapters.broker.qmt_broker import QmtBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = QmtBroker("unused", "test")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "account": {"account_id": broker.account_id,
                    "as_of": datetime.datetime.now().isoformat(), "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
        "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
    })
    with patch("backend.service.local_ledger.LocalLedgerRepo.get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0, "available": 1000,
        "broker_account_id": broker.account_id,
    }), patch("backend.service.local_ledger.LocalLedgerRepo.get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False), \
         patch.object(BrokerOrderReportRepo, "list_unfinished", return_value=[
             {"order_id": "123", "status": "pending"},
         ]), patch.object(BrokerOrderReportRepo, "list_unresolved", return_value=[]):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is False
    assert result["pending_journal_orders"][0]["order_id"] == "123"
    assert "未终结委托" in result["reason"]


def test_easytrader_reconciliation_blocks_unresolved_order_report_journal():
    from core.adapters.broker.easytrader_broker import EasytraderBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = EasytraderBroker(broker_account_id="TEST-001")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "account": {"account_id": broker.account_id,
                    "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
        "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
    })
    unresolved = [{"report_key": "conflict-1", "order_id": "E-1", "symbol": "000001.SZ",
                   "side": "buy", "quantity": 100, "filled_qty": 0,
                   "status": "pending", "source": "query", "disposition": "conflict",
                   "observed_at": "2026-10-29 10:00:00", "raw_json": "private-fields"}]
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": broker.account_id,
    }), patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False), \
         patch.object(BrokerOrderReportRepo, "list_unfinished", return_value=[]), \
         patch.object(BrokerOrderReportRepo, "list_unresolved", return_value=unresolved):
        result = LocalLedgerService.reconcile()

    assert result["can_execute"] is False
    assert result["reason"] == "券商回报日志中存在未核对委托"
    assert result["unresolved_order_reports"] == [{
        key: unresolved[0][key] for key in (
            "report_key", "order_id", "symbol", "side", "quantity", "filled_qty",
            "status", "source", "disposition", "observed_at",
        )
    }]


def test_easytrader_reconciliation_snapshot_refreshes_trade_report_health():
    from core.adapters.broker.easytrader_broker import EasytraderBroker
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker = EasytraderBroker(broker_account_id="TEST-002")
    broker._connected = True
    broker._trader = Mock(balance=[{
        "资金帐号": "TEST-002", "可用资金": 1000, "参考市值": 0, "总资产": 1000,
    }])
    observed = datetime.datetime.now().isoformat()
    with patch.object(broker, "get_account", return_value={
        "status": "fresh", "account_id": broker.account_id, "as_of": observed,
        "cash": 1000, "frozen": 0, "available": 1000,
    }), patch.object(broker, "get_positions", return_value=[]), \
         patch.object(broker, "get_orders", return_value=[]), \
         patch.object(broker, "get_trades", return_value=[]) as get_trades, \
         patch.object(BrokerTradeReportRepo, "get_health", return_value={
             "conflicts": 0, "unpriced": 1, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0,
         }) as get_health:
        snapshot = broker.get_reconciliation_snapshot()

    get_trades.assert_called_once_with(persist_reports=False)
    get_health.assert_called_once_with("easytrader", broker.account_id)
    assert snapshot["trade_report_health"]["unpriced"] == 1
    assert snapshot["snapshot_started_at"] <= snapshot["snapshot_completed_at"]


def test_easytrader_report_quarantine_lists_use_account_scoped_repositories():
    from backend.service.local_ledger import LocalLedgerService
    from core.data.repos.broker_order_report import BrokerOrderReportRepo
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker_account_id = "EASYT-TEST-003"
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "broker_account_id": broker_account_id,
    }), \
         patch.object(BrokerOrderReportRepo, "list_unresolved", return_value=[{"order_id": "O-1"}]) as orders, \
         patch.object(BrokerTradeReportRepo, "list_unresolved", return_value=[{"trade_id": "T-1"}]) as trades:
        order_reports = LocalLedgerService.list_unresolved_broker_reports(20)
        trade_reports = LocalLedgerService.list_unresolved_broker_trades(30)

    assert order_reports == [{"order_id": "O-1"}]
    assert trade_reports == [{"trade_id": "T-1"}]
    orders.assert_called_once_with("easytrader", broker_account_id, 20)
    trades.assert_called_once_with("easytrader", broker_account_id, 30)


def test_easytrader_report_replay_refreshes_order_and_trade_quarantine():
    from backend.service.local_ledger import LocalLedgerService
    from core.adapters.broker.easytrader_broker import EasytraderBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker = EasytraderBroker(broker_account_id="TEST-004")
    broker.get_orders = Mock(return_value=[{"id": "O-1"}])
    broker.get_trades = Mock(return_value=[{"id": "T-1"}])
    health = {"conflicts": 1, "unpriced": 2, "unbooked": 3, "unassigned": 4, "legacy_quarantined": 0}
    with patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(BrokerOrderReportRepo, "unresolved_count", return_value=1) as unresolved, \
         patch.object(BrokerTradeReportRepo, "get_health", return_value=health) as trade_health:
        result = LocalLedgerService.replay_broker_reports()

    assert result == {"observed_orders": 1, "observed_trades": 1,
                      "unresolved_reports": 1, "trade_report_health": health}
    broker.get_orders.assert_called_once_with()
    broker.get_trades.assert_called_once_with()
    unresolved.assert_called_once_with("easytrader", broker.account_id)
    trade_health.assert_called_once_with("easytrader", broker.account_id)


def test_report_archive_uses_explicit_configured_scope_before_account_binding():
    with patch.object(LocalLedgerRepo, "get_account", return_value={"broker_account_id": None}), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "TEST-005"):
        assert LocalLedgerService._get_report_scope() == ("qmt", "QMT-TEST-005")


def test_report_archive_refuses_unknown_bound_account_namespace():
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "broker_account_id": "BROKER-OTHER",
    }), pytest.raises(ValueError, match="标识无法解析"):
        LocalLedgerService._get_report_scope()


def test_broker_trade_report_health_separates_post_opening_unassigned_from_legacy_quarantine():
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    queries = []

    def fetchone(query, params=None):
        queries.append(query)
        if "active_unassigned" in query:
            return {"active_unassigned": 2, "legacy_quarantined": 3}
        return {"n": 0}

    with patch.object(Database, "fetchone", side_effect=fetchone):
        health = BrokerTradeReportRepo.get_health("qmt", "QMT-TEST")

    assert health == {
        "conflicts": 0, "unpriced": 0, "unbooked": 0,
        "unassigned": 2, "legacy_quarantined": 3,
    }
    assert "opening.event_type='opening'" in queries[-1]
    assert "r.observed_at>=opening.occurred_at" in queries[-1]


def test_broker_trade_unresolved_list_labels_pre_opening_rows_as_quarantined():
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    with patch.object(Database, "fetchall", return_value=[]) as fetchall:
        BrokerTradeReportRepo.list_unresolved("qmt", "QMT-TEST", 20)

    query, params = fetchall.call_args.args
    assert "'legacy_quarantined'" in query
    assert "r.observed_at>=opening.occurred_at" in query
    assert params[-1] == 20


def test_order_preparation_gate_allows_legacy_quarantine_but_rejects_new_unassigned_fill():
    from core.data.repos.local_ledger import require_fresh_broker_reconciliation

    account = {"version": 2, "cash": 1000, "frozen": 0}
    legacy_only = _ready_reconciliation(broker_trade_report_health={
        "conflicts": 0, "unpriced": 0, "unbooked": 0,
        "unassigned": 0, "legacy_quarantined": 3,
    })
    require_fresh_broker_reconciliation(legacy_only, account, "QMT-ONE")

    post_opening = _ready_reconciliation(broker_trade_report_health={
        "conflicts": 0, "unpriced": 0, "unbooked": 0,
        "unassigned": 1, "legacy_quarantined": 3,
    })
    with pytest.raises(ValueError, match="未解决问题"):
        require_fresh_broker_reconciliation(post_opening, account, "QMT-ONE")


@pytest.mark.parametrize("health,can_execute,reason,missing", [
    ({"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, True, None, []),
    ({"conflicts": 0, "unpriced": 2, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, False, "费用不完整", ["broker.trade_fees"]),
    ({"conflicts": 0, "unpriced": 0, "unbooked": 2, "unassigned": 0, "legacy_quarantined": 0}, False, "尚未入账", ["broker.trade_ledger_events"]),
    ({"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 2, "legacy_quarantined": 0}, False, "期初核准后", ["broker.trade_unassigned"]),
    ({"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 2}, True, None, []),
    (None, False, "完整性状态不可用", ["broker.trade_report_health"]),
    ({"conflicts": False, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, False, "完整性状态不可用", ["broker.trade_report_health"]),
])
def test_qmt_reconciliation_requires_complete_and_booked_trade_reports(health, can_execute, reason, missing):
    from core.adapters.broker.qmt_broker import QmtBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = QmtBroker("unused", "test")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "account": {"account_id": broker.account_id,
                    "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0}, "positions": [], "orders": [],
        "trade_report_health": health,
    })
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": broker.account_id,
    }), patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False), \
         patch.object(BrokerOrderReportRepo, "list_unfinished", return_value=[]), \
         patch.object(BrokerOrderReportRepo, "list_unresolved", return_value=[]):
        result = LocalLedgerService.reconcile()
    assert result["can_execute"] is can_execute
    assert result["missing_items"] == missing
    if reason:
        assert reason in result["reason"]


def test_qmt_reconciliation_blocks_unresolved_order_reports():
    from core.adapters.broker.qmt_broker import QmtBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = QmtBroker("unused", "test")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "account": {"account_id": broker.account_id,
                    "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True,
        "trade_report_health": {
            "conflicts": 0, "unpriced": 0, "unbooked": 0,
            "unassigned": 0, "legacy_quarantined": 0,
        },
        "positions": [], "orders": [],
    })
    unresolved = [{"report_key": "a" * 64, "order_id": "QMT-ORDER-1",
                   "symbol": "000001", "side": "buy", "quantity": 100,
                   "filled_qty": 0, "status": "unknown", "source": "query",
                   "disposition": "unresolved", "observed_at": "2026-10-30 10:00:00"}]
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": broker.account_id,
    }), patch.object(LocalLedgerRepo, "audit_integrity", return_value={
        "ok": True, "account_version": 2,
    }), patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False), \
         patch.object(BrokerOrderReportRepo, "list_unfinished", return_value=[]), \
         patch.object(BrokerOrderReportRepo, "list_unresolved", return_value=unresolved):
        result = LocalLedgerService.reconcile()

    assert result["can_execute"] is False
    assert result["reason"] == "券商回报日志中存在未核对委托"
    assert result["unresolved_order_reports"][0]["order_id"] == "QMT-ORDER-1"


def test_easytrader_reconciliation_accepts_verified_sellable_position():
    from core.adapters.broker.easytrader_broker import EasytraderBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo
    from core.data.repos.broker_trade_report import BrokerTradeReportRepo

    broker = EasytraderBroker(broker_account_id="TEST-EASY-001")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "account": {"account_id": broker.account_id,
                    "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": True, "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
        "positions": [{"symbol": "000001", "shares": 100, "sellable": 100}],
        "orders": [],
        "trade_report_health": {"conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0},
    })
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": broker.account_id,
    }), patch.object(LocalLedgerRepo, "audit_integrity", return_value={
        "ok": True, "account_version": 2,
    }), patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[{
             "code": "000001", "shares": 100,
         }]), patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False), \
         patch.object(BrokerOrderReportRepo, "list_unfinished", return_value=[]), \
         patch.object(BrokerOrderReportRepo, "list_unresolved", return_value=[]), \
         patch.object(BrokerTradeReportRepo, "get_health", return_value={
             "conflicts": 0, "unpriced": 0, "unbooked": 0, "unassigned": 0, "legacy_quarantined": 0,
         }), patch("core.shared.config.BROKER_TYPE", "easytrader"), \
         patch("core.shared.config.BROKER_FAKE", "fake"):
        result = LocalLedgerService.reconcile()

    assert result["broker_sellable"] == {"000001": 100}
    assert result["sellable_missing_codes"] == []
    assert result["can_execute"] is True


def test_qmt_reconciliation_preserves_all_missing_items():
    from core.adapters.broker.qmt_broker import QmtBroker
    from core.data.repos.broker_order_report import BrokerOrderReportRepo

    broker = QmtBroker("unused", "test")
    broker.get_reconciliation_snapshot = Mock(return_value={
        "account": {"account_id": broker.account_id,
                    "as_of": datetime.datetime.now().isoformat(),
                    "cash": 1000, "frozen": 0, "available": 1000},
        "identity_verified": False, "positions": [], "orders": [],
        "trade_report_health": {"conflicts": 0, "unpriced": 1, "unbooked": 1, "unassigned": 0, "legacy_quarantined": 0},
    })
    with patch.object(LocalLedgerRepo, "get_account", return_value={
        "status": "active", "version": 2, "cash": 1000, "frozen": 0,
        "available": 1000, "broker_account_id": broker.account_id,
    }), patch.object(LocalLedgerRepo, "get_legacy_counts", return_value={}), \
         patch("backend.service.local_ledger.Database.fetchall", return_value=[]), \
         patch("backend.api.trading.get_broker", return_value=broker), \
         patch.object(LocalOrderIntentRepo, "has_unresolved", return_value=False), \
         patch.object(BrokerOrderReportRepo, "list_unfinished", return_value=[]), \
         patch.object(BrokerOrderReportRepo, "list_unresolved", return_value=[]):
        result = LocalLedgerService.reconcile()

    assert result["can_execute"] is False
    assert result["missing_items"] == ["broker.trade_fees", "broker.verified_account_id"]


def test_real_broker_disconnect_does_not_fallback_to_fake():
    from backend.api import trading

    broker = Mock()
    broker.connected = False
    broker.account_id = "EASYT-YH"
    broker.connect.return_value = False
    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "easytrader"), \
         patch("core.shared.config.EASYTRAIDER_USER", "isolated-user"), \
         patch("core.shared.config.EASYTRAIDER_ACCOUNT_ID", "YH-001"), \
         patch("core.adapters.broker.create_broker", return_value=broker):
        with pytest.raises(RuntimeError, match="连接不可用"):
            trading.get_broker()
    assert broker.connected is False


def test_easytrader_without_broker_account_id_never_connects():
    from backend.api import trading

    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "easytrader"), \
         patch("core.shared.config.EASYTRAIDER_USER", "isolated-user"), \
         patch("core.shared.config.EASYTRAIDER_ACCOUNT_ID", ""), \
         patch("core.adapters.broker.create_broker", side_effect=AssertionError("不得连接未归属账户")):
        with pytest.raises(RuntimeError, match="ACCOUNT_ID 未配置"):
            trading.get_broker()


@pytest.mark.parametrize("connect_result", ["raises_after_connect", "false_after_connect"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_easytrader_partial_connection_is_cleaned_up_and_rejected(connect_result, cleanup_fails):
    from backend.api import trading

    broker = Mock()
    broker.connected = False
    broker.account_id = "EASYT-YH"

    def partial_connect():
        broker.connected = True
        if connect_result == "raises_after_connect":
            raise RuntimeError("session initialization failed")
        return False

    def disconnect():
        if cleanup_fails:
            raise RuntimeError("cleanup failed")
        broker.connected = False

    broker.connect.side_effect = partial_connect
    broker.disconnect.side_effect = disconnect
    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "easytrader"), \
         patch("core.shared.config.EASYTRAIDER_USER", "isolated-user"), \
         patch("core.shared.config.EASYTRAIDER_ACCOUNT_ID", "YH-001"), \
         patch("core.adapters.broker.create_broker", return_value=broker):
        with pytest.raises(RuntimeError, match="Easytrader 连接不可用"):
            trading.get_broker()

    assert broker.connected is cleanup_fails
    broker.disconnect.assert_called_once_with()
    assert trading._broker is None


def test_qmt_partial_initial_connection_is_rejected_when_cleanup_fails():
    from backend.api import trading
    from core.adapters.broker.qmt_broker import QmtBroker

    broker = QmtBroker("unused", "test")

    def partial_connect():
        broker._connected = True
        return False

    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "test"), \
         patch("core.adapters.broker.create_broker", return_value=broker), \
         patch.object(broker, "connect", side_effect=partial_connect), \
         patch.object(broker, "disconnect", side_effect=RuntimeError("cleanup failed")):
        with pytest.raises(RuntimeError, match="QMT 连接不可用"):
            trading.get_broker()
        assert broker.connected is True
        assert trading._broker is None


def test_failed_cached_qmt_reconnect_is_evicted_even_if_cleanup_fails():
    from backend.api import trading
    from core.adapters.broker.qmt_broker import QmtBroker

    cached = QmtBroker("unused", "test")
    replacement = Mock()
    replacement.connected = False
    replacement.account_id = "QMT-test"

    def partial_cached_connect():
        cached._connected = True
        raise RuntimeError("sync failed")

    def replacement_connect():
        replacement.connected = True
        return True

    replacement.connect.side_effect = replacement_connect
    with patch.object(trading, "_broker", cached), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "test"), \
         patch("core.adapters.broker.create_broker", return_value=replacement), \
         patch.object(cached, "connect", side_effect=partial_cached_connect), \
         patch.object(cached, "disconnect", side_effect=RuntimeError("cleanup failed")):
        with pytest.raises(RuntimeError, match="真实券商连接不可用"):
            trading.get_broker()
        assert trading._broker is None
        assert trading.get_broker() is replacement


def test_qmt_connect_exception_remains_unavailable_without_simulation_fallback():
    from backend.api import trading
    from core.adapters.broker.qmt_broker import QmtBroker

    broker = QmtBroker("unused", "test")
    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "test"), \
         patch("core.adapters.broker.create_broker", return_value=broker), \
         patch.object(broker, "connect", side_effect=RuntimeError("连接失败")), \
         patch.object(broker, "disconnect") as disconnect:
        with pytest.raises(RuntimeError, match="连接不可用"):
            trading.get_broker()
    assert broker.connected is False
    disconnect.assert_called_once_with()


def test_cached_qmt_reconnect_exception_cleans_half_connected_session():
    from backend.api import trading
    from core.adapters.broker.qmt_broker import QmtBroker

    broker = QmtBroker("unused", "test")

    def incomplete_connect():
        broker._connected = True
        raise RuntimeError("同步失败")

    with patch.object(trading, "_broker", broker), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "test"), \
         patch.object(broker, "connect", side_effect=incomplete_connect):
        with pytest.raises(RuntimeError, match="真实券商连接不可用"):
            trading.get_broker()
    assert broker.connected is False


def test_disconnected_broker_read_and_order_endpoints_return_503():
    from backend.api import trading

    calls = [
        trading.get_broker_info(),
        trading.get_account(),
        trading.place_order(trading.OrderRequest(symbol="000001", side="buy", quantity=100,
                                                price_type="limit", price=10), None),
        trading.cancel_order("123", None),
    ]
    with patch.object(trading, "get_broker", side_effect=RuntimeError("真实券商连接不可用")):
        for call in calls:
            with pytest.raises(HTTPException) as error:
                asyncio.run(call)
            assert error.value.status_code == 503


def test_real_direct_order_is_blocked_before_broker_submission_when_reconciliation_fails():
    from backend.api import trading

    broker = Mock()
    request = trading.OrderRequest(symbol="000001", side="buy", quantity=100,
                                   price_type="limit", price=10)
    with patch("backend.service.local_ledger.LocalLedgerService.reconcile", return_value={
        "can_execute": False, "reason": "券商存在未确认委托",
    }), patch("core.adapters.broker.is_fake_broker", return_value=False), \
         patch.object(trading, "get_broker", return_value=broker), \
         patch("core.shared.config.BROKER_TYPE", "qmt"):
        with pytest.raises(HTTPException) as error:
            asyncio.run(trading.place_order(request, None))

    assert error.value.status_code == 409
    assert "未确认委托" in error.value.detail
    broker.place_order.assert_not_called()


def test_broker_cache_rejects_stale_qmt_account():
    from backend.api import trading
    from core.adapters.broker.qmt_broker import QmtBroker

    old = QmtBroker("unused", "OLD")
    replacement = Mock()
    replacement.connected = True
    replacement.connect.return_value = True
    with patch.object(trading, "_broker", old), patch.object(old, "disconnect") as disconnect, \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "NEW"), \
         patch("core.adapters.broker.create_broker", return_value=replacement):
        result = trading.get_broker()
    assert result is replacement
    disconnect.assert_called_once_with()


def test_broker_cache_does_not_reuse_real_broker_after_simulation_switch():
    from backend.api import trading
    from core.adapters.broker.fake_broker import FakeBroker
    from core.adapters.broker.qmt_broker import QmtBroker

    old = QmtBroker("unused", "OLD")
    simulation = FakeBroker()
    with patch.object(trading, "_broker", old), patch.object(old, "disconnect") as disconnect, \
         patch("core.shared.config.BROKER_TYPE", "fake"), \
         patch("core.adapters.broker.create_broker", return_value=simulation), \
         patch("core.data.Database.is_available", return_value=False):
        result = trading.get_broker()
    assert result is simulation
    assert result.connected is True
    disconnect.assert_called_once_with()


def test_broker_cache_does_not_reuse_simulation_after_real_switch():
    from backend.api import trading
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker()
    replacement = Mock()
    replacement.connected = False
    replacement.connect.return_value = False
    with patch.object(trading, "_broker", simulation), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "NEW"), \
         patch("core.adapters.broker.create_broker", return_value=replacement):
        with pytest.raises(RuntimeError, match="连接不可用"):
            trading.get_broker()
    assert simulation.connected is False


def test_qmt_userdata_configuration_error_survives_broker_factory():
    from backend.api import trading

    replacement = Mock()
    replacement.connected = False
    replacement.connect.side_effect = RuntimeError("QMT userdata_mini 目录不存在或不可用")
    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "qmt"), \
         patch("core.shared.config.QMT_ACCOUNT", "TEST-ACCOUNT"), \
         patch("core.adapters.broker.create_broker", return_value=replacement):
        with pytest.raises(RuntimeError, match="userdata_mini 目录不存在或不可用"):
            trading.get_broker()
    replacement.disconnect.assert_called_once_with()
    assert trading._broker is None


def test_fake_broker_startup_restores_persisted_account_orders_and_trades():
    from backend.api import trading
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker()
    identity = {"account_id": "FAKE-LOCAL", "broker": "fake",
                "environment": "simulation", "source": "broker"}
    rows = [[{"id": "O-1", **identity}], [{"id": "T-1", **identity}]]
    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "fake"), \
         patch("core.adapters.broker.create_broker", return_value=simulation), \
         patch("core.data.Database.is_available", return_value=True), \
         patch("core.data.Database.fetchone", return_value={"cash": 1000, "frozen": 50}), \
         patch("core.data.Database.fetchall", side_effect=rows), \
         patch.object(simulation, "restore") as restore:
        result = trading.get_broker()
    assert result is simulation
    assert simulation.connected is True
    restore.assert_called_once_with(rows[0], rows[1])


@pytest.mark.parametrize("account", [
    {"cash": float("nan"), "frozen": 0},
    {"cash": 1000, "frozen": float("inf")},
    {"cash": 1000, "frozen": 1000.01},
])
def test_fake_broker_startup_rejects_invalid_persisted_cash_snapshot(account):
    from backend.api import trading
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker()
    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "fake"), \
         patch("core.adapters.broker.create_broker", return_value=simulation), \
         patch("core.data.Database.is_available", return_value=True), \
         patch("core.data.Database.fetchone", return_value=account), \
         patch("core.data.Database.fetchall", side_effect=[[], []]):
        with pytest.raises(RuntimeError, match="模拟账户状态恢复失败"):
            trading.get_broker()

    assert simulation.connected is False
    assert trading._broker is None


def test_fake_broker_restore_rebuilds_pending_buy_reserve_from_remaining_quantity():
    from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING

    simulation = FakeBroker(initial_cash=5000)
    simulation._frozen = 605
    order = {"id": "B-1", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
             "filled_qty": 40, "filled_price": 10, "price": 10, "status": STATUS_PENDING}
    trade = {"id": "T-B-1", "order_id": "B-1", "symbol": "000001.SZ", "side": "buy",
             "price": 10, "quantity": 40, "amount": 400, "commission": 5, "stamp_duty": 0,
             "trade_time": "2026-10-28 10:00:00"}
    simulation.restore([order], [trade])

    assert order["reserved_cash"] == 605
    assert simulation._frozen == 605


def test_fake_broker_restore_rejects_frozen_cash_mismatch():
    from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING

    simulation = FakeBroker(initial_cash=5000)
    simulation._frozen = 1004
    order = {"id": "B-2", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
             "filled_qty": 0, "price": 10, "status": STATUS_PENDING}

    with pytest.raises(ValueError, match="冻结资金与待处理买单预留不一致"):
        simulation.restore([order], [])


@pytest.mark.parametrize("cash,frozen", [
    (float("nan"), 0), (float("inf"), 0), (-1, 0), (True, 0),
    (1000, float("nan")), (1000, float("inf")), (1000, -1), (1000, True),
    (1000, 1000.01),
])
def test_fake_broker_restore_rejects_invalid_cash_snapshot(cash, frozen):
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker(initial_cash=cash)
    simulation._frozen = frozen

    with pytest.raises(ValueError, match="模拟账户资金快照无效|现金或冻结资金快照无效"):
        simulation.restore([], [])


@pytest.mark.parametrize("frozen", [604.99, 605.01])
def test_fake_broker_restore_rejects_one_cent_frozen_reserve_mismatch(frozen):
    from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING

    simulation = FakeBroker(initial_cash=5000)
    simulation._frozen = frozen
    order = {"id": "B-3", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
             "filled_qty": 40, "filled_price": 10, "price": 10, "status": STATUS_PENDING}
    trade = {"id": "T-B-3", "order_id": "B-3", "symbol": "000001.SZ", "side": "buy",
             "price": 10, "quantity": 40, "amount": 400, "commission": 5, "stamp_duty": 0,
             "trade_time": "2026-10-28 10:00:00"}

    with pytest.raises(ValueError, match="冻结资金与待处理买单预留不一致"):
        simulation.restore([order], [trade])


@pytest.mark.parametrize("price", [None, 0, -1, float("nan"), float("inf")])
def test_fake_broker_restore_rejects_invalid_pending_buy_reserve_price(price):
    from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING

    simulation = FakeBroker(initial_cash=5000)
    order = {"id": "B-4", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
             "filled_qty": 0, "price": price, "status": STATUS_PENDING}

    with pytest.raises(ValueError, match="待处理买单价格无效"):
        simulation.restore([order], [])


def test_fake_broker_restore_rejects_pending_sell_above_sellable_holdings():
    from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING

    simulation = FakeBroker(initial_cash=5000)
    trade = {"id": "T-1", "order_id": "B-1", "symbol": "000001.SZ", "side": "buy",
             "price": 10, "quantity": 100, "amount": 1000, "commission": 5,
             "stamp_duty": 0, "trade_time": "2026-10-28 10:00:00"}
    buy_order = {"id": "B-1", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
                 "filled_qty": 100, "filled_price": 10, "status": "filled"}
    order = {"id": "S-1", "symbol": "000001.SZ", "side": "sell", "quantity": 101,
             "filled_qty": 0, "price": 10, "status": STATUS_PENDING}

    with pytest.raises(ValueError, match="待处理卖单超过可卖持仓"):
        simulation.restore([buy_order, order], [trade])


def test_fake_broker_restore_uses_a_share_date_for_t_plus_one():
    from datetime import datetime
    from core.adapters.broker import fake_broker
    from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING

    simulation = FakeBroker(initial_cash=5000)
    trade = {"id": "T-TODAY", "order_id": "B-TODAY", "symbol": "000001.SZ", "side": "buy",
             "price": 10, "quantity": 100, "amount": 1000, "commission": 5,
             "stamp_duty": 0, "trade_time": "2026-10-28 00:30:00"}
    buy_order = {"id": "B-TODAY", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
                 "filled_qty": 100, "filled_price": 10, "status": "filled"}
    order = {"id": "S-TODAY", "symbol": "000001.SZ", "side": "sell", "quantity": 100,
             "filled_qty": 0, "price": 10, "status": STATUS_PENDING}

    with patch.object(fake_broker, "_market_now", return_value=datetime(2026, 10, 28, 9, 0)):
        with pytest.raises(ValueError, match="待处理卖单超过可卖持仓"):
            simulation.restore([buy_order, order], [trade])


@pytest.mark.parametrize("quantity", [100.5, True, 0, -100, "NaN"])
def test_fake_broker_restore_rejects_invalid_persisted_trade_quantity(quantity):
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker(initial_cash=5000)
    trade = {"id": "T-BAD", "symbol": "000001.SZ", "side": "buy", "quantity": quantity,
             "amount": 1000, "commission": 5, "trade_time": "2026-10-28 10:00:00"}

    with pytest.raises(ValueError, match="无效股数"):
        simulation.restore([], [trade])


def test_fake_broker_restore_rejects_unknown_persisted_trade_side():
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker(initial_cash=5000)
    trade = {"id": "T-BAD", "symbol": "000001.SZ", "side": "transfer", "quantity": 100,
             "amount": 1000, "commission": 5, "trade_time": "2026-10-28 10:00:00"}

    with pytest.raises(ValueError, match="成交方向无效"):
        simulation.restore([], [trade])


def test_fake_broker_restore_rejects_filled_order_without_matching_trade():
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker(initial_cash=5000)
    order = {"id": "B-MISSING-TRADE", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
             "filled_qty": 100, "filled_price": 10, "status": "filled"}

    with pytest.raises(ValueError, match="订单成交股数与成交事件不一致"):
        simulation.restore([order], [])


@pytest.mark.parametrize("trade_update,error", [
    ({"quantity": 99, "amount": 990}, "订单成交股数与成交事件不一致"),
    ({"side": "sell"}, "成交与委托证券或方向不一致"),
    ({"symbol": "000002.SZ"}, "成交与委托证券或方向不一致"),
    ({"amount": 1001}, "成交金额或费用无效"),
])
def test_fake_broker_restore_rejects_order_trade_snapshot_mismatch(trade_update, error):
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker(initial_cash=5000)
    order = {"id": "B-MISMATCH", "symbol": "000001.SZ", "side": "buy", "quantity": 100,
             "filled_qty": 100, "filled_price": 10, "status": "filled"}
    trade = {"id": "T-MISMATCH", "order_id": "B-MISMATCH", "symbol": "000001.SZ", "side": "buy",
             "price": 10, "quantity": 100, "amount": 1000, "commission": 5, "stamp_duty": 0,
             "trade_time": "2026-10-28 10:00:00"}
    trade.update(trade_update)

    with pytest.raises(ValueError, match=error):
        simulation.restore([order], [trade])


def test_fake_broker_restore_rejects_orphan_trade():
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker(initial_cash=5000)
    trade = {"id": "T-ORPHAN", "order_id": "MISSING", "symbol": "000001.SZ", "side": "buy",
             "price": 10, "quantity": 100, "amount": 1000, "commission": 5, "stamp_duty": 0,
             "trade_time": "2026-10-28 10:00:00"}

    with pytest.raises(ValueError, match="成交缺少对应委托"):
        simulation.restore([], [trade])


@pytest.mark.parametrize("quantity,filled_qty", [
    (100.5, 0), (100, 20.5), (100, 101), (True, 0), (100, True),
])
def test_fake_broker_restore_rejects_invalid_pending_order_quantities(quantity, filled_qty):
    from core.adapters.broker.fake_broker import FakeBroker, STATUS_PENDING

    simulation = FakeBroker(initial_cash=5000)
    simulation._frozen = 0
    order = {"id": "O-BAD", "symbol": "000001.SZ", "side": "buy", "quantity": quantity,
             "filled_qty": filled_qty, "price": 10, "status": STATUS_PENDING}

    with pytest.raises(ValueError, match="无效股数|无效的待处理委托数量|成交股数超过委托股数"):
        simulation.restore([order], [])


def test_fake_broker_startup_rejects_misidentified_persisted_trade():
    from backend.api import trading
    from core.adapters.broker.fake_broker import FakeBroker

    simulation = FakeBroker()
    rows = [[{"id": "O-1", "account_id": "FAKE-LOCAL", "broker": "fake",
              "environment": "simulation", "source": "broker"}],
            [{"id": "T-1", "account_id": "FAKE-LOCAL", "broker": "qmt",
              "environment": "live", "source": "broker"}]]
    with patch.object(trading, "_broker", None), \
         patch("core.shared.config.BROKER_TYPE", "fake"), \
         patch("core.adapters.broker.create_broker", return_value=simulation), \
         patch("core.data.Database.is_available", return_value=True), \
         patch("core.data.Database.fetchone", return_value={"cash": 1000, "frozen": 50}), \
         patch("core.data.Database.fetchall", side_effect=rows), \
         patch.object(simulation, "restore") as restore:
        with pytest.raises(RuntimeError, match="模拟账户状态恢复失败"):
            trading.get_broker()
    assert simulation.connected is False
    restore.assert_not_called()


def test_concurrent_broker_initialization_uses_one_session():
    from backend.api import trading
    from core.adapters.broker.fake_broker import FakeBroker

    def create_once(*args, **kwargs):
        time.sleep(0.03)
        return FakeBroker()

    with patch.object(trading, "_broker", None), patch("core.shared.config.BROKER_TYPE", "fake"), \
         patch("core.adapters.broker.create_broker", side_effect=create_once) as create, \
         patch("core.data.Database.is_available", return_value=False):
        with ThreadPoolExecutor(max_workers=2) as pool:
            brokers = list(pool.map(lambda _: trading.get_broker(), range(2)))
    assert brokers[0] is brokers[1]
    assert create.call_count == 1


def test_broker_binding_rejects_other_identity():
    cursor = Mock()
    cursor.fetchone.return_value = {
        "status": "active", "version": 2, "broker_account_id": "QMT-ONE", "cash": 1000,
    }
    with patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="另一券商资金账号"):
            LocalOrderIntentRepo.bind_broker("QMT-TWO", 2, "admin", None)
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("expected_version", [True, 2.5, 0])
def test_broker_binding_repository_rejects_invalid_account_version_before_transaction(expected_version):
    with patch("core.data.repos.local_order_intent.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="账户版本无效"):
            LocalOrderIntentRepo.bind_broker("QMT-ONE", expected_version, "admin", {})

    transaction.assert_not_called()


@pytest.mark.parametrize("broker_account_id", [None, 123, " QMT-ONE", "QMT-ONE "])
def test_broker_binding_repository_rejects_invalid_broker_identity_before_transaction(broker_account_id):
    with patch("core.data.repos.local_order_intent.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="券商资金账号无效"):
            LocalOrderIntentRepo.bind_broker(broker_account_id, 1, "admin", {})

    transaction.assert_not_called()


def test_broker_binding_rejects_unresolved_intent_inside_account_transaction():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "broker_account_id": None, "cash": 1000, "frozen": 0},
        {"intent_id": "00000000-0000-4000-8000-000000000099"},
    ]
    reconciliation = _ready_reconciliation()
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="未完成的本地委托意图"):
            LocalOrderIntentRepo.bind_broker("QMT-ONE", 2, "admin", reconciliation)

    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_broker_binding_rejects_expired_snapshot_inside_account_lock():
    cursor = Mock()
    cursor.fetchone.return_value = {
        "status": "active", "version": 2, "broker_account_id": None, "cash": 1000, "frozen": 0,
    }
    now = datetime.datetime.now()
    reconciliation = _ready_reconciliation(
        broker_as_of=(now - datetime.timedelta(seconds=70)).isoformat(),
    )
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="快照已过期"):
            LocalOrderIntentRepo.bind_broker("QMT-ONE", 2, "admin", reconciliation)
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_broker_binding_invalidates_prebinding_signals_and_returns_details():
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "broker_account_id": None, "cash": 1000, "frozen": 0},
        None,
    ]
    details = [{"signal_id": 91, "code": "000001", "side": "buy",
                "execution_status": "pending", "planned_shares": 100,
                "filled_shares": 0, "invalidated_remaining_shares": 100}]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch.object(RebalanceSignalRepo, "invalidate_stale_account_signals", return_value=details) as invalidate, \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        result = LocalOrderIntentRepo.bind_broker(
            "QMT-ONE", 2, "admin", _ready_reconciliation(broker_account_id="QMT-ONE"),
        )

    assert result["invalidated_signal_ids"] == [91]
    assert result["invalidated_signal_details"] == details
    invalidate.assert_called_once_with(cursor, "LOCAL-LIVE", 3)
    event_insert = next(call for call in cursor.execute.call_args_list
                        if "INSERT INTO local_ledger_events" in call.args[0])
    assert json.loads(event_insert.args[1][-1])["invalidated_signal_details"] == details


@pytest.mark.parametrize("field,value", [
    ("quantity", 100.5), ("quantity", True),
    ("signal_id", 7.5), ("expected_account_version", True),
    ("limit_price", "10.0001"), ("fee_cap", "1.001"),
])
def test_prepare_repository_rejects_inexact_or_invalid_values_before_transaction(field, value):
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "broker_account_id": "QMT-ONE", "expected_account_version": 2,
              "quantity": 100, "limit_price": "10.00", "fee_cap": "1.00",
              "actor": "admin"}
    detail[field] = value
    with patch("core.data.repos.local_order_intent.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="无效"):
            LocalOrderIntentRepo.prepare(detail, lambda rows: "fp")

    transaction.assert_not_called()


def test_prepare_intent_reserves_cash_and_writes_audit_event():
    from core.domain.portfolio.decision import decision_fingerprint, holdings_fingerprint_rows

    cursor = Mock()
    cursor.rowcount = 1
    holding = {"code": "000001", "shares": 100, "cost_price": 9.5,
               "buy_date": datetime.date(2026, 10, 29)}
    input_fingerprint = decision_fingerprint(holdings_fingerprint_rows([holding]))
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 2000, "frozen": 0,
         "broker_account_id": "QMT-ONE"}, None, None,
        {"account_id": "LOCAL-LIVE", "account_version": 2, "planned_shares": 100,
         "filled_shares": 0, "execution_status": "pending", "side": "buy",
         "code": "000001", "planned_price": 10, "input_fingerprint": input_fingerprint,
         "batch_id": "B1", "created_at": datetime.datetime.now(),
         "created_date": datetime.date.today(), "database_date": datetime.date.today()},
    ]
    cursor.fetchall.return_value = [holding]
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "broker_account_id": "QMT-ONE", "expected_account_version": 2,
              "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin",
              "reconciliation": _ready_reconciliation(
                  broker_cash=2000, broker_available=2000,
                  broker_positions={"000001": 100},
                  broker_sellable={"000001": 100},
                  local_positions={"000001": 100},
              )}
    details = [{"signal_id": 88, "code": "600000", "side": "buy",
                "execution_status": "pending", "planned_shares": 200,
                "filled_shares": 0, "invalidated_remaining_shares": 200}]
    with patch.object(RebalanceSignalRepo, "invalidate_stale_account_signals", return_value=details) as invalidate, \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        result = LocalOrderIntentRepo.prepare(detail, decision_fingerprint)
    assert result["reserved_cash"] == 1001
    assert result["state"] == "prepared"
    assert result["invalidated_signal_ids"] == [88]
    assert result["invalidated_signal_details"] == details
    invalidate.assert_called_once_with(cursor, "LOCAL-LIVE", 3)
    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("SELECT code,shares,cost_price,buy_date" in sql for sql in statements)
    assert any("SET frozen=frozen+%s" in sql for sql in statements)
    assert any("INSERT INTO local_order_intents" in sql for sql in statements)
    assert any("'reserve'" in sql for sql in statements)
    assert any("CURRENT_DATE() AS database_date" in sql for sql in statements)
    event_insert = next(call for call in cursor.execute.call_args_list
                        if "INSERT INTO local_ledger_events" in call.args[0])
    assert json.loads(event_insert.args[1][-1])["invalidated_signal_details"] == details


def test_prepare_intent_rejects_boolean_only_broker_reconciliation_evidence():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 2000, "frozen": 0,
         "broker_account_id": "QMT-ONE"}, None,
    ]
    detail = {
        "intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
        "broker_account_id": "QMT-ONE", "expected_account_version": 2,
        "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin",
        "reconciliation": {
            "can_execute": True, "broker_account_id": "QMT-ONE", "local_version": 2,
            "broker_received_at": datetime.datetime.now().isoformat(),
            "broker_as_of": datetime.datetime.now().isoformat(),
        },
    }
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="对账证据不完整"):
            LocalOrderIntentRepo.prepare(detail, lambda rows: "fp")
    assert not any("UPDATE " in call.args[0] or "INSERT " in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_prepare_intent_insufficient_cash_has_no_writes():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 500, "frozen": 0,
         "broker_account_id": "QMT-ONE"}, None, None,
        {"account_id": "LOCAL-LIVE", "account_version": 2, "planned_shares": 100,
         "filled_shares": 0, "execution_status": "pending", "side": "buy",
         "code": "000001", "planned_price": 10, "input_fingerprint": "fp",
         "created_at": datetime.datetime.now(),
         "created_date": datetime.date.today(), "database_date": datetime.date.today()},
    ]
    cursor.fetchall.return_value = []
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "broker_account_id": "QMT-ONE", "expected_account_version": 2,
              "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin",
              "reconciliation": _ready_reconciliation(
                  broker_cash=500, broker_available=500,
              )}
    with patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="可用现金不足"):
            LocalOrderIntentRepo.prepare(detail, lambda rows: "fp")
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_prepare_intent_uses_database_market_date_and_rejects_stale_signal():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 2000, "frozen": 0,
         "broker_account_id": "QMT-ONE"}, None, None,
        {"account_id": "LOCAL-LIVE", "account_version": 2, "planned_shares": 100,
         "filled_shares": 0, "execution_status": "pending", "side": "buy",
         "code": "000001", "planned_price": 10, "input_fingerprint": "fp", "batch_id": "B1",
         "created_date": datetime.date(2026, 10, 28),
         "database_date": datetime.date(2026, 10, 29)},
    ]
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "broker_account_id": "QMT-ONE", "expected_account_version": 2,
              "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin",
              "reconciliation": _ready_reconciliation(
                  broker_cash=2000, broker_available=2000,
              )}
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="信号已过期"):
            LocalOrderIntentRepo.prepare(detail, lambda rows: "fp")
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("invalid,expected", [
    ("received_stale", "快照已过期"),
    ("broker_stale", "快照已过期"),
    ("account_mismatch", "账户或版本已变化"),
    ("version_mismatch", "账户或版本已变化"),
    ("missing_time", "时间无效"),
])
def test_prepare_intent_rechecks_reconciliation_inside_account_lock(invalid, expected):
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 2000, "frozen": 0,
         "broker_account_id": "QMT-ONE"}, None,
    ]
    now = datetime.datetime.now()
    reconciliation = _ready_reconciliation(broker_cash=2000, broker_available=2000)
    if invalid == "received_stale":
        reconciliation["broker_received_at"] = (now - datetime.timedelta(seconds=70)).isoformat()
    elif invalid == "broker_stale":
        reconciliation["broker_as_of"] = (now - datetime.timedelta(seconds=70)).isoformat()
    elif invalid == "account_mismatch":
        reconciliation["broker_account_id"] = "QMT-TWO"
    elif invalid == "version_mismatch":
        reconciliation["local_version"] = 1
    else:
        reconciliation.pop("broker_as_of")
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "broker_account_id": "QMT-ONE", "expected_account_version": 2,
              "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin",
              "reconciliation": reconciliation}
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match=expected):
            LocalOrderIntentRepo.prepare(detail, lambda rows: "fp")
    assert not any("UPDATE " in call.args[0] for call in cursor.execute.call_args_list)


def test_corrupt_ledger_blocks_intent_reservation():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 2000, "frozen": 0}, None,
    ]
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "broker_account_id": "QMT-ONE", "expected_account_version": 2,
              "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin"}
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": False}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="账本完整性校验失败"):
            LocalOrderIntentRepo.prepare(detail, lambda rows: "fp")
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_corrupt_ledger_blocks_broker_fill_booking():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 2000, "frozen": 1001},
        {"account_id": "LOCAL-LIVE", "broker_order_id": "123", "broker_account_id": "QMT-ONE"},
        {"trade_id": "T-1"}, None, None,
    ]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": False}), \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="账本完整性校验失败"):
            LocalOrderIntentRepo.record_broker_fill("intent-1", "T-1", 0.3, 0, "admin", lambda rows: "fp")
    assert not any("UPDATE holdings" in call.args[0] for call in cursor.execute.call_args_list)


def test_manual_fill_trade_id_blocks_broker_fill_booking():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 2000, "frozen": 1001},
        {"account_id": "LOCAL-LIVE", "broker_order_id": "123", "broker_account_id": "QMT-ONE"},
        {"trade_id": "T-1"}, None, {"event_key": "manual:LOCAL-LIVE:existing"},
    ]
    with patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="已由人工成交入账"):
            LocalOrderIntentRepo.record_broker_fill("intent-1", "T-1", 0.3, 0, "admin", lambda rows: "fp")
    assert not any("UPDATE holdings" in call.args[0] for call in cursor.execute.call_args_list)


def test_cancel_prepared_releases_reservation_once():
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 3, "cash": 2000, "frozen": 1001},
        {"account_id": "LOCAL-LIVE", "state": "prepared", "remaining_reserved_cash": 1001,
         "signal_id": 7, "code": "000001", "side": "buy", "quantity": 100,
         "account_version": 3},
        {"batch_id": "batch-1", "account_id": "LOCAL-LIVE", "execution_status": "pending",
         "account_version": 3, "code": "000001", "planned_shares": 100, "filled_shares": 0},
    ]
    details = [{"signal_id": 92, "code": "600000", "side": "buy",
                "execution_status": "pending", "planned_shares": 200,
                "filled_shares": 0, "invalidated_remaining_shares": 200}]
    with patch.object(RebalanceSignalRepo, "invalidate_stale_account_signals", return_value=details) as invalidate, \
         patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        result = LocalOrderIntentRepo.cancel_prepared("intent-1", "admin")
    assert result["state"] == "cancelled"
    assert result["invalidated_signal_ids"] == [92]
    assert result["invalidated_signal_details"] == details
    invalidate.assert_called_once_with(cursor, "LOCAL-LIVE", 4)
    assert any("SET frozen=frozen-%s" in call.args[0] for call in cursor.execute.call_args_list)
    assert any("UPDATE rebalance_signals SET account_version=%s" in call.args[0]
               for call in cursor.execute.call_args_list)
    assert any("'reserve_release'" in call.args[0] for call in cursor.execute.call_args_list)
    event_insert = next(call for call in cursor.execute.call_args_list
                        if "INSERT INTO local_ledger_events" in call.args[0])
    assert json.loads(event_insert.args[1][-1])["invalidated_signal_details"] == details


def test_repeated_prepare_returns_original_intent_without_second_reserve():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 4, "cash": 2000, "frozen": 0},
        {"intent_id": "00000000-0000-4000-8000-000000000099", "account_id": "LOCAL-LIVE",
         "broker_account_id": "QMT-ONE", "signal_id": 7, "quantity": 100,
         "limit_price": 10, "fee_cap": 1, "state": "cancelled",
         "reserved_cash": 1001, "account_version": 3},
        {"payload_json": {"invalidated_signal_ids": [93],
                           "invalidated_signal_details": [{"signal_id": 93,
                                                           "invalidated_remaining_shares": 50}]}},
    ]
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "broker_account_id": "QMT-ONE", "expected_account_version": 2,
              "quantity": 100, "limit_price": 10, "fee_cap": 1, "actor": "admin"}
    with patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        result = LocalOrderIntentRepo.prepare(detail, lambda rows: "fp")
    assert result["already_prepared"] is True
    assert result["state"] == "cancelled"
    assert result["invalidated_signal_ids"] == [93]
    assert result["invalidated_signal_details"][0]["invalidated_remaining_shares"] == 50
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_nonprepared_intent_cannot_release_cash():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 3, "cash": 2000, "frozen": 1001},
        {"account_id": "LOCAL-LIVE", "state": "submitted", "reserved_cash": 1001},
    ]
    with patch("core.data.repos.local_order_intent.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="不能仅在本地取消"):
            LocalOrderIntentRepo.cancel_prepared("intent-1", "admin")
    assert not any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)


def test_prepare_buy_checks_strategy_before_reconciliation():
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "expected_account_version": 2, "quantity": 100,
              "limit_price": "10.000", "fee_cap": "1.00", "actor": "admin"}
    with patch("backend.service.local_ledger.LocalOrderIntentRepo.get_by_id", return_value=None), \
         patch("core.data.repos.rebalance.RebalanceSignalRepo.get_by_id", return_value={
             "side": "buy", "profile_id": "profile", "profile_version": 1,
             "profile_fingerprint": "old", "source_mode": "live",
         }), patch("backend.service.strategy_profile.StrategyProfileService.ensure_execution_allowed",
                  side_effect=ValueError("策略已变化")), \
         patch.object(LocalLedgerService, "reconcile") as reconcile:
        with pytest.raises(ValueError, match="策略已变化"):
            LocalLedgerService.prepare_intent(detail)
    reconcile.assert_not_called()


def test_prepare_service_passes_broker_snapshot_into_transaction():
    detail = {"intent_id": "00000000-0000-4000-8000-000000000099", "signal_id": 7,
              "expected_account_version": 2, "quantity": 100,
              "limit_price": "10.000", "fee_cap": "1.00", "actor": "admin"}
    reconciliation = {
        "can_execute": True, "broker_account_id": "QMT-ONE", "local_version": 2,
        "broker_received_at": datetime.datetime.now().isoformat(),
        "broker_as_of": datetime.datetime.now().isoformat(),
        "broker_sellable": {"000001": 100},
    }
    with patch.object(LocalOrderIntentRepo, "get_by_id", return_value=None), \
         patch("core.data.repos.rebalance.RebalanceSignalRepo.get_by_id", return_value={
             "side": "sell", "code": "000001",
         }), patch.object(LocalLedgerService, "reconcile", return_value=reconciliation), \
         patch.object(LocalOrderIntentRepo, "prepare", return_value={"state": "prepared"}) as prepare:
        result = LocalLedgerService.prepare_intent(detail)
    assert result["state"] == "prepared"
    assert prepare.call_args.args[0]["reconciliation"] is reconciliation


def test_live_rebalance_session_rechecks_account_version_inside_transaction():
    from core.data.repos.session import RebalanceSessionRepo
    from core.data.repos.strategy_profile import StrategyProfileRepo

    cursor = Mock()
    cursor.fetchone.return_value = {"status": "active", "version": 4}
    cursor.lastrowid = 12
    with patch.object(StrategyProfileRepo, "ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.session.Database.transaction", side_effect=lambda op: op(cursor)):
        result = RebalanceSessionRepo.create(
            "session-v1", "rebalance", "2026-10-29", {"cash": 1000}, [], [],
            account_id="LOCAL-LIVE", account_version=4,
        )
    assert result == 12
    assert any("SELECT status,version FROM local_ledger_accounts" in call.args[0]
               for call in cursor.execute.call_args_list)
    assert any("INSERT INTO rebalance_sessions" in call.args[0] for call in cursor.execute.call_args_list)


def test_live_rebalance_session_stale_version_does_not_persist():
    from core.data.repos.session import RebalanceSessionRepo
    from core.data.repos.strategy_profile import StrategyProfileRepo

    cursor = Mock()
    cursor.fetchone.return_value = {"status": "active", "version": 5}
    with patch.object(StrategyProfileRepo, "ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.session.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="账户状态或版本已变化"):
            RebalanceSessionRepo.create(
                "session-v2", "rebalance", "2026-10-29", {"cash": 1000}, [], [],
                account_id="LOCAL-LIVE", account_version=4,
            )
    assert not any("INSERT INTO rebalance_sessions" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_live_rebalance_session_rejects_corrupt_ledger():
    from core.data.repos.session import RebalanceSessionRepo
    from core.data.repos.strategy_profile import StrategyProfileRepo

    cursor = Mock()
    cursor.fetchone.return_value = {"status": "active", "version": 4}
    with patch.object(StrategyProfileRepo, "ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": False}), \
         patch("core.data.repos.session.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="账本完整性校验失败"):
            RebalanceSessionRepo.create(
                "session-corrupt", "rebalance", "2026-10-29", {"cash": 1000}, [], [],
                account_id="LOCAL-LIVE", account_version=4,
            )
    assert not any("INSERT INTO rebalance_sessions" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_rebalance_signal_read_queries_can_be_scoped_to_account():
    with patch("core.data.repos.rebalance.Database.fetchall", return_value=[] ) as fetchall, \
         patch("core.data.repos.rebalance.Database.fetchone", return_value=None) as fetchone:
        RebalanceSignalRepo.get_by_batch("shared-batch", "LOCAL-LIVE", execution_status="pending")
        RebalanceSignalRepo.get_by_code("000001")
        RebalanceSignalRepo.get_by_id(42, account_id="LOCAL-LIVE")
        RebalanceSignalRepo.get_history(days=7, limit=10, account_id="LOCAL-LIVE")

    assert fetchall.call_args_list[0].args[0].endswith("WHERE batch_id=%s AND account_id=%s AND execution_status=%s ORDER BY CASE urgency WHEN 'stop_loss' THEN 3 WHEN 'take_profit' THEN 2 ELSE 1 END DESC, id ASC")
    assert fetchall.call_args_list[0].args[1] == ("shared-batch", "LOCAL-LIVE", "pending")
    assert fetchall.call_args_list[1].args[0].endswith("WHERE code=%s AND execution_status=%s AND account_id=%s ORDER BY created_at DESC")
    assert fetchall.call_args_list[1].args[1] == ("000001", "pending", "LOCAL-LIVE")
    assert fetchone.call_args.args[0].endswith("WHERE id=%s AND account_id=%s")
    assert fetchone.call_args.args[1] == (42, "LOCAL-LIVE")
    assert "AND account_id=%s" in fetchall.call_args_list[2].args[0]
    assert fetchall.call_args_list[2].args[1] == (7, "LOCAL-LIVE", 10)


def test_rebalance_session_read_queries_can_be_scoped_to_account():
    from core.data.repos.session import RebalanceSessionRepo

    with patch("core.data.repos.session.Database.fetchall", return_value=[]) as fetchall, \
         patch("core.data.repos.session.Database.fetchone", return_value=None) as fetchone:
        RebalanceSessionRepo.get_recent(days=14, limit=5, account_id="LOCAL-LIVE")
        RebalanceSessionRepo.get_by_batch("shared-batch", account_id="LOCAL-LIVE")
        RebalanceSessionRepo.get_latest(check_type="rebalance", account_id="LOCAL-LIVE")

    assert "AND account_id=%s" in fetchall.call_args.args[0]
    assert fetchall.call_args.args[1] == (14, "LOCAL-LIVE", 5)
    assert fetchone.call_args_list[0].args[0].endswith("WHERE batch_id=%s AND account_id=%s")
    assert fetchone.call_args_list[0].args[1] == ("shared-batch", "LOCAL-LIVE")
    assert fetchone.call_args_list[1].args[0].endswith("WHERE account_id=%s AND check_type=%s ORDER BY eval_date DESC LIMIT 1")
    assert fetchone.call_args_list[1].args[1] == ("LOCAL-LIVE", "rebalance")


def test_mark_skipped_rejects_non_local_account_before_update():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active"},
        {"account_id": "OTHER-ACCOUNT"},
    ]
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="仅可更改本地实盘账户"):
            RebalanceSignalRepo.mark_skipped(42, "not my account")
    assert not any("UPDATE rebalance_signals" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_mark_skipped_rejects_uninitialized_account_before_signal_lookup():
    cursor = Mock()
    cursor.fetchone.return_value = {"status": "uninitialized"}
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="期初尚未核准"):
            RebalanceSignalRepo.mark_skipped(42, "not filled")
    assert cursor.execute.call_count == 1
    assert not any("UPDATE rebalance_signals" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_mark_skipped_scopes_transactional_update_to_local_account():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active"},
        {"account_id": "LOCAL-LIVE"},
        None,
    ]
    cursor.rowcount = 1
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        assert RebalanceSignalRepo.mark_skipped(42, "not filled") == 1
    update_call = next(call for call in cursor.execute.call_args_list
                       if "UPDATE rebalance_signals" in call.args[0])
    assert "id=%s AND account_id=%s AND execution_status='pending'" in update_call.args[0]
    assert update_call.args[1] == ('{"reason": "not filled"}', "not filled", 42, "LOCAL-LIVE")


def test_service_mark_skipped_rejects_other_account_without_mutation():
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value={
        "id": 42, "account_id": "OTHER-ACCOUNT", "execution_status": "pending",
    }), patch("core.data.RebalanceSignalRepo.mark_skipped") as mark_skipped:
        result = RebalanceService.mark_skipped(42, "skip")
    assert "仅可更改本地实盘账户" in result["error"]
    mark_skipped.assert_not_called()


def test_latest_session_route_is_not_captured_by_batch_route():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from core.data.repos.session import RebalanceSessionRepo
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    with patch.object(RebalanceSessionRepo, "get_latest", return_value={"batch_id": "latest-real"}) as get_latest, \
         patch.object(RebalanceSessionRepo, "get_by_batch") as get_by_batch:
        response = TestClient(app).get("/sessions/latest")

    assert response.status_code == 200
    assert response.json() == {"data": {"batch_id": "latest-real"}}
    get_latest.assert_called_once_with(check_type=None, account_id="LOCAL-LIVE")
    get_by_batch.assert_not_called()


def test_real_cancel_never_reaches_broker():
    import asyncio
    from fastapi import HTTPException
    from backend.api import trading

    broker = Mock()
    with patch.object(trading, "get_broker", return_value=broker), \
         patch("core.adapters.broker.is_fake_broker", return_value=False):
        with pytest.raises(HTTPException) as error:
            asyncio.run(trading.cancel_order("external-order"))
    assert error.value.status_code == 409
    broker.cancel_order.assert_not_called()


@pytest.mark.parametrize("status,available", [("active", 1000), ("uninitialized", None)])
def test_daily_risk_uses_local_account_and_blocks_unapproved_persistence(status, available):
    holding = {"code": "000001", "name": "平安银行", "shares": 100, "cost_price": 10}
    account = {"status": status, "version": 3 if status == "active" else 0,
               "cash": 1200 if status == "active" else None,
               "frozen": 200 if status == "active" else 0,
               "available": available}
    with patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={
             "ok": True, "account_version": account["version"],
         }), \
         patch("core.data.HoldingsRepo.get_all", return_value=[holding]) as get_holdings, \
         patch("backend.service.rebalance._get_prices_batch", return_value=({"000001": 8}, "realtime")), \
         patch("backend.service.rebalance._load_half_sold", return_value=set()), \
         patch("core.data.RebalanceSignalRepo.batch_create") as create_signals, \
         patch("core.data.RebalanceSessionRepo.create") as create_session:
        result = RebalanceService.daily_risk_check()
    get_holdings.assert_called_once_with(account_id="LOCAL-LIVE")
    assert result["signals"][0]["planned_shares"] == 100
    assert result["signals"][0]["suggested_shares"] == 100
    if status == "active":
        payload = create_signals.call_args.kwargs
        assert payload["account_id"] == "LOCAL-LIVE"
        assert payload["account_version"] == 3
        assert payload["session_payload"]["account_json"] == {
            "total_assets": 2000, "cash": 1000, "market_value": 800,
            "total_cash": 1200, "frozen_cash": 200,
            "cash_source": "local_ledger",
        }
        assert payload["session_payload"]["account_id"] == "LOCAL-LIVE"
        create_session.assert_not_called()
    else:
        assert result["execution_blocked"] is True
        create_signals.assert_not_called()
        create_session.assert_not_called()


def test_daily_risk_skips_signals_when_active_ledger_integrity_fails():
    holding = {"code": "000001", "name": "平安银行", "shares": 100, "cost_price": 10}
    account = {"status": "active", "version": 4, "available": 1000}
    with patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={
             "ok": False, "account_version": 4,
         }), \
         patch("core.data.HoldingsRepo.get_all", return_value=[holding]) as get_holdings, \
         patch("core.data.RebalanceSignalRepo.batch_create") as create_signals, \
         patch("core.data.RebalanceSessionRepo.create") as create_session:
        result = RebalanceService.daily_risk_check()

    assert result["execution_blocked"] is True
    assert result["data_status"] == "unavailable"
    assert "账本不一致" in result["execution_block_reason"]
    get_holdings.assert_not_called()
    create_signals.assert_not_called()
    create_session.assert_not_called()


@pytest.mark.parametrize("integrity", [
    {"ok": True, "account_version": 3},
    {"ok": False, "account_version": 3},
])
@pytest.mark.parametrize("preview_cash", [0, 500])
def test_explicit_rebalance_cash_is_preview_only_and_never_persists_signals(integrity, preview_cash):
    from types import SimpleNamespace

    decision_date = datetime.date.today()
    account = {"status": "active", "version": 3, "cash": 1200,
               "frozen": 200, "available": 1000}
    save = {"candidates": [{"code": "000001", "name": "测试股"}],
            "pool": "all", "eval_date": decision_date.isoformat()}

    class PreviewAdvisor:
        def __init__(self, **kwargs):
            self._allocator = SimpleNamespace(config=SimpleNamespace())

        def generate_advice_from_screening(self, **kwargs):
            return {"orders": [], "classification": {}, "constraints": [],
                    "turnover": 0, "observe_candidates": [], "top_n_codes": [], "skipped": []}

    with patch("core.shared.trading_time.get_latest_trading_day", return_value=decision_date), \
         patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value=integrity), \
         patch("core.data.SettingsRepo.get", return_value="2500"), \
         patch("core.data.ScreeningSaveRepo.get_by_id", return_value=save), \
         patch("core.data.HoldingsRepo.get_all", return_value=[]), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", PreviewAdvisor), \
         patch("backend.service.rebalance._get_prices_batch", return_value=({"000001": 10}, "realtime")), \
         patch("backend.service.rebalance._fetch_fundamental_quality", return_value={}), \
         patch("backend.service.rebalance._fetch_recent_amount", return_value={}), \
         patch("backend.service.rebalance._get_regime_stop_params", return_value={
             "stop_loss_warn": -0.04, "stop_loss_full": -0.08,
             "take_profit_half": 0.25, "take_profit_full": 0.5, "source": "test",
         }), \
         patch("backend.service.rebalance._load_half_sold", return_value=set()), \
         patch("core.data.RebalanceSignalRepo.batch_create") as create_signals:
        result = RebalanceService.run_rebalance(
            screening_save_id=1, cash=preview_cash, timing_ratio=1.0, panic=False,
            regime_info={}, enforce_strategy_profile=False,
            unmanaged_reason="管理员人工预览验证", unmanaged_actor="admin",
        )

    assert result["account"]["cash"] == preview_cash
    assert result["account"]["local_cash"] == (1000 if integrity["ok"] else None)
    assert result["account"]["manual_budget"] == 2500
    assert result["account"]["cash_source"] == "preview_budget"
    assert result["execution_blocked"] is True
    expected_reason = ("显式现金仅用于预览，当前方案不可执行" if integrity["ok"] else
                       "本地账本完整性校验失败，当前仅能预览")
    assert result["execution_block_reason"] == expected_reason
    create_signals.assert_not_called()


def test_legacy_rebalance_blocks_new_candidates_when_quality_gate_fails():
    from types import SimpleNamespace

    decision_date = datetime.date.today()
    account = {"status": "active", "version": 3, "cash": 1200,
               "frozen": 200, "available": 1000}
    save = {"candidates": [
                {"code": "000001", "name": "测试股"},
                {"code": "000002", "name": "ST测试股"},
            ],
            "pool": "all", "eval_date": decision_date.isoformat()}
    captured = {}

    class PreviewAdvisor:
        def __init__(self, **kwargs):
            self._allocator = SimpleNamespace(config=SimpleNamespace())

        def generate_advice_from_screening(self, **kwargs):
            captured["candidates"] = kwargs["screening_result"]["candidates"]
            return {"orders": [], "classification": {}, "constraints": [],
                    "turnover": 0, "observe_candidates": [], "top_n_codes": [], "skipped": []}

    with patch("core.shared.trading_time.get_latest_trading_day", return_value=decision_date), \
         patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True, "account_version": 3}), \
         patch("core.data.SettingsRepo.get", return_value="2500"), \
         patch("core.data.ScreeningSaveRepo.get_by_id", return_value=save), \
         patch("core.data.HoldingsRepo.get_all", return_value=[]), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", PreviewAdvisor), \
         patch("backend.service.rebalance._get_prices_batch", return_value=({"000001": 10}, "realtime")), \
         patch("backend.service.rebalance._fetch_fundamental_quality", side_effect=RuntimeError("quality unavailable")), \
         patch("backend.service.rebalance._get_regime_stop_params", return_value={
             "stop_loss_warn": -0.04, "stop_loss_full": -0.08,
             "take_profit_half": 0.25, "take_profit_full": 0.5, "source": "test",
         }), \
         patch("backend.service.rebalance._load_half_sold", return_value=set()):
        result = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            enforce_strategy_profile=False,
            unmanaged_reason="管理员人工预览验证", unmanaged_actor="admin",
            preview_only=True,
        )

    assert captured["candidates"] == []
    assert result["filtered_out"] == [{
        "code": "000002",
        "name": "ST测试股",
        "reason": "ST股票",
    }, {
        "code": "000001",
        "name": "测试股",
        "reason": "质量/流动性数据不可用，禁止新增风险",
    }]


def test_rebalance_total_assets_include_frozen_cash_but_plan_cash_is_available_cash():
    from types import SimpleNamespace

    decision_date = datetime.date.today()
    account = {"status": "active", "version": 3, "cash": 1200,
               "frozen": 200, "available": 1000}
    save = {"candidates": [{"code": "000001", "name": "测试股"}],
            "pool": "all", "eval_date": decision_date.isoformat()}

    class PreviewAdvisor:
        def __init__(self, **kwargs):
            self._allocator = SimpleNamespace(config=SimpleNamespace())

        def generate_advice_from_screening(self, **kwargs):
            assert kwargs["account_data"]["cash"] == 1000
            assert kwargs["account_data"]["total_cash"] == 1200
            assert kwargs["account_data"]["frozen_cash"] == 200
            assert kwargs["account_data"]["total_assets"] == 1200
            return {"orders": [], "classification": {}, "constraints": [],
                    "turnover": 0, "observe_candidates": [], "top_n_codes": [], "skipped": []}

    with patch("core.shared.trading_time.get_latest_trading_day", return_value=decision_date), \
         patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True, "account_version": 3}), \
         patch("core.data.SettingsRepo.get", return_value="2500"), \
         patch("core.data.ScreeningSaveRepo.get_by_id", return_value=save), \
         patch("core.data.HoldingsRepo.get_all", return_value=[]), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", PreviewAdvisor), \
         patch("backend.service.rebalance._get_prices_batch", return_value=({"000001": 10}, "realtime")), \
         patch("backend.service.rebalance._fetch_fundamental_quality", return_value={}), \
         patch("backend.service.rebalance._fetch_recent_amount", return_value={}), \
         patch("core.domain.screening.exclusion.prefilter_by_quality", return_value=(["000001"], [])), \
         patch("core.domain.screening.exclusion.prefilter_by_liquidity_capacity",
               return_value=(["000001"], [], {})), \
         patch("backend.service.rebalance._get_regime_stop_params", return_value={
             "stop_loss_warn": -0.04, "stop_loss_full": -0.08,
             "take_profit_half": 0.25, "take_profit_full": 0.5, "source": "test",
         }), \
         patch("backend.service.rebalance._load_half_sold", return_value=set()), \
         patch("core.data.RebalanceSignalRepo.batch_create", return_value=[]) as create_signals, \
         patch("core.data.RebalanceSessionRepo.create") as create_session:
        result = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            enforce_strategy_profile=False,
            unmanaged_reason="管理员人工预览验证", unmanaged_actor="admin",
            preview_only=True,
        )

    assert result["account"]["cash"] == 1000
    assert result["account"]["total_cash"] == 1200
    assert result["account"]["frozen_cash"] == 200
    assert result["account"]["total_assets"] == 1200
    assert result["account"]["local_cash"] == 1000
    assert result["execution_blocked"] is False
    assert result["preview_only"] is True
    assert result["committed"] is False
    assert len(result["plan_fingerprint"]) == 64
    create_signals.assert_not_called()
    create_session.assert_not_called()


def test_corrupt_active_ledger_uses_manual_budget_only_for_preview():
    from types import SimpleNamespace

    decision_date = datetime.date.today()
    account = {"status": "active", "version": 4, "cash": 1200,
               "frozen": 200, "available": 1000}
    save = {"candidates": [{"code": "000001", "name": "测试股"}],
            "pool": "all", "eval_date": decision_date.isoformat()}

    class PreviewAdvisor:
        def __init__(self, **kwargs):
            self._allocator = SimpleNamespace(config=SimpleNamespace())

        def generate_advice_from_screening(self, **kwargs):
            return {
                "orders": [{"code": "000001", "name": "测试股", "side": "buy",
                            "action": "买入", "amount": 1000, "urgency": "normal",
                            "reason": "预览计划", "current_weight": 0,
                            "target_weight": 0.1, "delta_weight": 0.1}],
                "classification": {}, "constraints": [], "turnover": 0,
                "observe_candidates": [], "top_n_codes": [], "skipped": [],
            }

    with patch("core.shared.trading_time.get_latest_trading_day", return_value=decision_date), \
         patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={
             "ok": False, "account_version": 4,
         }), patch("core.data.SettingsRepo.get", return_value="2500"), \
         patch("core.data.ScreeningSaveRepo.get_by_id", return_value=save), \
         patch("core.data.HoldingsRepo.get_all", return_value=[]), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", PreviewAdvisor), \
         patch("backend.service.rebalance._get_prices_batch", return_value=({"000001": 10}, "realtime")), \
         patch("backend.service.rebalance._fetch_fundamental_quality", return_value={}), \
         patch("backend.service.rebalance._fetch_recent_amount", return_value={}), \
         patch("core.domain.screening.exclusion.prefilter_by_liquidity_capacity",
               return_value=(["000001"], [], {})), \
         patch("backend.service.rebalance._get_regime_stop_params", return_value={
             "stop_loss_warn": -0.04, "stop_loss_full": -0.08,
             "take_profit_half": 0.25, "take_profit_full": 0.5, "source": "test",
         }), patch("backend.service.rebalance._load_half_sold", return_value=set()), \
         patch("core.data.RebalanceSignalRepo.batch_create") as create_signals, \
         patch("core.data.RebalanceSessionRepo.create") as create_session:
        result = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            enforce_strategy_profile=False,
            unmanaged_reason="管理员人工预览验证", unmanaged_actor="admin",
        )

    assert result["account"]["cash"] == 2500
    assert result["account"]["local_cash"] is None
    assert result["account"]["manual_budget"] == 2500
    assert result["account"]["cash_source"] == "preview_budget"
    assert result["execution_blocked"] is True
    assert result["execution_block_reason"] == "本地账本完整性校验失败，当前仅能预览"
    assert "signal_id" not in result["signals"][0]
    create_signals.assert_not_called()
    create_session.assert_not_called()


def test_unapproved_rebalance_returns_preview_without_persisting_actionable_signals():
    from types import SimpleNamespace

    decision_date = datetime.date.today()
    account = {"status": "uninitialized", "version": 0, "cash": None,
               "frozen": 0, "available": None}
    save = {"candidates": [{"code": "000001", "name": "测试股"}],
            "pool": "all", "eval_date": decision_date.isoformat()}

    class PreviewAdvisor:
        def __init__(self, **kwargs):
            self._allocator = SimpleNamespace(config=SimpleNamespace())

        def generate_advice_from_screening(self, **kwargs):
            return {
                "orders": [{"code": "000001", "name": "测试股", "side": "buy",
                            "action": "买入", "amount": 1000, "urgency": "normal",
                            "reason": "预览计划", "current_weight": 0,
                            "target_weight": 0.1, "delta_weight": 0.1}],
                "classification": {}, "constraints": [], "turnover": 0,
                "observe_candidates": [], "top_n_codes": [], "skipped": [],
            }

    with patch("core.shared.trading_time.get_latest_trading_day", return_value=decision_date), \
         patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch("core.data.SettingsRepo.get", return_value="2500"), \
         patch("core.data.ScreeningSaveRepo.get_by_id", return_value=save), \
         patch("core.data.HoldingsRepo.get_all", return_value=[]), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", PreviewAdvisor), \
         patch("backend.service.rebalance._get_prices_batch", return_value=({"000001": 10}, "realtime")), \
         patch("backend.service.rebalance._fetch_fundamental_quality", return_value={}), \
         patch("backend.service.rebalance._fetch_recent_amount", return_value={}), \
         patch("core.domain.screening.exclusion.prefilter_by_liquidity_capacity",
               return_value=(["000001"], [], {})), \
         patch("backend.service.rebalance._get_regime_stop_params", return_value={
             "stop_loss_warn": -0.04, "stop_loss_full": -0.08,
             "take_profit_half": 0.25, "take_profit_full": 0.5, "source": "test",
         }), \
         patch("backend.service.rebalance._load_half_sold", return_value=set()), \
         patch("core.data.RebalanceSignalRepo.batch_create") as create_signals, \
         patch("core.data.RebalanceSessionRepo.create") as create_session:
        result = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            enforce_strategy_profile=False,
            unmanaged_reason="管理员人工预览验证", unmanaged_actor="admin",
        )

    assert result["account"]["cash"] == 2500
    assert result["account"]["local_cash"] is None
    assert result["account"]["manual_budget"] == 2500
    assert result["account"]["cash_source"] == "preview_budget"
    assert result["execution_blocked"] is True
    assert result["execution_block_reason"] == "本地账户期初尚未核准，当前仅能预览"
    assert result["signals"][0]["side"] == "buy"
    assert result["signals"][0]["suggested_shares"] == 100
    assert "signal_id" not in result["signals"][0]
    create_signals.assert_not_called()
    create_session.assert_not_called()


def test_rebalance_plan_preview_is_read_only_and_commit_is_fingerprinted_and_idempotent():
    from types import SimpleNamespace

    from core.data.repos.local_ledger import LocalLedgerRepo
    from core.data.repos.session import RebalanceSessionRepo

    decision_date = datetime.date.today()
    account = {"status": "active", "version": 3, "cash": 1200,
               "frozen": 0, "available": 1200, "broker_account_id": "broker-1"}
    integrity = {"ok": True, "account_version": 3}
    save = {"candidates": [{"code": "000001", "name": "测试股"}],
            "pool": "all", "eval_date": decision_date.isoformat()}

    class PreviewAdvisor:
        def __init__(self, **kwargs):
            self._allocator = SimpleNamespace(config=SimpleNamespace())

        def generate_advice_from_screening(self, **kwargs):
            return {"orders": [{"code": "000001", "name": "测试股", "side": "buy",
                                "action": "买入", "amount": 1000, "urgency": "normal",
                                "reason": "计划预览", "current_weight": 0,
                                "target_weight": 0.1, "delta_weight": 0.1}],
                    "classification": {}, "constraints": [], "turnover": 0,
                    "observe_candidates": [], "top_n_codes": [], "skipped": []}

    with patch("core.shared.trading_time.get_latest_trading_day", return_value=decision_date), \
         patch.object(LocalLedgerRepo, "get_account", return_value=account), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value=integrity), \
         patch("backend.service.local_ledger.LocalLedgerService.reconcile", return_value={
             "can_execute": True, "broker_account_id": "broker-1", "local_version": 3,
         }) as reconcile, \
         patch("core.data.SettingsRepo.get", return_value=None), \
         patch("core.data.ScreeningSaveRepo.get_by_id", return_value=save), \
         patch("core.data.HoldingsRepo.get_all", return_value=[]), \
         patch("core.domain.portfolio.rebalance.WeeklyRebalanceAdvisor", PreviewAdvisor), \
         patch("backend.service.rebalance._get_prices_batch", return_value=({"000001": 10}, "realtime")), \
         patch("backend.service.rebalance._fetch_fundamental_quality", return_value={}), \
         patch("backend.service.rebalance._fetch_recent_amount", return_value={}), \
         patch("core.domain.screening.exclusion.prefilter_by_liquidity_capacity",
               return_value=(["000001"], [], {})), \
         patch("backend.service.rebalance._get_regime_stop_params", return_value={
             "stop_loss_warn": -0.04, "stop_loss_full": -0.08,
             "take_profit_half": 0.25, "take_profit_full": 0.5, "source": "test",
         }), \
         patch("backend.service.rebalance._load_half_sold", return_value=set()), \
         patch("core.data.RebalanceSignalRepo.batch_create", return_value=[42]) as create_signals, \
         patch.object(RebalanceSessionRepo, "get_by_commit_request", return_value=None), \
         patch.object(RebalanceSessionRepo, "ensure_commit_schema"), \
         patch("core.data.RebalanceSessionRepo.create") as create_empty_session:
        preview = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            enforce_strategy_profile=False, unmanaged_reason="管理员本次计划预览", unmanaged_actor="admin",
            preview_only=True,
        )
        create_signals.assert_not_called()
        create_empty_session.assert_not_called()
        assert preview["preview_only"] is True
        assert preview["committed"] is False
        assert preview["signals"][0]["suggested_shares"] == 100
        assert "signal_id" not in preview["signals"][0]
        reconcile.assert_not_called()
        assert len(preview["plan_fingerprint"]) == 64

        request_id = "00000000-0000-4000-8000-000000000123"
        committed = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            enforce_strategy_profile=False, unmanaged_reason="管理员本次计划预览", unmanaged_actor="admin",
            preview_only=False, commit_request_id=request_id,
            expected_plan_fingerprint=preview["plan_fingerprint"],
        )
        assert committed["committed"] is True
        assert committed["signals"][0]["signal_id"] == 42
        reconcile.assert_called_once()
        create_signals.assert_called_once()
        persisted_session = create_signals.call_args.kwargs["session_payload"]
        assert persisted_session["commit_request_id"] == request_id
        assert persisted_session["plan_fingerprint"] == preview["plan_fingerprint"]

        create_signals.reset_mock()
        previous_session = {"batch_id": request_id.replace("-", ""),
                            "plan_fingerprint": preview["plan_fingerprint"],
                            "eval_date": decision_date, "screening_save_id": 1,
                            "account_json": {}, "holdings_json": [], "constraints_json": []}
        with patch.object(RebalanceSessionRepo, "get_by_commit_request", return_value=previous_session), \
             patch("core.data.RebalanceSignalRepo.get_by_batch", return_value=[{"id": 42, "code": "000001", "side": "buy"}]):
            retried = RebalanceService.run_rebalance(
                screening_save_id=1, preview_only=False, commit_request_id=request_id,
                expected_plan_fingerprint=preview["plan_fingerprint"],
            )
        assert retried["already_committed"] is True
        assert retried["signals"][0]["id"] == 42
        create_signals.assert_not_called()

        create_signals.reset_mock()
        create_signals.side_effect = RuntimeError("duplicate commit request race")
        with patch.object(RebalanceSessionRepo, "get_by_commit_request",
                          side_effect=[None, previous_session]), \
             patch("core.data.RebalanceSignalRepo.get_by_batch",
                   return_value=[{"id": 42, "code": "000001", "side": "buy"}]):
            raced = RebalanceService.run_rebalance(
                screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
                preview_only=False, commit_request_id=request_id,
                expected_plan_fingerprint=preview["plan_fingerprint"],
                enforce_strategy_profile=False, unmanaged_reason="管理员本次计划预览", unmanaged_actor="admin",
            )
        assert raced.get("already_committed") is True, raced
        assert raced["signals"][0]["id"] == 42
        create_signals.assert_called_once()

        create_signals.reset_mock()
        create_signals.side_effect = ValueError("本地账户状态或版本已变化")
        rejected = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            preview_only=False, commit_request_id=request_id,
            expected_plan_fingerprint=preview["plan_fingerprint"],
            enforce_strategy_profile=False, unmanaged_reason="管理员本次计划预览", unmanaged_actor="admin",
        )
        assert rejected["commit_outcome"] == "not_committed"
        create_signals.assert_called_once()

        create_signals.reset_mock()
        reconcile.reset_mock()
        blocked_reconciliations = [
            {"can_execute": False, "broker_account_id": "broker-1", "local_version": 3,
             "reason": "券商持仓存在差异"},
            {"can_execute": True, "broker_account_id": "other-broker", "local_version": 3},
            {"can_execute": True, "broker_account_id": "broker-1", "local_version": 4},
            None,
        ]
        for index, snapshot in enumerate(blocked_reconciliations, start=124):
            reconcile.reset_mock()
            reconcile.return_value = snapshot
            blocked = RebalanceService.run_rebalance(
                screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
                preview_only=False,
                commit_request_id=f"00000000-0000-4000-8000-{index:012d}",
                expected_plan_fingerprint=preview["plan_fingerprint"],
                enforce_strategy_profile=False, unmanaged_reason="管理员本次计划预览", unmanaged_actor="admin",
            )
            assert "未保存实盘计划" in blocked["error"]
            assert blocked["commit_outcome"] == "not_committed"
            reconcile.assert_called_once()
            create_signals.assert_not_called()

        reconcile.reset_mock()
        reconcile.side_effect = RuntimeError("broker offline")
        unavailable = RebalanceService.run_rebalance(
            screening_save_id=1, timing_ratio=1.0, panic=False, regime_info={},
            preview_only=False, commit_request_id="00000000-0000-4000-8000-000000000125",
            expected_plan_fingerprint=preview["plan_fingerprint"],
            enforce_strategy_profile=False, unmanaged_reason="管理员本次计划预览", unmanaged_actor="admin",
        )
        assert unavailable["error"] == "券商对账不可用，未保存实盘计划"
        assert unavailable["commit_outcome"] == "not_committed"
        reconcile.assert_called_once()
        create_signals.assert_not_called()


@pytest.mark.parametrize("field", ["shares", "expected_account_version"])
def test_manual_fill_api_rejects_boolean_integer_fields_before_service(field):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    payload = {
        "shares": 100,
        "price": "10.00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    payload[field] = True
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "mark_executed") as record:
        response = TestClient(app).put("/signals/7/execute", json=payload)

    assert response.status_code == 422
    record.assert_not_called()


def test_manual_fill_api_keeps_flat_payload_for_valid_integer_fields():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    payload = {
        "shares": 100,
        "price": "10.00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "mark_executed", return_value={"execution_status": "partial"}) as record:
        response = TestClient(app).put("/signals/7/execute", json=payload)

    assert response.status_code == 200
    assert record.call_args.args == (7, {
        "shares": 100, "price": "10.00", "request_id": payload["request_id"],
        "expected_account_version": 1, "commission": "0.30", "stamp_duty": "0.00",
        "actor": "admin",
    })


def test_historical_fill_api_requires_admin_and_forwards_a_separate_ledger_event():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    payload = {
        "request_id": "00000000-0000-4000-8000-000000000321",
        "trade_date": "2026-10-29", "code": "000001", "side": "sell", "shares": 100,
        "price": "11.00", "expected_account_version": 3, "commission": "0.30",
        "stamp_duty": "0.55", "broker_trade_id": "broker-trade-321",
        "source_signal_id": 17, "reason": "依据券商交割单补记漏记成交",
    }
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(LocalLedgerService, "record_historical_fill", return_value={
             "already_recorded": False, "signal_updated": False}) as record:
        response = TestClient(app).post("/account/local/historical-fills", json=payload)

    assert response.status_code == 200
    forwarded = record.call_args.args[0]
    assert forwarded["actor"] == "admin"
    assert forwarded["request_id"] == payload["request_id"]
    assert forwarded["trade_date"] == payload["trade_date"]
    assert forwarded["source_signal_id"] == 17
    assert forwarded["allow_odd_lot_sell"] is False


def test_historical_fill_api_rejects_missing_evidence_before_service():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    payload = {
        "request_id": "00000000-0000-4000-8000-000000000321",
        "trade_date": "2026-10-29", "code": "000001", "side": "buy", "shares": 100,
        "price": "10.00", "expected_account_version": 3, "commission": "0.30",
        "stamp_duty": "0", "broker_trade_id": "",
        "reason": "依据券商交割单补记漏记成交",
    }
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(LocalLedgerService, "record_historical_fill") as record:
        response = TestClient(app).post("/account/local/historical-fills", json=payload)

    assert response.status_code == 422
    record.assert_not_called()


def test_rebalance_api_defaults_to_preview_and_requires_fingerprint_for_commit():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    preview_payload = {"screening_save_id": 1, "decision_id": "decision-1"}
    with patch.object(RebalanceService, "run_rebalance", return_value={"preview_only": True}) as run:
        preview = client.post("/run", json=preview_payload)
    assert preview.status_code == 200
    assert run.call_args.kwargs["preview_only"] is True
    assert run.call_args.kwargs["commit_request_id"] is None

    with patch.object(RebalanceService, "run_rebalance") as run:
        missing_hash = client.post("/run", json={**preview_payload, "preview_only": False,
                                                   "commit_request_id": "00000000-0000-4000-8000-000000000123"})
    assert missing_hash.status_code == 422
    run.assert_not_called()

    with patch.object(RebalanceService, "run_rebalance", return_value={"committed": True}) as run:
        commit = client.post("/run", json={**preview_payload, "preview_only": False,
            "commit_request_id": "00000000-0000-4000-8000-000000000123",
            "expected_plan_fingerprint": "a" * 64})
    assert commit.status_code == 200
    assert run.call_args.kwargs["preview_only"] is False
    assert run.call_args.kwargs["expected_plan_fingerprint"] == "a" * 64


@pytest.mark.parametrize("kwargs", [
    {},
    {"commit_request_id": "00000000-0000-4000-8000-000000000123"},
    {"expected_plan_fingerprint": "a" * 64},
])
def test_rebalance_service_rejects_commit_without_request_id_and_fingerprint(kwargs):
    import inspect

    signature = inspect.signature(RebalanceService.run_rebalance)
    assert signature.parameters["preview_only"].default is True
    with patch("core.data.repos.session.RebalanceSessionRepo.get_by_commit_request") as lookup:
        result = RebalanceService.run_rebalance(
            screening_save_id=1, preview_only=False, **kwargs)

    assert result == {"error": "调仓提交必须携带请求 ID 和预览方案指纹"}
    lookup.assert_not_called()


@pytest.mark.parametrize(("request_id", "fingerprint", "expected_error"), [
    ("not-a-uuid", "a" * 64, "提交请求 ID 必须是有效 UUID"),
    ("00000000-0000-4000-8000-000000000123", "z" * 64, "提交必须携带有效的预览方案指纹"),
])
def test_rebalance_service_rejects_invalid_commit_identity_before_database(request_id, fingerprint,
                                                                            expected_error):
    with patch("core.data.repos.session.RebalanceSessionRepo.get_by_commit_request") as lookup:
        result = RebalanceService.run_rebalance(
            screening_save_id=1, preview_only=False, commit_request_id=request_id,
            expected_plan_fingerprint=fingerprint)

    assert result == {"error": expected_error}
    lookup.assert_not_called()


def test_qmt_trade_sync_replaces_cached_row_with_authoritative_report():
    from types import SimpleNamespace
    from core.adapters.broker.qmt_broker import QmtBroker

    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_trades.return_value = [SimpleNamespace(
        traded_id="T-CACHE", order_id="O-CACHE", stock_code="000001.SZ",
        order_type=23, traded_price=10, traded_volume=100, traded_amount=1000,
        commission=1, traded_time="10:00:00",
    )]
    broker._trades = [{
        "id": "T-CACHE", "order_id": "O-CACHE", "symbol": "000001.SZ",
        "side": "buy", "price": 9.9, "quantity": 100, "amount": 990,
    }]

    with patch.object(broker, "_should_sync", return_value=True), \
         patch.object(broker, "_record_trade_report", return_value={"status": "duplicate"}), \
         patch.object(broker, "_persist_qmt_trade", return_value=True), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.list_unprojected", return_value=[]), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.unresolved_conflict_count", return_value=0):
        trades = broker.get_trades()

    assert len(trades) == 1
    assert trades[0]["price"] == 10
    assert trades[0]["amount"] == 1000


def test_qmt_unresolved_trade_conflict_blocks_stale_cached_rows():
    from core.adapters.broker.qmt_broker import QmtBroker

    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._stock_account = object()
    broker._trader.query_stock_trades.return_value = []
    broker._trades = [{"id": "T-STALE", "symbol": "000001.SZ", "price": 9.9}]

    with patch.object(broker, "_should_sync", return_value=True), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.list_unprojected", return_value=[]), \
         patch("core.data.repos.broker_trade_report.BrokerTradeReportRepo.unresolved_conflict_count", return_value=1):
        with pytest.raises(RuntimeError, match="查询或回报持久化不可用"):
            broker.get_trades()
    assert "未核对的矛盾成交回报" in broker._report_persistence_error


def test_qmt_empty_sync_exception_still_blocks_trade_query():
    from core.adapters.broker.qmt_broker import QmtBroker

    broker = QmtBroker("unused", "test")
    broker._connected = True
    broker._trader = Mock()
    broker._trader.query_stock_trades.side_effect = StopIteration()
    broker._trades = [{"id": "T-STALE", "symbol": "000001.SZ", "price": 9.9}]

    with patch.object(broker, "_should_sync", return_value=True):
        with pytest.raises(RuntimeError, match="查询或回报持久化不可用"):
            broker.get_trades()

    assert broker._report_persistence_error.startswith("StopIteration:")


def test_manual_fill_api_rejects_non_uuid_request_id_before_service():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    payload = {
        "shares": 100,
        "price": "10.00",
        "request_id": "manual-1",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "mark_executed") as record:
        response = TestClient(app).put("/signals/7/execute", json=payload)

    assert response.status_code == 422
    record.assert_not_called()


@pytest.mark.parametrize("field", ["signal_id", "shares", "expected_account_version"])
def test_batch_manual_fill_api_rejects_boolean_integer_fields_before_service(field):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    item = {
        "signal_id": 7,
        "shares": 100,
        "price": "10.00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    item[field] = True
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "batch_execute") as execute:
        response = TestClient(app).put("/signals/batch-execute", json={"signal_ids": [item]})

    assert response.status_code == 422
    execute.assert_not_called()


def test_batch_manual_fill_api_requires_ledger_idempotency_and_actual_fee_fields():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "batch_execute") as execute:
        response = TestClient(app).put("/signals/batch-execute", json={
            "signal_ids": [{"signal_id": 7, "shares": 100, "price": "10.00"}],
        })

    assert response.status_code == 422
    execute.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("signal_id", 0), ("signal_id", -1),
    ("signal_id", 2147483648),
    ("shares", 0), ("shares", -1),
    ("shares", 2147483648),
    ("expected_account_version", 0), ("expected_account_version", -1),
    ("expected_account_version", 2147483648),
])
def test_batch_manual_fill_api_rejects_non_positive_integer_fields_before_service(field, value):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    item = {
        "signal_id": 7,
        "shares": 100,
        "price": "10.00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    item[field] = value
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "batch_execute") as execute:
        response = TestClient(app).put("/signals/batch-execute", json={"signal_ids": [item]})

    assert response.status_code == 422
    execute.assert_not_called()


@pytest.mark.parametrize("field,value", [("request_id", "not-a-uuid"), ("unexpected", "value")])
def test_batch_manual_fill_api_rejects_invalid_idempotency_or_extra_fields(field, value):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    item = {
        "signal_id": 7,
        "shares": 100,
        "price": "10.00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    item[field] = value
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "batch_execute") as execute:
        response = TestClient(app).put("/signals/batch-execute", json={"signal_ids": [item]})

    assert response.status_code == 422
    execute.assert_not_called()


def test_batch_manual_fill_api_keeps_ledger_fields_and_authenticated_actor():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    item = {
        "signal_id": 7,
        "shares": 100,
        "price": "10.00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "batch_execute", return_value={"executed": 1}) as execute:
        response = TestClient(app).put("/signals/batch-execute", json={"signal_ids": [item]})

    assert response.status_code == 200
    submitted = execute.call_args.args[0][0]
    assert submitted == {**item, "actor": "admin"}


def test_batch_manual_fill_api_rejects_duplicate_request_ids_before_service():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    item = {
        "signal_id": 7,
        "shares": 100,
        "price": "10.00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "expected_account_version": 1,
        "commission": "0.30",
        "stamp_duty": "0.00",
    }
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(RebalanceService, "batch_execute") as execute:
        response = TestClient(app).put("/signals/batch-execute", json={
            "signal_ids": [item, {**item, "signal_id": 8}],
        })

    assert response.status_code == 422
    execute.assert_not_called()


@pytest.mark.parametrize("model,payload", [
    ("binding", {"expected_account_version": True}),
    ("intent", {"intent_id": "00000000-0000-4000-8000-000000000001", "signal_id": 7,
                 "expected_account_version": 1, "quantity": True,
                 "limit_price": "10.00", "fee_cap": "1.00"}),
    ("intent", {"intent_id": "00000000-0000-4000-8000-000000000001", "signal_id": True,
                 "expected_account_version": 1, "quantity": 100,
                 "limit_price": "10.00", "fee_cap": "1.00"}),
    ("close", {"expected_account_version": True, "reason": "核对后关闭剩余计划"}),
])
def test_local_ledger_write_models_reject_boolean_integer_fields(model, payload):
    from pydantic import ValidationError
    from backend.api.rebalance import BrokerBindingRequest, CloseRemainderRequest, OrderIntentRequest

    models = {"binding": BrokerBindingRequest, "intent": OrderIntentRequest,
              "close": CloseRemainderRequest}
    with pytest.raises(ValidationError):
        models[model].model_validate(payload)


@pytest.mark.parametrize("operation,forbidden_sql", [
    (lambda repo: repo.add("000001", "测试股", 100, "10.00"), "INSERT INTO holdings"),
    (lambda repo: repo.update("000001", reason="人工核对后的持仓数量更改",
                              expected_shares=100, expected_cost_price="10.00",
                              expected_buy_date="2026-10-28", expected_id=1, expected_version=1,
                              shares=200), "UPDATE holdings"),
    (lambda repo: repo.update("000001", reason="人工核对后的持仓成本更改",
                              expected_shares=100, expected_cost_price="10.00",
                              expected_buy_date="2026-10-28", expected_id=1, expected_version=1,
                              cost_price="11.00"), "UPDATE holdings"),
    (lambda repo: repo.update("000001", reason="人工核对后的买入日期更改",
                              expected_shares=100, expected_cost_price="10.00",
                              expected_buy_date="2026-10-28", expected_id=1, expected_version=1,
                              buy_date="2026-10-29"), "UPDATE holdings"),
    (lambda repo: repo.delete("000001", reason="人工核对后删除错误的本地持仓",
                              expected_shares=100, expected_cost_price="10.00",
                              expected_buy_date="2026-10-28", expected_id=1, expected_version=1), "DELETE FROM holdings"),
])
def test_legacy_holding_writes_cannot_change_active_ledger(operation, forbidden_sql):
    from core.data.repos.holdings import HoldingsRepo

    cursor = Mock()
    cursor.fetchone.return_value = {"status": "active"}
    with patch("core.data.repos.holdings.Database.transaction",
               side_effect=lambda callback: callback(cursor)):
        with pytest.raises(ValueError, match="本地账户已激活"):
            operation(HoldingsRepo)

    assert not any(forbidden_sql in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("method,expected_shares", [
    ("update", 100.5), ("update", True), ("delete", 100.5), ("delete", True),
])
def test_legacy_holding_mutations_reject_inexact_expected_share_snapshot(method, expected_shares):
    from core.data.repos.holdings import HoldingsRepo

    if method == "update":
        operation = lambda: HoldingsRepo.update(
            "000001", reason="核验后修正期初持仓数量", expected_shares=expected_shares,
            expected_cost_price="10.00", expected_buy_date="2026-10-28",
            expected_id=1, expected_version=1, shares=200,
        )
    else:
        operation = lambda: HoldingsRepo.delete(
            "000001", reason="核验后删除错误的期初持仓记录", expected_shares=expected_shares,
            expected_cost_price="10.00", expected_buy_date="2026-10-28", expected_version=1,
            expected_id=1,
        )
    with patch("core.data.repos.holdings.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="原持仓快照无效"):
            operation()

    transaction.assert_not_called()


@pytest.mark.parametrize("method,stored_shares", [
    ("update", 100.5), ("update", True), ("delete", 100.5), ("delete", True),
])
def test_legacy_holding_mutations_reject_corrupt_stored_share_snapshot(method, stored_shares):
    from decimal import Decimal
    from core.data.repos.holdings import HoldingsRepo

    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "uninitialized"},
        {"id": 1, "code": "000001", "shares": stored_shares, "cost_price": Decimal("10.00"),
         "buy_date": "2026-10-28", "name": "测试股"},
    ]
    if method == "update":
        operation = lambda: HoldingsRepo.update(
            "000001", reason="核验后修正期初持仓数量", expected_shares=100,
            expected_cost_price="10.00", expected_buy_date="2026-10-28",
            expected_id=1, expected_version=1, shares=200,
        )
        forbidden_sql = "UPDATE holdings"
    else:
        operation = lambda: HoldingsRepo.delete(
            "000001", reason="核验后删除错误的期初持仓记录", expected_shares=100,
            expected_cost_price="10.00", expected_buy_date="2026-10-28", expected_version=1,
            expected_id=1,
        )
        forbidden_sql = "DELETE FROM holdings"

    with patch("core.data.repos.holdings.Database.transaction",
               side_effect=lambda callback: callback(cursor)):
        with pytest.raises(ValueError, match="本地持仓股数无效"):
            operation()

    assert not any(forbidden_sql in call.args[0] for call in cursor.execute.call_args_list)


def test_daily_risk_half_signal_cannot_exceed_odd_lot_holding():
    signal = _check_risk_signal(
        "000001", "平安银行", 50, 10, 12, 600, -0.04, -0.08, 0.15, 0.30,
    )
    assert signal["side"] == "reduce"
    assert signal["planned_shares"] == signal["suggested_shares"] == 50
    assert signal["amount"] == 600


@pytest.mark.parametrize("account_id,scope,expected_params", [
    ("LOCAL-LIVE", "check_type=%s", ("daily_risk", "LOCAL-LIVE")),
])
def test_replacing_pending_signals_is_scoped_to_account(account_id, scope, expected_params):
    cursor = Mock()
    cursor.fetchone.side_effect = [{"status": "active", "version": 1}, None] if account_id else []
    selected = []

    def execute(sql, params=()):
        if "SELECT id,code,urgency,created_at" in sql:
            selected.append((sql, params))
            raise RuntimeError("stop after scoped selection")

    cursor.execute.side_effect = execute
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(RuntimeError, match="stop after scoped selection"):
            RebalanceSignalRepo.batch_create(
                "batch", None, "daily_risk",
                [{"code": "000001", "side": "sell", "planned_shares": 100}],
                replace_pending=True, input_fingerprint="fp", fingerprint_fn=lambda rows: "fp",
                account_id=account_id, account_version=1 if account_id else None,
            )
    assert len(selected) == 1
    assert scope in selected[0][0]
    assert "code IN" not in selected[0][0]
    if account_id:
        assert "account_version>0" in selected[0][0]
        assert "planned_shares>COALESCE(filled_shares,0)" in selected[0][0]
    assert selected[0][1] == expected_params


def test_empty_rebalance_batch_atomically_cancels_omitted_pending_signals():
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1}, None, None,
    ]
    cursor.fetchall.side_effect = [
        [{"id": 17, "code": "000002", "urgency": "normal",
          "created_at": datetime.datetime.now() - datetime.timedelta(days=1),
          "execution_status": "pending"}],
        [],
        [],
    ]
    session_payload = {"batch_id": "empty-batch", "check_type": "rebalance",
                       "screening_save_id": 8, "source_mode": "live"}
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.session.RebalanceSessionRepo.create") as create_session, \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        ids = RebalanceSignalRepo.batch_create(
            "empty-batch", 8, "rebalance", [], input_fingerprint="fp",
            replace_pending=True, fingerprint_fn=lambda rows: "fp",
            session_payload=session_payload, account_id="LOCAL-LIVE", account_version=1,
        )

    assert ids == []
    selected = next(call for call in cursor.execute.call_args_list
                    if "SELECT id,code,urgency,created_at" in call.args[0])
    assert "check_type=%s" in selected.args[0]
    assert "code IN" not in selected.args[0]
    assert selected.args[1] == ("rebalance", "LOCAL-LIVE")
    cancelled = next(call for call in cursor.execute.call_args_list
                     if "SET execution_status='cancelled'" in call.args[0])
    assert "unfilled_reason=%s" in cancelled.args[0]
    assert cancelled.args[1] == ("已由调仓批次 empty-batch 替换，原计划失效", 17)
    create_session.assert_called_once()


def test_empty_rebalance_batch_preserves_omitted_same_day_stop_loss():
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1}, None, None,
    ]
    cursor.fetchall.side_effect = [[{
        "id": 18, "code": "000003", "urgency": "stop_loss",
        "created_at": datetime.datetime.now(), "execution_status": "pending",
    }], [], []]
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.session.RebalanceSessionRepo.create") as create_session, \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        result = RebalanceSignalRepo.batch_create(
            "empty-batch", 8, "rebalance", [], input_fingerprint="fp",
            replace_pending=True, fingerprint_fn=lambda rows: "fp",
            session_payload={"batch_id": "empty-batch"},
            account_id="LOCAL-LIVE", account_version=1,
        )

    assert result == []
    assert not any("SET execution_status='cancelled'" in call.args[0]
                   for call in cursor.execute.call_args_list)
    create_session.assert_called_once()


def test_replacing_pending_signals_rejects_missing_account_before_database_access():
    with patch("core.data.repos.rebalance.Database.transaction") as transaction:
        with pytest.raises(ValueError, match="必须绑定本地账户"):
            RebalanceSignalRepo.batch_create(
                "batch", None, "daily_risk",
                [{"code": "000001", "side": "sell", "planned_shares": 100}],
                replace_pending=True, input_fingerprint="fp", fingerprint_fn=lambda rows: "fp",
                account_id=None, account_version=None,
            )

    transaction.assert_not_called()


def test_pending_list_excludes_legacy_and_other_account_signals():
    with patch("core.data.repos.rebalance.Database.fetchall", return_value=[]) as fetchall:
        RebalanceSignalRepo.get_pending(check_type="daily_risk", limit=20)
    sql, params = fetchall.call_args.args
    assert "account_id=%s" in sql
    assert "account_version>0" in sql
    assert "planned_shares>COALESCE(filled_shares,0)" in sql
    assert "execution_status='partial' OR" in sql
    assert "created_at >= CURDATE()" in sql
    assert "created_at DESC,id DESC" in sql
    assert params == ("LOCAL-LIVE", "daily_risk", 20, 0)


def test_pending_list_supports_stable_pagination():
    with patch("core.data.repos.rebalance.Database.fetchall", return_value=[]) as fetchall:
        RebalanceSignalRepo.get_pending(limit=200, offset=200)
    assert fetchall.call_args.args[1] == ("LOCAL-LIVE", 200, 200)


def test_old_partial_plan_blocks_replacement_until_explicit_close():
    cursor = Mock()
    cursor.fetchone.side_effect = [{"status": "active", "version": 2}, None]
    cursor.fetchall.return_value = [{
        "id": 7, "code": "000001", "urgency": "normal",
        "created_at": datetime.datetime.now() - datetime.timedelta(days=1),
        "execution_status": "partial",
    }]
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="部分成交的旧计划"):
            RebalanceSignalRepo.batch_create(
                "new-batch", None, "rebalance",
                [{"code": "000001", "side": "sell", "planned_shares": 100}],
                replace_pending=True, input_fingerprint="fp", fingerprint_fn=lambda rows: "fp",
                account_id="LOCAL-LIVE", account_version=2,
            )
    assert not any("INSERT INTO rebalance_signals" in call.args[0]
                   for call in cursor.execute.call_args_list)


def test_partial_remainder_close_records_zero_cash_event_and_preserves_fill():
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 900, "broker_account_id": None},
        None, None,
        {"account_id": "LOCAL-LIVE", "execution_status": "partial", "planned_shares": 100,
         "filled_shares": 40, "code": "000001", "side": "sell", "batch_id": "batch"},
    ]
    cursor.fetchall.return_value = []
    invalidated = [{"signal_id": 91, "code": "600000", "side": "buy",
                    "execution_status": "pending", "planned_shares": 100,
                    "filled_shares": 0, "invalidated_remaining_shares": 100}]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch.object(RebalanceSignalRepo, "invalidate_stale_account_signals", return_value=invalidated) as invalidate, \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        result = RebalanceSignalRepo.close_remainder(7, 2, "admin", "核对后关闭剩余计划")
    assert result == {"signal_id": 7, "closed_shares": 60, "account_version": 3,
                      "execution_status": "cancelled", "reason": "核对后关闭剩余计划",
                      "invalidated_signal_ids": [91], "invalidated_signal_details": invalidated,
                      "already_closed": False}
    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("SET execution_status='cancelled',unfilled_reason" in sql for sql in statements)
    assert not any("filled_shares=" in sql for sql in statements)
    assert any("'signal_close'" in sql and "cash_delta" in sql for sql in statements)
    invalidate.assert_called_once_with(cursor, "LOCAL-LIVE", 3)


def test_partial_remainder_close_duplicate_returns_prior_event_without_writes():
    payload = {"signal_id": 7, "closed_shares": 60, "account_version": 3,
               "execution_status": "cancelled", "reason": "核对后关闭剩余计划"}
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 3, "cash": 900},
        {"actor": "admin", "reason": "核对后关闭剩余计划", "payload_json": payload},
    ]
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        result = RebalanceSignalRepo.close_remainder(7, 2, "admin", "核对后关闭剩余计划")
    assert result["already_closed"] is True
    assert not any("UPDATE " in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("reconciliation,error", [
    (None, "券商对账未通过"),
    ({"can_execute": False}, "券商对账未通过"),
    ({"can_execute": True, "broker_account_id": "other", "local_version": 2}, "账户或版本已变化"),
])
def test_bound_account_remainder_close_requires_matching_reconciliation(reconciliation, error):
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 900, "frozen": 0, "broker_account_id": "broker-1"}, None,
    ]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match=error):
            RebalanceSignalRepo.close_remainder(7, 2, "admin", "核对后关闭剩余计划", reconciliation)
    assert not any("UPDATE " in call.args[0] for call in cursor.execute.call_args_list)


def test_bound_remainder_close_rejects_snapshot_aged_while_waiting_for_account_lock():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 900, "frozen": 0, "broker_account_id": "broker-1"}, None,
    ]
    reconciliation = _ready_reconciliation(
        broker_account_id="broker-1", broker_cash=900, broker_available=900,
        broker_as_of=(datetime.datetime.now() - datetime.timedelta(seconds=70)).isoformat(),
    )
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="快照已过期"):
            RebalanceSignalRepo.close_remainder(7, 2, "admin", "核对后关闭剩余计划", reconciliation)
    assert not any("UPDATE " in call.args[0] for call in cursor.execute.call_args_list)


def test_partial_remainder_close_rejects_unfinished_local_intent():
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 900, "broker_account_id": None},
        None, {"intent_id": "open-intent"},
    ]
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="未完成的本地委托意图"):
            RebalanceSignalRepo.close_remainder(7, 2, "admin", "核对后关闭剩余计划")
    assert not any("UPDATE " in call.args[0] for call in cursor.execute.call_args_list)


def test_partial_remainder_close_event_failure_rolls_back_transaction():
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 900, "broker_account_id": None},
        None, None,
        {"account_id": "LOCAL-LIVE", "execution_status": "partial", "planned_shares": 100,
         "filled_shares": 40, "code": "000001", "side": "sell", "batch_id": "batch"},
    ]
    cursor.fetchall.return_value = []

    def execute(sql, params=()):
        if "INSERT INTO local_ledger_events" in sql:
            raise RuntimeError("event insert failed")

    cursor.execute.side_effect = execute
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    with patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch.object(RebalanceSignalRepo, "invalidate_stale_account_signals", return_value=[]), \
         patch.object(Database, "_borrow", return_value=connection), \
         patch.object(Database, "_recycle"), \
         patch("core.data.connection._destroy_connection"):
        with pytest.raises(RuntimeError, match="event insert failed"):
            RebalanceSignalRepo.close_remainder(7, 2, "admin", "核对后关闭剩余计划")
    assert any("UPDATE local_ledger_accounts" in call.args[0] for call in cursor.execute.call_args_list)
    assert any("UPDATE rebalance_signals" in call.args[0] for call in cursor.execute.call_args_list)
    connection.rollback.assert_called_once()
    connection.commit.assert_not_called()


def test_bound_remainder_close_stops_when_broker_reconciliation_unavailable():
    with patch.object(LocalLedgerRepo, "get_account", return_value={"broker_account_id": "QMT-ONE"}), \
         patch.object(Database, "fetchone", return_value=None), \
         patch.object(LocalLedgerService, "reconcile", return_value={
             "can_execute": False, "reason": "券商数据不可用",
         }), \
         patch.object(RebalanceSignalRepo, "close_remainder") as close:
        with pytest.raises(ValueError, match="券商数据不可用"):
            LocalLedgerService.close_signal_remainder(7, 2, "admin", "核对后关闭剩余计划")
    close.assert_not_called()


def test_duplicate_remainder_close_does_not_need_new_broker_snapshot():
    with patch.object(LocalLedgerRepo, "get_account", return_value={"broker_account_id": "QMT-ONE"}), \
         patch.object(Database, "fetchone", return_value={"event_key": "prior"}), \
         patch.object(LocalLedgerService, "reconcile") as reconcile, \
         patch.object(RebalanceSignalRepo, "close_remainder", return_value={"already_closed": True}) as close:
        result = LocalLedgerService.close_signal_remainder(7, 2, "admin", "核对后关闭剩余计划")
    assert result["already_closed"] is True
    reconcile.assert_not_called()
    close.assert_called_once_with(7, 2, "admin", "核对后关闭剩余计划", None)


def test_quarantine_counts_signals_missing_plan_or_account_identity():
    with patch("core.data.repos.local_ledger.Database.fetchone", side_effect=[
        {"n": 1}, {"n": 1}, {"n": 3}, {"n": 2},
    ]) as fetchone, patch("core.data.repos.local_ledger.Database.fetchall",
                          side_effect=[[], [], [], [{"id": 9}]]) as fetchall:
        result = LocalLedgerRepo.get_quarantine(limit=10)
    assert result["counts"]["signals"] == 3
    assert result["counts"]["sessions"] == 2
    assert result["sessions"] == [{"id": 9}]
    count_sql = fetchone.call_args_list[2].args[0]
    list_sql = fetchall.call_args_list[2].args[0]
    session_count_sql = fetchone.call_args_list[3].args[0]
    session_list_sql = fetchall.call_args_list[3].args[0]
    assert MISSING_SIGNAL_PLAN in count_sql
    assert MISSING_SIGNAL_PLAN in list_sql
    assert "planned_shares" in list_sql
    assert "account_version" in session_count_sql
    assert "account_id" in session_list_sql


@pytest.mark.parametrize("side,quantity,error", [
    ("buy", 0, "缺少有效计划股数"),
    ("buy", 50, "100股整数倍"),
    ("reduce", None, "缺少有效计划股数"),
    ("other", 100, "方向无效"),
])
def test_live_plan_rejects_invalid_quantity_before_schema_or_transaction(side, quantity, error):
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema") as schema, \
         patch("core.data.repos.rebalance.Database.transaction") as transaction:
        with pytest.raises(ValueError, match=error):
            RebalanceSignalRepo.batch_create(
                "batch", None, "rebalance",
                [{"code": "000001", "side": side, "planned_shares": quantity}],
                replace_pending=True, input_fingerprint="fp", fingerprint_fn=lambda rows: "fp",
                account_id="LOCAL-LIVE", account_version=1,
            )
    schema.assert_not_called()
    transaction.assert_not_called()


def test_live_sell_plan_rejects_quantity_above_account_holding():
    cursor = Mock()
    cursor.fetchone.side_effect = [{"status": "active", "version": 1}, None]
    cursor.fetchall.side_effect = [[], [{"code": "000001", "shares": 50, "cost_price": 10}]]
    with patch("core.data.repos.strategy_profile.StrategyProfileRepo.ensure_schema"), \
         patch.object(LocalLedgerRepo, "audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda op: op(cursor)):
        with pytest.raises(ValueError, match="超过当前账户持仓"):
            RebalanceSignalRepo.batch_create(
                "batch", None, "rebalance",
                [{"code": "000001", "side": "reduce", "planned_shares": 100}],
                replace_pending=True, input_fingerprint="fp", fingerprint_fn=lambda rows: "fp",
                account_id="LOCAL-LIVE", account_version=1,
            )
    assert not any("INSERT INTO rebalance_signals" in call.args[0]
                   for call in cursor.execute.call_args_list)


@pytest.mark.parametrize("cash", [-1, float("nan"), float("inf"), float("-inf")])
def test_manual_rebalance_budget_rejects_non_finite_or_negative_values(cash):
    from backend.api import rebalance

    with patch("core.data.SettingsRepo.set") as save:
        with pytest.raises(HTTPException) as error:
            asyncio.run(rebalance.set_cash(cash))
    assert error.value.status_code == 422
    save.assert_not_called()


@pytest.mark.parametrize("stored", ["bad", "-1", "nan", "inf"])
def test_manual_rebalance_budget_read_rejects_invalid_stored_values(stored):
    from backend.api import rebalance

    with patch("core.data.SettingsRepo.get", return_value=stored):
        with pytest.raises(HTTPException) as error:
            asyncio.run(rebalance.get_cash())
    assert error.value.status_code == 503


def test_local_quarantine_api_requires_admin_and_returns_requested_page():
    from backend.api import rebalance

    page = {"counts": {"orders": 51, "trades": 0, "signals": 0}, "limit": 50,
            "offset": 50, "orders": [{"id": 51}], "trades": [], "signals": []}
    with patch("backend.api.auth.require_admin_actor", return_value="admin"), \
         patch.object(LocalLedgerRepo, "get_quarantine", return_value=page) as read:
        result = asyncio.run(rebalance.get_local_quarantine(50, 50, "Bearer admin-token"))

    assert result == {"data": page}
    read.assert_called_once_with(50, 50)


def test_local_quarantine_api_rejects_non_admin_before_reading():
    from backend.api import rebalance

    with patch("backend.api.auth.require_admin_actor",
               side_effect=HTTPException(403, "需要管理员权限")), \
         patch.object(LocalLedgerRepo, "get_quarantine") as read:
        with pytest.raises(HTTPException) as error:
            asyncio.run(rebalance.get_local_quarantine(50, 0, "Bearer user-token"))

    assert error.value.status_code == 403
    read.assert_not_called()
