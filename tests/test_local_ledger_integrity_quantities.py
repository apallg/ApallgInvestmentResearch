import json
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from core.data.repos.local_ledger import LocalLedgerRepo
from core.domain.portfolio.decision import decision_fingerprint


def _ledger_event(event_type, version, cash_delta, cash_after, payload, **fields):
    if event_type == "opening":
        payload = {**payload, "fingerprint": decision_fingerprint(payload["holdings"])}
    return {
        "event_type": event_type,
        "account_version": version,
        "cash_delta": Decimal(str(cash_delta)),
        "cash_after": Decimal(str(cash_after)),
        "payload_json": json.dumps(payload),
        "code": fields.get("code"),
        "side": fields.get("side"),
        "shares": fields.get("shares"),
        "price": Decimal(str(fields.get("price", 10))),
        "commission": Decimal(str(fields.get("commission", 0))),
        "stamp_duty": Decimal(str(fields.get("stamp_duty", 0))),
        "signal_id": fields.get("signal_id"),
        "actor": fields.get("actor", "admin"),
        "reason": fields.get("reason", "隔离账本测试"),
    }


def _opening_fingerprint(events):
    payload = events[0]["payload_json"]
    return json.loads(payload)["fingerprint"]


@pytest.mark.parametrize("bad_shares,actual_shares", [(100.5, 100), (True, 1), (None, 0)])
@pytest.mark.parametrize("event_type", ["opening", "manual_fill"])
def test_ledger_integrity_rejects_non_integral_event_shares(bad_shares, actual_shares, event_type):
    if event_type == "opening":
        ledger_events = [_ledger_event(
            "opening", 1, 1000, 1000,
            {"holdings": [{"code": "000001", "shares": bad_shares, "cost_price": 10}]},
        )]
        account_version = 1
        cash = 1000
        position_shares = actual_shares
        cost_price = Decimal("10")
    else:
        ledger_events = [
            _ledger_event("opening", 1, 1000, 1000,
                          {"holdings": [{"code": "000001", "shares": 1000, "cost_price": 10}]}),
            _ledger_event("manual_fill", 2, 500, 1500, {}, code="000001",
                          side="sell", shares=bad_shares),
        ]
        account_version = 2
        cash = 1500
        position_shares = 1000 - actual_shares
        cost_price = Decimal("10")

    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal(str(cash)), "frozen": Decimal("0"),
         "version": account_version, "broker_account_id": None,
         "opening_fingerprint": _opening_fingerprint(ledger_events)},
        {"events": account_version, "cash_total": Decimal(str(cash)),
         "last_version": account_version},
        {"reserved": Decimal("0")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [ledger_events, [], [
        {"code": "000001", "shares": position_shares, "cost_price": cost_price},
    ]]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is False
    assert any("持仓" in issue for issue in result["issues"])


@pytest.mark.parametrize("bad_shares", [100.5, True, None])
def test_ledger_integrity_rejects_non_integral_current_holding_shares(bad_shares):
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal("1000"), "frozen": Decimal("0"),
         "version": 1, "broker_account_id": None,
         "opening_fingerprint": _opening_fingerprint([_ledger_event(
             "opening", 1, 1000, 1000,
             {"holdings": [{"code": "000001", "shares": 100, "cost_price": 10}]},
         )])},
        {"events": 1, "cash_total": Decimal("1000"), "last_version": 1},
        {"reserved": Decimal("0")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [
        [_ledger_event("opening", 1, 1000, 1000,
                       {"holdings": [{"code": "000001", "shares": 100, "cost_price": 10}]})],
        [],
        [{"code": "000001", "shares": bad_shares, "cost_price": Decimal("10")}],
    ]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is False
    assert "本地持仓证券身份或股数无效" in result["issues"]


@pytest.mark.parametrize("shares", [100.5, True, None])
def test_opening_snapshot_rejects_non_integral_database_shares(shares):
    with patch("core.data.repos.local_ledger.Database.fetchall", return_value=[{
        "code": "000001", "shares": shares, "cost_price": Decimal("10"),
    }]):
        with pytest.raises(ValueError, match="账本股数缺失或无效"):
            LocalLedgerRepo.get_opening_snapshot()


@pytest.mark.parametrize("event_type", ["manual_fill", "broker_fill"])
def test_ledger_integrity_rejects_fill_cash_delta_inconsistent_with_price(event_type):
    intent_id = "intent-1"
    payload = {"amount": "999.00", "released_cash": "1000.30", "broker_trade_id": "T-1",
               "intent_id": intent_id, "quantity": 100} if event_type == "broker_fill" else {}
    events = [
        _ledger_event("opening", 1, 5000, 5000,
                      {"holdings": []}),
    ]
    intents = []
    if event_type == "broker_fill":
        intents = [{"intent_id": intent_id, "signal_id": 7, "code": "000001", "side": "buy",
                    "quantity": 100, "limit_price": Decimal("10"), "fee_cap": Decimal("0.30"),
                    "reserved_cash": Decimal("1000.30"), "remaining_reserved_cash": Decimal("0"),
                    "filled_quantity": 100, "state": "filled"}]
        events.append(_ledger_event("reserve", 2, 0, 5000,
                                    {"intent_id": intent_id, "reserved_cash": "1000.30",
                                     "quantity": 100, "limit_price": "10", "fee_cap": "0.30"},
                                    code="000001", side="buy", signal_id=7))
    version = len(events) + 1
    events.append(_ledger_event(event_type, version, -999.30, 4000.70, payload, code="000001",
                                side="buy", signal_id=7, shares=100, commission="0.30"))
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal("4000.70"), "frozen": Decimal("0"),
         "version": version, "broker_account_id": None,
         "opening_fingerprint": _opening_fingerprint(events)},
        {"events": version, "cash_total": Decimal("4000.70"), "last_version": version},
        {"reserved": Decimal("0")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [events, intents, [
        {"code": "000001", "shares": 100, "cost_price": Decimal("9.99")},
    ]]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is False
    assert "成交事件金额、费用或持仓数量无效" in result["issues"]


@pytest.mark.parametrize("event_type", ["manual_fill", "broker_fill"])
def test_ledger_integrity_accepts_fill_cash_delta_consistent_with_price(event_type):
    intent_id = "intent-1"
    payload = {"amount": "1000.00", "released_cash": "1000.30", "broker_trade_id": "T-1",
               "intent_id": intent_id, "quantity": 100} if event_type == "broker_fill" else {}
    events = [
        _ledger_event("opening", 1, 5000, 5000,
                      {"holdings": []}),
    ]
    intents = []
    if event_type == "broker_fill":
        intents = [{"intent_id": intent_id, "signal_id": 7, "code": "000001", "side": "buy",
                    "quantity": 100, "limit_price": Decimal("10"), "fee_cap": Decimal("0.30"),
                    "reserved_cash": Decimal("1000.30"), "remaining_reserved_cash": Decimal("0"),
                    "filled_quantity": 100, "state": "filled"}]
        events.append(_ledger_event("reserve", 2, 0, 5000,
                                    {"intent_id": intent_id, "reserved_cash": "1000.30",
                                     "quantity": 100, "limit_price": "10", "fee_cap": "0.30"},
                                    code="000001", side="buy", signal_id=7))
    version = len(events) + 1
    events.append(_ledger_event(event_type, version, -1000.30, 3999.70, payload, code="000001",
                                side="buy", signal_id=7, shares=100, commission="0.30"))
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal("3999.70"), "frozen": Decimal("0"),
         "version": version, "broker_account_id": None,
         "opening_fingerprint": _opening_fingerprint(events)},
        {"events": version, "cash_total": Decimal("3999.70"), "last_version": version},
        {"reserved": Decimal("0")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [events, intents, [
        {"code": "000001", "shares": 100, "cost_price": Decimal("10.00")},
    ]]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is True


def test_ledger_integrity_rejects_manual_fill_fees_above_gross_amount():
    events = [
        _ledger_event("opening", 1, 1000, 1000, {"holdings": []}),
        _ledger_event("manual_fill", 2, -3, 997, {}, code="000001", side="buy",
                      signal_id=7, shares=100, price="0.01", commission="2.00"),
    ]
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal("997"), "frozen": Decimal("0"),
         "version": 2, "broker_account_id": None,
         "opening_fingerprint": _opening_fingerprint(events)},
        {"events": 2, "cash_total": Decimal("997"), "last_version": 2},
        {"reserved": Decimal("0")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [events, [], [
        {"code": "000001", "shares": 100, "cost_price": Decimal("0.03")},
    ]]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is False
    assert "成交事件金额、费用或持仓数量无效" in result["issues"]


@pytest.mark.parametrize("event_type,payload", [
    ("reserve", {"reserved_cash": "0"}),
    ("reserve_release", {"released_cash": "0"}),
    ("broker_bind", {"broker_account_id": "QMT-TEST"}),
    ("broker_attach", {}),
    ("broker_terminal", {"released_cash": "0"}),
    ("signal_close", {"signal_id": 7, "closed_shares": 1, "account_version": 2,
                       "reason": "核对后关闭剩余计划"}),
])
def test_ledger_integrity_rejects_cash_delta_on_non_fill_event(event_type, payload):
    events = [
        _ledger_event("opening", 1, 1000, 1000, {"holdings": []}),
        _ledger_event(event_type, 2, 1, 1001, payload, signal_id=7, shares=1,
                      code="000001", side="buy", reason="核对后关闭剩余计划"),
    ]
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal("1001"), "frozen": Decimal("0"),
         "version": 2, "broker_account_id": None,
         "opening_fingerprint": _opening_fingerprint(events)},
        {"events": 2, "cash_total": Decimal("1001"), "last_version": 2},
        {"reserved": Decimal("0")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [events, [], []]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is False
    assert "非成交账本事件不得改变账户现金" in result["issues"]


def test_ledger_integrity_rejects_per_intent_reserve_drift_when_aggregate_matches():
    events = [_ledger_event("opening", 1, 5000, 5000, {"holdings": []})]
    intents = []
    for version, (intent_id, signal_id, event_reserved, current_reserved) in enumerate((
            ("intent-1", 1, 1000, 2000), ("intent-2", 2, 2000, 1000)), start=2):
        events.append(_ledger_event(
            "reserve", version, 0, 5000,
            {"intent_id": intent_id, "reserved_cash": str(event_reserved), "quantity": 100,
             "limit_price": "10", "fee_cap": "0"},
            code=f"00000{signal_id}", side="buy", signal_id=signal_id,
        ))
        intents.append({
            "intent_id": intent_id, "signal_id": signal_id, "code": f"00000{signal_id}", "side": "buy",
            "quantity": 100, "limit_price": Decimal("10"), "fee_cap": Decimal("0"),
            "reserved_cash": Decimal(str(event_reserved)), "remaining_reserved_cash": Decimal(str(current_reserved)),
            "filled_quantity": 0, "state": "prepared",
        })
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal("5000"), "frozen": Decimal("3000"),
         "version": 3, "broker_account_id": None,
         "opening_fingerprint": _opening_fingerprint(events)},
        {"events": 3, "cash_total": Decimal("5000"), "last_version": 3},
        {"reserved": Decimal("3000")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [events, intents, []]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is False
    assert "委托意图冻结金额或成交数量与事件回放不一致" in result["issues"]


@pytest.mark.parametrize("mismatch", ["account", "event"])
def test_ledger_integrity_rejects_opening_fingerprint_mismatch(mismatch):
    event = _ledger_event("opening", 1, 1000, 1000,
                          {"holdings": [{"code": "000001", "shares": 100,
                                         "cost_price": 10}]})
    expected_fingerprint = _opening_fingerprint([event])
    account_fingerprint = expected_fingerprint
    if mismatch == "account":
        account_fingerprint = "f" * 64
    else:
        payload = json.loads(event["payload_json"])
        payload["fingerprint"] = "f" * 64
        event["payload_json"] = json.dumps(payload)
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "cash": Decimal("1000"), "frozen": Decimal("0"),
         "version": 1, "broker_account_id": None,
         "opening_fingerprint": account_fingerprint},
        {"events": 1, "cash_total": Decimal("1000"), "last_version": 1},
        {"reserved": Decimal("0")},
        {"n": 0},
    ]
    cursor.fetchall.side_effect = [[event], [], [
        {"code": "000001", "shares": 100, "cost_price": Decimal("10")},
    ]]

    with patch("core.data.repos.local_ledger.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = LocalLedgerRepo.audit_integrity()

    assert result["ok"] is False
    assert "期初持仓事件无效" in result["issues"]
