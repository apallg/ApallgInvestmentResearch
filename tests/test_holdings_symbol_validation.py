from unittest.mock import Mock

import pytest

from core.data.connection import Database
from core.data.repos.holdings import HoldingsRepo
from core.shared.order_validation import normalize_a_share_symbol


@pytest.mark.parametrize(
    "operation",
    [
        lambda: HoldingsRepo.add("999999", "invalid", 100, 1),
        lambda: HoldingsRepo.update("600000.SZ", risk_threshold=8),
        lambda: HoldingsRepo.delete(
            "300001.SH", "核对后删除本地持仓", 100, 10, "2026-10-30",
        ),
        lambda: HoldingsRepo.get_by_code("999999"),
        lambda: HoldingsRepo.get_adjustments("600000.SZ"),
    ],
)
def test_holdings_repository_rejects_invalid_a_share_identity_before_database(operation, monkeypatch):
    transaction = Mock(side_effect=AssertionError("database must not be reached"))
    fetchone = Mock(side_effect=AssertionError("database must not be reached"))
    fetchall = Mock(side_effect=AssertionError("database must not be reached"))
    monkeypatch.setattr(Database, "transaction", transaction)
    monkeypatch.setattr(Database, "fetchone", fetchone)
    monkeypatch.setattr(Database, "fetchall", fetchall)

    with pytest.raises(ValueError, match="股票代码或交易所无效"):
        operation()

    transaction.assert_not_called()
    fetchone.assert_not_called()
    fetchall.assert_not_called()


def test_holding_lookup_normalizes_exchange_suffix(monkeypatch):
    fetchone = Mock(return_value={"code": "000001"})
    monkeypatch.setattr(Database, "fetchone", fetchone)

    result = HoldingsRepo.get_by_code("000001.SZ")

    assert result == {"code": "000001"}
    assert fetchone.call_args.args[1] == ("000001", "LOCAL-LIVE")


@pytest.mark.parametrize("code", ["399001", "399001.SZ", "sz399001", "399006.SZ", "000688.SH"])
def test_market_index_identities_are_rejected_as_equity_symbols(code):
    with pytest.raises(ValueError):
        normalize_a_share_symbol(code)


def test_exchange_less_000001_remains_the_equity_symbol():
    assert normalize_a_share_symbol("000001") == "000001"
