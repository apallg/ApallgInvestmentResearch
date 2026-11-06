from types import ModuleType, SimpleNamespace

import pytest


def _broker_with_response(monkeypatch, response=None, error=None):
    from core.adapters.broker.qmt_broker import QmtBroker

    xtquant = ModuleType("xtquant")
    xtquant.xtconstant = SimpleNamespace(STOCK_BUY=23, STOCK_SELL=24, FIX_PRICE=11)
    monkeypatch.setitem(__import__("sys").modules, "xtquant", xtquant)
    broker = QmtBroker("unused", "account")
    broker._connected = True
    broker._stock_account = object()

    class Trader:
        calls = 0

        def order_stock(self, *args):
            self.calls += 1
            if error:
                raise error
            return response

    broker._trader = Trader()
    return broker


@pytest.mark.unit
@pytest.mark.p1
@pytest.mark.parametrize("response", [None, "123", True])
def test_qmt_invalid_order_id_is_reported_as_uncertain(monkeypatch, response):
    broker = _broker_with_response(monkeypatch, response=response)

    with pytest.raises(RuntimeError, match="结果不明确.*禁止直接重试"):
        broker.place_order("000001.SZ", "平安银行", "buy", 100, "limit", 10.0)

    assert broker._trader.calls == 1


@pytest.mark.unit
@pytest.mark.p1
def test_qmt_order_call_exception_is_reported_as_uncertain(monkeypatch):
    broker = _broker_with_response(monkeypatch, error=TimeoutError("response timeout"))

    with pytest.raises(RuntimeError, match="结果不明确.*禁止直接重试"):
        broker.place_order("000001.SZ", "平安银行", "buy", 100, "limit", 10.0)

    assert broker._trader.calls == 1
