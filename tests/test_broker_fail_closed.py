from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import trading
from core.shared import config


@pytest.mark.parametrize(
    "broker_type,account_updates,expected_error",
    [
        ("qmt", {"QMT_ACCOUNT": "QMT-ONE"}, "QMT 连接不可用"),
        ("easytrader", {
            "EASYTRAIDER_USER": "user-1",
            "EASYTRAIDER_ACCOUNT_ID": "ACCOUNT-ONE",
        }, "Easytrader 连接不可用"),
    ],
)
def test_real_broker_connection_failure_returns_unavailable_without_fake_fallback(
        monkeypatch, broker_type, account_updates, expected_error):
    monkeypatch.setattr(trading, "_broker", None)
    monkeypatch.setattr(config, "BROKER_TYPE", broker_type)
    for key, value in account_updates.items():
        monkeypatch.setattr(config, key, value)
    broker = Mock()
    broker.connect.return_value = False
    broker.connected = False
    app = FastAPI()
    app.include_router(trading.router)

    with patch("core.adapters.broker.create_broker", return_value=broker) as create_broker:
        response = TestClient(app).get("/account")

    assert response.status_code == 503
    assert response.json()["detail"] == expected_error
    assert create_broker.call_args.args[0] == broker_type
    broker.disconnect.assert_called_once()
    assert trading._broker is None


def test_cached_fake_instance_is_discarded_when_configuration_selects_real_broker(monkeypatch):
    monkeypatch.setattr(config, "BROKER_TYPE", config.BROKER_QMT)
    monkeypatch.setattr(config, "QMT_ACCOUNT", "QMT-ONE")
    stale_fake = Mock()
    stale_fake.connected = True
    monkeypatch.setattr(trading, "_broker", stale_fake)
    real_broker = Mock()
    real_broker.connect.return_value = False
    real_broker.connected = False

    with patch("core.adapters.broker.create_broker", return_value=real_broker) as create_broker:
        with pytest.raises(RuntimeError, match="QMT 连接不可用"):
            trading.get_broker()

    stale_fake.disconnect.assert_called_once()
    assert create_broker.call_args.args[0] == config.BROKER_QMT
    assert trading._broker is None


@pytest.mark.parametrize(
    "broker_type,account_updates,adapter_path,account_id,user",
    [
        ("qmt", {"QMT_ACCOUNT": "QMT-ONE"}, "core.adapters.broker.qmt_broker.QmtBroker", "QMT-QMT-ONE", None),
        ("easytrader", {
            "EASYTRAIDER_USER": "user-1",
            "EASYTRAIDER_ACCOUNT_ID": "ACCOUNT-ONE",
        }, "core.adapters.broker.easytrader_broker.EasytraderBroker", "EASYT-ACCOUNT-ONE", "user-1"),
    ],
)
def test_cached_real_broker_reconnect_failure_returns_unavailable_without_fallback(
        monkeypatch, broker_type, account_updates, adapter_path, account_id, user):
    from importlib import import_module

    monkeypatch.setattr(config, "BROKER_TYPE", broker_type)
    for key, value in account_updates.items():
        monkeypatch.setattr(config, key, value)
    adapter_module, adapter_name = adapter_path.rsplit(".", 1)
    adapter_type = getattr(import_module(adapter_module), adapter_name)
    cached_broker = adapter_type.__new__(adapter_type)
    cached_broker._account_id = account_id
    cached_broker._connected = False
    cached_broker.connect = Mock(return_value=False)
    cached_broker.disconnect = Mock()
    if user:
        cached_broker._user = user
    monkeypatch.setattr(trading, "_broker", cached_broker)
    app = FastAPI()
    app.include_router(trading.router)

    with patch("core.adapters.broker.create_broker") as create_broker:
        response = TestClient(app).get("/account")

    assert response.status_code == 503
    assert response.json()["detail"] == "真实券商连接不可用"
    cached_broker.connect.assert_called_once()
    cached_broker.disconnect.assert_called_once()
    create_broker.assert_not_called()
    assert trading._broker is None
