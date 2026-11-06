import asyncio
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.api.portfolio import HoldingCreate, add_holding, delete_holding, update_holding


@pytest.mark.parametrize(
    "payload",
    [
        {"shares": True, "cost_price": 10},
        {"shares": 100, "cost_price": True},
        {"shares": 100.0, "cost_price": 10},
        {"shares": 100.5, "cost_price": 10},
        {"shares": 100, "cost_price": 10, "risk_threshold": True},
        {"shares": 100, "cost_price": 10, "alerts_enabled": 1},
    ],
)
def test_holding_create_rejects_boolean_or_non_integer_share_inputs(payload):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HoldingCreate(code="000001", name="test", **payload)


def test_holding_create_accepts_decimal_string_cost_from_form():
    item = HoldingCreate(code="000001", name="test", shares=100, cost_price="10.50")

    assert item.shares == 100
    assert item.cost_price == 10.5
    assert str(item.cost_price) == "10.50"


def test_holding_create_rejects_float_cost_price_to_preserve_decimal_precision():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="十进制字符串"):
        HoldingCreate(code="000001", name="test", shares=100, cost_price=10.5)


def test_holding_create_preserves_overprecision_until_repository_validation():
    from core.data.repos.holdings import HoldingsRepo

    item = HoldingCreate(code="000001", name="test", shares=100, cost_price="10.0000000000000001")
    with pytest.raises(ValueError, match="两位小数"):
        HoldingsRepo.add(item.code, item.name, item.shares, item.cost_price)


@pytest.mark.parametrize(
    "payload",
    [
        {"code": "000001", "name": "test", "shares": True, "cost_price": 10},
        {"code": "000001", "name": "test", "shares": 100, "cost_price": True},
        {"code": "000001", "name": "test", "shares": 100, "cost_price": 10.5},
        {"code": "000001", "name": "test", "shares": 100, "cost_price": 10, "risk_threshold": True},
        {"code": "000001", "name": "test", "shares": 100, "cost_price": 10, "alerts_enabled": 1},
    ],
)
def test_holding_create_route_rejects_boolean_numbers_before_handler(payload):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.portfolio import router

    app = FastAPI()
    app.include_router(router)

    response = TestClient(app).post("/holdings", json=payload)

    assert response.status_code == 422


def test_duplicate_holding_create_maps_database_conflict_to_409(monkeypatch):
    import pymysql
    from backend.api import auth
    from core.data import Database, HoldingsRepo

    monkeypatch.setattr(auth, "require_admin_actor", lambda authorization: "admin")
    monkeypatch.setattr(Database, "is_available", lambda: True)
    monkeypatch.setattr(HoldingsRepo, "add", Mock(side_effect=pymysql.err.IntegrityError(1062, "duplicate")))
    holding = HoldingCreate(code="000001", name="test", shares=100, cost_price=10)

    with pytest.raises(HTTPException) as error:
        asyncio.run(add_holding(holding, authorization="admin"))

    assert error.value.status_code == 409
    assert error.value.detail == "该股票已存在，请使用编辑操作"


def test_update_of_missing_holding_maps_to_404(monkeypatch):
    from backend.api import auth
    from core.data import HoldingsRepo

    monkeypatch.setattr(auth, "require_admin_actor", lambda authorization: "admin")
    monkeypatch.setattr(HoldingsRepo, "update", Mock(return_value=0))

    with pytest.raises(HTTPException) as error:
        asyncio.run(update_holding("000001", {"risk_threshold": 8, "expected_id": 1, "expected_version": 1},
                                   authorization="admin"))

    assert error.value.status_code == 404


def test_update_rejects_float_cost_price_before_repository(monkeypatch):
    from backend.api import auth
    from core.data import HoldingsRepo

    monkeypatch.setattr(auth, "require_admin_actor", lambda authorization: "admin")
    update = Mock()
    monkeypatch.setattr(HoldingsRepo, "update", update)

    with pytest.raises(HTTPException) as error:
        asyncio.run(update_holding(
            "000001", {"cost_price": 10.5, "expected_id": 1, "expected_version": 1},
            authorization="admin",
        ))

    assert error.value.status_code == 422
    update.assert_not_called()


@pytest.mark.parametrize(
    "handler,args",
    [
        (add_holding, (HoldingCreate(code="000001", name="test", shares=100, cost_price=10),)),
        (update_holding, ("000001", {"risk_threshold": 8})),
        (delete_holding, ("000001", {"reason": "核对后删除本地记录", "expected_shares": 100,
                                     "expected_cost_price": 10, "expected_buy_date": "2026-10-30"})),
    ],
)
def test_holding_mutations_require_admin_before_repository_access(monkeypatch, handler, args):
    from backend.api import auth, settings

    monkeypatch.setattr(settings, "_verify_admin", lambda authorization: False)
    monkeypatch.setattr(auth, "decode_token", lambda token: None)

    with pytest.raises(HTTPException) as error:
        asyncio.run(handler(*args, authorization=None))

    assert error.value.status_code == 403
