from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.stock import router


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_stock_search_uses_current_stock_identity_and_preserves_namespace():
    rows = [{"ts_code": "600519.SH", "symbol": "600519", "name": "贵州茅台", "exchange": "SSE"}]
    with patch("core.data.repos.stock_identity.StockIdentityRepo.search_current", return_value=rows) as search:
        response = _client().get("/search", params={"q": "贵州茅台"})

    assert response.status_code == 200
    assert response.json()["data"] == [{
        "code": "600519",
        "name": "贵州茅台",
        "type": "A股",
        "asset_type": "stock",
        "exchange": "SH",
        "instrument_id": "stock:SH:600519",
    }]
    search.assert_called_once_with("贵州茅台", 10)


def test_unknown_stock_search_does_not_echo_query_as_a_valid_instrument():
    with patch("core.data.repos.stock_identity.StockIdentityRepo.search_current", return_value=[]), \
         patch("backend.api.stock.fetcher.get_stock_name", side_effect=AssertionError("unverified fallback")) as fallback:
        response = _client().get("/search", params={"q": "not-a-stock"})

    assert response.status_code == 200
    assert response.json()["data"] == []
    fallback.assert_not_called()


def test_index_and_stock_with_same_numeric_code_keep_distinct_ids():
    rows = [{"ts_code": "000001.SZ", "symbol": "000001", "name": "平安银行", "exchange": "SZSE"}]
    with patch("core.data.repos.stock_identity.StockIdentityRepo.search_current", return_value=rows):
        response = _client().get("/search", params={"q": "000001"})

    assert response.status_code == 200
    results = response.json()["data"]
    assert {item["instrument_id"] for item in results} == {"stock:SZ:000001", "index:SH:000001"}
    assert {item["asset_type"] for item in results} == {"stock", "index"}


def test_market_index_name_search_returns_explicit_index_identity():
    with patch("core.data.repos.stock_identity.StockIdentityRepo.search_current", return_value=[]):
        response = _client().get("/search", params={"q": "上证指数"})

    assert response.status_code == 200
    assert response.json()["data"] == [{
        "code": "sh000001",
        "name": "上证指数",
        "type": "指数",
        "asset_type": "index",
        "exchange": "SH",
        "instrument_id": "index:SH:000001",
    }]


def test_stock_detail_requires_a_current_registered_stock_identity():
    rows = [{"ts_code": "600519.SH", "symbol": "600519", "name": "贵州茅台", "exchange": "SSE"}]
    with patch("core.data.repos.stock_identity.StockIdentityRepo.get_current_identity", return_value=rows[0]), \
         patch("backend.api.stock.fetcher.get_stock_name", return_value="贵州茅台"):
        response = _client().get("/600519.SH")

    assert response.status_code == 200
    assert response.json() == {"code": "600519", "name": "贵州茅台"}


def test_stock_detail_rejects_index_alias_before_loading_stock_data():
    with patch("core.data.repos.stock_identity.StockIdentityRepo.get_current_identity", side_effect=AssertionError("index must be rejected before DB lookup")):
        response = _client().get("/000001.SH")

    assert response.status_code == 404
    assert "当前已登记的A股证券" in response.json()["detail"]


def test_stock_identity_resolution_rejects_unknown_or_delisted_rows():
    from backend.service.instrument_lookup import resolve_current_stock_code

    with patch("core.data.repos.stock_identity.StockIdentityRepo.get_current_identity", return_value=None) as lookup:
        assert resolve_current_stock_code("999999") is None
        assert resolve_current_stock_code("index:SH:000001") is None

    lookup.assert_called_once_with("999999", None)


def test_ths_stock_routes_share_the_current_identity_gate():
    from backend.api.ths import router as ths_router

    app = FastAPI()
    app.include_router(ths_router)
    with patch("core.data.repos.stock_identity.StockIdentityRepo.get_current_identity", return_value=None), \
         patch("backend.api.ths.Database.fetchall", side_effect=AssertionError("identity gate must run before data query")):
        response = TestClient(app).get("/api/ths/stock/999999/daily")

    assert response.status_code == 404
    assert "当前已登记的A股证券" in response.json()["detail"]


def test_current_identity_repository_uses_active_listing_and_exact_exchange():
    from core.data.connection import Database
    from core.data.repos.stock_identity import StockIdentityRepo

    row = {"ts_code": "000001.SZ", "symbol": "000001", "name": "平安银行", "exchange": "SZSE"}
    with patch.object(Database, "fetchall", return_value=[row]) as fetch:
        result = StockIdentityRepo.get_current_identity("000001", "sz")

    assert result == row
    assert "list_status='L'" in fetch.call_args.args[0]
    assert "UPPER(ts_code)=%s" in fetch.call_args.args[0]
    assert fetch.call_args.args[1] == ("000001", "000001.SZ")
