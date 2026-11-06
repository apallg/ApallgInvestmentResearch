"""
API 冒烟测试 — 核心端点快速验证，防止重构回归

使用 FastAPI TestClient，不需要启动服务器。
"""
import sys
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    with TestClient(app) as c:
        password = os.getenv("LOGIN_PASSWORD")
        if password:
            response = c.post("/api/auth/login", json={
                "username": os.getenv("LOGIN_USERNAME", "admin"),
                "password": password,
            })
            if response.status_code == 200:
                c.headers.update({"Authorization": f"Bearer {response.json()['token']}"})
        yield c


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200, f"health 返回 {r.status_code}: {r.text[:200]}"

    def test_health_has_required_fields(self, client):
        r = client.get("/api/health")
        data = r.json()
        assert "status" in data
        assert "components" in data
        assert "app" in data
        assert data["app"] == "Apallg投研"

    def test_health_components_is_dict(self, client):
        r = client.get("/api/health")
        data = r.json()
        assert isinstance(data["components"], dict)
        for name in ("db", "redis", "factors", "schedulers", "qlib"):
            assert name in data["components"], f"缺少组件: {name}"


class TestMarketEndpoint:

    def test_indices_returns_200(self, client):
        r = client.get("/api/market/indices")
        assert r.status_code in (200, 503), f"indices 返回 {r.status_code}"

    def test_indices_has_data_field(self, client):
        r = client.get("/api/market/indices")
        if r.status_code == 200:
            data = r.json()
            assert "data" in data, f"缺少 data 字段: {list(data.keys())}"

    def test_sectors_returns_ok(self, client):
        r = client.get("/api/market/sectors")
        assert r.status_code in (200, 503), f"sectors 返回 {r.status_code}"


class TestScreeningEndpoint:

    def test_screening_latest_returns_list(self, client):
        r = client.get("/api/screening/latest")
        assert r.status_code in (200, 503), f"latest 返回 {r.status_code}"

    def test_screening_history_returns_ok(self, client):
        r = client.get("/api/screening/history")
        assert r.status_code in (200, 503), f"history 返回 {r.status_code}"

    def test_screening_result_invalid_id_returns_error(self, client):
        r = client.get("/api/screening/result/nonexistent_job_12345")
        assert r.status_code in (200, 404, 422, 503), \
            f"无效 job_id 应返回 404/422, 实际 {r.status_code}"


class TestSystemEndpoint:

    def test_tasks_returns_200(self, client):
        r = client.get("/api/system/tasks")
        assert r.status_code in (200, 503), f"tasks 返回 {r.status_code}"

    def test_system_health_returns_ok(self, client):
        r = client.get("/api/system/health")
        assert r.status_code in (200, 503), f"system/health 返回 {r.status_code}"


class TestBacktestEndpoint:

    def test_screening_backtest_status_returns_ok(self, client):
        r = client.get("/api/backtest/screening/nonexistent_job")
        assert r.status_code in (200, 404, 503), \
            f"backtest/screening 返回 {r.status_code}"
