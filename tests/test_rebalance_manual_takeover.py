import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException


@pytest.mark.unit
@pytest.mark.p0
def test_service_rejects_unaudited_manual_takeover():
    from backend.service.rebalance import RebalanceService

    with patch("core.shared.trading_time.get_latest_trading_day"):
        result = RebalanceService.run_rebalance(
            screening_save_id=1,
            enforce_strategy_profile=False,
            unmanaged_reason="",
            unmanaged_actor=None,
        )

    assert "必须由管理员发起" in result["error"]


@pytest.mark.p0
def test_api_forwards_manual_takeover_audit_fields(monkeypatch):
    from backend.api.rebalance import run_rebalance
    from backend.service.rebalance import RebalanceService

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("backend.api.auth.require_admin_actor", lambda value: "admin-user")
    monkeypatch.setattr(RebalanceService, "run_rebalance", fake_run)

    result = asyncio.run(
        run_rebalance(
            screening_save_id=1,
            decision_id="decision-1",
            profile_id=None,
            profile_version=None,
            profile_overrides=None,
            enforce_strategy_profile=False,
            unmanaged_reason="临时验证非Champion快照",
            authorization="Bearer token",
        )
    )

    assert result == {"data": {"ok": True}}
    assert captured["unmanaged_actor"] == "admin-user"
    assert captured["unmanaged_reason"] == "临时验证非Champion快照"


@pytest.mark.p0
def test_api_rejects_short_manual_takeover_reason(monkeypatch):
    from backend.api.rebalance import run_rebalance

    monkeypatch.setattr("backend.api.auth.require_admin_actor", lambda value: "admin-user")

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            run_rebalance(
                screening_save_id=1,
                decision_id="decision-1",
                profile_id=None,
                profile_version=None,
                profile_overrides=None,
                enforce_strategy_profile=False,
                unmanaged_reason="太短",
                authorization="Bearer token",
            )
        )

    assert error.value.status_code == 422
