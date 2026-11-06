import pytest
from fastapi import HTTPException

from backend.api.factor import register_factor_endpoint


pytestmark = pytest.mark.unit


def test_manual_factor_registration_cannot_self_certify_pit_safety(monkeypatch):
    register = lambda **_kwargs: pytest.fail("self-certified factor must not be registered")
    monkeypatch.setattr("core.domain.factors.codegen.register_factor_safe", register)

    with pytest.raises(HTTPException) as exc:
        register_factor_endpoint("manual", "Manual", data_safety="safe")

    assert exc.value.status_code == 422


def test_manual_factor_registration_defaults_to_unknown_safety(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "core.domain.factors.codegen.register_factor_safe",
        lambda **kwargs: captured.update(kwargs) or True,
    )
    monkeypatch.setattr(
        "core.domain.factors.registry.get_pool",
        lambda: type("Pool", (), {"get_all_factors": lambda _self: []})(),
    )

    result = register_factor_endpoint("manual", "Manual", data_safety="unknown")

    assert captured["data_safety"] == "unknown"
    assert result["data"]["registered"] is True
