import pytest
from fastapi import HTTPException
from unittest.mock import patch

from backend.api import auth, settings


def test_admin_auth_rejects_jwt_when_signing_secret_is_missing(monkeypatch):
    monkeypatch.setenv("LOGIN_USERNAME", "admin")
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiJ9.AA"

    with patch.object(auth, "_get_jwt_secret", return_value=""), \
         patch.object(settings, "_verify_admin", return_value=False):
        with pytest.raises(HTTPException) as error:
            auth.require_admin_actor(f"Bearer {token}")

    assert error.value.status_code == 403
