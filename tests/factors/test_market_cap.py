import numpy as np

from core.data import Database
from core.domain.factors.market_cap import get_market_cap_map


def test_market_cap_accepts_numpy_codes(monkeypatch):
    monkeypatch.setattr(Database, "is_available", lambda: False)

    result = get_market_cap_map(np.array(["000001", "600519"]), "2024-01-02")

    assert result == {}
