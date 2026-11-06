import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from core.shared.persistence import load_json


def _prices():
    dates = pd.bdate_range("2026-01-02", periods=22)
    return pd.DataFrame({
        "date": dates, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
        "volume": 100, "amount": 1000, "pct_chg": 0.0, "turnover": 0.0,
        "adj_factor": 1.0,
    })


def _stored(frame):
    return frame.rename(columns={"date": "trade_date"})[
        ["trade_date", "open", "high", "low", "close"]
    ].to_dict("records")


@pytest.mark.unit
@pytest.mark.p0
def test_direct_rebuild_is_closed():
    from scripts.rebuild_qfq_prices import rebuild_codes

    with pytest.raises(RuntimeError, match="直接覆盖正式行情已关闭"):
        rebuild_codes(["000001"], "20260102", "20260202")


@pytest.mark.unit
@pytest.mark.p0
def test_shadow_classifies_suspension_by_date(tmp_path, monkeypatch):
    from scripts import rebuild_qfq_prices as script

    full = _prices()
    suspended = full.loc[5, "date"]
    source = full.drop(index=5).reset_index(drop=True)
    stored = full.copy()
    stored.loc[5, ["open", "high", "low", "close"]] = None
    monkeypatch.setattr(script, "get_daily_kline", lambda *args, **kwargs: source.copy())
    monkeypatch.setattr(script.Database, "fetchall", lambda *args, **kwargs: _stored(stored))
    monkeypatch.setattr(script.Database, "fetchone", lambda *args, **kwargs: {"ts_code": "000001.SZ"})
    history = SimpleNamespace(error_code="0")
    history.next = lambda: not getattr(history, "done", False)
    def get_row_data():
        history.done = True
        return [str(suspended.date()), "10.0", "0"]
    history.get_row_data = get_row_data
    monkeypatch.setitem(sys.modules, "baostock", SimpleNamespace(
        login=lambda: SimpleNamespace(error_code="0"),
        query_history_k_data_plus=lambda *args, **kwargs: history,
        logout=lambda: None,
    ))

    output = tmp_path / "shadow"
    result = script.build_shadow_dataset(["000001"], "20260102", "20260202", output)

    assert result["rebuilt"] == 1
    assert load_json(output / "manifest.json")["codes"][0]["confirmed_suspension_count"] == 1
    assert script.validate_shadow_dataset(output)["valid"] is True


@pytest.mark.unit
@pytest.mark.p0
def test_shadow_quarantines_price_conflict(tmp_path, monkeypatch):
    from scripts import rebuild_qfq_prices as script

    frame = _prices()
    stored = frame.copy()
    stored.loc[0, "close"] = 12.0
    monkeypatch.setattr(script, "get_daily_kline", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(script.Database, "fetchall", lambda *args, **kwargs: _stored(stored))
    monkeypatch.setattr(script.Database, "fetchone", lambda *args, **kwargs: {})

    output = tmp_path / "shadow"
    result = script.build_shadow_dataset(["000001"], "20260102", "20260202", output)

    assert result["skipped"] == 1
    assert load_json(output / "manifest.json")["skipped_codes"][0]["price_conflict_dates"] == ["2026-01-02"]
    assert script.validate_shadow_dataset(output)["valid"] is False


@pytest.mark.unit
@pytest.mark.p0
def test_apply_only_fills_missing_fields(tmp_path, monkeypatch):
    from scripts import rebuild_qfq_prices as script

    frame = _prices()
    stored = frame.copy()
    stored.loc[0, "close"] = None
    monkeypatch.setattr(script, "get_daily_kline", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(script.Database, "fetchall", lambda *args, **kwargs: _stored(stored))
    monkeypatch.setattr(script.Database, "fetchone", lambda *args, **kwargs: {})
    writes = []
    monkeypatch.setattr(script.DailyQuotesRepo, "save_batch", lambda code, data: writes.append((code, data.copy(), dict(data.attrs["provenance"]))))
    monkeypatch.setattr(script, "version", lambda package: "test-version")

    output = tmp_path / "shadow"
    script.build_shadow_dataset(["000001"], "20260102", "20260202", output)
    assert script.validate_shadow_dataset(output)["valid"] is True

    result = script.apply_shadow_dataset(output)

    assert result["applied"] == 1
    assert writes[0][0] == "000001"
    assert len(writes[0][1]) == len(frame)
    assert writes[0][2]["anchor_factor"] == 1.0


@pytest.mark.unit
@pytest.mark.p0
def test_shadow_allows_unknown_optional_market_fields(tmp_path, monkeypatch):
    from scripts import rebuild_qfq_prices as script

    frame = _prices()
    frame["turnover"] = None
    monkeypatch.setattr(script, "get_daily_kline", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(script.Database, "fetchall", lambda *args, **kwargs: _stored(frame))
    monkeypatch.setattr(script.Database, "fetchone", lambda *args, **kwargs: {})

    output = tmp_path / "shadow"
    script.build_shadow_dataset(["000001"], "20260102", "20260202", output)

    assert script.validate_shadow_dataset(output) == {"valid": True, "checked": 1, "errors": []}


@pytest.mark.unit
@pytest.mark.p0
def test_shadow_rejects_partial_coverage_policy(tmp_path):
    from scripts.rebuild_qfq_prices import build_shadow_dataset

    with pytest.raises(ValueError, match="min_coverage 固定为 1.0"):
        build_shadow_dataset([], "20260102", "20260202", tmp_path / "shadow", min_coverage=0.98)
