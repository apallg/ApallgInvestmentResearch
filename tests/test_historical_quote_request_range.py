import pandas as pd
import pytest

from core.data.sources import historical_quotes


def _tushare_daily(dates):
    return pd.DataFrame({
        "ts_code": ["000001.SZ"] * len(dates),
        "trade_date": dates,
        "open": [10.0] * len(dates),
        "high": [11.0] * len(dates),
        "low": [9.0] * len(dates),
        "close": [10.5] * len(dates),
        "pre_close": [10.0] * len(dates),
        "change": [0.5] * len(dates),
        "pct_chg": [5.0] * len(dates),
        "vol": [100.0] * len(dates),
        "amount": [1000.0] * len(dates),
    })


def _tushare_factors(dates):
    return pd.DataFrame({
        "ts_code": ["000001.SZ"] * len(dates),
        "trade_date": dates,
        "adj_factor": [1.0] * len(dates),
    })


def _install_tushare(monkeypatch, daily, factors):
    class Pro:
        def daily(self, **kwargs):
            return daily

        def adj_factor(self, **kwargs):
            return factors

    monkeypatch.setattr(historical_quotes, "get_pro", lambda: Pro())
    monkeypatch.setattr(historical_quotes, "_rate_limit", lambda: None)


def test_tushare_rejects_daily_row_outside_requested_range(monkeypatch):
    _install_tushare(
        monkeypatch,
        _tushare_daily(["20261001", "20261003"]),
        _tushare_factors(["20261001", "20261003"]),
    )

    with pytest.raises(RuntimeError, match="TuShare pro.daily 返回请求区间外"):
        historical_quotes.fetch_tushare_raw_daily("000001.SZ", "20261001", "20261002")


def test_tushare_rejects_factor_row_outside_range_even_when_daily_is_empty(monkeypatch):
    _install_tushare(
        monkeypatch,
        pd.DataFrame(columns=_tushare_daily([]).columns),
        _tushare_factors(["20261003"]),
    )

    with pytest.raises(RuntimeError, match="TuShare pro.adj_factor 返回请求区间外"):
        historical_quotes.fetch_tushare_raw_daily(
            "000001.SZ", "20261001", "20261002", allow_empty_daily=True,
        )


def test_tushare_accepts_rows_at_inclusive_range_boundaries(monkeypatch):
    _install_tushare(
        monkeypatch,
        _tushare_daily(["20261001", "20261002"]),
        _tushare_factors(["20261001", "20261002"]),
    )

    result = historical_quotes.fetch_tushare_raw_daily("000001.SZ", "20261001", "20261002")

    assert result["normalized"]["trade_date"].dt.strftime("%Y%m%d").tolist() == ["20261001", "20261002"]


def test_tushare_anchor_factor_rejects_future_source_row(monkeypatch):
    class Pro:
        def adj_factor(self, **kwargs):
            return _tushare_factors(["20261003"])

    monkeypatch.setattr(historical_quotes, "get_pro", lambda: Pro())
    monkeypatch.setattr(historical_quotes, "_rate_limit", lambda: None)

    with pytest.raises(RuntimeError, match="TuShare复权锚点因子 返回请求区间外"):
        historical_quotes.fetch_tushare_anchor_factor("000001.SZ", "2026-10-02")


@pytest.mark.parametrize("dates", [["2026-10-01", "2026-10-03"], ["2026-09-30"]])
def test_baostock_rejects_rows_outside_requested_range(monkeypatch, dates):
    raw = pd.DataFrame({
        "date": dates,
        "open": [10.0] * len(dates),
        "high": [11.0] * len(dates),
        "low": [9.0] * len(dates),
        "close": [10.5] * len(dates),
        "volume": [100.0] * len(dates),
        "amount": [1000.0] * len(dates),
        "turn": [1.0] * len(dates),
        "tradestatus": [1] * len(dates),
        "pctChg": [5.0] * len(dates),
    })
    monkeypatch.setattr(historical_quotes.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})())
    monkeypatch.setattr(pd, "read_parquet", lambda path: raw)
    monkeypatch.setattr(historical_quotes.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="BaoStock 3 次隔离采集均失败: BaoStock 返回请求区间外"):
        historical_quotes.fetch_baostock_raw_daily("000001", "sz", "2026-10-01", "2026-10-02")


def test_baostock_accepts_rows_at_inclusive_range_boundaries(monkeypatch):
    raw = pd.DataFrame({
        "date": ["2026-10-01", "2026-10-02"],
        "open": [10.0, 10.0],
        "high": [11.0, 11.0],
        "low": [9.0, 9.0],
        "close": [10.5, 10.5],
        "volume": [100.0, 100.0],
        "amount": [1000.0, 1000.0],
        "turn": [1.0, 1.0],
        "tradestatus": [1, 1],
        "pctChg": [5.0, 5.0],
    })
    monkeypatch.setattr(historical_quotes.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})())
    monkeypatch.setattr(pd, "read_parquet", lambda path: raw)

    result = historical_quotes.fetch_baostock_raw_daily("000001", "sz", "20261001", "20261002")

    assert result["normalized"]["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-01", "2026-10-02"]
