import datetime

import pandas as pd
import pytest

from core.domain.screening._screener import BuySideScreener
from core.domain.screening.scoring import rank_l1


@pytest.fixture(autouse=True)
def listed_stocks(monkeypatch):
    from core.data import StockIdentityRepo

    monkeypatch.setattr(
        StockIdentityRepo, "get_listing_dates",
        classmethod(lambda cls, codes: {code: datetime.date(2020, 1, 1) for code in codes}),
    )


class _Fetcher:
    def get_industry_map(self):
        return {
            "000001": {"name": "甲", "industry": "银行"},
            "000002": {"name": "乙", "industry": "房地产"},
            "000003": {"name": "丙", "industry": "电子"},
        }


def _snapshot():
    dates = pd.date_range(end="2026-08-01", periods=90, freq="D")
    result = {}
    for index, code in enumerate(("000001", "000002", "000003")):
        close = 10.0 + index
        df = pd.DataFrame({
            "trade_date": dates,
            "close": [close] * len(dates),
            "volume": [1_000_000.0] * len(dates),
            "amount": [50_000_000.0] * len(dates),
            "pct_chg": [0.0] * len(dates),
        })
        result[code] = (df, _Fetcher().get_industry_map()[code]["name"], close, 1_000_000.0)
    return result


@pytest.mark.p0
def test_qlib_primary_order_is_not_changed_by_auxiliary_scores(monkeypatch):
    from core.data import Database
    from core.data.sources import etf_flow

    monkeypatch.setattr(Database, "fetchall", lambda *args, **kwargs: [])
    monkeypatch.setattr(etf_flow, "get_etf_flow_by_industry", lambda **kwargs: {})
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())
    scores = {"000001": 0.2, "000002": 0.9, "000003": 0.5}

    result = screener._select_qlib_top_candidates(
        list(scores), scores, 3,
        eval_date=datetime.date(2026, 8, 1), snapshot=_snapshot(),
    )
    result["candidates"][0]["fundamental_score"] = -1.0
    result["candidates"][2]["fundamental_score"] = 1.0
    ranked = rank_l1(result["candidates"])

    assert [row["code"] for row in ranked] == ["000002", "000003", "000001"]
    assert result["weights"] == {
        "qlib": 1.0,
        "fundamental": 0.0,
        "flow": 0.0,
        "north_flow": 0.0,
        "sentiment": 0.0,
        "etf_flow": 0.0,
    }
    assert result["auxiliary_coverage"]["industry"] == {
        "covered": 3, "total": 3, "rate": 1.0,
    }
    for source in ("fundamental", "flow", "north_flow", "sentiment", "etf_flow"):
        assert result["auxiliary_coverage"][source] == {
            "covered": 0, "total": 3, "rate": 0.0,
        }


@pytest.mark.p0
def test_qlib_primary_rejects_empty_predictions():
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())

    with pytest.raises(RuntimeError, match="辅助数据不会替代排序"):
        screener._select_qlib_top_candidates(
            ["000001"], {}, 1,
            eval_date=datetime.date(2026, 8, 1), snapshot=_snapshot(),
        )


@pytest.mark.p0
def test_screen_pipeline_uses_qlib_primary_contract(monkeypatch):
    from core.data import Database
    from core.data.sources import etf_flow

    monkeypatch.setattr(Database, "fetchall", lambda *args, **kwargs: [])
    monkeypatch.setattr(etf_flow, "get_etf_flow_by_industry", lambda **kwargs: {})
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())
    scores = {"000001": 0.2, "000002": 0.9, "000003": 0.5}
    monkeypatch.setattr(screener, "_load_qlib_predictions", lambda *args, **kwargs: scores)

    result = screener.screen(
        pool="all", top_n=3, qlib_model=("experiment", "run"),
        eval_date=datetime.date(2026, 8, 1), snapshot=_snapshot(),
        source_weights={"fundamental": 1.0},
        skip_l2=True, skip_debate=True, skip_backfill=True,
        sector_fusion=False, ranking_mode="qlib_primary",
    )

    assert result["status"] == "done"
    assert result["ranking_mode"] == "qlib_primary"
    assert [row["code"] for row in result["candidates"]] == ["000002", "000003", "000001"]
    assert [row["l0_score"] for row in result["candidates"]] == [0.9, 0.5, 0.2]
    assert all(row["confidence"] is None for row in result["recommended"])
    assert all(row["recommendation_level"] == "量化排名" for row in result["recommended"])
    assert result["auxiliary_coverage"]["industry"]["rate"] == 1.0


@pytest.mark.p0
def test_screen_pipeline_exposes_score_ledger(monkeypatch):
    from core.data import Database
    from core.data.sources import etf_flow
    from core.domain.screening._screener import normalize_qlib_scores

    monkeypatch.setattr(Database, "fetchall", lambda *args, **kwargs: [])
    monkeypatch.setattr(etf_flow, "get_etf_flow_by_industry", lambda **kwargs: {})
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())
    details = normalize_qlib_scores(
        {"000001": -0.1, "000002": 0.8, "000003": 0.2},
        "2026-08-01", neutralize=False, return_details=True,
    )
    monkeypatch.setattr(screener, "_load_qlib_predictions", lambda *args, **kwargs: details)

    result = screener.screen(
        pool="all", top_n=3, qlib_model=("experiment", "run"),
        eval_date=datetime.date(2026, 8, 1), snapshot=_snapshot(),
        source_weights={"qlib": 1.0},
        skip_l2=True, skip_debate=True, skip_backfill=True,
        sector_fusion=False, ranking_mode="qlib_primary",
    )

    best = result["candidates"][0]
    assert best["code"] == "000002"
    assert best["raw_model_score"] == 0.8
    assert best["raw_rank"] == 1
    assert best["rank_percentile"] == pytest.approx(2 / 3)
    assert result["score_processing"]["neutralize_status"] == "disabled"
    assert result["score_processing"]["raw_score_available"] is True


@pytest.mark.p0
def test_screen_pipeline_rejects_prediction_older_than_decision_date(monkeypatch):
    from core.data import Database
    from core.data.sources import etf_flow
    from core.domain.screening._screener import normalize_qlib_scores

    monkeypatch.setattr(Database, "fetchall", lambda *args, **kwargs: [])
    monkeypatch.setattr(etf_flow, "get_etf_flow_by_industry", lambda **kwargs: {})
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())
    details = normalize_qlib_scores(
        {"000001": 0.2, "000002": 0.9, "000003": 0.5},
        "2026-08-01", neutralize=False, return_details=True,
    )
    details["audit"]["context"]["prediction_date"] = "2026-07-31"
    monkeypatch.setattr(screener, "_load_qlib_predictions", lambda *args, **kwargs: details)

    result = screener.screen(
        pool="all", top_n=3, qlib_model=("experiment", "run"),
        eval_date=datetime.date(2026, 8, 1), snapshot=_snapshot(),
        source_weights={"qlib": 1.0},
        skip_l2=True, skip_debate=True, skip_backfill=True,
        sector_fusion=False, ranking_mode="qlib_primary",
    )

    assert result["status"] == "error"
    assert "Qlib预测日期 2026-07-31 与决策日 2026-08-01 不一致" in result["error"]


@pytest.mark.p0
def test_qlib_primary_hard_filter_refills_and_can_be_disabled(monkeypatch):
    from core.data import Database
    from core.data.sources import etf_flow

    monkeypatch.setattr(Database, "fetchall", lambda *args, **kwargs: [])
    monkeypatch.setattr(etf_flow, "get_etf_flow_by_industry", lambda **kwargs: {})
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())
    snapshot = _snapshot()
    df, _, close, volume = snapshot["000002"]
    snapshot["000002"] = (df, "ST乙", close, volume)
    scores = {"000001": 0.7, "000002": 0.9, "000003": 0.8}

    filtered = screener._select_qlib_top_candidates(
        list(scores), scores, 2,
        eval_date=datetime.date(2026, 8, 1), snapshot=snapshot,
    )
    unfiltered = screener._select_qlib_top_candidates(
        list(scores), scores, 2,
        eval_date=datetime.date(2026, 8, 1), snapshot=snapshot,
        decision_day_filter=False,
    )

    assert [row["code"] for row in filtered["candidates"]] == ["000003", "000001"]
    assert filtered["excluded_reasons"]["st"] == 1
    assert [row["code"] for row in unfiltered["candidates"]] == ["000002", "000003"]


@pytest.mark.p0
@pytest.mark.parametrize("state_models", [None, {"high_vol": ["e", "r"], "low_vol": ["e", "t"]}])
def test_neutral_volatility_returns_empty_before_model_inference(monkeypatch, state_models):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from core.domain.regime.types import Regime

    date = datetime.date(2026, 8, 1)
    rr = SimpleNamespace(
        regime=Regime.STEADY_UP, confidence=0.8, sub_scores={}, as_of_date=date,
        details={"p_vol": 0.5, "vol_direction": "trend_momentum"},
    )
    monkeypatch.setattr("core.domain.regime.unified.get_unified_regime", lambda **kwargs: rr)
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())
    inference = Mock(side_effect=AssertionError("空仓不得进入L0"))
    monkeypatch.setattr(screener, "_l0_filter", inference)
    result = screener.screen(
        pool="all", top_n=1, eval_date=date, snapshot=_snapshot(),
        regime="auto", use_vol_switch=True, use_conditional_switch=True,
        conditional_market_env={date.isoformat(): (0.5, 0.0)}, state_models=state_models,
        ranking_mode="qlib_primary", source_weights={"qlib": 1.0},
        skip_l2=True, skip_debate=True, skip_backfill=True,
        macro_bias=False, macro_regime=False, sector_fusion=False,
    )

    assert result["status"] == "done"
    assert result["vol_switch_empty"] is True
    assert result["candidates"] == result["recommended"] == []
    assert result["direction"] == "neutral"
    inference.assert_not_called()


def test_sentiment_query_binds_dates_and_enforces_historical_cutoff(monkeypatch):
    from core.data import Database
    from core.data.sources import etf_flow

    date = datetime.date(2026, 8, 1)
    calls = []

    def fetchall(sql, params):
        if "FROM sentiment_daily" not in sql:
            return []
        calls.append((sql, params))
        assert params[:4] == (date,) * 4
        assert params[4:-2] == ("000002", "000001")
        assert params[-2:] == (date, date)
        assert "AND trade_date <= %s" in sql
        return [{"code": "000002", "s_7d": 0.6, "conf_7d": 0.8, "s_23d": 0.2}]

    monkeypatch.setattr(Database, "fetchall", fetchall)
    monkeypatch.setattr(etf_flow, "get_etf_flow_by_industry", lambda **kwargs: {})
    screener = BuySideScreener(fetcher=_Fetcher(), analyzer=object())
    result = screener._select_qlib_top_candidates(
        ["000001", "000002"], {"000001": 0.2, "000002": 0.9}, 2,
        eval_date=date, snapshot=_snapshot(),
    )

    assert len(calls) == 1
    assert result["candidates"][0]["sentiment_score"] > 0
