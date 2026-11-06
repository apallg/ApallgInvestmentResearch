import pytest

from backend.service.portfolio_decision import PortfolioDecisionService, load_amount_median_20d, load_fundamental_quality


pytestmark = [pytest.mark.p0, pytest.mark.unit]


def test_fundamental_quality_is_cut_off_by_known_date(monkeypatch):
    captured = {}

    def fake_fetchall(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return [{
            "code": "000001", "report_date": "2023-12-31", "ann_date": "2024-01-30",
            "source_observed_at": "2024-01-30 18:00:00", "source_capture_id": "capture-1",
            "source_record_hash": "a" * 64, "source_name": "tushare_fina_indicator",
            "version_kind": "provider_observation", "payload_json": '{"profit_dedt":1,"debt_to_assets":2}',
        }]

    monkeypatch.setattr("core.data.connection.Database.fetchall", fake_fetchall)
    monkeypatch.setattr(
        "core.data.fetcher.DataFetcher.get_industry_map",
        lambda self: {"000001": {"name": "平安银行", "industry": "J66货币金融服务"}},
    )

    result = load_fundamental_quality(["000001"], "2024-01-31")

    assert "fundamental_data_versions" in captured["sql"]
    assert "source_name='tushare_fina_indicator'" in captured["sql"]
    assert "version_kind='provider_observation'" in captured["sql"]
    assert "payload_json" in captured["sql"]
    assert "DATE(source_observed_at)<%s" in captured["sql"]
    assert "fundamental_data f" not in captured["sql"]
    assert captured["params"] == ("2024-01-31", "2024-01-31", "000001")
    assert result["000001"]["profit_dedt"] == 1
    assert result["000001"]["industry"] == "J66货币金融服务"


def test_fundamental_quality_uses_latest_eligible_immutable_version(monkeypatch):
    monkeypatch.setattr(
        "core.data.connection.Database.fetchall",
        lambda sql, params: [
            {
                "code": "000001", "report_date": "2023-12-31", "ann_date": "2024-01-25",
                "source_observed_at": "2024-01-26 09:00:00", "source_capture_id": "capture-1",
                "source_record_hash": "a" * 64, "source_name": "tushare_fina_indicator",
                "version_kind": "provider_observation",
                "payload_json": '{"profit_dedt":"10","debt_to_assets":"40"}',
            },
            {
                "code": "000001", "report_date": "2023-12-31", "ann_date": "2024-01-30",
                "source_observed_at": "2024-01-30 18:00:00", "source_capture_id": "capture-2",
                "source_record_hash": "b" * 64, "source_name": "tushare_fina_indicator",
                "version_kind": "provider_observation",
                "payload_json": '{"profit_dedt":"12","debt_to_assets":"38"}',
            },
            {
                "code": "000001", "report_date": "2023-12-31", "ann_date": "2024-01-30",
                "source_observed_at": "2024-01-29 18:00:00", "source_capture_id": "official-1",
                "source_record_hash": "c" * 64, "source_name": "official_report",
                "version_kind": "official_field_reconciliation",
                "payload_json": '{"official_values":{"profit_dedt":"13","debt_to_assets":"35"}}',
            },
        ],
    )
    monkeypatch.setattr(
        "core.data.fetcher.DataFetcher.get_industry_map",
        lambda self: {},
    )

    result = load_fundamental_quality(["000001"], "2024-01-31")

    assert result["000001"]["profit_dedt"] == 13
    assert result["000001"]["debt_to_assets"] == 35


def test_amount_median_20d_is_cut_off_by_trade_date(monkeypatch):
    captured = {}

    def fake_fetchall(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return (
            [{"code": "000001", "amount": amount} for amount in range(100, 1600, 100)]
            + [{"code": "000002", "amount": 500}] * 15
        )

    monkeypatch.setattr("core.data.connection.Database.fetchall", fake_fetchall)

    result = load_amount_median_20d(["000001"], "2024-01-31")

    assert "trade_date<=%s" in captured["sql"]
    assert "ROW_NUMBER() OVER" in captured["sql"]
    assert "row_num<=20" in captured["sql"]
    assert captured["params"] == ("000001", "2024-01-31")
    assert result == {"000001": 800.0, "000002": 500.0}


def test_amount_median_requires_fifteen_valid_days(monkeypatch):
    monkeypatch.setattr(
        "core.data.connection.Database.fetchall",
        lambda sql, params: [{"code": "000001", "amount": 100}] * 14,
    )

    assert load_amount_median_20d(["000001"], "2024-01-31") == {}


def test_preview_does_not_restore_hard_exclusions_when_candidates_are_insufficient(monkeypatch):
    monkeypatch.setattr(
        "core.data.ScreeningSaveRepo.get_by_id",
        lambda save_id: {
            "id": save_id,
            "eval_date": "2024-01-31",
            "candidates": [
                {"code": "000001"}, {"code": "000002"},
                {"code": "600001", "name": "ST风险"},
            ],
        },
    )
    monkeypatch.setattr("core.data.HoldingsRepo.get_all", lambda: [])
    monkeypatch.setattr("core.data.PortfolioDecisionRepo.save_run", lambda run, candidates: None)
    monkeypatch.setattr(
        PortfolioDecisionService,
        "_policy",
        classmethod(lambda cls, policy_id: {
            "policy_id": "legacy", "version": 1, "mode": "legacy",
            "candidate_pool": 50, "buffer_multiple": 2, "status": "active",
        }),
    )
    monkeypatch.setattr(
        "backend.service.portfolio_decision.load_fundamental_quality",
        lambda codes, as_of: {
            "000001": {"profit_dedt": -1, "debt_to_assets": 20},
            "000002": {"profit_dedt": 1, "debt_to_assets": 20},
        },
    )
    monkeypatch.setattr(
        "backend.service.portfolio_decision.load_amount_median_20d",
        lambda codes, as_of: {code: 20_000_000 for code in codes},
    )

    result = PortfolioDecisionService.preview(1, top_n=5)

    assert [row["code"] for row in result["selected"]] == ["000002"]
    assert {(row["code"], row["exclusion_reason"]) for row in result["excluded"]} == {
        ("000001", "扣非净利润为负"),
        ("600001", "ST股票"),
    }


def test_shadow_v2_policy_is_registered_as_non_executable_shadow(monkeypatch):
    captured = {}

    def fake_upsert(policy, status):
        captured["policy"] = policy
        captured["status"] = status

    monkeypatch.setattr("core.data.PortfolioDecisionRepo.upsert_policy", fake_upsert)

    policy = PortfolioDecisionService._policy("shadow_v2")

    assert policy["score_mode"] == "rank_overlay_v2"
    assert policy["status"] == "shadow"
    assert captured["status"] == "shadow"
