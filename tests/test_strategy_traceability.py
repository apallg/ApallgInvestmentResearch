import pytest
from unittest.mock import Mock


pytestmark = [pytest.mark.p0, pytest.mark.unit]


def test_rebalance_session_persists_all_fingerprints_and_runtime_equity(monkeypatch):
    from core.data.repos.session import RebalanceSessionRepo
    from core.data.repos.strategy_profile import StrategyProfileRepo
    from core.data.connection import Database

    calls = {}
    monkeypatch.setattr(StrategyProfileRepo, "ensure_schema", classmethod(lambda cls: None))
    monkeypatch.setattr(
        StrategyProfileRepo, "record_runtime_equity",
        classmethod(lambda cls, *args: calls.setdefault("equity", args)),
    )
    cursor = Mock()
    cursor.fetchone.return_value = {"status": "active", "version": 1}
    cursor.lastrowid = 17
    cursor.execute.side_effect = lambda sql, params=(): calls.update(sql=sql, params=params)
    monkeypatch.setattr("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity", lambda cur: {"ok": True})
    monkeypatch.setattr(Database, "transaction", classmethod(lambda cls, operation: operation(cursor)))

    result = RebalanceSessionRepo.create(
        "batch", "rebalance", "2026-09-15", {"total_assets": 80000}, [], [],
        profile_id="profile", profile_version=2, profile_fingerprint="p" * 64,
        data_fingerprint="d" * 64, model_fingerprint="m" * 64, factor_fingerprint="f" * 64,
        account_id="LOCAL-LIVE", account_version=1,
    )

    assert result == 17
    assert "data_fingerprint,model_fingerprint,factor_fingerprint" in calls["sql"]
    assert calls["params"][17:20] == ("d" * 64, "m" * 64, "f" * 64)
    assert calls["equity"] == ("profile", 2, 80000, "2026-09-15")


def test_rebalance_signal_persists_all_fingerprints(monkeypatch):
    from core.data.repos.rebalance import RebalanceSignalRepo
    from core.data.repos.strategy_profile import StrategyProfileRepo
    from core.data.connection import Database

    calls = {}
    monkeypatch.setattr(StrategyProfileRepo, "ensure_schema", classmethod(lambda cls: None))
    cursor = Mock()
    cursor.fetchone.side_effect = [{"status": "active", "version": 1}, None]
    cursor.fetchall.side_effect = [[], [], [{"id": 9}]]
    cursor.executemany.side_effect = lambda sql, rows: calls.update(sql=sql, rows=rows)
    monkeypatch.setattr("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity", lambda cur: {"ok": True})
    monkeypatch.setattr(Database, "transaction", classmethod(lambda cls, operation: operation(cursor)))

    ids = RebalanceSignalRepo.batch_create(
        "batch", 1, "rebalance", [{"code": "000001", "side": "buy", "planned_shares": 100}],
        profile_id="profile", profile_version=2, profile_fingerprint="p" * 64,
        data_fingerprint="d" * 64, model_fingerprint="m" * 64, factor_fingerprint="f" * 64,
        source_mode="live", account_id="LOCAL-LIVE", account_version=1,
        replace_pending=True, input_fingerprint="snapshot", fingerprint_fn=lambda rows: "snapshot",
    )

    assert ids == [9]
    assert "data_fingerprint,model_fingerprint" in calls["sql"]
    assert calls["rows"][0][16:19] == ("d" * 64, "m" * 64, "f" * 64)
    assert calls["rows"][0][20] == "live"


def test_portfolio_decision_persists_all_fingerprints(monkeypatch):
    from core.data.repos.portfolio_decision import PortfolioDecisionRepo
    from core.data.repos.strategy_profile import StrategyProfileRepo
    from core.data.connection import Database

    calls = {}
    monkeypatch.setattr(PortfolioDecisionRepo, "ensure_schema", classmethod(lambda cls: None))
    monkeypatch.setattr(StrategyProfileRepo, "ensure_schema", classmethod(lambda cls: None))
    monkeypatch.setattr(Database, "execute", classmethod(lambda cls, sql, params=(): calls.update(sql=sql, params=params)))
    monkeypatch.setattr(Database, "executemany", classmethod(lambda cls, sql, rows: None))
    run = {
        "decision_id": "decision", "screening_save_id": 1, "eval_date": "2026-09-15",
        "data_cutoff": "2026-09-15", "policy_id": "policy", "policy_version": 1,
        "mode": "baseline", "top_n": 5, "source_fingerprint": "s" * 64,
        "result_fingerprint": "r" * 64, "result": {}, "profile_id": "profile",
        "profile_version": 2, "profile_fingerprint": "p" * 64,
        "data_fingerprint": "d" * 64, "model_fingerprint": "m" * 64,
        "factor_fingerprint": "f" * 64,
    }

    PortfolioDecisionRepo.save_run(run, [])

    assert "data_fingerprint,model_fingerprint,factor_fingerprint" in calls["sql"]
    assert calls["params"][-3:] == ("d" * 64, "m" * 64, "f" * 64)


def test_runtime_equity_ignores_uninitialized_zero(monkeypatch):
    from core.data.repos.strategy_profile import StrategyProfileRepo
    from core.data.connection import Database

    monkeypatch.setattr(StrategyProfileRepo, "ensure_schema", classmethod(lambda cls: None))
    monkeypatch.setattr(StrategyProfileRepo, "get_runtime_health", classmethod(lambda cls, _profile_id, _version: None))
    monkeypatch.setattr(Database, "execute", classmethod(lambda cls, sql, params=(): pytest.fail("不应初始化零净值基线")))
    assert StrategyProfileRepo.record_runtime_equity("profile", 1, 0) is None
