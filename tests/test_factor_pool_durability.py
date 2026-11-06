import copy

import pytest

from core.data import FactorPoolStateRepo
from core.domain.factors import pool


def test_database_read_failure_does_not_load_json_snapshot(monkeypatch, tmp_path):
    json_path = tmp_path / "factor_pool_state.json"
    json_path.write_text('{"white_box":{"stale":{"active":true}}}', encoding="utf-8")
    monkeypatch.setattr(pool, "STATE_PATH", str(json_path))
    monkeypatch.setattr(pool, "LEGACY_STATE_PATH", str(tmp_path / "legacy.json"))
    monkeypatch.setattr(pool.os, "environ", {"APALLG_FACTOR_POOL_STATE_PATH": str(json_path)})
    monkeypatch.setattr(FactorPoolStateRepo, "load", classmethod(lambda cls: (_ for _ in ()).throw(ConnectionError("db down"))))
    monkeypatch.setattr(FactorPoolStateRepo, "load_json", classmethod(lambda cls, path: pytest.fail("JSON fallback used")))

    with pytest.raises(RuntimeError, match="权威数据库不可用"):
        pool.FactorPoolManager()


def test_missing_database_record_with_json_snapshot_requires_explicit_recovery(monkeypatch, tmp_path):
    json_path = tmp_path / "factor_pool_state.json"
    json_path.write_text('{"white_box":{"stale":{"active":true}}}', encoding="utf-8")
    monkeypatch.setattr(pool, "STATE_PATH", str(json_path))
    monkeypatch.setattr(pool, "LEGACY_STATE_PATH", str(tmp_path / "legacy.json"))
    monkeypatch.setenv("APALLG_FACTOR_POOL_STATE_PATH", str(json_path))
    monkeypatch.setattr(FactorPoolStateRepo, "load", classmethod(lambda cls: None))
    monkeypatch.setattr(FactorPoolStateRepo, "load_json", classmethod(lambda cls, path: pytest.fail("JSON fallback used")))

    with pytest.raises(RuntimeError, match="显式核验并恢复"):
        pool.FactorPoolManager()


def test_database_write_failure_rolls_back_memory_and_skips_json(monkeypatch, tmp_path):
    monkeypatch.setattr(pool, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(FactorPoolStateRepo, "load", classmethod(lambda cls: {"white_box": {"old": {"active": True}}}))
    monkeypatch.setattr(FactorPoolStateRepo, "now_iso", classmethod(lambda cls: "2026-10-31T00:00:00"))
    monkeypatch.setattr(FactorPoolStateRepo, "save", classmethod(lambda cls, state: (_ for _ in ()).throw(ConnectionError("db write failed"))))
    monkeypatch.setattr(FactorPoolStateRepo, "save_json", classmethod(lambda cls, state, path: pytest.fail("JSON export used")))
    manager = pool.FactorPoolManager()
    persisted = copy.deepcopy(manager.state)
    manager.state["white_box"]["new"] = {"active": True}

    with pytest.raises(RuntimeError, match="变更已回滚"):
        manager._save()

    assert manager.state == persisted
    assert manager._persisted_state == persisted


def test_json_export_failure_keeps_committed_database_state(monkeypatch, tmp_path):
    monkeypatch.setattr(pool, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(FactorPoolStateRepo, "load", classmethod(lambda cls: {"white_box": {"old": {"active": True}}}))
    monkeypatch.setattr(FactorPoolStateRepo, "now_iso", classmethod(lambda cls: "2026-10-31T00:00:00"))
    committed = []
    monkeypatch.setattr(FactorPoolStateRepo, "save", classmethod(lambda cls, state: committed.append(copy.deepcopy(state))))
    monkeypatch.setattr(FactorPoolStateRepo, "save_json", classmethod(lambda cls, state, path: (_ for _ in ()).throw(OSError("disk full"))))
    manager = pool.FactorPoolManager()
    manager.state["white_box"]["new"] = {"active": True}

    manager._save()

    assert len(committed) == 1
    assert manager._persisted_state == committed[0]
    assert manager.state == committed[0]
