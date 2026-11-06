import datetime

import pytest

from backend.service.strategy_profile import PIT_CONTRACT_VERSION, StrategyProfileService


def test_snapshot_pit_gate_rejects_legacy_and_mismatched_contracts():
    assert not StrategyProfileService.snapshot_has_verified_pit({"pit_status": "pass"})
    assert not StrategyProfileService.snapshot_has_verified_pit({
        "pit_status": "pass",
        "pit_contract_version": PIT_CONTRACT_VERSION,
        "snapshot_json": {"pit_contract_version": "legacy"},
    })
    assert StrategyProfileService.snapshot_has_verified_pit({
        "pit_status": "pass",
        "pit_contract_version": PIT_CONTRACT_VERSION,
        "snapshot_json": {
            "pit_contract_version": PIT_CONTRACT_VERSION,
            "security_reference_status": "pass",
            "security_reference_coverage": 1.0,
            "security_reference_fingerprint": "b" * 64,
            "security_reference_input_fingerprint": "a" * 64,
        },
    })


def test_snapshot_current_integrity_requires_pit_and_content(monkeypatch):
    calls = []
    monkeypatch.setattr(
        StrategyProfileService,
        "snapshot_has_verified_pit",
        staticmethod(lambda _snapshot: True),
    )
    monkeypatch.setattr(
        StrategyProfileService,
        "_snapshot_content_matches",
        classmethod(lambda cls, _snapshot: calls.append("content") or True),
    )

    assert StrategyProfileService.snapshot_has_current_integrity({"snapshot_id": "snapshot"})
    assert calls == ["content"]

    monkeypatch.setattr(
        StrategyProfileService,
        "snapshot_has_verified_pit",
        staticmethod(lambda _snapshot: False),
    )
    calls.clear()
    assert not StrategyProfileService.snapshot_has_current_integrity({"snapshot_id": "legacy"})
    assert calls == []


def test_snapshot_pit_gate_rejects_missing_or_partial_security_reference_contract():
    base = {
        "pit_status": "pass",
        "pit_contract_version": PIT_CONTRACT_VERSION,
        "snapshot_json": {
            "pit_contract_version": PIT_CONTRACT_VERSION,
            "security_reference_status": "pass",
            "security_reference_coverage": 1.0,
            "security_reference_fingerprint": "b" * 64,
            "security_reference_input_fingerprint": "a" * 64,
        },
    }

    assert not StrategyProfileService.snapshot_has_verified_pit({**base, "snapshot_json": {}})
    assert not StrategyProfileService.snapshot_has_verified_pit({
        **base,
        "snapshot_json": {**base["snapshot_json"], "security_reference_status": "partial"},
    })
    assert not StrategyProfileService.snapshot_has_verified_pit({
        **base,
        "snapshot_json": {**base["snapshot_json"], "security_reference_coverage": 0.998},
    })
    assert not StrategyProfileService.snapshot_has_verified_pit({
        **base,
        "snapshot_json": {**base["snapshot_json"], "security_reference_input_fingerprint": "invalid"},
    })


def test_champion_runtime_guard_rejects_legacy_snapshot(monkeypatch):
    from core.data import StrategyProfileRepo

    monkeypatch.setattr(
        StrategyProfileRepo,
        "get_snapshot",
        classmethod(lambda cls, _snapshot_id: {"snapshot_id": "legacy", "pit_status": "pass"}),
    )
    monkeypatch.setattr(StrategyProfileRepo, "get_runtime_health", classmethod(lambda cls, *_args: None))
    monkeypatch.setattr(StrategyProfileService, "_snapshot_content_matches", classmethod(lambda cls, _snapshot: True))

    failures = StrategyProfileService.runtime_health_failures({
        "profile_id": "profile", "version": 1,
        "config": {"data": {"snapshot_id": "legacy"}, "model": {"refs": []}},
    })

    assert any("PIT契约" in reason for reason in failures)


def test_security_reference_import_keeps_admin_claims_unverified(monkeypatch):
    from core.data import StrategyProfileRepo

    saved = {}
    monkeypatch.setattr(
        StrategyProfileRepo,
        "save_security_references",
        classmethod(lambda cls, as_of, rows: saved.update(as_of=as_of, rows=rows) or len(rows)),
    )
    result = StrategyProfileService.import_security_references(
        "2026-10-31",
        [{
            "code": "600000", "name": "浦发银行", "industry": "银行", "board": "main",
            "listed_date": "1999-11-10",
        }],
        "vendor_archive", "2026Q3", "2026-10-30T16:00:00", "a" * 64,
    )

    assert result["status"] == "imported_unverified"
    assert saved["rows"][0]["provenance_status"] == "unverified"
    assert saved["rows"][0]["source_available_at"] == datetime.datetime(2026, 10, 30, 16)


def test_security_reference_import_rejects_source_available_on_decision_date(monkeypatch):
    with pytest.raises(ValueError, match="之前已公开"):
        StrategyProfileService.import_security_references(
            "2026-10-31",
            [{
                "code": "600000", "name": "浦发银行", "industry": "银行", "board": "main",
                "listed_date": "1999-11-10",
            }],
            "vendor_archive", "2026Q3", "2026-10-31T09:00:00", "a" * 64,
        )


def test_snapshot_without_source_visibility_evidence_stays_unverified(monkeypatch):
    as_of = datetime.date(2026, 10, 31)
    latest = datetime.date(2026, 10, 30)
    saved = {}

    monkeypatch.setattr(
        "core.data.Database.fetchone",
        lambda sql, params=None: {"d": latest},
    )

    def fetchall(sql, params=None):
        if "COUNT(*) AS observation_count" in sql:
            return [{
                "code": "600000",
                "observation_count": 500,
                "first_trade_date": datetime.date(2024, 1, 1),
                "last_trade_date": latest,
            }]
        if "FROM stock_sectors" in sql:
            return []
        if "SELECT code,volume" in sql:
            return [{"code": "600000", "volume": 100}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(
        "core.data.StrategyProfileRepo.market_data_manifest",
        lambda requested_as_of: {"daily_quotes": {"max_date": latest.isoformat()}},
    )
    monkeypatch.setattr("core.data.StrategyProfileRepo.get_snapshot_for_date", lambda *args: None)
    monkeypatch.setattr("core.data.StrategyProfileRepo.get_security_references", lambda *args: [])
    monkeypatch.setattr("core.data.StrategyProfileRepo.save_security_references", lambda *args: 1)
    monkeypatch.setattr(
        "core.data.StrategyProfileRepo.save_snapshot",
        lambda snapshot, members: saved.update(snapshot=snapshot, members=members) or snapshot,
    )

    result = StrategyProfileService.build_data_snapshot(as_of)

    assert result["pit_status"] == "unverified"
    assert result["pit_contract_version"] == PIT_CONTRACT_VERSION
    assert "source_available_at" in result["pit_status_reason"]
    assert saved["snapshot"]["pit_status"] == "unverified"


def test_legacy_cached_snapshot_marked_pass_is_rebuilt(monkeypatch):
    as_of = datetime.date(2026, 10, 31)
    latest = datetime.date(2026, 10, 30)
    monkeypatch.setattr(
        "core.data.Database.fetchone",
        lambda sql, params=None: {"d": latest},
    )

    def fetchall(sql, params=None):
        if "COUNT(*) AS observation_count" in sql:
            return [{
                "code": "600000",
                "observation_count": 500,
                "first_trade_date": datetime.date(2024, 1, 1),
                "last_trade_date": latest,
            }]
        if "FROM stock_sectors" in sql:
            return []
        if "SELECT code,volume" in sql:
            return [{"code": "600000", "volume": 100}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(
        "core.data.StrategyProfileRepo.market_data_manifest",
        lambda requested_as_of: {"daily_quotes": {"max_date": latest.isoformat()}},
    )
    monkeypatch.setattr("backend.service.strategy_profile._json_fingerprint", lambda value: "f" * 64)
    monkeypatch.setattr(
        "core.data.StrategyProfileRepo.get_snapshot_for_date",
        lambda *args: {
            "snapshot_id": "legacy",
            "as_of": as_of.isoformat(),
            "latest_trade_date": latest.isoformat(),
            "market_data_content_fingerprint": "f" * 64,
            "security_reference_status": "pass",
            "member_count": 1,
            "pit_status": "pass",
        },
    )
    monkeypatch.setattr("core.data.StrategyProfileRepo.get_security_references", lambda *args: [])
    monkeypatch.setattr("core.data.StrategyProfileRepo.save_security_references", lambda *args: 1)
    monkeypatch.setattr("core.data.StrategyProfileRepo.save_snapshot", lambda snapshot, members: snapshot)

    result = StrategyProfileService.build_data_snapshot(as_of)

    assert result["snapshot_id"] != "legacy"
    assert result["pit_status"] == "unverified"


def test_cached_snapshot_is_rebuilt_when_imported_reference_evidence_changes(monkeypatch):
    as_of = datetime.date(2026, 10, 31)
    latest = datetime.date(2026, 10, 30)
    monkeypatch.setattr("core.data.Database.fetchone", lambda sql, params=None: {"d": latest})

    def fetchall(sql, params=None):
        if "COUNT(*) AS observation_count" in sql:
            return [{
                "code": "600000", "observation_count": 500,
                "first_trade_date": datetime.date(2024, 1, 1), "last_trade_date": latest,
            }]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(
        "core.data.StrategyProfileRepo.market_data_manifest",
        lambda requested_as_of: {"daily_quotes": {"max_date": latest.isoformat()}},
    )
    monkeypatch.setattr(
        "backend.service.strategy_profile._json_fingerprint",
        lambda value: "market" if value == {"daily_quotes": {"max_date": latest.isoformat()}}
        else "references" if value == [] else "rebuilt",
    )
    monkeypatch.setattr(
        "core.data.StrategyProfileRepo.get_security_references",
        lambda requested_as_of: [],
    )
    monkeypatch.setattr(
        "core.data.StrategyProfileRepo.get_snapshot_for_date",
        lambda *args: {
            "snapshot_id": "cached", "pit_contract_version": PIT_CONTRACT_VERSION,
            "latest_trade_date": latest.isoformat(), "market_data_content_fingerprint": "market",
            "security_reference_input_fingerprint": "prior-import", "member_count": 1,
        },
    )
    monkeypatch.setattr("core.data.StrategyProfileRepo.save_security_references", lambda *args: 1)
    monkeypatch.setattr("core.data.StrategyProfileRepo.save_snapshot", lambda snapshot, members: snapshot)

    result = StrategyProfileService.build_data_snapshot(as_of)

    assert result["snapshot_id"] == "rebuilt"


def test_snapshot_content_gate_rejects_security_reference_change_even_when_market_cache_is_warm(monkeypatch):
    from backend.service.strategy_profile import (
        _json_fingerprint,
        _security_reference_fingerprint,
        _security_reference_input_fingerprint,
    )
    from core.data import StrategyProfileRepo

    references = [{
        "code": "600000", "name": "浦发银行", "industry": "银行", "board": "main",
        "is_st": False, "is_suspended": False, "listed_date": "1999-11-10", "delisted_date": None,
        "source": "pit_import", "provenance_status": "verified", "source_name": "archive",
        "source_version": "v1", "source_available_at": "2026-10-30 16:00:00",
        "source_evidence_sha256": "a" * 64,
    }]
    manifest = {"daily_quotes": {"max_date": "2026-10-30"}}
    StrategyProfileService._snapshot_integrity_cache.clear()
    monkeypatch.setattr(StrategyProfileRepo, "get_security_references", lambda _as_of: references)
    monkeypatch.setattr(StrategyProfileRepo, "get_snapshot_members", lambda _snapshot_id: [{"code": "600000"}])
    monkeypatch.setattr(StrategyProfileRepo, "market_data_manifest", lambda _as_of: manifest)
    snapshot = {
        "snapshot_id": "security-reference-drift",
        "as_of": "2026-10-31",
        "snapshot_json": {
            "market_data_content_fingerprint": _json_fingerprint(manifest),
            "security_reference_status": "pass",
            "security_reference_coverage": 1.0,
            "security_reference_fingerprint": _security_reference_fingerprint(references),
            "security_reference_input_fingerprint": _security_reference_input_fingerprint(references),
            "member_count": 1,
        },
    }

    assert StrategyProfileService._snapshot_content_matches(snapshot)
    references[0]["provenance_status"] = "unverified"

    assert not StrategyProfileService._snapshot_content_matches(snapshot)


def test_snapshot_content_gate_recomputes_verified_member_coverage(monkeypatch):
    from backend.service.strategy_profile import (
        _json_fingerprint,
        _security_reference_fingerprint,
        _security_reference_input_fingerprint,
    )
    from core.data import StrategyProfileRepo

    references = [{
        "code": "600000", "name": "浦发银行", "industry": "银行", "board": "main",
        "is_st": False, "is_suspended": False, "listed_date": "1999-11-10", "delisted_date": None,
        "source": "pit_import", "provenance_status": "unverified", "source_name": "archive",
        "source_version": "v1", "source_available_at": "2026-10-30 16:00:00",
        "source_evidence_sha256": "a" * 64,
    }]
    manifest = {"daily_quotes": {"max_date": "2026-10-30"}}
    StrategyProfileService._snapshot_integrity_cache.clear()
    monkeypatch.setattr(StrategyProfileRepo, "get_security_references", lambda _as_of: references)
    monkeypatch.setattr(StrategyProfileRepo, "get_snapshot_members", lambda _snapshot_id: [{"code": "600000"}])
    monkeypatch.setattr(StrategyProfileRepo, "market_data_manifest", lambda _as_of: manifest)
    snapshot = {
        "snapshot_id": "unverified-security-reference",
        "as_of": "2026-10-31",
        "snapshot_json": {
            "market_data_content_fingerprint": _json_fingerprint(manifest),
            "security_reference_status": "pass",
            "security_reference_coverage": 1.0,
            "security_reference_fingerprint": _security_reference_fingerprint(references),
            "security_reference_input_fingerprint": _security_reference_input_fingerprint(references),
            "member_count": 1,
        },
    }

    assert not StrategyProfileService._snapshot_content_matches(snapshot)


def test_champion_runtime_guard_blocks_unverified_current_security_references(monkeypatch):
    from backend.service.strategy_profile import (
        _json_fingerprint,
        _security_reference_fingerprint,
        _security_reference_input_fingerprint,
    )
    from core.data import StrategyProfileRepo

    references = [{
        "code": "600000", "name": "浦发银行", "industry": "银行", "board": "main",
        "is_st": False, "is_suspended": False, "listed_date": "1999-11-10", "delisted_date": None,
        "source": "pit_import", "provenance_status": "unverified", "source_name": "archive",
        "source_version": "v1", "source_available_at": "2026-10-30 16:00:00",
        "source_evidence_sha256": "a" * 64,
    }]
    manifest = {"daily_quotes": {"max_date": "2026-10-30"}}
    snapshot = {
        "snapshot_id": "champion-security-reference-drift",
        "as_of": "2026-10-31",
        "pit_status": "pass",
        "pit_contract_version": PIT_CONTRACT_VERSION,
        "snapshot_json": {
            "pit_contract_version": PIT_CONTRACT_VERSION,
            "market_data_content_fingerprint": _json_fingerprint(manifest),
            "security_reference_status": "pass",
            "security_reference_coverage": 1.0,
            "security_reference_fingerprint": _security_reference_fingerprint(references),
            "security_reference_input_fingerprint": _security_reference_input_fingerprint(references),
            "member_count": 1,
        },
    }
    StrategyProfileService._snapshot_integrity_cache.clear()
    monkeypatch.setattr(StrategyProfileRepo, "get_snapshot", classmethod(lambda cls, _snapshot_id: snapshot))
    monkeypatch.setattr(StrategyProfileRepo, "get_snapshot_members", lambda _snapshot_id: [{"code": "600000"}])
    monkeypatch.setattr(StrategyProfileRepo, "get_security_references", lambda _as_of: references)
    monkeypatch.setattr(StrategyProfileRepo, "market_data_manifest", lambda _as_of: manifest)
    monkeypatch.setattr(StrategyProfileRepo, "get_runtime_health", classmethod(lambda cls, *_args: None))

    failures = StrategyProfileService.runtime_health_failures({
        "profile_id": "profile", "version": 1,
        "config": {"data": {"snapshot_id": snapshot["snapshot_id"]}, "model": {"refs": []}},
    })

    assert any("数据内容发生漂移" in reason for reason in failures)
