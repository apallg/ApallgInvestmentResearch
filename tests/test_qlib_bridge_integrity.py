import datetime
import multiprocessing
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from core.adapters.qlib.bridge import FIELDS, QlibDataBridge
from core.shared.persistence import save_json


def _stage_qlib_generation_process(root, started, finished, result_queue):
    from core.adapters.qlib.generation import stage_qlib_generation

    started.set()
    stage = stage_qlib_generation(data_root=root)
    stage_dir = Path(stage["stage_dir"])
    result_queue.put((str(stage_dir), (stage_dir / "calendars" / "day.txt").read_text(encoding="utf-8")))
    finished.set()


def _crash_process_during_qlib_generation_copy(root, generation_id, copy_started):
    from core.adapters.qlib import generation

    def copy_then_crash(source, destination, *args, **kwargs):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "partial-copy-marker").write_text("partial", encoding="utf-8")
        copy_started.set()
        os._exit(86)

    generation.shutil.copytree = copy_then_crash
    generation.stage_qlib_generation(data_root=root, generation_id=generation_id)


def _crash_process_after_qlib_generation_stage(root, generation_id, stage_ready):
    from core.adapters.qlib.generation import stage_qlib_generation

    stage_qlib_generation(data_root=root, generation_id=generation_id, lifecycle_lease=True)
    stage_ready.set()
    os._exit(87)


def _crash_process_after_qlib_generation_promotion(root, generation_id, promoted):
    from core.adapters.qlib import generation

    stage = generation.stage_qlib_generation(
        data_root=root, generation_id=generation_id, lifecycle_lease=True,
    )
    save_json_atomic = generation.save_json_atomic

    def crash_before_pointer(value, path):
        if Path(path).name == "current.json":
            promoted.set()
            os._exit(88)
        return save_json_atomic(value, path)

    generation.save_json_atomic = crash_before_pointer
    generation.publish_qlib_generation(stage)


def _crash_process_after_qlib_generation_pointer(root, generation_id, pointer_saved):
    from core.adapters.qlib import generation

    stage = generation.stage_qlib_generation(
        data_root=root, generation_id=generation_id, lifecycle_lease=True,
    )
    save_json_atomic = generation.save_json_atomic

    def crash_after_pointer(value, path):
        save_json_atomic(value, path)
        if Path(path).name == "current.json":
            pointer_saved.set()
            os._exit(89)

    generation.save_json_atomic = crash_after_pointer
    generation.publish_qlib_generation(stage)


def _hold_qlib_stage_lease_process(root, generation_id, lease_ready, release_lease):
    from core.adapters.qlib.generation import stage_qlib_generation

    stage_qlib_generation(
        data_root=root,
        generation_id=generation_id,
        lifecycle_lease=True,
        owner={"type": "isolated_lease_test"},
    )
    lease_ready.set()
    release_lease.wait(15)


@pytest.mark.unit
@pytest.mark.p0
def test_bridge_constructor_does_not_mutate_managed_generation_paths(tmp_path, monkeypatch):
    import shutil

    from core.adapters.qlib import generation
    from core.shared import config

    root = tmp_path / "qlib-root"
    source = tmp_path / "qlib-source"
    for directory in ("calendars", "instruments", "features"):
        (source / directory).mkdir(parents=True)
    (source / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (source / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    stage = generation.stage_qlib_generation(source_dir=source, data_root=root)
    published = generation.publish_qlib_generation(stage, expected_generation_id=None)
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", root)
    monkeypatch.setattr(config, "QLIB_DATA_DIR", Path(published["data_dir"]))

    QlibDataBridge(root)

    assert not (root / "calendars").exists()
    assert not (root / "instruments").exists()
    assert not (root / "features").exists()

    features_dir = Path(published["data_dir"]) / "features"
    shutil.rmtree(features_dir)
    with pytest.raises(RuntimeError, match="受管Qlib数据代目录不完整，拒绝自动创建"):
        QlibDataBridge(published["data_dir"])
    assert not features_dir.exists()


@pytest.mark.unit
@pytest.mark.p0
def test_full_sync_is_read_only_and_exports_vwap_on_qfq_basis(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    dates = [datetime.date(2026, 9, 1), datetime.date(2026, 9, 2)]
    rows = [
        {"code": "600000", "trade_date": dates[0], "open": 50, "high": 50,
         "low": 50, "close": 50, "volume": 100, "amount": 10000,
         "adj_factor": 1},
        {"code": "600000", "trade_date": dates[1], "open": 100, "high": 100,
         "low": 100, "close": 100, "volume": 100, "amount": 10000,
         "adj_factor": 2},
    ]
    execute_calls = []

    def fetchall(sql, params=None):
        if "SELECT DISTINCT trade_date" in sql:
            return [{"trade_date": value} for value in dates]
        if "MIN(trade_date)" in sql:
            return [{"code": "600000", "start": dates[0], "end": dates[1], "cnt": 600}]
        if "WHERE code IN" in sql:
            return rows
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr("core.data.Database.execute", lambda *args, **kwargs: execute_calls.append(args))
    monkeypatch.setattr("core.adapters.qlib.bridge.verify_price_adjustment", lambda: None)
    monkeypatch.setattr(
        "core.data.repos.instruments.InstrumentRepo.get_stable_history_500d_ranges",
        lambda codes=None: {"600000": (dates[1], dates[1])},
    )

    bridge.sync_all()

    values = np.fromfile(bridge.features_dir / "sh600000" / "vwap.day.bin", dtype="<f")
    assert execute_calls == []
    assert values.tolist() == [0.0, 50.0, 100.0]
    assert (bridge.instruments_dir / "a500.txt").read_text(encoding="utf-8") == (
        "sh600000\t2026-09-02\t2026-09-02\n"
    )


@pytest.mark.unit
@pytest.mark.p0
def test_incremental_exports_vwap_on_qfq_basis(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    dates = [datetime.date(2026, 9, 1), datetime.date(2026, 9, 2)]
    (bridge.calendar_dir / "day.txt").write_text(
        "2026-09-01\n2026-09-02\n", encoding="utf-8"
    )
    (bridge.instruments_dir / "all.txt").write_text(
        "sh600000\t2026-09-01\t2026-09-02\n", encoding="utf-8"
    )
    rows = [
        {"code": "600000", "trade_date": dates[0], "open": 50, "high": 50,
         "low": 50, "close": 50, "volume": 100, "amount": 10000,
         "adj_factor": 1},
        {"code": "600000", "trade_date": dates[1], "open": 100, "high": 100,
         "low": 100, "close": 100, "volume": 100, "amount": 10000,
         "adj_factor": 2},
    ]

    def fetchall(sql, params=None):
        if "trade_date >" in sql:
            return []
        if "trade_date IN" in sql or "WHERE code IN" in sql:
            return rows
        if "COUNT(*) as cnt" in sql:
            return [{"code": "600000", "cnt": 600}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(bridge, "_refresh_index_instruments", lambda existing: False)
    monkeypatch.setattr(
        "core.data.repos.instruments.InstrumentRepo.get_stable_history_500d_ranges",
        lambda codes=None: {"600000": (dates[1], dates[1])},
    )

    bridge.sync_incremental()

    values = np.fromfile(bridge.features_dir / "sh600000" / "vwap.day.bin", dtype="<f")
    assert values.tolist() == [0.0, 50.0, 100.0]
    assert (bridge.instruments_dir / "a500.txt").read_text(encoding="utf-8") == (
        "sh600000\t2026-09-02\t2026-09-02\n"
    )


@pytest.mark.unit
@pytest.mark.p0
def test_incremental_refresh_without_new_calendar_date_preserves_instrument_end(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    dates = [datetime.date(2026, 9, 1) + datetime.timedelta(days=offset) for offset in range(15)]
    date_strings = [value.isoformat() for value in dates]
    (bridge.calendar_dir / "day.txt").write_text("\n".join(date_strings) + "\n", encoding="utf-8")
    (bridge.instruments_dir / "all.txt").write_text(
        "sh600000\t2026-09-01\t2026-09-15\n", encoding="utf-8"
    )
    stock_dir = bridge.features_dir / "sh600000"
    stock_dir.mkdir(parents=True)
    for field in FIELDS:
        np.hstack([np.float32(0), np.ones(15, dtype=np.float32)]).astype("<f").tofile(
            stock_dir / f"{field}.day.bin"
        )

    def fetchall(sql, params=None):
        if "trade_date >" in sql:
            return []
        if "trade_date IN" in sql:
            return [
                {
                    "code": "600000", "trade_date": value,
                    "open": 10, "high": 11, "low": 9, "close": 10,
                    "volume": 100, "amount": 1000,
                }
                for value in dates[-10:]
            ]
        if "COUNT(*) as cnt" in sql:
            return [{"code": "600000", "cnt": 600}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(bridge, "_refresh_index_instruments", lambda existing: False)

    result = bridge.sync_incremental()

    assert result["new_dates"] == 0
    assert (bridge.instruments_dir / "all.txt").read_text(encoding="utf-8").strip() == (
        "sh600000\t2026-09-01\t2026-09-15"
    )


@pytest.mark.unit
@pytest.mark.p0
def test_incremental_new_stock_uses_its_actual_first_exported_date(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    (bridge.calendar_dir / "day.txt").write_text(
        "2026-09-01\n2026-09-02\n", encoding="utf-8"
    )
    (bridge.instruments_dir / "all.txt").write_text("", encoding="utf-8")

    def fetchall(sql, params=None):
        if "trade_date >" in sql:
            return [
                {"trade_date": datetime.date(2026, 9, 3)},
                {"trade_date": datetime.date(2026, 9, 4)},
            ]
        if "trade_date IN" in sql:
            return [{
                "code": "600000", "trade_date": datetime.date(2026, 9, 4),
                "open": 10, "high": 11, "low": 9, "close": 10,
                "volume": 100, "amount": 1000,
            }]
        if "COUNT(*) as cnt" in sql:
            return [{"code": "600000", "cnt": 1}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(bridge, "_refresh_index_instruments", lambda existing: False)

    bridge.sync_incremental()

    assert (bridge.instruments_dir / "all.txt").read_text(encoding="utf-8").strip() == (
        "sh600000\t2026-09-04\t2026-09-04"
    )


@pytest.mark.unit
@pytest.mark.p0
def test_incremental_keeps_calendar_gaps_as_nan(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    (bridge.calendar_dir / "day.txt").write_text(
        "2026-09-01\n2026-09-02\n2026-09-03\n", encoding="utf-8"
    )
    (bridge.instruments_dir / "all.txt").write_text(
        "sh600000\t2026-09-01\t2026-09-03\n", encoding="utf-8"
    )

    def fetchall(sql, params=None):
        if "trade_date >" in sql:
            return []
        if "trade_date IN" in sql:
            return [
                {"code": "600000", "trade_date": datetime.date(2026, 9, 1),
                 "open": 10, "high": 10, "low": 10, "close": 10,
                 "volume": 100, "amount": 1000, "adj_factor": 1},
                {"code": "600000", "trade_date": datetime.date(2026, 9, 3),
                 "open": 12, "high": 12, "low": 12, "close": 12,
                 "volume": 100, "amount": 1200, "adj_factor": 1},
            ]
        if "COUNT(*) as cnt" in sql:
            return [{"code": "600000", "cnt": 2}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(bridge, "_refresh_index_instruments", lambda existing: False)

    bridge.sync_incremental()

    values = np.fromfile(
        bridge.features_dir / "sh600000" / "close.day.bin", dtype="<f"
    )
    assert values[0] == 0
    assert values[1] == 10
    assert np.isnan(values[2])
    assert values[3] == 12


@pytest.mark.unit
@pytest.mark.p0
def test_adj_factor_change_rewrites_full_instrument_history(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    (bridge.calendar_dir / "day.txt").write_text(
        "2026-09-01\n2026-09-02\n2026-09-03\n2026-09-04\n", encoding="utf-8"
    )
    (bridge.instruments_dir / "all.txt").write_text(
        "sh600000\t2026-09-01\t2026-09-04\n", encoding="utf-8"
    )
    stock_dir = bridge.features_dir / "sh600000"
    stock_dir.mkdir(parents=True)
    for field in FIELDS:
        np.hstack([np.float32(0), np.full(4, 100, dtype=np.float32)]).astype("<f").tofile(
            stock_dir / f"{field}.day.bin"
        )

    recent = [
        {"code": "600000", "trade_date": datetime.date(2026, 9, day),
         "open": value, "high": value, "low": value, "close": value,
         "volume": 100, "amount": value * 100, "adj_factor": factor}
        for day, value, factor in ((1, 50, 1), (2, 50, 1), (3, 50, 1), (4, 100, 2))
    ]

    full_query_calls = []

    def fetchall(sql, params=None):
        if "trade_date >" in sql:
            return []
        if "trade_date IN" in sql:
            return recent[-2:]
        if "WHERE code IN" in sql:
            full_query_calls.append(sql)
            return recent
        if "COUNT(*) as cnt" in sql:
            return [{"code": "600000", "cnt": 600}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(bridge, "_refresh_index_instruments", lambda existing: False)

    result = bridge.sync_incremental()

    values = np.fromfile(stock_dir / "close.day.bin", dtype="<f")
    assert values.tolist() == [0.0, 50.0, 50.0, 50.0, 100.0]
    assert result["rebased_stocks"] == 1

    second = bridge.sync_incremental()

    assert second["rebased_stocks"] == 0
    assert len(full_query_calls) == 1


@pytest.mark.unit
@pytest.mark.p0
def test_anchor_change_rebases_when_factor_change_is_outside_overlap(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    dates = [datetime.date(2026, 9, day) for day in range(1, 5)]
    (bridge.calendar_dir / "day.txt").write_text(
        "\n".join(value.isoformat() for value in dates) + "\n", encoding="utf-8"
    )
    (bridge.instruments_dir / "all.txt").write_text(
        "sh600000\t2026-09-01\t2026-09-04\n", encoding="utf-8"
    )
    save_json({"600000": 1.0}, tmp_path / "incremental_anchor_state.json")
    stock_dir = bridge.features_dir / "sh600000"
    stock_dir.mkdir(parents=True)
    for field in FIELDS:
        np.hstack([np.float32(0), np.full(4, 100, dtype=np.float32)]).astype("<f").tofile(
            stock_dir / f"{field}.day.bin"
        )

    full_rows = [
        {"code": "600000", "trade_date": value, "open": price, "high": price,
         "low": price, "close": price, "volume": 100, "amount": price * 100,
         "adj_factor": factor}
        for value, price, factor in zip(dates, [50, 50, 100, 100], [1, 1, 2, 2])
    ]
    full_query_calls = []

    def fetchall(sql, params=None):
        if "trade_date >" in sql:
            return []
        if "trade_date IN" in sql:
            return full_rows[-2:]
        if "WHERE code IN" in sql:
            full_query_calls.append(sql)
            return full_rows
        if "COUNT(*) as cnt" in sql:
            return [{"code": "600000", "cnt": 600}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(bridge, "_refresh_index_instruments", lambda existing: False)

    result = bridge.sync_incremental()

    assert result["rebased_stocks"] == 1
    assert len(full_query_calls) == 1


@pytest.mark.unit
@pytest.mark.p0
def test_date_coverage_checks_actual_close_value(tmp_path):
    bridge = QlibDataBridge(tmp_path)
    (bridge.calendar_dir / "day.txt").write_text(
        "2026-09-22\n2026-09-23\n", encoding="utf-8"
    )
    for code, values in (("sz000001", [10.0, 11.0]), ("sz000002", [10.0, np.nan])):
        stock_dir = bridge.features_dir / code
        stock_dir.mkdir(parents=True)
        np.hstack([np.float32(0), np.array(values, dtype=np.float32)]).astype("<f").tofile(
            stock_dir / "close.day.bin"
        )

    coverage = bridge.get_date_coverage("2026-09-23", ["000001", "000002"])

    assert coverage == {
        "expected": 2,
        "available": 1,
        "coverage_rate": 0.5,
        "missing_codes": ["000002"],
    }


@pytest.mark.unit
@pytest.mark.p0
def test_full_rebase_can_extend_before_existing_feature_start(tmp_path, monkeypatch):
    bridge = QlibDataBridge(tmp_path)
    dates = [datetime.date(2026, 9, day) for day in range(1, 5)]
    (bridge.calendar_dir / "day.txt").write_text(
        "\n".join(value.isoformat() for value in dates) + "\n", encoding="utf-8"
    )
    (bridge.instruments_dir / "all.txt").write_text(
        "sh600000\t2026-09-03\t2026-09-04\n", encoding="utf-8"
    )
    stock_dir = bridge.features_dir / "sh600000"
    stock_dir.mkdir(parents=True)
    for field in FIELDS:
        np.array([2, 100, 100], dtype="<f").tofile(stock_dir / f"{field}.day.bin")

    full_rows = [
        {"code": "600000", "trade_date": value, "open": price, "high": price,
         "low": price, "close": price, "volume": 100, "amount": price * 100,
         "adj_factor": factor}
        for value, price, factor in zip(dates, [50, 50, 50, 100], [1, 1, 1, 2])
    ]

    def fetchall(sql, params=None):
        if "trade_date >" in sql:
            return []
        if "trade_date IN" in sql:
            return full_rows[-2:]
        if "WHERE code IN" in sql:
            return full_rows
        if "COUNT(*) as cnt" in sql:
            return [{"code": "600000", "cnt": 600}]
        return []

    monkeypatch.setattr("core.data.Database.fetchall", fetchall)
    monkeypatch.setattr(bridge, "_refresh_index_instruments", lambda existing: False)

    bridge.sync_incremental()

    values = np.fromfile(stock_dir / "close.day.bin", dtype="<f")
    assert values.tolist() == [0.0, 50.0, 50.0, 50.0, 100.0]


@pytest.mark.unit
@pytest.mark.p0
def test_managed_generation_sync_failure_discards_staging_directory(tmp_path, monkeypatch):
    from core.adapters.qlib import generation
    from core.shared import config

    root = tmp_path / "qlib"
    for relative in ("calendars", "instruments", "features"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text(
        "sz000001\t2020-01-02\t2020-01-02\n", encoding="utf-8",
    )
    initial_stage = generation.stage_qlib_generation(data_root=root)
    initial = generation.publish_qlib_generation(initial_stage, expected_generation_id=None)
    active_dir = Path(initial["data_dir"])
    bridge = QlibDataBridge(active_dir)

    def fail_sync():
        raise RuntimeError("isolated sync failure")

    monkeypatch.setattr(config, "QLIB_DATA_ROOT", str(root))
    monkeypatch.setattr(config, "QLIB_DATA_DIR", active_dir)
    monkeypatch.setattr(QlibDataBridge, "_sync_incremental_in_place", lambda self: fail_sync())

    with pytest.raises(RuntimeError, match="isolated sync failure"):
        bridge.sync_incremental()

    generations_dir = root / "generations"
    assert generation.resolve_qlib_data_dir(root).name == initial["generation_id"]
    assert list(generations_dir.glob(".staging-*")) == []


@pytest.mark.unit
@pytest.mark.p0
@pytest.mark.parametrize("status,blocked", [("database_published", True), ("published", False)])
def test_init_qlib_checks_active_historical_release_marker(tmp_path, monkeypatch, status, blocked):
    from core.adapters.qlib import config as qlib_config
    from core.adapters.qlib.generation import mark_historical_release_active
    from core.shared import config
    from core.shared.persistence import save_json_atomic

    root = tmp_path / "data" / "qlib_data"
    root.mkdir(parents=True)
    release_id = "a" * 12
    manifest_path = tmp_path / "releases" / release_id / "manifest.json"
    save_json_atomic({"release_id": release_id, "status": status}, manifest_path)
    mark_historical_release_active(release_id, manifest_path, data_root=root)
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", str(root))
    monkeypatch.setattr(qlib_config, "_initialized", True)

    if blocked:
        with pytest.raises(RuntimeError, match="Qlib被未完成历史发布阻断"):
            QlibDataBridge(root)
        with pytest.raises(RuntimeError, match="Qlib被未完成历史发布阻断"):
            qlib_config.init_qlib()
    else:
        qlib_config.init_qlib()


@pytest.mark.unit
@pytest.mark.p0
def test_direct_qlib_read_services_check_active_historical_release_marker(tmp_path, monkeypatch):
    from core.adapters.qlib.generation import mark_historical_release_active
    from core.shared import config

    root = tmp_path / "data" / "qlib_data"
    root.mkdir(parents=True)
    release_id = "b" * 12
    manifest_path = tmp_path / "releases" / release_id / "manifest.json"
    from core.shared.persistence import save_json_atomic
    save_json_atomic({"release_id": release_id, "status": "database_published"}, manifest_path)
    mark_historical_release_active(release_id, manifest_path, data_root=root)
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", str(root))

    from backend.service.data_quality import DataQualityService
    from backend.service.instrument_resolver import get_qlib_data_dir
    from backend.service.pit import PitService
    from backend.service.historical_quality import get_historical_quality

    guarded_calls = (
        DataQualityService.validate_qlib_data,
        get_qlib_data_dir,
        PitService.check_pit,
        get_historical_quality,
    )
    for call in guarded_calls:
        with pytest.raises(RuntimeError, match="Qlib被未完成历史发布阻断"):
            call()


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_read_lease_serializes_generation_mutation(tmp_path):
    from core.adapters.qlib.generation import (
        _generation_pointer_lock,
        qlib_generation_mutation_lock,
        qlib_generation_read_lock,
    )

    root = tmp_path / "qlib"
    mutation_started = threading.Event()
    mutation_finished = threading.Event()

    def mutate():
        mutation_started.set()
        with qlib_generation_mutation_lock(root):
            mutation_finished.set()

    with qlib_generation_read_lock(root):
        worker = threading.Thread(target=mutate)
        worker.start()
        assert mutation_started.wait(1)
        assert not mutation_finished.wait(0.1)
    assert mutation_finished.wait(2)
    worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.unit
@pytest.mark.p0
@pytest.mark.parametrize("read_method", ["get_status", "get_date_coverage"])
def test_qlib_bridge_file_reads_hold_generation_read_lease(tmp_path, monkeypatch, read_method):
    from core.adapters.qlib.generation import qlib_generation_mutation_lock
    from core.shared import config

    root = tmp_path / "qlib"
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", str(root))
    bridge = QlibDataBridge(root)
    (root / "calendars" / "day.txt").write_text("2026-10-29\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sh600000\n", encoding="utf-8")
    started = threading.Event()
    finished = threading.Event()
    result = {}

    def read_status():
        started.set()
        if read_method == "get_status":
            result["value"] = bridge.get_status()
        else:
            result["value"] = bridge.get_date_coverage("2026-10-29", [])
        finished.set()

    with qlib_generation_mutation_lock(root):
        worker = threading.Thread(target=read_status)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(0.1)

    worker.join(timeout=2)
    assert not worker.is_alive()
    if read_method == "get_status":
        assert result["value"]["synced"] is True
        assert result["value"]["last_date"] == "2026-10-29"
    else:
        assert result["value"]["expected"] == 0
        assert result["value"]["available"] == 0


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_read_lease_rejects_generation_mutation_upgrade(tmp_path):
    from core.adapters.qlib.generation import (
        _generation_pointer_lock,
        qlib_generation_mutation_lock,
        qlib_generation_read_lock,
    )

    with qlib_generation_read_lock(tmp_path / "qlib"):
        with pytest.raises(RuntimeError, match="持有Qlib读租约期间拒绝升级"):
            with qlib_generation_mutation_lock(tmp_path / "qlib"):
                pass
        with pytest.raises(RuntimeError, match="持有Qlib读租约期间拒绝获取generation写锁"):
            with _generation_pointer_lock(tmp_path / "qlib"):
                pass


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_mutation_lock_is_reentrant_for_nested_publish_lock(tmp_path):
    from core.adapters.qlib.generation import _generation_pointer_lock, qlib_generation_mutation_lock

    with qlib_generation_mutation_lock(tmp_path / "qlib"):
        with _generation_pointer_lock(tmp_path / "qlib"):
            pass


@pytest.mark.unit
@pytest.mark.p0
@pytest.mark.parametrize(
    "guard_name",
    ["qlib_generation_read_lock", "qlib_generation_mutation_lock", "assert_no_active_historical_release"],
)
def test_qlib_generation_guards_reject_unmanaged_provider_uri_in_legacy_mode(
    tmp_path, monkeypatch, guard_name,
):
    from core.adapters.qlib import generation
    from core.shared import config

    root = (tmp_path / "qlib").resolve()
    unmanaged_provider = (tmp_path / "external-provider").resolve()
    monkeypatch.setattr(generation, "_configured_qlib_root_at_import", root)
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", root)
    monkeypatch.setattr(config, "QLIB_DATA_DIR", root)
    monkeypatch.setattr(config, "QLIB_DATA_GENERATION_ID", None)
    monkeypatch.setattr(config, "QLIB_PROVIDER_URI", str(unmanaged_provider))

    with pytest.raises(RuntimeError, match="QLIB_PROVIDER_URI 与受管Qlib数据目录不一致"):
        guard = getattr(generation, guard_name)
        if guard_name == "assert_no_active_historical_release":
            guard(root)
        else:
            with guard(root):
                pass


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_staging_waits_for_legacy_mutation_lock(tmp_path):
    from core.adapters.qlib.generation import qlib_generation_mutation_lock, stage_qlib_generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features" / "sh000001").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("old\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("old\n", encoding="utf-8")
    (root / "features" / "sh000001" / "close.day.bin").write_bytes(b"old")
    started = threading.Event()
    finished = threading.Event()
    result = {}

    def stage():
        started.set()
        result["stage"] = stage_qlib_generation(data_root=root)
        finished.set()

    with qlib_generation_mutation_lock(root, require_legacy=True):
        worker = threading.Thread(target=stage)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(0.1)
        (root / "calendars" / "day.txt").write_text("new\n", encoding="utf-8")
        (root / "instruments" / "all.txt").write_text("new\n", encoding="utf-8")
        (root / "features" / "sh000001" / "close.day.bin").write_bytes(b"new")

    worker.join(timeout=2)
    assert not worker.is_alive()
    stage_dir = Path(result["stage"]["stage_dir"])
    assert (stage_dir / "calendars" / "day.txt").read_text(encoding="utf-8") == "new\n"
    assert (stage_dir / "instruments" / "all.txt").read_text(encoding="utf-8") == "new\n"
    assert (stage_dir / "features" / "sh000001" / "close.day.bin").read_bytes() == b"new"


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_staging_waits_for_cross_process_mutation_lock(tmp_path):
    from core.adapters.qlib.generation import qlib_generation_mutation_lock

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("old\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("old\n", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    finished = context.Event()
    result_queue = context.Queue()
    process = context.Process(
        target=_stage_qlib_generation_process,
        args=(str(root), started, finished, result_queue),
    )

    with qlib_generation_mutation_lock(root, require_legacy=True):
        process.start()
        assert started.wait(5)
        assert not finished.wait(0.2)
        (root / "calendars" / "day.txt").write_text("new\n", encoding="utf-8")
        (root / "instruments" / "all.txt").write_text("new\n", encoding="utf-8")

    process.join(timeout=10)
    assert not process.is_alive()
    assert process.exitcode == 0
    stage_dir, calendar = result_queue.get(timeout=2)
    assert calendar == "new\n"
    assert (Path(stage_dir) / "instruments" / "all.txt").read_text(encoding="utf-8") == "new\n"


@pytest.mark.unit
@pytest.mark.p0
def test_interrupted_qlib_generation_copy_is_recovered_before_next_stage(tmp_path, monkeypatch):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    interrupted_id = "c" * 32
    copy_dir = root / "generations" / f".copying-{interrupted_id}"
    intent_path = root / "generations" / f".stage-copy-intent-{interrupted_id}.json"
    copytree = generation.shutil.copytree

    def copy_then_interrupt(source, destination, *args, **kwargs):
        copytree(source, destination, *args, **kwargs)
        raise SystemExit("simulated copy interruption")

    with monkeypatch.context() as patcher:
        patcher.setattr(generation.shutil, "copytree", copy_then_interrupt)
        with pytest.raises(SystemExit, match="copy interruption"):
            generation.stage_qlib_generation(data_root=root, generation_id=interrupted_id)

    assert copy_dir.is_dir()
    assert intent_path.is_file()
    assert not (root / "generations" / f".staging-{interrupted_id}").exists()

    stage = generation.stage_qlib_generation(data_root=root, generation_id="d" * 32)

    assert not copy_dir.exists()
    assert not intent_path.exists()
    assert Path(stage["stage_dir"]).is_dir()
    generation.discard_qlib_generation_stage(stage)


@pytest.mark.unit
@pytest.mark.p0
def test_stage_intent_persistence_failure_releases_lease_without_copy(tmp_path, monkeypatch):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    generation_id = "c" * 32
    intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"

    def fail_initial_intent(value, path):
        if Path(path) == intent_path:
            raise OSError("injected initial intent persistence failure")
        return generation.save_json_atomic(value, path)

    monkeypatch.setattr(generation, "save_json_atomic", fail_initial_intent)
    with pytest.raises(OSError, match="initial intent persistence failure"):
        generation.stage_qlib_generation(
            data_root=root, generation_id=generation_id, lifecycle_lease=True,
        )

    assert not intent_path.exists()
    assert not (root / "generations" / f".copying-{generation_id}").exists()
    assert not (root / "generations" / f".staging-{generation_id}").exists()
    assert generation._try_acquire_stage_lease(root, generation_id) is not None
    generation._release_stage_lease(root, generation_id)


@pytest.mark.unit
@pytest.mark.p0
def test_ready_intent_persistence_failure_removes_promoted_stage_and_releases_lease(tmp_path, monkeypatch):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    generation_id = "d" * 32
    intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"
    save_json_atomic = generation.save_json_atomic

    def fail_ready_intent(value, path):
        if Path(path) == intent_path and value.get("status") == "ready":
            raise OSError("injected ready intent persistence failure")
        return save_json_atomic(value, path)

    monkeypatch.setattr(generation, "save_json_atomic", fail_ready_intent)
    with pytest.raises(OSError, match="ready intent persistence failure"):
        generation.stage_qlib_generation(
            data_root=root, generation_id=generation_id, lifecycle_lease=True,
        )

    assert not intent_path.exists()
    assert not (root / "generations" / f".copying-{generation_id}").exists()
    assert not (root / "generations" / f".staging-{generation_id}").exists()
    assert generation._try_acquire_stage_lease(root, generation_id) is not None
    generation._release_stage_lease(root, generation_id)


@pytest.mark.unit
@pytest.mark.p0
def test_failed_stage_cleanup_keeps_recovery_intent_for_next_attempt(tmp_path, monkeypatch):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    interrupted_id = "a" * 32
    copy_dir = root / "generations" / f".copying-{interrupted_id}"
    intent_path = root / "generations" / f".stage-copy-intent-{interrupted_id}.json"
    copytree = generation.shutil.copytree
    rmtree = generation.shutil.rmtree

    def copy_then_fail(source, destination, *args, **kwargs):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "partial-copy-marker").write_text("partial", encoding="utf-8")
        raise OSError("injected copy failure")

    def fail_copy_cleanup(path, *args, **kwargs):
        if Path(path) == copy_dir:
            raise OSError("injected cleanup failure")
        return rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(generation.shutil, "copytree", copy_then_fail)
        patcher.setattr(generation.shutil, "rmtree", fail_copy_cleanup)
        with pytest.raises(RuntimeError, match="保留恢复清单待下次恢复"):
            generation.stage_qlib_generation(
                data_root=root, generation_id=interrupted_id, lifecycle_lease=True,
            )

    assert (copy_dir / "partial-copy-marker").is_file()
    assert intent_path.is_file()
    assert generation._try_acquire_stage_lease(root, interrupted_id) is not None
    generation._release_stage_lease(root, interrupted_id)

    stage = generation.stage_qlib_generation(data_root=root, generation_id="b" * 32)

    assert not copy_dir.exists()
    assert not intent_path.exists()
    assert Path(stage["stage_dir"]).is_dir()
    generation.discard_qlib_generation_stage(stage)


@pytest.mark.unit
@pytest.mark.p0
def test_interrupted_qlib_generation_copy_is_preserved_after_parent_change(tmp_path, monkeypatch):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    interrupted_id = "e" * 32
    copy_dir = root / "generations" / f".copying-{interrupted_id}"
    intent_path = root / "generations" / f".stage-copy-intent-{interrupted_id}.json"

    def copy_then_interrupt(source, destination, *args, **kwargs):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "partial-copy-marker").write_text("partial", encoding="utf-8")
        raise SystemExit("simulated copy interruption")

    with monkeypatch.context() as patcher:
        patcher.setattr(generation.shutil, "copytree", copy_then_interrupt)
        with pytest.raises(SystemExit, match="copy interruption"):
            generation.stage_qlib_generation(data_root=root, generation_id=interrupted_id)

    monkeypatch.setattr(generation, "_pointer", lambda _root: {"generation_id": "another-generation"})
    with pytest.raises(RuntimeError, match="父代已变化，保留候选复制现场待恢复"):
        generation.stage_qlib_generation(data_root=root, generation_id="f" * 32)

    assert (copy_dir / "partial-copy-marker").read_text(encoding="utf-8") == "partial"
    assert intent_path.is_file()


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_generation_stage_copy_recovers_after_os_process_crash(tmp_path):
    from core.adapters.qlib.generation import discard_qlib_generation_stage, stage_qlib_generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    interrupted_id = "9" * 32
    copy_dir = root / "generations" / f".copying-{interrupted_id}"
    intent_path = root / "generations" / f".stage-copy-intent-{interrupted_id}.json"
    context = multiprocessing.get_context("spawn")
    copy_started = context.Event()
    process = context.Process(
        target=_crash_process_during_qlib_generation_copy,
        args=(str(root), interrupted_id, copy_started),
    )

    process.start()
    assert copy_started.wait(10)
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("Qlib候选复制进程未按故障注入终止")
    assert process.exitcode == 86
    assert copy_dir.is_dir()
    assert intent_path.is_file()

    stage = stage_qlib_generation(data_root=root, generation_id="8" * 32)

    assert not copy_dir.exists()
    assert not intent_path.exists()
    assert Path(stage["stage_dir"]).is_dir()
    discard_qlib_generation_stage(stage)


@pytest.mark.unit
@pytest.mark.p0
def test_lifecycle_leased_stage_survives_active_owner_and_recovers_after_crash(tmp_path):
    from core.adapters.qlib.generation import discard_qlib_generation_stage, stage_qlib_generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    active = stage_qlib_generation(
        data_root=root, generation_id="a" * 32, lifecycle_lease=True,
        owner={"type": "benchmark_sync"},
    )

    next_stage = stage_qlib_generation(data_root=root, generation_id="b" * 32)
    assert Path(active["stage_dir"]).is_dir()
    assert (root / "generations" / f".stage-copy-intent-{'a' * 32}.json").is_file()
    discard_qlib_generation_stage(next_stage)
    discard_qlib_generation_stage(active)

    context = multiprocessing.get_context("spawn")
    stage_ready = context.Event()
    crashed_id = "c" * 32
    process = context.Process(
        target=_crash_process_after_qlib_generation_stage,
        args=(str(root), crashed_id, stage_ready),
    )
    process.start()
    assert stage_ready.wait(10)
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("Qlib候选生命周期租约进程未按故障注入终止")
    assert process.exitcode == 87
    crashed_stage_dir = root / "generations" / f".staging-{crashed_id}"
    crashed_intent = root / "generations" / f".stage-copy-intent-{crashed_id}.json"
    assert crashed_stage_dir.is_dir()
    assert crashed_intent.is_file()

    recovered = stage_qlib_generation(data_root=root, generation_id="d" * 32)

    assert not crashed_stage_dir.exists()
    assert not crashed_intent.exists()
    discard_qlib_generation_stage(recovered)


@pytest.mark.unit
@pytest.mark.p0
def test_lifecycle_publish_recovers_after_process_crash_between_promotion_and_pointer(tmp_path):
    from core.adapters.qlib.generation import discard_qlib_generation_stage, stage_qlib_generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    generation_id = "9" * 32
    final_dir = root / "generations" / generation_id
    intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"
    context = multiprocessing.get_context("spawn")
    promoted = context.Event()
    process = context.Process(
        target=_crash_process_after_qlib_generation_promotion,
        args=(str(root), generation_id, promoted),
    )

    process.start()
    assert promoted.wait(10)
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("Qlib候选发布进程未在指针提交前按故障注入终止")
    assert process.exitcode == 88
    assert final_dir.is_dir()
    assert intent_path.is_file()
    assert not (root / "current.json").exists()

    recovered = stage_qlib_generation(data_root=root, generation_id="a" * 32)

    assert not final_dir.exists()
    assert not intent_path.exists()
    assert Path(recovered["stage_dir"]).is_dir()
    discard_qlib_generation_stage(recovered)


@pytest.mark.unit
@pytest.mark.p0
def test_lifecycle_publish_recovers_after_pointer_commit_before_intent_cleanup(tmp_path):
    import json
    from core.adapters.qlib.generation import discard_qlib_generation_stage, stage_qlib_generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    generation_id = "b" * 32
    final_dir = root / "generations" / generation_id
    intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"
    context = multiprocessing.get_context("spawn")
    pointer_saved = context.Event()
    process = context.Process(
        target=_crash_process_after_qlib_generation_pointer,
        args=(str(root), generation_id, pointer_saved),
    )

    process.start()
    assert pointer_saved.wait(10)
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("Qlib候选发布进程未在恢复intent清理前按故障注入终止")
    assert process.exitcode == 89
    assert final_dir.is_dir()
    assert intent_path.is_file()
    pointer_before = json.loads((root / "current.json").read_text(encoding="utf-8"))
    assert pointer_before["generation_id"] == generation_id

    recovered = stage_qlib_generation(data_root=root, generation_id="c" * 32)

    assert final_dir.is_dir()
    assert not intent_path.exists()
    assert json.loads((root / "current.json").read_text(encoding="utf-8")) == pointer_before
    discard_qlib_generation_stage(recovered)


@pytest.mark.unit
@pytest.mark.p0
def test_publish_returns_committed_result_when_intent_cleanup_fails(tmp_path, monkeypatch):
    import json
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    generation_id = "f" * 32
    intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"
    stage = generation.stage_qlib_generation(
        data_root=root, generation_id=generation_id, lifecycle_lease=True,
    )
    unlink = Path.unlink

    def fail_intent_cleanup(path, *args, **kwargs):
        if path == intent_path:
            raise OSError("injected intent cleanup failure")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "unlink", fail_intent_cleanup)
        published = generation.publish_qlib_generation(stage)

    assert published["generation_id"] == generation_id
    assert published["recovery_pending"] is True
    assert json.loads((root / "current.json").read_text(encoding="utf-8"))["generation_id"] == generation_id
    assert Path(published["data_dir"]).is_dir()
    assert intent_path.is_file()
    assert generation._try_acquire_stage_lease(root, generation_id) is not None
    generation._release_stage_lease(root, generation_id)

    recovered = generation.stage_qlib_generation(data_root=root, generation_id="e" * 32)

    assert Path(published["data_dir"]).is_dir()
    assert not intent_path.exists()
    assert json.loads((root / "current.json").read_text(encoding="utf-8"))["generation_id"] == generation_id
    generation.discard_qlib_generation_stage(recovered)


@pytest.mark.unit
@pytest.mark.p0
def test_publish_recognizes_pointer_committed_before_atomic_writer_raises(tmp_path, monkeypatch):
    import json
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    generation_id = "0" * 32
    stage = generation.stage_qlib_generation(
        data_root=root, generation_id=generation_id, lifecycle_lease=True,
    )
    save_json_atomic = generation.save_json_atomic

    def persist_pointer_then_raise(value, path):
        save_json_atomic(value, path)
        if Path(path).name == "current.json":
            raise OSError("injected post-replace pointer persistence failure")

    monkeypatch.setattr(generation, "save_json_atomic", persist_pointer_then_raise)
    published = generation.publish_qlib_generation(stage)

    assert published["generation_id"] == generation_id
    assert published["recovery_pending"] is False
    assert json.loads((root / "current.json").read_text(encoding="utf-8"))["generation_id"] == generation_id
    assert not (root / "generations" / f".stage-copy-intent-{generation_id}.json").exists()
    assert Path(published["data_dir"]).is_dir()


@pytest.mark.unit
@pytest.mark.p0
def test_lifecycle_stage_cleans_candidate_if_publish_fails_after_directory_promotion(tmp_path, monkeypatch):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    stage = generation.stage_qlib_generation(
        data_root=root, generation_id="e" * 32, lifecycle_lease=True,
    )
    save_json_atomic = generation.save_json_atomic

    def fail_pointer_save(value, path):
        if Path(path).name == "current.json":
            raise OSError("injected pointer persistence failure")
        return save_json_atomic(value, path)

    monkeypatch.setattr(generation, "save_json_atomic", fail_pointer_save)
    with pytest.raises(OSError, match="pointer persistence failure"):
        generation.publish_qlib_generation(stage)

    generation.discard_qlib_generation_stage(stage)

    assert not (root / "generations" / ("e" * 32)).exists()
    assert not (root / "generations" / f".stage-copy-intent-{'e' * 32}.json").exists()


@pytest.mark.unit
@pytest.mark.p0
def test_lifecycle_stage_refuses_publish_without_recovery_intent(tmp_path):
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    stage = generation.stage_qlib_generation(
        data_root=root, generation_id="g" * 32, lifecycle_lease=True,
    )
    intent_path = root / "generations" / f".stage-copy-intent-{'g' * 32}.json"
    intent_path.unlink()

    with pytest.raises(RuntimeError, match="生命周期恢复清单缺失"):
        generation.publish_qlib_generation(stage)

    assert Path(stage["stage_dir"]).is_dir()
    assert not (root / "generations" / ("g" * 32)).exists()
    assert not (root / "current.json").exists()


@pytest.mark.unit
@pytest.mark.p0
def test_lifecycle_stage_refuses_publish_with_mismatched_recovery_owner(tmp_path):
    import json
    from core.adapters.qlib import generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    stage = generation.stage_qlib_generation(
        data_root=root, generation_id="h" * 32, lifecycle_lease=True,
        owner={"type": "isolated_owner"},
    )
    intent_path = root / "generations" / f".stage-copy-intent-{'h' * 32}.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["owner"] = {"type": "other_owner"}
    generation.save_json_atomic(intent, intent_path)

    with pytest.raises(RuntimeError, match="恢复清单身份不匹配"):
        generation.publish_qlib_generation(stage)

    assert Path(stage["stage_dir"]).is_dir()
    assert not (root / "generations" / ("h" * 32)).exists()
    assert not (root / "current.json").exists()


@pytest.mark.unit
@pytest.mark.p0
def test_historical_release_stage_is_not_reclaimed_by_generic_stage_recovery(tmp_path):
    import json
    from core.adapters.qlib.generation import discard_qlib_generation_stage, stage_qlib_generation

    root = tmp_path / "qlib"
    generations = root / "generations"
    stage_id = "f" * 32
    stage_dir = generations / f".staging-{stage_id}"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    stage_dir.mkdir(parents=True)
    (stage_dir / "calendars").mkdir()
    intent_path = generations / f".stage-copy-intent-{stage_id}.json"
    intent_path.write_text(json.dumps({
        "generation_id": stage_id,
        "data_root": str(root.resolve()),
        "parent_generation_id": None,
        "copy_dir": str(generations / f".copying-{stage_id}"),
        "stage_dir": str(stage_dir),
        "status": "ready",
        "lifecycle_lease": True,
        "owner": {"type": "historical_release", "release_id": "release-1"},
    }), encoding="utf-8")

    next_stage = stage_qlib_generation(data_root=root, generation_id="1" * 32)

    assert stage_dir.is_dir()
    assert intent_path.is_file()
    discard_qlib_generation_stage({
        "data_root": root,
        "generation_id": stage_id,
        "stage_dir": stage_dir,
        "parent_generation_id": None,
        "lifecycle_lease": True,
        "owner": {"type": "historical_release", "release_id": "release-1"},
    })
    discard_qlib_generation_stage(next_stage)
    assert not stage_dir.exists()


@pytest.mark.unit
@pytest.mark.p0
def test_cross_process_stage_lease_blocks_recovery_until_owner_exits(tmp_path):
    from core.adapters.qlib.generation import discard_qlib_generation_stage, stage_qlib_generation

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    lease_ready = context.Event()
    release_lease = context.Event()
    leased_id = "2" * 32
    leased_stage_dir = root / "generations" / f".staging-{leased_id}"
    process = context.Process(
        target=_hold_qlib_stage_lease_process,
        args=(str(root), leased_id, lease_ready, release_lease),
    )
    process.start()
    assert lease_ready.wait(10)
    try:
        other_stage = stage_qlib_generation(data_root=root, generation_id="3" * 32)
        assert leased_stage_dir.is_dir()
        assert (root / "generations" / f".stage-copy-intent-{leased_id}.json").is_file()
        discard_qlib_generation_stage(other_stage)
    finally:
        release_lease.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("Qlib候选租约持有进程未退出")
    assert process.exitcode == 0

    recovered = stage_qlib_generation(data_root=root, generation_id="4" * 32)

    assert not leased_stage_dir.exists()
    assert not (root / "generations" / f".stage-copy-intent-{leased_id}.json").exists()
    lease_file = root / "generations" / ".stage-leases" / f"{leased_id}.lock"
    assert lease_file.is_file()
    discard_qlib_generation_stage(recovered)


@pytest.mark.unit
@pytest.mark.p0
def test_benchmark_sync_locks_managed_stage_writes(tmp_path, monkeypatch):
    import pandas as pd
    from contextlib import nullcontext
    from types import SimpleNamespace

    from backend.service import qlib_benchmarks
    from core.adapters.qlib import generation
    from core.data import fetcher
    from core.shared import config

    root = tmp_path / "qlib"
    stage_dir = root / "generations" / f".staging-{'a' * 32}"
    (stage_dir / "features").mkdir(parents=True)
    (stage_dir / "instruments").mkdir()
    (stage_dir / "instruments" / "all.txt").write_text("sz399001\t2020-01-02\t2020-01-02\n", encoding="utf-8")
    bridge = SimpleNamespace(
        data_dir=stage_dir,
        features_dir=stage_dir / "features",
        instruments_dir=stage_dir / "instruments",
    )
    frame = pd.DataFrame([{
        "date": "2020-01-02", "open": 10, "high": 11, "low": 9, "close": 10,
        "volume": 100, "amount": 1000,
    }])
    lock_calls = []

    class FakeFetcher:
        def get_index_daily(self, index_name, days):
            return frame.copy()

    def tracked_lock(data_root, require_legacy=False):
        lock_calls.append((Path(data_root).resolve(), require_legacy))
        return nullcontext()

    monkeypatch.setattr(fetcher, "DataFetcher", FakeFetcher)
    monkeypatch.setattr(generation, "qlib_generation_mutation_lock", tracked_lock)
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", root)

    result = qlib_benchmarks.sync_benchmark_indices(
        bridge, ["2020-01-02"], {"2020-01-02": 0}, require_current=True,
    )

    assert result["indices"] == ["sh000300", "sh000001", "sz399001", "sz399006"]
    assert lock_calls == [(root.resolve(), False)]
    assert (stage_dir / "features" / "sh000300" / "close.day.bin").is_file()


@pytest.mark.unit
@pytest.mark.p0
def test_benchmark_sync_rejects_published_generation_write(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace

    from backend.service import qlib_benchmarks
    from core.data import fetcher
    from core.shared import config

    root = tmp_path / "qlib"
    published_dir = root / "generations" / ("b" * 32)
    (published_dir / "features").mkdir(parents=True)
    (published_dir / "instruments").mkdir()
    frame = pd.DataFrame([{
        "date": "2020-01-02", "open": 10, "high": 11, "low": 9, "close": 10,
        "volume": 100, "amount": 1000,
    }])

    class FakeFetcher:
        def get_index_daily(self, index_name, days):
            return frame.copy()

    monkeypatch.setattr(fetcher, "DataFetcher", FakeFetcher)
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", root)
    bridge = SimpleNamespace(
        data_dir=published_dir,
        features_dir=published_dir / "features",
        instruments_dir=published_dir / "instruments",
    )

    with pytest.raises(RuntimeError, match="拒绝修改已发布generation"):
        qlib_benchmarks.sync_benchmark_indices(
            bridge, ["2020-01-02"], {"2020-01-02": 0}, require_current=True,
        )


@pytest.mark.unit
@pytest.mark.p0
def test_qlib_read_gate_blocks_process_bound_to_previous_generation(tmp_path, monkeypatch):
    from core.adapters.qlib import generation
    from core.shared import config

    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars" / "day.txt").write_text("2026-09-01\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sh600000\t2026-09-01\t2026-09-01\n", encoding="utf-8")
    stage = generation.stage_qlib_generation(data_root=root)
    published = generation.publish_qlib_generation(stage, expected_generation_id=None)

    monkeypatch.setattr(generation, "_configured_qlib_root_at_import", root.resolve())
    monkeypatch.setattr(config, "QLIB_DATA_ROOT", str(root))
    monkeypatch.setattr(config, "QLIB_DATA_GENERATION_ID", "previousgeneration")
    monkeypatch.setattr(config, "QLIB_DATA_DIR", root / "generations" / "previousgeneration")

    with pytest.raises(RuntimeError, match="Qlib当前数据代与进程启动配置不一致"):
        generation.assert_no_active_historical_release(root)
    assert published["generation_id"] != config.QLIB_DATA_GENERATION_ID
