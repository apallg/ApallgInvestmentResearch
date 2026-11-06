import datetime as dt


def test_tuning_service_injects_memmap_panel_and_cleans_up(monkeypatch):
    import backend.service.tuning as module

    events = []

    class Store:
        backing_mode = "memmap"

        def __len__(self):
            return 2

        def clear(self):
            events.append("store.clear")

    store = Store()

    class Tuner:
        def _get_eval_dates(self, weeks, step):
            return [dt.date(2024, 1, 5), dt.date(2024, 1, 10)]

        def phase2_weight_optimization(self, **kwargs):
            assert kwargs["nm_snapshot_cache"] is store
            return {"ok": True}

    monkeypatch.setattr(module, "get_tuner", lambda: Tuner())
    monkeypatch.setattr(
        module.FactorPanelService,
        "build_shared_memmap",
        lambda self, *args, **kwargs: (
            "manifest.json", {"mapped_bytes": 1024}, ["600001"],
        ),
    )
    monkeypatch.setattr(
        module.FactorPanelService,
        "open_shared_memmap",
        lambda self, *args, **kwargs: (store, {"values": object()}, {}),
    )
    monkeypatch.setattr(
        module.FactorPanelService,
        "close_shared_memmap",
        lambda self, arrays: events.append("arrays.close"),
    )
    monkeypatch.setattr(
        module.FactorPanelService,
        "cleanup_shared_memmap",
        lambda self, directory: events.append("directory.cleanup"),
    )

    result = module.TuningService().phase2_weight_optimization(
        active_signals=["tech", "qlib"], eval_weeks=12, job_id="test",
    )

    assert result == {"ok": True}
    assert events == ["store.clear", "arrays.close", "directory.cleanup"]


def test_nelder_mead_uses_injected_cache_without_legacy_precompute(monkeypatch):
    from core.domain.tuning._tuner import ScreeningConfigTuner

    tuner = ScreeningConfigTuner()
    monkeypatch.setattr(
        tuner,
        "_precompute_snapshots_parallel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy precompute called")),
    )
    monkeypatch.setattr(
        tuner,
        "historical_validate",
        lambda *args, **kwargs: {"results": [{"fitness": 0.2}]},
    )
    result = tuner._nelder_mead_optimize(
        ["tech", "qlib"],
        {"skip_tech": False, "skip_qlib": False},
        ["tech", "qlib"],
        pool="csi300",
        opt_weeks=6,
        top_k=10,
        code_prefixes=None,
        progress={},
        regime=None,
        snapshot_cache={"2024-01-05": {}},
    )

    assert result is not None
    assert abs(sum(result["source_weights"].values()) - 1) < 1e-9
