import pytest

from scripts.atr_research_common import find_seen_data_overlaps, require_oos_or_override


class Runner:
    def _load_models_cached(self, directions):
        return {"trend_momentum": ("1", "run-id")}


@pytest.mark.p0
def test_missing_model_segments_fail_closed(monkeypatch, tmp_path):
    from core.shared import config

    monkeypatch.setattr(config, "QLIB_MLFLOW_URI", f"file:{tmp_path}")
    overlaps = find_seen_data_overlaps(Runner(), "2026-07-01", "2026-08-14")
    assert overlaps[0]["split"] == "unknown"
    with pytest.raises(SystemExit):
        require_oos_or_override(
            Runner(), "2026-07-01", "2026-08-14", allow_seen_data=False)
