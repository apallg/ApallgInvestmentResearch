from pathlib import Path

import pytest


@pytest.mark.unit
@pytest.mark.p0
def test_factor_pool_state_is_outside_repository_during_tests():
    from core.domain.factors.pool import STATE_PATH

    repository_state = Path(__file__).parents[1] / "core" / "domain" / "factors" / "factor_pool_state.json"
    assert Path(STATE_PATH).resolve() != repository_state.resolve()


@pytest.mark.unit
@pytest.mark.p0
def test_factor_pool_json_writer_creates_runtime_directory(tmp_path):
    from core.data.repos.factors import FactorPoolStateRepo

    target = tmp_path / "nested" / "factor_pool_state.json"
    FactorPoolStateRepo.save_json({"version": 1}, str(target))

    assert target.read_text(encoding="utf-8").strip().startswith("{")
