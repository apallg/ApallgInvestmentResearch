import json
import hashlib

import pytest

from core.domain.factors.genetic import GeneticFactorMiner


def _fake_batch_ic(self, expressions, dates, progress_cb=None, return_metrics=False, include_series=False):
    result = []
    for expression in expressions:
        value = (int(hashlib.sha256(expression.encode("utf-8")).hexdigest()[:8], 16) % 1000) / 10000
        result.append({
            "mean": value,
            "std": 0.1,
            "icir": value / 0.1,
            "positive_ratio": 0.6,
            "coverage": 1.0,
            "count": len(dates),
        })
    return result if return_metrics else [item["mean"] for item in result]


def test_genetic_mining_checkpoint_resume_matches_uninterrupted_run(monkeypatch):
    monkeypatch.setattr(GeneticFactorMiner, "_batch_ic", _fake_batch_ic)
    miner = GeneticFactorMiner({"2026-01-01": {}, "2026-01-02": {}})
    params = {
        "population": 12,
        "generations": 5,
        "top_n": 4,
        "train_dates": ["2026-01-01"],
        "valid_dates": ["2026-01-02"],
        "seed": 81,
        "max_depth": 3,
    }
    expected = miner.mine(**params)
    checkpoints = []

    def save_checkpoint(state):
        checkpoints.append(json.loads(json.dumps(state, sort_keys=True)))
        if state["next_generation"] == 2:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        miner.mine(**params, checkpoint_cb=save_checkpoint)

    actual = miner.mine(**params, resume_state=checkpoints[-1])
    assert actual == expected
    assert checkpoints[-1]["next_generation"] == 2


def test_genetic_mining_rejects_checkpoint_with_mismatched_population(monkeypatch):
    monkeypatch.setattr(GeneticFactorMiner, "_batch_ic", _fake_batch_ic)
    miner = GeneticFactorMiner({"2026-01-01": {}})
    state = {
        "version": 1,
        "next_generation": 1,
        "population": [],
        "archive": [],
        "rng_state": {},
    }
    with pytest.raises(ValueError, match="种群与当前配置不匹配"):
        miner.mine(population=4, generations=2, train_dates=["2026-01-01"], resume_state=state)
