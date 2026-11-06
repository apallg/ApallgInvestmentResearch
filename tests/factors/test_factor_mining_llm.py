import json
import datetime as dt

import pytest

from backend.service.factor_mining import FactorMiningService
from core.adapters.nlp.factor_candidates import CandidateProposalError, LLMFactorCandidateSource
from core.domain.factors.candidate import candidate_key


class FakeLLMClient:
    provider = "deepseek"
    model_name = "fake-factor-model"

    def __init__(self, content):
        self.content = content
        self.messages = None

    def chat_detailed(self, messages, temperature, max_tokens):
        self.messages = messages
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model_name,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }


def _context():
    config = FactorMiningService._normalize_config({
        "train_start": "2020-01-01", "train_end": "2023-12-31",
        "valid_start": "2024-01-01", "valid_end": "2024-12-31",
        "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
        "candidate_sources": ["llm"], "llm_candidate_count": 2,
    })
    return FactorMiningService._llm_proposal_context("run-test", config)


def test_llm_candidate_source_parses_and_audits_json():
    payload = {"proposals": [{
        "expr": "Delta($close, 20) / (Ref($close, 20) + 0.00000001)",
        "hypothesis": "中期价格变化延续",
        "rationale": "趋势信息可能缓慢扩散",
        "expected_sign": "positive",
        "tags": ["momentum"],
    }]}
    client = FakeLLMClient(f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```")
    result = LLMFactorCandidateSource(client=client).propose(_context())

    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["proposal_id"]
    assert result["audit"]["model"] == "fake-factor-model"
    assert result["audit"]["usage"]["total_tokens"] == 30
    prompt = json.dumps(client.messages, ensure_ascii=False)
    assert "2023-12-31" in prompt
    assert "2025-01-01" not in prompt
    assert "2025-12-31" not in prompt


def test_llm_candidate_source_rejects_non_json_response():
    source = LLMFactorCandidateSource(client=FakeLLMClient("这里有三个不错的因子"))
    with pytest.raises(CandidateProposalError, match="JSON"):
        source.propose(_context())


def test_llm_context_excludes_validation_and_final_oos_evidence():
    context = _context()
    assert context["train_cutoff"] == "2023-12-31"
    assert "valid_start" not in context
    assert "valid_end" not in context
    assert "final_oos_start" not in context
    assert "final_oos_end" not in context
    assert "metrics" not in context


def test_llm_config_and_candidate_key_are_source_aware():
    config = FactorMiningService._normalize_config({
        "train_start": "2020-01-01", "train_end": "2023-12-31",
        "valid_start": "2024-01-01", "valid_end": "2024-12-31",
        "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
        "candidate_sources": ["gp", "llm"], "llm_candidate_count": 12,
    })
    assert config["candidate_sources"] == ["gp", "llm"]
    assert config["llm_candidate_count"] == 12
    assert candidate_key("$close", source="llm").startswith("llm_")
    assert candidate_key("$close").startswith("gp_")


def test_llm_config_rejects_unbounded_candidate_budget():
    with pytest.raises(ValueError, match="llm_candidate_count"):
        FactorMiningService._normalize_config({
            "train_start": "2020-01-01", "train_end": "2023-12-31",
            "valid_start": "2024-01-01", "valid_end": "2024-12-31",
            "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
            "candidate_sources": ["llm"], "llm_candidate_count": 1000,
        })


def test_llm_resume_replays_persisted_proposals_without_new_call():
    config = FactorMiningService._normalize_config({
        "train_start": "2020-01-01", "train_end": "2023-12-31",
        "valid_start": "2024-01-01", "valid_end": "2024-12-31",
        "final_oos_start": "2025-01-01", "final_oos_end": "2025-12-31",
        "candidate_sources": ["llm"], "llm_candidate_count": 1,
    })
    persisted = [{
        "proposal_id": "proposal-1", "call_id": "call-1", "candidate_key": None,
        "expr": "Delta($close, 20)", "hypothesis": "动量延续", "rationale": "信息扩散",
        "expected_sign": "positive", "tags": ["momentum"], "provider": "deepseek",
        "model": "deepseek-chat", "prompt_profile": "factor_hypothesis_v1",
        "prompt_hash": "prompt", "response_hash": "response", "latency_ms": 10,
        "usage": {}, "status": "proposed", "rejection_reason": "",
    }]
    service = FactorMiningService(
        proposal_source_factory=lambda _: (_ for _ in ()).throw(AssertionError("不应再次调用 LLM"))
    )
    service._repo = lambda method, *args, **kwargs: persisted if method == "list_proposals" else None

    class FakeMiner:
        calls = 0

        def _batch_ic(self, exprs, dates, progress_cb=None, return_metrics=False):
            self.calls += 1
            return [{
                "mean": 0.03, "std": 0.02, "icir": 1.5,
                "positive_ratio": 0.7, "coverage": 1.0, "count": 10,
            } for _ in exprs]

    miner = FakeMiner()
    candidates = []
    accepted, audit = service._generate_llm_candidates(
        "resume-run", config, miner,
        [dt.date(2023, 12, 1)], [dt.date(2024, 1, 1)], candidates, [],
    )
    assert len(accepted) == 1
    assert accepted[0]["source"] == "llm"
    assert audit["replayed"] is True
    assert miner.calls == 2
