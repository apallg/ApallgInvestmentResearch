import datetime as dt
import hashlib
import json
import logging
import math
import os
import threading
import traceback
import uuid
from pathlib import Path

from core.adapters.nlp.factor_candidates import LLMFactorCandidateSource
from core.data.repos.factor_mining import factor_mining_repo
from core.domain.factors.archive import expression_fingerprint
from core.domain.factors.candidate import build_candidate, candidate_key, canonical_expression
from core.domain.factors.fitness import compute_training_fitness
from core.domain.factors.promotion import evaluate_promotion_gate
from core.domain.factors.walkforward import (
    aggregate_fold_metrics,
    build_l1_validation_dates,
    build_walkforward_folds,
    purge_dates_before_boundary,
    sample_trading_dates,
)
from core.shared.infrastructure.async_task import TaskCancelled, task_manager
from backend.service.qlib_access import run_with_qlib_read_access
from core.shared.persistence import load_json, save_json
from core.shared import config as app_config


logger = logging.getLogger(__name__)
REPORT_ROOT = os.path.join("data", "factor_mining")


class FactorMiningService:
    def __init__(self, proposal_source_factory=None):
        self._stage_clocks = {}
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._proposal_source_factory = proposal_source_factory or (
            lambda config: LLMFactorCandidateSource(provider=config["llm_provider"])
        )

    def _ensure_schema(self):
        if self._schema_ready:
            return
        with self._schema_lock:
            if not self._schema_ready:
                self._repo_required("ensure_tables")
                self._schema_ready = True

    def _record_stage(self, run_id, stage):
        now = dt.datetime.now()
        state = self._stage_clocks.get(run_id)
        if state is None:
            self._stage_clocks[run_id] = {
                "current": stage, "started_at": now, "items": {},
            }
            return
        if state["current"] == stage:
            return
        elapsed = (now - state["started_at"]).total_seconds()
        state["items"][state["current"]] = {
            "started_at": state["started_at"].isoformat(timespec="seconds"),
            "finished_at": now.isoformat(timespec="seconds"),
            "duration_seconds": round(elapsed, 3),
        }
        state["current"] = stage
        state["started_at"] = now

    def _finish_stage_clock(self, run_id):
        state = self._stage_clocks.pop(run_id, None)
        if state is None:
            return {}
        now = dt.datetime.now()
        state["items"][state["current"]] = {
            "started_at": state["started_at"].isoformat(timespec="seconds"),
            "finished_at": now.isoformat(timespec="seconds"),
            "duration_seconds": round((now - state["started_at"]).total_seconds(), 3),
        }
        return state["items"]

    def _repo(self, method, *args, **kwargs):
        try:
            return getattr(factor_mining_repo, method)(*args, **kwargs)
        except Exception as exc:
            logger.warning("因子挖矿持久化失败 %s: %s", method, exc)
            return None

    def _repo_required(self, method, *args, **kwargs):
        return getattr(factor_mining_repo, method)(*args, **kwargs)

    def create_run(self, config):
        run_id = uuid.uuid4().hex[:12]
        normalized = self._normalize_config(config)
        if normalized["candidate_sources"] == ["llm"] and not self._llm_provider_configured(
            normalized["llm_provider"]
        ):
            raise ValueError(f"{normalized['llm_provider']} API Key 未配置，无法启动纯 LLM 挖掘")
        progress = self._progress_payload("planning", 0, 1, normalized)
        self._ensure_schema()
        self._ensure_archive()
        self._repo_required("create_run", run_id, normalized, "planning", progress)
        task_manager.submit_with_id(
            run_id,
            lambda: run_with_qlib_read_access(self._run, run_id, normalized),
            f"因子挖矿 {normalized['pool']} {normalized['train_start']}~{normalized['valid_end']}",
            pool="light",
        )
        return {"run_id": run_id, "status": "pending", "progress": progress}

    @staticmethod
    def _mining_input_fingerprint(config, snapshots, market_caps, extra_factors, industry_map):
        import numpy as np
        import pandas as pd

        digest = hashlib.sha256()

        def add_json(value):
            digest.update(json.dumps(
                value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
            ).encode("utf-8"))

        add_json({
            "version": 1,
            "config": config,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        })
        for date_key in sorted(snapshots):
            add_json(str(date_key))
            day = snapshots[date_key]
            for code in sorted(day):
                item = day[code]
                add_json(str(code))
                if isinstance(item, (tuple, list)) and item:
                    frame = item[0]
                    add_json(item[1:])
                else:
                    frame = item
                if hasattr(frame, "columns") and hasattr(frame, "index"):
                    add_json({
                        "columns": list(frame.columns),
                        "dtypes": [str(value) for value in frame.dtypes],
                        "index_name": frame.index.name,
                    })
                    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
                else:
                    add_json(frame)
        add_json({
            "market_caps": market_caps,
            "extra_factors": extra_factors,
            "industry_map": industry_map,
        })
        root = Path(__file__).resolve().parents[2]
        for relative_path in (
            "backend/service/factor_mining.py",
            "backend/service/factor_panel.py",
            "core/data/repos/factor_panel.py",
            "core/domain/factors/candidate.py",
            "core/domain/factors/expression.py",
            "core/domain/factors/fitness.py",
            "core/domain/factors/genetic.py",
            "core/domain/factors/panel.py",
            "core/domain/factors/promotion.py",
            "core/domain/factors/validation.py",
            "core/domain/factors/walkforward.py",
            "core/domain/backtest/engine.py",
            "core/domain/backtest/screening.py",
        ):
            source = root / relative_path
            digest.update(relative_path.encode("utf-8"))
            digest.update(source.read_bytes())
        return digest.hexdigest()

    def resume_run(self, run_id):
        self._ensure_schema()
        if task_manager.is_active(run_id):
            raise ValueError("任务仍在运行，不能重复恢复")
        claimed = self._repo_required("claim_checkpoint_resume", run_id)
        if not claimed:
            raise ValueError("任务没有可用检查点，或最近仍有活动心跳")
        config = claimed.get("config") or {}
        checkpoint = claimed.get("checkpoint") or {}
        if checkpoint.get("version") != 1 or checkpoint.get("stage") != "gp_generation":
            self._repo_required("clear_checkpoint", run_id)
            self._repo_required("update_run", run_id, status="failed", error="检查点版本或阶段不受支持")
            raise ValueError("检查点版本或阶段不受支持，请重新提交任务")
        try:
            task_manager.submit_with_id(
                run_id,
                lambda: run_with_qlib_read_access(self._run, run_id, config, checkpoint),
                f"恢复因子挖矿 {config.get('pool', '')}",
                pool="light",
            )
        except Exception as exc:
            self._repo("update_run", run_id, status="failed", error=f"恢复任务启动失败: {exc}")
            raise
        return {"run_id": run_id, "status": "resuming", "progress": checkpoint.get("progress") or {}}

    @staticmethod
    def _llm_provider_configured(provider):
        from core.shared import config as shared_config
        if provider == "deepseek":
            return bool(shared_config.DEEPSEEK_API_KEY)
        if provider == "qwen":
            return bool(shared_config.QWEN_API_KEY)
        return False

    @staticmethod
    def _normalize_config(config):
        compatibility_mode = bool(config.get("compatibility_mode", False))
        final_oos_start = config.get("final_oos_start")
        final_oos_end = config.get("final_oos_end")
        if not final_oos_start or not final_oos_end:
            if not compatibility_mode:
                raise ValueError("正式因子挖掘必须填写最终 OOS 起止日期")
            final_oos_start = config["valid_start"]
            final_oos_end = config["valid_end"]
        raw_sources = config.get("candidate_sources") or ["gp"]
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        candidate_sources = list(dict.fromkeys(str(item).lower() for item in raw_sources))
        if not candidate_sources or set(candidate_sources) - {"gp", "llm"}:
            raise ValueError("candidate_sources 只能包含 gp 或 llm")
        result = {
            "pool": config.get("pool", "csi300"),
            "train_start": config["train_start"],
            "train_end": config["train_end"],
            "valid_start": config["valid_start"],
            "valid_end": config["valid_end"],
            "step": int(config.get("step", 5)),
            "population": int(config.get("population", 100)),
            "generations": int(config.get("generations", 10)),
            "top_n": int(config.get("top_n", 20)),
            "max_depth": int(config.get("max_depth", 5)),
            "forward_days": int(config.get("forward_days", 20)),
            "risk_neutralize": bool(config.get("risk_neutralize", True)),
            "validation_profile": config.get("validation_profile", "default"),
            "baseline_set": config.get("baseline_set", "current_curated"),
            "baseline_model_experiment_id": str(config.get("baseline_model_experiment_id") or "").strip(),
            "baseline_model_run_id": str(config.get("baseline_model_run_id") or "").strip(),
            "seed": int(config.get("seed", 42)),
            "backtest_profile": config.get("backtest_profile", "production_screening"),
            "backtest_freq": config.get("backtest_freq", "week"),
            "backtest_top_n": int(config.get("backtest_top_n", 10)),
            "backtest_use_timing": bool(config.get("backtest_use_timing", True)),
            "backtest_macro_bias": bool(config.get("backtest_macro_bias", True)),
            "backtest_sector_fusion": config.get("backtest_sector_fusion"),
            "overlay_weight": float(config.get("overlay_weight", 0.15)),
            "commission_rate": float(config.get("commission_rate", 0.0003)),
            "slippage_rate": float(config.get("slippage_rate", 0.0001)),
            "stamp_duty": float(config.get("stamp_duty", 0.0005)),
            "final_oos_start": final_oos_start,
            "final_oos_end": final_oos_end,
            "compatibility_mode": compatibility_mode,
            "candidate_sources": candidate_sources,
            "llm_provider": str(config.get("llm_provider") or "deepseek").lower(),
            "llm_candidate_count": int(config.get("llm_candidate_count", 10)),
            "llm_temperature": float(config.get("llm_temperature", 0.2)),
            "llm_max_tokens": int(config.get("llm_max_tokens", 4096)),
            "llm_prompt_profile": str(config.get("llm_prompt_profile") or "factor_hypothesis_v1"),
            "llm_research_focus": str(config.get("llm_research_focus") or "").strip()[:500],
            "allow_historical_revisit": bool(config.get("allow_historical_revisit", False)),
            "campaign_id": str(config.get("campaign_id") or "").strip(),
            "batch_index": int(config.get("batch_index", 0)),
            "defer_final_oos": bool(config.get("defer_final_oos", False)),
        }
        for key in ("train_start", "train_end", "valid_start", "valid_end"):
            dt.date.fromisoformat(result[key])
        if not result["train_start"] < result["train_end"] < result["valid_start"] < result["valid_end"]:
            raise ValueError("训练和验证日期必须严格递增且互不重叠")
        dt.date.fromisoformat(result["final_oos_start"])
        dt.date.fromisoformat(result["final_oos_end"])
        if result["final_oos_start"] > result["final_oos_end"]:
            raise ValueError("最终 OOS 日期范围无效")
        if result["final_oos_start"] <= result["valid_end"] and not compatibility_mode:
            raise ValueError("最终 OOS 必须严格晚于验证区间")
        if result["step"] <= 0 or result["population"] < 2 or result["generations"] < 1:
            raise ValueError("step、population、generations 必须为正数，population 至少为 2")
        if result["top_n"] < 1 or result["forward_days"] < 1:
            raise ValueError("top_n 和 forward_days 必须为正数")
        if not 2 <= result["max_depth"] <= 8:
            raise ValueError("max_depth 必须在 2 到 8 之间")
        if result["backtest_top_n"] < 1 or not 0 <= result["overlay_weight"] <= 0.5:
            raise ValueError("backtest_top_n 或候选因子权重无效")
        if bool(result["baseline_model_experiment_id"]) != bool(result["baseline_model_run_id"]):
            raise ValueError("手工选择模型时必须同时提供 experiment_id 和 run_id")
        if result["llm_provider"] not in ("deepseek", "qwen"):
            raise ValueError("llm_provider 只能是 deepseek 或 qwen")
        if not 1 <= result["llm_candidate_count"] <= 50:
            raise ValueError("llm_candidate_count 必须在 1 到 50 之间")
        if not 0 <= result["llm_temperature"] <= 1:
            raise ValueError("llm_temperature 必须在 0 到 1 之间")
        if not 512 <= result["llm_max_tokens"] <= 8192:
            raise ValueError("llm_max_tokens 必须在 512 到 8192 之间")
        if result["llm_prompt_profile"] != "factor_hypothesis_v1":
            raise ValueError("不支持的 llm_prompt_profile")
        return result

    def _ensure_archive(self):
        if (self._repo("archive_count") or 0) > 0:
            return
        for candidate in self._repo("list_all_candidates", 10000) or []:
            self._repo(
                "upsert_archive", candidate["run_id"], candidate,
                expression_fingerprint(candidate["expr"]),
            )

    @staticmethod
    def _pool_expression_fingerprints():
        fingerprints = set()
        try:
            from core.domain.factors.registry import get_pool
            pool = get_pool()
            for category in ("white_box", "traditional", "hybrid"):
                for entry in (pool.state.get(category) or {}).values():
                    if entry.get("expr"):
                        fingerprints.add(expression_fingerprint(entry["expr"]))
        except Exception:
            pass
        return fingerprints

    @staticmethod
    def _llm_proposal_context(run_id, config):
        return {
            "run_id": run_id,
            "pool": config["pool"],
            "train_cutoff": config["train_end"],
            "forward_days": config["forward_days"],
            "candidate_count": config["llm_candidate_count"],
            "temperature": config["llm_temperature"],
            "max_tokens": config["llm_max_tokens"],
            "prompt_profile": config["llm_prompt_profile"],
            "research_focus": config["llm_research_focus"],
            "allowed_fields": [
                "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg",
            ],
            "allowed_operators": [
                "Ref", "Delay", "Mean", "Std", "Sum", "Delta", "Corr", "Cov",
                "Ts_Rank", "Abs", "Log", "Sign", "Max", "Min", "Rank", "Scale",
            ],
            "preferred_windows": [5, 10, 20, 40, 60, 120],
        }

    def _generate_llm_candidates(self, run_id, config, miner, train_dates, valid_dates,
                                 candidates, warnings, historical_fingerprints=None):
        context = self._llm_proposal_context(run_id, config)
        persisted_proposals = self._repo("list_proposals", run_id) or []
        if persisted_proposals:
            first = persisted_proposals[0]
            result = {
                "proposals": [{
                    "proposal_id": item["proposal_id"], "expr": item["expr"],
                    "hypothesis": item.get("hypothesis"), "rationale": item.get("rationale"),
                    "expected_sign": item.get("expected_sign"), "tags": item.get("tags") or [],
                } for item in persisted_proposals],
                "audit": {
                    "call_id": first.get("call_id"), "provider": first.get("provider"),
                    "model": first.get("model"), "prompt_profile": first.get("prompt_profile"),
                    "prompt_hash": first.get("prompt_hash"), "response_hash": first.get("response_hash"),
                    "latency_ms": first.get("latency_ms"), "usage": first.get("usage") or {},
                    "requested_count": config["llm_candidate_count"],
                    "returned_count": len(persisted_proposals), "replayed": True,
                },
            }
        else:
            try:
                source = self._proposal_source_factory(config)
                result = source.propose(context)
            except Exception as exc:
                failure_audit = {
                    "call_id": hashlib.sha1(f"{run_id}:llm-failed".encode("utf-8")).hexdigest()[:16],
                    "provider": config["llm_provider"],
                    "prompt_profile": config["llm_prompt_profile"],
                    "requested_count": config["llm_candidate_count"],
                    "returned_count": 0,
                    "usage": {},
                }
                self._repo("save_llm_call", run_id, failure_audit, "failed", str(exc))
                if config["candidate_sources"] == ["llm"]:
                    raise RuntimeError(f"LLM 候选生成失败: {exc}") from exc
                warnings.append(f"LLM 候选生成失败，已降级继续 GP: {exc}")
                return [], failure_audit

            audit = result.get("audit") or {}
            self._repo("save_llm_call", run_id, audit, "completed", "")
            for proposal in result.get("proposals") or []:
                self._repo(
                    "save_proposal", run_id, audit.get("call_id"), proposal,
                    "proposed", None, "",
                )

        audit = result.get("audit") or {}
        accepted = []
        seen = {canonical_expression(item["expr"]) for item in candidates}
        historical = set(historical_fingerprints or ())
        proposals = result.get("proposals") or []
        prepared = []
        for index, proposal in enumerate(proposals, 1):
            task_manager.check_cancelled(run_id)
            expr = proposal.get("expr") or ""
            try:
                built = build_candidate(expr, source="llm", seed=config["seed"])
                canonical = canonical_expression(expr)
                if canonical in seen:
                    raise ValueError("与已有候选表达式等价")
                if expression_fingerprint(expr) in historical and not config["allow_historical_revisit"]:
                    raise ValueError("历史任务已挖掘等价表达式")
                seen.add(canonical)
                prepared.append((index, proposal, built))
            except Exception as exc:
                warnings.append(f"LLM 提议 {proposal.get('proposal_id') or index} 被拒绝: {exc}")
                self._repo(
                    "save_proposal", run_id, audit.get("call_id"), proposal,
                    "rejected", None, str(exc),
                )

        exprs = [item[1]["expr"] for item in prepared]

        def llm_ic_progress(current_date, total):
            task_manager.check_cancelled(run_id)
            task_manager.update_progress(run_id, {
                "stage": "train_filter", "current": 0, "total": max(len(proposals), 1),
                "candidate_count": len(candidates), "generation_source": "llm",
                "last_date": current_date.isoformat() if hasattr(current_date, "isoformat") else str(current_date),
            })

        train_results = miner._batch_ic(
            exprs, [d.isoformat() for d in train_dates],
            progress_cb=llm_ic_progress, return_metrics=True,
        ) if exprs else []
        valid_results = miner._batch_ic(
            exprs, [d.isoformat() for d in valid_dates],
            progress_cb=llm_ic_progress, return_metrics=True,
        ) if exprs else []
        for (index, proposal, built), train_metrics, valid_metrics in zip(
            prepared, train_results, valid_results
        ):
            task_manager.check_cancelled(run_id)
            expr = proposal["expr"]
            key = None
            rejection_reason = ""
            try:
                train_ic = train_metrics.get("mean")
                if train_ic is None or not math.isfinite(float(train_ic)):
                    raise ValueError("训练期没有可用 IC")
                key = candidate_key(expr, source="llm")
                candidate = {
                    **built,
                    "candidate_key": key,
                    "train_metrics": {
                        **train_metrics,
                        "ic": float(train_ic),
                        "validation_ic": valid_metrics.get("mean"),
                        "fitness": compute_training_fitness(
                            train_metrics, complexity=built["complexity"]),
                    },
                    "oos_metrics": {},
                    "validation_layers": {},
                    "correlation_metrics": {},
                    "strategy_metrics": {},
                    "status": "train_passed",
                    "source_metadata": {
                        "proposal_id": proposal.get("proposal_id"),
                        "hypothesis": proposal.get("hypothesis"),
                        "rationale": proposal.get("rationale"),
                        "expected_sign": proposal.get("expected_sign"),
                        "tags": proposal.get("tags") or [],
                        "provider": audit.get("provider"),
                        "model": audit.get("model"),
                        "prompt_profile": audit.get("prompt_profile"),
                    },
                }
                candidates.append(candidate)
                accepted.append(candidate)
                self._repo("save_candidate", run_id, candidate)
            except Exception as exc:
                rejection_reason = str(exc)
                warnings.append(f"LLM 提议 {proposal.get('proposal_id') or index} 被拒绝: {exc}")
            self._repo(
                "save_proposal", run_id, audit.get("call_id"), proposal,
                "accepted" if key else "rejected", key, rejection_reason,
            )
            self._set_progress(
                run_id, "train_filter", index, max(len(proposals), 1), config,
                candidate_count=len(candidates), generation_source="llm",
            )
        if config["candidate_sources"] == ["llm"] and not accepted:
            raise RuntimeError("LLM 未产生通过表达式和训练期检查的候选")
        return accepted, audit

    @staticmethod
    def _progress_payload(stage, current, total, config, **extra):
        payload = {
            "stage": stage,
            "current": current,
            "total": total,
            "candidate_count": extra.pop("candidate_count", 0),
            "passed_count": extra.pop("passed_count", 0),
            "best_train_ic": extra.pop("best_train_ic", None),
            "best_oos_ic": extra.pop("best_oos_ic", None),
            "last_date": extra.pop("last_date", None),
        }
        payload.update(extra)
        return payload

    def _set_progress(self, run_id, stage, current, total, config, persist=False, **extra):
        self._record_stage(run_id, stage)
        payload = self._progress_payload(stage, current, total, config, **extra)
        task_manager.check_cancelled(run_id)
        task_manager.update_progress(run_id, payload)
        if persist:
            self._repo_required("update_run", run_id, status="running", progress=payload)
        return payload

    @staticmethod
    def _memory_progress():
        try:
            import psutil
            info = psutil.Process().memory_info()
            peak = getattr(info, "peak_wset", info.rss)
            return {
                "rss_mb": round(info.rss / (1024 * 1024), 1),
                "peak_rss_mb": round(peak / (1024 * 1024), 1),
            }
        except Exception:
            return {}

    def _run(self, run_id, config, resume_checkpoint=None):
        config = self._normalize_config(config)
        candidates = []
        warnings = []
        started = dt.datetime.now().isoformat(timespec="seconds")
        self._stage_clocks.pop(run_id, None)
        heartbeat_stop = threading.Event()

        def heartbeat_loop():
            while not heartbeat_stop.wait(30):
                task_manager.heartbeat(run_id)
                self._repo("touch_run", run_id)

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        try:
            self._set_progress(run_id, "planning", 1, 1, config, persist=True)
            from backend.service.factor_panel import FactorPanelService
            panel_service = FactorPanelService()
            trading_dates = panel_service.load_trading_dates(
                config["train_start"], config["final_oos_end"],
            )
            raw_train_dates = sample_trading_dates(
                trading_dates, config["train_start"], config["train_end"], config["step"],
            )
            raw_valid_dates = sample_trading_dates(
                trading_dates, config["valid_start"], config["valid_end"], config["step"],
            )
            final_oos_dates = [] if config["defer_final_oos"] else sample_trading_dates(
                trading_dates, config["final_oos_start"], config["final_oos_end"], config["step"],
            )
            purge_gap = config["forward_days"] + 1
            train_dates = purge_dates_before_boundary(
                raw_train_dates, config["valid_start"], trading_dates, purge_gap)
            valid_dates = raw_valid_dates if config["compatibility_mode"] else purge_dates_before_boundary(
                raw_valid_dates, config["final_oos_start"], trading_dates, purge_gap)
            l1_dates = build_l1_validation_dates(
                config["valid_start"], trading_dates, purge_gap=purge_gap, step=5,
            )
            folds = build_walkforward_folds(
                train_dates, valid_dates, n_folds=5, train_months=36,
                purge_gap=purge_gap, trading_dates=trading_dates,
            )
            all_dates = sorted(set(
                list(train_dates) + list(valid_dates) + list(final_oos_dates)
                + [value for values in l1_dates.values() for value in values]
            ))
            if not train_dates or not valid_dates or (
                not config["defer_final_oos"] and not final_oos_dates
            ):
                raise RuntimeError("训练、验证或最终 OOS 区间没有可用交易日")
            if config["final_oos_start"] <= config["valid_end"]:
                warnings.append("最终 OOS 与验证区间重叠；建议下一次任务提供冻结后的新日期区间")
            preflight_backtest_context = None
            if not config["compatibility_mode"]:
                preflight_backtest_context = self._resolve_backtest_context(config)

            from core.domain.factors.genetic import GeneticFactorMiner, build_mining_context
            from core.domain.factors.selection import adjust_selection_bias
            from core.domain.factors.registry import get_screener
            screener = get_screener()
            self._set_progress(run_id, "snapshot", 0, len(all_dates), config, persist=True)

            snapshot_mode = "panel" if getattr(app_config, "FACTOR_PANEL_ENABLED", True) else "legacy"

            def snapshot_progress(done, total, dkey):
                self._set_progress(
                    run_id, "snapshot", done, total, config,
                    last_date=dkey,
                    candidate_count=len(candidates),
                    snapshot_mode=snapshot_mode,
                    **self._memory_progress(),
                )

            if snapshot_mode == "panel":
                try:
                    snapshots, mktcap_maps, extra_factors = panel_service.build_mining_context(
                        config["pool"], all_dates,
                        risk_neutralize=config["risk_neutralize"],
                        progress_cb=snapshot_progress,
                    )
                except Exception as exc:
                    snapshot_mode = "legacy"
                    warnings.append(f"共享研究面板构建失败，已回退兼容快照: {exc}")
                    logger.warning("共享研究面板构建失败，回退兼容快照", exc_info=True)
                    snapshots, mktcap_maps, extra_factors = build_mining_context(
                        screener, config["pool"], all_dates,
                        risk_neutralize=config["risk_neutralize"],
                        progress_cb=snapshot_progress,
                    )
            else:
                snapshots, mktcap_maps, extra_factors = build_mining_context(
                    screener, config["pool"], all_dates,
                    risk_neutralize=config["risk_neutralize"],
                    progress_cb=snapshot_progress,
                )
            task_manager.check_cancelled(run_id)
            if not snapshots:
                raise RuntimeError("无可用快照 (股票池无数据或数据源不可用)")
            panel_mb = round(getattr(snapshots, "memory_bytes", 0) / (1024 * 1024), 1) or None
            self._set_progress(run_id, "generate", 0, 1, config,
                               candidate_count=0, last_date=all_dates[-1].isoformat(), persist=True,
                               snapshot_mode=snapshot_mode, panel_mb=panel_mb,
                               **self._memory_progress())
            from core.data.fetcher import DataFetcher
            try:
                industry_map = DataFetcher().get_industry_map() or {}
            except Exception as exc:
                warnings.append(f"行业映射不可用: {exc}")
                industry_map = {}
            miner = GeneticFactorMiner(
                snapshots, forward_days=config["forward_days"],
                mktcap_maps=mktcap_maps, industry_map=industry_map,
                extra_factors=extra_factors,
            )
            industry_map = industry_map or {}
            input_fingerprint = None
            engine_checkpoint = None

            def persist_generation_checkpoint(state):
                progress = self._progress_payload(
                    "generate", state["next_generation"], config["generations"], config,
                    candidate_count=len(candidates), generation_source="gp",
                    checkpoint_generation=state["next_generation"],
                    checkpoint_available=True,
                )
                checkpoint = {
                    "version": 1,
                    "stage": "gp_generation",
                    "input_fingerprint": input_fingerprint,
                    "engine_state": state,
                    "progress": progress,
                }
                task_manager.check_cancelled(run_id)
                self._repo_required("save_checkpoint", run_id, checkpoint, progress)
                task_manager.update_progress(run_id, progress)

            def stage_progress(stage, current, total):
                self._set_progress(
                    run_id, stage, current, total, config,
                    candidate_count=len(candidates),
                    best_train_ic=(max((abs(x.get("train_ic", 0)) for x in candidates), default=None)),
                )

            def generation_progress(generation, total, best):
                self._set_progress(
                    run_id, "generate", generation, total, config,
                    candidate_count=len(candidates), best_train_ic=best,
                )

            def ic_progress(current_date, total):
                if current_date is not None:
                    task_manager.check_cancelled(run_id)
                    task_manager.update_progress(run_id, {
                        "stage": "train_filter", "current": 0, "total": total,
                        "candidate_count": len(candidates),
                        "last_date": current_date.isoformat() if hasattr(current_date, "isoformat") else str(current_date),
                    })

            gp_result = []
            llm_audit = {}
            historical_fingerprints = set(self._repo("archive_fingerprints") or [])
            pool_fingerprints = self._pool_expression_fingerprints()
            excluded_fingerprints = set(pool_fingerprints)
            if not config["allow_historical_revisit"]:
                excluded_fingerprints.update(historical_fingerprints)
            base_fingerprint = self._mining_input_fingerprint(
                config, snapshots, mktcap_maps, extra_factors, industry_map,
            )
            input_fingerprint = hashlib.sha256(json.dumps(
                {"base": base_fingerprint, "excluded_fingerprints": sorted(excluded_fingerprints)},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            if resume_checkpoint is not None:
                if resume_checkpoint.get("input_fingerprint") != input_fingerprint:
                    self._repo_required("clear_checkpoint", run_id)
                    raise RuntimeError("检查点绑定的数据或算法指纹已变化，拒绝恢复；请重新提交任务")
                engine_checkpoint = resume_checkpoint.get("engine_state")
            if excluded_fingerprints:
                warnings.append(
                    f"生成阶段已加载 {len(historical_fingerprints)} 个历史表达式和 "
                    f"{len(pool_fingerprints)} 个因子池表达式用于防重复"
                )
            if "gp" in config["candidate_sources"]:
                gp_result = miner.mine(
                    population=config["population"], generations=config["generations"], top_n=config["top_n"],
                    train_dates=[d.isoformat() for d in train_dates],
                    valid_dates=[d.isoformat() for d in valid_dates], seed=config["seed"],
                    max_depth=config["max_depth"],
                    progress_cb=generation_progress, stage_cb=stage_progress,
                    ic_progress_cb=ic_progress, excluded_fingerprints=excluded_fingerprints,
                    resume_state=engine_checkpoint,
                    checkpoint_cb=persist_generation_checkpoint,
                )
                self._repo_required("clear_checkpoint", run_id)
                for item in gp_result:
                    task_manager.check_cancelled(run_id)
                    built = build_candidate(
                        item["expr"], source="gp", seed=config["seed"],
                        metrics={"train_ic": item.get("train_ic"), "valid_ic": item.get("valid_ic"),
                                 "train_fitness": item.get("train_fitness")},
                    )
                    candidate = {
                        **built,
                        "candidate_key": candidate_key(item["expr"]),
                        "train_metrics": {
                            **(item.get("train_metrics") or {}),
                            "ic": item.get("train_ic"),
                            "validation_ic": item.get("valid_ic"),
                            "fitness": item.get("train_fitness"),
                        },
                        "oos_metrics": {},
                        "validation_layers": {},
                        "correlation_metrics": {},
                        "strategy_metrics": {},
                        "status": "train_passed" if item.get("train_ic") is not None and math.isfinite(float(item.get("train_ic"))) else "generated",
                        "expr_fingerprint": item.get("expr_fingerprint") or expression_fingerprint(item["expr"]),
                        "generation": item.get("generation"),
                    }
                    candidates.append(candidate)
                    self._repo("save_candidate", run_id, candidate)
                    self._repo(
                        "upsert_archive", run_id, candidate,
                        candidate["expr_fingerprint"],
                    )

            if "llm" in config["candidate_sources"]:
                self._set_progress(
                    run_id, "generate", 0, 1, config,
                    candidate_count=len(candidates), generation_source="llm", persist=True,
                )
                _, llm_audit = self._generate_llm_candidates(
                    run_id, config, miner, train_dates, valid_dates, candidates, warnings,
                    historical_fingerprints=excluded_fingerprints,
                )

            for candidate in candidates:
                self._repo(
                    "upsert_archive", run_id, candidate,
                    candidate.get("expr_fingerprint") or expression_fingerprint(candidate["expr"]),
                )

            if not candidates:
                raise RuntimeError("没有候选通过生成和训练期检查")
            self._set_progress(run_id, "train_filter", len(candidates), len(candidates), config,
                               candidate_count=len(candidates), persist=True)

            self._set_progress(run_id, "validation", 0, len(candidates), config,
                               candidate_count=len(candidates), persist=True)
            from core.domain.factors.validation import OverfittingChecker
            checker = OverfittingChecker()
            train_variants = [
                {"variant_key": item["candidate_key"], "expr": item["expr"]}
                for item in candidates if item["status"] == "train_passed"
            ]
            checker.set_expr_variants(train_variants)
            kept_keys = set(checker.deduplicate_variants(
                train_variants,
                pool=config["pool"],
                eval_date=dt.date.fromisoformat(config["valid_start"]),
                corr_thr=0.9,
                snapshots=snapshots,
                selection_scores={
                    item["candidate_key"]: item["train_metrics"].get("fitness")
                    for item in candidates if item["status"] == "train_passed"
                },
            ))
            for candidate in candidates:
                if candidate["status"] == "train_passed" and candidate["candidate_key"] not in kept_keys:
                    candidate["status"] = "rejected"
                    candidate["rejection_reason"] = "与同批候选截面相关性超过 0.90"
                    self._repo("save_candidate", run_id, candidate)
            for index, candidate in enumerate(candidates, 1):
                task_manager.check_cancelled(run_id)
                if candidate["status"] != "train_passed":
                    self._set_progress(
                        run_id, "validation", index, len(candidates), config,
                        candidate_count=len(candidates),
                        passed_count=sum(x["status"] == "validation_passed" for x in candidates),
                    )
                    continue
                variant = {"variant_key": candidate["candidate_key"], "expr": candidate["expr"]}
                try:
                    layers = checker.check_all_layers(
                        variant, eval_weeks=18, pool=config["pool"], top_k=10,
                        snapshots=snapshots, mode="full",
                        anchor_date=dt.date.fromisoformat(config["valid_start"]),
                        l1_dates=l1_dates,
                        eval_dates=l1_dates.get("test") or valid_dates,
                        progress_cb=lambda _payload: task_manager.check_cancelled(run_id),
                    )
                except Exception as exc:
                    layers = {"all_pass": False, "layers": [], "error": str(exc), "max_corr": None}
                    warnings.append(f"{candidate['candidate_key']} 校验异常: {exc}")
                candidate["validation_layers"] = layers
                candidate["correlation_metrics"] = {
                    "max_correlation": layers.get("max_corr"),
                    "max_correlation_with": layers.get("max_corr_with"),
                    "evidence": layers.get("correlation_evidence") or {},
                    "evaluated": layers.get("max_corr") is not None,
                }
                if layers.get("all_pass"):
                    candidate["status"] = "validation_passed"
                    candidate.pop("rejection_reason", None)
                else:
                    failed_layers = [
                        f"L{item['id']} {item.get('detail', '')}"
                        for item in layers.get("layers", [])
                        if item.get("status") == "failed" or item.get("pass") is False
                    ]
                    candidate["status"] = "rejected"
                    candidate["rejection_reason"] = "；".join(failed_layers) or layers.get("error") or "未通过现有层级校验"
                self._repo("save_candidate", run_id, candidate)
                self._repo("save_score", run_id, candidate["candidate_key"], "validation", layers)
                self._set_progress(
                    run_id, "validation", index, len(candidates), config,
                    candidate_count=len(candidates), passed_count=sum(x["status"] == "validation_passed" for x in candidates),
                )

            eligible = [x for x in candidates if x["status"] == "validation_passed"]
            self._set_progress(run_id, "walk_forward", 0, 5, config,
                               candidate_count=len(candidates), passed_count=len(eligible), persist=True)
            for candidate in eligible:
                fold_results = []
                for fold in folds:
                    task_manager.check_cancelled(run_id)
                    train_detail = miner._batch_ic(
                        [candidate["expr"]], fold["train_dates"],
                        return_metrics=True, include_series=True,
                    )[0]
                    oos_detail = miner._batch_ic(
                        [candidate["expr"]], fold["valid_dates"],
                        return_metrics=True, include_series=True,
                    )[0]
                    train_ic = train_detail.get("mean")
                    oos_ic = oos_detail.get("mean")
                    train_ic = float(train_ic) if train_ic is not None and math.isfinite(train_ic) else None
                    oos_ic = float(oos_ic) if oos_ic is not None and math.isfinite(oos_ic) else None
                    fold_result = {
                        "fold": fold["fold"], "train_ic": train_ic, "oos_ic": oos_ic,
                        "train_ic_series": train_detail.get("series", []),
                        "oos_ic_series": oos_detail.get("series", []),
                        "valid_date_count": len(fold["valid_dates"]),
                        "passed": oos_ic is not None and oos_ic >= 0.02,
                        "train_start": fold["train_start"], "train_end": fold["train_end"],
                        "valid_start": fold["valid_start"], "valid_end": fold["valid_end"],
                        "purge_gap": fold["purge_gap"],
                    }
                    fold_results.append(fold_result)
                    self._repo("save_score", run_id, candidate["candidate_key"], "walk_forward", fold_result, fold["fold"])
                    self._set_progress(
                        run_id, "walk_forward", fold["fold"], max(5, len(folds)), config,
                        candidate_count=len(candidates), passed_count=sum(x["status"] in ("validation_passed", "strategy_passed", "awaiting_promotion") for x in candidates),
                        last_date=fold["valid_end"],
                    )
                aggregate = aggregate_fold_metrics(fold_results)
                train_ic = abs(float(candidate["train_metrics"].get("ic") or 0.0))
                oos_ic = abs(float(aggregate.get("oos_ic") or 0.0))
                decay = max(0.0, train_ic - oos_ic) / train_ic if train_ic > 1e-12 else 1.0
                candidate["oos_metrics"] = {
                    **aggregate, "folds_detail": fold_results, "decay": decay,
                }
                candidate["status"] = "strategy_passed" if aggregate.get("passed_folds", 0) >= 3 else "validation_passed"
                self._repo("save_candidate", run_id, candidate)

            self._set_progress(run_id, "deduplicate", 0, len(candidates), config,
                               candidate_count=len(candidates), persist=True)
            seen = {}
            for candidate in candidates:
                key = canonical_expression(candidate["expr"])
                if key in seen:
                    candidate["status"] = "rejected"
                    candidate["rejection_reason"] = "与同批候选表达式等价"
                else:
                    seen[key] = candidate["candidate_key"]
                self._repo("save_candidate", run_id, candidate)
            self._set_progress(run_id, "deduplicate", len(candidates), len(candidates), config,
                               candidate_count=len(candidates), persist=True)

            if hasattr(snapshots, "clear"):
                snapshots.clear()
            miner.snapshots = {}
            import gc
            gc.collect()

            eligible_strategy = [x for x in candidates if x["status"] == "strategy_passed"]
            selection_count = len([x for x in candidates if x.get("oos_metrics") and x["status"] != "rejected"])
            campaign_attempt_count = max(
                int(self._repo("archive_count") or 0), selection_count)
            for candidate in candidates:
                oos_metrics = candidate.get("oos_metrics") or {}
                if not oos_metrics:
                    continue
                selection = adjust_selection_bias(
                    oos_metrics.get("oos_ic", oos_metrics.get("ic")),
                    campaign_attempt_count, oos_metrics.get("folds"),
                )
                candidate["oos_metrics"].update(selection)
                self._repo("save_candidate", run_id, candidate)
            if not config["compatibility_mode"]:
                backtest_context = preflight_backtest_context
                model_info = backtest_context["model_info"]
            else:
                backtest_context = {
                    "requested_baseline_set": config["baseline_set"],
                    "baseline_set": config["baseline_set"],
                    "factor_count": 0, "model_info": {},
                    "skipped": "legacy_compatibility" if config["compatibility_mode"] else "no_candidate",
                }
                model_info = {}
            if model_info.get("baseline_coverage", model_info.get("factor_overlap", 1.0)) < 0.8:
                warnings.append(
                    f"基线模型与 {backtest_context['baseline_set']} 因子重合度仅 "
                    f"{model_info.get('baseline_coverage', model_info.get('factor_overlap', 0.0)):.1%}，已在报告中标记")
            qlib_model = tuple(model_info["model"]) if model_info.get("model") else None
            total_ab_steps = max(1, len(folds) * (1 + len(eligible_strategy)) + len(eligible_strategy) + 1)
            self._set_progress(run_id, "incremental_backtest", 0, total_ab_steps, config, persist=True)
            backtest_fingerprint = self._backtest_fingerprint(config, backtest_context)
            completed_ab_steps = 0
            baseline_fold_metrics = {}
            if eligible_strategy and len(folds) == 5 and not config["compatibility_mode"]:
                for fold in folds:
                    task_manager.check_cancelled(run_id)
                    result = self._run_production_backtest(
                        config, start=fold["valid_start"], end=fold["valid_end"],
                        qlib_model=qlib_model, run_id=run_id)
                    metrics = self._backtest_metrics(result)
                    baseline_fold_metrics[fold["fold"]] = metrics
                    self._repo("save_backtest", run_id, "__baseline__", "baseline",
                               metrics, backtest_fingerprint, fold["fold"])
                    completed_ab_steps += 1
                    self._set_progress(
                        run_id, "incremental_backtest", completed_ab_steps, total_ab_steps, config,
                        candidate_count=len(candidates), passed_count=0,
                        last_date=fold["valid_end"],
                    )

            final_candidates = []
            for candidate in candidates:
                task_manager.check_cancelled(run_id)
                if candidate["status"] != "strategy_passed":
                    candidate["strategy_metrics"] = {
                        "evaluation_mode": "production_screening",
                        "status": "not_eligible",
                        "reason": candidate.get("rejection_reason") or "未通过滚动 OOS",
                    }
                    candidate["promotion_gate"] = {
                        "passed": False,
                        "reasons": [candidate.get("rejection_reason") or "未通过滚动 OOS"],
                    }
                elif config["compatibility_mode"]:
                    candidate["strategy_metrics"] = {
                        "evaluation_mode": "production_screening",
                        "status": "not_eligible", "reason": "legacy_compatibility",
                    }
                    candidate["promotion_gate"] = {
                        "passed": False, "reasons": ["旧接口兼容任务不允许进入生产 A/B 或晋级"],
                    }
                elif len(folds) != 5:
                    candidate["strategy_metrics"] = {
                        "evaluation_mode": "production_screening",
                        "status": "not_eligible", "folds": [], "fold_wins": 0,
                    }
                    candidate["promotion_gate"] = {"passed": False, "reasons": ["滚动验证未生成完整 5 折"]}
                else:
                    strategy_folds = []
                    for fold in folds:
                        task_manager.check_cancelled(run_id)
                        candidate_result = self._run_production_backtest(
                            config,
                            factor_overlay={
                                "expr": candidate["expr"], "weight": config["overlay_weight"],
                                "risk_neutralize": config["risk_neutralize"],
                            },
                            start=fold["valid_start"], end=fold["valid_end"],
                            qlib_model=qlib_model, run_id=run_id,
                        )
                        candidate_metrics = self._backtest_metrics(candidate_result)
                        baseline_metrics_for_fold = baseline_fold_metrics[fold["fold"]]
                        delta = self._backtest_delta(candidate_metrics, baseline_metrics_for_fold)
                        fold_passed, fold_reasons = self._strategy_gate(delta)
                        evidence = {
                            "fold": fold["fold"], "passed": fold_passed,
                            "reasons": fold_reasons, "candidate": candidate_metrics,
                            "baseline": baseline_metrics_for_fold, "delta": delta,
                            "valid_start": fold["valid_start"], "valid_end": fold["valid_end"],
                        }
                        strategy_folds.append(evidence)
                        self._repo("save_backtest", run_id, candidate["candidate_key"], "candidate",
                                   evidence, backtest_fingerprint, fold["fold"])
                        completed_ab_steps += 1
                        self._set_progress(
                            run_id, "incremental_backtest", completed_ab_steps, total_ab_steps, config,
                            candidate_count=len(candidates),
                            passed_count=len(final_candidates), last_date=fold["valid_end"],
                        )
                    fold_wins = sum(bool(item["passed"]) for item in strategy_folds)
                    candidate["strategy_metrics"] = {
                        "evaluation_mode": "production_screening",
                        "status": "fold_evaluated", "config_fingerprint": backtest_fingerprint,
                        "baseline_set": backtest_context["baseline_set"],
                        "model": model_info, "folds": strategy_folds, "fold_wins": fold_wins,
                    }
                    if fold_wins >= 3:
                        final_candidates.append(candidate)
                    else:
                        reason = f"策略滚动折仅 {fold_wins}/5 优于基线"
                        candidate["status"] = "rejected"
                        candidate["rejection_reason"] = reason
                        candidate["promotion_gate"] = {
                            "passed": False, "reasons": [reason],
                        }
                self._repo("save_candidate", run_id, candidate)

            baseline_result = None
            baseline_metrics = None
            if config["defer_final_oos"]:
                for candidate in final_candidates:
                    candidate["status"] = "campaign_shortlisted"
                    candidate["promotion_gate"] = {
                        "passed": False,
                        "reasons": ["持续挖掘批次仅归档候选，冻结最终 OOS 尚未执行"],
                    }
                    self._repo("save_candidate", run_id, candidate)
                final_candidates = []
            if final_candidates:
                baseline_result = self._run_production_backtest(
                    config, qlib_model=qlib_model, run_id=run_id)
                baseline_metrics = self._backtest_metrics(baseline_result)
                self._repo("save_backtest", run_id, "__baseline__", "baseline",
                           baseline_metrics, backtest_fingerprint)
                completed_ab_steps += 1

            for candidate in final_candidates:
                task_manager.check_cancelled(run_id)
                claim = None
                try:
                    claim = self._reserve_oos(
                        run_id, candidate, config, backtest_fingerprint)
                    if not claim.get("created"):
                        if claim.get("run_id") == run_id and claim.get("status") in ("passed", "failed"):
                            locked = claim.get("metrics") or {}
                            candidate_metrics = locked.get("candidate") or {}
                            delta = locked.get("delta") or self._backtest_delta(
                                candidate_metrics, baseline_metrics or {})
                        else:
                            raise RuntimeError("最终 OOS 已被锁定或曾在中断任务中暴露，禁止重复执行")
                    else:
                        candidate_result = self._run_production_backtest(
                            config, factor_overlay={
                                "expr": candidate["expr"], "weight": config["overlay_weight"],
                                "risk_neutralize": config["risk_neutralize"],
                            }, qlib_model=qlib_model, run_id=run_id)
                        candidate_metrics = self._backtest_metrics(candidate_result)
                        delta = self._backtest_delta(candidate_metrics, baseline_metrics or {})
                        self._finalize_oos(
                            claim["lock_key"], candidate_metrics, baseline_metrics,
                            delta, candidate_result.get("overlay_coverage"),
                            "passed" if self._strategy_gate(delta)[0] else "failed",
                        )
                    candidate["strategy_metrics"].update({
                        "status": "evaluated", "candidate": candidate_metrics,
                        "baseline": baseline_metrics, "delta": delta,
                        "overlay_coverage": (claim.get("metrics") or {}).get("overlay_coverage")
                        if claim and not claim.get("created") else candidate_result.get("overlay_coverage"),
                    })
                    candidate["strategy_metrics"]["oos_lock"] = self._repo(
                        "get_oos_lock", lock_key=claim["lock_key"])
                    self._repo("save_backtest", run_id, candidate["candidate_key"], "candidate_final",
                               candidate_metrics, backtest_fingerprint)
                    strategy_passed, strategy_reasons = self._strategy_gate(delta)
                    gate_metrics = {
                        "oos_ic": candidate["oos_metrics"].get("oos_ic", candidate["oos_metrics"].get("ic")),
                        "selection_adjusted_oos_ic": candidate["oos_metrics"].get("selection_adjusted_oos_ic"),
                        "oos_icir": candidate["oos_metrics"].get("icir"),
                        "positive_ratio": candidate["oos_metrics"].get("positive_ratio", 0.0),
                        "passed_folds": candidate["oos_metrics"].get("passed_folds", 0),
                        "folds": candidate["oos_metrics"].get("folds", 0),
                        "decay": candidate["oos_metrics"].get("decay"),
                        "max_correlation": candidate["correlation_metrics"].get("max_correlation", 0.0),
                        "coverage": candidate["oos_metrics"].get("coverage", 0.0),
                        "strategy_net_return": candidate_metrics.get("total_return"),
                        "strategy_passed": strategy_passed and candidate["strategy_metrics"]["fold_wins"] >= 3,
                        "strategy_reasons": strategy_reasons,
                        "oos_independent": config["final_oos_start"] > config["valid_end"],
                    }
                    candidate["promotion_gate"] = evaluate_promotion_gate(
                        gate_metrics, baseline={"net_return": (baseline_metrics or {}).get("total_return")},
                    )
                    candidate["strategy_metrics"]["promotion_gate"] = candidate["promotion_gate"]
                    if candidate["promotion_gate"]["passed"] and strategy_passed:
                        candidate["status"] = "awaiting_promotion"
                except Exception as exc:
                    if claim and claim.get("created"):
                        self._finalize_oos(claim["lock_key"], {}, baseline_metrics, {}, None, "failed", str(exc))
                    candidate["strategy_metrics"].update({
                        "status": "error", "error": str(exc),
                        "config_fingerprint": backtest_fingerprint,
                    })
                    candidate["promotion_gate"] = {"passed": False, "reasons": [f"生产 A/B 失败: {exc}"]}
                    warnings.append(f"{candidate['candidate_key']} 生产 A/B 失败: {exc}")
                self._repo("save_candidate", run_id, candidate)
                completed_ab_steps += 1
                self._set_progress(
                    run_id, "incremental_backtest", min(completed_ab_steps, total_ab_steps), total_ab_steps, config,
                    candidate_count=len(candidates),
                    passed_count=sum(x["status"] == "awaiting_promotion" for x in candidates),
                )

            for candidate in candidates:
                fingerprint = candidate.get("expr_fingerprint") or expression_fingerprint(candidate["expr"])
                candidate["expr_fingerprint"] = fingerprint
                self._repo("upsert_archive", run_id, candidate, fingerprint)
            passed = [x for x in candidates if x["status"] == "awaiting_promotion"]
            self._set_progress(run_id, "report", 1, 1, config, candidate_count=len(candidates),
                               passed_count=len(passed), persist=True)
            report = {
                "run_id": run_id,
                "request": config,
                "data_version": {
                    "snapshot_cutoff": all_dates[-1].isoformat(),
                    "source": "qlib_adjusted_daily",
                    "snapshot_mode": snapshot_mode,
                    "panel_dtype": getattr(snapshots, "dtype", getattr(app_config, "FACTOR_PANEL_DTYPE", "float64")),
                    "panel_mb": panel_mb,
                    "final_oos_start": config["final_oos_start"],
                    "final_oos_end": config["final_oos_end"],
                    "final_oos_evaluated": not config["defer_final_oos"],
                    "overlaps_validation": config["final_oos_start"] <= config["valid_end"],
                    "independent": config["final_oos_start"] > config["valid_end"],
                },
                "seed": config["seed"],
                "lineage": [{
                    "candidate_key": x["candidate_key"], "expr": x["expr"],
                    "source": x.get("source", "gp"), "parents": x["parent_exprs"],
                } for x in candidates],
                "candidate_generation": {
                    "sources": config["candidate_sources"],
                    "gp_returned_count": len(gp_result),
                    "llm": llm_audit,
                    "llm_proposals": self._repo("list_proposals", run_id) or [],
                    "oos_exposed_to_llm": False,
                },
                "started_at": started,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                "stage_timings": {},
                "folds": folds,
                "candidates": candidates,
                "backtest": {
                    "profile": config["backtest_profile"],
                    "config_fingerprint": backtest_fingerprint,
                    "baseline_context": backtest_context,
                    "baseline": baseline_metrics,
                    "candidates": {
                        x["candidate_key"]: x.get("strategy_metrics")
                        for x in candidates if x.get("strategy_metrics")
                    },
                },
                "promotion_conclusion": (
                    "campaign_archived" if config["defer_final_oos"]
                    else "awaiting_manual_confirmation" if passed else "no_candidate_passed"
                ),
                "result_summary": {
                    "candidate_count": len(candidates),
                    "validation_passed_count": sum(
                        item["status"] in (
                            "validation_passed", "strategy_passed", "campaign_shortlisted",
                            "awaiting_promotion",
                        )
                        for item in candidates
                    ),
                    "walkforward_evaluated_count": sum(
                        bool((item.get("oos_metrics") or {}).get("folds")) for item in candidates
                    ),
                    "strategy_evaluated_count": sum(
                        (item.get("strategy_metrics") or {}).get("status") == "evaluated"
                        for item in candidates
                    ),
                    "strategy_fold_evaluated_count": sum(
                        (item.get("strategy_metrics") or {}).get("status") == "fold_evaluated"
                        for item in candidates
                    ),
                    "campaign_shortlisted_count": sum(
                        item["status"] == "campaign_shortlisted" for item in candidates
                    ),
                    "rejected_count": sum(item["status"] == "rejected" for item in candidates),
                },
                "selection_control": {
                    "candidate_count": selection_count,
                    "historical_candidate_count": campaign_attempt_count,
                    "method": "log_candidate_haircut",
                    "final_oos_independent": config["final_oos_start"] > config["valid_end"],
                },
                "memory": self._memory_progress(),
                "warnings": warnings,
                "errors": [],
            }
            final_stage = "awaiting_promotion" if passed else "completed"
            final_progress = self._set_progress(
                run_id, final_stage, 1, 1, config,
                candidate_count=len(candidates), passed_count=len(passed), persist=False,
            )
            report["stage_timings"] = self._finish_stage_clock(run_id)
            report_path = os.path.join(REPORT_ROOT, run_id, "report.json")
            save_json(report, report_path)
            self._repo_required("update_run", run_id, status=final_stage, progress=final_progress, report=report)
            return {
                "_task_status": "awaiting_promotion" if passed else "completed",
                "summary": {"candidate_count": len(candidates), "passed_count": len(passed)},
                "report_path": report_path,
                "report": report,
            }
        except TaskCancelled:
            self._finish_stage_clock(run_id)
            self._repo_required("clear_checkpoint", run_id)
            self._repo_required("update_run", run_id, status="cancelled",
                                progress=self._progress_payload("cancelled", 0, 1, config))
            raise
        except Exception as exc:
            stage_timings = self._finish_stage_clock(run_id)
            error = traceback.format_exc()
            existing = task_manager.get_dict(run_id) or {}
            failed_progress = {
                **(existing.get("progress") or {}),
                "stage": "failed", "error": str(exc), "traceback": error,
            }
            task_manager.update_progress(run_id, failed_progress)
            failed_report = {
                "run_id": run_id, "request": config,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                "stage_timings": stage_timings,
                "candidates": candidates, "promotion_conclusion": "failed",
                "candidate_generation": {
                    "sources": config.get("candidate_sources", ["gp"]),
                    "llm_proposals": self._repo("list_proposals", run_id) or [],
                    "oos_exposed_to_llm": False,
                },
                "warnings": warnings, "errors": [error],
            }
            try:
                save_json(failed_report, os.path.join(REPORT_ROOT, run_id, "report.json"))
            except Exception:
                pass
            self._repo("update_run", run_id, status="failed", error=error,
                       progress=failed_progress, report=failed_report)
            raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)

    @staticmethod
    def _backtest_fingerprint(config, context=None):
        fields = {
            key: config.get(key) for key in (
                "pool", "final_oos_start", "final_oos_end", "backtest_profile",
                "backtest_freq", "backtest_top_n", "backtest_use_timing",
                "backtest_macro_bias", "backtest_sector_fusion", "overlay_weight",
                "risk_neutralize",
                "baseline_set",
                "commission_rate", "slippage_rate", "stamp_duty",
            )
        }
        if context:
            fields["resolved_baseline_set"] = context.get("baseline_set")
            fields["model_fingerprint"] = (context.get("model_info") or {}).get("artifact_fingerprint")
        return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_baseline(baseline_set):
        from core.domain.factors.registry import get_pool
        pool = get_pool()
        requested = baseline_set or "current_curated"
        if requested == "current_curated":
            entry = pool.get_curated_set("latest")
            if entry:
                baseline_set = entry["name"]
                factors = entry.get("flat_list") or []
            else:
                baseline_set = "trend_momentum"
                factors = pool.get_curated_factors(baseline_set)
        else:
            baseline_set = requested
            factors = pool.get_curated_factors(baseline_set)
        if not factors and baseline_set != "trend_momentum":
            baseline_set = "trend_momentum"
            factors = pool.get_curated_factors(baseline_set)
        if not factors:
            raise RuntimeError("没有可用于基线的精选因子集")
        return requested, baseline_set, factors

    def list_models(self, pool, baseline_set, train_end):
        dt.date.fromisoformat(train_end)
        from core.adapters.qlib.model_selection import list_models_for_factor_set
        requested, resolved, factors = self._resolve_baseline(baseline_set)
        models = list_models_for_factor_set(factors, cutoff=train_end, pool=pool)
        return {
            "requested_baseline_set": requested,
            "baseline_set": resolved,
            "factor_count": len(factors),
            "models": models,
        }

    def _resolve_backtest_context(self, config):
        from core.adapters.qlib.model_selection import (
            get_model_for_factor_set,
            select_model_for_factor_set,
        )
        requested, baseline_set, factors = self._resolve_baseline(config.get("baseline_set"))
        experiment_id = config.get("baseline_model_experiment_id")
        run_id = config.get("baseline_model_run_id")
        if experiment_id and run_id:
            model_info = get_model_for_factor_set(
                experiment_id, run_id, factors,
                cutoff=config["train_end"], pool=config["pool"],
            )
            selection_mode = "manual"
        else:
            model_info = select_model_for_factor_set(
                factors, cutoff=config["train_end"], pool=config["pool"])
            selection_mode = "auto"
        if not model_info:
            raise RuntimeError(
                f"未找到数据截止日不晚于训练截止日 {config['train_end']} 且可用于 {baseline_set} 的 Qlib 模型")
        return {
            "requested_baseline_set": requested,
            "baseline_set": baseline_set,
            "factor_count": len(factors),
            "model_selection_mode": selection_mode,
            "model_info": model_info,
        }

    @staticmethod
    def _backtest_metrics(result):
        summary = dict(result.get("summary") or {})
        diagnostics = result.get("trade_diagnostics") or {}
        periods = int(summary.get("n_periods") or len(result.get("periods") or []))
        summary.update({
            "turnover": (float(diagnostics.get("traded_notional") or 0.0) / max(periods, 1)),
            "transaction_cost": diagnostics.get("transaction_cost"),
            "traded_notional": diagnostics.get("traded_notional"),
        })
        return summary

    @staticmethod
    def _backtest_delta(candidate, baseline):
        return {
            "total_return": (candidate.get("total_return") or 0.0) - (baseline.get("total_return") or 0.0),
            "annual_return": (candidate.get("annual_return") or 0.0) - (baseline.get("annual_return") or 0.0),
            "sharpe": (candidate.get("sharpe") or 0.0) - (baseline.get("sharpe") or 0.0),
            "max_drawdown": (candidate.get("max_drawdown") or 0.0) - (baseline.get("max_drawdown") or 0.0),
            "turnover_ratio": (candidate.get("turnover") or 0.0) / max(float(baseline.get("turnover") or 0.0), 1e-9),
        }

    @staticmethod
    def _strategy_gate(delta):
        reasons = []
        if delta.get("total_return", 0.0) < 0.0:
            reasons.append("最终 OOS 净收益低于基线")
        if delta.get("sharpe", 0.0) < -0.05:
            reasons.append("最终 OOS Sharpe 比基线低超过 0.05")
        if delta.get("max_drawdown", 0.0) < -0.02:
            reasons.append("最大回撤恶化超过 2 个百分点")
        if delta.get("turnover_ratio", 0.0) > 1.20:
            reasons.append("换手率超过基线 120%")
        return not reasons, reasons

    def _run_production_backtest(self, config, factor_overlay=None, start=None, end=None,
                                 qlib_model=None, run_id=None):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        return runner.run(
            start=start or config["final_oos_start"], end=end or config["final_oos_end"],
            pool=config["pool"], top_n=config["backtest_top_n"],
            freq=config["backtest_freq"], use_timing=config["backtest_use_timing"],
            macro_bias=config["backtest_macro_bias"],
            sector_fusion=config["backtest_sector_fusion"],
            factor_overlay=factor_overlay, qlib_model=qlib_model,
            progress_cb=(lambda _current, _total, _date: task_manager.heartbeat(run_id)) if run_id else None,
            cancel_cb=(lambda: task_manager.check_cancelled(run_id)) if run_id else None,
            commission_rate=config.get("commission_rate", 0.0003),
            slippage_rate=config.get("slippage_rate", 0.0001),
            stamp_duty=config.get("stamp_duty", 0.0005),
        )

    @staticmethod
    def _oos_lock_key(candidate, config):
        payload = "|".join([
            canonical_expression(candidate["expr"]),
            config["final_oos_start"], config["final_oos_end"], config["final_oos_end"],
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _reserve_oos(self, run_id, candidate, config, fingerprint):
        lock_key = self._oos_lock_key(candidate, config)
        return self._repo_required(
            "reserve_oos", lock_key, run_id, candidate["candidate_key"],
            config["final_oos_start"], config["final_oos_end"],
            config["final_oos_end"], fingerprint,
        )

    def _finalize_oos(self, lock_key, candidate_metrics, baseline_metrics, delta,
                      overlay_coverage, status, error=""):
        metrics = {
            "candidate": candidate_metrics, "baseline": baseline_metrics,
            "delta": delta, "overlay_coverage": overlay_coverage,
        }
        if error:
            metrics["error"] = error
        payload = json.dumps(metrics, sort_keys=True, ensure_ascii=False, default=str)
        result_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._repo_required("finalize_oos", lock_key, result_hash, status, metrics)

    def get_run(self, run_id):
        task = task_manager.get_dict(run_id)
        persisted = self._repo("get_run", run_id)
        if not persisted:
            return task
        checkpoint = self._repo("get_checkpoint", run_id) if persisted.get("checkpoint_available") else None
        resume_available = bool(
            persisted.get("status") in ("planning", "running", "resuming", "failed")
            and
            isinstance(checkpoint, dict)
            and checkpoint.get("version") == 1
            and checkpoint.get("stage") == "gp_generation"
            and "gp" in (persisted.get("config") or {}).get("candidate_sources", [])
        )
        if persisted.get("status") not in ("planning", "running", "resuming"):
            return {**persisted, "resume_available": resume_available}
        if task_manager.is_active(run_id):
            return task or persisted
        if task and task.get("status") in (
            "completed", "partial", "failed", "cancelled", "awaiting_promotion",
        ):
            return {**task, "resume_available": resume_available}
        return {
            **persisted,
            "status": "interrupted",
            "resume_available": resume_available,
            "progress": {
                **(persisted.get("progress") or {}),
                "stage": "interrupted",
                "message": (
                    "后端任务线程已不存在；存在代际边界检查点，可显式恢复并校验输入指纹"
                    if resume_available else
                    "后端任务线程已不存在；当前阶段没有可验证的恢复检查点，请显式重新提交"
                ),
            },
        }

    def list_candidates(self, run_id):
        return self._repo("list_candidates", run_id) or []

    def get_candidate_detail(self, run_id, candidate_key_value):
        candidate = next(
            (item for item in self.list_candidates(run_id)
             if item.get("candidate_key") == candidate_key_value),
            None,
        )
        if candidate is None:
            return None
        candidate["scores"] = self._repo("get_scores", run_id, candidate_key_value) or []
        candidate["backtests"] = self._repo("list_backtests", run_id, candidate_key_value) or []
        proposals = self._repo("list_proposals", run_id, candidate_key_value) or []
        candidate["proposal"] = proposals[0] if proposals else None
        return candidate

    def list_proposals(self, run_id):
        return self._repo("list_proposals", run_id) or []

    def get_scores(self, run_id, candidate_key=None):
        return self._repo("get_scores", run_id, candidate_key) or []

    def list_backtests(self, run_id, candidate_key=None):
        return self._repo("list_backtests", run_id, candidate_key) or []

    def get_report(self, run_id):
        path = os.path.join(REPORT_ROOT, run_id, "report.json")
        report = load_json(path)
        if report is not None:
            return report
        persisted = self._repo("get_run", run_id)
        return (persisted or {}).get("report") or {}

    def cancel(self, run_id):
        ok = task_manager.request_cancel(run_id)
        if ok:
            self._repo("update_run", run_id, status="cancel_requested")
        return {"run_id": run_id, "cancel_requested": ok}

    def decide(self, run_id, candidate_key_value, target_pool, action="promote", reason=""):
        candidates = self.list_candidates(run_id)
        candidate = next((x for x in candidates if x.get("candidate_key") == candidate_key_value), None)
        if candidate is None:
            raise ValueError("候选不存在")
        if action == "promote":
            if candidate.get("status") != "awaiting_promotion":
                raise ValueError("候选尚未通过人工晋级门槛")
            from core.domain.factors.registry import get_pool
            actual_target = get_pool().add_factor(
                key=candidate_key_value,
                label=f"{candidate.get('source', 'gp').upper()}-{candidate['expr'][:36]}",
                expr=candidate["expr"],
                metrics={"train": candidate.get("train_metrics"), "oos": candidate.get("oos_metrics")},
                category="white_box",
                curated_set=target_pool,
            )
            candidate["status"] = "promoted"
            candidate["target_pool"] = actual_target
        elif action == "reject":
            candidate["status"] = "rejected"
            candidate["rejection_reason"] = reason or "人工拒绝"
        else:
            raise ValueError("action 只能是 promote 或 reject")
        self._repo("save_candidate", run_id, candidate)
        self._repo(
            "save_promotion", run_id, candidate_key_value,
            candidate.get("target_pool") or target_pool, action, reason)
        return candidate


factor_mining_service = FactorMiningService()
